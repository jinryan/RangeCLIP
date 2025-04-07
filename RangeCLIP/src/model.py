import os, sys
import torch, torchvision
sys.path.insert(0, os.getcwd())
from utils.src.networks import DepthEncoder, TextEncoder, ImageEncoder
import utils.src.loss_utils as loss_utils
import utils.src.log_utils as log_utils
from utils.src.log_utils import apply_colormap
from transformers import CLIPModel

import torch
import torch.nn as nn
import torch.nn.functional as F

def check_extreme(tensor, name):
    with torch.no_grad():
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        
        # print(f"--- {name} Analysis ---")
        # print(f"Contains NaN: {has_nan}")
        # print(f"Contains Inf: {has_inf}")
        # print(f"Min value: {min_val}")
        # print(f"Max value: {max_val}")
        # print(f"Mean value: {mean_val}")
        # print(f"Std deviation: {std_val}")
        
        # Check for extremely large values
        large_values = (torch.abs(tensor) > 1e10).sum().item()
        if large_values > 0:
            print(f"{name}: WARNING: {large_values} elements have absolute value > 1e10")
            
        # Check for extremely small values close to zero (potential division issues)
        small_values = ((torch.abs(tensor) > 0) & (torch.abs(tensor) < 1e-10)).sum().item()
        if small_values > 0:
            print(f"{name}: WARNING: {small_values} non-zero elements have absolute value < 1e-10")
        
        return not (has_nan or has_inf or large_values > 0)
    
def normalize_depth_map(depth_map, eps=1e-8):
    """
    Normalize depth map to 0-1 range with protection against zero division.
    
    Args:
        depth_map: Input depth map tensor
        eps: Small epsilon value to prevent division by zero
    
    Returns:
        Normalized depth map
    """
    min_val = depth_map.min()
    max_val = depth_map.max()
    
    # Check if the range is too small (flat depth map)
    if max_val - min_val < eps:
        return torch.zeros_like(depth_map)  # Return zeros if depth map is flat
    
    # Otherwise normalize as usual
    return (depth_map - min_val) / (max_val - min_val)

class DepthClipModel(nn.Module):
    def __init__(self, 
                 depth_encoder_type,
                 clip_model_name,
                 device,
                 temperature=0.07):
        """
        Open-vocabulary depth map classifier using contrastive learning.
        
        Args:
            depth_encoder (DepthEncoder): Encoder for processing depth maps
            text_encoder (TextEncoder): Encoder for processing text/class descriptions
            temperature (float): Temperature scaling for contrastive learning
        """
        super().__init__()
        
        self.device = device
        
        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.text_encoder = TextEncoder(clip_model, device)
            self.image_encoder = ImageEncoder(clip_model)
            embedding_dim = clip_model.config.projection_dim
        except:
            raise ValueError(f'Unsupported CLIP model {clip_model_name}')
        
        # Store the encoders
        if depth_encoder_type == 'resnet':
            self.depth_encoder = DepthEncoder(n_layer=18,
                                              input_channels=1,
                                              n_filters=[32, 64, 128, 256, 512],
                                              embedding_dim=embedding_dim,
                                              weight_initializer='kaiming_uniform',
                                              activation_func='relu',
                                              use_instance_norm=False,
                                              use_batch_norm=True )
        else:
            raise ValueError(f'Unsupported depth encoder {depth_encoder_type}')
        
        self.depth_encoder.to(self.device)
        self.text_encoder.to(self.device)
        self.image_encoder.to(self.device)
        
        # Learnable temperature parameter for scaling logits
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, depth_map, class_descriptions, image=None):
        """
        Forward pass: Handles both training (with images) and inference (depth + text only).
        
        Args:
            depth_map (Tensor): Depth input (batch, 1, H, W)
            class_descriptions (List[str]): List of text class descriptions
            image (Tensor, optional): RGB image (batch, 3, H, W). Default is None for inference.
        
        Returns:
            depth_embedding (Tensor): Depth feature embedding
            text_embeddings (Tensor): Class description embeddings
            image_embedding (Tensor, optional): Image feature embedding (only during training)
        """
        depth_embedding = self.depth_encoder(depth_map)  # Always compute depth embedding
        
        text_embeddings = self.text_encoder(class_descriptions)  # Always compute text embeddings
        
        if torch.isnan(depth_embedding).any() or torch.isinf(depth_embedding).any():
            print("NaN or Inf detected in depth embeddings")

        if image is not None:  # Training mode: also compute image embeddings
            image_embedding = self.image_encoder(image)
            return depth_embedding, image_embedding, text_embeddings
        else:  # Inference mode: Only return depth & text embeddings
            return depth_embedding, text_embeddings


    def predict(self, depth_map, class_descriptions):
        """
        Predict class of an object given a depth map and possible text class descriptions.
        
        Args:
            depth_map (Tensor): Depth map input (1, 1, H, W)
            class_descriptions (List[str]): List of text class descriptions
        
        Returns:
            predicted_class (int): Index of the predicted class
            confidence_score (float): Softmax probability of the predicted class
        """
        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            depth_embedding, text_embeddings = self(depth_map, class_descriptions)  # No image input

            # Compute similarity
            logits = torch.matmul(depth_embedding, text_embeddings.T) / self.temperature.exp()
            confidence_scores = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(confidence_scores, dim=-1)

            return predicted_class.item(), confidence_scores.max().item()

    def compute_loss(self, depth_maps, images, labels, w_text=0.5):
        """
        Compute contrastive loss aligning depth with text & depth with image.
        Handles batch processing for improved contrastive learning.
        
        Args:
            depth_maps (Tensor): Batch of depth maps (batch_size, 1, H, W)
            images (Tensor): Batch of RGB images (batch_size, 3, H, W)
            class_descriptions (List[str]): List of all possible class descriptions
            ground_truth_indices (Tensor): Ground truth indices for each example in batch
            w_text (float): Weight for text loss (depth-to-image loss weight will be 1-w_text)
            
        Returns:
            loss (Tensor): Combined contrastive loss
            loss_info (dict): Dictionary containing loss components
        """
        batch_size = depth_maps.shape[0]
        
        try:
            # Get normalized embeddings for all modalities
            if images is not None:
                depth_embeddings, image_embeddings, text_embeddings = self.forward(
                    depth_maps, labels, images
                )
            else:
                depth_embeddings, text_embeddings = self.forward(
                    depth_maps, labels
                )
                
            check_extreme(depth_embeddings, 'depth embedding')
            check_extreme(text_embeddings, 'text embedding')
            # Depth-to-Text Contrastive Loss (InfoNCE)
            # Calculate similarity with numerical stability check
            sim_d_t = torch.matmul(depth_embeddings, text_embeddings.T)
            check_extreme(sim_d_t, 'sim_d_t')
            # Check similarity values

            # Apply temperature with a safety check
            temp_value = self.temperature.exp().item()
            if temp_value < 1e-10 or temp_value > 1e10:
                print(f"WARNING: Extreme temperature value: {temp_value}")
                # Use a safe default instead
                sim_d_t = sim_d_t / 0.07  # Common default temperature
            else:
                sim_d_t = sim_d_t / self.temperature.exp()
            
            check_extreme(sim_d_t, 'post temp sim_d_t')
            
            # Cross entropy loss with ground truth indices
            try:
                gt = torch.arange(batch_size, device='cuda:0')

                loss_d_t = F.cross_entropy(sim_d_t, gt)
                # print(f"======= Loss is {loss_d_t} ========")
                
                # Check if loss is valid
                if torch.isnan(loss_d_t) or torch.isinf(loss_d_t) or loss_d_t < 1e-10:
                    print(f"WARNING: Problematic depth-text loss value: {loss_d_t.item()}\nsim_d_t: {sim_d_t}\n\nground_truth: {ground_truth_indices}")
                    loss_d_t = torch.tensor(1.0, requires_grad=True, device=self.device)  # Safe fallback
            except Exception as e:
                print(f"ERROR in cross entropy calculation: {e}\nsim_d_t: {sim_d_t}\n\nground_truth: {ground_truth_indices}")
                print(f"sim_d_t shape: {sim_d_t}, ground_truth_indices shape: {ground_truth_indices.shape}")
                loss_d_t = torch.tensor(1.0, requires_grad=True, device=self.device)  # Safe fallback
            
            loss_info = {}
            
            if images is not None:
                # Calculate similarity between all pairs of depth and image embeddings
                sim_d_i = torch.matmul(depth_embeddings, image_embeddings.T)
                
                check_extreme(sim_d_i, 'sim_d_i')
                # Check similarity values
                if torch.isnan(sim_d_i).any() or torch.isinf(sim_d_i).any():
                    print("WARNING: NaN or Inf values in depth-image similarity matrix")
                    sim_d_i = torch.zeros_like(sim_d_i)
                
                # Apply temperature with safety check
                sim_d_i = sim_d_i / self.temperature.exp()
                check_extreme(sim_d_i, 'post temp sim_d_i')
                
                # Check modified similarity values
                if torch.isnan(sim_d_i).any() or torch.isinf(sim_d_i).any():
                    print("WARNING: NaN or Inf values after temperature scaling (d-i)")
                    sim_d_i = torch.zeros_like(sim_d_i)
                
                try:
                    # InfoNCE loss - each depth map should match with its corresponding image
                    targets = torch.arange(batch_size, device=self.device)
                    loss_d_i = F.cross_entropy(sim_d_i, targets)
                    
                    # Check if loss is valid
                    if torch.isnan(loss_d_i) or torch.isinf(loss_d_i) or loss_d_i < 1e-10:
                        print(f"WARNING: Problematic depth-image loss value: {loss_d_i.item()}")
                        loss_d_i = torch.tensor(1.0, requires_grad=True, device=self.device)  # Safe fallback
                except Exception as e:
                    print(f"ERROR in depth-image cross entropy calculation: {e}")
                    loss_d_i = torch.tensor(1.0, requires_grad=True, device=self.device)  # Safe fallback
                    
                loss_info['loss_d_i'] = loss_d_i.item()
                # Combine losses with stability check
                loss = w_text * loss_d_t + (1 - w_text) * loss_d_i
            else:
                loss = loss_d_t
            
            # Final safety check
            if torch.isnan(loss) or torch.isinf(loss) or loss < 1e-10:
                print(f"WARNING: Final loss problematic: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                return torch.tensor(1.0, requires_grad=True, device=self.device), {'loss_d_i': 1.0, 'loss': 1.0, 'loss_d_t': 1.0}
            
            loss_info['loss'] = loss.item()
            loss_info['loss_d_t'] = loss_d_t.item()
            
            # print(f"Ground truth indices: {ground_truth_indices}\n")
            return loss, loss_info
            
        except Exception as e:
            check_extreme(sim_d_t, 'sim_d_t_exception')
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Error details: {repr(e)}")
            import traceback
            traceback.print_exc()
            # Return safe values that can be backpropagated
            return torch.tensor(1.0, requires_grad=True, device=self.device), {'loss_d_i': 1.0, 'loss': 1.0, 'loss_d_t': 1.0}
    
    def compute_loss2(self, depth_maps, images, class_descriptions, ground_truth_indices, w_text=0.5):
        """
        Compute contrastive loss aligning depth with text & depth with image.
        Handles batch processing for improved contrastive learning.
        
        Args:
            depth_maps (Tensor): Batch of depth maps (batch_size, 1, H, W)
            images (Tensor): Batch of RGB images (batch_size, 3, H, W)
            class_descriptions (List[str]): List of all possible class descriptions
            ground_truth_indices (Tensor): Ground truth class indices for each example in batch
            w_text (float): Weight for text loss (depth-to-image loss weight will be 1-w_text)
            
        Returns:
            loss (Tensor): Combined contrastive loss
            loss_d_t (Tensor): Depth-to-text loss component
            loss_d_i (Tensor): Depth-to-image loss component
        """
        batch_size = depth_maps.shape[0]
        
        # Get normalized embeddings for all modalities
        if images is not None:
            depth_embeddings, image_embeddings, text_embeddings = self.forward(
                depth_maps, class_descriptions, images
            )
        else:
            depth_embeddings, text_embeddings = self.forward(
                depth_maps, class_descriptions, images
            )
            
        # Depth-to-Text Contrastive Loss (InfoNCE)
        # Calculate similarity between each depth embedding and all text embeddings
        sim_d_t = torch.matmul(depth_embeddings, text_embeddings.T) / self.temperature.exp()
        
        # Cross entropy loss with ground truth indices
        loss_d_t = F.cross_entropy(sim_d_t, ground_truth_indices.to(self.device))
        
        loss_info = {}
        
        if images is not None:
            # Calculate similarity between all pairs of depth and image embeddings
            sim_d_i = torch.matmul(depth_embeddings, image_embeddings.T) / self.temperature.exp()
            
            # InfoNCE loss - each depth map should match with its corresponding image
            loss_d_i = F.cross_entropy(sim_d_i, torch.arange(batch_size, device=self.device))
            loss_info['loss_d_i'] = loss_d_i.item()
            # Combine losses
            loss = w_text * loss_d_t + (1 - w_text) * loss_d_i
        else:
            loss = loss_d_t
        
        loss_info['loss'] = loss.item()
        loss_info['loss_d_t'] = loss_d_t.item()
        
        return loss, loss_info
    
    
    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return list(self.depth_encoder.parameters()) + list(self.text_encoder.parameters()) + list(self.image_encoder.parameters())

    def train(self):
        '''
        Sets model to training mode
        '''
        self.depth_encoder.train()
        pass

    def eval(self):
        '''
        Sets model to evaluation mode
        '''
        self.depth_encoder.eval()
        pass

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.depth_encoder.to(device)
        self.text_encoder.to(device)
        self.image_encoder.to(device)

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                lists of paths to model weights
            optimizer : torch.optim or None
                current optimizer
        Returns:
            int : training step
            torch.optim : restored optimizer or None if no optimizer is passed in
        '''

        checkpoint = torch.load(restore_path, map_location=self.device)
        
        if isinstance(self, torch.nn.DataParallel):
            self.module.depth_encoder.load_state_dict(checkpoint['encoder'])  # Corrected line
        else:
            self.depth_encoder.load_state_dict(checkpoint['encoder'])
        
        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        train_step = checkpoint.get('train_step', 0)
        
        return train_step, optimizer


    def save_model(self, checkpoint_path, step, optimizer=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                list path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        checkpoint = {
            'train_step': step
        }
        
        if isinstance(self, torch.nn.DataParallel):
            checkpoint['encoder'] = self.module.depth_encoder.state_dict()
        else:
            checkpoint['encoder'] = self.depth_encoder.state_dict()
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)

    def log_summary(self,
                summary_writer,
                tag,
                step,
                image,
                depth,
                text,
                scalars={},
                n_sample_per_summary=4):

        with torch.no_grad():
            display_summary_image = []

            # Log input image
            if image is not None:
                input_image_summary = image[:min(len(image), n_sample_per_summary)]
                input_image_summary = input_image_summary.to('cpu')
                display_summary_image.append(input_image_summary)

            # Log depth map with colormap
            if depth is not None:
                depth_summary = depth[:min(len(depth), n_sample_per_summary)]
                depth_colored = apply_colormap(depth_summary)  # Apply colormap
                depth_colored = depth_colored.to('cpu')
                display_summary_image.append(depth_colored)

            # Log scalars
            for name, value in scalars.items():
                summary_writer.add_scalar(f"{tag}_{name}", value, global_step=step)

            # Log text to TensorBoard
            if text is not None:
                # Create a concatenated string of the text for each sample
                text_summary = "\n".join([f"{i+1}: {t}" for i, t in enumerate(text[:min(len(text), n_sample_per_summary)])])
                summary_writer.add_text(f"{tag}_text", text_summary, global_step=step)

            # Log images to TensorBoard
            if display_summary_image:
                display_summary_image = torch.cat(display_summary_image, dim=2)  # Concatenate along width
                summary_writer.add_image(
                    tag,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_sample_per_summary),
                    global_step=step)
