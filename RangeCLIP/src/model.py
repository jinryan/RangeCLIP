import os, sys
from typing import Optional, List, Tuple
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
        
        # For scaling logits
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(
        self,
        depth_map: torch.Tensor,
        class_descriptions: Optional[List[str]] = None,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass: Handles both training (with images) and inference (depth + text only).
        
        Args:
            depth_map (Tensor): Depth input (batch, 1, H, W)
            class_descriptions (List[str], optional): List of text class descriptions
            image (Tensor, optional): RGB image (batch, 3, H, W). Default is None for inference.
        
        Returns:
            depth_embedding (Tensor): Depth feature embedding
            text_embeddings (Tensor, optional): Class description embeddings
            image_embedding (Tensor, optional): Image feature embedding
        """
        
        if image is not None and image.size(0) != depth_map.size(0):
            raise ValueError("Batch size mismatch between depth_map and image.")
        
        depth_embedding = self.depth_encoder(depth_map)
        
        text_embeddings = None
        image_embedding = None
        if class_descriptions is not None:
            text_embeddings = self.text_encoder(class_descriptions)

        if image is not None:
            image_embedding = self.image_encoder(image)
            
        return depth_embedding, text_embeddings, image_embedding


    def predict(self, depth_map, class_descriptions):
        """
        Predict class of an object given a depth map and possible text class descriptions.
        
        Args:
            depth_map (Tensor): Depth map input (1, 1, H, W)
            candidate_labels (List[str]): List of text class descriptions
        
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

    def compute_trimodal_loss(self,
                     depth_embeddings,
                     image_embeddings,
                     candidate_labels,
                     ground_truth_indices,
                     w_text=0.8):
        
        batch_size = depth_embeddings.shape[0]
        text_embeddings = self.text_encoder(candidate_labels)
        
        check_extreme(depth_embeddings, 'Depth Embeddings')
        check_extreme(text_embeddings, 'Text Embeddings')
        check_extreme(image_embeddings, 'Image Embeddings')
        
        # Loss between depth and text
        sim_d_t = torch.matmul(depth_embeddings, text_embeddings.T)
        check_extreme(sim_d_t, 'sim_d_t')
        
        temp_value = self.temperature.exp().item()
        
        if temp_value < 1e-10 or temp_value > 1e10:
            print(f"WARNING: Extreme temperature value: {temp_value}")
        
        sim_d_t = sim_d_t / self.temperature.exp()
        check_extreme(sim_d_t, 'post temp sim_d_t')
        
        loss_d_t = F.cross_entropy(sim_d_t, ground_truth_indices)
        
        if torch.isnan(loss_d_t) or torch.isinf(loss_d_t) or loss_d_t < 1e-10:
            print(f"WARNING: Problematic depth-text loss value: {loss_d_t.item()}\nsim_d_t: {sim_d_t}\n\nground_truth: {ground_truth_indices}")
            loss_d_t = torch.tensor(1.0, requires_grad=True, device=self.device)  # Safe fallback
            
        # Loss between depth and image
        sim_d_i = torch.matmul(depth_embeddings, image_embeddings.T)
        check_extreme(sim_d_i, 'sim_d_i')
        sim_d_i = sim_d_i / self.temperature.exp()
        check_extreme(sim_d_i, 'post temp sim_d_i')
        
        targets = torch.arange(batch_size, device=self.device)
        loss_d_i = F.cross_entropy(sim_d_i, targets)
        
        if torch.isnan(loss_d_i) or torch.isinf(loss_d_i) or loss_d_i < 1e-10:
            print(f"WARNING: Problematic depth-image loss value: {loss_d_i.item()}\nsim_d_i: {sim_d_i.item()}\ntargets: {targets.item()}")
            loss_d_i = torch.tensor(1.0, requires_grad=True, device=self.device)  # Safe fallback
        
        loss = w_text * loss_d_t + (1 - w_text) * loss_d_i
        loss_info = {'loss_d_i': loss_d_i.item(), 'loss_d_t': loss_d_t.item(), 'trimodal_loss': loss.item()}
        
        return loss, loss_info
        
        
    def compute_bimodal_loss(self, depth_embeddings, other_embeddings):
        batch_size = depth_embeddings.shape[0]
        logits = torch.matmul(F.normalize(depth_embeddings, dim=-1),
                      F.normalize(other_embeddings, dim=-1).T) / self.temperature.exp()
        targets = torch.arange(batch_size, device=self.device)
        loss = F.cross_entropy(logits, targets)
        loss_info = {"bimodal_loss": loss}
        return loss, loss_info
        
    
    # def parameters(self):
    #     '''
    #     Returns the list of parameters in the model

    #     Returns:
    #         list[torch.Tensor[float32]] : list of parameters
    #     '''

    #     return list(self.depth_encoder.parameters()) + list(self.text_encoder.parameters()) + list(self.image_encoder.parameters())

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