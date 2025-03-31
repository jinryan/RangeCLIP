import os, sys
import torch, torchvision
sys.path.insert(0, os.getcwd())
from utils.src.networks import DepthEncoder, TextEncoder, ImageEncoder
import utils.src.loss_utils as loss_utils
import utils.src.log_utils as log_utils
from transformers import CLIPModel


import torch
import torch.nn as nn
import torch.nn.functional as F

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
            self.text_encoder = TextEncoder(clip_model)
            self.image_encoder = ImageEncoder(clip_model)
            clip_model = CLIPModel.from_pretrained(clip_model_name)
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
                                              use_batch_norm=True)
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

    def compute_loss(self, depth_maps, images, class_descriptions, ground_truth_indices, w_text=0.5):
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
        
        
        loss_d_i = 0
        if images is not None:
            # Calculate similarity between all pairs of depth and image embeddings
            sim_d_i = torch.matmul(depth_embeddings, image_embeddings.T) / self.temperature.exp()
            
            # InfoNCE loss - each depth map should match with its corresponding image
            loss_d_i = F.cross_entropy(sim_d_i, torch.arange(batch_size, device=self.device))
        
        # Combine losses
        loss = w_text * loss_d_t + (1 - w_text) * loss_d_i
        
        return loss, loss_d_t, loss_d_i
    
    def transform_input(self, depth_maps, images, class_names):
        '''
        Transforms input based on model arguments and settings

        Arg(s):
            depth_maps : torch.Tensor[float32]
                N x 1 x H x W images
            images : torch.Tensor[float32]
                N x C x H x W images
        Returns:
            torch.Tensor[float32] : transformed N x C x H x W images
        '''
        
        depth_normalized = [
            (dm - dm.min()) / (dm.max() - dm.min()) 
            for dm in depth_maps
        ]

        return depth_normalized, images, class_names
    
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

        # TODO: Restore the weights from checkpoint
        pass

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

        # TODO: Save the weights into checkpoint
        pass

    def log_summary(self):
        # TODO
        pass