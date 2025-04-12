import os
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import CLIPModel

from utils.src.networks import DepthEncoder, TextEncoder, ImageEncoder
from utils.src.log_utils import validate_tensor





def normalize_depth_map(depth_map, eps=1e-8):
    """
    Normalize depth map to 0-1 range with protection against zero division.
    
    Args:
        depth_map: Input depth map tensor
        eps: Small epsilon value to prevent division by zero
    
    Returns:
        Normalized depth map
    """
    min_val = depth_map.amin(dim=(1, 2, 3), keepdim=True)
    max_val = depth_map.amax(dim=(1, 2, 3), keepdim=True)

    # Check if the range is too small (flat depth map)
    if (max_val - min_val < eps).any():
        # Only normalize depth maps with sufficient range
        normalized = torch.zeros_like(depth_map)
        valid_maps = (max_val - min_val >= eps).squeeze()
        if valid_maps.sum() > 0:
            normalized[valid_maps] = (depth_map[valid_maps] - min_val[valid_maps]) / (max_val[valid_maps] - min_val[valid_maps])
        return normalized
    
    # Otherwise normalize as usual
    return (depth_map - min_val) / (max_val - min_val + eps)


class DepthClipModel(nn.Module):
    def __init__(self, 
                 depth_encoder_type,
                 clip_model_name,
                 device,
                 temperature=0.07):
        """
        Open-vocabulary depth map classifier using contrastive learning.
        
        Args:
            depth_encoder_type (str): Type of depth encoder ('resnet', etc.)
            clip_model_name (str): Name of CLIP model to use
            device: Device to run model on
            temperature (float): Temperature scaling for contrastive learning
        """
        super().__init__()
        
        self.device = device
        
        # Initialize CLIP-based encoders
        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            embedding_dim = clip_model.config.projection_dim
            self.text_encoder = TextEncoder(clip_model, device)
            self.image_encoder = ImageEncoder(clip_model)
        except Exception as e:
            raise ValueError(f'Failed to initialize CLIP model {clip_model_name}: {str(e)}')
        
        # Initialize depth encoder
        if depth_encoder_type == 'resnet':
            self.depth_encoder = DepthEncoder(n_layer=18,
                                              input_channels=1,
                                              n_filters=[32, 64, 128, 256, 512],
                                              embedding_dim=embedding_dim,
                                              weight_initializer='kaiming_uniform',
                                              activation_func='relu',
                                              use_instance_norm=False,
                                              use_batch_norm=True)
        else:
            raise ValueError(f'Unsupported depth encoder type: {depth_encoder_type}')
        
        # Temperature parameter for scaling logits
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
        # Move models to device
        self.to(device)
        

    def forward(self,
                depth_map: torch.Tensor,
                class_descriptions: Optional[List[str]] = None,
                image: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if image is not None and image.size(0) != depth_map.size(0):
            raise ValueError("Batch size mismatch between depth_map and image.")

        with autocast():
            depth_embedding = self.depth_encoder(depth_map)
            
            text_embeddings = None
            if class_descriptions is not None:
                text_embeddings = self.text_encoder(class_descriptions)

            image_embedding = None
            if image is not None:
                image_embedding = self.image_encoder(image)

        return depth_embedding, text_embeddings, image_embedding


    def predict_image(self, image, class_descriptions, top_k=5):
        """
        Predicts top-k classes for a batch of images.

        Args:
            image: Batch of images with shape [B, C, H, W]
            class_descriptions: List of text class descriptions
            top_k: Number of top predictions to return

        Returns:
            predicted_classes: Top-k class indices for each image
            top_k_confidence_scores: Confidence scores for each prediction
        """
        with torch.no_grad():
            # Get embeddings
            image_embeddings = self.image_encoder(image)  # Shape: [B, D]
            text_embeddings = self.text_encoder(class_descriptions)  # Shape: [N, D]

            # Normalize embeddings
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

            # Compute similarity logits: [B, N]
            logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
            confidence_scores = F.softmax(logits, dim=-1)

            # Top-k prediction: [B, top_k]
            top_k_scores, top_k_indices = torch.topk(confidence_scores, k=min(top_k, len(class_descriptions)), dim=-1)

            return top_k_indices, top_k_scores
        
    def predict(self, depth_map, class_descriptions, top_k=5):
        """
        Predicts top-k classes for a batch of depth maps.
        
        Args:
            depth_map: Tensor of shape [B, 1, H, W]
            class_descriptions: List of strings (length N)
            top_k: Number of top predictions to return
            
        Returns:
            predicted_classes: Indices of top-k predictions
            top_k_confidence_scores: Confidence scores for top-k predictions
        """
        with torch.no_grad():
            # Get embeddings
            depth_embeddings = self.depth_encoder(depth_map)           # [B, D]
            text_embeddings = self.text_encoder(class_descriptions)    # [N, D]

            # Normalize embeddings
            depth_embeddings = F.normalize(depth_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

            # Compute similarity and get top-k predictions
            logits = torch.matmul(depth_embeddings, text_embeddings.T) / self.temperature
            confidence_scores = F.softmax(logits, dim=-1)
            top_k_scores, top_k_indices = torch.topk(confidence_scores, k=min(top_k, len(class_descriptions)), dim=-1)

            return top_k_indices, top_k_scores

    @property
    def temperature(self):
        """
        Get the temperature value (exp of the log_temperature parameter).
        This ensures temperature is always positive.
        """
        return torch.exp(self.log_temperature)

    def compute_trimodal_loss(self, embedding_1, embedding_2, embedding_3, ground_truth_indices, w_1_2=0.2):
        """
        Compute contrastive loss between three modalities.
        
        Args:
            embedding_1: Depth embeddings [B, D]
            embedding_2: Image embeddings [B, D]
            embedding_3: Text embeddings [N, D]
            ground_truth_indices: Ground truth indices for text embeddings
            w_1_2: Weight balancing factor between depth-text and depth-image losses
            
        Returns:
            loss: Combined loss value
            loss_info: Dictionary of individual loss components
        """
        
        batch_size = embedding_1.shape[0]
        with autocast():
            # Normalize embeddings for cosine similarity
            depth_embed_norm = F.normalize(embedding_1, dim=-1)
            image_embed_norm = F.normalize(embedding_2, dim=-1)
            text_embed_norm = F.normalize(embedding_3, dim=-1)
            
            # Validate embeddings
            validate_tensor(depth_embed_norm, 'Normalized Depth Embeddings')
            validate_tensor(image_embed_norm, 'Normalized Image Embeddings')
            validate_tensor(text_embed_norm, 'Normalized Text Embeddings')
            
            # Loss between depth and text
            sim_d_t = torch.matmul(depth_embed_norm, text_embed_norm.T) / self.temperature
            loss_d_t = F.cross_entropy(sim_d_t, ground_truth_indices)
            
            # Loss between depth and image
            sim_d_i = torch.matmul(depth_embed_norm, image_embed_norm.T) / self.temperature
            targets = torch.arange(batch_size, device=self.device)
            loss_d_i = F.cross_entropy(sim_d_i, targets)
            
            # Combined loss
            loss = w_1_2 * loss_d_t + (1 - w_1_2) * loss_d_i
            loss_info = {
                'loss_d_i': loss_d_i.item(), 
                'loss_d_t': loss_d_t.item(), 
                'trimodal_loss': loss.item(),
                'temperature': self.temperature.item()
            }
        
        return loss, loss_info
        
    def compute_bimodal_loss(self, embedding_1, embedding_2):
        """
        Compute contrastive loss between two modalities.
        
        Args:
            embedding_1: First embeddings [B, D]
            embedding_2: Second embeddings [B, D]
            
        Returns:
            loss: Contrastive loss value
            loss_info: Dictionary with loss information
        """
        batch_size = embedding_1.shape[0]
        
        with autocast():
            # Normalize embeddings
            norm_embedding_1 = F.normalize(embedding_1, dim=-1)
            norm_embedding_2 = F.normalize(embedding_2, dim=-1)
            
            # Compute similarity and loss
            logits = torch.matmul(norm_embedding_1, norm_embedding_2.T) / self.temperature
            targets = torch.arange(batch_size, device=self.device)
            loss = F.cross_entropy(logits, targets)
            
            loss_info = {
                "bimodal_loss": loss.item(),
                "temperature": self.temperature.item()
            }
            
        return loss, loss_info
    
    def get_text_encoder(self):
        """Returns the text encoder component."""
        return self.text_encoder
    
    def get_image_encoder(self):
        """Returns the image encoder component."""
        return self.image_encoder
    
    def get_depth_encoder(self):
        """Returns the depth encoder component."""
        return self.depth_encoder

    def train(self, mode=True):
        """
        Sets model to training mode.
        """
        self.depth_encoder.train(mode)
        return self

    def eval(self):
        """Sets model to evaluation mode."""
        self.depth_encoder.eval()
        return self

    def restore_model(self, restore_path, optimizer=None):
        """
        Loads weights from checkpoint.
        
        Args:
            restore_path: Path to model weights
            optimizer: Current optimizer (optional)
            
        Returns:
            train_step: Training step from checkpoint
            optimizer: Restored optimizer or None
        """
        try:
            checkpoint = torch.load(restore_path, map_location=self.device)
            
            # Load depth encoder weights
            if isinstance(self, torch.nn.DataParallel):
                self.module.depth_encoder.load_state_dict(checkpoint['encoder'])
            else:
                self.depth_encoder.load_state_dict(checkpoint['encoder'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Get training step
            train_step = checkpoint.get('train_step', 0)
            
            print(f"Model restored from {restore_path} at step {train_step}")
            return train_step, optimizer
            
        except Exception as e:
            print(f"Error restoring model from {restore_path}: {str(e)}")
            return 0, optimizer

    def save_model(self, checkpoint_path, step, optimizer=None):
        """
        Save weights of the model to checkpoint path.
        
        Args:
            checkpoint_path: Path to save checkpoint
            step: Current training step
            optimizer: Optimizer to save (optional)
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'train_step': step
        }
        
        # Save depth encoder weights
        if isinstance(self, torch.nn.DataParallel):
            checkpoint['encoder'] = self.module.depth_encoder.state_dict()
        else:
            checkpoint['encoder'] = self.depth_encoder.state_dict()
        
        # Save optimizer state if provided
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Model saved to {checkpoint_path} at step {step}")