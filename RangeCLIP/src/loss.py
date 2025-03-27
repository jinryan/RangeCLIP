from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class ClipLoss2D(nn.Module):

    def __init__(self, image_loss_weight=0.8):
        super().__init__()

        self.image_loss_weight=image_loss_weight


    def forward(self, image_embeddings, depth_embeddings, logit_scale=10, output_dict=False):
        device = image_embeddings.device
        depth_image_sim = logit_scale * torch.matmul(depth_embeddings, image_embeddings.T)
        
        batch_size = len(depth_embeddings)
        labels = torch.arange(batch_size).to(device)
        
        loss_depth_image = F.cross_entropy(depth_image_sim, labels)
        loss_image_depth = F.cross_entropy(depth_image_sim.T, labels)

        total_loss = (loss_depth_image + loss_image_depth) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
class ClipLoss3D(nn.Module):

    def __init__(self, image_loss_weight=0.8):
        super().__init__()

        self.image_loss_weight=image_loss_weight

    def get_similarity_matrices(self, image_embeddings, text_embeddings, depth_embeddings, logit_scale):
        depth_image_sim = logit_scale * torch.matmul(depth_embeddings, image_embeddings.T)
        depth_text_sim = logit_scale * torch.matmul(depth_embeddings, text_embeddings.T)
        
        return depth_image_sim, depth_text_sim

    def forward(self, image_embeddings, text_embeddings, depth_embeddings, logit_scale=10, output_dict=False):
        device = image_embeddings.device
        depth_image_sim, depth_text_sim = self.get_similarity_matrices(image_embeddings, text_embeddings, depth_embeddings, logit_scale)
        
        batch_size = len(depth_embeddings)
        labels = torch.arange(batch_size).to(device)
        
        # Compute losses for each modality pair
        loss_depth_image = F.cross_entropy(depth_image_sim, labels)
        loss_depth_text = F.cross_entropy(depth_text_sim, labels)
        
        # Symmetric losses
        loss_image_depth = F.cross_entropy(depth_image_sim.T, labels)
        loss_text_depth = F.cross_entropy(depth_text_sim.T, labels)

        total_loss = (
            self.image_loss_weight * (loss_depth_image + loss_image_depth) + # Image losses
            (1- self.image_loss_weight) * (loss_depth_text + loss_text_depth) # Text losses
        ) / 4

        return {"contrastive_loss": total_loss} if output_dict else total_loss