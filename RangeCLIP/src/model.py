import os, sys
import torch, torchvision
sys.path.insert(0, os.getcwd())
from utils.src.networks import DepthEncoder, TextEncoder
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
        
        # Store the encoders
        if depth_encoder_type == 'resnet':
            self.depth_encoder = DepthEncoder(n_layer=18,
                                              input_channels=1,
                                              n_filters=[32, 64, 128, 256, 512],
                                              embedding_dim=776,
                                              weight_initializer='kaiming_uniform',
                                              activation_func='leaky_relu',
                                              use_batch_norm=True)
        else:
            raise ValueError(f'Unsupported depth encoder {depth_encoder_type}')
        
        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.text_encoder = TextEncoder(clip_model)
        except:
            raise ValueError(f'Unsupported CLIP model {clip_model_name}')
        
        self.depth_encoder.to(self.device)
        self.text_encoder.to(self.device)
        
        # Learnable temperature parameter for scaling logits
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, depth_map, class_descriptions):
        """
        Classify depth map against given class descriptions.
        """
        depth_embedding = self.depth_encoder(depth_map)
        text_embeddings = self.text_encoder(class_descriptions)
        
        # Compute similarity between depth embedding and class text embeddings
        # We use dot product as our similarity metric
        logits = torch.matmul(depth_embedding, text_embeddings.T) / self.temperature.exp()
        
        return logits

    def predict(self, depth_map, class_descriptions):
        logits = self.forward(depth_map, class_descriptions)
        confidence_scores = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(confidence_scores, dim=-1)
        
        return predicted_class.item(), confidence_scores.max().item()

    def compute_loss(self, depth_map, class_descriptions, ground_truth_index):
        """
        Compute contrastive loss for training, though only between depth and text
        """
        
        logits = self.forward(depth_map, class_descriptions)
        loss = F.cross_entropy(logits, torch.tensor([ground_truth_index], device=self.device))
        
        return loss
    
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

        return list(self.depth_encoder.parameters()) + list(self.text_encoder.parameters())

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