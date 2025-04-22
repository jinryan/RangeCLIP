import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils.src.encoder import DepthEncoder
from utils.src.decoder import DepthDecoder
from torch.cuda.amp import autocast
import numpy as np

class DepthUNet(nn.Module):
    """
    UNet architecture for depth map segmentation
    
    Arg(s):
        unet_type (str):
            Type of encoder/decoder ('resnet', etc.)
        clip_model_name (str):
            Name of CLIP model to use
        device: Device to run model on
        n_layer : int
            ResNet encoder architecture: 18, 34, 50
        input_channels : int
            number of input channels in the depth map
        encoder_filters : list
            number of filters for each encoder block
        decoder_filters : list
            number of filters for each decoder block
        embedding_dim : int
            dimension of the encoder's output embedding
        weight_initializer : str
            weight initialization method
        activation_func : str
            activation function to use
        use_batch_norm : bool
            if set, use batch normalization
        use_instance_norm : bool
            if set, use instance normalization
        temperature (float):
            Temperature scaling for contrastive learning
    """
    def __init__(self,
                 unet_type,
                 device,
                 n_layer=18,
                 input_channels=1,
                 encoder_filters=[32, 64, 128, 256, 512],
                 embedding_dim=256,
                 weight_initializer='kaiming_uniform',
                 activation_func='relu',
                 use_batch_norm=False,
                 use_instance_norm=False,
                 temperature=0.07):
        super(DepthUNet, self).__init__()
        
        self.device = device
        
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
        if unet_type == 'resnet':
            decoder_filters = encoder_filters[::-1]
            
            self.freeze_encoder = False
            self.depth_encoder = DepthEncoder(
                n_layer=n_layer,
                input_channels=input_channels,
                n_filters=encoder_filters,
                embedding_dim=embedding_dim,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm
            )
            
            self.depth_decoder = DepthDecoder(
                n_filters=decoder_filters,
                encoder_filters=encoder_filters,
                embedding_dim=embedding_dim,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm
            )
        else:
            raise ValueError(f'Unsupported depth encoder type: {unet_type}')
        
    
    def forward(self, depth_map):
        """
        Forward pass of the UNet model
        
        Arg(s):
            depth_map : torch.Tensor
                input depth map tensor
        
        Returns:
            torch.Tensor : segmentation output
        """
        with autocast():
            target_shape = depth_map.shape[2:]
            _, encoder_features, final_feature_map = self.depth_encoder(depth_map)        
            output = self.depth_decoder(final_feature_map, encoder_features, target_shape)
        
        return output, self.temperature
    
    def predict(self, depth_maps, candidate_text_embeddings, segmentation, num_negatives=300):
        """
        Predict segmentation labels for the input depth maps using a reduced candidate set.

        Args:
            depth_maps (torch.Tensor): input depth maps [B, 1, H, W]
            candidate_text_embeddings (torch.Tensor): all text embeddings [C, D]
            segmentation (torch.Tensor): ground truth segmentation [B, H, W], used to extract true label indices
            num_negatives (int): number of additional negative labels to sample

        Returns:
            pred_indices (torch.LongTensor): predicted label indices in the ORIGINAL index space [B, H, W]
            pixel_embeddings (torch.Tensor): output embeddings from the model [B, D, H, W]
            temperature (float): learned temperature parameter
        """
        self.eval()
            
        B, _, H, W = depth_maps.shape
        total_candidates, D = candidate_text_embeddings.shape

        with torch.no_grad(), autocast():
            # Forward pass
            _, encoder_features, final_feature_map = self.depth_encoder(depth_maps)
            pixel_embeddings = self.depth_decoder(final_feature_map, encoder_features, (H, W))  # [B, D, H, W]
            pixel_embeddings = F.normalize(pixel_embeddings, dim=1)

            # --- Step 1: Build reduced candidate set ---
            if segmentation is not None:
                unique_labels = torch.unique(segmentation).tolist()
            else:
                raise ValueError("segmentation must be provided for reduced-candidate prediction")

            all_indices = list(range(total_candidates))
            gt_indices = set(unique_labels)
            sample_pool = list(set(all_indices) - gt_indices)
            sampled_negatives = random.sample(sample_pool, min(num_negatives, len(sample_pool)))
            reduced_indices = sorted(list(gt_indices.union(sampled_negatives)))

            # Mapping from reduced -> global
            index_tensor = torch.tensor(reduced_indices, device=depth_maps.device)  # [C_reduced]
            reduced_candidate_embeddings = candidate_text_embeddings[index_tensor]  # [C_reduced, D]
            reduced_candidate_embeddings = F.normalize(reduced_candidate_embeddings, dim=1)

            # --- Step 2: Predict using reduced candidates ---
            pixel_flat = pixel_embeddings.view(B, D, H * W)
            logits = torch.einsum('bdn,cd->bcn', pixel_flat, reduced_candidate_embeddings)
            pred_indices_reduced = logits.argmax(dim=1).view(B, H, W)  # [B, H, W]

            # --- Step 3: Map reduced indices back to original index space ---
            pred_indices = index_tensor[pred_indices_reduced]  # [B, H, W], now in original index space

            return pred_indices, pixel_embeddings, self.temperature

        

    def compute_loss(
        self,
        pred,
        target_indices,
        candidate_text_embeddings,
        temperature,
        label_similarity_sets,
        percent_image=0.7,
        k_distractors=100,
        pct_medium=0.0,
        pct_hard=0.75,
        pct_rand=0.25,
        lambda_smooth=1e2,
    ):
        """
        Hybrid contrastive loss with difficulty-aware distractors.
        """
        assert abs(pct_medium + pct_hard + pct_rand - 1.0) < 1e-4, "Sum of percentages must be 1."

        B, D, H, W = [int(s) for s in pred.shape]
        C = candidate_text_embeddings.shape[0]
        device = pred.device
        num_samples = int(percent_image * H * W)

        # Flatten and sample
        pred_flat = pred.reshape(B, D, -1)
        target_flat = target_indices.reshape(B, -1)
        rand_indices = torch.randint(0, H * W, (B, num_samples), device=device)
        pred_samples = torch.gather(pred_flat, 2, rand_indices.unsqueeze(1).expand(-1, D, -1))
        label_samples = torch.gather(target_flat, 1, rand_indices)
        pred_samples = pred_samples.transpose(1, 2).reshape(-1, D)
        label_samples = label_samples.reshape(-1)

        if pred_samples.numel() == 0:
            raise RuntimeError("No valid pixels found for loss computation.")

        unique_labels = torch.unique(label_samples)

        distractor_indices = set()

        n_medium = int(k_distractors * pct_medium)
        n_hard = int(k_distractors * pct_hard)
        n_rand = k_distractors - n_medium - n_hard

        # Get medium/hard sets
        if n_medium > 0:
            for label in unique_labels.tolist():
                distractor_indices.update(label_similarity_sets['medium'][label])
        if n_hard > 0:
            for label in unique_labels.tolist():
                distractor_indices.update(label_similarity_sets['hard'][label])

        distractor_indices = list(distractor_indices)
        distractor_indices = [d for d in distractor_indices if d not in unique_labels]

        medium_and_hard = (
            np.random.choice(distractor_indices, size=n_medium + n_hard, replace=False)
            if len(distractor_indices) >= n_medium + n_hard
            else distractor_indices
        )
        medium_and_hard = torch.tensor(medium_and_hard, device=device, dtype=torch.long)

        # Random distractors
        all_indices = torch.arange(C, device=device)
        mask = ~torch.isin(all_indices, torch.cat([unique_labels, medium_and_hard], dim=0))
        remaining = all_indices[mask]

        rand_distractors = remaining[torch.randperm(len(remaining))[:n_rand]] if n_rand > 0 else torch.tensor([], device=device, dtype=torch.long)

        all_distractors = torch.cat([medium_and_hard, rand_distractors], dim=0)

        # Final contrast set
        contrast_indices = torch.cat([unique_labels, all_distractors], dim=0)
        contrast_text_embeddings = candidate_text_embeddings[contrast_indices]

        # Map labels
        mapping_array = torch.full((C,), -1, dtype=torch.long, device=device)
        mapping_array[contrast_indices] = torch.arange(contrast_indices.shape[0], device=device)
        label_samples_mapped = mapping_array[label_samples]

        if (label_samples_mapped == -1).any():
            raise ValueError("Some labels in the batch were not found in the contrastive index set.")

        logits = pred_samples @ contrast_text_embeddings.T
        logits /= temperature
        contrastive_loss = F.cross_entropy(logits, label_samples_mapped)
        
        # pred: [B, D, H, W] — normalized embeddings
        tv_h = F.l1_loss(pred[:, :, :, :-1], pred[:, :, :, 1:])
        tv_v = F.l1_loss(pred[:, :, :-1, :], pred[:, :, 1:, :])
        smoothness_loss = (tv_h + tv_v) * lambda_smooth
        
        total_loss = contrastive_loss + smoothness_loss


        loss_info = {
            'total_loss': total_loss.detach().item(),
            'contrastive_loss': contrastive_loss.detach().item(),
            'smoothness_loss': smoothness_loss.detach().item(),
            'temperature': temperature.item(),
        }

        return total_loss, loss_info




    
    @property
    def temperature(self):
        """
        Get the temperature value (exp of the log_temperature parameter).
        This ensures temperature is always positive.
        """
        return torch.exp(self.log_temperature)
    
    def train(self, mode=True):
        """
        Sets model to training mode.
        """
        self.depth_encoder.train(mode)
        self.depth_decoder.train(mode and not self.freeze_encoder)
        return self

    def eval(self):
        """Sets model to evaluation mode."""
        self.depth_encoder.eval()
        self.depth_decoder.eval()
        return self
    
    def save_model(self, checkpoint_path, step, optimizer=None):
        """
        Save the full model checkpoint

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer state to save
        """
        checkpoint = {
            'train_step': step
        }

        # Save encoder state dicts
        encoder = self.depth_encoder
        if hasattr(encoder, "module"):
            encoder = encoder.module
        checkpoint["encoder"] = encoder.state_dict()
        
        # Save decoder state dict
        decoder = self.depth_decoder
        if hasattr(decoder, "module"):
            decoder = decoder.module
        checkpoint["decoder"] = decoder.state_dict()

        # Save temperature parameter
        checkpoint["temperature"] = self.temperature.data

        # Optionally save optimizer state
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    
    def restore_model(self, restore_path, optimizer=None):
        """
        Restore model from checkpoint
        
        Arg(s):
            restore_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer to restore
                
        Returns:
            int : training step
            torch.optim : restored optimizer (if provided)
        """
        checkpoint = torch.load(restore_path, map_location=self.device)

        encoder = self.depth_encoder
        if hasattr(encoder, "module"):
            encoder = encoder.module
        
        encoder.load_state_dict(checkpoint['encoder'])
        decoder = self.depth_decoder
        if hasattr(decoder, "module"):
            decoder = decoder.module
            
        decoder.load_state_dict(checkpoint['decoder'])
        
        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

        try:
            train_step = checkpoint['train_step']
        except Exception:
            train_step = 0
            
        if "temperature" in checkpoint:
            self.temperature = nn.Parameter(checkpoint["temperature"])
            
        return train_step, optimizer if optimizer else None
    
    def restore_depth_encoder(self, encoder_path, freeze_encoder=False):
        """
        Restore model from checkpoint
        
        Arg(s):
            restore_path : str
                path to checkpoint
            optimizer : torch.optim
                optimizer to restore
                
        Returns:
            int : training step
            torch.optim : restored optimizer (if provided)
        """
        checkpoint = torch.load(encoder_path, map_location=self.device)

        if isinstance(self.depth_encoder, torch.nn.DataParallel):
            self.depth_encoder.module.load_state_dict(checkpoint['encoder'])
        else:
            self.depth_encoder.load_state_dict(checkpoint['encoder'])
            
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            # Freeze encoder parameters
            
            self.depth_encoder.eval()
            for param in self.depth_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.depth_encoder.parameters():
                param.requires_grad = True
    
    
