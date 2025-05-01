# model.py (Changes indicated by ###)
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

# ### Helper function for masked average pooling (can be placed inside the class or outside) ###
def masked_average_pooling(pixel_embeddings, segmentation_map, object_indices):
    """
    Pools pixel embeddings based on segmentation masks for specified object indices.

    Args:
        pixel_embeddings (torch.Tensor): Pixel embeddings (B, D, H, W).
        segmentation_map (torch.Tensor): Segmentation map (B, H, W).
        object_indices (torch.Tensor): Long tensor of unique object indices [N_obj].
                                       Assumes these indices correspond to values in segmentation_map.

    Returns:
        torch.Tensor: Area embeddings for each object index (N_obj, D).
                      Returns zeros if an object index is not found in the map.
    """
    B, D, H, W = pixel_embeddings.shape
    device = pixel_embeddings.device
    area_embeddings = torch.zeros((len(object_indices), D), device=device, dtype=pixel_embeddings.dtype)

    # Ensure segmentation_map is on the same device and suitable for broadcasting
    segmentation_map = segmentation_map.to(device).unsqueeze(1) # -> (B, 1, H, W)

    for i, obj_idx in enumerate(object_indices):
        # Create a mask for the current object index across the batch
        mask = (segmentation_map == obj_idx) # -> (B, 1, H, W) boolean mask

        # Check if the object exists in the batch
        if mask.any():
            # Expand mask to match embedding dimensions: (B, D, H, W)
            mask_expanded = mask.expand_as(pixel_embeddings)

            # Select relevant embeddings and sum them
            masked_pixels = torch.where(mask_expanded, pixel_embeddings, torch.zeros_like(pixel_embeddings))
            summed_embeddings = masked_pixels.sum(dim=(0, 2, 3)) # Sum over Batch, H, W -> (D,)

            # Count the number of pixels for normalization (across the batch)
            pixel_count = mask.sum()

            # Calculate average if pixels exist
            if pixel_count > 0:
                area_embeddings[i] = summed_embeddings / pixel_count
            # else: area_embeddings[i] remains zeros

    return area_embeddings


class DepthUNet(nn.Module):
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
                 temperature_text=0.07,
                 temperature_image=0.1): # ### Added image temperature ###
        super(DepthUNet, self).__init__()

        self.device = device

        # ### Separate temperatures for text and image contrastive losses ###
        self.log_temperature_text = nn.Parameter(torch.log(torch.tensor(temperature_text)))
        self.log_temperature_image = nn.Parameter(torch.log(torch.tensor(temperature_image)))

        # --- (Rest of __init__ remains the same) ---
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
        """ Forward pass of the UNet model """
        with autocast():
            target_shape = depth_map.shape[2:]
            _, encoder_features, final_feature_map = self.depth_encoder(depth_map)
            # ### Renamed output for clarity ###
            pixel_embeddings = self.depth_decoder(final_feature_map, encoder_features, target_shape)
            current_temp_text = torch.exp(self.log_temperature_text)
            current_temp_image = torch.exp(self.log_temperature_image)
        
        # ### Return pixel embeddings and BOTH temperatures ###
        # ### Note: Returning temperature directly is less common than accessing via property ###
        # ### We will use properties self.temperature_text and self.temperature_image later ###
        return pixel_embeddings, current_temp_text, current_temp_image

    def predict(self, depth_maps, candidate_text_embeddings, segmentation, num_negatives=300, top_k=5):
        """
        Predict segmentation labels for the input depth maps using a reduced candidate set.

        Args:
            depth_maps (torch.Tensor): input depth maps [B, 1, H, W]
            candidate_text_embeddings (torch.Tensor): all text embeddings [C, D]
            segmentation (torch.Tensor): ground truth segmentation [B, H, W], used to extract true label indices
            num_negatives (int): number of additional negative labels to sample
            top_k (int): number of top labels to return per pixel

        Returns:
            topk_pred_indices (torch.LongTensor): top-k predicted label indices in the ORIGINAL index space [B, k, H, W]
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
            logits = torch.einsum('bdn,cd->bcn', pixel_flat, reduced_candidate_embeddings)  # [B, C_reduced, H*W]

            # Top-k prediction
            topk = min(top_k, logits.shape[1])
            topk_indices_reduced = logits.topk(topk, dim=1).indices  # [B, k, H*W]
            topk_indices_reduced = topk_indices_reduced.view(B, topk, H, W)  # [B, k, H, W]

            # Map reduced indices back to original candidate label indices
            topk_pred_indices = index_tensor[topk_indices_reduced]  # [B, k, H, W]

            return topk_pred_indices, pixel_embeddings, self.temperature_text


    def compute_loss(
        self,
        pixel_embeddings, # ### Renamed from 'pred' ###
        target_indices, # Ground truth segmentation map
        candidate_text_embeddings,
        label_similarity_sets,
        area_embeddings, # Tensor of area embeddings (N_instances, D)
        image_embeddings, # Tensor of corresponding CLIP image embeddings (N_instances, D)
        # ### Loss weights ###
        W_text = 1.0,
        W_image = 0.5, # Example weight
        W_smooth = 2e2, # Example weight (formerly lambda_smooth)
        # --- Parameters for text contrastive loss ---
        percent_image_sampling=0.7, # Percentage of pixels to sample for text loss
        k_distractors=50,
        pct_medium=0.0,
        pct_hard=0.75,
        pct_rand=0.25,
    ):
        """
        Hybrid contrastive loss including pixel-text and area-image alignment.
        """
        # --- 1. Pixel-Text Contrastive Loss (Existing Logic, adapted) ---
        text_contrastive_loss = torch.tensor(0.0, device=pixel_embeddings.device)
        if W_text > 0:
            assert abs(pct_medium + pct_hard + pct_rand - 1.0) < 1e-4, "Sum of text percentages must be 1."

            B, D, H, W = pixel_embeddings.shape
            C = candidate_text_embeddings.shape[0]
            device = pixel_embeddings.device
            num_samples = int(percent_image_sampling * H * W) # Renamed var

            # Flatten and sample pixels
            pred_flat = pixel_embeddings.reshape(B, D, -1)
            target_flat = target_indices.reshape(B, -1)
            # Ensure sampling within valid range if H*W is small
            num_pixels_total = H * W
            if num_pixels_total == 0:
                 raise RuntimeError("Input dimensions H or W are zero.")
            actual_num_samples = min(num_samples, num_pixels_total)
            if actual_num_samples == 0 and num_pixels_total > 0:
                 actual_num_samples = num_pixels_total # Sample all if percent is too low

            rand_indices = torch.randint(0, num_pixels_total, (B, actual_num_samples), device=device)

            pred_samples = torch.gather(pred_flat, 2, rand_indices.unsqueeze(1).expand(-1, D, -1))
            label_samples = torch.gather(target_flat, 1, rand_indices)

            # Filter out background pixels (assuming index 0 is background) before reshaping
            valid_mask = label_samples > 0
            pred_samples = pred_samples.permute(0, 2, 1)[valid_mask].reshape(-1, D) # (N_valid_samples, D)
            label_samples = label_samples[valid_mask].reshape(-1) # (N_valid_samples,)


            if pred_samples.numel() > 0 and label_samples.numel() > 0:
                unique_labels_in_samples = torch.unique(label_samples)

                # --- Build contrast set for text loss (same logic as before) ---
                distractor_indices = set()
                n_medium = int(k_distractors * pct_medium)
                n_hard = int(k_distractors * pct_hard)
                n_rand = k_distractors - n_medium - n_hard

                if n_medium > 0:
                    for label in unique_labels_in_samples.tolist():
                         if label in label_similarity_sets['medium']: # Check label exists
                            distractor_indices.update(label_similarity_sets['medium'][label])
                if n_hard > 0:
                     for label in unique_labels_in_samples.tolist():
                          if label in label_similarity_sets['hard']: # Check label exists
                             distractor_indices.update(label_similarity_sets['hard'][label])

                distractor_indices = list(distractor_indices)
                # Ensure distractors are not ground truth labels present in *this sample*
                distractor_indices = [d for d in distractor_indices if d not in unique_labels_in_samples.tolist()]

                medium_and_hard_count = n_medium + n_hard
                medium_and_hard = (
                    np.random.choice(distractor_indices, size=medium_and_hard_count, replace=False)
                    if len(distractor_indices) >= medium_and_hard_count
                    else distractor_indices
                )
                medium_and_hard = torch.tensor(medium_and_hard, device=device, dtype=torch.long)

                all_indices = torch.arange(C, device=device)
                # Exclude GT labels and already chosen medium/hard distractors
                mask = ~torch.isin(all_indices, torch.cat([unique_labels_in_samples, medium_and_hard], dim=0))
                remaining = all_indices[mask]

                rand_distractors = remaining[torch.randperm(len(remaining))[:n_rand]] if n_rand > 0 and len(remaining) > 0 else torch.tensor([], device=device, dtype=torch.long)
                all_distractors = torch.cat([medium_and_hard, rand_distractors], dim=0)
                contrast_indices = torch.unique(torch.cat([unique_labels_in_samples, all_distractors], dim=0)) # Use unique to handle overlaps
                # --- End build contrast set ---

                if len(contrast_indices) > 1: # Need at least one positive and one negative
                     contrast_text_embeddings = F.normalize(candidate_text_embeddings[contrast_indices], dim=1) # Normalize embeddings
                     pred_samples_norm = F.normalize(pred_samples, dim=1) # Normalize pixel embeddings

                     # Map labels to the reduced contrast set index
                     mapping_array = torch.full((C,), -1, dtype=torch.long, device=device)
                     mapping_array[contrast_indices] = torch.arange(contrast_indices.shape[0], device=device)
                     label_samples_mapped = mapping_array[label_samples]

                     if (label_samples_mapped == -1).any():
                          print("Warning: Some sampled labels not found in text contrast set. Skipping text loss calculation for them.")
                          valid_map_mask = label_samples_mapped != -1
                          label_samples_mapped = label_samples_mapped[valid_map_mask]
                          pred_samples_norm = pred_samples_norm[valid_map_mask]


                     if len(label_samples_mapped) > 0:
                            # Calculate logits using normalized embeddings
                            logits_text = pred_samples_norm @ contrast_text_embeddings.T
                            logits_text /= self.temperature_text # Use text temperature
                            text_contrastive_loss = F.cross_entropy(logits_text, label_samples_mapped)
                     else:
                            print("Warning: No valid mapped labels remained for text contrastive loss.")
                            text_contrastive_loss = torch.tensor(0.0, device=device)
                else:
                     print("Warning: Not enough indices for text contrastive loss (need > 1).")
                     text_contrastive_loss = torch.tensor(0.0, device=device)

            else: # No valid non-background pixels sampled
                 print("Warning: No valid foreground pixels sampled for text contrastive loss.")
                 text_contrastive_loss = torch.tensor(0.0, device=device)


        # --- 2. Area-Image Contrastive Loss (New Logic) ---
        image_contrastive_loss = torch.tensor(0.0, device=pixel_embeddings.device)
        # Check if we have enough instances for contrast (at least 2)
        if area_embeddings is not None and image_embeddings is not None and area_embeddings.shape[0] > 1:
            n_instances = area_embeddings.shape[0]
            # Normalize embeddings
            area_embeddings_norm = F.normalize(area_embeddings, dim=1)
            image_embeddings_norm = F.normalize(image_embeddings, dim=1)

            # Calculate pairwise similarities (logits)
            # Logits matrix: Rows = Area Embeddings, Columns = Image Embeddings
            logits_image = area_embeddings_norm @ image_embeddings_norm.T # Shape: (N_instances, N_instances)
            logits_image /= self.temperature_image # Use image temperature

            # Labels: Each area embedding should match the image embedding at the same index
            image_contrast_labels = torch.arange(n_instances, device=device)

            # Calculate cross-entropy loss (InfoNCE)
            image_contrastive_loss = F.cross_entropy(logits_image, image_contrast_labels)
        elif W_image > 0:
            # Not enough instances in the batch/data provided for contrast
            print("Warning: Not enough instances (<=1) or missing embeddings for image contrastive loss.")
            dummy = torch.tensor(1.0, device=pixel_embeddings.device, requires_grad=True)
            image_contrastive_loss = dummy * self.temperature_image * 0.0


        # --- 3. Smoothness Loss (Existing Logic, applied to UNet output) ---
        smoothness_loss = torch.tensor(0.0, device=pixel_embeddings.device)
        if W_smooth > 0:
             # Use pixel_embeddings before normalization for smoothness
             # Ensure pixel_embeddings require grad if they don't already
             # if not pixel_embeddings.requires_grad: pixel_embeddings.requires_grad_(True)

             tv_h = F.l1_loss(pixel_embeddings[:, :, :, :-1], pixel_embeddings[:, :, :, 1:])
             tv_v = F.l1_loss(pixel_embeddings[:, :, :-1, :], pixel_embeddings[:, :, 1:, :])
             # No need to multiply by weight here, done in total loss
             smoothness_loss = tv_h + tv_v


        # --- 4. Total Loss ---
        total_loss = (W_text * text_contrastive_loss +
                      W_image * image_contrastive_loss +
                      W_smooth * smoothness_loss)

        # --- 5. Loss Info Dict ---
        loss_info = {
            'total_loss': total_loss.detach().item(),
            'text_contrastive_loss': text_contrastive_loss.detach().item() if W_text > 0 else 0,
            'image_contrastive_loss': image_contrastive_loss.detach().item() if W_image > 0 else 0,
            'smoothness_loss': smoothness_loss.detach().item() if W_smooth > 0 else 0,
            'temperature_text': self.temperature_text.item(),
            'temperature_image': self.temperature_image.item(),
            'W_text': W_text,
            'W_image': W_image,
            'W_smooth': W_smooth
        }

        return total_loss, loss_info


    @property
    def temperature_text(self):
        """ Get the temperature for text contrastive loss. """
        return torch.exp(self.log_temperature_text)

    @property
    def temperature_image(self):
        """ Get the temperature for image contrastive loss. """
        return torch.exp(self.log_temperature_image)

    # ### Deprecated single temperature property ###
    # @property
    # def temperature(self):
    #     return torch.exp(self.log_temperature)

    # --- (train, eval, save_model, restore_model, restore_depth_encoder remain mostly the same) ---
    # Ensure save/restore handles the two temperature parameters correctly
    def save_model(self, checkpoint_path, step, optimizer=None):
         checkpoint = { 'train_step': step }
         encoder = self.depth_encoder.module if hasattr(self.depth_encoder, "module") else self.depth_encoder
         decoder = self.depth_decoder.module if hasattr(self.depth_decoder, "module") else self.depth_decoder
         checkpoint["encoder"] = encoder.state_dict()
         checkpoint["decoder"] = decoder.state_dict()
         # ### Save both temperatures ###
         checkpoint["log_temperature_text"] = self.log_temperature_text.data
         checkpoint["log_temperature_image"] = self.log_temperature_image.data
         if optimizer is not None:
             checkpoint["optimizer"] = optimizer.state_dict()
         torch.save(checkpoint, checkpoint_path)

    def restore_model(self, restore_path, optimizer=None):
        checkpoint = torch.load(restore_path, map_location=self.device)
        encoder = self.depth_encoder.module if hasattr(self.depth_encoder, "module") else self.depth_encoder
        decoder = self.depth_decoder.module if hasattr(self.depth_decoder, "module") else self.depth_decoder
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        train_step = checkpoint.get('train_step', 0)
        # ### Restore both temperatures, provide default if missing ###
        default_log_temp_text = torch.log(torch.tensor(0.07))
        default_log_temp_image = torch.log(torch.tensor(0.1))
        self.log_temperature_text = nn.Parameter(checkpoint.get("log_temperature_text", default_log_temp_text))
        self.log_temperature_image = nn.Parameter(checkpoint.get("log_temperature_image", default_log_temp_image))
        return train_step, optimizer if optimizer else None

    # ... (rest of the methods: train, eval, restore_depth_encoder) ...
    def train(self, mode=True):
        """ Sets model to training mode. """
        # Handle frozen encoder correctly
        self.depth_encoder.train(mode and not getattr(self, 'freeze_encoder', False))
        self.depth_decoder.train(mode)
        return self

    def eval(self):
        """Sets model to evaluation mode."""
        self.depth_encoder.eval()
        self.depth_decoder.eval()
        return self