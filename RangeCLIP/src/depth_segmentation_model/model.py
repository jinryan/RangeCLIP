import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils.src.encoder import DepthEncoder
from utils.src.decoder import DepthDecoder
from transformers import CLIPModel
from torch.cuda.amp import autocast
from utils.src.networks import TextEncoder


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
                 clip_model_name,
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
        
        try:
            clip_model = CLIPModel.from_pretrained(clip_model_name)
            embedding_dim = clip_model.config.projection_dim
            print(f"Using CLIP model {clip_model_name} with embedding dimension {embedding_dim}")
            self.text_encoder = TextEncoder(clip_model, device)
        except Exception as e:
            raise ValueError(f'Failed to initialize CLIP model {clip_model_name}: {str(e)}')
        
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
        if unet_type == 'resnet':
            decoder_filters = encoder_filters[-1:0:-1]
            
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
        
        self.to(device)
    
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
        
        return output
    
    def predict(self, depth_maps, candidate_labels, top_k=5, return_text=False):
        """
        Predict segmentation labels for the input depth maps
        
        Arg(s):
            depth_maps : torch.Tensor
                input depth map tensor of shape [B, 1, H, W]
            candidate_labels : list of str
                list of candidate class labels
            top_k : int
                number of top predictions to return per pixel
            return_text : bool
                whether to return predicted class names (optional)
        
        Returns:
            torch.Tensor : predicted indices [B, H, W, K] or [B, H, W]
            torch.Tensor : predicted scores [B, H, W, K] or [B, H, W]
            Optional[List[List[str]]] : predicted class names
        """
        with torch.no_grad():
            depth_embeddings = self.forward(depth_maps)  # [B, D, H, W]
            B, D, H, W = depth_embeddings.shape

            candidate_text_embeddings = self.text_encoder(candidate_labels)  # [C, D]

            # Normalize for cosine similarity
            depth_embeddings = F.normalize(depth_embeddings, p=2, dim=1)
            candidate_text_embeddings = F.normalize(candidate_text_embeddings, p=2, dim=1)

            # Reshape to [B*H*W, D]
            pixel_embeddings = depth_embeddings.permute(0, 2, 3, 1).reshape(-1, D)

            # Cosine similarity + softmax over class dimension
            similarity = torch.matmul(pixel_embeddings, candidate_text_embeddings.T)  # [N, C]
            full_scores = F.softmax(similarity, dim=1)  # [N, C]

            # Top-k prediction per pixel
            topk_scores, topk_indices = full_scores.topk(k=top_k, dim=1)  # [N, K]

            if top_k == 1:
                predicted = topk_indices.view(B, H, W)
                predicted_scores = topk_scores.view(B, H, W)
            else:
                predicted = topk_indices.view(B, H, W, top_k)
                predicted_scores = topk_scores.view(B, H, W, top_k)

            if return_text:
                # Always treat top_k as dimension - even if 1
                topk_indices_flat = topk_indices.view(-1, top_k)
                predicted_classes = [
                    [candidate_labels[i.item()] for i in row]
                    for row in topk_indices_flat
                ]
                return predicted, predicted_scores, predicted_classes
            else:
                return predicted, predicted_scores

    def compute_loss(self, pred, target_indices, candidate_text_embeddings, percent_image=0.5, k_distractors=100):
        """
        Hybrid contrastive loss: in-batch class labels + random distractors.
        """
        B, D, H, W = pred.shape
        num_samples = int(percent_image * H * W)
        device = pred.device
        C = candidate_text_embeddings.shape[0]

        # Normalize predicted features
        pred = F.normalize(pred, p=2, dim=1)

        pred_samples, label_samples = [], []

        for b in range(B):
            flat_indices = torch.arange(H * W, device=self.device)

            sampled_idx = flat_indices[torch.randperm(len(flat_indices))[:min(num_samples, len(flat_indices))]]
            h_idx = sampled_idx // W
            w_idx = sampled_idx % W

            sampled_pred = pred[b, :, h_idx, w_idx].T  # [K, D]
            sampled_label = target_indices[b, h_idx, w_idx]  # [K]

            pred_samples.append(sampled_pred)
            label_samples.append(sampled_label)


        if len(pred_samples) == 0:
            raise RuntimeError("No valid pixels found for loss computation.")

        pred_samples = torch.cat(pred_samples, dim=0)        # [N, D]
        label_samples = torch.cat(label_samples, dim=0)      # [N]

        # Step 1: get unique labels in this batch
        unique_labels = torch.unique(label_samples)

        # Step 2: sample distractors
        all_indices = torch.arange(C, device=device)
        remaining = all_indices[~torch.isin(all_indices, unique_labels)]
        distractor_labels = remaining[torch.randperm(len(remaining))[:min(k_distractors, len(remaining))]]

        # Step 3: combine and extract contrastive set
        contrast_indices = torch.cat([unique_labels, distractor_labels], dim=0)
        contrast_text_embeddings = F.normalize(candidate_text_embeddings[contrast_indices], p=2, dim=1)  # [C', D]

        # Step 4: remap labels
        label_map = {idx.item(): new_idx for new_idx, idx in enumerate(contrast_indices)}
        label_samples_mapped = torch.tensor([label_map[l.item()] for l in label_samples], device=device)

        # Step 5: logits + loss
        logits = pred_samples @ contrast_text_embeddings.T  # [N, C']
        logits /= self.temperature
        loss = F.cross_entropy(logits, label_samples_mapped)

        loss_info = {
            'loss': loss.item(),
            'temperature': self.temperature.item(),
            'unique_labels': len(unique_labels),
            'distractors': len(distractor_labels),
            'total_classes': logits.shape[1]
        }

        return loss, loss_info


    
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

        if isinstance(self.depth_encoder, torch.nn.DataParallel):
            checkpoint['encoder'] = self.depth_encoder.module.state_dict()
        else:
            checkpoint['encoder'] = self.depth_encoder.state_dict()

        if isinstance(self.depth_decoder, torch.nn.DataParallel):
            checkpoint['decoder'] = self.depth_decoder.module.state_dict()
        else:
            checkpoint['decoder'] = self.depth_decoder.state_dict()

        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()

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

        if isinstance(self.depth_encoder, torch.nn.DataParallel):
            self.depth_encoder.module.load_state_dict(checkpoint['encoder'])
        else:
            self.depth_encoder.load_state_dict(checkpoint['encoder'])

        if isinstance(self.depth_decoder, torch.nn.DataParallel):
            self.depth_decoder.module.load_state_dict(checkpoint['decoder'])
        else:
            self.depth_decoder.load_state_dict(checkpoint['decoder'])

        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

        try:
            train_step = checkpoint['train_step']
        except Exception:
            train_step = 0

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
                
    def build_target_embedding_map(target_indices, candidate_text_embeddings):
        """
        Vectorized version: no for-loop.
        """
        B, H, W = target_indices.shape
        C, D = candidate_text_embeddings.shape

        flat_indices = target_indices.view(B, H * W)  # [B, HW]

        # [C, D] -> [1, C, D]
        text_emb = candidate_text_embeddings.unsqueeze(0)  # [1, C, D]

        # Index embeddings for each pixel
        gathered = text_emb.expand(B, C, D).gather(1, flat_indices.unsqueeze(-1).expand(-1, -1, D))  # [B, HW, D]
        
        target_embedding_map = gathered.permute(0, 2, 1).reshape(B, D, H, W)
        return target_embedding_map


                
                
import time

if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Time: Model Initialization ---
    t0 = time.time()
    model = DepthUNet(
        unet_type='resnet',
        clip_model_name='openai/clip-vit-base-patch32',  # Or another available CLIP model
        device=device,
        n_layer=18,
        embedding_dim=768,
        input_channels=1,
        encoder_filters=[32, 64, 128, 256, 512],
        weight_initializer='kaiming_uniform',
        activation_func='relu',
        use_batch_norm=True,
        use_instance_norm=False,
    )
    t1 = time.time()
    print(f"Model initialization took {t1 - t0:.3f} seconds")

    # Create a dummy batch of depth maps [B, C, H, W]
    B, C, H, W = 32, 1, 200, 200
    dummy_depth = torch.randn(B, C, H, W).to(device)

    import random
    import string

    # Generate 6000 random dummy labels
    num_total_labels = 6000
    dummy_labels = [f"label_{i}" for i in range(num_total_labels)]

    # Randomly select 40 labels to be used in this batch's target
    selected_indices = random.sample(range(num_total_labels), 40)
    selected_labels = [dummy_labels[i] for i in selected_indices]

    # Map selected class indices to 0–39 (for dummy target)
    index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(selected_indices)}

    # Create dummy target segmentation map with values in [0, 39]
    dummy_target = torch.randint(0, 40, size=(B, H, W)).to(device)

    # Remap target indices to match the indices in dummy_labels
    # (Later you'll reverse this during loss by mapping back to full label space)


    model.eval()
    with torch.no_grad():
        # --- Time: Forward Pass ---
        t2 = time.time()
        embedding_output = model(dummy_depth)  # [B, D, H, W]    
        t3 = time.time()
        print(f"Forward pass took {t3 - t2:.3f} seconds")
        print("Forward pass output shape:", embedding_output.shape)

        # --- Time: Text Encoder ---
        t4 = time.time()
        dummy_text_embeddings = model.text_encoder(dummy_labels)
        t5 = time.time()
        print(f"Text encoding on candidate labels took {t5 - t4:.3f} seconds")

        # --- Time: Loss Computation ---
        t6 = time.time()
        loss, info = model.compute_loss(embedding_output, dummy_target, dummy_text_embeddings)
        t7 = time.time()
        print(f"Loss computation took {t7 - t6:.3f} seconds")

        print("Dummy loss:", loss.item())
        print("Loss info:", info)

        # --- Optional: Prediction ---
        predicted_indices, predicted_scores, predicted_classes = model.predict(
            dummy_depth, dummy_labels, top_k=1, return_text=True
        )
        print("Predicted indices shape:", predicted_indices.shape)
        print("Predicted class names (first sample):", predicted_classes[:H * W][:5])
