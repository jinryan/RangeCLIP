# datasets.py

import os
import random
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms

# ... (Keep helper functions like normalize_depth, default_image_transform) ...
# --- ADD A RESIZE TRANSFORM if not already part of your specific transforms ---
# --- This is crucial for this approach to work with default_collate ---
# Example: Define the target size
TARGET_HEIGHT = 224 # Or whatever your model expects
TARGET_WIDTH = 224  # Or whatever your model expects

# You would typically integrate Resize into your specific image/seg/depth transforms
# For example, modify default_image_transform or create specific ones.
# Example transform including resize:
# def create_image_transform(target_h, target_w):
#      return transforms.Compose([
#          transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BICUBIC),
#          transforms.Normalize(
#              mean=[0.48145466, 0.4578275, 0.40821073],
#              std=[0.26862954, 0.26130258, 0.27577711]
#          )
#      ])
# Make sure segmentation/depth transforms also resize to the *same* H, W


class ImageDepthTextDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, labels_path, image_transform=None, depth_transform=None, segmentation_transform=None, bbox_padding=10):
        # ... (Initialization logic) ...
        # IMPORTANT: Ensure the transforms WILL resize to a fixed size
        self.metadata = pd.read_csv(metadata_file)
        self.image_transform = image_transform # Should include Resize & Normalize
        self.depth_transform = depth_transform # Should include Resize
        self.segmentation_transform = segmentation_transform # Should include Resize (Nearest neighbor)
        self.root_dir = os.path.dirname(metadata_file)
        self.bbox_padding = bbox_padding
        # --- (Label Loading logic) ---
        df = pd.read_csv(labels_path, usecols=['label', 'index'], na_values=[], keep_default_na=False)
        df = df.sort_values(by='index', ascending=True)
        object_labels_list = df['label'].tolist()
        actual_indices = df['index'].tolist()
        expected_indices = list(range(1, len(object_labels_list) + 1))
        assert actual_indices == expected_indices, \
            f"Indices in {labels_path} must be 1-based, consecutive, and sorted. Found: {actual_indices}"
        self.dummy_label = "unavailable"
        self.labels = [self.dummy_label] + object_labels_list
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        # --- End Label Setup ---

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        depth_path = os.path.join(self.root_dir, row['depth_path'])
        segmentation_path = os.path.join(self.root_dir, row['label_path'])

        # --- Load PIL Images ---
        img_orig_pil = Image.open(img_path).convert("RGB")
        depth_pil = Image.open(depth_path).convert("I")
        segmentation_pil = Image.open(segmentation_path).convert("I")
        # --- End Loading ---

        # --- Process Image, Depth, Segmentation Tensors ---
        # Convert original PIL image to tensor FIRST
        img_orig_tensor = F.to_tensor(img_orig_pil)
        depth_tensor = F.to_tensor(depth_pil).float()
        segmentation_np = np.array(segmentation_pil) # Keep np version for now if transform needs it

        img_processed = self.image_transform(img_orig_tensor)
        depth_processed = self.depth_transform(depth_tensor)
        if self.segmentation_transform is not None:
            segmentation_processed = self.segmentation_transform(segmentation_np) # e.g., Resize(NN)
            if not isinstance(segmentation_processed, torch.Tensor):
                 segmentation_processed = torch.as_tensor(segmentation_processed, dtype=torch.long)
            elif segmentation_processed.dtype != torch.long:
                 segmentation_processed = segmentation_processed.long()
        else:
             # If no transform, convert to tensor BUT size will be variable! This breaks the approach.
             segmentation_processed = torch.from_numpy(segmentation_np).long()
             # Ensure segmentation_transform includes resizing if using this strategy!

        # --- Get Processed Shape ---
        if not isinstance(img_processed, torch.Tensor) or img_processed.dim() < 2:
             raise ValueError("img_processed is not a valid tensor after transform")
        if not isinstance(segmentation_processed, torch.Tensor) or segmentation_processed.dim() < 2:
             raise ValueError("segmentation_processed is not a valid tensor after transform")

        seg_shape = segmentation_processed.shape
        if len(seg_shape) == 3: # C, H, W
            H_proc, W_proc = seg_shape[1], seg_shape[2]
        elif len(seg_shape) == 2: # H, W
            H_proc, W_proc = seg_shape[0], seg_shape[1]
        else:
            raise ValueError(f"Unexpected segmentation_processed shape: {seg_shape}")


        # --- Find Random Object Bounding Box and Label on PROCESSED segmentation ---
        object_bbox = (0, 0, W_proc, H_proc)
        object_label = 0

        # Use PyTorch tensor operations for finding bbox
        unique_labels_in_image = torch.unique(segmentation_processed)

        # Determine excluded indices safely (using string labels)
        excluded_indices_vals = {0} # Exclude dummy index 0
        background_idx = self.label_to_index.get("background", -1)
        wall_idx = self.label_to_index.get("wall", -1)
        if background_idx != -1: excluded_indices_vals.add(background_idx)
        if wall_idx != -1: excluded_indices_vals.add(wall_idx)

        # Create mask for valid foreground labels present in the image
        # Convert excluded indices set to a tensor for isin
        excluded_tensor = torch.tensor(list(excluded_indices_vals), device=unique_labels_in_image.device, dtype=unique_labels_in_image.dtype)
        # Find labels that are NOT excluded AND are valid indices
        valid_label_mask = ~torch.isin(unique_labels_in_image, excluded_tensor)
        valid_label_mask &= (unique_labels_in_image > 0) & (unique_labels_in_image < len(self.labels))
        foreground_labels_present = unique_labels_in_image[valid_label_mask]

        random_label_idx = None
        if foreground_labels_present.numel() > 0:
            # random.choice works on tensors too
            random_label_idx = random.choice(foreground_labels_present).item() # Get scalar value
            # Find pixel coordinates (indices) for the chosen label using torch.nonzero
            # nonzero returns shape [N, Dims], where Dims is number of dimensions (2 for H,W)
            object_pixels = torch.nonzero(segmentation_processed == int(random_label_idx), as_tuple=False)

            if object_pixels.numel() > 0:
                # Find min/max coordinates (y coordinates are dim 0, x coordinates are dim 1)
                ymin, xmin = object_pixels.min(dim=0).values # .values extracts from torch.return_types.min
                ymax, xmax = object_pixels.max(dim=0).values # .values extracts from torch.return_types.max

                # Convert to scalar ints
                ymin, xmin = ymin.item(), xmin.item()
                ymax, xmax = ymax.item(), xmax.item()

                # Add Padding and Clip to PROCESSED Image Boundaries (H_proc, W_proc)
                ymin_pad = max(0, ymin - self.bbox_padding)
                xmin_pad = max(0, xmin - self.bbox_padding)
                # Add 1 to max index for slicing (exclusive end)
                ymax_pad = min(H_proc, ymax + 1 + self.bbox_padding)
                xmax_pad = min(W_proc, xmax + 1 + self.bbox_padding)

                if xmax_pad > xmin_pad and ymax_pad > ymin_pad:
                    # Bbox format (left, upper, right, lower) for processed dimensions
                    object_bbox = (int(xmin_pad), int(ymin_pad), int(xmax_pad), int(ymax_pad))
                    object_label = int(random_label_idx)
                else:
                    object_label = 0
                    object_bbox = (0, 0, W_proc, H_proc) # Reset if padding invalidates
            else:
                 object_label = 0
                 object_bbox = (0, 0, W_proc, H_proc) # Reset if no pixels found for label

        # --- End BBox Logic ---
        object_bbox_tensor = torch.tensor(object_bbox, dtype=torch.long)
        
        return {
            'depth': depth_processed,                # Processed (resized) depth tensor
            'image': img_processed,                  # Processed (resized, normalized) image tensor
            'segmentation': segmentation_processed,  # Processed (resized) segmentation tensor
            # 'image_orig': NO LONGER RETURNED
            'object_bbox': object_bbox_tensor,              # BBox tuple relative to PROCESSED dims
            'object_label': object_label             # Label index corresponding to bbox
        }
        
    def get_candidate_labels(self):
        return self.labels