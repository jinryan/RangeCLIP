# datasets.py

import os
import random
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms


# Temporary hard code
TARGET_HEIGHT = 224
TARGET_WIDTH = 224


class ImageDepthTextDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, labels_path, image_transform=None, depth_transform=None, segmentation_transform=None, bbox_padding=10):
        # ... (Initialization logic) ...
        self.metadata = pd.read_csv(metadata_file)
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.segmentation_transform = segmentation_transform
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

        # --- Process Image, Depth, Segmentation Tensors ---
        img_orig_tensor = F.to_tensor(img_orig_pil)
        depth_tensor = F.to_tensor(depth_pil).float()
        segmentation_np = np.array(segmentation_pil)

        img_processed = self.image_transform(img_orig_tensor)
        depth_processed = self.depth_transform(depth_tensor)
        if self.segmentation_transform is not None:
            segmentation_processed = self.segmentation_transform(segmentation_np) # e.g., Resize(NN)
            if not isinstance(segmentation_processed, torch.Tensor):
                 segmentation_processed = torch.as_tensor(segmentation_processed, dtype=torch.long)
            elif segmentation_processed.dtype != torch.long:
                 segmentation_processed = segmentation_processed.long()
        else:
             segmentation_processed = torch.from_numpy(segmentation_np).long()

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
            random_label_idx = random.choice(foreground_labels_present).item()
            object_pixels = torch.nonzero(segmentation_processed == int(random_label_idx), as_tuple=False)

            if object_pixels.numel() > 0:
                ymin, xmin = object_pixels.min(dim=0).values
                ymax, xmax = object_pixels.max(dim=0).values

                ymin, xmin = ymin.item(), xmin.item()
                ymax, xmax = ymax.item(), xmax.item()

                ymin_pad = max(0, ymin - self.bbox_padding)
                xmin_pad = max(0, xmin - self.bbox_padding)
                ymax_pad = min(H_proc, ymax + 1 + self.bbox_padding)
                xmax_pad = min(W_proc, xmax + 1 + self.bbox_padding)

                if xmax_pad > xmin_pad and ymax_pad > ymin_pad:
                    object_bbox = (int(xmin_pad), int(ymin_pad), int(xmax_pad), int(ymax_pad))
                    object_label = int(random_label_idx)
                else:
                    object_label = 0
                    object_bbox = (0, 0, W_proc, H_proc)
            else:
                 object_label = 0
                 object_bbox = (0, 0, W_proc, H_proc)

        # --- End BBox Logic ---
        object_bbox_tensor = torch.tensor(object_bbox, dtype=torch.long)
        
        return {
            'depth': depth_processed,
            'image': img_processed,
            'segmentation': segmentation_processed,
            'object_bbox': object_bbox_tensor,
            'object_label': object_label
        }
        
    def get_candidate_labels(self):
        return self.labels