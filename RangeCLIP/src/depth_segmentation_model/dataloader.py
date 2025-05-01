import random
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

def setup_dataloaders(metadata_file,
                      labels_file,
                      resize_shape,
                      batch_size,
                      n_thread,
                      n_epoch):
    
    image_transform = transforms.Compose([
        transforms.Resize(resize_shape)
    ])
    

    def depth_transform_fn(depth_tensor, interpolation_mode='nearest'):
        """
        Resize and normalize a depth map tensor using the median value for normalization.

        Args:
            depth_tensor (torch.Tensor): Input depth map. Shape can be [H, W] or [C, H, W].
            interpolation_mode (str): Interpolation mode for resizing (default: 'nearest').

        Returns:
            torch.Tensor: Normalized depth map resized to resize_shape, same dimensions as input.
        """
        original_dims = depth_tensor.dim()

        if original_dims == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif original_dims == 3:
            if depth_tensor.shape[0] != 1:
                raise ValueError(f"Expected single-channel depth, got shape {depth_tensor.shape}")
            depth_tensor = depth_tensor.unsqueeze(0)

        resized = F.interpolate(
            depth_tensor,
            size=resize_shape,
            mode=interpolation_mode
        )

        median_val = resized.median()

        if median_val.abs() < 1e-6:
            normalized = torch.zeros_like(resized)
        else:
            normalized = resized / median_val

        if original_dims == 2:
            return normalized.squeeze(0).squeeze(0)
        elif original_dims == 3:
            return normalized.squeeze(0)
        else:
            return normalized

    def resize_segmentation(segmentation):
        """
        Resize segmentation map using nearest neighbor interpolation.

        Args:
            segmentation (np.ndarray or torch.Tensor): Segmentation map as a 2D array of class indices.
            resize_shape (tuple): (H, W) target shape.

        Returns:
            torch.Tensor: Resized segmentation map (H, W) with integer labels.
        """
        if isinstance(segmentation, np.ndarray):
            segmentation = torch.from_numpy(segmentation).long()

        if segmentation.dim() == 2:
            segmentation = segmentation.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        resized = F.interpolate(segmentation.float(), size=resize_shape, mode='nearest')  # Use float for interpolate
        return resized.squeeze(0).squeeze(0).long()  # [H, W] with int labels

    depth_transform = depth_transform_fn
    segmentation_transform = resize_segmentation
        
    dataset = datasets.ImageDepthTextDataset(metadata_file=metadata_file,
                                                  labels_path=labels_file,
                                                  image_transform=image_transform,
                                                  depth_transform=depth_transform,
                                                  segmentation_transform=segmentation_transform)

    labels = dataset.get_candidate_labels()
    
    # Split dataset into train, validation, and test sets
    random.seed(42)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    split1 = int(0.6 * len(dataset))
    split2 = int(0.8 * len(dataset))

    train_indices = indices[:split1]
    val_indices = indices[split1:split2]
    test_indices = indices[split2:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # Create samplers for distributed training

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=n_thread
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=n_thread
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=n_thread
    )
    
    n_train_samples = len(train_dataset)
    n_train_steps = ((n_train_samples + batch_size - 1) // batch_size) * n_epoch
    
    return train_loader, val_loader, test_loader, train_sampler, n_train_steps, labels

    
    
    
import ast  # To safely parse list strings like "[1, 2, 3]"

def load_equivalence_dict(csv_path):
    df = pd.read_csv(csv_path)
    equivalence_dict = {}
    for _, row in df.iterrows():
        idx = int(row['index'])
        same = set(ast.literal_eval(row['same']))  # safely parse the string list
        same.add(idx)  # include self
        equivalence_dict[idx] = same
    return equivalence_dict

import torch

def build_equivalence_tensor(equivalence_dict, num_classes):
    # print(f"Num classes is {num_classes}")
    lookup = torch.zeros((num_classes, num_classes), dtype=torch.bool)
    for gt, equivalents in equivalence_dict.items():
        for pred in equivalents:
            lookup[gt, pred] = True
    return lookup

import pandas as pd
import ast

def load_label_similarity_sets(path, num_classes):
    df = pd.read_csv(path)
    medium_sets = [[] for _ in range(num_classes)]
    hard_sets = [[] for _ in range(num_classes)]

    for _, row in df.iterrows():
        idx = int(row['index'])
        medium_sets[idx] = ast.literal_eval(row['medium'])
        hard_sets[idx] = ast.literal_eval(row['hard'])

    return {
        'medium': medium_sets,
        'hard': hard_sets,
    }

from collections import defaultdict
import torch
from torch.cuda.amp import autocast
import tqdm


def build_equivalence_class_map(equivalence_tensor, device):
    """
    Given an equivalence_tensor, map each label index to a representative
    label ID of its equivalence class (smallest index in the class).
    """
    num_labels = equivalence_tensor.shape[0]
    equiv_class_map = torch.arange(num_labels).to(device)
    for i in range(num_labels):
        equiv = (equivalence_tensor[i] == 1).nonzero(as_tuple=True)[0]
        if len(equiv) > 0:
            equiv_class_map[i] = torch.min(equiv)
    return equiv_class_map


@torch.no_grad()
def prepare_image_contrast_data(
    image_processed_batch,
    object_bbox_batch,
    object_label_batch,
    segmentation_batch,
    pixel_embeddings_batch,
    clip_image_encoder,
    clip_processor,
    device
):
    """
    Prepares area embeddings and CLIP image embeddings for image contrastive loss,
    cropping the PROCESSED image based on bbox derived from PROCESSED segmentation.
    """
    if not isinstance(image_processed_batch, torch.Tensor) or image_processed_batch.dim() != 4:
         print(f"Warning: 'image_processed_batch' is not a 4D tensor.")
         return None, None
    if not isinstance(object_bbox_batch, torch.Tensor) or object_bbox_batch.dim() != 2 or object_bbox_batch.shape[1] != 4:
         print(f"Warning: 'object_bbox_batch' is not a [B, 4] tensor.")
         return None, None
    if not isinstance(object_label_batch, torch.Tensor):
         print(f"Warning: 'object_label_batch' is not a tensor.")
         return None, None

    B, C, H_proc, W_proc = image_processed_batch.shape
    if B == 0: return None, None
    D = pixel_embeddings_batch.shape[1]

    crops_for_clip = []
    valid_indices = []
    labels_for_pooling = []

    for b_idx in range(B):
        img_tensor = image_processed_batch[b_idx]
        bbox_tensor = object_bbox_batch[b_idx]
        label_idx = object_label_batch[b_idx].item()

        # --- Basic Validation ---
        # if label_idx <= 0:
            # print(f"Debug: Skipping item {b_idx}, label is {label_idx} (background/invalid).")
            # continue

        try:
            xmin, ymin, xmax, ymax = bbox_tensor[0].item(), bbox_tensor[1].item(), bbox_tensor[2].item(), bbox_tensor[3].item()
        except IndexError:
            print(f"Warning: Skipping item {b_idx}, failed to unpack bbox tensor: {bbox_tensor}")
            continue

        # --- Crop PROCESSED Image Tensor ---
        # Coordinates are already relative to processed dimensions (H_proc, W_proc)
        if xmax > xmin and ymax > ymin and xmin >= 0 and ymin >= 0 and xmax <= W_proc and ymax <= H_proc:
            cropped_processed_tensor = img_tensor[:, ymin:ymax, xmin:xmax]

            if cropped_processed_tensor.numel() > 0:
                # Detach?
                crops_for_clip.append(cropped_processed_tensor)
                valid_indices.append(b_idx)
                labels_for_pooling.append(label_idx)
            else:
                print(f"Warning: Skipping item {b_idx}, cropped processed tensor is empty for bbox [{xmin},{ymin},{xmax},{ymax}].")
        else:
            print(f"Debug: Skipping item {b_idx}, invalid bbox [{xmin},{ymin},{xmax},{ymax}] relative to processed dims ({H_proc}, {W_proc}) for label {label_idx}.")

    # --- Process Valid Crops ---
    if not crops_for_clip:
        return None, None

   
    try:
        image_inputs = clip_processor(images=crops_for_clip, return_tensors="pt", padding=True, do_rescale=False).to(device)
    except Exception as e:
        print(f"Error during clip_processor processing cropped tensors: {e}")
        return None, None

    # Get image embeddings from CLIP
    try:
        image_embeddings = clip_image_encoder.get_image_features(pixel_values=image_inputs['pixel_values'])
    except Exception as e:
        raise TypeError(f"CLIP image encoder failed: {e}. Ensure it's a compatible model.") from e

    # --- Calculate Area Embeddings ---
    final_area_embeddings = torch.zeros_like(image_embeddings)
    for i, original_b_idx in enumerate(valid_indices):
        label_idx = labels_for_pooling[i]
        # Use the corresponding item from the original batch index
        single_pixel_embeddings = pixel_embeddings_batch[original_b_idx]
        single_segmentation_map = segmentation_batch[original_b_idx]

        # Handle potential channel dim in segmentation_batch
        if single_segmentation_map.dim() == 3:
            single_segmentation_map = single_segmentation_map.squeeze(0)

        mask = (single_segmentation_map == label_idx).unsqueeze(0)
        pixel_count = mask.sum()
        if pixel_count > 0:
            mask_expanded = mask.expand_as(single_pixel_embeddings)
            masked_pixels = torch.where(mask_expanded, single_pixel_embeddings, torch.zeros_like(single_pixel_embeddings))
            summed_embeddings = masked_pixels.sum(dim=(1, 2))
            final_area_embeddings[i] = summed_embeddings / pixel_count

    return final_area_embeddings, image_embeddings