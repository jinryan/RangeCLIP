from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
import datasets
from torch.utils.data import DataLoader

def setup_dataloaders(metadata_file,
                      labels_file,
                      resize_shape,
                      batch_size,
                      n_thread,
                      n_epoch):
    
    image_transform = transforms.Compose([
        transforms.Resize(resize_shape),
    ])
    

    def depth_transform_fn(depth_tensor):
        """
        Resize and normalize a depth map tensor to [0, 1].

        Args:
            depth_tensor (torch.Tensor): Input depth map. Shape can be [H, W] or [C, H, W].
            resize_shape (tuple): Desired (H, W) shape.
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
            mode='nearest'
        )

        min_val, max_val = resized.min(), resized.max()

        if (max_val - min_val).abs() < 1e-6:
            normalized = torch.zeros_like(resized)
        else:
            normalized = (resized - min_val) / (max_val - min_val)

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
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_thread
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_thread
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_thread
    )
    
    n_train_samples = len(train_dataset)
    n_train_steps = ((n_train_samples + batch_size - 1) // batch_size) * n_epoch
    
    return train_loader, val_loader, test_loader, n_train_steps, labels
    
    