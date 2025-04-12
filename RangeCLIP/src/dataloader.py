from torchvision import transforms
import torch
import datasets
from torch.utils.data import DataLoader

def setup_dataloaders(labeled_metadata_file,
                      unlabeled_metadata_file,
                      labels_path,
                      resize_shape,
                      batch_size,
                      n_thread,
                      n_epoch):
    
    image_transform = transforms.Compose([
        transforms.Resize(resize_shape),
    ])
    
    def depth_transform_fn(depth_tensor):
        # Resize depth map
        resized = transforms.functional.resize(depth_tensor, resize_shape)
        
        min_val = resized.min()
        max_val = resized.max()
        
        # If depth is uniform, return zero tensor or normalized to 0.5 (optional)
        if (max_val - min_val).abs() < 1e-6:
            return torch.zeros_like(resized)  # or torch.full_like(resized, 0.5)
        
        normalized = (resized - min_val) / (max_val - min_val)
        return normalized

    depth_transform = depth_transform_fn
        
    labeled_dataset = datasets.ImageDepthTextDataset(metadata_file=labeled_metadata_file,
                                                  labels_path=labels_path,
                                                  image_transform=image_transform,
                                                  depth_transform=depth_transform)
    
    unlabeled_dataset = datasets.ImageDepthDataset(metadata_file=unlabeled_metadata_file,
                                                   image_transform=image_transform,
                                                   depth_transform=depth_transform)
    labels = labeled_dataset.get_labels()
    
    labeled_dataset = datasets.TaggedDataset(labeled_dataset, tag='labeled')
    unlabeled_dataset = datasets.TaggedDataset(unlabeled_dataset, tag='unlabeled')
    
    generator = torch.Generator().manual_seed(42)
    l_train_dataset, l_val_dataset = torch.utils.data.random_split(labeled_dataset, [0.8, 0.2], generator=generator)
    u_train_dataset, u_val_dataset, _ = torch.utils.data.random_split(unlabeled_dataset, [0.02, 0.005, 0.975], generator=generator)
    
    
    def custom_collate(batch):
        labeled = {'image': [], 'depth': [], 'id': []}
        unlabeled = {'image': [], 'depth': []}

        for sample in batch:
            tag = sample.pop('__tag__')

            if tag == 'labeled':
                labeled['image'].append(sample['image'])
                labeled['depth'].append(sample['depth'])
                labeled['id'].append(sample['id'])

            elif tag == 'unlabeled':
                unlabeled['image'].append(sample['image'])
                unlabeled['depth'].append(sample['depth'])

        # Convert lists to tensors where applicable
        if labeled['image']:
            labeled['image'] = torch.stack(labeled['image'])
            labeled['depth'] = torch.stack(labeled['depth'])
            labeled['id'] = torch.stack(labeled['id'])

        if unlabeled['image']:
            unlabeled['image'] = torch.stack(unlabeled['image'])
            unlabeled['depth'] = torch.stack(unlabeled['depth'])  # shape: [B, N, ...]
        
        return {'labeled': labeled, 'unlabeled': unlabeled}

    train_dataset = torch.utils.data.ConcatDataset([l_train_dataset, u_train_dataset])
    val_dataset = torch.utils.data.ConcatDataset([l_val_dataset, u_val_dataset])
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=n_thread
    )
    
    # More accurate calculation of train steps
    n_train_samples = len(train_dataset)
    n_train_steps = ((n_train_samples + batch_size - 1) // batch_size) * n_epoch
    
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=custom_collate,
                                shuffle=False,
                                num_workers=n_thread)
    
    
    
    return train_loader, val_loader, n_train_steps, labels
    
    