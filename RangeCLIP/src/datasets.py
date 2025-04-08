import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F
from torchvision import transforms

class TaggedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tag):
        self.dataset = dataset
        self.tag = tag  # "labeled" or "unlabeled"

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['__tag__'] = self.tag
        return sample

    def __len__(self):
        return len(self.dataset)

def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def default_image_transform():
    return transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP mean
        std=[0.26862954, 0.26130258, 0.27577711]   # CLIP std
    )

class ImageDepthDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, image_transform=None, depth_transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.image_transform = image_transform or default_image_transform()
        self.depth_transform = depth_transform or normalize_depth

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx, 0]
        depth_path = self.metadata.iloc[idx, 1]

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        img = F.to_tensor(img)
        depth = F.to_tensor(depth)  # shape: [1, H, W]

        img = self.image_transform(img)
        depth = self.depth_transform(depth)

        return {
            'depth': depth,
            'image': img
        }
        
        
        
class ImageDepthTextDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, labels_path, image_transform=None, depth_transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.image_transform = image_transform or default_image_transform()
        self.depth_transform = depth_transform or normalize_depth
        
        with open(labels_path, 'r') as file:
            lines = file.readlines()

        self.all_candidate_labels = ['a depth map of ' + line.strip() for line in lines]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx, 0]
        depth_path = self.metadata.iloc[idx, 1]

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        
        img = F.to_tensor(img)        
        depth = F.to_tensor(depth)
        
        img = self.image_transform(img)
        depth = self.depth_transform(depth)
                        
        object_id = self.metadata.iloc[idx, 3] - 1 # 1-based indexing
            
        return {
            'depth': depth,
            'image': img,
            'id': object_id}
    
    def get_labels(self):
        """Method to access all the label names."""
        return self.all_candidate_labels
        