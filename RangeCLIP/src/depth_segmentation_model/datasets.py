import os
import torch
from PIL import Image
import pandas as pd
import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms


def normalize_depth(depth_tensor):
    # Depth expected as 1xHxW torch tensor
    return (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)

def default_image_transform():
    return transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP mean
        std=[0.26862954, 0.26130258, 0.27577711]   # CLIP std
    )

class ImageDepthTextDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, labels_path, image_transform=None, depth_transform=None, segmentation_transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.image_transform = image_transform or default_image_transform()
        self.depth_transform = depth_transform or normalize_depth
        self.segmentation_transform = segmentation_transform
        self.root_dir = os.path.dirname(metadata_file)
        
        df = pd.read_csv(labels_path, usecols=['label', 'index'], na_values=[], keep_default_na=False)
        df = df.sort_values(by='index', ascending=True)
        self.labels = df['label'].tolist()

        assert df['index'].tolist() == list(range(1, len(self.labels) + 1)), "Indices must be 1-based and consecutive"


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, row['image_path'])
        depth_path = os.path.join(self.root_dir, row['depth_path'])
        segmentation_path = os.path.join(self.root_dir, row['label_path'])
        

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("I")
        segmentation = Image.open(segmentation_path).convert("I")

        img = F.to_tensor(img)
        depth = F.to_tensor(depth)
        depth = depth.float()
        segmentation = np.array(segmentation)

        img = self.image_transform(img)
        depth = self.depth_transform(depth)

        if self.segmentation_transform is not None:
            segmentation = self.segmentation_transform(segmentation)

        return {
            'depth': depth,
            'image': img,
            'segmentation': segmentation,
        }

    def get_candidate_labels(self):
        return self.labels
