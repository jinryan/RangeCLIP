import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
import pandas as pd

# Function to find bounding boxes with padding
def get_padded_bounding_boxes(label_map, padding=10):
    """Find bounding boxes of unique objects in a label map with padding."""
    unique_labels = np.unique(label_map)
    bounding_boxes = []
    
    height, width = label_map.shape
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        mask = (label_map == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Apply padding while ensuring the box stays within image boundaries
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + w + padding, width)
            y2 = min(y + h + padding, height)

            bounding_boxes.append((label, x1, y1, x2, y2))
    
    return bounding_boxes


output_dir = "training/nyu_depth_v2/labelled_patches"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

# Load the dataset
file_path = "setup/nyu_depth_v2/nyu_depth_v2_labeled.mat"
with h5py.File(file_path, "r") as data:
    images = np.array(data["images"])  # (3, H, W, N)
    depths = np.array(data["depths"])  # (H, W, N)
    labels = np.array(data["labels"])  # (H, W, N)

images = images.transpose(0, 3, 2, 1)  # (N, H, W, 3)
depths = depths.transpose(0, 2, 1)  # (N, H, W)
labels = labels.transpose(0, 2, 1)  # (N, H, W)


# Metadata storage
metadata = []

# Generate dataset
sample_count = images.shape[0]  # Use full dataset
padding_size = 20

for i in range(sample_count):
    img, depth, label_map = images[i], depths[i], labels[i]
    object_bboxes = get_padded_bounding_boxes(label_map, padding=padding_size)
    
    for obj_id, x1, y1, x2, y2 in object_bboxes:
        # Crop images
        img_crop = img[y1:y2, x1:x2]
        depth_crop = depth[y1:y2, x1:x2]
        label_crop = label_map[y1:y2, x1:x2]

        # Resize for consistency
        img_crop = cv2.resize(img_crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        depth_crop = cv2.resize(depth_crop, (128, 128), interpolation=cv2.INTER_NEAREST)
        label_crop = cv2.resize(label_crop, (128, 128), interpolation=cv2.INTER_NEAREST)

        # Define filenames
        filename = f"{i}_{obj_id}.png"
        img_path = os.path.join(output_dir, "images", filename)
        depth_path = os.path.join(output_dir, "depths", filename)
        label_path = os.path.join(output_dir, "labels", filename)

        # Save images
        cv2.imwrite(img_path, cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(depth_path, (depth_crop * 255 / np.max(depth_crop)).astype(np.uint8))
        cv2.imwrite(label_path, (label_crop * 255 / np.max(label_crop)).astype(np.uint8))

        # Store metadata
        metadata.append([img_path, depth_path, label_path, obj_id, (x1, y1, x2, y2)])

# Save metadata CSV
df = pd.DataFrame(metadata, columns=["image_filename", "depth_filename", "label_filename", "object_id", "bounding_box"])
df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

print(f"Dataset creation complete! Total samples: {len(metadata)}")