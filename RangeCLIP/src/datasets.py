import numpy as np
import cv2, torch
import torchvision.transforms.functional as TF
from PIL import Image
import random
import os
import pandas as pd

class ImageDepthDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 depth_paths,
                 resize_shape,
                 augmentation_random_brightness=[-1, -1],
                 augmentation_random_contrast=[-1, -1],
                 augmentation_random_hue=[-1, -1],
                 augmentation_random_saturation=[-1, -1],
                 augmentation_random_flip_type=['none']):

        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.n_sample = len(self.image_paths)

        self.n_height, self.n_width = resize_shape
        
        # Set up data augmentations
        self.augmentation_random_brightness = augmentation_random_brightness
        self.augmentation_random_contrast = augmentation_random_contrast
        self.augmentation_random_hue = augmentation_random_hue
        self.augmentation_random_saturation = augmentation_random_saturation
        self.augmentation_random_flip_type = augmentation_random_flip_type
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        depth_path = self.depth_paths[index]
        
        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        
        image = image.resize((self.n_width, self.n_height), Image.BILINEAR)
        depth = depth.resize((self.n_width, self.n_height), Image.BILINEAR)

        image = augment_image(image,
                      self.augmentation_random_brightness,
                      self.augmentation_random_contrast,
                      self.augmentation_random_hue,
                      self.augmentation_random_saturation,
                      self.augmentation_random_flip_type)

        image = np.array(image)
        depth = np.array(depth)

        # Resize image to n_height by n_width with bilinear interpolation using cv2
        # Note: cv2 convention in dimensions is different from torch, numpy, etc.
        image = cv2.resize(image, (self.n_width, self.n_height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.n_width, self.n_height), interpolation=cv2.INTER_LINEAR)

        # Permute dimensions from HxWxC to CxHxW
        image = np.transpose(image, (2, 0, 1))
        depth = depth[None, :, :]
        
        depth = depth.astype(np.float32)
        image = image.astype(np.float32)
        
        return depth, image

    def __len__(self):
        return self.n_sample
    
    

class ImageDepthTextDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, root_dir, labels_dir, transform=None):
        """
        Args:
            metadata_file (string): Path to the metadata CSV file.
            root_dir (string): Directory with all the image, depth, and label files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.root_dir = root_dir
        self.transform = transform
        
        with open(labels_dir, 'r') as file:
            lines = file.readlines()

        self.labels = ['a depth map of ' + line.strip() for line in lines]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.metadata.iloc[idx, 0])
        depth_path = os.path.join(self.root_dir, self.metadata.iloc[idx, 1])

        img = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")
        object_id = self.metadata.iloc[idx, 3]

        if self.transform:
            img = self.transform(img)
            depth = self.transform(depth)

        return {
            'image': img,
            'depth': depth,
            'object_id': self.labels[object_id],
        }
    
    def get_labels(self):
        """Method to access all the label names."""
        return self.labels
        
def valid_augmentation(arr, l_bound, u_bound):
        return arr and len(arr) == 2 and arr[0] >= l_bound and arr[1] <= u_bound and arr[0] <= arr[1] and arr[1] > l_bound
    
    
'''
Helper function for data augmentation
'''
def augment_image(image,
                  random_brightness=[-1, -1],
                  random_contrast=[-1, -1],
                  random_hue=[-1, -1],
                  random_saturation=[-1, -1],
                  random_flip_type=['none']):
    '''
    Applies data augmentations to image

    Arg(s):
        image : PIL.Image
            Pillow Image object
        random_brightness : list[float]
            range of brightness adjustment
        random_contrast : list[float]
            range of contrast adjustment
        random_hue : list[float]
            range of hue adjustment between -0.5 to 0.5
        random_saturation : list[float]
            range of saturation adjustment
        random_flip_type : list[str]
            horizontal and vertical flip

    Returns:
        PIL.Image : Pillow Image object after augmentation
    '''

    do_random_brightness = valid_augmentation(random_brightness, -1, 1)
    do_random_contrast = valid_augmentation(random_contrast, -1, 1)
    do_random_hue = valid_augmentation(random_hue, -0.5, 0.5)
    do_random_saturation = valid_augmentation(random_saturation, -1, 1)
    do_random_horizontal_flip = random_flip_type and 'horizontal' in random_flip_type
    do_random_vertical_flip = random_flip_type and 'vertical' in random_flip_type

    '''
    Perform data augmentation
    '''

    if do_random_brightness and random.random() < 0.5:

        brightness_min, brightness_max = random_brightness
        factor = random.uniform(brightness_min, brightness_max)
        image = TF.adjust_brightness(image, factor)

    if do_random_contrast and random.random() < 0.5:

        contrast_min, contrast_max = random_contrast
        factor = random.uniform(contrast_min, contrast_max)
        image = TF.adjust_contrast(image, factor)

    if do_random_hue and random.random() < 0.5:

        hue_min, hue_max = random_hue
        factor = random.uniform(hue_min, hue_max)
        image = TF.adjust_hue(image, factor)

    if do_random_saturation and random.random() < 0.5:

        saturation_min, saturation_max = random_saturation
        factor = random.uniform(saturation_min, saturation_max)
        image = TF.adjust_saturation(image, factor)

    if do_random_horizontal_flip and random.random() < 0.5:
        image = TF.hflip(image)

    if do_random_vertical_flip and random.random() < 0.5:
        image = TF.vflip(image)

    return image
