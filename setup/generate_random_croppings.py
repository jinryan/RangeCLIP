'''
Generate pseudo ground truth dataset by randomly cropping
patches from images and their corresponding depth maps
'''

import os
import sys
import argparse
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, os.getcwd())   
import numpy as np
import utils.src.data_utils as data_utils
parser = argparse.ArgumentParser()

parser.add_argument('--image_paths', type=str, required=True, help='path to txt file that includes the paths to all the images (.txt)')
parser.add_argument('--depth_paths', type=str, required=True, help='path to txt file that includes the paths to all the depth maps (.txt)')
parser.add_argument('--output_path', type=str, required=True, help='output directory for patches')
parser.add_argument('--num_crops', type=int, default=4, help='Number of crops per image')


args = parser.parse_args()


def stratified_random_crop(
    image,
    depth_map,
    min_crop_size = (64, 64),
    max_crops = 16,
    overlap_threshold = 0.1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Perform stratified random cropping with improved sampling strategy.
    
    Args:
        image: Source image (PIL or NumPy)
        depth_map: Corresponding depth map (PIL or NumPy)
        min_crop_size: Minimum crop dimensions (height, width)
        max_crops: Maximum number of crops to generate
        overlap_threshold: Maximum allowed overlap between crops
    
    Returns:
        List of (image_crop, depth_crop) pairs
    """
    # Convert to NumPy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    if isinstance(depth_map, Image.Image):
        depth_map = np.array(depth_map)
    
    height, width = image.shape[:2]
    min_h, min_w = min_crop_size
    
    # Validate minimum crop size
    if min_h > height or min_w > width:
        raise ValueError("Crop size larger than image dimensions")
    
    # Adaptive stratification based on image size
    max_strata = min(8, max(4, int(min(height, width) / (min_h * 2))))
    
    crops = []
    existing_crops = []
    
    for strata_size in range(2, max_strata + 1):
        strata_height = height // strata_size
        strata_width = width // strata_size
        
        for i in range(strata_size):
            for j in range(strata_size):
                # Define stratum boundaries
                y_start = i * strata_height
                x_start = j * strata_width
                
                # Randomize crop within stratum
                max_y = max(0, strata_height - min_h)
                max_x = max(0, strata_width - min_w)
                
                offset_y = np.random.randint(0, max_y + 1)
                offset_x = np.random.randint(0, max_x + 1)
                
                crop_y = y_start + offset_y
                crop_x = x_start + offset_x
                
                image_crop = image[crop_y:crop_y+min_h, crop_x:crop_x+min_w]
                depth_crop = depth_map[crop_y:crop_y+min_h, crop_x:crop_x+min_w]
                
                # Check for significant overlap with existing crops
                if not _has_significant_overlap(
                    (crop_y, crop_x, min_h, min_w), 
                    existing_crops, 
                    overlap_threshold
                ):
                    crops.append((image_crop, depth_crop))
                    existing_crops.append((crop_y, crop_x, min_h, min_w))
                
                # Stop if we've reached max crops
                if len(crops) >= max_crops:
                    return crops
    
    return crops

def _has_significant_overlap(
    new_crop: Tuple[int, int, int, int], 
    existing_crops: List[Tuple[int, int, int, int]], 
    threshold: float = 0.1
) -> bool:
    """
    Check if a new crop significantly overlaps with existing crops.
    
    Args:
        new_crop: (y, x, height, width) of the new crop
        existing_crops: List of existing crop coordinates
        threshold: Maximum allowed overlap ratio
    
    Returns:
        Boolean indicating significant overlap
    """
    ny, nx, nh, nw = new_crop
    
    for ey, ex, eh, ew in existing_crops:
        # Compute intersection
        intersection_y = max(0, min(ny + nh, ey + eh) - max(ny, ey))
        intersection_x = max(0, min(nx + nw, ex + ew) - max(nx, ex))
        
        intersection_area = intersection_y * intersection_x
        new_crop_area = nh * nw
        
        if intersection_area / new_crop_area > threshold:
            return True
    
    return False

def generate_cropped_patches():
    # Create output directories if they don't exist
    image_crop_output_path = os.path.join(args.output_path, 'images')
    depth_crop_output_path = os.path.join(args.output_path, 'depth')

    os.makedirs(image_crop_output_path, exist_ok=True)
    os.makedirs(depth_crop_output_path, exist_ok=True)
    
    # Data source
    image_paths = data_utils.read_paths(args.image_paths)
    depth_paths = data_utils.read_paths(args.depth_paths)
    
    # Validate that image and depth lists have the same length
    if len(image_paths) != len(depth_paths):
        print(f"Error: Number of images ({len(image_paths)}) doesn't match number of depth maps ({len(depth_paths)})")
        return
    
    if len(image_paths) != len(set(image_paths)):
        print(f"Duplicate images")
        return
    
    

    for index, image_path, depth_path in tqdm(zip(range(1, len(image_paths)+1), image_paths, depth_paths), total=len(image_paths)):
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        
        image = np.array(image.convert("RGB"))  # Ensuring it's an RGB image
        depth = np.array(depth.convert("L"))  # Ensuring it's grayscale
        
        height, width = image.shape[:2]
        
        
        for i in range(2, 10):
            scale_factor = np.random.uniform(0.25, 0.75)  # Random scale factor between 25% and 75% of original size
            desired_width = max(64, int(width * scale_factor))
            desired_height = max(64, int(height * scale_factor))
            
            crops = stratified_random_crop(image, depth, min_crop_size=(desired_height, desired_width))

            for crop_index, (cropped_image, cropped_depth) in enumerate(crops):
                cropped_image_path = os.path.join(image_crop_output_path, f"{index}_{crop_index}.png")
                cropped_depth_path = os.path.join(depth_crop_output_path, f"{index}_{crop_index}.png")
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    Image.fromarray(cropped_image).save(cropped_image_path)
                    Image.fromarray(cropped_depth).save(cropped_depth_path)

if __name__ == '__main__':
    generate_cropped_patches()
   
