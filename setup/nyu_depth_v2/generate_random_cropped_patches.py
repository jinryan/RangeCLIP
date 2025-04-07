import os
import cv2
import h5py
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional
from tqdm import tqdm
from functools import partial
from PIL import Image
import glob

import argparse
    
parser = argparse.ArgumentParser(description="Generate image and depth patches from H5 files")
parser.add_argument("--h5_directory", type=str, default="/media/common/datasets/nyu_depth_v2/train",
                    help="Directory containing H5 files with 'rgb' and 'depth' keys")
parser.add_argument("--output_path", type=str, required=True,
                    help="Directory to save output patches")
parser.add_argument("--num_crops", type=int, default=4,
                    help="Number of crops to generate per H5 file")
parser.add_argument("--num_workers", type=int, default=None,
                    help="Number of worker processes (defaults to CPU count - 1)")

args = parser.parse_args()

class FastH5PatchGenerator:
    def __init__(self, args):
        """
        Initialize the patch generator with configuration parameters.
        
        Args:
            args: Parsed command-line arguments with the following attributes:
                - h5_directory: Directory containing H5 files
                - output_path: Directory to save the output crops
                - num_crops: Number of crops to generate per image
        """
        self.h5_directory = args.h5_directory
        self.output_path = args.output_path
        self.num_crops = args.num_crops
        
        # Find all .h5 files in the directory
        self.h5_paths = glob.glob(os.path.join(self.h5_directory, '**/*.h5'), recursive=True)
        
        # Create output directories
        self.image_crop_output_path = os.path.join(self.output_path, 'images')
        self.depth_crop_output_path = os.path.join(self.output_path, 'depth')
        os.makedirs(self.image_crop_output_path, exist_ok=True)
        os.makedirs(self.depth_crop_output_path, exist_ok=True)
        
        # Validate input data
        self._validate_input_data()
    
    def _validate_input_data(self):
        """
        Perform comprehensive validation of input H5 files.
        """
        if len(self.h5_paths) == 0:
            raise ValueError(f"No H5 files found in directory: {self.h5_directory}")
        
        # Check the first file to ensure it has the required keys
        try:
            with h5py.File(self.h5_paths[0], 'r') as f:
                if 'rgb' not in f or 'depth' not in f:
                    raise ValueError(f"H5 file missing required 'rgb' and/or 'depth' keys: {self.h5_paths[0]}")
        except Exception as e:
            raise ValueError(f"Error validating first H5 file: {str(e)}")
        
        print(f"Found {len(self.h5_paths)} valid H5 files for processing")
    
    def _load_image_and_depth_from_h5(self, h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficiently load image and depth map from H5 file as NumPy arrays.
        
        Args:
            h5_path: Path to the H5 file containing 'rgb' and 'depth' keys
        
        Returns:
            Tuple of image and depth map as NumPy arrays
        """
        with h5py.File(h5_path, 'r') as f:
            # Load RGB image and convert to uint8 if necessary
            image = np.array(f['rgb'])
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Load depth map and normalize if necessary
            depth = np.array(f['depth'])
            if depth.dtype != np.uint8:
                # Normalize depth to 0-255 range for visualization and storage
                depth_min = np.min(depth)
                depth_max = np.max(depth)
                if depth_max > depth_min:
                    depth = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                else:
                    depth = np.zeros_like(depth, dtype=np.uint8)
        
        # Ensure RGB format
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:  # Handle RGBA
            image = image[:, :, :3]
        
        # Ensure depth is 2D
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        return image, depth
    
    def generate_flexible_crops(
        self,
        image: np.ndarray, 
        depth: np.ndarray, 
        num_crops: int = 4, 
        min_crop_size: int = 64, 
        max_crop_size: Optional[int] = None,
        max_attempts: int = 1000,
        max_overlap_ratio: float = 0.5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate random, non-significantly overlapping crops with flexible positioning and sizing.
        
        Args:
            image: Input image as NumPy array
            depth: Corresponding depth map
            num_crops: Number of crops to generate
            min_crop_size: Minimum width/height of a crop
            max_crop_size: Maximum width/height of a crop (defaults to min(image height, image width))
            max_attempts: Maximum number of attempts to generate crops
            max_overlap_ratio: Maximum allowed overlap between crops (0.0 to 1.0)
        
        Returns:
            List of (image_crop, depth_crop) tuples
        """
        height, width = image.shape[:2]
        
        # Set max crop size if not specified
        if max_crop_size is None:
            max_crop_size = min(height, width)
        
        # Validate input parameters
        if min_crop_size > max_crop_size:
            raise ValueError("Minimum crop size cannot be larger than maximum crop size")
        
        def compute_crop_overlap(crop1, crop2):
            """
            Compute the overlap ratio between two crops.
            
            Args:
                crop1: (y_start, x_start, height, width) of first crop
                crop2: (y_start, x_start, height, width) of second crop
            
            Returns:
                Overlap ratio between 0 and 1
            """
            y1_start, x1_start, h1, w1 = crop1
            y2_start, x2_start, h2, w2 = crop2
            
            # Compute intersection
            y_overlap_start = max(y1_start, y2_start)
            x_overlap_start = max(x1_start, x2_start)
            y_overlap_end = min(y1_start + h1, y2_start + h2)
            x_overlap_end = min(x1_start + w1, x2_start + w2)
            
            # Compute intersection area
            intersection_height = max(0, y_overlap_end - y_overlap_start)
            intersection_width = max(0, x_overlap_end - x_overlap_start)
            intersection_area = intersection_height * intersection_width
            
            # Compute area of the first crop as reference
            crop1_area = h1 * w1
            
            # Compute overlap ratio
            return intersection_area / crop1_area if crop1_area > 0 else 0.0
        
        crops = []
        crop_coords = []
        attempts = 0
        
        while len(crops) < num_crops and attempts < max_attempts:
            # Randomly choose crop size
            crop_height = np.random.randint(min_crop_size, max_crop_size + 1)
            crop_width = np.random.randint(min_crop_size, max_crop_size + 1)
            
            # Randomly choose crop position
            max_y = height - crop_height
            max_x = width - crop_width
            
            if max_y < 0 or max_x < 0:
                attempts += 1
                continue
                
            y_start = np.random.randint(0, max_y + 1)
            x_start = np.random.randint(0, max_x + 1)
            
            # Extract crop
            image_crop = image[y_start:y_start+crop_height, x_start:x_start+crop_width]
            depth_crop = depth[y_start:y_start+crop_height, x_start:x_start+crop_width]
            
            # Check overlap with existing crops
            is_valid_crop = True
            for existing_crop_coords in crop_coords:
                overlap = compute_crop_overlap(
                    (y_start, x_start, crop_height, crop_width), 
                    existing_crop_coords
                )
                
                if overlap > max_overlap_ratio:
                    is_valid_crop = False
                    break
            
            # Add crop if valid
            if is_valid_crop:
                crops.append((image_crop, depth_crop))
                crop_coords.append((y_start, x_start, crop_height, crop_width))
            
            attempts += 1
        
        return crops
    
    def _process_single_h5_file(self, args):
        """
        Process a single H5 file and generate crops.
        
        Args:
            args: Tuple (index, h5_path)
        """
        index, h5_path = args  # Unpack the tuple
        try:
            # Extract directory and filename to create a unique identifier
            rel_path = os.path.relpath(h5_path, self.h5_directory)
            base_name = os.path.splitext(rel_path)[0].replace('/', '_')
            
            image, depth = self._load_image_and_depth_from_h5(h5_path)
            
            crops = self.generate_flexible_crops(
                image, 
                depth,
                num_crops=self.num_crops,
                min_crop_size=64,
                max_attempts=20,
                max_overlap_ratio=0.3
            )
            
            # Save crops
            for crop_index, (cropped_image, cropped_depth) in enumerate(crops):
                cropped_image_path = os.path.join(
                    self.image_crop_output_path, f"{base_name}_{crop_index}.png"
                )
                cropped_depth_path = os.path.join(
                    self.depth_crop_output_path, f"{base_name}_{crop_index}.png"
                )
                
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cv2.imwrite(cropped_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(cropped_depth_path, cropped_depth)
            
            return len(crops)
        
        except Exception as e:
            import traceback
            print(f"Error processing H5 file {h5_path}: {e}")
            print(traceback.format_exc())
            return 0
    
    def generate_patches(self, num_workers: int = None):
        """
        Generate patches using multiprocessing.
        
        Args:
            num_workers: Number of parallel processes (defaults to CPU count)
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        total_crops = 0
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(
                    self._process_single_h5_file, 
                    enumerate(self.h5_paths)
                ), 
                total=len(self.h5_paths),
                desc="Processing H5 files"
            ))
            
            total_crops = sum(results)
        
        print(f"Generated {total_crops} crops from {len(self.h5_paths)} H5 files")


if __name__ == "__main__":
    generator = FastH5PatchGenerator(args)
    generator.generate_patches(num_workers=args.num_workers)