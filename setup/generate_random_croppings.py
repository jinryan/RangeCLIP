import os
import sys
import argparse
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils
import random
import cv2
import traceback


class FastPatchGenerator:
    def __init__(self, args):
        """
        Initialize the patch generator with configuration parameters.
        
        Args:
            args: Parsed command-line arguments
        """
        self.image_paths = data_utils.read_paths(args.image_paths)
        self.depth_paths = data_utils.read_paths(args.depth_paths)
        self.output_path = args.output_path
        self.num_crops = args.num_crops
        
        # Create output directories
        self.image_crop_output_path = os.path.join(self.output_path, 'images')
        self.depth_crop_output_path = os.path.join(self.output_path, 'depth')
        os.makedirs(self.image_crop_output_path, exist_ok=True)
        os.makedirs(self.depth_crop_output_path, exist_ok=True)
        
        # Validate input data
        self._validate_input_data()
    
    def _validate_input_data(self):
        """
        Perform comprehensive validation of input image and depth paths.
        """
        if len(self.image_paths) != len(self.depth_paths):
            raise ValueError(f"Mismatch in image and depth map counts: {len(self.image_paths)} vs {len(self.depth_paths)}")
        
        if len(self.image_paths) != len(set(self.image_paths)):
            raise ValueError("Duplicate image paths detected")
    
    def _load_image_and_depth(self, image_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Efficiently load image and depth map as NumPy arrays.
        
        Args:
            image_path: Path to the input image
            depth_path: Path to the corresponding depth map
        
        Returns:
            Tuple of image and depth map as uint8 NumPy arrays
        """
        image = np.array(Image.open(image_path), dtype=np.uint8)
        depth = np.array(Image.open(depth_path), dtype=np.uint8)
        
        # Ensure RGB and grayscale
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:  # Handle RGBA
            image = image[:, :, :3]
        
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
        
        # Warn if unable to generate requested number of crops
        return crops
    
    def _process_single_image(self, args):
        """
        Process a single image-depth pair and generate crops.
        
        Args:
            args: Tuple (index, image_path, depth_path)
        """
        index, (image_path, depth_path) = args  # Unpack the tuple
        try:
            image, depth = self._load_image_and_depth(image_path, depth_path)
            
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
                    self.image_crop_output_path, f"{index}_{crop_index}.png"
                )
                cropped_depth_path = os.path.join(
                    self.depth_crop_output_path, f"{index}_{crop_index}.png"
                )
                
                if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                    cv2.imwrite(cropped_image_path, cropped_image)
                    cv2.imwrite(cropped_depth_path, cropped_depth)
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            print(traceback.format_exc())
            exit(1)
            
    
    def generate_patches(self, num_workers: int = None):
        """
        Generate patches using multiprocessing.
        
        Args:
            num_workers: Number of parallel processes (defaults to CPU count)
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        with mp.Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(
                    partial(self._process_single_image), 
                    enumerate(zip(self.image_paths, self.depth_paths), 1)
                ), 
                total=len(self.image_paths)
            ))

def main():
    parser = argparse.ArgumentParser(description="Optimized Patch Generation")
    parser.add_argument('--image_paths', type=str, required=True, 
                        help='Path to txt file with image paths')
    parser.add_argument('--depth_paths', type=str, required=True, 
                        help='Path to txt file with depth map paths')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Output directory for patches')
    parser.add_argument('--num_crops', type=int, default=4, 
                        help='Number of crops per image')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    generator = FastPatchGenerator(args)
    generator.generate_patches(num_workers=args.workers)

if __name__ == '__main__':
    main()