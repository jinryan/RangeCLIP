import os
import cv2
import h5py
import numpy as np
import multiprocessing as mp
import pandas as pd
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from functools import partial
from PIL import Image
import glob
import csv
import tempfile

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
        
        # Set up metadata CSV path
        self.metadata_path = os.path.join(self.output_path, 'metadata.csv')
        
        # Create metadata CSV with headers
        with open(self.metadata_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'source_h5_file',
                'image_crop_path',
                'depth_crop_path',
                'crop_index',
                'crop_height',
                'crop_width',
                'crop_y_start',
                'crop_x_start',
                'original_height',
                'original_width'
            ])
        
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
    
    def _load_image_and_depth_from_h5(self, h5_path):
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
    
    def generate_flexible_crops(self, image, depth, num_crops=4, min_crop_size=32, 
                               max_crop_size=None, max_attempts=1000, max_overlap_ratio=0.5):
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
            List of dicts with crop data and coordinates
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
                crops.append({
                    'image_crop': image_crop,
                    'depth_crop': depth_crop,
                    'y_start': y_start,
                    'x_start': x_start,
                    'height': crop_height,
                    'width': crop_width
                })
                crop_coords.append((y_start, x_start, crop_height, crop_width))
            
            attempts += 1
        
        return crops

# Static function to be used with multiprocessing
def process_h5_file(args_tuple, h5_directory, image_crop_output_path, depth_crop_output_path, num_crops):
    """
    Process a single H5 file and generate crops.
    
    Args:
        args_tuple: Tuple containing (index, h5_path)
        h5_directory: Base directory for H5 files
        image_crop_output_path: Output directory for image crops
        depth_crop_output_path: Output directory for depth crops
        num_crops: Number of crops to generate per file
    
    Returns:
        Tuple containing (num_crops_generated, metadata_rows)
    """
    index, h5_path = args_tuple
    try:
        # Extract directory and filename to create a unique identifier
        rel_path = os.path.relpath(h5_path, h5_directory)
        base_name = os.path.splitext(rel_path)[0].replace('/', '_')
        
        # Load image and depth data
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
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure depth is 2D
        if depth.shape[0] == 1:
            depth = np.transpose(depth, (1, 2, 0))
        
        orig_height, orig_width = image.shape[:2]
        
        # Generate crops
        crops = []
        crop_coords = []
        attempts = 0
        max_attempts = 20
        min_crop_size = 32
        max_crop_size = min(orig_height, orig_width)
        max_overlap_ratio = 0.3
        
        while len(crops) < num_crops and attempts < max_attempts:
            # Randomly choose crop size
            crop_height = np.random.randint(min_crop_size, max_crop_size + 1)
            crop_width = np.random.randint(min_crop_size, max_crop_size + 1)
            
            # Randomly choose crop position
            max_y = orig_height - crop_height
            max_x = orig_width - crop_width
            
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
                y1_start, x1_start, h1, w1 = existing_crop_coords
                
                # Compute intersection
                y_overlap_start = max(y_start, y1_start)
                x_overlap_start = max(x_start, x1_start)
                y_overlap_end = min(y_start + crop_height, y1_start + h1)
                x_overlap_end = min(x_start + crop_width, x1_start + w1)
                
                # Compute intersection area
                intersection_height = max(0, y_overlap_end - y_overlap_start)
                intersection_width = max(0, x_overlap_end - x_overlap_start)
                intersection_area = intersection_height * intersection_width
                
                # Compute area of the first crop as reference
                crop1_area = crop_height * crop_width
                
                # Compute overlap ratio
                overlap = intersection_area / crop1_area if crop1_area > 0 else 0.0
                
                if overlap > max_overlap_ratio:
                    is_valid_crop = False
                    break
            
            # Add crop if valid
            if is_valid_crop:
                crops.append({
                    'image_crop': image_crop,
                    'depth_crop': depth_crop,
                    'y_start': y_start,
                    'x_start': x_start,
                    'height': crop_height,
                    'width': crop_width
                })
                crop_coords.append((y_start, x_start, crop_height, crop_width))
            
            attempts += 1
        
        metadata_rows = []
        
        # Save crops
        for crop_index, crop_data in enumerate(crops):
            image_crop = crop_data['image_crop']
            depth_crop = crop_data['depth_crop']
            
            # Generate output filenames
            image_filename = f"{base_name}_{crop_index}.png"
            depth_filename = f"{base_name}_{crop_index}.png"
            
            image_crop_path = os.path.join(image_crop_output_path, image_filename)
            depth_crop_path = os.path.join(depth_crop_output_path, depth_filename)
            
            # Save the crop images
            if image_crop.shape[0] > 0 and image_crop.shape[1] > 0:
                cv2.imwrite(image_crop_path, cv2.cvtColor(image_crop, cv2.COLOR_RGB2BGR))
                cv2.imwrite(depth_crop_path, depth_crop)
                
                # Prepare metadata row
                metadata_rows.append([
                    h5_path,
                    os.path.abspath(image_crop_path),
                    os.path.abspath(depth_crop_path),
                    crop_index,
                    crop_data['height'],
                    crop_data['width'],
                    crop_data['y_start'],
                    crop_data['x_start'],
                    orig_height,
                    orig_width
                ])
        
        # Return metadata rows to be written in the main process
        return len(crops), metadata_rows
    
    except Exception as e:
        import traceback
        print(f"Error processing H5 file {h5_path}: {e}")
        print(traceback.format_exc())
        return 0, []

class FastH5PatchGeneratorFixed:
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
        
        # Set up metadata CSV path
        self.metadata_path = os.path.join(self.output_path, 'metadata.csv')
        
        # Create metadata CSV with headers
        with open(self.metadata_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'source_h5_file',
                'image_crop_path',
                'depth_crop_path',
                'crop_index',
                'crop_height',
                'crop_width',
                'crop_y_start',
                'crop_x_start',
                'original_height',
                'original_width'
            ])
        
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
    
    def _write_metadata_rows(self, all_metadata_rows):
        """
        Write all metadata rows to the CSV file.
        
        Args:
            all_metadata_rows: List of rows to write to the CSV
        """
        with open(self.metadata_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_metadata_rows)
    
    def generate_patches(self, num_workers=None):
        """
        Generate patches using multiprocessing.
        
        Args:
            num_workers: Number of parallel processes (defaults to CPU count)
        """
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        print(f"Starting patch generation with {num_workers} workers...")
        
        total_crops = 0
        all_metadata_rows = []
        
        # Use partial to fix arguments to the process_h5_file function
        process_func = partial(
            process_h5_file,
            h5_directory=self.h5_directory,
            image_crop_output_path=self.image_crop_output_path,
            depth_crop_output_path=self.depth_crop_output_path,
            num_crops=self.num_crops
        )
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(
                    process_func,
                    enumerate(self.h5_paths)
                ), 
                total=len(self.h5_paths),
                desc="Processing H5 files"
            ))
        
        # Collect results and metadata rows
        for num_crops, metadata_rows in results:
            total_crops += num_crops
            all_metadata_rows.extend(metadata_rows)
        
        # Write all metadata at once
        self._write_metadata_rows(all_metadata_rows)
        
        print(f"Generated {total_crops} crops from {len(self.h5_paths)} H5 files")
        print(f"Metadata saved to: {self.metadata_path}")
        
        # Generate a summary dataframe
        try:
            df = pd.read_csv(self.metadata_path)
            print(f"\nMetadata Summary:")
            print(f"  Total crops: {len(df)}")
            print(f"  Unique source files: {df['source_h5_file'].nunique()}")
            print(f"  Average crop dimensions: {df['crop_height'].mean():.1f}x{df['crop_width'].mean():.1f}")
        except Exception as e:
            print(f"Could not generate metadata summary: {e}")


# Example usage (if run as main script)
if __name__ == "__main__":
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
    
    generator = FastH5PatchGeneratorFixed(args)
    generator.generate_patches(num_workers=args.num_workers)