'''
Generate pseudo ground truth dataset by cropping
patches from images and their corresponding depth maps
'''

import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import math
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils

parser = argparse.ArgumentParser()

parser.add_argument('--image_paths', type=str, required=True, help='path to txt file that includes the paths to all the images (.txt)')
parser.add_argument('--depth_paths', type=str, required=True, help='path to txt file that includes the paths to all the depth maps (.txt)')
parser.add_argument('--label_path', type=str, required=True, help='path to labels')
parser.add_argument('--output_path', type=str, required=True, help='output directory for patches')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
parser.add_argument('--top_k_classes', type=int, default=5, help='for each batch, we select top k most confident classes')
parser.add_argument('--classes_path', type=str, required=False, help='path to lvis file that includes all the classes')

args = parser.parse_args()
categories = data_utils.get_categories_from_vild_json_file(args.classes_path)

def get_detected_objects(detection_path):
    '''
    Sample detected object txt file:
    
    554 0.551533 0.594581 0.119481 0.241865 0.52267
    703 0.957308 0.400842 0.0853839 0.489669 0.487107
    236 0.0769664 0.816906 0.153642 0.0478897 0.40897
    364 0.499899 0.854104 0.999608 0.291792 0.35211
    265 0.825288 0.62603 0.273138 0.144442 0.329448
    '''
    try:
        content = data_utils.read_file(detection_path)
        detected_objects = content.splitlines()
        detected_objects = [x.split(' ') for x in detected_objects]
        detected_objects = [[int(float(x[0])), *[float(v) for v in x[1:]]] for x in detected_objects]

        detected_objects = [{
            'class': categories[int(x[0])],
            'x': x[1],
            'y': x[2],
            'w': x[3],
            'h': x[4],
            'conf': x[5]
        } for x in detected_objects]
        
        return detected_objects
    except (FileNotFoundError, IndexError) as e:
        print(f"Error processing detection file {detection_path}: {e}")
        return []

def get_balanced_top_k_classes(detected_objects_batch, k):
    '''
    Balance between high confidence scores and frequency of detection
    '''
    class_stats = {}
    
    # Process all detected objects in the batch
    for obj_list in detected_objects_batch:
        for obj in obj_list:
            class_name = obj['class']
            conf = obj['conf']
            
            if class_name not in class_stats:
                class_stats[class_name] = {
                    'count': 0,
                    'total_conf': 0,
                    'max_conf': 0
                }
            
            class_stats[class_name]['count'] += 1
            class_stats[class_name]['total_conf'] += conf
            class_stats[class_name]['max_conf'] = max(conf, class_stats[class_name]['max_conf'])
    
    # Calculate average confidence and a combined score
    for class_name, stats in class_stats.items():
        stats['avg_conf'] = stats['total_conf'] / stats['count']
        
        # Combined score: balance between frequency and confidence
        # You can adjust the weights (0.4 and 0.6) to favor frequency or confidence
        stats['score'] = (0.4 * (stats['count'] / len(detected_objects_batch))) + (0.6 * stats['max_conf'])
    
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    return [class_name for class_name, _ in sorted_classes[:k]]

def get_top_k_confident_classes(detected_objects_batch, k):
    '''
    Given a batch of detected objects, return k class names corresponding
    to the classes that have been detected with the highest confidence
    among all detected objects in the batch
    '''
    class_to_conf = {}
    
    # Process all detected objects in the batch
    for obj_list in detected_objects_batch:
        for obj in obj_list:
            class_name = obj['class']
            conf = obj['conf']
            
            if class_name not in class_to_conf or conf > class_to_conf[class_name]:
                class_to_conf[class_name] = conf
    
    sorted_classes = sorted(class_to_conf.items(), key=lambda x: x[1], reverse=True)
    return [class_name for class_name, _ in sorted_classes[:k]]

def safe_crop(image, box):
    """Safely crop an image with error handling and boundary checks"""
    try:
        width, height = image.size
        left, top, right, bottom = box
        
        # Ensure box coordinates are within image boundaries
        left = max(0, min(width-1, left))
        top = max(0, min(height-1, top))
        right = max(left+1, min(width, right))
        bottom = max(top+1, min(height, bottom))
        
        return image.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None

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
    
    batch_size = min(args.batch_size, len(image_paths))
    num_batches = math.ceil(len(image_paths) / batch_size)
    
    # Output images will be indexed
    num_sample_each_class = {}
    
    image_base_path = os.path.commonpath(image_paths)
    
    # Process images in batches
    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            
            # Get images and depth maps for this batch
            image_batch_paths = image_paths[start_idx:end_idx]
            depth_batch_paths = depth_paths[start_idx:end_idx]
            
            # Construct detection paths based on image names
            detection_batch_paths = []
            
            for img_path in image_batch_paths:
                rel_path = os.path.relpath(img_path, start=image_base_path)
                detection_path = os.path.join(args.label_path, os.path.splitext(rel_path)[0] + ".txt")
                detection_batch_paths.append(detection_path)
            
            # Get all detected objects in a batch
            detected_objects_batch = []
            batch_data = []
            
            for i, (img_path, depth_path, detection_path) in enumerate(zip(image_batch_paths, depth_batch_paths, detection_batch_paths)):
                detected_objects = get_detected_objects(detection_path)
                detected_objects_batch.append(detected_objects)
                batch_data.append({
                    'img_path': img_path,
                    'depth_path': depth_path,
                    'detected_objects': detected_objects
                })
            
            # Find top k classes across the entire batch
            top_k_classes = get_balanced_top_k_classes(detected_objects_batch, args.top_k_classes)
            
            # Process each image in the batch
            for data in batch_data:
                img_path = data['img_path']
                depth_path = data['depth_path']
                detected_objects = data['detected_objects']
                
                # Load images only if needed (when objects of interest are found)
                relevant_objects = [obj for obj in detected_objects if obj['class'] in top_k_classes]
                if not relevant_objects:
                    continue
                
                try:
                    # Only load images when needed
                    scene_image = Image.open(img_path)
                    scene_depth = Image.open(depth_path)
                    
                    width = scene_image.width
                    height = scene_image.height
                    
                    # Process each object
                    for obj in relevant_objects:
                        detected_class = obj['class']
                        
                        # Calculate crop coordinates
                        x, y, w, h = obj['x'] * width, obj['y'] * height, obj['w'] * width, obj['h'] * height
                        left = int(x - w / 2.0)
                        top = int(y - h / 2.0)
                        right = int(x + w / 2.0)
                        bottom = int(y + h / 2.0)
                        
                        # Crop images
                        cropped_image = safe_crop(scene_image, (left, top, right, bottom))
                        cropped_depth = safe_crop(scene_depth, (left, top, right, bottom))
                        
                        if cropped_image is None or cropped_depth is None:
                            continue
                        
                        # Prepare output directories
                        cropped_image_dir = os.path.join(image_crop_output_path, detected_class)
                        cropped_depth_dir = os.path.join(depth_crop_output_path, detected_class)
                        
                        os.makedirs(cropped_image_dir, exist_ok=True)
                        os.makedirs(cropped_depth_dir, exist_ok=True)
                        
                        # Save output file and name by index number.png
                        sample_index = num_sample_each_class.get(detected_class, 0)
                        cropped_image_path = os.path.join(cropped_image_dir, f"{sample_index}.png")
                        cropped_depth_path = os.path.join(cropped_depth_dir, f"{sample_index}.png")
                        
                        cropped_image.save(cropped_image_path)
                        cropped_depth.save(cropped_depth_path)
                        
                        # Update the counter
                        num_sample_each_class[detected_class] = sample_index + 1
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                
            # Update progress bar
            pbar.update(end_idx - start_idx)
            pbar.set_description(f"Processed {end_idx}/{len(image_paths)} images")
    
    # Print summary
    print("Cropping completed.")
    print(f"Generated crops for {len(num_sample_each_class)} classes:")
    for class_name, count in sorted(num_sample_each_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {class_name}: {count} samples")

if __name__ == '__main__':
    generate_cropped_patches()