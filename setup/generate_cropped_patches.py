'''
Generate pseudo ground truth for range map object detection
by using YOLO to perform object detection on corresponding images
'''

import os, sys, argparse
from PIL import Image
from tqdm import tqdm
import math
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils

parser = argparse.ArgumentParser()

parser.add_argument('--image_paths', type=str, required=True, help='path to txt file that includes the paths to all the images (.txt)')
parser.add_argument('--depth_paths', type=str, required=True, help='path to txt file that includes the paths to all the depth maps (.txt)')
parser.add_argument('--label_path', type=str, required=True, help='path to labels')
parser.add_argument('--categories_path', type=str, required=True, help='path to lvis category json file', default='../data/lvis/lvis_val_100.json')
parser.add_argument('--output_path', type=str, required=True, help='output directory for patches')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
parser.add_argument('--top_k_classes', type=int, default=5, help='for each batch, we select top k most confident classes')
parser.add_argument('--classes_path', type=str, required=False, help='path to file that includes all the classes')

args = parser.parse_args()
categories=data_utils.get_categories_from_vild_json_file(args.categories_path)

def get_detected_objects(detection_path):
    detected_objects = data_utils.read_file(detection_path).splitlines()
    detected_objects = [x.split(' ') for x in detected_objects]
    detected_objects = [[float(v) for v in x] for x in detected_objects]

    detected_objects = [{
        'class': categories[x[0]],
        'x': int(x[1]),
        'y': int(x[2]),
        'w': int(x[3]),
        'h': int(x[4]),
        'conf': x[5]
    } for x in detected_objects]
    
    return detected_objects

def get_top_k_confident_objects(detected_objects, k=5):
    class_to_conf = {}
    
    for obj in detected_objects:
        class_name = obj['class']
        conf = obj['conf']
        
        if class_name not in class_to_conf or conf > class_to_conf[class_name]:
            class_to_conf[class_name] = conf
    
    sorted_classes = sorted(class_to_conf.items(), key=lambda x: x[1], reverse=True)
    return [class_name for class_name, _ in sorted_classes[:k]]

def generate_cropped_patches():

    image_paths = data_utils.read_paths(args.image_paths)
    depth_paths = data_utils.read_paths(args.depth_paths)

    batch_size = args.batch_size if args.batch_size else len(image_paths)
    num_batches = math.ceil(len(image_paths) / batch_size)
    
    image_crop_output_path = os.path.join(args.output_path, 'images')
    depth_crop_output_path = os.path.join(args.output_path, 'depth')
    
    num_sample_each_class = {}
    

    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for i in range(0, len(image_paths), batch_size):
            # Get images and depth maps
            image_batch_paths = image_paths[i:i + batch_size]
            depth_batch_paths = depth_paths[i:i + batch_size]
            detection_batch_paths = [args.output_path + x.split('/')[-1][:-3] + 'txt' for x in image_batch_paths]
            
            batch_dict = {det: {img: img, depth: depth} for (det, img, depth) in zip(detection_batch_paths, image_batch_paths, depth_batch_paths)}
            
            detected_objects = []
            for detection_path in detection_batch_paths:
                detection_batch_paths[detection_path]['det_objs'] = get_detected_objects(detection_path)
                detected_objects.append(get_detected_objects(detection_path))
            
            top_k_classes = get_top_k_confident_objects(detected_objects, args.top_k_classes)
            
            for _, value in batch_dict:
                for (img, depth, det_objs) in zip(value['img'], value['depth'], value['det_objs']):
                    for detected_object in det_objs:
                        detected_class = detected_object['class']
                        if detected_class in top_k_classes:
                            # Crop image and depth map
                            scene_image = Image.open(img)
                            scene_depth = Image.open(depth)
                            
                            cropped_image = scene_image.crop((detected_object['x'] - detected_object['w'] / 2.0, # left
                                                            detected_object['y'] + detected_object['h'] / 2.0, # top
                                                            detected_object['x'] + detected_object['w'] / 2.0, # right
                                                            detected_object['y'] - detected_object['h'] / 2.0)) # bottom
                            
                            cropped_depth = scene_depth.crop((detected_object['x'] - detected_object['w'] / 2.0, # left
                                                            detected_object['y'] + detected_object['h'] / 2.0, # top
                                                            detected_object['x'] + detected_object['w'] / 2.0, # right
                                                            detected_object['y'] - detected_object['h'] / 2.0)) # bottom
                        
                            cropped_image_dir = os.path.join(image_crop_output_path, detected_class)
                            cropped_depth_dir = os.path.join(depth_crop_output_path, detected_class)
                            
                            os.makedirs(cropped_image_dir, exist_ok=True)
                            os.makedirs(cropped_depth_dir, exist_ok=True)
                            
                            cropped_image_path = os.path.join(cropped_image_dir, str(num_sample_each_class.get(detected_class, 0)) + '.png')
                            cropped_depth_path = os.path.join(cropped_depth_dir, str(num_sample_each_class.get(detected_class, 0)) + '.png')
                            
                            cropped_image.save(cropped_image_path)
                            cropped_depth.save(cropped_depth_path)
                
                    
                
                # Save to output

                pbar.update(len(batch))
                
                pbar.set_description(f"Processed {i + len(batch)}/{len(image_paths)} images")

    print("Inference completed.")

if __name__ == '__main__':
    generate_cropped_patches()