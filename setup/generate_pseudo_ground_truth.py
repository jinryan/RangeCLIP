from ultralytics import YOLO
import os, sys, argparse
from tqdm import tqdm
import math
import numpy as np
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=False, help='path to YOLO model (.pt)')
parser.add_argument('--image_paths', type=str, required=True, help='path to txt file that includes the paths to all the images (.txt)')
parser.add_argument('--output_path', type=str, required=True, help='output directory for labels')
parser.add_argument('--start', type=int, required=False, help='start index')
parser.add_argument('--end', type=int, required=False, help='end index')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
parser.add_argument('--classes_path', type=str, required=False, help='path to file that includes all the classes')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for cross-class NMS')

args = parser.parse_args()

def box_iou(box1, box2):
    """
    Calculate IoU between two boxes in xywh format (normalized)
    """
    # Convert xywh to xyxy
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def cross_class_nms(boxes, scores, class_ids, iou_threshold=0.5):
    """
    Perform NMS across all classes, keeping only the highest confidence prediction
    for each group of overlapping boxes regardless of class
    """
    if len(boxes) == 0:
        return [], [], []
    
    # Convert to numpy arrays for easier handling
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    # Sort by confidence
    indices = np.argsort(-scores)
    boxes = boxes[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    
    keep = []
    while len(boxes) > 0:
        # Keep the box with highest confidence
        keep.append(indices[0])
        
        # Calculate IoU of the kept box with the rest
        ious = np.array([box_iou(boxes[0], box) for box in boxes[1:]])
        
        # Find boxes with IoU less than threshold
        mask = ious < iou_threshold
        indices = indices[1:][mask]
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        class_ids = class_ids[1:][mask]
    
    return keep

def generate_pseudo_ground_truth():
    model = YOLO(args.model_path if args.model_path else "yolov8x-worldv2.pt")

    image_paths = data_utils.read_paths(args.image_paths)
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(image_paths)
    image_paths = image_paths[start_idx:end_idx]

    batch_size = args.batch_size
    num_batches = math.ceil(len(image_paths) / batch_size)

    if args.classes_path:
        categories = data_utils.get_categories_from_vild_json_file(args.classes_path)
        model.set_classes(categories)

    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Processing {len(image_paths)} images in {num_batches} batches of {batch_size}...")

    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]

            # Run YOLO inference without auto-saving
            results = model.predict(
                source=batch, 
                save_txt=False,  # Prevent automatic saving
                verbose=False,
                save_conf=True
            )

            for img_path, result in zip(batch, results):
                # Compute relative path from original image directory
                rel_path = os.path.relpath(img_path, start=os.path.commonpath(image_paths))
                label_path = os.path.join(args.output_path, os.path.splitext(rel_path)[0] + ".txt")

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(label_path), exist_ok=True)

                # Extract all detections
                all_boxes = []
                all_scores = []
                all_cls_ids = []
                
                for box in result.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf[0])
                    x, y, w, h = float(box.xywhn[0][0]), float(box.xywhn[0][1]), float(box.xywhn[0][2]), float(box.xywhn[0][3])
                    all_boxes.append([x, y, w, h])
                    all_scores.append(conf)
                    all_cls_ids.append(cls_id)
                
                # Apply custom cross-class NMS
                if all_boxes:
                    keep_indices = cross_class_nms(all_boxes, all_scores, all_cls_ids, args.iou_threshold)
                    
                    # Write only the kept detections
                    with open(label_path, "w") as f:
                        for idx in keep_indices:
                            cls = all_cls_ids[idx]
                            x, y, w, h = all_boxes[idx]
                            conf = all_scores[idx]
                            f.write(f"{cls} {x} {y} {w} {h} {conf}\n")
                else:
                    # Create empty file if no detections
                    open(label_path, "w").close()

            pbar.update(len(batch))
            pbar.set_description(f"Processed {i + len(batch)}/{len(image_paths)} images")

    print("Inference completed.")

if __name__ == '__main__':
    generate_pseudo_ground_truth()