from ultralytics import YOLO
import os, sys, argparse
from tqdm import tqdm
import math
sys.path.insert(0, os.getcwd())
import utils.src.data_utils as data_utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, required=False, help='path to model')
parser.add_argument('--image_paths', type=str, required=True, help='path to txt file that includes the paths to all the images')
parser.add_argument('--output_path', type=str, required=True, help='output for labels')
parser.add_argument('--start', type=int, required=False, help='start index')
parser.add_argument('--end', type=int, required=False, help='end index')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')

args = parser.parse_args()

def generate_pseudo_ground_truth():
    model = YOLO(args.model_path if args.model_path else "yolo11m.pt")

    image_paths = data_utils.read_paths(args.image_paths)
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(image_paths)
    image_paths = image_paths[start_idx:end_idx]

    batch_size = args.batch_size
    num_batches = math.ceil(len(image_paths) / batch_size)

    print(f"Processing {len(image_paths)} images in {num_batches} batches of {batch_size}...")

    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            
            model.predict(
                source=batch, 
                save_txt=True, 
                project=args.output_path, 
                name="detection_labels", 
                exist_ok=True,
                batch=batch_size, 
                verbose=False
            )

            pbar.update(len(batch))
            
            pbar.set_description(f"Processed {i + len(batch)}/{len(image_paths)} images")

    print("Inference completed.")

if __name__ == '__main__':
    generate_pseudo_ground_truth()