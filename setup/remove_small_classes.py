import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--patches_path', type=str, required=True, help='path to patches directory. This directory should have two sub-directories: images and depth')

def remove_small_classes(root_dir, threshold=80):
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            num_files = len([f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))])
            if num_files < threshold:
                print(f"Removing '{category}' ({category_path}) with {num_files} files...")
                shutil.rmtree(category_path)

# Example usage:
args = parser.parse_args()
root = args.patches_path
remove_small_classes(os.path.join(root, 'depth'), threshold=80)
remove_small_classes(os.path.join(root, 'images'), threshold=80)
