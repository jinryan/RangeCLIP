import os
import csv
import argparse

def generate_metadata(directory):
    depth_dir = os.path.join(directory, 'depth')
    images_dir = os.path.join(directory, 'images')
    output_csv = os.path.join(directory, 'metadata.csv')

    if not os.path.isdir(depth_dir) or not os.path.isdir(images_dir):
        raise ValueError("The directory must contain 'depth' and 'images' subdirectories.")

    # Get all PNG files from the depth directory
    depth_files = set(f for f in os.listdir(depth_dir) if f.endswith('.png'))
    image_files = set(f for f in os.listdir(images_dir) if f.endswith('.png'))

    # Only keep files that appear in both
    common_files = sorted(depth_files & image_files)

    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', 'depth_path', 'image_path'])

        for fname in common_files:
            depth_path = os.path.join('depth', fname)
            image_path = os.path.join('images', fname)
            writer.writerow([fname, depth_path, image_path])

    print(f"metadata.csv written to {output_csv} with {len(common_files)} entries.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate metadata.csv from depth and image directories.')
    parser.add_argument('directory', help='Path to the root directory containing depth and images folders')
    args = parser.parse_args()

    generate_metadata(args.directory)
