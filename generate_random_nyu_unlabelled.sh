#!/bin/bash

h5_directory="/media/common/datasets/nyu_depth_v2/train"
output_directory="/media/home/ryjin/depthclip/RangeCLIP/training/nyu_depth_v2/unlabelled_patches"
num_crops=4
num_workers=8

python setup/nyu_depth_v2/generate_random_cropped_patches.py \
--h5_directory "$h5_directory" \
--output_path  "$output_directory" \
--num_crops $num_crops \
--num_workers $num_workers

python setup/nyu_depth_v2/generate_csv_paths.py \
--directory "$output_directory"