#!/bin/bash

dataset_path="training/nyu_depth_v2/labelled_patches"
metadata_path="$dataset_path/metadata.csv"
labels_path="$dataset_path/labels.txt"
batch_size=32
n_height=256
n_width=256
depth_encoder_type="resnet"
learning_rates=(1e-5 5e-6 2e-6 1e-6 5e-7)
learning_schedule=(10 20 30 40 50)
scheduler_type="cosine_annealing"
w_weight_decay=0
checkpoint_path="checkpoints_model"
n_step_per_checkpoint=2000
n_step_per_summary=5000
n_sample_per_summary=4
restore_path_model="checkpoints/checkpoints/resnet_depth_encoder-140000.pth"
validation_start_step=2000
device="gpu"
n_thread=4

python RangeCLIP/src/train_model.py \
    --dataset_path "$dataset_path" \
    --metadata_path "$metadata_path" \
    --labels_path "$labels_path" \
    --batch_size $batch_size \
    --n_height $n_height \
    --n_width $n_width \
    --depth_encoder_type "$depth_encoder_type" \
    --learning_rates "${learning_rates[@]}" \
    --learning_schedule "${learning_schedule[@]}" \
    --scheduler_type "$scheduler_type" \
    --w_weight_decay $w_weight_decay \
    --checkpoint_path "$checkpoint_path" \
    --n_step_per_checkpoint $n_step_per_checkpoint \
    --n_step_per_summary $n_step_per_summary \
    --n_sample_per_summary $n_sample_per_summary \
    --restore_path_model "$restore_path_model" \
    --validation_start_step $validation_start_step \
    --device "$device" \
    --n_thread $n_thread
