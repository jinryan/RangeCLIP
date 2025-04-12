#!/bin/bash

# Paths
labeled_metadata_path="training/labeled_metadata.csv"
unlabeled_metadata_path="training/unlabeled_metadata.csv"
labels_path="training/nyu_depth_v2/labelled_patches/labels.txt"

# Training & model config
batch_size=32
n_height=256
n_width=256
depth_encoder_type="resnet"
clip_model_name="openai/clip-vit-base-patch32"
learning_rates=(2e-4 1e-4 5e-5 1e-5)
learning_schedule=(10 20 30 35)
scheduler_type="multi_step"
w_weight_decay=1e-4

# Checkpoint & logging
checkpoint_path="checkpoints"
n_step_per_checkpoint=1000
n_step_per_summary=1000
n_sample_per_summary=4
validation_start_step=5000

# System
device="gpu"
n_thread=4

python RangeCLIP/src/train_model.py \
    --labeled_metadata_path "$labeled_metadata_path" \
    --unlabeled_metadata_path "$unlabeled_metadata_path" \
    --labels_path "$labels_path" \
    --batch_size $batch_size \
    --n_height $n_height \
    --n_width $n_width \
    --depth_encoder_type "$depth_encoder_type" \
    --clip_model_name "$clip_model_name" \
    --learning_rates "${learning_rates[@]}" \
    --learning_schedule "${learning_schedule[@]}" \
    --scheduler_type "$scheduler_type" \
    --w_weight_decay $w_weight_decay \
    --checkpoint_path "$checkpoint_path" \
    --n_step_per_checkpoint $n_step_per_checkpoint \
    --n_step_per_summary $n_step_per_summary \
    --n_sample_per_summary $n_sample_per_summary \
    --validation_start_step $validation_start_step \
    --device "$device" \
    --n_thread $n_thread
