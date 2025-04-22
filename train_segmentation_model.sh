#!/bin/bash

# Paths
labeled_metadata_path="data/sunrgbd/SUNRGBD/metadata.csv"
labels_path="data/sunrgbd/SUNRGBD/candidate_labels.csv"
equivalence_dict_path="data/sunrgbd/SUNRGBD/label_similarity_sets.csv"

# Training & model config
batch_size=2
n_height=256
n_width=256

unet_architecture="resnet"
clip_model_name="openai/clip-vit-base-patch32"
learning_rates=(2e-4 1e-4 5e-5 1e-5)
learning_schedule=(15 25 35 40)
scheduler_type="multi_step"
w_weight_decay=1e-4

# Checkpoint & logging
checkpoint_path="checkpoints"
n_step_per_checkpoint=1000
n_step_per_summary=500
n_sample_per_summary=32
validation_start_step=5000
restore_path_model="checkpoints/checkpoints/depth_segmentation_model-25000.pth"       # Set to path if resuming from checkpoint
restore_path_encoder=""     # Set to path if restoring encoder separately

# System
device="gpu"
n_thread=8

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.6
torchrun --nproc_per_node=2 RangeCLIP/src/depth_segmentation_model/train.py \
    --labeled_metadata_path "$labeled_metadata_path" \
    --labels_path "$labels_path" \
    --equivalence_dict_path "$equivalence_dict_path" \
    --batch_size $batch_size \
    --n_height $n_height \
    --n_width $n_width \
    --unet_architecture "$unet_architecture" \
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
    --restore_path_model "$restore_path_model" \
    --restore_path_encoder "$restore_path_encoder" \
    --device "$device" \
    --n_thread $n_thread
