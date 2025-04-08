import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import tqdm
import sys
import os
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import datasets
from utils.src.log_utils import log
from RangeCLIP.src.log import log_input_settings, log_loss_func_settings, log_network_settings, log_system_settings, log_training_settings
from RangeCLIP.src.model import DepthClipModel

torch.cuda.empty_cache()


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--labeled_metadata_path',
    type=str, required=True, help='Path to labeled dataset metadata.csv')
parser.add_argument('--unlabeled_metadata_path',
    type=str, required=True, help='Path to unlabeled dataset metadata.csv')
parser.add_argument('--labels_path',
    type=str, required=True, help='Path to dataset labels')

# Batch parameters
parser.add_argument('--batch_size',
    type=int, default=16, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=256, help='Height of of sample for resizing')
parser.add_argument('--n_width',
    type=int, default=256, help='Width of each sample for resizing')

# Network settings
parser.add_argument('--depth_encoder_type',
    type=str, required=True, help='Available: resnet')
parser.add_argument('--clip_model_name',
    type=str, default='openai/clip-vit-base-patch32', help='Available: openai/clip-vit-base-patch32')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[2e-4, 1e-4, 5e-5, 1e-5], help='Space delimited list of learning rates')
parser.add_argument('--scheduler_type',
    type=str, default='multi_step', help='Options: multi_step, cosine_annealing, reduce_on_plateau')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[10, 20, 30, 35], help='Space delimited list to change learning rate')

# Loss settings
parser.add_argument('--w_weight_decay',
    type=float, default=0.0, help='Weight of weight decay loss')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=5000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=1000, help='Number of iterations before logging summary')
parser.add_argument('--n_sample_per_summary',
    type=int, default=4, help='Number of samples to include in visual display summary')
parser.add_argument('--validation_start_step',
    type=int, default=5000, help='Number of steps before starting validation')
parser.add_argument('--restore_path_model',
    type=str, default=None, help='Paths to restore depth model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def contains_nan(tensor):
    return torch.isnan(tensor).any().item()

def train_depth_clip_model(
        labeled_metadata_path,
        unlabeled_metadata_path,
        labels_path,
        batch_size,
        n_height,
        n_width,
        depth_encoder_type,
        learning_rates,
        learning_schedule,
        scheduler_type,
        w_weight_decay,
        checkpoint_path,
        n_step_per_checkpoint,
        n_step_per_summary,
        n_sample_per_summary,
        validation_start_step,
        restore_path_model,
        clip_model_name,
        device='cuda',
        n_thread=8):
    """
    Train a depth encoder model to learn depth representations that align with image embeddings.
    
    Args:
        train_image_path (str): Path to training images
        train_depth_path (str): Path to training depth maps
        batch_size (int): Number of samples per batch
        n_height (int): Target height for resizing images
        n_width (int): Target width for resizing images
        depth_encoder_type (str): Type of encoder architecture ('resnet' supported)
        augmentations (dict): Data augmentation settings
        learning_rates (list): Learning rates to use during training
        learning_schedule (list): Schedule for learning rate adjustments
        scheduler_type (str): Type of learning rate scheduler
        w_weight_decay (float): Weight decay coefficient for optimizer
        checkpoint_path (str): Path to save model checkpoints
        n_step_per_checkpoint (int): Steps between saving checkpoints
        n_step_per_summary (int): Steps between logging summaries
        n_sample_per_summary (int): Number of samples to visualize in summaries
        validation_start_step (int): Step to begin validation
        restore_path_model (str): Path to restore model from (None for new training)
        clip_model_name (str): Name of the CLIP model to use
        device (str): Device to use ('cuda' or 'cpu')
        n_thread (int): Number of data loading threads
        
    Returns:
        None: Model checkpoints and logs are saved to disk
    """
    # Set up device
    device = setup_device(device)
    
    # Set up directories for checkpoints and logs
    model_checkpoint_path, log_path, event_path = setup_checkpoint_and_event_paths(
        checkpoint_path, 'depth_clip_model')
    
    # Initialize tracking for best validation results
    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imce': np.infty,
    }
    
    # Calculate total number of epochs
    n_epoch = learning_schedule[-1]
    
    # Prepare data loaders
    resize_shape = (n_height, n_width)
    train_dataloader, val_dataloader, n_train_step, candidate_labels = setup_dataloaders(
        labeled_metadata_file=labeled_metadata_path,
        unlabeled_metadata_file=unlabeled_metadata_path,
        labels_path=labels_path,
        resize_shape=resize_shape, 
        batch_size=batch_size,
        n_thread=n_thread,
        n_epoch=n_epoch
    )
    
    model = DepthClipModel(depth_encoder_type, clip_model_name, device)
    
    # Set up optimizer and learning rate scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model.parameters(), learning_rates, w_weight_decay, 
        scheduler_type, learning_schedule
    )
    
    # Restore model if specified
    train_step, optimizer = restore_model_if_needed(model, restore_path_model, optimizer)
    start_step = train_step


    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rates[0]
    
    # Configure model for training
    model.train()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Log configuration settings
    log_configuration(
        log_path, labeled_metadata_path, unlabeled_metadata_path, 
        batch_size, n_height, n_width,
        depth_encoder_type, model.parameters(),
        batch_size, len(train_dataloader.dataset), n_train_step,
        learning_rates, learning_schedule, scheduler_type,
        w_weight_decay,
        checkpoint_path, n_step_per_checkpoint, event_path,
        n_step_per_summary, n_sample_per_summary,
        validation_start_step, restore_path_model,
        device, n_thread
    )
    
    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')
    
    # Begin training
    time_start = time.time()
    log('Begin training...', log_path)
    
    # Main training loop
    for epoch in range(1, n_epoch + 1):
        total_loss = 0
        for batch in tqdm.tqdm(train_dataloader, desc=f'{epoch}/{n_epoch}'):
            train_step += 1
            
            try:
                labeled_batch = batch['labeled']
                unlabeled_batch = batch['unlabeled']
                if labeled_batch.get('image') is not None and len(labeled_batch['image']) > 0:  # Check if there are any labeled samples
                    labeled_images = labeled_batch['image'].to(device)          # shape: [B_l, C, H, W]
                    labeled_depths = labeled_batch['depth'].to(device)          # shape: [B_l, 1, H, W]
                    labeled_ground_truth_indices = labeled_batch['id']                       # list of strings

                    # Compute embeddings
                    depth_embed_l, _, image_embed_l = model.forward(
                        depth_map=labeled_depths,
                        image=labeled_images
                    )

                    labeled_loss, loss_info_labeled = unwrap_model(model).compute_trimodal_loss(
                        depth_embeddings=depth_embed_l,
                        image_embeddings=image_embed_l,
                        candidate_labels=candidate_labels,
                        ground_truth_indices=labeled_ground_truth_indices,
                        w_text=0.8
                    )
                else:
                    labeled_loss = 0
                    loss_info_labeled = {}
                    
                if unlabeled_batch.get('image') is not None and len(unlabeled_batch['image']) > 0:  # Check if there are any unlabeled samples
                    unlabeled_images = unlabeled_batch['image'].to(device)        # shape: [B_u, C, H, W]
                    unlabeled_depths = unlabeled_batch['depth'].to(device)        # shape: [B_u, N, 1, H, W]


                    # Repeat RGB image for each patch

                    depth_embed_u, _, image_embed_u = model.forward(
                        depth_map=unlabeled_depths,
                        image=unlabeled_images
                    )

                    unsup_loss, loss_info_unsup = unwrap_model(model).compute_bimodal_loss(
                        depth_embeddings=depth_embed_u,
                        other_embeddings=image_embed_u
                    )
                else:
                    unsup_loss = 0
                    loss_info_unsup = {}
                    
                # Combine losses
                if labeled_loss != 0 and unsup_loss != 0:
                    loss = labeled_loss + unsup_loss
                else:
                    loss = labeled_loss or unsup_loss  # whichever is nonzero
                
                loss_info = {**loss_info_labeled, **loss_info_unsup}

                loss.backward()
                # Gradient clipping
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
            
            except Exception as e:
                print(f"Runtime error {e}")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                print(f"Error details: {repr(e)}")
                import traceback
                traceback.print_exc()
                if "CUBLAS_STATUS" in str(e):
                    print(f"CUDA error encountered: {e}")
                    print("Attempting to recover...")
                    torch.cuda.empty_cache()
                    # Skip this batch
                    continue
            
            # Logging and validation
            if (train_step % n_step_per_summary) == 0:
                log_training_summary(
                    train_summary_writer,
                    train_step,
                    labeled_images,
                    labeled_depths,
                    labeled_ground_truth_indices,
                    unlabeled_images,
                    unlabeled_depths,
                    loss_info,
                    n_sample_per_summary
                )
                
                # Validate model
                with torch.no_grad():
                    unwrap_model(model).eval()
                    best_results, _ = validate_model(
                        model, candidate_labels, val_dataloader,
                        train_step, best_results, device,
                        val_summary_writer, n_sample_per_summary, log_path
                    )
                    unwrap_model(model).train()
            
            # Save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                save_checkpoint_and_log_progress(
                    model, model_checkpoint_path, train_step,
                    n_train_step, start_step, loss.item(),
                    time_start, log_path, optimizer
                )
        
        log(f"Total loss: {total_loss}", log_path)
        
        # Update learning rate scheduler
        update_scheduler(scheduler, scheduler_type, model, candidate_labels,
                         val_dataloader, train_step,
                         best_results, device, val_summary_writer,
                         n_sample_per_summary, log_path)
    
    # Save final model
    save_checkpoint_and_log_progress(
        model, model_checkpoint_path, train_step,
        n_train_step, train_step - n_train_step, loss.item(),
        time_start, log_path, optimizer
    )


def setup_device(device):
    """
    Set up and validate the computing device.
    
    Args:
        device (str): The requested device ('cuda', 'gpu', or 'cpu')
        
    Returns:
        torch.device: The configured device
    """
    device = device.lower()
    device = 'cuda' if device in ['gpu', 'cuda'] and torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def setup_optimizer_and_scheduler(parameters, learning_rates, w_weight_decay, 
                                 scheduler_type, learning_schedule):
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        parameters: Model parameters to optimize
        learning_rates (list): Learning rates
        w_weight_decay (float): Weight decay coefficient
        scheduler_type (str): Type of learning rate scheduler
        learning_schedule (list): Schedule for learning rate adjustments
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Initialize optimizer with first learning rate
    optimizer = optim.Adam(
        parameters,
        lr=learning_rates[0],
        weight_decay=w_weight_decay
    )
    
    # Initialize learning rate scheduler based on type
    if scheduler_type == 'multi_step':
        # Reduces learning rate at predefined milestones
        scheduler = MultiStepLR(
            optimizer, 
            milestones=learning_schedule, 
            gamma=0.1  # reduces learning rate by 10x at each milestone
        )
    
    elif scheduler_type == 'cosine_annealing':
        # Cosine annealing learning rate, good for converging to a minimum
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=learning_schedule[-1],  # total number of epochs
            eta_min=learning_rates[-1]  # minimum learning rate
        )
    
    elif scheduler_type == 'reduce_on_plateau':
        # Reduces learning rate when a metric has stopped improving
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',  # look for minimum of the metric (e.g., loss)
            factor=0.1,  # reduce learning rate by 10x
            patience=5,  # number of epochs with no improvement
            min_lr=learning_rates[-1]
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
    return optimizer, scheduler


def restore_model_if_needed(model, restore_path_model, optimizer):
    """
    Restore model from checkpoint if path is provided.
    
    Args:
        model: The depth encoder model
        restore_path_model (str): Path to restore model from
        optimizer: The optimizer
        learning_rates (list): Learning rates
        
    Returns:
        int: Starting step (0 for new training, restored step for continuing)
    """
    if restore_path_model is not None:
        start_step, optimizer = model.restore_model(
            restore_path_model, optimizer=optimizer)
    else:
        start_step = 0
        
    return start_step, optimizer


def log_configuration(log_path, labeled_metadata_path, unlabeled_metadata_path, batch_size, n_height, n_width,
                     model_architecture, parameters_model, n_batch, n_train_sample, n_train_step,
                     learning_rates, learning_schedule, scheduler_type, w_weight_decay,
                     checkpoint_path, n_step_per_checkpoint, summary_event_path,
                     n_step_per_summary, n_sample_per_summary,
                     validation_start_step, restore_path_model, device, n_thread):
    """
    Log all configuration settings for the training run.
    
    Args:
        Various configuration parameters
    """
    # Log input paths
    log('Training input paths:', log_path)
    log(labeled_metadata_path, log_path)
    log(unlabeled_metadata_path, log_path)

    # Log batch settings
    log_input_settings(
        log_path,
        n_batch=batch_size,
        n_height=n_height,
        n_width=n_width
    )

    # Log network settings
    log_network_settings(
        log_path,
        model_architecture=model_architecture,
        parameters_model=parameters_model
    )

    # Log training settings
    log_training_settings(
        log_path,
        n_batch=n_batch,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        scheduler_type=scheduler_type
    )

    # Log loss function settings
    log_loss_func_settings(
        log_path,
        w_weight_decay=w_weight_decay
    )

    # Log system settings
    log_system_settings(
        log_path,
        checkpoint_path=checkpoint_path,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=summary_event_path,
        n_step_per_summary=n_step_per_summary,
        n_sample_per_summary=n_sample_per_summary,
        validation_start_step=validation_start_step,
        restore_path_model=restore_path_model,
        device=device,
        n_thread=n_thread
    )


import torch
import torchvision
from torchvision.utils import make_grid

def apply_colormap(depth_tensor):
    """Apply colormap to a batch of depth maps for visualization."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Normalize to 0–1
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min() + 1e-8)
    depth_tensor = depth_tensor.squeeze(1)  # [B, H, W]

    # Apply colormap per sample
    depth_colored = []
    for i in range(depth_tensor.size(0)):
        depth_np = depth_tensor[i].cpu().numpy()
        depth_cm = plt.get_cmap('plasma')(depth_np)[:, :, :3]  # Drop alpha
        depth_cm_tensor = torch.tensor(depth_cm).permute(2, 0, 1).float()  # [3, H, W]
        depth_colored.append(depth_cm_tensor)

    return torch.stack(depth_colored)  # [B, 3, H, W]

def log_training_summary(
        train_summary_writer,
        train_step,
        labeled_images,
        labeled_depths,
        labeled_ground_truth_indices,
        unlabeled_images,
        unlabeled_depths,
        loss_info,
        n_sample_per_summary
    ):
    """
    Log training summary to TensorBoard.
    """
    with torch.no_grad():
        display_images = []

        # Labeled image
        if labeled_images is not None and len(labeled_images) > 0:
            labeled_images_vis = labeled_images[:n_sample_per_summary].cpu()
            display_images.append(labeled_images_vis)

        # Labeled depth
        if labeled_depths is not None and len(labeled_depths) > 0:
            labeled_depths_vis = labeled_depths[:n_sample_per_summary]
            labeled_depths_colored = apply_colormap(labeled_depths_vis)
            display_images.append(labeled_depths_colored)

        # Unlabeled depth
        if unlabeled_depths is not None and len(unlabeled_depths) > 0:
            # Select 1 patch per example (e.g., first patch)
            B, N, _, H, W = unlabeled_depths.shape
            patches = unlabeled_depths[:, 0]  # [B, 1, H, W]
            patches = patches[:n_sample_per_summary]
            patches_colored = apply_colormap(patches)
            display_images.append(patches_colored)

        # Unlabeled image (optional, but good to track RGB reference)
        if unlabeled_images is not None and len(unlabeled_images) > 0:
            unlabeled_images_vis = unlabeled_images[:n_sample_per_summary].cpu()
            display_images.append(unlabeled_images_vis)

        # Log concatenated image strip
        if display_images:
            grid = make_grid(torch.cat(display_images, dim=2), nrow=n_sample_per_summary)  # concat width-wise
            train_summary_writer.add_image('train/visual_summary', grid, global_step=train_step)

        # Log scalar loss values
        for name, val in loss_info.items():
            train_summary_writer.add_scalar(f'train/loss_{name}', val, global_step=train_step)

        # Log text labels
        if labeled_ground_truth_indices is not None and len(labeled_ground_truth_indices) > 0:
            text_summary = "\n".join([f"{i+1}: {t}" for i, t in enumerate(labeled_ground_truth_indices[:n_sample_per_summary])])
            train_summary_writer.add_text('train/labels', text_summary, global_step=train_step)


def save_checkpoint_and_log_progress(model, model_checkpoint_path, train_step,
                                    n_train_step, start_step, loss_value,
                                    time_start, log_path, optimizer):
    """
    Save model checkpoint and log training progress.
    
    Args:
        model: The model
        model_checkpoint_path (str): Path to save checkpoint
        train_step (int): Current training step
        n_train_step (int): Total number of training steps
        start_step (int): Starting step
        loss_value (float): Current loss value
        time_start (float): Training start time
        log_path (str): Path to log file
        optimizer: The optimizer
    """
    time_elapse = (time.time() - time_start) / 3600
    if train_step > start_step:
        time_remain = (n_train_step - train_step + start_step) * time_elapse / (train_step - start_step)
    else:
        time_remain = 0

    log(
        'Step={:6}/{}  Loss={:.7f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
            train_step, n_train_step + start_step, loss_value, time_elapse, time_remain
        ),
        log_path
    )

    unwrap_model(model).save_model(
        model_checkpoint_path.format(train_step), train_step, optimizer
    )


def update_scheduler(scheduler, scheduler_type, model, candidate_labels,
                    val_dataloader, train_step, best_results, device,
                    val_summary_writer, n_sample_per_summary, log_path):
    """
    Update the learning rate scheduler.
    
    Args:
        scheduler: The learning rate scheduler
        scheduler_type (str): Type of scheduler
        model: The depth encoder model
        image_encoder: The image encoder model
        val_dataloader: Validation data loader
        train_step (int): Current training step
        best_results (dict): Best validation results
        device (torch.device): The device
        val_summary_writer: Validation summary writer
        n_sample_per_summary (int): Number of samples to visualize
        log_path (str): Path to log file
    """
    if scheduler_type == 'reduce_on_plateau':
        # For ReduceLROnPlateau, pass the validation loss
        unwrap_model(model).eval()
        _, val_imce_loss = validate_model(
            model, candidate_labels, val_dataloader,
            train_step, best_results, device,
            val_summary_writer, n_sample_per_summary, log_path
        )
        unwrap_model(model).train()
        
        scheduler.step(val_imce_loss)
    else:
        scheduler.step()

def validate_model(model,
                  candidate_labels,
                  dataloader,
                  step,
                  best_results,
                  device,
                  summary_writer,
                  n_sample_per_summary=4,
                  log_path=None):
    """
    Validate the model on a dataset.
    
    Args:
        model: The DepthClipModel to validate
        candidate_labels: List of text labels for classification
        dataloader: DataLoader for validation data
        step: Current training step
        best_results: Dictionary containing best validation results so far
        device: Device to run validation on
        summary_writer: TensorBoard summary writer
        n_sample_per_summary: Number of samples to log to TensorBoard
        log_path: Path to log results to
        
    Returns:
        best_results: Updated best_results dictionary
        accuracy: Classification accuracy on validation set
    """
    model.eval()  # Set model to evaluation mode
    
    
    # Metrics for embedding quality
    total_loss = 0.0
    
    image_summary = []
    depth_summary = []
    text_summary = []
    
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for idx, batch in enumerate(dataloader):
            # Unpack batch to device
            labeled_batch = batch['labeled']
            unlabeled_batch = batch['unlabeled']
            
            if not labeled_batch.get('image') and not unlabeled_batch.get('image'):
                print(f"[Validation] Warning: Empty batch at index {idx}. Skipping.")
                continue  # Skip this iteration
            
            if labeled_batch.get('image') is not None and len(labeled_batch['image']) > 0:  # Check if there are any labeled samples
                labeled_images = labeled_batch['image'].to(device)          # shape: [B_l, C, H, W]
                labeled_depths = labeled_batch['depth'].to(device)          # shape: [B_l, 1, H, W]
                labeled_ground_truth_indices = labeled_batch['id']                       # list of strings
                
                for i in range(len(labeled_depths)):
                    single_depth = labeled_depths[i:i+1]  # Shape: [1, 1, H, W]
                    predicted_class, confidence = model.predict(single_depth.to(device), candidate_labels)

                    true_class = labeled_ground_truth_indices[i].item()

                    if predicted_class == true_class:
                        correct_predictions += 1

                    total_predictions += 1
                    
                # Compute embeddings
                depth_embed_l, _, image_embed_l = model.forward(
                    depth_map=labeled_depths,
                    image=labeled_images
                )

                labeled_loss, loss_info_labeled = unwrap_model(model).compute_trimodal_loss(
                    depth_embeddings=depth_embed_l,
                    image_embeddings=image_embed_l,
                    candidate_labels=candidate_labels,
                    ground_truth_indices=labeled_ground_truth_indices,
                    w_text=0.8
                )
            else:
                labeled_loss = 0
                loss_info_labeled = {}
                print(f"[Validation] No labeled data in batch at index {idx}. Skipping labeled loss.")

                
            if unlabeled_batch.get('image') is not None and len(unlabeled_batch['image']) > 0:  # Check if there are any unlabeled samples
                unlabeled_images = unlabeled_batch['image'].to(device)        # shape: [B_u, C, H, W]
                unlabeled_depths = unlabeled_batch['depth'].to(device)        # shape: [B_u, N, 1, H, W]


                # Repeat RGB image for each patch

                depth_embed_u, _, image_embed_u = model.forward(
                    depth_map=unlabeled_depths,
                    image=unlabeled_images
                )

                unsup_loss, loss_info_unsup = unwrap_model(model).compute_bimodal_loss(
                    depth_embeddings=depth_embed_u,
                    other_embeddings=image_embed_u
                )
            else:
                unsup_loss = 0
                loss_info_unsup = {}
                
            # Combine losses
            if labeled_loss != 0 and unsup_loss != 0:
                loss = labeled_loss + unsup_loss
            else:
                loss = labeled_loss or unsup_loss  # whichever is nonzero
            
            loss_info = {**loss_info_labeled, **loss_info_unsup}
            total_loss += loss.item()

            
            # Collect samples for TensorBoard display
            if (idx % max(1, len(dataloader) // n_sample_per_summary)) == 0 and summary_writer is not None:
                n = min(len(labeled_images), n_sample_per_summary)

                # Collect image and depth examples
                image_summary.append(labeled_images[:n].cpu())
                depth_summary.append(labeled_depths[:n].cpu())

                batch_texts = [candidate_labels[i] for i in labeled_ground_truth_indices[:n]]

                text_summary.extend(batch_texts)
    
    # Compute average metrics
    avg_loss = total_loss / len(dataloader)
    val_accuracy = correct_predictions / (total_predictions + 1e-8)
    print(f"[Validation] Accuracy: {val_accuracy:.4f}")
    summary_writer.add_scalar("val/accuracy", val_accuracy, global_step=step)
    
    # Log to TensorBoard
    # Concatenate for logging
    if image_summary and depth_summary:
        summary_images = torch.cat(image_summary, dim=0)
        summary_depths = torch.cat(depth_summary, dim=0)
        summary_texts = text_summary

        log_training_summary(
            model=model,
            train_summary_writer=summary_writer,
            train_step=step,
            labeled_images=summary_images,
            labeled_depths=summary_depths,
            labeled_ground_truth_indices=summary_texts,
            unlabeled_images=None,
            unlabeled_depths=None,
            loss_info=loss_info,
            n_sample_per_summary=n_sample_per_summary
        )
        
    # Print validation results to console
    if log_path:
        log('Validation results:', log_path)
        log('{:>8}  {:>8}  {:>10}'.format(
            'Step', 'Loss', 'Accuracy'),
            log_path)
        log('{:8}  {:8.4f}  {:10.4f}'.format(
            step, avg_loss, val_accuracy),
            log_path)
    
    # Check if model improved
    improved = False
    
    # For initial run when best_results might be empty
    if 'accuracy' not in best_results or best_results['accuracy'] < val_accuracy:
        improved = True
    
    # Update best results if improved
    if improved:
        best_results['step'] = step
        best_results['loss'] = avg_loss
        best_results['accuracy'] = val_accuracy
    
    if log_path:
        log('Best results:', log_path)
        log('{:>8}  {:>8}  {:>10}'.format(
            'Step', 'Loss', 'Accuracy'),
            log_path)
        log('{:8}  {:8.4f}  {:10.4f}'.format(
            best_results['step'],
            best_results['loss'],
            best_results['accuracy']), log_path)

    
    return best_results, val_accuracy


    
    
def setup_dataloaders(labeled_metadata_file,
                      unlabeled_metadata_file,
                      labels_path,
                      resize_shape,
                      batch_size,
                      n_thread,
                      n_epoch):
    
    image_transform = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    def depth_transform_fn(depth_tensor):
        # Resize depth map
        resized = transforms.functional.resize(depth_tensor, resize_shape)
        
        min_val = resized.min()
        max_val = resized.max()
        
        # If depth is uniform, return zero tensor or normalized to 0.5 (optional)
        if (max_val - min_val).abs() < 1e-6:
            return torch.zeros_like(resized)  # or torch.full_like(resized, 0.5)
        
        normalized = (resized - min_val) / (max_val - min_val)
        return normalized

    depth_transform = depth_transform_fn
        
    labeled_dataset = datasets.ImageDepthTextDataset(metadata_file=labeled_metadata_file,
                                                  labels_path=labels_path,
                                                  image_transform=image_transform,
                                                  depth_transform=depth_transform)
    
    unlabeled_dataset = datasets.ImageDepthDataset(metadata_file=unlabeled_metadata_file,
                                                   image_transform=image_transform,
                                                   depth_transform=depth_transform)
    labels = labeled_dataset.get_labels()
    
    labeled_dataset = datasets.TaggedDataset(labeled_dataset, tag='labeled')
    unlabeled_dataset = datasets.TaggedDataset(unlabeled_dataset, tag='unlabeled')
    
    generator = torch.Generator().manual_seed(42)
    l_train_dataset, l_val_dataset = torch.utils.data.random_split(labeled_dataset, [0.8, 0.2], generator=generator)
    u_train_dataset, u_val_dataset = torch.utils.data.random_split(unlabeled_dataset, [0.8, 0.2], generator=generator)
    
    
    def custom_collate(batch):
        labeled = {'image': [], 'depth': [], 'id': []}
        unlabeled = {'image': [], 'depth': []}

        for sample in batch:
            tag = sample.pop('__tag__')

            if tag == 'labeled':
                labeled['image'].append(sample['image'])
                labeled['depth'].append(sample['depth'])
                labeled['id'].append(sample['id'])

            elif tag == 'unlabeled':
                unlabeled['image'].append(sample['image'])
                unlabeled['depth'].append(sample['depth'])

        # Convert lists to tensors where applicable
        if labeled['image']:
            labeled['image'] = torch.stack(labeled['image'])
            labeled['depth'] = torch.stack(labeled['depth'])
            # labeled['text'] stays as list of strings

        if unlabeled['image']:
            unlabeled['image'] = torch.stack(unlabeled['image'])
            unlabeled['depth'] = torch.stack(unlabeled['depth'])  # shape: [B, N, ...]
        
        return {'labeled': labeled, 'unlabeled': unlabeled}

    train_dataset = torch.utils.data.ConcatDataset([l_train_dataset, u_train_dataset])
    val_dataset = torch.utils.data.ConcatDataset([l_val_dataset, u_val_dataset])
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=n_thread
    )
    
    # More accurate calculation of train steps
    n_train_samples = len(train_dataset)
    n_train_steps = ((n_train_samples + batch_size - 1) // batch_size) * n_epoch
    
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                collate_fn=custom_collate,
                                shuffle=False,
                                num_workers=n_thread)
    
    
    
    return train_loader, val_loader, n_train_steps, labels
    
    
    
def setup_checkpoint_and_event_paths(checkpoint_path, model_name):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_checkpoint_path = os.path.join(
        checkpoint_path,
        'checkpoints',
        f'{model_name}' + '-{}.pth'
    )
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'tensorboard')

    os.makedirs(event_path, exist_ok=True)
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    
    return model_checkpoint_path, log_path, event_path


def set_global_seed(seed=42):
    """Set global seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



'''
Helper functions for logging
'''



if __name__ == '__main__':

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)
    set_global_seed()

    train_depth_clip_model(
          labeled_metadata_path=args.labeled_metadata_path,
          unlabeled_metadata_path=args.unlabeled_metadata_path,
          labels_path=args.labels_path,
          # Batch settings
          batch_size=args.batch_size,
          n_height=args.n_height,
          n_width=args.n_width,
          # Network settings
          depth_encoder_type=args.depth_encoder_type,
          clip_model_name=args.clip_model_name,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          scheduler_type=args.scheduler_type,
          # Loss function settings
          w_weight_decay=args.w_weight_decay,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_sample_per_summary=args.n_sample_per_summary,
          validation_start_step=args.validation_start_step,
          restore_path_model=args.restore_path_model,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)