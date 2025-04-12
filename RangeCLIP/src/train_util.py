import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import sys
import os

from RangeCLIP.src.model import DepthClipModel
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from RangeCLIP.src.dataloader import setup_dataloaders
from utils.src.log_utils import log
from RangeCLIP.src.log import log_training_summary, log_configuration
from RangeCLIP.src.validate import validate_model, get_clip_baseline_performance

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def contains_nan(tensor):
    return torch.isnan(tensor).any().item()

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
    
    
    try:
        model = DepthClipModel(depth_encoder_type, clip_model_name, device)
    except Exception as e:
        print(f"Error loading DepthClipModel: {e}")
        import traceback
        traceback.print_exc()
        
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
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    if isinstance(model, torch.nn.DataParallel):
        print(f"DataParallel using {len(model.device_ids)} GPUs: {model.device_ids}")

    unwrapped_model = unwrap_model(model)
    get_clip_baseline_performance(_model=model,
                                  candidate_labels=candidate_labels,
                                  dataloader=val_dataloader,
                                  summary_writer=val_summary_writer,
                                  device=device,
                                  log_path=log_path)
    unwrapped_model.train()
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
    
    text_embeddings = unwrapped_model.get_text_encoder(candidate_labels)
    # Main training loop
    for epoch in range(1, n_epoch + 1):
        total_loss = 0
        for batch in tqdm.tqdm(train_dataloader, desc=f'{epoch}/{n_epoch}'):
            train_step += 1
            
            try:
                labeled_batch = batch['labeled']
                unlabeled_batch = batch['unlabeled']
                if 'image' in labeled_batch and labeled_batch['image'].size(0) > 1:
                    labeled_images = labeled_batch['image'].to(device)          # shape: [B_l, C, H, W]
                    labeled_depths = labeled_batch['depth'].to(device)          # shape: [B_l, 1, H, W]
                    labeled_ground_truth_indices = labeled_batch['id'].to(device)

                    # Compute embeddings
                    depth_embed_l, _, image_embed_l = model.forward(
                        depth_map=labeled_depths,
                        image=labeled_images
                    )

                    labeled_loss, loss_info_labeled = unwrapped_model.compute_trimodal_loss(
                        embedding_1=depth_embed_l,
                        embedding_2=image_embed_l,
                        embedding_3=text_embeddings,
                        ground_truth_indices=labeled_ground_truth_indices,
                        w_1_2=0.2
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

                    unsup_loss, loss_info_unsup = unwrapped_model.compute_bimodal_loss(
                        embedding_1=depth_embed_u,
                        embedding_2=image_embed_u
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
                loss_info['loss'] = loss.item()
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
                    unwrapped_model.eval()
                    best_results = validate_model(
                        model, candidate_labels, val_dataloader,
                        train_step, best_results, device,
                        val_summary_writer, n_sample_per_summary, log_path
                    )
                    unwrapped_model.train()
            
            # Save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                save_checkpoint_and_log_progress(
                    model, model_checkpoint_path, train_step,
                    n_train_step, start_step, loss.item(),
                    time_start, log_path, optimizer
                )
        
        log(f"Total loss: {total_loss}", log_path)
        
        # Update learning rate scheduler
        update_scheduler(scheduler, scheduler_type, unwrapped_model, candidate_labels,
                         val_dataloader, train_step,
                         best_results, device, val_summary_writer,
                         n_sample_per_summary, log_path)
    
    # Save final model
    save_checkpoint_and_log_progress(
        model, model_checkpoint_path, train_step,
        n_train_step, train_step - n_train_step, loss.item(),
        time_start, log_path, optimizer
    )


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


def update_scheduler(scheduler, scheduler_type, unwrapped_model, candidate_labels,
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
        unwrapped_model.eval()
        _, val_imce_loss = validate_model(
            model, candidate_labels, val_dataloader,
            train_step, best_results, device,
            val_summary_writer, n_sample_per_summary, log_path
        )
        unwrapped_model.train()
        
        scheduler.step(val_imce_loss)
    else:
        scheduler.step()
        
    
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

    unwrapped_model.save_model(
        model_checkpoint_path.format(train_step), train_step, optimizer
    )


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

