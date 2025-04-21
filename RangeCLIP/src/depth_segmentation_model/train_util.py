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
import traceback
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from utils.src.networks import TextEncoder
from transformers import CLIPModel
import matplotlib.pyplot as plt
from matplotlib import cm


from RangeCLIP.src.depth_segmentation_model.model import DepthUNet
from RangeCLIP.src.depth_segmentation_model.dataloader import setup_dataloaders, load_equivalence_dict, build_equivalence_tensor, load_label_similarity_sets
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.depth_segmentation_model.log import log_training_summary, log_configuration, visualize_batch_predictions
from RangeCLIP.src.depth_segmentation_model.validate import validate_model


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

def get_curriculum_schedule(epoch: int, total_epochs: int) -> dict:
    pct = epoch / total_epochs
    pct_medium = max(0.0, 1.0 - 4.0 * pct)   # from 1.0 → 0.0 over 25% of training
    pct_hard = min(0.8, pct * 1.2)           # from 0.0 → 0.8 linearly
    pct_rand = 1.0 - pct_medium - pct_hard

    return {
        "pct_medium": round(pct_medium, 4),
        "pct_hard": round(pct_hard, 4),
        "pct_rand": round(pct_rand, 4),
    }




def train_depth_clip_model(
        labeled_metadata_path,
        labels_path,
        equivalence_dict_path,
        batch_size,
        n_height,
        n_width,
        unet_architecture,
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
        restore_path_encoder,
        clip_model_name,
        device='cuda',
        n_thread=8,
        accumulation_steps = 8,
        local_rank=0):
    
    scaler = GradScaler()

    # Set up directories for checkpoints and logs
    model_checkpoint_path, log_path, event_path = setup_checkpoint_and_event_paths(
        checkpoint_path, 'depth_segmentation_model')
    
    # Initialize tracking for best validation results
    best_results = {
        'step': -1,
        'loss': np.infty,
    }
    
    # Calculate total number of epochs
    n_epoch = learning_schedule[-1]
    
    # Prepare data loaders
    resize_shape = (n_height, n_width)
    train_dataloader, val_dataloader, test_dataloader, train_sampler, n_train_step, candidate_labels = setup_dataloaders(
        metadata_file=labeled_metadata_path,
        labels_file=labels_path,
        resize_shape=resize_shape, 
        batch_size=batch_size,
        n_thread=n_thread,
        n_epoch=n_epoch
    )
    
    equivalence_dict = load_equivalence_dict(equivalence_dict_path)
    equivalence_tensor = build_equivalence_tensor(equivalence_dict, num_classes=len(candidate_labels)).to(device)
    equivalence_tensor = equivalence_tensor.to(device)
    
    similarity_sets = load_label_similarity_sets(equivalence_dict_path, len(candidate_labels))
    

    
    try:
        
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        embedding_dim = clip_model.config.projection_dim
        text_encoder = TextEncoder(
            clip_model=clip_model,
            device=device
        )
        
        model = DepthUNet(
            unet_type=unet_architecture,
            device=device,
            n_layer=18,
            input_channels=1,
            encoder_filters=[32, 64, 128, 256, 512],
            embedding_dim=embedding_dim,
            weight_initializer='kaiming_uniform',
            activation_func='relu',
            use_batch_norm=True,
            use_instance_norm=False
        )
        
        
    except Exception as e:
        print(f"Error loading DepthUnet: {e}")
        traceback.print_exc()
    
    

    optimizer, scheduler = setup_optimizer_and_scheduler(
        model.parameters(), learning_rates, w_weight_decay, 
        scheduler_type, learning_schedule
    )
    
    # Restore model if specified
    if restore_path_encoder is not None and restore_path_encoder != '':
        model.restore_depth_encoder(restore_path_encoder, freeze_encoder=True)
        train_step = 0
    else:
        train_step, optimizer = restore_model_if_needed(model, restore_path_model, optimizer)
        
    start_step = train_step
    
    if start_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[0]
    
    # Configure model for training
    
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)


    # Log configuration settings
    log_configuration(
        log_path, labeled_metadata_path, 
        batch_size, n_height, n_width,
        unet_architecture, model.parameters(),
        len(train_dataloader.dataset), n_train_step,
        learning_rates, learning_schedule, scheduler_type,
        w_weight_decay,
        checkpoint_path, n_step_per_checkpoint, event_path,
        n_step_per_summary, n_sample_per_summary,
        validation_start_step, restore_path_model,
        device, n_thread
    )
    
    
    
    # Set up tensorboard summary writers
    if local_rank == 0:
        train_summary_writer = SummaryWriter(event_path + '-train')
        val_summary_writer = SummaryWriter(event_path + '-val')
    else:
        train_summary_writer = val_summary_writer = None

    with torch.no_grad():
        candidate_text_embeddings = text_encoder(candidate_labels)

    if local_rank == 0:
        model.eval()
        best_results = validate_model(model=model,
                                    candidate_text_embeddings=candidate_text_embeddings,
                                    candidate_labels=candidate_labels,
                                    equivalence_tensor=equivalence_tensor,
                                    similarity_sets=similarity_sets,
                                    curriculum=get_curriculum_schedule(0, n_epoch),
                                    dataloader=val_dataloader,
                                    step=train_step,
                                    best_results=best_results,
                                    device=device,
                                    summary_writer=val_summary_writer,
                                    n_sample_per_summary=n_sample_per_summary,
                                    log_path=log_path
        )
    
    model.train()
    # Begin training
    time_start = time.time()
    log('Begin training...', log_path)
    
    
    
    
    loss = None
    output = None
    
    # ================ Main training loop ================
    for epoch in range(1, n_epoch + 1):
        train_sampler.set_epoch(epoch)  # Inside the training loop, once per epoch
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for batch in tqdm.tqdm(train_dataloader, desc=f'{local_rank}: {epoch}/{n_epoch}'):
            if loss is not None:
                del loss
            if output is not None:
                del output
            
            torch.cuda.empty_cache()
            curriculum = get_curriculum_schedule(epoch, n_epoch)

            n_batches += 1
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Unpack batch
            depth = batch['depth']
            image = batch['image']
            segmentation = batch['segmentation']
            
            # Forward through network
            
            
            with autocast():
                output, temperature = model(depth)
                loss, loss_info = unwrap_model(model).compute_loss(
                    output,
                    segmentation,
                    candidate_text_embeddings,
                    temperature,
                    label_similarity_sets=similarity_sets,
                    pct_medium=curriculum['pct_medium'],
                    pct_hard=curriculum['pct_hard'],
                    pct_rand=curriculum['pct_rand'],
                )
            
            # Zero gradients, backpropagate, and update weights
            loss = loss / accumulation_steps  # Normalize loss
            scaler.scale(loss).backward()

            if (n_batches % accumulation_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                train_step += 1
                
                total_loss += loss.item()
                
                
                
                # Logging and validation
                if (train_step % n_step_per_summary) == 0 and local_rank == 0:
                    log_training_summary(
                        train_summary_writer=train_summary_writer,
                        train_step=train_step,
                        image_samples=image,
                        depth_samples=depth,
                        ground_truth_segmentation_samples=segmentation,
                        output_segmentation_samples=output,
                        candidate_labels=candidate_labels,
                        loss_info=loss_info,
                        n_sample_per_summary=n_sample_per_summary
                    )
                    
                    train_summary_writer.add_scalars("train/curriculum", {
                        "pct_medium": curriculum["pct_medium"],
                        "pct_hard": curriculum["pct_hard"],
                        "pct_rand": curriculum["pct_rand"],
                    }, global_step=train_step)
                    
                    # Validate model
                    with torch.no_grad():
                        model.eval()
                        best_results = validate_model(model=unwrap_model(model),
                                    candidate_text_embeddings=candidate_text_embeddings,
                                    candidate_labels=candidate_labels,
                                    equivalence_tensor=equivalence_tensor,
                                    similarity_sets=similarity_sets,
                                    curriculum=curriculum,
                                    dataloader=val_dataloader,
                                    step=train_step,
                                    best_results=best_results,
                                    device=device,
                                    summary_writer=val_summary_writer,
                                    n_sample_per_summary=n_sample_per_summary,
                                    max_failed_log=16,
                                    log_path=log_path
                        )
                        model.train()
                
                # Save checkpoints
                if (train_step % n_step_per_checkpoint) == 0 and local_rank == 0:
                    save_checkpoint_and_log_progress(
                        unwrap_model(model), model_checkpoint_path, train_step,
                        n_train_step, start_step, loss.item(),
                        time_start, log_path, optimizer
                    )
                
            
    
        avg_loss = total_loss / max(n_batches, 1)
        log(f"Rank {local_rank} | Epoch {epoch} | Step {train_step} | Average loss: {avg_loss}", log_path)
        
        # Update learning rate scheduler
        scheduler.step()
    
    # Save final model
    if local_rank == 0:
        save_checkpoint_and_log_progress(
            model, model_checkpoint_path, train_step,
            n_train_step, train_step - n_train_step, loss.item(),
            time_start, log_path, optimizer
        )
    
    if dist.is_initialized():
        dist.barrier()  # wait for all processes
        dist.destroy_process_group()



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
    if restore_path_model is not None and restore_path_model != '':
        start_step, optimizer = model.restore_model(
            restore_path_model, optimizer=optimizer)
    else:
        start_step = 0
        
    return start_step, optimizer



        
    
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

