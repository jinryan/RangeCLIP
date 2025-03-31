import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import tqdm
import sys
import os
import random
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.src import data_utils, eval_utils
import datasets
from utils.src.log_utils import log
from utils.src.networks import DepthEncoder, ImageEncoder
from transformers import CLIPModel
from RangeCLIP.src.log import log_input_settings, log_loss_func_settings, log_network_settings, log_system_settings, log_training_settings

parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to list of training image paths')
parser.add_argument('--train_depth_path',
    type=str, default=None, help='Path to list of training depth paths')

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

# Photometric data augmentations
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='Range of brightness adjustments for augmentation, if does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1], help='Range of contrast adjustments for augmentation, if does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_hue',
    nargs='+', type=float, default=[-1, -1], help='Range of hue adjustments for augmentation, if does not contain -1, apply random hue')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1], help='Range of saturation adjustments for augmentation, if does not contain -1, apply random saturation')

# Geometric data augmentations
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Random flip type for data augmentation: horizontal, vertical')

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

def train_depth_encoder(
        train_image_path,
        train_depth_path,
        batch_size,
        n_height,
        n_width,
        depth_encoder_type,
        augmentations,
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
    device = _setup_device(device)
    
    # Set up directories for checkpoints and logs
    model_checkpoint_path, log_path, event_path = setup_checkpoint_and_event_paths(
        checkpoint_path, 'resnet_depth_encoder')
    
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
    train_dataloader, val_dataloader, n_train_step = _setup_dataloaders(
        train_image_path, train_depth_path, resize_shape, 
        augmentations, learning_schedule, batch_size, n_thread, n_epoch
    )
    
    # Initialize models
    depth_encoder, image_encoder = _initialize_models(
        depth_encoder_type, clip_model_name, device
    )
    
    # Set up optimizer and learning rate scheduler
    optimizer, scheduler = _setup_optimizer_and_scheduler(
        depth_encoder.parameters(), learning_rates, w_weight_decay, 
        scheduler_type, learning_schedule
    )
    
    # Restore model if specified
    start_step = train_step = _restore_model_if_needed(
        depth_encoder, restore_path_model, optimizer, learning_rates, device
    )
    
    # Configure model for training
    depth_encoder.train()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        depth_encoder = nn.DataParallel(depth_encoder)
    
    # Log configuration settings
    _log_configuration(
        log_path, train_image_path, 
        batch_size, n_height, n_width,
        depth_encoder_type, depth_encoder.parameters(),
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
        for batch in tqdm.tqdm(train_dataloader, desc=f'{epoch}/{n_epoch}'):
            train_step += 1

            # Process batch
            depth, image = _prepare_batch(batch, device)
            
            # Forward pass
            depth_embedding = depth_encoder(depth)
            image_embedding = image_encoder(image)
            
            # Compute loss
            loss, loss_info = depth_encoder.module.compute_loss(depth_embedding, image_embedding)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging and validation
            if (train_step % n_step_per_summary) == 0:
                _log_training_summary(
                    depth_encoder, train_summary_writer, train_step,
                    image, depth, loss_info, batch_size, n_sample_per_summary
                )
                
                # Validate model
                with torch.no_grad():
                    depth_encoder.eval()
                    best_results, _ = validate_encoder(
                        depth_encoder, image_encoder, val_dataloader,
                        train_step, best_results, device,
                        val_summary_writer, n_sample_per_summary, log_path
                    )
                    depth_encoder.train()
            
            # Save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                _save_checkpoint_and_log_progress(
                    depth_encoder, model_checkpoint_path, train_step,
                    n_train_step, start_step, loss.item(),
                    time_start, log_path, optimizer
                )
        
        # Update learning rate scheduler
        _update_scheduler(scheduler, scheduler_type, depth_encoder, 
                         image_encoder, val_dataloader, train_step,
                         best_results, device, val_summary_writer,
                         n_sample_per_summary, log_path)
    
    # Save final model
    _save_checkpoint_and_log_progress(
        depth_encoder, model_checkpoint_path, train_step,
        n_train_step, train_step - n_train_step, loss.item(),
        time_start, log_path, optimizer
    )


def _setup_device(device):
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


def _setup_dataloaders(train_image_path, train_depth_path, resize_shape, 
                     augmentations, learning_schedule, batch_size, n_thread, n_epoch):
    """
    Set up data loaders for training and validation.
    
    Args:
        train_image_path (str): Path to training images
        train_depth_path (str): Path to training depth maps
        resize_shape (tuple): Target shape for resizing
        augmentations (dict): Data augmentation settings
        learning_schedule (list): Learning rate schedule
        batch_size (int): Number of samples per batch
        n_thread (int): Number of data loading threads
        n_epoch (int): Total number of epochs
        
    Returns:
        tuple: (train_dataloader, val_dataloader, n_train_step)
    """
    return setup_dataloader(
        train_image_path=train_image_path,
        train_depth_path=train_depth_path,
        resize_shape=resize_shape,
        augmentations=augmentations,
        batch_size=batch_size,
        n_thread=n_thread,
        n_epoch=learning_schedule[-1]
    )


def _initialize_models(depth_encoder_type, clip_model_name, device):
    """
    Initialize depth encoder and image encoder models.
    
    Args:
        depth_encoder_type (str): Type of depth encoder architecture
        clip_model_name (str): Name of the CLIP model
        device (torch.device): Device to load models on
        
    Returns:
        tuple: (depth_encoder, image_encoder)
    """
    # Load CLIP model for image embedding
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    embedding_dim = clip_model.config.projection_dim
    
    # Initialize depth encoder based on specified type
    if depth_encoder_type == 'resnet':
        depth_encoder = DepthEncoder(
            n_layer=18,
            input_channels=1,
            n_filters=[32, 64, 128, 256, 512],
            embedding_dim=embedding_dim,
            weight_initializer='kaiming_uniform',
            activation_func='relu',
            use_batch_norm=True,
            use_instance_norm=False
        )
    else:
        raise ValueError(f'{depth_encoder_type} not supported as depth encoder. Supported: resnet')
    
    # Initialize image encoder with CLIP model
    image_encoder = ImageEncoder(clip_model)
    
    # Move models to device
    depth_encoder = depth_encoder.to(device)
    image_encoder = image_encoder.to(device)
    
    return depth_encoder, image_encoder


def _setup_optimizer_and_scheduler(parameters, learning_rates, w_weight_decay, 
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


def _restore_model_if_needed(depth_encoder, restore_path_model, optimizer, learning_rates, device):
    """
    Restore model from checkpoint if path is provided.
    
    Args:
        depth_encoder: The depth encoder model
        restore_path_model (str): Path to restore model from
        optimizer: The optimizer
        learning_rates (list): Learning rates
        device (torch.device): The device
        
    Returns:
        int: Starting step (0 for new training, restored step for continuing)
    """
    if restore_path_model is not None:
        start_step, optimizer = depth_encoder.restore_encoder(
            restore_path_model, device, optimizer=optimizer)
        
        # Reset learning rate to initial value
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rates[0]
    else:
        start_step = 0
        
    return start_step


def _log_configuration(log_path, train_image_path, batch_size, n_height, n_width,
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
    log(train_image_path, log_path)

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


def _prepare_batch(batch, device):
    """
    Prepare a batch of data by moving it to the appropriate device.
    
    Args:
        batch: A batch of data (depth, image)
        device (torch.device): The device to move data to
        
    Returns:
        tuple: (depth, image) on the target device
    """
    depth, image = batch
    depth = depth.to(device)
    image = image.to(device)
    return depth, image


def _log_training_summary(depth_encoder, summary_writer, step, image, depth, 
                        loss_info, batch_size, n_sample_per_summary):
    """
    Log training summary to tensorboard.
    
    Args:
        depth_encoder: The depth encoder model
        summary_writer: Tensorboard summary writer
        step (int): Current training step
        image: Batch of images
        depth: Batch of depth maps
        loss_info: Loss information
        batch_size (int): Batch size
        n_sample_per_summary (int): Number of samples to visualize
    """
    depth_encoder.module.log_summary(
        summary_writer=summary_writer,
        tag='train',
        step=step,
        image=image,
        depth=depth,
        scalars=loss_info,
        n_sample_per_summary=min(batch_size, n_sample_per_summary)
    )


def _save_checkpoint_and_log_progress(depth_encoder, model_checkpoint_path, train_step,
                                    n_train_step, start_step, loss_value,
                                    time_start, log_path, optimizer):
    """
    Save model checkpoint and log training progress.
    
    Args:
        depth_encoder: The depth encoder model
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

    depth_encoder.module.save_encoder(
        model_checkpoint_path.format(train_step), train_step, optimizer
    )


def _update_scheduler(scheduler, scheduler_type, depth_encoder, image_encoder,
                    val_dataloader, train_step, best_results, device,
                    val_summary_writer, n_sample_per_summary, log_path):
    """
    Update the learning rate scheduler.
    
    Args:
        scheduler: The learning rate scheduler
        scheduler_type (str): Type of scheduler
        depth_encoder: The depth encoder model
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
        depth_encoder.eval()
        _, val_imce_loss = validate_encoder(
            depth_encoder, image_encoder, val_dataloader,
            train_step, best_results, device,
            val_summary_writer, n_sample_per_summary, log_path
        )
        depth_encoder.train()
        
        scheduler.step(val_imce_loss)
    else:
        scheduler.step()

def validate_encoder(depth_encoder,
                     image_encoder,
                    dataloader,
                    step,
                    best_results,
                    device,
                    summary_writer,
                    n_sample_per_summary=4,
                    log_path=None):

    n_sample = len(dataloader)
    imce = np.zeros(n_sample)
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)

    image_summary = []
    depth_summary = []

    for idx, batch in enumerate(dataloader):

        # Unpack batch
        depth, image = batch

        # Move noisy image to device
        image = image.to(device)
        depth = depth.to(device)

        # Forward through the network, return_all_outputs=False
        depth_embedding = depth_encoder(depth)
        image_embedding = image_encoder(image)
        
        # Collect samples for Tensorboard display
        if (idx % (n_sample // n_sample_per_summary)) == 0 and summary_writer is not None:
            image_summary.append(image)
            depth_summary.append(depth)

        # Move output image to CPU and convert output and ground truth to numpy to validate
        depth_embedding = depth_embedding.detach().cpu().numpy()
        image_embedding = image_embedding.detach().cpu().numpy()


        # Compute evaluation metrics using eval_utils
        imce[idx] = eval_utils.info_nce(depth_embedding, image_embedding)
        mae[idx] = eval_utils.mean_abs_err(depth_embedding, image_embedding)
        rmse[idx] = eval_utils.root_mean_sq_err(depth_embedding, image_embedding)
        

    # Compute mean scores
    mae = mae.mean()
    rmse = rmse.mean()
    imce = imce.mean()

    # Log to lists of inputs, outputs, ground truth summaries to Tensorboard
    # Concatenate lists along batch dimension, create dictionary with metrics as keys and scores as value
    # for logging into Tensorboard
    if summary_writer is not None:
        depth_encoder.module.log_summary(
            summary_writer=summary_writer,
            tag='val',
            step=step,
            image=torch.cat(image_summary, dim=0),
            depth=torch.cat(depth_summary, dim=0),
            scalars={'MAE': mae, 'RMSE': rmse, 'InfoNCE': imce},
        )

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse),
        log_path)

    # If the model improves over all results on both metrics by 2nd decimal
    improved = best_results['mae'] > mae and best_results['rmse'] > rmse and best_results['imce'] > imce


    # Update best results
    if improved:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imce'] = imce

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'InfoMCE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imce']), log_path)

    return best_results, imce


    
    
def setup_dataloader(train_image_path, train_depth_path, augmentations, resize_shape, batch_size, n_thread, n_epoch):
    train_image_paths = data_utils.read_paths(train_image_path)
    train_depth_paths = data_utils.read_paths(train_depth_path)
    
    full_dataset = datasets.ImageDepthDataset(image_paths=train_image_paths,
                                               depth_paths=train_depth_paths,
                                               resize_shape=resize_shape,
                                               augmentation_random_brightness=augmentations.get('random_brightness', None),
                                               augmentation_random_contrast=augmentations.get('random_contrast', None),
                                               augmentation_random_flip_type=augmentations.get('random_flip_type', None),
                                               augmentation_random_hue=augmentations.get('random_hue', None),
                                               augmentation_random_saturation=augmentations.get('random_saturation', None)
                                               )
    
    # Use provided seed or generate a random one
    generator = torch.Generator()
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_thread)
    
    # More accurate calculation of train steps
    n_train_samples = len(train_dataset)
    n_train_steps = ((n_train_samples + batch_size - 1) // batch_size) * n_epoch
    
    val_dataloader = DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,  # Changed to False for validation
                                                 num_workers=n_thread)
    
    return train_dataloader, val_dataloader, n_train_steps
    
    
    
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

    train_depth_encoder(train_image_path=args.train_image_path,
          train_depth_path=args.train_depth_path,
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
          # Photometric data augmentations
          augmentations={
            "random_brightness": args.augmentation_random_brightness,
            "random_contrast": args.augmentation_random_contrast,
            "random_hue": args.augmentation_random_hue,
            "random_saturation": args.augmentation_random_saturation,
            # Geometric data augmentations
            "random_flip_type": args.augmentation_random_flip_type,
          },
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