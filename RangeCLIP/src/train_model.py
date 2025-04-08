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
parser.add_argument('--dataset_path',
    type=str, required=True, help='Path to dataset')
parser.add_argument('--metadata_path',
    type=str, required=True, help='Path to dataset metadata')
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

def contains_nan(tensor):
    return torch.isnan(tensor).any().item()

def train_depth_clip_model(
        root_dir,
        metadata_file,
        labels_path,
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
        root_dir, metadata_file, labels_path, resize_shape, 
        batch_size, n_thread, n_epoch
    )
    
    model = DepthClipModel(depth_encoder_type, clip_model_name, device)
    
    # Set up optimizer and learning rate scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model.parameters(), learning_rates, w_weight_decay, 
        scheduler_type, learning_schedule
    )
    
    # Restore model if specified
    start_step = train_step = 0
    
    restore_model_if_needed(
        model, restore_path_model, None
    )
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rates[0]
    
    # Configure model for training
    model.train()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Log configuration settings
    log_configuration(
        log_path, metadata_file, 
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
                # Unpack batch to device
                # We are guaranteed depth and images but not necessarily labels
                depth_maps, images, label_indices = prepare_batch(batch, device)
                            
                batch_size = depth_maps.shape[0]
                
                depth_embeddings, _, image_embeddings = model.forward(
                    depth_map=depth_maps,
                    image=images
                )
                
                if candidate_labels != None:
                    # Get loss values for embedding quality evaluation
                    loss, loss_info = model.module.compute_trimodal_loss(
                        depth_embeddings=depth_embeddings,
                        image_embeddings=image_embeddings,
                        candidate_labels=candidate_labels,
                        ground_truth_indices=label_indices,
                        w_text=0.8
                    )
                else:
                    loss, loss_info = model.module.compute_bimodal_loss(
                        depth_embeddings=depth_embeddings,
                        other_embeddings=image_embeddings
                    )
                
                total_loss += loss
                
                # print(f"Loss: {loss_info}")
                                        
                # Backward pass and optimization
                optimizer.zero_grad()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print("Loss is NaN or Inf, skipping batch")
                    continue
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
                    model, train_summary_writer, train_step,
                    images, depth_maps, label_indices, loss_info, batch_size, n_sample_per_summary
                )
                
                # Validate model
                with torch.no_grad():
                    model.module.eval()
                    best_results, _ = validate_model(
                        model, candidate_labels, val_dataloader,
                        train_step, best_results, device,
                        val_summary_writer, n_sample_per_summary, log_path
                    )
                    model.module.train()
            
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
        
    return start_step


def log_configuration(log_path, train_image_path, batch_size, n_height, n_width,
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


def prepare_batch(batch, device):
    depth, image, label = batch
    
    depth = depth.to(device)
    image = image.to(device)
    
    return depth, image, label


def log_training_summary(model, summary_writer, step, image, depth, text, 
                        loss_info, batch_size, n_sample_per_summary):
    """
    Log training summary to tensorboard.
    
    Args:
        model: The model
        summary_writer: Tensorboard summary writer
        step (int): Current training step
        image: Batch of images
        depth: Batch of depth maps
        loss_info: Loss information
        batch_size (int): Batch size
        n_sample_per_summary (int): Number of samples to visualize
    """
    model.module.log_summary(
        summary_writer=summary_writer,
        tag='train',
        step=step,
        image=image,
        depth=depth,
        text=text,
        scalars=loss_info,
        n_sample_per_summary=min(batch_size, n_sample_per_summary)
    )


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

    model.module.save_model(
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
        model.module.eval()
        _, val_imce_loss = validate_model(
            model, candidate_labels, val_dataloader,
            train_step, best_results, device,
            val_summary_writer, n_sample_per_summary, log_path
        )
        model.module.train()
        
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
    
    n_sample = 0
    correct_predictions = 0
    
    # Metrics for embedding quality
    total_loss = 0.0
    
    image_summary = []
    depth_summary = []
    text_summary = []
    
    with torch.no_grad():  # Disable gradient computation for validation
        for idx, batch in enumerate(dataloader):
            # Unpack batch to device
            depth_maps, images, label_indices = prepare_batch(batch, device)
            
            labels = [candidate_labels[i-1] for i in label_indices]
            
                        
            batch_size = depth_maps.shape[0]
            
            # Get loss values for embedding quality evaluation
            loss, loss_info = model.module.compute_loss(
                depth_maps=depth_maps,
                images=images,
                labels=labels,
                w_text=0.8
            )
            
            # Accumulate losses
            total_loss += loss.item() * batch_size
            
            # For classification accuracy, run prediction on each depth map
            for i in range(batch_size):
                single_depth = depth_maps[i:i+1]  # Get single sample with batch dimension
                predicted_class, confidence = model.predict(single_depth, candidate_labels)
                
                if predicted_class == label_indices[i].item():
                    correct_predictions += 1
            
            # Collect samples for TensorBoard display
            if (idx % max(1, len(dataloader) // n_sample_per_summary)) == 0 and summary_writer is not None:
                image_summary.append(images[:min(batch_size, n_sample_per_summary)])
                depth_summary.append(depth_maps[:min(batch_size, n_sample_per_summary)])
                
                # Add text descriptions
                batch_text = [candidate_labels[i-1] for i in label_indices[:min(batch_size, n_sample_per_summary)]]
                text_summary.extend(batch_text)
    
    # Compute average metrics
    avg_loss = total_loss / n_sample
    accuracy = correct_predictions / n_sample if n_sample > 0 else 0.0
    
    # Log to TensorBoard
    if summary_writer is not None and image_summary and depth_summary:
        model.log_summary(
            summary_writer=summary_writer,
            tag='val',
            step=step,
            image=torch.cat(image_summary, dim=0) if image_summary else None,
            depth=torch.cat(depth_summary, dim=0) if depth_summary else None,
            text=text_summary if text_summary else None,
            scalars={
                'Loss': avg_loss,
                'Accuracy': accuracy
            }
        )
    
    # Print validation results to console
    if log_path:
        log('Validation results:', log_path)
        log('{:>8}  {:>8}  {:>10}'.format(
            'Step', 'Loss', 'Accuracy'),
            log_path)
        log('{:8}  {:8.4f}  {:10.4f}'.format(
            step, avg_loss, accuracy),
            log_path)
    
    # Check if model improved
    improved = False
    
    # For initial run when best_results might be empty
    if 'accuracy' not in best_results or best_results['accuracy'] < accuracy:
        improved = True
    
    # Update best results if improved
    if improved:
        best_results['step'] = step
        best_results['loss'] = avg_loss
        best_results['accuracy'] = accuracy
    
    if log_path:
        log('Best results:', log_path)
        log('{:>8}  {:>8}  {:>10}'.format(
            'Step', 'Loss', 'Accuracy'),
            log_path)
        log('{:8}  {:8.4f}  {:12.4f}  {:12.4f}  {:10.4f}'.format(
            best_results['step'],
            best_results['loss'],
            best_results['accuracy']), 
            log_path)
    
    return best_results, accuracy


    
    
def setup_dataloaders(root_dir, metadata_file, labels_path, resize_shape, batch_size, n_thread, n_epoch):
    resize_transform = transforms.Resize(resize_shape)
    
    full_dataset = datasets.ImageDepthTextDataset(metadata_file=metadata_file,
                                                  root_dir=root_dir,
                                                  labels_path=labels_path,
                                                  transform=resize_transform)
    
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
    
    labels = full_dataset.get_labels()
    
    return train_dataloader, val_dataloader, n_train_steps, labels
    
    
    
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
          root_dir=args.dataset_path,
          metadata_file=args.metadata_path,
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