import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
import tqdm
import utils.src.data_utils as data_utils
import datasets
from utils.src.log_utils import log
from utils.src.networks import DepthEncoder, ImageEncoder
from transformers import CLIPModel

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
    type=str, help='Available: resnet')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[2e-4, 1e-4, 5e-5, 1e-5], help='Space delimited list of learning rates')
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
parser.add_argument('--n_image_per_summary',
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

def train_depth_encoder(train_image_path,
                            train_depth_path,
                            batch_size,
                            n_height,
                            n_width,
                            depth_encoder_type,
                            augmentations,
                            learning_rates,
                            learning_schedule,
                            w_weight_decay,
                            checkpoint_path,
                            n_steps_per_checkpoint,
                            n_steps_per_summary,
                            n_samples_per_summary,
                            validation_start_step,
                            restore_path_model,
                            clip_model_name='openai/clip-vit-base-patch32',
                            device='cuda',
                            n_thread=8):
    
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model_checkpoint_path, log_path, event_path = setup_checkpoint_and_event_paths(checkpoint_path, 'resnet_depth_encoder')
    
    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty
    }
    
    resize_shape = (n_height, n_width)
    train_dataloader, n_train_step = setup_dataloader(train_image_path=train_image_path,
                                                    train_depth_path=train_depth_path,
                                                    resize_shape=resize_shape,
                                                    augmentations=augmentations,
                                                    learning_schedule=learning_schedule,
                                                    batch_size=batch_size,
                                                    n_thread=n_thread
                                        )
    
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    embedding_dim = clip_model.config.projection_dim
    
    if depth_encoder_type == 'resnet':
        depth_encoder = DepthEncoder(n_layer=18,
                            input_channels=1,
                            n_filters=[32, 64, 128, 256, 512],
                            embedding_dim=embedding_dim,
                            weight_initializer='kaiming_uniform',
                            activation_func='leaky_relu',
                            use_batch_norm=True)
    else:
        raise ValueError(f'{depth_encoder_type} not supported as depth encoder')
    
    image_encoder = ImageEncoder(clip_model)
    
    parameters_model = depth_encoder.parameters()
    
    n_epoch = learning_schedule[-1]
    
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]
    
    optimizer = torch.optim.Adam(parameters_model,
                                 lr=learning_rate,
                                 weight_decay=w_weight_decay)
    
    # Restore the model
    if restore_path_model is not None:
        start_step, optimizer = depth_encoder.restore_model(restore_path_model, optimizer=optimizer)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    else:
        start_step = 0
        
    train_step = start_step
    
    depth_encoder.train()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        depth_encoder = nn.DataParallel(depth_encoder)
        
        
    time_start = time.time()
    
    log('Begin training...', log_path)
    
    for epoch in range(1, n_epoch + 1):
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        for batch in tqdm.tqdm(train_dataloader, desc='{}/{}'.format(epoch, n_epoch)):

            train_step = train_step + 1

            # Move batch to device
            batch = [
                in_.to(device) for in_ in batch
            ]

            # Unpack batch
            depth, image = batch

            # Forward through the network, set return_all_outputs=True
            depth_embedding = depth_encoder(depth)
            image_embedding = image_encoder(image)

            # Compute loss
            loss = depth_encoder.compute_loss(depth_embedding, image_embedding)

            # Zero gradient, backpropagate, and update weights with optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        time_elapse = (time.time() - time_start) / 3600
        time_remain = (n_train_step - train_step + start_step) * time_elapse / (train_step - start_step)

        log('Step={:6}/{}  Loss={:.7f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
            train_step, n_train_step + start_step, loss.item(), time_elapse, time_remain),
            log_path)
        
        depth_encoder.save_model(model_checkpoint_path.format(train_step), train_step, optimizer)
    
    
    

def setup_dataloader(train_image_path, train_depth_path, augmentations, resize_shape, learning_schedule, batch_size, n_thread):
    train_image_paths = data_utils.read_paths(train_image_path)
    train_depth_paths = data_utils.read_paths(train_depth_path)
    
    n_train_sample = len(train_image_paths)
    n_train_step = learning_schedule[-1] * (n_train_sample // batch_size)
    
    train_dataset = datasets.ImageDepthDataset(image_paths=train_image_paths,
                                               depth_paths=train_depth_paths,
                                               resize_shape=resize_shape,
                                               augmentation_random_brightness=augmentations.get('random_brightness', None),
                                               augmentation_random_contrast=augmentations.get('random_contrast', None),
                                               augmentation_random_flip_type=augmentations.get('random_flip_type', None),
                                               augmentation_random_hue=augmentations.get('random_hue', None),
                                               augmentation_random_saturation=augmentations.get('random_saturation', None)
                                               )
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size = batch_size,
                                                   shuffle=True,
                                                   num_workers=n_thread)
    
    return train_dataloader, n_train_step
    
    
    
def setup_checkpoint_and_event_paths(checkpoint_path, model_name):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_checkpoint_path = os.path.join(
        checkpoint_path,
        'checkpoints',
        f'{model_name}.pth'
    )
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'tensorboard')

    os.makedirs(event_path, exist_ok=True)
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    
    return model_checkpoint_path, log_path, event_path

class MultiModalContrastiveLearner:
    def __init__(self, 
                 depth_encoder, 
                 image_encoder, 
                 text_encoder,
                 loss_fn,
                 temperature=0.07, 
                 learning_rate=1e-4):
        """
        Multi-modal contrastive learning framework.
        """
        
        self.depth_encoder = depth_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        self.loss_fn = loss_fn
        
        # Freeze text encoder
        for param in self.text_encoder.clip_model.parameters():
            param.requires_grad = False
        
        # Freeze image encoder
        for param in self.image_encoder.clip_model.parameters():
            param.requires_grad = False
        
        # Trainable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.optimizer = optim.Adam([
            {'params': self.depth_encoder.parameters()},
            {'params': [self.temperature]}
        ], lr=learning_rate)
        
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch across multi-modal data.
        """
        self.depth_encoder.train()
        
        total_loss = 0.0
        for batch in dataloader:
            
            # Unpack batch
            depth_maps = batch['depth']
            images = batch['image']
            text_descriptions = batch['text']
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Encode modalities
            depth_embeddings = self.depth_encoder(depth_maps)
            image_embeddings = self.image_encoder(images)
            text_embeddings = self.text_encoder(text_descriptions)
            
            # Compute contrastive loss
            loss = self.loss_fn(
                depth_embeddings, 
                image_embeddings, 
                text_embeddings
            )
            
            # Backpropagate and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, 
              train_dataloader, 
              val_dataloader=None, 
              epochs=10, 
              early_stopping_patience=3):
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train epoch
            train_loss = self.train_epoch(train_dataloader)
            train_losses.append(train_loss)
            
            # Validate if validation data provided
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
                
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # torch.save(self.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                # Stop if no improvement
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def validate(self, val_dataloader):
        """
        Validate model on validation dataset.
        """
        self.depth_encoder.eval()
        
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                depth_maps = batch['depth']
                images = batch['image']
                text_descriptions = batch['text']
                
                # Encode modalities
                depth_embeddings = self.depth_encoder(depth_maps)
                image_embeddings = self.image_encoder(images)
                text_embeddings = self.text_encoder(text_descriptions)
                
                # Compute contrastive loss
                loss = self.compute_contrastive_loss(
                    depth_embeddings, 
                    image_embeddings, 
                    text_embeddings
                )
                
                total_val_loss += loss.item()
        
        return total_val_loss / len(val_dataloader)
    

if __name__ == '__main__':

    # Network settings
    args.model_architecture = args.model_architecture.lower()

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    args.augmentation_random_flip_type = [
        flip_type.lower() for flip_type in args.augmentation_random_flip_type
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    train_depth_encoder(train_image_path=args.train_image_path,
          train_depth_path=args.train_depth_path,
          # Batch settings
          batch_size=args.batch_size,
          n_height=args.n_height,
          n_width=args.n_width,
          # Network settings
          depth_encoder_type=args.depth_encoder_type,
          # Training settings
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          # Photometric data augmentations
          augmentations={
            "augmentation_random_brightness": args.augmentation_random_brightness,
            "augmentation_random_contrast": args.augmentation_random_contrast,
            "augmentation_random_hue": args.augmentation_random_hue,
            "augmentation_random_saturation": args.augmentation_random_saturation,
            # Geometric data augmentations
            "augmentation_random_flip_type": args.augmentation_random_flip_type,
          },
          # Loss function settings
          w_losses=args.w_losses,
          w_weight_decay=args.w_weight_decay,
          # Checkpoint settings
          checkpoint_path=args.checkpoint_path,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_image_per_summary=args.n_image_per_summary,
          validation_start_step=args.validation_start_step,
          restore_path_model=args.restore_path_model,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
