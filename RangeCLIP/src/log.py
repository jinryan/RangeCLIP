import os
import torch
import sys
import torch
from torchvision.utils import make_grid
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from utils.src.log_utils import log

def log_input_settings(log_path,
                       n_batch=None,
                       n_height=None,
                       n_width=None):

    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    log('', log_path)

def log_network_settings(log_path,
                         # Network settings
                         model_architecture,
                         # Weight settings
                         parameters_model=[]):

    # Computer number of parameters
    n_parameter = sum(p.numel() for p in parameters_model)

    log('Network settings:', log_path)
    log('model_architecture={}'.format(model_architecture),
        log_path)

    log('Weight settings:', log_path)
    log('n_parameter={}'.format(n_parameter),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          scheduler_type):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('scheduler type: ' + str(scheduler_type), log_path)
    log('scheduler: {}, learning_schedule=[{}]'.format(
        scheduler_type,
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates))),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_losses={},
                           w_weight_decay=None):

    w_losses_text = ''
    for idx, (key, value) in enumerate(w_losses.items()):

        if idx > 0 and idx % 3 == 0:
            w_losses_text = w_losses_text + '\n'

        w_losses_text = w_losses_text + '{}={:.1e}'.format(key, value) + '  '

    log('Loss function settings:', log_path)
    if len(w_losses_text) > 0:
        log(w_losses_text, log_path)

    if w_weight_decay is not None:
        log('w_weight_decay={:.1e}'.format(
            w_weight_decay),
            log_path)

    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_sample_per_summary=None,
                        validation_start_step=None,
                        restore_path_model=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_step_per_checkpoint), log_path)

        if validation_start_step is not None:
            log('validation_start_step={}'.format(validation_start_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'log_summary_frequency={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_sample_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_sample_per_summary={}'
        summary_settings_vars.append(n_sample_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_path_model is not None and restore_path_model != '':
        log('restore_path={}'.format(restore_path_model),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
    

def log_configuration(log_path, labeled_metadata_path, unlabeled_metadata_path, batch_size, n_height, n_width,
                     model_architecture, parameters_model, n_train_sample, n_train_step,
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
        n_batch=batch_size,
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
            for i in range(min(n_sample_per_summary, labeled_images.shape[0])):
                display_images.append(labeled_images[i].cpu())

        # Labeled depth
        if labeled_depths is not None and len(labeled_depths) > 0:
            labeled_depths_vis = labeled_depths[:n_sample_per_summary]
            labeled_depths_colored = apply_colormap(labeled_depths_vis)
            for i in range(labeled_depths_colored.shape[0]):
                display_images.append(labeled_depths_colored[i])

        # Unlabeled depth
        if unlabeled_depths is not None and len(unlabeled_depths) > 0:
            B, N, H, W = unlabeled_depths.shape
            patches = unlabeled_depths[:, 0].unsqueeze(1)  # [B, 1, H, W]
            patches = patches[:n_sample_per_summary]
            patches_colored = apply_colormap(patches)
            for i in range(patches_colored.shape[0]):
                display_images.append(patches_colored[i])


        # Unlabeled image (optional, but good to track RGB reference)
        if unlabeled_images is not None and len(unlabeled_images) > 0:
            unlabeled_images_vis = unlabeled_images[:n_sample_per_summary].cpu()
            for i in range(unlabeled_images_vis.shape[0]):
                display_images.append(unlabeled_images_vis[i])


        
        # Log concatenated image strip
        if display_images:
            for i, img in enumerate(display_images):
                assert img.ndim == 3 and img.shape[0] == 3, f"Image {i} shape mismatch: {img.shape}"
                
            grid = make_grid(torch.cat(display_images, dim=2), nrow=n_sample_per_summary)  # concat width-wise
            train_summary_writer.add_image('train/visual_summary', grid, global_step=train_step)

        # Log scalar loss values
        for name, val in loss_info.items():
            train_summary_writer.add_scalar(f'train/loss_{name}', val, global_step=train_step)

        # Log text labels
        if labeled_ground_truth_indices is not None and len(labeled_ground_truth_indices) > 0:
            text_summary = "\n".join([f"{i+1}: {t}" for i, t in enumerate(labeled_ground_truth_indices[:n_sample_per_summary])])
            train_summary_writer.add_text('train/labels', text_summary, global_step=train_step)



