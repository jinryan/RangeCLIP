import os
import torch
import sys
import torch
from torchvision.utils import make_grid
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torchvision.utils import make_grid
from io import BytesIO
from PIL import Image
from torchvision import transforms
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
    

def log_configuration(log_path, labeled_metadata_path, batch_size, n_height, n_width,
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
        image_samples,
        depth_samples,
        ground_truth_segmentation_samples,
        output_segmentation_samples,
        candidate_labels,
        loss_info,
        n_sample_per_summary
    ):
    """
    Log training summary to TensorBoard.
    """
    with torch.no_grad():

        # Log scalar loss values
        for name, val in loss_info.items():
            train_summary_writer.add_scalar(f'train/loss_{name}', val, global_step=train_step)

def log_validation_summary(
        val_summary_writer,
        val_step,
        scalar_info,
    ):
    """
    Log validation summary to TensorBoard.
    """
    with torch.no_grad():
        for name, val in scalar_info.items():
            val_summary_writer.add_scalar(f'val/{name}', val, global_step=val_step)




def visualize_batch_predictions(
    images,               # Tensor [B, 3, H, W]
    depths,               # Tensor [B, 1, H, W]
    seg_gt,               # Tensor [B, H, W] or [B, 1, H, W]
    seg_pred,             # Tensor [B, H, W] or [B, 1, H, W]
    candidate_labels,     # List[str]
    save_dir="output/visualizations_preds"
):
    os.makedirs(save_dir, exist_ok=True)
    B = images.size(0)

    tab20 = cm.get_cmap('tab20', 20)
    colors = (tab20(np.linspace(0, 1, 20))[:, :3] * 255).astype(np.uint8)

    for i in range(B):
        # RGB image (CLIP de-normalization)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.2686, 0.2613, 0.2758]) + np.array([0.4815, 0.4578, 0.4082])
        img = np.clip(img, 0, 1)

        # Depth
        depth_np = depths[i][0].cpu().numpy()

        # Segmentation GT and prediction
        gt = seg_gt[i][0] if seg_gt[i].dim() == 3 else seg_gt[i]
        pred = seg_pred[i][0] if seg_pred[i].dim() == 3 else seg_pred[i]
        gt = gt.cpu().numpy().astype(np.int32)
        pred = pred.cpu().numpy().astype(np.int32)

        # Color visualizations
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        pred_vis = np.zeros((*pred.shape, 3), dtype=np.uint8)

        for label_idx in np.unique(gt):
            if label_idx == 0:
                continue
            gt_vis[gt == label_idx] = colors[label_idx % len(colors)]

        for label_idx in np.unique(pred):
            if label_idx == 0:
                continue
            pred_vis[pred == label_idx] = colors[label_idx % len(colors)]

        # Plot
        fig, axs = plt.subplots(1, 4, figsize=(22, 6))

        axs[0].imshow(img)
        axs[0].set_title("RGB Image")
        axs[0].axis("off")

        axs[1].imshow(depth_np, cmap="plasma")
        axs[1].set_title("Depth Map")
        axs[1].axis("off")

        axs[2].imshow(gt_vis)
        axs[2].set_title("Ground Truth Seg")
        axs[2].axis("off")

        axs[3].imshow(pred_vis)
        axs[3].set_title("Predicted Seg")
        axs[3].axis("off")

        # Overlay labels for GT
        for uid in np.unique(gt):
            if uid == 0 or uid >= len(candidate_labels):
                continue
            ys, xs = np.where(gt == uid)
            if len(xs) == 0:
                continue
            x_c, y_c = int(xs.mean()), int(ys.mean())
            axs[2].text(
                x_c, y_c, candidate_labels[uid],
                color="white", fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
            )

        # Overlay labels for Prediction
        for uid in np.unique(pred):
            if uid == 0 or uid >= len(candidate_labels):
                continue
            ys, xs = np.where(pred == uid)
            if len(xs) == 0:
                continue
            x_c, y_c = int(xs.mean()), int(ys.mean())
            axs[3].text(
                x_c, y_c, candidate_labels[uid],
                color="white", fontsize=8, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i}.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
def visualize_tensorboard_image(depth, image, seg_gt, seg_pred, candidate_labels):
    """
    Returns a grid image visualizing RGB, depth, GT and predicted segmentation maps
    for a single sample. All inputs should be torch.Tensors.
    """
    

    B = image.shape[0]
    depth = depth.cpu()
    image = image.cpu()
    seg_gt = seg_gt.cpu()
    seg_pred = seg_pred.cpu()

    fig_images = []

    n_labels = len(candidate_labels)
    cmap = plt.get_cmap('gist_ncar', n_labels)
    colors = (cmap(np.arange(n_labels))[:, :3] * 255).astype(np.uint8)


    for i in range(B):
        img = image[i].numpy().transpose(1, 2, 0)
        img = img * np.array([0.2686, 0.2613, 0.2758]) + np.array([0.4815, 0.4578, 0.4082])
        img = np.clip(img, 0, 1)

        depth_np = depth[i][0].numpy()
        gt = seg_gt[i].numpy()
        pred = seg_pred[i].numpy()
        
        

        def colorize(seg):
            seg = seg.astype(np.int32)
            color_map = np.zeros((*seg.shape, 3), dtype=np.uint8)
            for label_idx in np.unique(seg):
                
                if label_idx == 0:
                    continue
                if label_idx < len(colors):
                    color_map[seg == label_idx] = colors[label_idx]
            return color_map

        gt_vis = colorize(gt)
        pred_vis = colorize(pred)

        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        axs[0].imshow(img)
        axs[0].set_title("RGB")
        axs[0].axis("off")

        axs[1].imshow(depth_np, cmap='plasma')
        axs[1].set_title("Depth")
        axs[1].axis("off")

        axs[2].imshow(gt_vis)
        axs[2].set_title("GT Seg")
        axs[2].axis("off")

        axs[3].imshow(pred_vis)
        axs[3].set_title("Pred Seg")
        axs[3].axis("off")
        
        # Get unique labels from GT and Pred
        unique_labels = np.unique(np.concatenate([gt.flatten(), pred.flatten()]))
        unique_labels = [l for l in unique_labels if l != 0]
        
        legend_y = 0.95
        legend_gap = 0.05

        for j, label in enumerate(unique_labels[:30]):
            y = legend_y - j * legend_gap
            if y < 0: break  # Avoid drawing below the plot
            axs[4].add_patch(plt.Rectangle((0.0, y), 0.1, 0.04, color=colors[label] / 255.0, transform=axs[4].transAxes))
            axs[4].text(0.15, y + 0.015, candidate_labels[label], fontsize=8, verticalalignment='center', transform=axs[4].transAxes)

        axs[4].axis("off")

        plt.subplots_adjust(wspace=0.05)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        fig_images.append(img_tensor)
        plt.close()

    return make_grid(fig_images, nrow=1)
