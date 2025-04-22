import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
import os
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.depth_segmentation_model.log import log_validation_summary, visualize_tensorboard_image
from torchvision.transforms.functional import to_pil_image
import numpy as np
from torch.cuda.amp import autocast
from collections import defaultdict


def log_gpu_usage(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{tag}] Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")

def unwrap_model(model):
    """
    Unwraps the model if it is wrapped in DataParallel or DistributedDataParallel.
    """
    if hasattr(model, 'module'):
        return model.module
    return model



def validate_model(
    model,
    candidate_text_embeddings,
    candidate_labels,
    equivalence_tensor,
    equiv_class_map,
    similarity_sets,
    curriculum,
    dataloader,
    step,
    best_results,
    device,
    summary_writer=None,
    n_sample_per_summary=16,
    log_path=None
):
    model.eval()
    torch.cuda.empty_cache()

    # === Setup ===
    total_loss = total_contrastive_loss = total_smoothness_loss = n_batches = 0
    num_predicted = 0
    correct_pixels = 0
    total_pixels = 0
    intersection = defaultdict(int)
    union = defaultdict(int)

    with torch.no_grad(), autocast():
        for batch in tqdm.tqdm(dataloader, desc=f'Validation Step {step}'):
            # === Load and move batch to device ===
            depth = batch['depth'].to(device)               # [B, 1, H, W]
            segmentation = batch['segmentation'].to(device) # [B, H, W]
            
            # === Forward pass ===
            pred_segmentation, output, temperature = unwrap_model(model).predict(
                depth_maps=depth,
                candidate_text_embeddings=candidate_text_embeddings,
                segmentation=segmentation,
                num_negatives=50,
            )

            # === Flatten predictions and ground truth ===
            gt_flat = segmentation.view(-1)
            pred_flat = pred_segmentation.view(-1)

            # === Pixel accuracy using equivalence ===
            correct_mask = equivalence_tensor[gt_flat, pred_flat]
            correct_pixels += correct_mask.sum().item()
            total_pixels += correct_mask.numel()

            # === mIoU using equivalence-mapped labels ===
            gt_equiv = equiv_class_map[gt_flat]
            pred_equiv = equiv_class_map[pred_flat]
            unique_equiv_labels = torch.unique(torch.cat([gt_equiv, pred_equiv]))

            for label in unique_equiv_labels:
                label = label.item()
                pred_mask = pred_equiv == label
                gt_mask = gt_equiv == label
                intersection[label] += torch.logical_and(pred_mask, gt_mask).sum().item()
                union[label] += torch.logical_or(pred_mask, gt_mask).sum().item()

            # === Qualitative Logging ===
            if num_predicted < n_sample_per_summary and summary_writer is not None:
                image = batch['image'].to(device)
                grid_image = visualize_tensorboard_image(
                    depth, image, segmentation, pred_segmentation, candidate_labels
                )
                summary_writer.add_image(f"val/qualitative_preds/{num_predicted}", grid_image, global_step=step)

            # === Compute loss ===
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
            total_loss += loss_info['total_loss']
            total_contrastive_loss += loss_info['contrastive_loss']
            total_smoothness_loss += loss_info['smoothness_loss']
            n_batches += 1

            # === Cleanup ===
            del output, loss, pred_segmentation, loss_info
            torch.cuda.empty_cache()
            num_predicted += len(batch['depth'])

    # === Final Metrics ===
    avg_loss = total_loss / max(n_batches, 1)
    avg_contrastive_loss = total_contrastive_loss / max(n_batches, 1)
    avg_smoothness_loss = total_smoothness_loss / max(n_batches, 1)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    ious = [intersection[l] / union[l] for l in union if union[l] > 0]
    miou = sum(ious) / len(ious) if ious else 0.0

    # === Logging ===
    log(f"[Step {step}] Validation pixel accuracy: {pixel_accuracy:.4f}", log_path)
    log(f"[Step {step}] Validation mIoU (w/ equivalence): {miou:.4f}", log_path)
    log(f"[Val] Step {step} | Loss: {avg_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f}", log_path)

    # === Best Results Tracking ===
    if best_results.get("loss", float('inf')) > avg_loss:
        best_results["loss"] = avg_loss
        best_results["step"] = step
    log(f"Best validation loss: {best_results['loss']:.4f} at step {best_results['step']}", log_path)
    
    # === TensorBoard Summary ===
    if summary_writer is not None:
        summary_writer.add_scalar("val/loss", avg_loss, global_step=step)
        scalar_info = {}
        scalar_info['avg_loss'] = avg_loss
        scalar_info['pixel_accuracy'] = pixel_accuracy
        scalar_info['mIoU'] = miou
        scalar_info['contrastive_loss'] = avg_contrastive_loss
        scalar_info['smoothness_loss'] = avg_smoothness_loss
        scalar_info['temperature'] = temperature
        log_validation_summary(
            summary_writer,
            step,
            scalar_info,
        )

    return best_results
