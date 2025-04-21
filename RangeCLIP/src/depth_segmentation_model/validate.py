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
    dataloader,
    step,
    best_results,
    device,
    summary_writer=None,
    n_sample_per_summary=8,
    max_failed_log=16,
    log_path=None
):
    model.eval()

    total_loss = 0.0
    n_batches = 0
    
    num_predicted = 0
    torch.cuda.empty_cache()
    with torch.no_grad(), autocast():
        # print(f"[Step {step}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB, Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        correct_pixels = 0
        total_pixels = 0

        for batch in tqdm.tqdm(dataloader, desc=f'Validation Step {step}'):
            
            
            depth = batch['depth'].to(device)        # [B, 1, H, W]
            segmentation = batch['segmentation'].to(device)  # [B, H, W]
            # print(f"[Step {step}] Depth shape: {depth.shape}, Segmentation shape: {segmentation.shape}")
            
            pred_segmentation, output, temperature = unwrap_model(model).predict(
                depth_maps=depth,
                candidate_text_embeddings=candidate_text_embeddings,
                segmentation=segmentation,  # required for extracting true labels
                num_negatives=300  # can tune based on memory
            )
            
            # log_gpu_usage("After forward")
            gt_flat = segmentation.view(-1)                     # shape [BHW]
            pred_flat = pred_segmentation.view(-1)              # shape [BHW]

            # Vectorized batch lookup
            correct_mask = equivalence_tensor[gt_flat, pred_flat]  # shape [BHW], bool
            correct_pixels += correct_mask.sum().item()
            total_pixels += correct_mask.numel()


            
            if num_predicted < n_sample_per_summary and summary_writer is not None:
                image = batch['image'].to(device)        # [B, 3, H, W]
                grid_image = visualize_tensorboard_image(
                    depth, image, segmentation, pred_segmentation, candidate_labels
                )
                summary_writer.add_image(f"val/qualitative_preds/{num_predicted}", grid_image, global_step=step)


            # Compute loss
            loss, loss_info = unwrap_model(model).compute_loss(
                output, segmentation, candidate_text_embeddings, temperature
            )
            # log_gpu_usage("After loss computation")

            total_loss += loss.item()
            n_batches += 1
            
            del output, loss, pred_segmentation
            torch.cuda.empty_cache()
            
            num_predicted += len(batch['depth'])
            # log_gpu_usage("After batch cleanup")


    avg_loss = total_loss / max(n_batches, 1)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    log(f"[Step {step}] Validation pixel accuracy: {pixel_accuracy:.4f}", log_path)


    # Optionally log to summary writer
    if summary_writer is not None:
        summary_writer.add_scalar("val/loss", avg_loss, global_step=step)
        log_validation_summary(
            summary_writer,
            step,
            loss_info,
        )
    # Save best validation loss
    if best_results.get("loss", float('inf')) > avg_loss:
        best_results["loss"] = avg_loss
        best_results["step"] = step

    log(f"[Val] Step {step} | Loss: {avg_loss:.4f}", log_path)
    log(f"Best validation loss: {best_results['loss']:.4f} at step {best_results['step']}", log_path)

        
    return best_results