import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
import os
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.depth_segmentation_model.log import log_validation_summary
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import ImageDraw, ImageFont
import matplotlib.cm as cm
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
    candidate_labels,
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
    

    with torch.no_grad():
        print(f"[Step {step}] Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB, Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

        candidate_text_embeddings = unwrap_model(model).text_encoder(candidate_labels).to(device)
        for batch in tqdm.tqdm(dataloader, desc=f'Validation Step {step}'):

            depth = batch['depth'].to(device)        # [B, 1, H, W]
            segmentation = batch['segmentation'].to(device)  # [B, H, W]
            
            log_gpu_usage("Before forward")
            # Forward pass
            output = model(depth)                       # [B, D, H, W]
            log_gpu_usage("After forward")

            # Compute loss
            torch.cuda.empty_cache()
            log_gpu_usage("Empty cache before loss")
            loss, loss_info = unwrap_model(model).compute_loss(
                output, segmentation, candidate_text_embeddings
            )
            log_gpu_usage("After loss computation")

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

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