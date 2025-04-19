import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
import os
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.depth_segmentation_model.log import log_validation_summary
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import ImageDraw, ImageFont
import matplotlib.cm as cm
from collections import defaultdict

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

    # Precompute candidate text embeddings
    candidate_text_embeddings = model.text_encoder(candidate_labels).to(device)

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc=f'Validation Step {step}'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            depth = batch['depth']                      # [B, 1, H, W]
            image = batch['image']                      # e.g., RGB images
            segmentation = batch['segmentation']        # [B, 1, H, W] or [B, H, W]
            unique_labels = batch['unique_labels']      # Not used here, but maybe for logging

            # Forward pass
            output = model(depth)                       # [B, D, H, W]

            # Compute loss
            loss, loss_info = model.compute_loss(
                output, segmentation, candidate_text_embeddings
            )

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