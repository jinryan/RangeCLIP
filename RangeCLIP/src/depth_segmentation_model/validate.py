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
    total_pixels = 0
    
    intersection_top1 = defaultdict(int)
    union_top1 = defaultdict(int)
    intersection_topk = defaultdict(int)
    union_topk = defaultdict(int)
    correct_pixels_top1 = 0
    correct_pixels_topk = 0
    total_pixels = 0

    with torch.no_grad(), autocast():
        for batch in tqdm.tqdm(dataloader, desc=f'Validation Step {step}'):
            # === Load and move batch to device ===
            depth = batch['depth'].to(device)               # [B, 1, H, W]
            segmentation = batch['segmentation'].to(device) # [B, H, W]
            
            # === Forward pass ===
            pred_topk, output, temperature = unwrap_model(model).predict(
                depth_maps=depth,
                candidate_text_embeddings=candidate_text_embeddings,
                segmentation=segmentation,
                num_negatives=50,
                top_k=5,  # set your k here
            )

            B, k, H, W = pred_topk.shape
            gt_flat = segmentation.view(-1)  # [B*H*W]
            
            pred_segmentation = pred_topk[:, 0]  # Top-1 prediction
            top1_flat = pred_segmentation.contiguous().view(-1)  # [B*H*W]
            topk_flat = pred_topk.permute(0, 2, 3, 1).reshape(-1, k)  # [B*H*W, k]

            # === Top-1 pixel accuracy using equivalence ===
            correct_top1_mask = equivalence_tensor[gt_flat, top1_flat]
            correct_pixels_top1 += correct_top1_mask.sum().item()
            total_pixels += correct_top1_mask.numel()

            # === Top-k pixel accuracy (any of top-k match the GT) ===
            # Expand gt to [B*H*W, 1] and compare with [B*H*W, k]
            gt_flat_exp = gt_flat.unsqueeze(1).expand_as(topk_flat)
            correct_topk_mask = equivalence_tensor[gt_flat_exp, topk_flat]  # [B*H*W, k]
            correct_pixels_topk += correct_topk_mask.any(dim=1).sum().item()

            # === mIoU using top-1 prediction (equivalence-mapped labels) ===
            gt_equiv = equiv_class_map[gt_flat]
            pred_equiv_top1 = equiv_class_map[top1_flat]
            unique_equiv_labels = torch.unique(torch.cat([gt_equiv, pred_equiv_top1]))

            for label in unique_equiv_labels:
                label = label.item()
                pred_mask = pred_equiv_top1 == label
                gt_mask = gt_equiv == label
                intersection_top1[label] += torch.logical_and(pred_mask, gt_mask).sum().item()
                union_top1[label] += torch.logical_or(pred_mask, gt_mask).sum().item()
                
            # === mIoU - Top-k ===
            # Step 1: Expand equivalence map to [N, k] to compare with all k predictions
            topk_equiv = equiv_class_map[topk_flat]  # [N, k]

            # Create an oracle prediction mask using top-k
            oracle_pred = top1_flat.clone()  # Start with top-1 predictions
            for label in unique_equiv_labels:
                label_val = label.item()
                gt_mask = (gt_equiv == label_val)
                
                # Find pixels where GT is this label AND any top-k prediction matches
                correct_pixels = gt_mask & (topk_equiv == label_val).any(dim=1)
                
                # Replace those pixels in oracle prediction
                oracle_pred[correct_pixels] = label_val

            # Then calculate standard IoU using oracle_pred
            for label in unique_equiv_labels:
                label_val = label.item()
                pred_mask = oracle_pred == label_val
                gt_mask = gt_equiv == label_val
                intersection_topk[label_val] += torch.logical_and(pred_mask, gt_mask).sum().item()
                union_topk[label_val] += torch.logical_or(pred_mask, gt_mask).sum().item()
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

    def compute_mIoU(intersection, union, valid_labels=None):
        ious = []
        for label in union:
            if valid_labels is not None and label not in valid_labels:
                continue
            if union[label] > 0:
                iou = intersection[label] / union[label]
                ious.append(iou)
        return sum(ious) / len(ious) if ious else 0.0


    # Equivalence-aware metrics
    gt_equiv_all = equiv_class_map[segmentation.view(-1)]
    valid_labels = set(gt_equiv_all.tolist())  # Or union of all GTs from all batches

    miou_top1 = compute_mIoU(intersection_top1, union_top1, valid_labels)
    miou_topk = compute_mIoU(intersection_topk, union_topk, valid_labels)


    pixel_acc_top1 = correct_pixels_top1 / total_pixels if total_pixels > 0 else 0.0
    pixel_acc_topk = correct_pixels_topk / total_pixels if total_pixels > 0 else 0.0

    # Loss values
    avg_loss = total_loss / max(n_batches, 1)
    avg_contrastive_loss = total_contrastive_loss / max(n_batches, 1)
    avg_smoothness_loss = total_smoothness_loss / max(n_batches, 1)


    # === Logging ===
    log(f"[Val] [Step {step}] Top-1 pixel accuracy (equiv): {pixel_acc_top1:.4f}", log_path)
    log(f"[Val] [Step {step}] Top-k pixel accuracy (equiv): {pixel_acc_topk:.4f}", log_path)
    log(f"[Val] [Step {step}] Top-1 mIoU (equiv): {miou_top1:.4f}", log_path)
    log(f"[Val] [Step {step}] Top-k mIoU (equiv): {miou_topk:.4f}", log_path)
    log(f"[Val] Step {step} | # of labels in Top-1 mIoU: {len(intersection_top1)}", log_path)
    log(f"[Val] Step {step} | # of labels in Top-k mIoU: {len(intersection_topk)}", log_path)

    log(f"[Val] Step {step} | Loss: {avg_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f}", log_path)

    # === Best Results Tracking ===
    if best_results.get("mIoU_tk", 0) < miou_topk:
        best_results["loss"] = avg_loss
        best_results["step"] = step
        best_results["mIoU_t1"] = miou_top1
        best_results["mIoU_tk"] = miou_topk
        best_results["pixel_accuracy_t1"] = pixel_acc_top1
        best_results["pixel_accuracy_tk"] = pixel_acc_topk
        best_results["temperature"] = temperature
        best_results["contrastive_loss"] = avg_contrastive_loss
        best_results["smoothness_loss"] = avg_smoothness_loss
    log(f"Best validation loss: {best_results['loss']:.4f} at step {best_results['step']}", log_path)
    
    # === TensorBoard Summary ===
    if summary_writer is not None:
        summary_writer.add_scalar("val/loss", avg_loss, global_step=step)
        scalar_info = {}
        scalar_info['avg_loss'] = avg_loss
        scalar_info['pixel_accuracy'] = pixel_acc_top1
        scalar_info["pixel_accuracy_tk"] = pixel_acc_topk
        scalar_info['mIoU'] = miou_top1
        scalar_info["mIoU_tk"] = miou_topk
        scalar_info['contrastive_loss'] = avg_contrastive_loss
        scalar_info['smoothness_loss'] = avg_smoothness_loss
        scalar_info['temperature'] = temperature
        log_validation_summary(
            summary_writer,
            step,
            scalar_info,
        )

    return best_results
