import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
import os
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.log import log_training_summary
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import ImageDraw, ImageFont
import matplotlib.cm as cm
from collections import defaultdict

def apply_colormap_to_depth(depth_tensor, colormap='viridis'):
    """
    Takes a [1, H, W] tensor and returns a [3, H, W] colorized tensor in RGB.
    More efficient implementation with fewer conversions.
    """
    # Move to CPU once at the beginning if needed
    if depth_tensor.device.type != 'cpu':
        depth_tensor = depth_tensor.cpu()
    
    # Get dimensions directly from tensor
    depth = depth_tensor.squeeze().numpy()  # [H, W]
    
    # Handle edge case of constant depth
    depth_range = depth.max() - depth.min()
    if depth_range < 1e-8:
        depth_norm = np.zeros_like(depth)
    else:
        depth_norm = (depth - depth.min()) / depth_range
    
    # Apply colormap in one operation
    cmap = cm.get_cmap(colormap)
    depth_colored = cmap(depth_norm)[:, :, :3]  # Drop alpha channel
    
    # Convert to tensor in one step with correct scaling
    depth_tensor_color = torch.from_numpy(depth_colored.transpose(2, 0, 1)).float()  # [3, H, W]
    
    return depth_tensor_color

def log_failed_samples_to_tensorboard(writer, step, failed_images, failed_depths, text_summaries, max_log=4):
    """
    Logs a batch of failed predictions to TensorBoard as composite depth-image pairs with text overlays.
    """
    n_samples = min(len(text_summaries), max_log)
    font = ImageFont.load_default()
    
    # Create a batch of images to log at once
    composite_grid = []
    
    for i in range(n_samples):
        image = failed_images[i]           # Tensor [3, H, W]
        depth = failed_depths[i]           # Tensor [1, H, W]
        text = text_summaries[i]           # Formatted string: "GT: ... | Pred: ..."
        
        # Apply colormap
        depth_vis = apply_colormap_to_depth(depth, colormap='viridis')
        
        # Concatenate depth and image horizontally
        combined = torch.cat([depth_vis, image], dim=2)  # [3, H, 2W]
        
        # Convert to PIL and draw text
        grid_img = to_pil_image(combined.cpu())
        draw = ImageDraw.Draw(grid_img)
        draw.text((5, 5), text, font=font, fill=(255, 255, 255))
        
        # Convert PIL image to tensor
        to_tensor = transforms.ToTensor()

        img_tensor = to_tensor(grid_img)  # [3, H, W], automatically normalized to [0, 1]
        
        composite_grid.append(img_tensor)
    
    # Combine all images into a grid and log once
    if composite_grid and writer is not None:
        all_samples = torch.stack(composite_grid)  # [N, 3, H, W]
        writer.add_images('failures/samples', all_samples, global_step=step)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def get_clip_baseline_performance(model,
                                  candidate_labels,
                                  dataloader,
                                  summary_writer,
                                  device,
                                  log_path=None):
    print("Getting CLIP Baseline Performance (image/text)")
    model.eval()

    total_loss = 0.0
    total_predictions = 0

    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    with torch.no_grad():
        text_encoder = model.get_text_encoder()
        image_encoder = model.get_image_encoder()
        
        text_embeddings = F.normalize(text_encoder(candidate_labels).to(device), dim=-1)

        gt_label_fail_counts = defaultdict(int)
        gt_label_total_counts = defaultdict(int)

        for batch in tqdm.tqdm(dataloader, desc='Baseline'):
            labeled_batch = batch['labeled']

            if 'image' not in labeled_batch or len(labeled_batch['image']) < 1:
                continue

            images = labeled_batch['image'].to(device)  # [B, C, H, W]
            gt_indices = labeled_batch['id'].to(device)  # [B]
            gt_indices_list = gt_indices.tolist()

            # ======= Batched Prediction =======
            
            image_embeddings = F.normalize(image_encoder(images), dim=-1) # [B, D]
            logits = torch.matmul(image_embeddings, text_embeddings.T)  # [B, N]
            probs = F.softmax(logits, dim=-1)

            top_k = 5
            topk_scores, topk_preds = torch.topk(probs, k=top_k, dim=-1)  # [B, 5]

            # ======= Accuracy Calculation =======
            top_1_correct += (topk_preds[:, 0] == gt_indices).sum().item()
            top_3_correct += sum(gt in pred[:3] for gt, pred in zip(gt_indices_list, topk_preds.tolist()))
            top_5_correct += sum(gt in pred[:5] for gt, pred in zip(gt_indices_list, topk_preds.tolist()))
            total_predictions += gt_indices.size(0)
            
            # Vectorized failure detection
            gt_expanded = gt_indices.unsqueeze(1).expand_as(topk_preds)
            failed_mask = ~(gt_expanded == topk_preds).any(dim=1)  # [B]

            failed_gt_indices = gt_indices[failed_mask]

            # Update failure counts
            for label_idx in failed_gt_indices.tolist():
                gt_label_fail_counts[label_idx] += 1

            for label_idx in gt_indices_list:
                gt_label_total_counts[label_idx] += 1

            # ======= Loss Calculation =======
            selected_text_embeddings = text_embeddings[gt_indices]  # [B, D]

            bimodal_loss, _ = model.compute_bimodal_loss(
                embedding_1=image_embeddings,
                embedding_2=selected_text_embeddings
            )
            total_loss += bimodal_loss.item()

    # ======= Metrics =======
    avg_loss = total_loss / len(dataloader)
    val_top1_accuracy = top_1_correct / (total_predictions + 1e-8)
    val_top3_accuracy = top_3_correct / (total_predictions + 1e-8)
    val_top5_accuracy = top_5_correct / (total_predictions + 1e-8)

    # ======= Logging =======
    summary_writer.add_scalar("baseline/top1_accuracy", val_top1_accuracy, global_step=0)
    summary_writer.add_scalar("baseline/top3_accuracy", val_top3_accuracy, global_step=0)
    summary_writer.add_scalar("baseline/top5_accuracy", val_top5_accuracy, global_step=0)
    
    # ====== Failures ======
    # Convert counts to label names
    failed_class_stats = [(candidate_labels[i], count) for i, count in gt_label_fail_counts.items()]
    failed_class_stats.sort(key=lambda x: x[1], reverse=True)

    # Print top-k most failed labels
    log("\nTop Failed GT Classes (CLIP Baseline):", log_path)
    for label, count in failed_class_stats[:10]:
        msg = f"  {label}: {count} incorrect predictions"
        log(msg, log_path)
    
    for label_idx, total in gt_label_total_counts.items():
        print(f"{candidate_labels[label_idx]}: {total} total")

    # Log to TensorBoard
    for label_idx, fail_count in gt_label_fail_counts.items():
        total_count = gt_label_total_counts[label_idx]
        fail_rate = fail_count / total_count if total_count > 0 else 0
        summary_writer.add_scalar(f'baseline_failures/failure_rate/{candidate_labels[label_idx]}', fail_rate, global_step=0)


    if log_path:
        log('CLIP Baseline results:', log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Loss', 'Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc'), log_path)
        log('{:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}'.format(
            avg_loss, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy), log_path)


def validate_model(model,
                  candidate_labels,
                  dataloader,
                  step,
                  best_results,
                  device,
                  summary_writer,
                  n_sample_per_summary=4,
                  max_failed_log = 16,
                  log_path=None):
    print("Entering Validation")
    model.eval()
    
    
    # Metrics for embedding quality
    total_loss = 0.0
    
    image_summary = []
    depth_summary = []
    text_summary = []
    
    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0
    total_predictions = 0
    
    # Failures
    logged_failures = 0
    all_failed_images = []
    all_failed_depths = []
    all_failed_texts = []
    
    gt_label_fail_counts = defaultdict(int)
    
    with torch.no_grad():
        # FIX: Added .to(device) to text_embeddings
        text_encoder = model.get_text_encoder()
        text_embeddings = text_encoder(candidate_labels).to(device)
        # print("Text embedding mean:", text_embeddings.mean().item(), "std:", text_embeddings.std().item())

        
        for idx, batch in tqdm.tqdm(enumerate(dataloader), desc='Validation', total=len(dataloader)):
            labeled_batch = batch['labeled']
            unlabeled_batch = batch['unlabeled']

            has_labeled = 'image' in labeled_batch and len(labeled_batch['image']) > 0
            has_unlabeled = 'image' in unlabeled_batch and len(unlabeled_batch['image']) > 0

            if not has_labeled and not has_unlabeled:
                continue

            # Initialize loss components to zero (FIX: proper scalar/tensor initialization)
            supervised_loss = torch.tensor(0.0, device=device)
            unsupervised_loss = torch.tensor(0.0, device=device)
            loss_info = {}

            # Handle labeled data
            if has_labeled:
                # Move tensors to device once
                images = labeled_batch['image'].to(device)
                depths = labeled_batch['depth'].to(device)
                gt_indices = labeled_batch['id'].to(device)

                # Get predictions
                predicted_classes_batch, confidence_scores_batch = model.predict(
                    depth_map=depths, class_descriptions=candidate_labels
                )

                # Vectorized Top-1 accuracy
                top_1_correct += (predicted_classes_batch[:, 0] == gt_indices).sum().item()

                # Top-3 and Top-5 accuracy (vectorized)
                gt_indices_exp = gt_indices.view(-1, 1)
                in_top_3 = (gt_indices_exp == predicted_classes_batch[:, :3]).any(dim=1)
                in_top_5 = (gt_indices_exp == predicted_classes_batch[:, :5]).any(dim=1)
                top_3_correct += in_top_3.sum().item()
                top_5_correct += in_top_5.sum().item()
                total_predictions += gt_indices.size(0)

                # Identify failures (vectorized)
                failed_mask = ~in_top_5  # you could use top-1 or top-3 depending on what you define as a "fail"
                failed_indices = failed_mask.nonzero(as_tuple=False).squeeze()

                # Track failures by class
                if failed_indices.numel() > 0:
                    failed_gt_indices = gt_indices[failed_indices]
                    for idx in failed_gt_indices:
                        gt_label_fail_counts[idx.item()] += 1

                    # Logging
                    remaining = max_failed_log - logged_failures
                    if remaining > 0:
                        n_to_log = min(remaining, failed_indices.numel())

                        log_indices = failed_indices[:n_to_log]
                        failed_images = images[log_indices].cpu()
                        failed_depths = depths[log_indices].cpu()
                        failed_gt = gt_indices[log_indices]
                        failed_preds = predicted_classes_batch[log_indices]
                        failed_confs = confidence_scores_batch[log_indices]

                        all_failed_images.append(failed_images)
                        all_failed_depths.append(failed_depths)

                        for i in range(n_to_log):
                            gt = candidate_labels[failed_gt[i].item()]
                            preds = failed_preds[i].tolist()
                            confs = failed_confs[i].tolist()
                            guesses = [f"{candidate_labels[p]} ({conf:.2f})" for p, conf in zip(preds, confs)]
                            all_failed_texts.append(f"GT: {gt} | Pred: {', '.join(guesses)}")

                        logged_failures += n_to_log

                # Compute supervised loss
                d_emb, _, i_emb = model(depth_map=depths, image=images)
                sup_loss, sup_info = model.compute_trimodal_loss(
                    embedding_1=d_emb,
                    embedding_2=i_emb,
                    embedding_3=text_embeddings,
                    ground_truth_indices=gt_indices,
                    w_1_2=0.2
                )
                supervised_loss = sup_loss
                loss_info.update(sup_info)

            # Handle unlabeled data
            if has_unlabeled:
                u_images = unlabeled_batch['image'].to(device)
                u_depths = unlabeled_batch['depth'].to(device)

                d_emb_u, _, i_emb_u = model(depth_map=u_depths, image=u_images)
                unsup_loss, unsup_info = model.compute_bimodal_loss(
                    embedding_1=d_emb_u, embedding_2=i_emb_u
                )
                unsupervised_loss = unsup_loss
                loss_info.update(unsup_info)

            # Combine losses
            total_batch_loss = supervised_loss + unsupervised_loss
            total_loss += total_batch_loss.item()
    
    if all_failed_images and all_failed_depths and summary_writer is not None and logged_failures > 0:
        log_failed_samples_to_tensorboard(
            writer=summary_writer,
            step=step,
            failed_images=torch.cat(all_failed_images),
            failed_depths=torch.cat(all_failed_depths),
            text_summaries=all_failed_texts,
            max_log=max_failed_log
        )
    
    # Convert index to label string
    failed_class_stats = [
        (candidate_labels[i], count) for i, count in gt_label_fail_counts.items()
    ]

    # Sort by most failed
    failed_class_stats.sort(key=lambda x: x[1], reverse=True)

    # Log top 10 hardest classes to TensorBoard
    if summary_writer is not None:  # FIX: Check if summary_writer exists
        for i, (label, count) in enumerate(failed_class_stats[:10]):
            summary_writer.add_scalar(f'failures/by_class/{label}', count, global_step=step)

    # Optional: also print to console
    log("\nTop Failed GT Classes:", log_path)
    for label, count in failed_class_stats[:10]:
        log(f"  {label}: {count} incorrect predictions", log_path)

    # FIX: Handle the case where dataloader might be empty
    num_batches = len(dataloader)
    if num_batches == 0:
        log("Warning: Dataloader was empty during validation", log_path)
        return best_results

    # Compute average metrics
    avg_loss = total_loss / num_batches
    val_top1_accuracy = top_1_correct / (total_predictions + 1e-8)
    val_top3_accuracy = top_3_correct / (total_predictions + 1e-8)
    val_top5_accuracy = top_5_correct / (total_predictions + 1e-8)

    log(f"[Validation] Top-1 Accuracy: {val_top1_accuracy:.4f} | Top-3 Accuracy: {val_top3_accuracy:.4f} | Top-5 Accuracy: {val_top5_accuracy:.4f}", log_path)

    if summary_writer is not None:  # FIX: Check if summary_writer exists
        summary_writer.add_scalar("val/top1_accuracy", val_top1_accuracy, global_step=step)
        summary_writer.add_scalar("val/top3_accuracy", val_top3_accuracy, global_step=step)
        summary_writer.add_scalar("val/top5_accuracy", val_top5_accuracy, global_step=step)
    
    # Log to TensorBoard
    # Concatenate for logging
    if image_summary and depth_summary and summary_writer is not None:  # FIX: Check if everything exists
        summary_images = torch.cat(image_summary, dim=0)
        summary_depths = torch.cat(depth_summary, dim=0)
        summary_texts = text_summary

        log_training_summary(
            train_summary_writer=summary_writer,
            train_step=step,
            labeled_images=summary_images,
            labeled_depths=summary_depths,
            labeled_ground_truth_indices=summary_texts,
            unlabeled_images=None,
            unlabeled_depths=None,
            loss_info=loss_info,
            n_sample_per_summary=n_sample_per_summary
        )
        
    # Print validation results to console
    if log_path:
        log('Validation results:', log_path)
        log('{:>8}  {:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Step', 'Loss', 'Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc'),
            log_path)
        log('{:8}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}'.format(
            step, avg_loss, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy),
            log_path)

    # FIX: Initialize best_results if empty
    if not best_results:
        best_results = {
            'step': step,
            'loss': avg_loss,
            'top1_accuracy': val_top1_accuracy,
            'top3_accuracy': val_top3_accuracy,
            'top5_accuracy': val_top5_accuracy
        }
        improved = True
    else:
        # Determine if this is the best result so far (based on top-5 accuracy)
        improved = best_results['top5_accuracy'] < val_top5_accuracy

        # Update best results if improved
        if improved:
            best_results['step'] = step
            best_results['loss'] = avg_loss
            best_results['top1_accuracy'] = val_top1_accuracy
            best_results['top3_accuracy'] = val_top3_accuracy
            best_results['top5_accuracy'] = val_top5_accuracy

    # Logging best results
    if log_path:
        log('Best results:', log_path)
        log('{:>8}  {:>10}  {:>12}  {:>12}  {:>12}'.format(
            'Step', 'Loss', 'Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc'),
            log_path)
        log('{:8}  {:10.4f}  {:12.4f}  {:12.4f}  {:12.4f}'.format(
            best_results['step'],
            best_results['loss'],
            best_results['top1_accuracy'],
            best_results['top3_accuracy'],
            best_results['top5_accuracy']), log_path)
        
    return best_results