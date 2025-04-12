import torch
import torch.nn.functional as F
import sys
import os
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.log import log_training_summary
from torchvision.transforms.functional import to_pil_image
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
        buf = io.BytesIO()
        grid_img.save(buf, format='PNG')
        buf.seek(0)
        img_np = np.array(Image.open(buf))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        
        composite_grid.append(img_tensor)
    
    # Combine all images into a grid and log once
    if composite_grid:
        all_samples = torch.stack(composite_grid)  # [N, 3, H, W]
        writer.add_images('failures/samples', all_samples, global_step=step)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def get_clip_baseline_performance(_model,
                                  candidate_labels,
                                  dataloader,
                                  summary_writer,
                                  device,
                                  log_path=None):
    print("Getting CLIP Baseline Performance (image/text)")
    model = unwrap_model(_model)
    model.eval()

    total_loss = 0.0
    total_predictions = 0

    top_1_correct = 0
    top_3_correct = 0
    top_5_correct = 0

    with torch.no_grad():
        text_embeddings = model.get_text_encoder(candidate_labels).to(device)
        gt_label_fail_counts = defaultdict(int)
        gt_label_total_counts = defaultdict(int)

        for batch in tqdm.tqdm(dataloader, desc='Baseline'):
            labeled_batch = batch['labeled']

            if 'image' not in labeled_batch or labeled_batch['image'].size(0) < 1:
                continue

            images = labeled_batch['image'].to(device)  # [B, C, H, W]
            gt_indices = labeled_batch['id'].to(device)  # [B]
            gt_indices_list = gt_indices.tolist()

            # ======= Batched Prediction =======
            image_embeddings = model.get_image_encoder(images)  # [B, D]
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
    print("\nTop Failed GT Classes (CLIP Baseline):")
    for label, count in failed_class_stats[:10]:
        msg = f"  {label}: {count} incorrect predictions"
        log(msg, log_path)
        print(msg)


    # Log to TensorBoard
    for label_idx, fail_count in gt_label_fail_counts.items():
        total_count = gt_label_total_counts[label_idx]
        fail_rate = fail_count / total_count
        summary_writer.add_scalar(f'baseline_failures/failure_rate/{candidate_labels[label_idx]}', fail_rate, global_step=0)


    if log_path:
        log('CLIP Baseline results:', log_path)
        log('{:>10}  {:>10}  {:>10}  {:>10}'.format(
            'Loss', 'Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc'), log_path)
        log('{:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}'.format(
            avg_loss, val_top1_accuracy, val_top3_accuracy, val_top5_accuracy), log_path)


def validate_model(_model,
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
    model = unwrap_model(_model)
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
    with torch.no_grad():
        text_embeddings = model.get_text_encoder(candidate_labels)
        # Outside your main dataloader loop
        
        logged_failures = 0
        all_failed_images = []
        all_failed_depths = []
        all_failed_texts = []
        
        gt_label_fail_counts = defaultdict(int)

        for idx, batch in tqdm.tqdm(enumerate(dataloader), desc='Validation', total=len(dataloader)):
            labeled_batch = batch['labeled']
            unlabeled_batch = batch['unlabeled']

            has_labeled = 'image' in labeled_batch and labeled_batch['image'].size(0) > 0
            has_unlabeled = 'image' in unlabeled_batch and unlabeled_batch['image'].size(0) > 0

            if not has_labeled and not has_unlabeled:
                continue

            # Handle labeled data
            if has_labeled:
                images = labeled_batch['image'].to(device)
                depths = labeled_batch['depth'].to(device)
                gt_indices = labeled_batch['id'].to(device)

                # BATCHED PREDICTION
                predicted_classes_batch, confidence_scores_batch = model.predict(depth_map=depths, class_descriptions=candidate_labels)

                # Accuracy metrics
                top_1_correct += (predicted_classes_batch[:, 0] == gt_indices).sum().item()
                top_3_correct += sum(gt in pred[:3] for gt, pred in zip(gt_indices, predicted_classes_batch))
                top_5_correct += sum(gt in pred[:5] for gt, pred in zip(gt_indices, predicted_classes_batch))
                total_predictions += len(gt_indices)

                # --------- Identify Failed Predictions (vectorized) ----------
                # Convert predicted_classes_batch to set for fast lookup per GT
                gt_expanded = gt_indices.unsqueeze(1).expand_as(predicted_classes_batch)
                failed_mask = ~(gt_expanded == predicted_classes_batch).any(dim=1)  # [B]

                n_failed = failed_mask.sum().item()
                remaining = max_failed_log - logged_failures
                if remaining > 0 and n_failed > 0:
                    n_to_log = min(n_failed, remaining)

                    # Use only the first n_to_log failed samples
                    failed_indices = failed_mask.nonzero(as_tuple=False).view(-1)[:n_to_log]

                    failed_images = images[failed_indices]
                    failed_depths = depths[failed_indices]
                    failed_gt_indices = gt_indices[failed_indices]
                    failed_preds = predicted_classes_batch[failed_indices]
                    failed_confs = confidence_scores_batch[failed_indices]
                    
                    for label_idx in failed_gt_indices.tolist():
                        gt_label_fail_counts[label_idx] += 1

                    # Append to log pool
                    all_failed_images.append(failed_images.cpu())
                    all_failed_depths.append(failed_depths.cpu())

                    for i in range(n_to_log):
                        gt = candidate_labels[failed_gt_indices[i].item()]
                        preds = failed_preds[i].tolist()
                        confs = failed_confs[i].tolist()
                        guesses = [f"{candidate_labels[p]} ({conf:.2f})" for p, conf in zip(preds, confs)]
                        formatted = f"GT: {gt} | Pred: {', '.join(guesses)}"
                        all_failed_texts.append(formatted)

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
            else:
                sup_loss, sup_info = 0, {}

            # Handle unlabeled data
            if has_unlabeled:
                u_images = unlabeled_batch['image'].to(device)
                u_depths = unlabeled_batch['depth'].to(device)

                d_emb_u, _, i_emb_u = model(depth_map=u_depths, image=u_images)
                unsup_loss, unsup_info = model.compute_bimodal_loss(
                    embedding_1=d_emb_u, embedding_2=i_emb_u
                )
            else:
                unsup_loss, unsup_info = 0, {}

            # Combine losses
            depth_image_loss = 0.0
            supervised_loss = sup_loss if has_labeled else 0.0
            unsupervised_loss = unsup_loss if has_unlabeled else 0.0
            total_batch_loss = supervised_loss + unsupervised_loss

            total_loss += total_batch_loss.item()
            loss_info = {**sup_info, **unsup_info}
    
    if summary_writer is not None and logged_failures > 0:
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
    for i, (label, count) in enumerate(failed_class_stats[:10]):
        summary_writer.add_scalar(f'failures/by_class/{label}', count, global_step=step)

    # Optional: also print to console
    log("\nTop Failed GT Classes:", log_path)
    for label, count in failed_class_stats[:10]:
        log(f"  {label}: {count} incorrect predictions", log_path)


    # Compute average metrics
    avg_loss = total_loss / len(dataloader)
    val_top1_accuracy = top_1_correct / (total_predictions + 1e-8)
    val_top3_accuracy = top_3_correct / (total_predictions + 1e-8)
    val_top5_accuracy = top_5_correct / (total_predictions + 1e-8)

    log(f"[Validation] Top-1 Accuracy: {val_top1_accuracy:.4f} | Top-3 Accuracy: {val_top3_accuracy:.4f} | Top-5 Accuracy: {val_top5_accuracy:.4f}", log_path)

    summary_writer.add_scalar("val/top1_accuracy", val_top1_accuracy, global_step=step)
    summary_writer.add_scalar("val/top3_accuracy", val_top3_accuracy, global_step=step)
    summary_writer.add_scalar("val/top5_accuracy", val_top5_accuracy, global_step=step)

    
    # Log to TensorBoard
    # Concatenate for logging
    if image_summary and depth_summary:
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

    # For initial run when best_results might be empty
    # Determine if this is the best result so far (based on top-5 accuracy)
    improved = 'top5_accuracy' not in best_results or \
                best_results['top5_accuracy'] < val_top5_accuracy

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

