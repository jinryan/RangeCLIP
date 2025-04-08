import torch
import sys
import os
import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from utils.src.log_utils import log
from RangeCLIP.src.log import log_training_summary

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def validate_model(model,
                  candidate_labels,
                  dataloader,
                  step,
                  best_results,
                  device,
                  summary_writer,
                  n_sample_per_summary=4,
                  log_path=None):
    """
    Validate the model on a dataset.
    
    Args:
        model: The DepthClipModel to validate
        candidate_labels: List of text labels for classification
        dataloader: DataLoader for validation data
        step: Current training step
        best_results: Dictionary containing best validation results so far
        device: Device to run validation on
        summary_writer: TensorBoard summary writer
        n_sample_per_summary: Number of samples to log to TensorBoard
        log_path: Path to log results to
        
    Returns:
        best_results: Updated best_results dictionary
        accuracy: Classification accuracy on validation set
    """
    print("Entering Validation")
    unwrap_model(model).eval()  # Set model to evaluation mode
    
    
    # Metrics for embedding quality
    total_loss = 0.0
    
    image_summary = []
    depth_summary = []
    text_summary = []
    
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():  # Disable gradient computation for validation
        for idx, batch in tqdm.tqdm(enumerate(dataloader), desc='Validation', total=len(dataloader)):
            # Unpack batch to device
            labeled_batch = batch['labeled']
            unlabeled_batch = batch['unlabeled']
            
            has_labeled_images = 'image' in labeled_batch and labeled_batch['image'].size(0) > 1
            has_unlabeled_images = 'image' in unlabeled_batch and unlabeled_batch['image'].size(0) > 1

            if not has_labeled_images and not has_unlabeled_images:
                print(f"[Validation] Warning: Empty batch at index {idx}. Skipping.")
                continue  # Skip this iteration
            
            if has_labeled_images:
                labeled_images = labeled_batch['image'].to(device)
                labeled_depths = labeled_batch['depth'].to(device)
                labeled_ground_truth_indices = labeled_batch['id'].to(device)
                
                for i in range(len(labeled_depths)):
                    single_depth = labeled_depths[i:i+1]  # Shape: [1, 1, H, W]
                    predicted_class, confidence = unwrap_model(model).predict(single_depth.to(device), candidate_labels)

                    true_class = labeled_ground_truth_indices[i].item()

                    if predicted_class == true_class:
                        correct_predictions += 1

                    total_predictions += 1
                    
                # Compute embeddings
                depth_embed_l, _, image_embed_l = model.forward(
                    depth_map=labeled_depths,
                    image=labeled_images
                )

                labeled_loss, loss_info_labeled = unwrap_model(model).compute_trimodal_loss(
                    depth_embeddings=depth_embed_l,
                    image_embeddings=image_embed_l,
                    candidate_labels=candidate_labels,
                    ground_truth_indices=labeled_ground_truth_indices,
                    w_text=0.8
                )
            else:
                labeled_loss = 0
                loss_info_labeled = {}
                print(f"[Validation] No labeled data in batch at index {idx}. Skipping labeled loss.")

                
            if has_unlabeled_images:
                unlabeled_images = unlabeled_batch['image'].to(device)
                unlabeled_depths = unlabeled_batch['depth'].to(device)


                # Repeat RGB image for each patch

                depth_embed_u, _, image_embed_u = model.forward(
                    depth_map=unlabeled_depths,
                    image=unlabeled_images
                )

                unsup_loss, loss_info_unsup = unwrap_model(model).compute_bimodal_loss(
                    depth_embeddings=depth_embed_u,
                    other_embeddings=image_embed_u
                )
            else:
                unsup_loss = 0
                loss_info_unsup = {}
                
            # Combine losses
            if labeled_loss != 0 and unsup_loss != 0:
                loss = labeled_loss + unsup_loss
            else:
                loss = labeled_loss or unsup_loss  # whichever is nonzero
            
            loss_info = {**loss_info_labeled, **loss_info_unsup}
            total_loss += loss.item()

            
            # Collect samples for TensorBoard display
            if (idx % max(1, len(dataloader) // n_sample_per_summary)) == 0 and summary_writer is not None:
                n = min(len(labeled_images), n_sample_per_summary)

                # Collect image and depth examples
                image_summary.append(labeled_images[:n].cpu())
                depth_summary.append(labeled_depths[:n].cpu())

                batch_texts = [candidate_labels[i] for i in labeled_ground_truth_indices[:n]]

                text_summary.extend(batch_texts)
    
    # Compute average metrics
    avg_loss = total_loss / len(dataloader)
    val_accuracy = correct_predictions / (total_predictions + 1e-8)
    print(f"[Validation] Accuracy: {val_accuracy:.4f}")
    summary_writer.add_scalar("val/accuracy", val_accuracy, global_step=step)
    
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
        log('{:>8}  {:>8}  {:>10}'.format(
            'Step', 'Loss', 'Accuracy'),
            log_path)
        log('{:8}  {:8.4f}  {:10.4f}'.format(
            step, avg_loss, val_accuracy),
            log_path)
    
    # Check if model improved
    improved = False
    
    # For initial run when best_results might be empty
    if 'accuracy' not in best_results or best_results['accuracy'] < val_accuracy:
        improved = True
    
    # Update best results if improved
    if improved:
        best_results['step'] = step
        best_results['loss'] = avg_loss
        best_results['accuracy'] = val_accuracy
    
    if log_path:
        log('Best results:', log_path)
        log('{:>8}  {:>8}  {:>10}'.format(
            'Step', 'Loss', 'Accuracy'),
            log_path)
        log('{:8}  {:8.4f}  {:10.4f}'.format(
            best_results['step'],
            best_results['loss'],
            best_results['accuracy']), log_path)

    
    return best_results, val_accuracy

