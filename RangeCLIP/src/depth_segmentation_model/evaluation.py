import torch
import random
from transformers import CLIPModel
import numpy as np

class MajorityBaseline:
    def __init__(self, majority_label_index):
        self.majority_label_index = majority_label_index

    def predict(self, depth_maps, candidate_text_embeddings, segmentation=None, num_negatives=None):
        B, _, H, W = depth_maps.shape
        device = depth_maps.device
        # Predict majority class for all pixels
        pred_segmentation = torch.full((B, H, W), self.majority_label_index, dtype=torch.long, device=device)
        output = None
        temperature = None
        return pred_segmentation, output, temperature


class RandomWithNegativesBaseline:
    def __init__(self, candidate_labels, num_negatives=300):
        self.candidate_labels = candidate_labels  # list of all label indices
        self.num_negatives = num_negatives

    def predict(self, depth_maps, candidate_text_embeddings, segmentation, num_negatives=None):
        """
        Predicts random labels from a reduced candidate set (ground truth + sampled negatives).
        Returns label indices in the original index space.
        """
        B, _, H, W = depth_maps.shape
        device = depth_maps.device

        if segmentation is None:
            raise ValueError("segmentation must be provided to extract true label indices.")

        batch_preds = []

        for b in range(B):
            gt_labels = torch.unique(segmentation[b]).tolist()
            all_labels = set(range(len(self.candidate_labels)))
            negatives_pool = list(all_labels - set(gt_labels))

            k = self.num_negatives if num_negatives is None else num_negatives
            sampled_negatives = random.sample(negatives_pool, min(k, len(negatives_pool)))

            reduced_set = sorted(list(set(gt_labels + sampled_negatives)))

            # Predict: randomly sample from reduced set
            pred = torch.randint(
                low=0,
                high=len(reduced_set),
                size=(H, W),
                device=device
            )
            pred_mapped = torch.tensor(reduced_set, device=device)[pred]  # map to original label space
            batch_preds.append(pred_mapped)

        pred_segmentation = torch.stack(batch_preds, dim=0)  # [B, H, W]
        output = None
        temperature = None
        return pred_segmentation, output, temperature


import torch
import tqdm
from utils.src.log_utils import log  # or just use print()

def evaluate_majority_model(
    dataloader,
    majority_label_index,
    equivalence_tensor=None,  # optional
    log_path=None,
    device="cuda"
):
    baseline_model = MajorityBaseline(majority_label_index)
    baseline_model_device = torch.device(device)
    
    correct_pixels = 0
    total_pixels = 0
    
    baseline_model = baseline_model  # no .to() needed, it's not a real model
    torch.cuda.empty_cache()

    for batch in tqdm.tqdm(dataloader, desc="Evaluating Majority Baseline"):
        depth_maps = batch["depth"].to(baseline_model_device)  # just for shape
        segmentation = batch["segmentation"].to(baseline_model_device)  # ground truth

        pred_segmentation, _, _ = baseline_model.predict(
            depth_maps=depth_maps,
            candidate_text_embeddings=None,
            segmentation=segmentation
        )

        # Flatten predictions and labels
        gt_flat = segmentation.view(-1)
        pred_flat = pred_segmentation.view(-1)

        if equivalence_tensor is not None:
            correct_mask = equivalence_tensor[gt_flat, pred_flat]
            correct_pixels += correct_mask.sum().item()
        else:
            correct_pixels += (gt_flat == pred_flat).sum().item()

        total_pixels += gt_flat.numel()

    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    log(f"[Majority Baseline] Pixel accuracy: {pixel_accuracy:.4f}", log_path)

    return pixel_accuracy

def evaluate_random_model(
    dataloader,
    num_candidate_labels,
    num_negatives=300,
    equivalence_tensor=None,
    log_path=None,
    device="cuda"
):
    random_baseline = RandomWithNegativesBaseline(
        candidate_labels=list(range(num_candidate_labels)),
        num_negatives=num_negatives
    )
    device = torch.device(device)

    correct_pixels = 0
    total_pixels = 0

    torch.cuda.empty_cache()

    for batch in tqdm.tqdm(dataloader, desc="Evaluating Random Baseline"):
        depth_maps = batch["depth"].to(device)
        segmentation = batch["segmentation"].to(device)

        pred_segmentation, _, _ = random_baseline.predict(
            depth_maps=depth_maps,
            candidate_text_embeddings=None,
            segmentation=segmentation
        )

        gt_flat = segmentation.view(-1)
        pred_flat = pred_segmentation.view(-1)

        if equivalence_tensor is not None:
            correct_mask = equivalence_tensor[gt_flat, pred_flat]
            correct_pixels += correct_mask.sum().item()
        else:
            correct_pixels += (gt_flat == pred_flat).sum().item()

        total_pixels += gt_flat.numel()

    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    log(f"[Random Baseline] Pixel accuracy: {pixel_accuracy:.4f}", log_path)

    return pixel_accuracy

import torch
import torch.nn.functional as F
import tqdm
import random
from utils.src.log_utils import log  # or just use print()

def evaluate_mask_clip(
    dataloader,
    clip_model,
    candidate_text_embeddings,
    num_negatives=300,
    equivalence_tensor=None,
    log_path=None,
    device="cuda"
):
    device = torch.device(device)
    clip_model.eval()
    clip_model.to(device)

    total_labels = list(range(candidate_text_embeddings.shape[0]))  # [C]
    embedding_dim = candidate_text_embeddings.shape[1]

    correct_pixels = 0
    total_pixels = 0

    for batch in tqdm.tqdm(dataloader, desc="Evaluating MaskCLIP Baseline"):
        images = batch["image"].to(device)              # [B, 3, H, W]
        segmentation = batch["segmentation"].to(device) # [B, H, W]
        B, _, H, W = images.shape

        with torch.no_grad():
            # --- Step 1: Determine reduced label set (GT + distractors) ---
            unique_labels = torch.unique(segmentation).tolist()  # Ground truth labels
            distractor_pool = list(set(total_labels) - set(unique_labels))
            sampled_negatives = random.sample(distractor_pool, min(num_negatives, len(distractor_pool)))
            reduced_indices = sorted(list(set(unique_labels + sampled_negatives)))  # [C_reduced]

            reduced_text_embeddings = candidate_text_embeddings[reduced_indices].to(device)  # [C_reduced, D]
            reduced_text_embeddings = F.normalize(reduced_text_embeddings, dim=-1)

            # --- Step 2: Get visual patch embeddings ---
            vision_out = clip_model.vision_model(images)
            patch_feats = vision_out.last_hidden_state[:, 1:, :]  # [B, P, D_vit]

            # Project to shared embedding space using visual_projection
            patch_feats_proj = clip_model.visual_projection(patch_feats)  # [B, P, D_clip]
            patch_feats_proj = F.normalize(patch_feats_proj, dim=-1)


            B, P, D = patch_feats_proj.shape
            patch_h = patch_w = int(P ** 0.5)
            assert patch_h * patch_w == P, f"Non-square patch grid (P={P})"

            pixel_feats = patch_feats_proj.permute(0, 2, 1).reshape(B, D, patch_h, patch_w)  # [B, D, h, w]
            pixel_feats = F.interpolate(pixel_feats, size=(H, W), mode="bilinear", align_corners=False)  # [B, D, H, W]

            # --- Step 4: Cosine similarity and prediction ---
            pixel_feats = F.normalize(pixel_feats, dim=1)
            pixel_flat = pixel_feats.view(B, D, H * W)  # [B, D, N]
            logits = torch.einsum("bdn,cd->bcn", pixel_flat, reduced_text_embeddings)  # [B, C_reduced, N]
            pred_reduced = logits.argmax(dim=1).view(B, H, W)  # [B, H, W]

            # --- Step 5: Map back to original label indices ---
            index_tensor = torch.tensor(reduced_indices, device=device)  # [C_reduced]
            pred_segmentation = index_tensor[pred_reduced]  # [B, H, W]

        # --- Step 6: Accuracy computation ---
        gt_flat = segmentation.view(-1)
        pred_flat = pred_segmentation.view(-1)

        if equivalence_tensor is not None:
            correct_mask = equivalence_tensor[gt_flat, pred_flat]
            correct_pixels += correct_mask.sum().item()
        else:
            correct_pixels += (gt_flat == pred_flat).sum().item()

        total_pixels += gt_flat.numel()

    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    log(f"[MaskCLIP Baseline] Pixel accuracy (GT + {num_negatives} distractors): {pixel_accuracy:.4f}", log_path)

    return pixel_accuracy

import torch
import torch.nn.functional as F
import tqdm
import random
from utils.src.log_utils import log  # Replace with print if not using a logger

def evaluate_seg_former(
    dataloader,
    model,
    image_processor,
    candidate_labels,  # List of all class indices or label strings
    num_negatives=300,
    equivalence_tensor=None,
    log_path=None,
    device="cuda"
):
    model.eval()
    model.to(device)

    correct_pixels = 0
    total_pixels = 0
    all_class_indices = list(range(len(candidate_labels)))

    for batch in tqdm.tqdm(dataloader, desc="Evaluating SegFormer"):
        images = batch["image"].to(device)              # [B, 3, H, W]
        segmentation = batch["segmentation"].to(device) # [B, H, W]
        # Convert to numpy, move from CHW to HWC
        images_np = []
        for img in images:
            img = img.detach().cpu()
            if torch.isnan(img).any() or torch.isinf(img).any():
                print("⚠️ Skipping image with NaN or Inf")
                continue
            img = torch.clamp(img, 0.0, 1.0).permute(1, 2, 0).numpy()  # [H, W, C]
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            images_np.append(img)

        if len(images_np) == 0:
            continue  # skip batch entirely if all images are bad




        with torch.no_grad():
            # Preprocess images using the image processor
            inputs = image_processor(images=images_np, return_tensors="pt", do_rescale=False, size=(512, 512)).to(device)
            if inputs['pixel_values'].shape[0] == 0:
                continue
            
            outputs = model(**inputs)
            logits = outputs.logits  # [B, num_classes, H, W]

            # Resize logits to match the size of the segmentation masks
            logits = F.interpolate(logits, size=segmentation.shape[-2:], mode="bilinear", align_corners=False)

            # --- Restrict to GT + sampled distractors ---
            unique_labels = torch.unique(segmentation).tolist()
            distractors = list(set(all_class_indices) - set(unique_labels))
            sampled_distractors = random.sample(distractors, min(num_negatives, len(distractors)))
            reduced_indices = sorted(list(set(unique_labels + sampled_distractors)))  # final indices to use

            reduced_logits = logits[:, reduced_indices, :, :]  # [B, C_reduced, H, W]
            pred_reduced = reduced_logits.argmax(dim=1)        # indices in reduced space [B, H, W]

            # Map back to original label indices
            index_tensor = torch.tensor(reduced_indices, device=device)  # [C_reduced]
            preds = index_tensor[pred_reduced]  # [B, H, W]

        # Flatten predictions and labels
        gt_flat = segmentation.view(-1)
        pred_flat = preds.view(-1)

        if equivalence_tensor is not None:
            correct_mask = equivalence_tensor[gt_flat, pred_flat]
            correct_pixels += correct_mask.sum().item()
        else:
            correct_pixels += (gt_flat == pred_flat).sum().item()

        total_pixels += gt_flat.numel()

    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    log(f"[SegFormer Evaluation] Pixel accuracy (GT + {num_negatives} distractors): {pixel_accuracy:.4f}", log_path)

    return pixel_accuracy
