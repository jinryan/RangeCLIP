# CLIPSeg Evaluation Script with Top-K Metrics, Equivalence Classes, and Brightness Variation (Batch + Visualization)

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tqdm import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import ast
import warnings
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
warnings.filterwarnings("ignore", message="The following named arguments are not valid for .*ViTImageProcessor.*")

# --- CONFIGURATION ---
BASE_DIR = '/media/home/ryjin/depthclip/RangeCLIP/data/sunrgbd/SUNRGBD/'
METADATA_PATH = os.path.join(BASE_DIR, 'metadata.csv')
LABELS_PATH = os.path.join(BASE_DIR, 'candidate_labels.csv')
SIMILARITY_PATH = os.path.join(BASE_DIR, 'label_similarity_sets.csv')
NUM_DISTRACTORS = 20
K_FOR_TOPK = 5
BRIGHTNESS_LEVELS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
SATURATION_LEVELS = [1.0, 0.7, 0.5, 0.2, 0.1, 0.1, 0.05]
NUM_SAMPLES_TO_VISUALIZE = 10
NUM_SAMPLES_TO_EVALUATE = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_VIS_DIR = './benchmark/clipseg_visualizations/'
PLOT_OUTPUT_PATH_BASE = './benchmark/clipseg_brightness_plot'
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# --- HELPER FUNCTIONS ---
def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    idx_to_name = dict(zip(df['index'], df['label']))
    name_to_idx = dict(zip(df['label'], df['index']))
    return idx_to_name, name_to_idx

def load_equivalence_sets(csv_path):
    df = pd.read_csv(csv_path)
    eq_dict = {}
    for _, row in df.iterrows():
        idx = int(row['index'])
        same = ast.literal_eval(row['same']) if pd.notna(row['same']) else []
        eq_dict[idx] = set([idx] + same)
    return eq_dict

def apply_brightness_variation(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)

def apply_saturation_variation(image, saturation_factor):
    """Applies saturation variation to an image."""
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation_factor)

def calculate_topk_metrics(logits, gt_map_np, original_size_hw, global_indices_for_prompts, equivalence_dict, k):
    """
    Computes top-1 accuracy, top-k accuracy, and equivalence-aware mIoU for a single sample.
    """
    # Resize logits to original image size
    with torch.no_grad():
        resized_logits = F.interpolate(logits.unsqueeze(0), size=original_size_hw, mode='bilinear', align_corners=False)[0]

    H, W = original_size_hw
    logits_per_pixel = resized_logits.permute(1, 2, 0)  # [H, W, num_classes]
    topk_scores, topk_indices = torch.topk(logits_per_pixel, k, dim=-1)  # [H, W, k]
    prompt_idx_to_global_idx = np.array(global_indices_for_prompts)  # length = 40
    topk_prompt_indices = topk_indices.cpu().numpy()  # shape = [H, W, k]
    topk_global_indices = np.take(prompt_idx_to_global_idx, topk_prompt_indices)  # map [0–39] → actual global indices
    
    assert logits.shape[0] == len(global_indices_for_prompts), \
        "Mismatch: logits channels must match number of prompts"


    pred_top1 = topk_global_indices[..., 0]  # [H, W]
    gt_flat = gt_map_np.flatten()
    pred_top1_flat = pred_top1.flatten()
    topk_flat = topk_global_indices.reshape(-1, k)

    # --- Equivalence-aware pixel accuracy ---
    correct_top1 = np.array([
        pred in equivalence_dict.get(gt, {gt})
        for pred, gt in zip(pred_top1_flat, gt_flat)
    ])
    correct_topk = np.array([
        any(pred in equivalence_dict.get(gt, {gt}) for pred in pred_k)
        for pred_k, gt in zip(topk_flat, gt_flat)
    ])

    top1_acc = correct_top1.mean()
    topk_acc = correct_topk.mean()

    # --- Equivalence-aware mIoU ---
    gt_equiv = np.array([min(equivalence_dict.get(g, {g})) for g in gt_flat])
    pred_equiv_top1 = np.array([min(equivalence_dict.get(p, {p})) for p in pred_top1_flat])

    from collections import Counter
    print("GT label counts:", Counter(gt_equiv.tolist()))
    print("Predicted label counts (top1):", Counter(pred_equiv_top1.tolist()))

    # Top-1 mIoU
    intersection_top1 = {}
    union_top1 = {}
    for label in np.unique(np.concatenate([gt_equiv, pred_equiv_top1])):
        gt_mask = gt_equiv == label
        pred_mask = pred_equiv_top1 == label
        intersection_top1[label] = np.logical_and(gt_mask, pred_mask).sum()
        union_top1[label] = np.logical_or(gt_mask, pred_mask).sum()
    miou_top1 = np.mean([
        intersection_top1[l] / union_top1[l]
        for l in union_top1 if union_top1[l] > 0
    ]) if union_top1 else 0.0

    # Top-k mIoU (oracle-style)
    oracle_pred = pred_equiv_top1.copy()
    topk_equiv = np.array([
        [min(equivalence_dict.get(p, {p})) for p in pred_k]
        for pred_k in topk_flat
    ])
    for i, (gt_val, topk_vals) in enumerate(zip(gt_equiv, topk_equiv)):
        if gt_val in topk_vals:
            oracle_pred[i] = gt_val

    intersection_topk = {}
    union_topk = {}
    for label in np.unique(np.concatenate([gt_equiv, oracle_pred])):
        gt_mask = gt_equiv == label
        pred_mask = oracle_pred == label
        intersection_topk[label] = np.logical_and(gt_mask, pred_mask).sum()
        union_topk[label] = np.logical_or(gt_mask, pred_mask).sum()
    miou_topk = np.mean([
        intersection_topk[l] / union_topk[l]
        for l in union_topk if union_topk[l] > 0
    ]) if union_topk else 0.0

    return top1_acc, miou_top1, topk_acc, miou_topk


def visualize_and_save_sample(input_image, gt_map, pred_map, sample_idx, image_rel_path, label_mapping_global, idx_sample_to_global, output_dir):
    for pred_idx, global_idx in idx_sample_to_global.items():
        pred_map[pred_map == pred_idx] = global_idx
    try:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(input_image)
        ax[0].set_title(f"Input Image (Sample {sample_idx})")
        ax[0].axis('off')

        all_indices = np.unique(np.concatenate((np.unique(gt_map), np.unique(pred_map))))
        plot_indices = sorted([idx for idx in all_indices if idx in label_mapping_global])

        num_plot_classes = len(plot_indices)
        cmap = None
        norm = None
        if num_plot_classes > 0:
            if num_plot_classes == 1:
                cmap = plt.get_cmap('viridis', 2)
                boundaries = np.array([plot_indices[0]-0.5, plot_indices[0]+0.5])
                norm = mcolors.BoundaryNorm(boundaries, cmap.N)
            else:
                cmap_name = 'tab20' if num_plot_classes <= 20 else 'viridis'
                cmap = plt.get_cmap(cmap_name, num_plot_classes)
                boundaries = np.array(plot_indices + [plot_indices[-1]+1]) - 0.5
                norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        if cmap and norm:
            ax[1].imshow(gt_map, cmap=cmap, norm=norm)
        else:
            ax[1].imshow(gt_map)
            ax[1].text(0.5, 0.5, 'Coloring Error/No Classes', ha='center', va='center', transform=ax[1].transAxes)
        ax[1].set_title("Ground Truth Segmentation")
        ax[1].axis('off')

        if cmap and norm:
            ax[2].imshow(pred_map, cmap=cmap, norm=norm)
        else:
            ax[2].imshow(pred_map)
            ax[2].text(0.5, 0.5, 'Coloring Error/No Classes', ha='center', va='center', transform=ax[2].transAxes)
        ax[2].set_title("Predicted Segmentation")
        ax[2].axis('off')

        unique_gt_labels = np.unique(gt_map)
        for label_idx in unique_gt_labels:
            if label_idx == 0: continue
            label_name = label_mapping_global.get(label_idx)
            if not label_name: continue
            coords = np.where(gt_map == label_idx)
            if coords[0].size == 0: continue
            y_c, x_c = int(coords[0].mean()), int(coords[1].mean())
            ax[1].text(x_c, y_c, label_name, color="white", fontsize=8, ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        unique_pred_labels = np.unique(pred_map)
        for label_idx in unique_pred_labels:
            if label_idx == 0: continue
            label_name = label_mapping_global.get(label_idx)
            if not label_name: continue
            coords = np.where(pred_map == label_idx)
            if coords[0].size == 0: continue
            y_c, x_c = int(coords[0].mean()), int(coords[1].mean())
            ax[2].text(x_c, y_c, label_name, color="white", fontsize=8, ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.suptitle(f"Sample {sample_idx}: {os.path.basename(image_rel_path)}", fontsize=14)
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        save_path = os.path.join(output_dir, f"sample_{sample_idx:06d}_visualization.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Visualization saved to: {save_path}")
        plt.close(fig)

    except Exception as e:
        print(f"  Error generating visualization for sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals(): plt.close(fig)

# (Keep imports and other helper functions as they are)

# ... load_label_mapping, load_equivalence_sets, apply_brightness_variation, apply_saturation_variation, calculate_topk_metrics ...

# MODIFY this function:
def visualize_full_variation(gt_map, original_img, variant_imgs, original_pred, variant_preds,
                             sample_idx, image_rel_path, label_mapping_global,
                             idx_sample_to_global, output_dir,
                             # Add these parameters:
                             brightness_levels_used, saturation_levels_used):
    """
    Visualize original and multiple brightness/saturation variations of an image and predictions.
    Accepts the actual brightness/saturation levels used for variants.

    Layout:
        Row 1: GT, Original Image, Bright/Sat Variants
        Row 2: -,  Original Pred, Bright/Sat Preds
    """
    # Map prediction indices to global indices (Important: Do this *before* calculating unique indices for colormap)
    for pred_idx, global_idx in idx_sample_to_global.items():
        if original_pred is not None: # Handle potential None if original image inference failed
             original_pred[original_pred == pred_idx] = global_idx
        for i in range(len(variant_preds)):
            if variant_preds[i] is not None: # Handle potential None
                variant_preds[i][variant_preds[i] == pred_idx] = global_idx

    try:
        n_variants = len(variant_imgs)
        total_cols = 1 + n_variants
        fig, ax = plt.subplots(2, total_cols, figsize=(5 * total_cols, 10))

        # Use passed levels for titles
        top_titles = ['GT'] + [f"Bright={b:.2f}/Sat={s:.2f}" for b, s in zip(brightness_levels_used, saturation_levels_used)]
        top_images = [gt_map] + variant_imgs
        bottom_titles = ['-',] + [f"Pred Bright={b:.2f}/Sat={s:.2f}" for b, s in zip(brightness_levels_used, saturation_levels_used)]
        # Add None placeholder for GT position in bottom row
        bottom_images = [None] + variant_preds

        # Build color map based on ALL valid segmentation maps
        valid_maps = [m for m in bottom_images if isinstance(m, np.ndarray)] + [gt_map]
        all_indices = np.unique(np.concatenate([np.unique(m) for m in valid_maps]))
        plot_indices = sorted([idx for idx in all_indices if idx in label_mapping_global and idx != 0]) # Exclude background 0 if desired
        num_plot_classes = len(plot_indices)

        cmap = None
        norm = None
        if num_plot_classes > 0:
             # Simpler cmap selection
             cmap_name = 'tab20' if num_plot_classes <= 20 else 'viridis'
             cmap = plt.get_cmap(cmap_name, num_plot_classes)
             boundaries = np.array(plot_indices + [plot_indices[-1]+1]) - 0.5
             norm = mcolors.BoundaryNorm(boundaries, cmap.N)
        else: # Handle case with no valid non-zero classes found
            cmap = plt.get_cmap('gray') # Default cmap
            norm = mcolors.Normalize(vmin=0, vmax=1) # Simple norm

        # Row 0: GT and input images
        for col in range(total_cols):
            ax[0, col].axis('off')
            if top_images[col] is None: continue # Skip if placeholder

            ax[0, col].set_title(top_titles[col])
            is_seg_map = isinstance(top_images[col], np.ndarray) and col == 0 # Only GT map in top row

            ax[0, col].imshow(top_images[col],
                              cmap=cmap if is_seg_map else None,
                              norm=norm if is_seg_map else None)

            # Annotate GT only
            if col == 0:
                unique_labels = np.unique(gt_map)
                for label_idx in unique_labels:
                    if label_idx == 0 or label_idx not in label_mapping_global: continue # Skip background or unknown
                    label_name = label_mapping_global.get(label_idx)
                    coords = np.where(gt_map == label_idx)
                    if coords[0].size == 0: continue
                    y_c, x_c = int(coords[0].mean()), int(coords[1].mean())
                    ax[0, col].text(x_c, y_c, label_name, color="white", fontsize=8, ha='center', va='center',
                                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        # Row 1: prediction maps
        for col in range(total_cols):
            ax[1, col].axis('off')
            if bottom_images[col] is None: continue # Skip placeholder or failed preds

            ax[1, col].imshow(bottom_images[col], cmap=cmap, norm=norm)
            ax[1, col].set_title(bottom_titles[col])

            unique_pred_labels = np.unique(bottom_images[col])
            for label_idx in unique_pred_labels:
                if label_idx == 0 or label_idx not in label_mapping_global: continue # Skip background or unknown
                label_name = label_mapping_global.get(label_idx)
                coords = np.where(bottom_images[col] == label_idx)
                if coords[0].size == 0: continue
                y_c, x_c = int(coords[0].mean()), int(coords[1].mean())
                ax[1, col].text(x_c, y_c, label_name, color="white", fontsize=8, ha='center', va='center',
                                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.suptitle(f"Sample {sample_idx}: {os.path.basename(image_rel_path)}", fontsize=16)
        plt.tight_layout(rect=[0, 0.01, 1, 0.95])
        save_path = os.path.join(output_dir, f"sample_{sample_idx:06d}_variation.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved extended visualization to: {save_path}")

    except Exception as e:
        print(f"  Visualization error for sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals():
            plt.close(fig)

# --- MAIN EVALUATION ---
label_map_idx_to_name, label_map_name_to_idx = load_label_mapping(LABELS_PATH)
equivalence_dict = load_equivalence_sets(SIMILARITY_PATH)
metadata = pd.read_csv(METADATA_PATH)
metadata = metadata.sample(n=NUM_SAMPLES_TO_EVALUATE, random_state=42) # Added random_state for reproducibility
all_indices = list(metadata.index) # Use actual index from sampled metadata
vis_indices = set(np.random.choice(all_indices, min(NUM_SAMPLES_TO_VISUALIZE, len(all_indices)), replace=False))
print(f"Total samples to evaluate: {len(metadata)}")
print(f"Visualization indices: {vis_indices}")

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(DEVICE)
model.eval()

# Use a dictionary to accumulate results per brightness level
brightness_results_accumulator = {
    level: {'acc_list': [], 'miou_list': [], 'topk_acc_list': [], 'topk_miou_list': []}
    for level in BRIGHTNESS_LEVELS
}

# Use the actual index from the sampled dataframe
for idx in tqdm(all_indices, total=len(all_indices)):
    row = metadata.loc[idx]
    try:
        # --- Load data ---
        img_path = os.path.join(BASE_DIR, row['image_path'])
        label_path = os.path.join(BASE_DIR, row['label_path'])
        image = Image.open(img_path).convert("RGB") # Original Image
        gt_map = np.array(Image.open(label_path))
        original_size = gt_map.shape

        unique_gt_indices = sorted([i for i in np.unique(gt_map) if i != 0])
        if not unique_gt_indices:
            print(f"Skipping sample {idx}: No non-zero ground truth labels found.")
            continue # Skip if no GT labels

        # --- Prepare prompts ---
        distractor_pool = [i for i in label_map_idx_to_name if i not in unique_gt_indices and i != 0]
        num_distractors_to_add = min(NUM_DISTRACTORS, len(distractor_pool))
        distractors = np.random.choice(distractor_pool, num_distractors_to_add, replace=False).tolist() if num_distractors_to_add > 0 else []
        global_indices = unique_gt_indices + distractors
        prompts = [label_map_idx_to_name[i] for i in global_indices]
        prompt_idx_to_global_idx = {i: j for i, j in enumerate(global_indices)}

        # --- Prepare storage for visualization (if this sample needs it) ---
        visualization_data = None
        if idx in vis_indices:
            print(f"\nProcessing sample {idx} for visualization...")
            visualization_data = {
                'original_image': image.copy(), # Store a copy
                'original_pred': None,
                'variant_images': [],
                'variant_preds': [],
                'levels': [] # Store the (brightness, saturation) pairs
            }
            # Get prediction for the original image ONCE
            try:
                 original_inputs = processor(text=prompts, images=[image]*len(prompts), return_tensors="pt", padding="max_length").to(DEVICE)
                 with torch.no_grad():
                    original_logits = model(**original_inputs).logits.cpu()
                 visualization_data['original_pred'] = torch.argmax(original_logits, dim=0).numpy()
                 print(f"  Computed original prediction for sample {idx}")
            except Exception as e_orig:
                 print(f"  Error computing original prediction for sample {idx}: {e_orig}")
                 # We can continue, but the visualization will lack the original prediction


        # --- Loop through brightness/saturation levels ---
        for brightness, saturation in zip(BRIGHTNESS_LEVELS, SATURATION_LEVELS):
            print(f"  Processing sample {idx} for Brightness={brightness:.2f}, Saturation={saturation:.2f}") # More verbose logging

            # --- Apply variation ---
            image_sat = apply_saturation_variation(image, saturation)
            image_bright = apply_brightness_variation(image_sat, brightness)

            # --- Run Inference (ONLY ONCE per level) ---
            inputs = processor(text=prompts, images=[image_bright]*len(prompts), return_tensors="pt", padding="max_length").to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits.cpu() # Logits for the current variation

            print(f"logits.shape = {logits.shape}, len(prompts) = {len(prompts)}")
            # --- Calculate Metrics (using current logits) ---
            # Make sure global_indices is not empty (shouldn't be if unique_gt_indices wasn't)
            if not global_indices:
                 print(f"  Warning: No global indices (GT + distractors) for sample {idx}. Skipping metrics for this level.")
                 continue

            acc, miou, topk_acc, topk_miou = calculate_topk_metrics(
                logits, gt_map, original_size, global_indices, equivalence_dict, K_FOR_TOPK)

            # --- Accumulate metrics for this brightness level's average ---
            brightness_results_accumulator[brightness]['acc_list'].append(acc)
            brightness_results_accumulator[brightness]['miou_list'].append(miou)
            brightness_results_accumulator[brightness]['topk_acc_list'].append(topk_acc)
            brightness_results_accumulator[brightness]['topk_miou_list'].append(topk_miou)
            print(f"    Metrics for sample {idx} @ B={brightness:.2f}/S={saturation:.2f}: Acc={acc:.4f}, mIoU={miou:.4f}, TopK Acc={topk_acc:.4f}, TopK mIoU={topk_miou:.4f}")


            # --- Store data for visualization (if needed) ---
            if visualization_data is not None: # Check if we are tracking this sample for viz
                current_pred = torch.argmax(logits, dim=0).numpy()
                visualization_data['variant_images'].append(image_bright) # Store the varied image
                visualization_data['variant_preds'].append(current_pred)
                visualization_data['levels'].append((brightness, saturation))
                print(f"    Stored variant prediction for sample {idx}")


        # --- Generate visualization AFTER iterating through all levels for this sample ---
        if visualization_data is not None:
            print(f"  Generating full variation visualization for sample {idx}...")
            visualize_full_variation(
                gt_map,
                visualization_data['original_image'],
                visualization_data['variant_images'], # List of varied images
                visualization_data['original_pred'], # Single original prediction
                visualization_data['variant_preds'], # List of varied predictions
                idx,
                row['image_path'],
                label_map_idx_to_name,
                prompt_idx_to_global_idx,
                OUTPUT_VIS_DIR,
                [lvl[0] for lvl in visualization_data['levels']], # Pass actual brightness levels used
                [lvl[1] for lvl in visualization_data['levels']]  # Pass actual saturation levels used
            )

    except Exception as e:
        print(f"ERROR processing sample {idx} (Index in sampled metadata): {e}")
        import traceback
        traceback.print_exc() # Add traceback for debugging
        continue # Continue to the next sample

# --- Aggregate results ---
print("\n--- Aggregating Results ---")
brightness_results = {}
for level in BRIGHTNESS_LEVELS: # Iterate through the defined levels to maintain order
    metrics = brightness_results_accumulator[level]
    count = len(metrics['acc_list']) # Number of valid samples processed for this level
    if count > 0:
        brightness_results[level] = {
            'accuracy': np.mean(metrics['acc_list']),
            'miou': np.mean(metrics['miou_list']),
            'topk_accuracy': np.mean(metrics['topk_acc_list']),
            'topk_miou': np.mean(metrics['topk_miou_list'])
        }
        print(f"Level B={level:.2f}: Processed {count} samples. Avg Acc={brightness_results[level]['accuracy']:.4f}, Avg mIoU={brightness_results[level]['miou']:.4f}")
    else:
         brightness_results[level] = { # Handle cases where no samples were valid for a level
            'accuracy': 0.0, 'miou': 0.0, 'topk_accuracy': 0.0, 'topk_miou': 0.0
         }
         print(f"Level B={level:.2f}: No valid samples processed.")


# --- SAVE PERFORMANCE PLOT ---
print("\n--- Generating Performance Plots ---")
levels_with_results = sorted(brightness_results.keys()) # Get levels that actually have results

# Define plot details
plot_metrics = {
    'accuracy': "Top-1 Accuracy vs. Brightness/Saturation",
    'miou': "Top-1 mIoU vs. Brightness/Saturation",
    'topk_accuracy': f"Top-{K_FOR_TOPK} Accuracy vs. Brightness/Saturation",
    'topk_miou': f"Top-{K_FOR_TOPK} mIoU vs. Brightness/Saturation"
}
# Create x-axis labels explaining the coupling
x_labels = [f"B={b:.2f}\nS={s:.2f}" for b, s in zip(BRIGHTNESS_LEVELS, SATURATION_LEVELS)]
x_ticks = np.arange(len(levels_with_results))

for key, title in plot_metrics.items():
    metric_values = [brightness_results[l][key] for l in levels_with_results]
    metric_values.reverse()

    plt.figure(figsize=(12, 7)) # Slightly larger figure
    plt.plot(x_ticks, metric_values, marker='o', linestyle='-') # Use markers
    plt.xlabel("Brightness (B) / Saturation (S) Factor Pairs")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right') # Set custom x-ticks
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout() # Adjust layout

    plot_output_path = f"{PLOT_OUTPUT_PATH_BASE}_{key}.png" # Use f-string
    plt.savefig(plot_output_path, dpi=150)
    print(f"Saved performance plot to {plot_output_path}")
    plt.close()

print("\n--- Evaluation Script Finished ---")