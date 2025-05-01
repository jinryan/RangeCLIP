import os
import csv
import numpy as np
from PIL import Image
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


# File paths
candidate_label_path = "/media/home/ryjin/depthclip/RangeCLIP/data/sunrgbd/SUNRGBD/candidate_labels.csv"
label_map_dir = "/media/home/ryjin/depthclip/RangeCLIP/data/sunrgbd/SUNRGBD/labels"
new_label_map_dir = "/media/home/ryjin/depthclip/RangeCLIP/data/sunrgbd/SUNRGBD/labels_new"
new_candidate_label_path = "/media/home/ryjin/depthclip/RangeCLIP/data/sunrgbd/SUNRGBD/candidate_labels_new.csv"
label_frequency_path = "/media/home/ryjin/depthclip/RangeCLIP/data/sunrgbd/SUNRGBD/label_frequency_new.csv"

os.makedirs(new_label_map_dir, exist_ok=True)

# Step 1: Load and deduplicate candidate labels
df = pd.read_csv(candidate_label_path, na_values=[], keep_default_na=False)

unique_labels = sorted(set(label.strip().lower() for label in df['label'].tolist()))  # case-insensitive deduplication

# Re-index alphabetically
new_index_map = {label: i + 1 for i, label in enumerate(unique_labels)}
reverse_index_map = {v: k for k, v in new_index_map.items()}  # For output

# Save new candidate label CSV
with open(new_candidate_label_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["label", "index"])
    for label, idx in new_index_map.items():
        writer.writerow([label, idx])

# Step 2: Create old index -> new index mapping
old_index_to_label = {row['index']: row['label'].strip().lower() for _, row in df.iterrows()}
# old_label_to_index = {row['label'].strip().lower(): row['index'] for _, row in df.iterrows()}
old_index_to_new_index = {}
for old_idx, old_label in old_index_to_label.items():
    new_idx = new_index_map.get(old_label)
    if new_idx is None:
        print(f"Warning: Label '{old_label}' not found in new index map. Skipping.")
        exit(1)
    if new_idx:
        old_index_to_new_index[int(old_idx)] = new_idx

assert len(old_index_to_new_index) == len(df), "Mismatch in number of unique labels and old index to new index mapping."

print("Old index to new index mapping:")
for old_idx, new_idx in old_index_to_new_index.items():
    print(f"Old index: {old_idx}, New index: {new_idx}")


# Step 3: Remap label maps with tqdm progress bar
label_frequency = defaultdict(int)
label_files = [f for f in os.listdir(label_map_dir) if f.endswith(".png")]
total_files = len(label_files)

for fname in tqdm(label_files, desc="Remapping label maps", unit="file"):
    img_path = os.path.join(label_map_dir, fname)
    img = Image.open(img_path)
    data = np.array(img)

    # Remap indices
    new_data = np.zeros_like(data)
    all_labels = np.unique(data)
    for label in all_labels:
        if label not in old_index_to_new_index:
            print(f"Label {label} not in old_index_to_new_index, skipping.")
            continue
        
        label_frequency[label] += np.sum(data == label)
        new_data[data == label] = old_index_to_new_index[label]
        
    # for old_idx, new_idx in old_index_to_new_index.items():
    #     mask = data == old_idx
    #     new_data[mask] = new_idx
    #     label_frequency[new_idx] += np.sum(mask)

    # Save remapped image
    new_img = Image.fromarray(new_data.astype(np.uint16))
    new_img.save(os.path.join(new_label_map_dir, fname))

# Step 4: Save label frequency CSV sorted by count
with open(label_frequency_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["label", "index", "count"])
    for idx in sorted(label_frequency, key=label_frequency.get, reverse=True):
        writer.writerow([reverse_index_map[idx], idx, label_frequency[idx]])
