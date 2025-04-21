import pandas as pd
import torch
from transformers import CLIPTokenizer, CLIPModel
import numpy as np
from tqdm import tqdm

# Load and sort labels
df = pd.read_csv("data/sunrgbd/SUNRGBD/candidate_labels.csv", na_values=[], keep_default_na=False)
df = df.sort_values(by="index", ascending=True).reset_index(drop=True)
labels = df['label'].tolist()
labels = [''] + labels  # Add empty label for index 0

# Load CLIP model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()

# Encode labels
with torch.no_grad():
    inputs = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").to("cuda")
    text_features = model.get_text_features(**inputs)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

# Cosine similarity matrix
sim_matrix = text_features @ text_features.T
sim_matrix = sim_matrix.cpu().numpy()

# Define raw similarity thresholds
thresholds = {
    "same": 0.9,
    "hard": (0.8, 0.85),
    "medium": (0.75, 0.8)
}

# Collect similarity sets
index_sets = []
string_sets = []

for i in tqdm(range(len(labels))):
    hard_similarities = []
    medium_similarities = []

    same_indices = []
    hard_indices = []
    medium_indices = []

    for j in range(len(labels)):
        if i == j:
            continue
        sim = sim_matrix[i, j]
        if sim >= thresholds["same"]:
            same_indices.append(j)
        elif thresholds["hard"][0] <= sim < thresholds["hard"][1]:
            hard_similarities.append((j, sim))
        elif thresholds["medium"][0] <= sim < thresholds["medium"][1]:
            medium_similarities.append((j, sim))

    # Sort and truncate
    hard_indices = [j for j, _ in sorted(hard_similarities, key=lambda x: x[1], reverse=False)[:50]]
    medium_indices = [j for j, _ in sorted(medium_similarities, key=lambda x: x[1], reverse=False)[:50]]

    index_sets.append({
        "index": i,
        "same": same_indices,
        "hard": hard_indices,
        "medium": medium_indices
    })

    string_sets.append({
        "label": labels[i],
        "same": [labels[j] for j in same_indices],
        "hard": [labels[j] for j in hard_indices],
        "medium": [labels[j] for j in medium_indices]
    })

# Convert and save
index_df = pd.DataFrame(index_sets)
string_df = pd.DataFrame(string_sets)

index_df.to_csv("data/sunrgbd/SUNRGBD/label_similarity_sets.csv", index=False)
string_df.to_csv("data/sunrgbd/SUNRGBD/label_similarity_sets_string.csv", index=False)

print("Saved:")
print(" → label_similarity_sets.csv (with indices)")
print(" → label_similarity_sets_string.csv (with label strings)")
