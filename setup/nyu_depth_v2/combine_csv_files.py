import pandas as pd
import os

# Paths to the metadata files
nyu_csv = "training/nyu_depth_v2/unlabelled_patches/metadata.csv"
void_csv = "training/void_1500/random_patches/metadata.csv"
output_csv = "training/unlabeled_metadata.csv"

nyu_base = "training/nyu_depth_v2/unlabelled_patches"
void_base = "training/void_1500/random_patches"
# Read the CSVs
nyu_df = pd.read_csv(nyu_csv)
void_df = pd.read_csv(void_csv)

# Keep only the 'depth_path' and 'image_path' columns
nyu_df = nyu_df[['depth_path', 'image_path']]
void_df = void_df[['depth_path', 'image_path']]

# Prepend the current directory path to the relative paths
nyu_df['depth_path'] = nyu_df['depth_path'].apply(lambda x: os.path.join(nyu_base, x))
nyu_df['image_path'] = nyu_df['image_path'].apply(lambda x: os.path.join(nyu_base, x))

void_df['depth_path'] = void_df['depth_path'].apply(lambda x: os.path.join(void_base, x))
void_df['image_path'] = void_df['image_path'].apply(lambda x: os.path.join(void_base, x))

# Concatenate both DataFrames
merged_df = pd.concat([nyu_df, void_df], ignore_index=True)

# Save the merged DataFrame
merged_df.to_csv(output_csv, index=False)

print(f"Merged metadata saved to {output_csv}")
