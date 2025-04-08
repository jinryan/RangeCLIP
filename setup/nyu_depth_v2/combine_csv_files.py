import pandas as pd
import os

# Paths to the metadata files
nyu_csv = "training/nyu_depth_v2/unlabelled_patches/metadata.csv"
void_csv = "training/void_1500/random_patches/metadata.csv"
output_csv = "training/metadata.csv"

# Read the CSVs
nyu_df = pd.read_csv(nyu_csv)
void_df = pd.read_csv(void_csv)

# Keep only the 'depth_path' and 'image_path' columns
nyu_df = nyu_df[['depth_path', 'image_path']]
void_df = void_df[['depth_path', 'image_path']]

# Concatenate both DataFrames
merged_df = pd.concat([nyu_df, void_df], ignore_index=True)

# Save the merged DataFrame
merged_df.to_csv(output_csv, index=False)

print(f"Merged metadata saved to {output_csv}")