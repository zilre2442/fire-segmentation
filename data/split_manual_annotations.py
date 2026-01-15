#!/usr/bin/env python3
"""
Manual Annotations Dataset Split Script
"""

import os
import random
import csv
from tqdm import tqdm

# Configuration
DATASET_ROOT = "dataset/mannual_annotations"
IMAGE_DIR = os.path.join(DATASET_ROOT, "landsat_patches")
MASK_DIR = os.path.join(DATASET_ROOT, "manual_annotations_patches")
OUTPUT_DIR = "data/splits_manual"

# Split ratios
TRAIN_RATIO = 0.0
VAL_RATIO = 0.0
TEST_RATIO = 1.0

RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Get all mask files
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.tif')]
    
    pairs = []
    for mask_file in tqdm(mask_files, desc="Matching pairs"):
        # Construct image filename: remove '_v1'
        # mask: LC08_..._RT_v1_pXXXXX.tif
        # image: LC08_..._RT_pXXXXX.tif
        
        image_file = mask_file.replace('_v1', '')
        
        image_path = os.path.join(IMAGE_DIR, image_file)
        mask_path = os.path.join(MASK_DIR, mask_file)
        
        if os.path.exists(image_path):
            pairs.append((image_path, mask_path))
        else:
            print(f"Warning: Image not found for mask {mask_file}")
            
    # Shuffle and split
    random.shuffle(pairs)
    total = len(pairs)
    
    # All to test
    test_pairs = pairs
    
    print(f"Total pairs: {total}")
    print(f"Test: {len(test_pairs)}")
    
    # Write CSVs
    def write_csv(filename, data_pairs):
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'mask_path'])
            writer.writerows(data_pairs)
        print(f"Saved {filepath}")

    write_csv('manual_test.csv', test_pairs)
    
    # Clean up old train/val files if they exist
    for f in ['manual_train.csv', 'manual_val.csv']:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed {p}")

if __name__ == "__main__":
    main()
