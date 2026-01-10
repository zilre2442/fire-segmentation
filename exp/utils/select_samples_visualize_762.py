import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.gridspec import GridSpec

# Configuration
ACTIVEFIRE_CSV = "data/splits_activefire/voting_train.csv"
LAND8FIRE_CSV = "data/splits_land8fire/Land8Fire_train.csv"
OUTPUT_PDF = "output/visualization_samples_762.pdf"
BANDS = (7, 6, 2)

def get_pixel_count(mask_path):
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            return np.sum(mask > 0)
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return -1

def find_samples(csv_path, categories, samples_per_category=1):
    """
    Find samples for each category.
    categories: list of tuples (min_pixels, max_pixels)
    """
    samples = {i: [] for i in range(len(categories))}
    
    # Read all rows first
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        rows = list(reader)
    
    # Shuffle rows to get random samples
    random.shuffle(rows)
    
    for row in rows:
        img_path, mask_path = row
        if not os.path.exists(mask_path):
            continue
            
        count = get_pixel_count(mask_path)
        
        for i, (min_p, max_p) in enumerate(categories):
            if len(samples[i]) < samples_per_category:
                if min_p <= count <= max_p:
                    samples[i].append((img_path, mask_path, count))
                    break
        
        # Check if we have enough samples for all categories
        if all(len(samples[i]) >= samples_per_category for i in range(len(categories))):
            break
            
    return [samples[i][0] for i in range(len(categories))]

def load_image_762(img_path):
    with rasterio.open(img_path) as src:
        # Read bands 7, 6, 2
        # Note: rasterio uses 1-based indexing
        # Check if bands exist
        if src.count < 7:
             # Fallback or error? Assuming Landsat 8 has enough bands.
             # If it's a patch, it might have all bands.
             # Let's assume the file has at least 7 bands.
             img = src.read((7, 6, 2))
        else:
             img = src.read((7, 6, 2))
             
    # Normalize
    img = img.astype(np.float32) / 65535.0
    
    # Transpose to (H, W, C) for matplotlib
    img = np.transpose(img, (1, 2, 0))
    
    # Simple enhancement for visualization (clip and scale)
    # Adjust these values as needed for better visualization
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip(img, p2, p98)
    img = (img - p2) / (p98 - p2)
    img = np.clip(img, 0, 1)
    
    return img

def load_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask

def main():
    # Define categories: 1-10, 10-100, 100-1000, >1000
    # Using float('inf') for >1000
    categories = [
        (1, 10),
        (11, 100),
        (101, 1000),
        (1001, float('inf'))
    ]
    
    print("Finding samples for ActiveFire...")
    activefire_samples = find_samples(ACTIVEFIRE_CSV, categories)
    
    print("Finding samples for Land8Fire...")
    land8fire_samples = find_samples(LAND8FIRE_CSV, categories)
    
    # Check if we found enough samples
    if len(activefire_samples) < 4:
        print("Warning: Not enough ActiveFire samples found.")
    if len(land8fire_samples) < 4:
        print("Warning: Not enough Land8Fire samples found.")

    # Create PDF
    print("Generating visualization...")
    
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05)
    
    col_labels = ["(a)", "(b)", "(c)", "(d)"]
    
    # Row 1: ActiveFire Images
    for i, sample in enumerate(activefire_samples):
        ax = fig.add_subplot(gs[0, i])
        img = load_image_762(sample[0])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("ActiveFire\nImages", fontsize=14, rotation=90, labelpad=10)
        ax.set_title(col_labels[i], fontsize=14, pad=10)

    # Row 2: ActiveFire Masks
    for i, sample in enumerate(activefire_samples):
        ax = fig.add_subplot(gs[1, i])
        mask = load_mask(sample[1])
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("ActiveFire\nMasks", fontsize=14, rotation=90, labelpad=10)

    # Row 3: Land8Fire Images
    for i, sample in enumerate(land8fire_samples):
        ax = fig.add_subplot(gs[2, i])
        img = load_image_762(sample[0])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Land8Fire\nImages", fontsize=14, rotation=90, labelpad=10)

    # Row 4: Land8Fire Masks
    for i, sample in enumerate(land8fire_samples):
        ax = fig.add_subplot(gs[3, i])
        mask = load_mask(sample[1])
        ax.imshow(mask, cmap='gray', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Land8Fire\nMasks", fontsize=14, rotation=90, labelpad=10)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    
    plt.savefig(OUTPUT_PDF, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
