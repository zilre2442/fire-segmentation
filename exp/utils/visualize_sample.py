import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt


def visualize_sample(image_path, mask_path, output_path):
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return

    print(f"Processing image: {image_path}")
    print(f"Processing mask: {mask_path}")

    # Read image (all bands for后续任意组合)
    with rasterio.open(image_path) as src:
        count = src.count
        print(f"Image has {count} bands.")
        img_all = src.read()  # [C, H, W]

    # Read mask
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)

    # 为每个单独波段生成一张灰度 SVG 图像
    base_out, ext = os.path.splitext(output_path)
    for b in range(1, count + 1):
        band_data = img_all[b - 1].astype(float)
        min_val = np.percentile(band_data, 2)
        max_val = np.percentile(band_data, 98)
        if max_val > min_val:
            band_norm = (band_data - min_val) / (max_val - min_val)
        else:
            band_norm = np.zeros_like(band_data)
        band_norm = np.clip(band_norm, 0, 1)

        plt.figure(figsize=(6, 6))
        plt.imshow(band_norm, cmap='gray')
        plt.axis('off')

        img_out_path = f"{base_out}_band_{b:02d}.svg"
        plt.savefig(img_out_path, format='svg', bbox_inches='tight', pad_inches=0)
        print(f"Saved single-band visualization (band {b}) to {img_out_path}")
        plt.close()

    # Plot Mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask_data, cmap='gray')
    plt.axis('off')

    mask_out_path = output_path.replace('.svg', '_mask.svg')
    plt.savefig(mask_out_path, format='svg', bbox_inches='tight', pad_inches=0)
    print(f"Saved mask visualization to {mask_out_path}")
    plt.close()

if __name__ == "__main__":
    # Paths relative to the workspace root
    # Assuming the script is run from the workspace root or we can locate the files relative to this script
    
    # Get the absolute path of the workspace root (assuming this script is in exp/utils/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    img_rel_path = "dataset/activefire/images/patches/LC08_L1TP_133015_20200825_20200825_01_RT_p00712.tif"
    mask_rel_path = "dataset/activefire/masks/voting/LC08_L1TP_133015_20200825_20200825_01_RT_voting_p00712.tif"
    
    img_path = os.path.join(workspace_root, img_rel_path)
    mask_path = os.path.join(workspace_root, mask_rel_path)
    
    output_path = os.path.join(script_dir, "visualization.svg")

    visualize_sample(img_path, mask_path, output_path)
