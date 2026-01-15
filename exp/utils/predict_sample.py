
import os
import sys
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Add workspace root to sys.path to import models
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

from models.DualSight_Fire import RGSNetV4

# Paths
MODEL_PATH = os.path.join(ROOT_DIR, "output/RGS_Net_V4/voting_202512162143/weights/model_best.pth")
IMG_PATH = os.path.join(ROOT_DIR, "dataset/activefire/images/patches/LC08_L1TP_133015_20200825_20200825_01_RT_p00712.tif")
MASK_PATH = os.path.join(ROOT_DIR, "dataset/activefire/masks/voting/LC08_L1TP_133015_20200825_20200825_01_RT_voting_p00712.tif")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/pred")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(path, bands=(7, 6, 2)):
    with rasterio.open(path) as src:
        img = src.read(bands)  # (3, H, W)
    
    # Normalize
    img = img.astype(np.float32) / 65535.0
    return img

def load_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1) # (H, W)
    return mask

def save_svg(data, output_path, cmap=None, vmin=None, vmax=None):
    if data.ndim == 3: # RGB
        # Clip top/bottom percentile for better contrast
        p2, p98 = np.percentile(data, (2, 98), axis=(0, 1))
        data_norm = np.clip(data, p2, p98)
        data_norm = (data_norm - p2) / (p98 - p2)
        data_norm = np.clip(data_norm, 0, 1)
        plt.imsave(output_path, data_norm, format="svg")
    else:
        # Save grayscale/heatmap
        # Avoid margins
        fig = plt.figure(figsize=(data.shape[1]/100, data.shape[0]/100), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        fig.savefig(output_path, format="svg")
        plt.close(fig)

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Load Image and Mask
    print(f"Loading image from {IMG_PATH}")
    img_np = load_image(IMG_PATH) # (3, H, W)
    print(f"Loading mask from {MASK_PATH}")
    mask_np = load_mask(MASK_PATH) # (H, W)
    
    # Prepare input tensor
    # Model expects (B, C, H, W)
    input_tensor = torch.from_numpy(img_np).unsqueeze(0).float().cuda()
    
    # Load Model
    print(f"Loading model from {MODEL_PATH}")
    model = RGSNetV4(n_channels=3, n_classes=1)
    checkpoint = torch.load(MODEL_PATH)
    # Handle DataParallel state_dict if necessary
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # If saved with DataParallel, remove "module." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        seg_logits, rec_logits, fused_logits = model(input_tensor)
        
        # Sigmoid to get probabilities
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        rec_prob = torch.sigmoid(rec_logits).squeeze().cpu().numpy()
        fuse_prob = torch.sigmoid(fused_logits).squeeze().cpu().numpy()
        
        # Binary thresholding
        seg_bin = (seg_prob > 0.5).astype(np.float32)
        rec_bin = (rec_prob > 0.5).astype(np.float32)
        fuse_bin = (fuse_prob > 0.5).astype(np.float32)

    # Save outputs
    # 1. Image (H, W, 3) for plt
    img_viz = np.transpose(img_np, (1, 2, 0))
    save_svg(img_viz, os.path.join(OUTPUT_DIR, "Image.svg"))
    print(f"Saved Image.svg")
    
    # 2. Masks
    save_svg(mask_np, os.path.join(OUTPUT_DIR, "Masks.svg"), cmap="gray")
    print(f"Saved Masks.svg")
    
    # 3. Seg
    save_svg(seg_bin, os.path.join(OUTPUT_DIR, "Seg.svg"), cmap="gray", vmin=0, vmax=1)
    print(f"Saved Seg.svg")
    
    # 4. Rec
    save_svg(rec_bin, os.path.join(OUTPUT_DIR, "Rec.svg"), cmap="gray", vmin=0, vmax=1)
    print(f"Saved Rec.svg")
    
    # 5. Fuse
    save_svg(fuse_bin, os.path.join(OUTPUT_DIR, "Fuse.svg"), cmap="gray", vmin=0, vmax=1)
    print(f"Saved Fuse.svg")

if __name__ == "__main__":
    main()
