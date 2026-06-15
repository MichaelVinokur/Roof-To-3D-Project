import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import pipeline

# Load depth estimation model
print("Loading depth model...")
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# Create output folder
os.makedirs("data/depths", exist_ok=True)

# Run depth estimation on each roof crop
crop_files = sorted(os.listdir("data/crops/"))
print(f"Processing {len(crop_files)} roof crops...")

for filename in crop_files:
    # Load the roof crop
    img_path = f"data/crops/{filename}"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Skip if image is too small
    if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
        print(f"Skipping {filename} - too small")
        continue

    # Run depth estimation
    from PIL import Image
    pil_img = Image.fromarray(img_rgb)
    result = depth_estimator(pil_img)
    depth_map = np.array(result["depth"])

    # Percentile clipping before normalization.
    # Plain NORM_MINMAX anchors to the background zeros, squashing real roof depth
    # into a narrow band. Clipping to the 2nd-98th percentile of non-zero pixels
    # stretches the actual roof depth across the full 0-255 range.
    nonzero = depth_map[depth_map > 0]
    if len(nonzero) > 100:
        p_low, p_high = np.percentile(nonzero, 5), np.percentile(nonzero, 95)
    else:
        p_low, p_high = depth_map.min(), depth_map.max()
    depth_clipped = np.clip(depth_map, p_low, p_high)
    depth_normalized = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save depth map
    out_path = f"data/depths/{filename.replace('.png', '_depth.png')}"
    cv2.imwrite(out_path, depth_normalized)
    print(f"Saved {out_path}")

print("Done - all depth maps saved to data/depths/")