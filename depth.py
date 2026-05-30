import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import pipeline

# Load depth estimation model
print("Loading depth model...")
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

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

    # Normalize depth to 0-255 for saving
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # Save depth map
    out_path = f"data/depths/{filename.replace('.png', '_depth.png')}"
    cv2.imwrite(out_path, depth_normalized)
    print(f"Saved {out_path}")

print("Done - all depth maps saved to data/depths/")