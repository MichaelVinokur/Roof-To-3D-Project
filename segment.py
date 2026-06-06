import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

os.makedirs("data/crops", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

print("Loading SAM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
sam = sam_model_registry["vit_h"](checkpoint="Image_proccesing/sam_vit_h-001.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,        # was 32 — scan more densely to find roof sub-regions
    pred_iou_thresh=0.86,      # was 0.88 — higher confidence only
    stability_score_thresh=0.90,  # new — masks must be very stable
    min_mask_region_area=1000
)

scene_dir = "data/TLV/Small_Photos"
scene_files = sorted([f for f in os.listdir(scene_dir) if f.endswith((".png", ".jpg"))])
print(f"Found {len(scene_files)} scene images")

for scene_filename in scene_files:
    scene_name = os.path.splitext(scene_filename)[0]
    img_path = f"{scene_dir}/{scene_filename}"

    img = cv2.imread(img_path)
    if img is None:
        from PIL import Image as PilImage
        pil = PilImage.open(img_path).convert("RGB")
        img_rgb = np.array(pil)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Resize if too large for GPU (max 2048 on longest side)
    MAX_SIZE = 2048
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
        h, w = img_rgb.shape[:2]
        print(f"\nProcessing {scene_filename} (resized to {w}x{h})...")
    else:
        print(f"\nProcessing {scene_filename} ({w}x{h})...")

    masks = mask_generator.generate(img_rgb)
    print(f"  SAM found {len(masks)} segments total")

    image_area = h * w
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    def is_roof(m):
        seg = m["segmentation"]
        # Coverage filter: must be 3%-35% of image
        coverage = seg.sum() / image_area
        if not (0.10 < coverage < 0.80):
            return False
        # Aspect ratio filter: reject long thin strips (roads, fences)
        x, y, bw, bh = m["bbox"]
        if max(bw, bh) / (min(bw, bh) + 1e-6) > 3.5:
            return False
        # Color filter: reject green vegetation (grass, trees)
        region_hsv = img_hsv[seg]
        hue = region_hsv[:, 0]
        sat = region_hsv[:, 1]
        green_pixels = ((hue > 35) & (hue < 85) & (sat > 40)).sum()
        if green_pixels / len(hue) > 0.35:
            return False
        # Color filter: reject bright blue (sky, pools, cars)
        #blue_pixels = ((hue > 100) & (hue < 130) & (sat > 60)).sum()
        #if blue_pixels / len(hue) > 0.4:
            #return False
        # Large segments must contain the image center (roof is always main subject)
        #if coverage > 0.35:
            #cy_img, cx_img = h // 2, w // 2
            #if not seg[cy_img, cx_img]:
                #return False
        return True

    roof_masks = [m for m in masks if is_roof(m)]
    roof_masks = sorted(roof_masks, key=lambda x: x["segmentation"].sum(), reverse=True)
    print(f"  {len(roof_masks)} roof candidates after filtering")

    for i, m in enumerate(roof_masks):
        seg = m["segmentation"]
        x, y, bw, bh = [int(v) for v in m["bbox"]]
        roof_crop = img_rgb[y:y+bh, x:x+bw].copy()
        roof_crop[~seg[y:y+bh, x:x+bw]] = 0
        crop_path = f"data/crops/{scene_name}_roof_{i:02d}.png"
        cv2.imwrite(crop_path, cv2.cvtColor(roof_crop, cv2.COLOR_RGB2BGR))

    overlay = img_rgb.copy()
    for m in roof_masks:
        overlay[m["segmentation"]] = np.random.randint(100, 255, 3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(overlay)
    axes[1].set_title(f"{len(roof_masks)} roofs detected")
    plt.tight_layout()
    plt.savefig(f"data/masks/{scene_name}_result.png")
    plt.close()
    print(f"  Saved {len(roof_masks)} crops + visualization")

print("\nDone! Check data/crops/ and data/masks/")
