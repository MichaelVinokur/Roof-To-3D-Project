import cv2 #used for reading and manipulating images
import numpy as np #used for numerical operations on arrays (like masks)
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
#sam_model_registry, SamAutomaticMaskGenerator — these are the two specific tools we need from SAM. sam_model_registry loads the model, SamAutomaticMaskGenerator runs it

img  = cv2.imread("inria/data/train/images/austin1.tif") # reads an image file from disk into memory as a grid of numbers 
gt_mask = cv2.imread("inria/data/train/gt/austin1.tif", 0) 
# reads the corresponding ground truth mask (the "gt" stands for "ground truth"). 
#The 0 argument means we want to read it as a grayscale image, which gives us a 2D array where each pixel value indicates whether it's part of a building (non-zero) or not (zero).

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#OpenCV loads images in BGR order (Blue, Green, Red) but SAM expects RGB (Red, Green, Blue). 
# So cv2.cvtColor(...COLOR_BGR2RGB) swaps the channel order.

# Crop a small 1024x1024 patch (the full 5000x5000 is too big for SAM)
patch = img_rgb[0:1024, 0:1024]
gt_patch = gt_mask[0:1024, 0:1024]

# Load SAM
print("Loading SAM model...")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,        # how densely SAM scans the image
    pred_iou_thresh=0.90,      # confidence threshold - only keep masks SAM is 90%+ confident about
    min_mask_region_area=500   # ignore tiny masks (trees, cars)
)

# Run SAM - masks is a list of dictionaries — each dictionary represents one detected object and contains its mask, confidence score, bounding box etc.
print("Running segmentation...")
masks = mask_generator.generate(patch)
print(f"Found {len(masks)} segments") # using f in print we can add variables directly into the string. Here it will print the number of segments found by SAM.

# Filter: only keep masks that overlap significantly with INRIA building mask
roof_masks = [] #empty list to store the masks that we will keep after filtering
for m in masks:
    seg = m["segmentation"].astype(np.uint8)
    overlap = np.logical_and(seg, gt_patch > 0).sum()
    coverage = overlap / seg.sum() if seg.sum() > 0 else 0
    if coverage > 0.7:  # at least 70% of the mask must be on a building
        roof_masks.append(m)

print(f"Roof masks after filtering: {len(roof_masks)}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(patch)
axes[0].set_title("Original patch")

axes[1].imshow(gt_patch, cmap="gray")
axes[1].set_title("INRIA building mask")

overlay = patch.copy()
for m in roof_masks:
    color = np.random.randint(0, 255, 3)
    overlay[m["segmentation"]] = color
axes[2].imshow(overlay)
axes[2].set_title(f"{len(roof_masks)} roofs detected")

plt.savefig("segmentation_result.png")
print("Saved segmentation_result.png")