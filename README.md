# Aerial Roof 3D Reconstruction Pipeline

This project takes aerial photos of buildings and automatically detects rooftops,
estimates their 3D shape, and outputs textured 3D meshes with building walls.

---

## What the project does

Given an aerial photograph of a city or neighborhood, the pipeline:
1. Detects and isolates individual rooftops using AI segmentation
2. Estimates the depth (height variation) of each roof using AI depth estimation
3. Converts the depth information into a 3D point cloud
4. Builds a textured 3D mesh with walls, ready to view in MeshLab or Blender

---

## Pipeline steps

### Step 1 — `segment.py` (Roof Segmentation)

Uses **SAM (Segment Anything Model, ViT-H)** by Meta to automatically detect all
regions in the aerial image, then filters them to keep only rooftops.

**How it works:**
- SAM scans the image with a dense grid of prompt points (`points_per_side=32`)
  and generates candidate masks for every region it finds
- Each mask is then filtered using three rules:
  - **Coverage:** the region must cover 10%–80% of the image (rejects tiny debris
    and regions that are the whole image)
  - **Aspect ratio:** rejects long thin strips (roads, fences, walls)
  - **Color:** rejects green vegetation (grass, trees) using HSV hue analysis
- The surviving masks are saved as individual cropped PNG images with the
  non-roof background set to black (pure zero pixels)

**Input:** `data/TLV/Small_Photos/` — aerial images (.jpg or .png)
**Output:** `data/crops/` — one PNG per detected roof, `data/masks/` — overlay visualizations

---

### Step 2 — `depth.py` (Depth Estimation)

Uses **Depth-Anything-V2-Large** (HuggingFace) to estimate the height/depth of
every pixel in each roof crop.

**How it works:**
- The model predicts a relative depth map for each crop — brighter pixels are
  closer to the camera (higher elevation), darker pixels are further away
- **Percentile clipping:** instead of simple min-max normalization (which gets
  anchored to the black background pixels and squashes the real depth range),
  we compute the 5th–95th percentile of non-background pixels and stretch only
  that range to 0–255. This gives much more contrast and sensitivity to subtle
  height differences like roof slopes
- The result is saved as a grayscale PNG depth map

**Why Large and not Small:**
The Large model (335M parameters vs 25M) handles scenes with both large depth
jumps (level-to-level steps) and subtle slopes within a single level — which is
exactly what multi-level roofs require.

**Input:** `data/crops/`
**Output:** `data/depths/` — grayscale depth map for each crop

---

### Step 3 — `pointcloud.py` (Point Cloud Generation)

Converts each (crop image, depth map) pair into a 3D point cloud.

**How it works:**
- Each pixel becomes a 3D point: X and Y come from the pixel's position in the
  image, Z comes from the depth map value scaled to real units (`* 50.0`)
- Points belonging to the black background (all channels < 10) are excluded
- Statistical outlier removal (`remove_statistical_outlier`) cleans up noise
- The result is saved as a `.ply` file (standard point cloud format)

**Input:** `data/crops/` + `data/depths/`
**Output:** `data/pointclouds/` — one .ply file per roof

---

### Step 4 — `reconstruct.py` (3D Mesh Reconstruction)

Builds a full triangulated 3D mesh from the depth map, and adds visualization
walls to make the output look like a building.

**How it works:**

**Mesh surface:**
- A `medianBlur` (kernel 5) is applied to the depth map before meshing to
  suppress pixel-level noise — without this, every noise spike becomes a
  physical bump on the surface, especially visible on sloped roofs
- The valid roof area is determined by brightness (`> 15`) to exclude the black
  background, then:
  - `MORPH_CLOSE` fills interior holes (shadows, AC units, rooftop equipment)
  - `erode` (5 iterations) pulls the boundary inward to remove "icicle" artifacts
    where the roof edge meets the background
- Each valid pixel at step-2 intervals becomes a vertex; adjacent vertices are
  connected into triangles to form the mesh surface
- Z scale is `* 80.0` — controls how tall the depth variation appears

**Walls:**
- After building the roof surface, every boundary edge (an edge where one side
  has no mesh) is detected and a wall rectangle is dropped from that edge down
  to `WALL_BOTTOM = -max(h, w) * 0.3`
- Wall height scales with the crop size so small roofs get proportional walls
  and large roofs get taller walls
- Wall vertices are coloured grey (0.65, 0.65, 0.70) to distinguish them from
  the textured roof

**Input:** `data/crops/` + `data/depths/`
**Output:** `outputs/meshes/` — one .obj file per roof (open in MeshLab or Blender)

---

### `pipeline.py` (Run everything)

Runs all 4 steps in sequence. If any step fails it stops immediately.

```
segment.py → depth.py → pointcloud.py → reconstruct.py
```

---

## Setup instructions (starting from only the .py files)

### 1. Requirements

- Python 3.9 or newer
- Git (to install segment-anything)
- An NVIDIA GPU is strongly recommended (CPU works but is much slower)

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

### 3. Install PyTorch

**With NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU only (slower):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining packages

```bash
pip install opencv-python numpy Pillow matplotlib open3d transformers huggingface_hub
```

### 5. Install segment-anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 6. Download the SAM model checkpoint

Download `sam_vit_h_4b8939.pth` (~2.4 GB) from:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Rename it to `sam_vit_h-001.pth` and place it at:
```
Image_proccesing/sam_vit_h-001.pth
```

The Depth-Anything-V2-Large model (~1.3 GB) downloads automatically from
HuggingFace the first time `depth.py` runs.

### 7. Add your aerial images

Place aerial photos (.jpg or .png) in:
```
data/TLV/Small_Photos/
```

### 8. Run

```bash
python pipeline.py
```

Or run steps individually:
```bash
python segment.py      # detect and crop rooftops
python depth.py        # estimate depth for each crop
python pointcloud.py   # build 3D point clouds
python reconstruct.py  # build 3D meshes with walls
```

---

## Output files

| Folder | Contents |
|---|---|
| `data/crops/` | Cropped roof images with black background |
| `data/masks/` | Overlay visualizations showing which roofs were detected |
| `data/depths/` | Grayscale depth maps (brighter = higher) |
| `data/pointclouds/` | 3D point clouds (.ply) |
| `outputs/meshes/` | 3D meshes with walls (.obj) — open in MeshLab or Blender |

---

## Key tunable parameters

| File | Parameter | Effect |
|---|---|---|
| `segment.py` | `points_per_side=32` | Higher = finds more segments, slower |
| `segment.py` | `pred_iou_thresh=0.86` | Higher = only very confident masks |
| `segment.py` | `min_mask_region_area=1000` | Minimum roof size in pixels |
| `depth.py` | percentile `5, 95` | Tighten for more depth contrast, loosen to keep more range |
| `reconstruct.py` | `* 80.0` (Z scale) | How tall the roof surface appears in 3D |
| `reconstruct.py` | `medianBlur(..., 5)` | Higher = smoother surface, lower = more detail |
| `reconstruct.py` | erosion `iterations=5` | Higher = fewer icicles but loses more edge detail |
| `reconstruct.py` | `WALL_BOTTOM = -max(h,w) * 0.3` | Wall height multiplier — increase for taller walls |
