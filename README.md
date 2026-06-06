# SAM3 Roof Segmentation & 3D Reconstruction

Takes aerial photos of rooftops and runs them through a 4-step pipeline:
1. **Segment** — SAM detects and crops individual rooftops
2. **Depth** — Depth-Anything-V2 estimates depth for each crop
3. **Point Cloud** — converts depth maps to 3D point clouds (`.ply`)
4. **Reconstruct** — builds a 3D mesh (`.obj`) for each roof

---

## Setup on a new machine

### 1. Clone the repo

```bash
git clone https://github.com/maortab/Image_SAM3.git
cd Image_SAM3
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
```

Activate it:
- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install PyTorch

This step depends on whether your laptop has an NVIDIA GPU.

**No GPU (CPU only) — most laptops:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

> Note: Without a GPU the pipeline will still work but will be significantly slower, especially the SAM segmentation step.

### 4. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

### 5. Install segment-anything from GitHub

The `segment-anything` package requires one extra step:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> If `git` is not installed, download and install it from https://git-scm.com/

### 6. Download the SAM model checkpoint

Download the file `sam_vit_h-001.pth` (~2.4 GB) and place it at:

```
Image_proccesing/sam_vit_h-001.pth
```

Download link:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

After downloading, rename the file to `sam_vit_h-001.pth` and put it in the `Image_proccesing/` folder.

> The Depth-Anything-V2 model is downloaded automatically the first time `depth.py` runs (requires internet connection).

---

## Add your images

Place your aerial/satellite images (`.jpg` or `.png`) in:

```
data/TLV/Small_Photos/
```

Create these folders if they don't exist:
```bash
mkdir -p data/TLV/Small_Photos
```
(On Windows: `mkdir data\TLV\Small_Photos`)

---

## Run the pipeline

Run all 4 steps in sequence with one command:

```bash
python pipeline.py
```

Or run each step individually:

```bash
python segment.py      # Step 1 — segment rooftops
python depth.py        # Step 2 — estimate depth
python pointcloud.py   # Step 3 — build point clouds
python reconstruct.py  # Step 4 — build 3D meshes
```

### Output folders

| Folder | Contents |
|---|---|
| `data/crops/` | Cropped roof images (PNG) |
| `data/depths/` | Depth maps (grayscale PNG) |
| `data/pointclouds/` | 3D point clouds (PLY) |
| `outputs/meshes/` | 3D meshes (OBJ) — open in Blender or MeshLab |

---

## Project structure

```
Image_SAM3/
├── pipeline.py        # Runs all 4 steps in order
├── segment.py         # Step 1: SAM segmentation
├── depth.py           # Step 2: depth estimation
├── pointcloud.py      # Step 3: point cloud generation
├── reconstruct.py     # Step 4: 3D mesh reconstruction
├── requirements.txt   # Python dependencies
├── Image_proccesing/
│   └── sam_vit_h-001.pth   # SAM model (download separately, not in git)
└── data/
    └── TLV/Small_Photos/   # Put your input images here
```

---

## Troubleshooting

**`No module named segment_anything`**
Run: `pip install git+https://github.com/facebookresearch/segment-anything.git`

**`FileNotFoundError: sam_vit_h-001.pth`**
Download the model file and place it in `Image_proccesing/` — see Step 6 above.

**`CUDA out of memory`**
The code will automatically fall back to CPU. Or reduce image size before running.

**`No images found`**
Make sure your images are in `data/TLV/Small_Photos/` and are `.jpg` or `.png` files.
