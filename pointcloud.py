import cv2
import numpy as np
import open3d as o3d
import os
from PIL import Image
from transformers import pipeline

# Create output folder
os.makedirs("data/pointclouds", exist_ok=True)

# Load depth model
print("Loading depth model...")
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

# Fake camera intrinsics (since we don't have real camera data)
# These simulate a reasonable aerial camera
FOCAL_LENGTH = 500.0  # pixels
CX, CY = 512, 512     # principal point (image center)

def depth_to_pointcloud(img_rgb, depth_map, mask=None):
    h, w = depth_map.shape
    cx, cy = w / 2, h / 2  # center of this specific image

    # Create a grid of (x, y) pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Back-project pixels into 3D space
    z = depth_map.astype(np.float32)
    x = (u - cx) * z / FOCAL_LENGTH
    y = (v - cy) * z / FOCAL_LENGTH

    # Stack into (N, 3) array of 3D points
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Get colors for each point
    colors = img_rgb.reshape(-1, 3) / 255.0

    # Remove points where depth is 0 (black background)
    brightness = img_rgb.reshape(-1, 3).mean(axis=1)
    valid = (z.reshape(-1) > 5) & (brightness > 15)
    points = points[valid]
    colors = colors[valid]

    return points, colors

# Process each roof crop
crop_files = sorted(os.listdir("data/crops/"))
print(f"Processing {len(crop_files)} roofs...")

for filename in crop_files:
    img_path = f"data/crops/{filename}"
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
        continue

    # Get depth map
    pil_img = Image.fromarray(img_rgb)
    result = depth_estimator(pil_img)
    depth_map = np.array(result["depth"])

    # Convert to point cloud
    points, colors = depth_to_pointcloud(img_rgb, depth_map)

    if len(points) < 100:
        print(f"Skipping {filename} - too few points")
        continue

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Clean the point cloud - remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Save as .ply file
    out_path = f"data/pointclouds/{filename.replace('.png', '.ply')}"
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved {out_path} ({len(pcd.points)} points)")

print("Done - all point clouds saved to data/pointclouds/")
# Visualize one point cloud
pcd_check = o3d.io.read_point_cloud("data/pointclouds/austin1_roof_04.ply")
o3d.visualization.draw_geometries([pcd_check], window_name="Roof Point Cloud")