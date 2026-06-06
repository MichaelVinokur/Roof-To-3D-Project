import cv2
import numpy as np
import open3d as o3d
import os

os.makedirs("data/pointclouds", exist_ok=True)

FOCAL_LENGTH = 500.0

def depth_to_pointcloud(img_rgb, depth_map, mask):
    h, w = depth_map.shape
    cx, cy = w / 2, h / 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # X and Y come purely from pixel position — simple and stable
    x = (u - cx).astype(np.float32)
    y = (v - cy).astype(np.float32)

    # Z comes from depth, scaled to show slopes clearly but not too extreme
    z = (depth_map.astype(np.float32) / 255.0) * 50.0

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = img_rgb.reshape(-1, 3) / 255.0

    # Use SAM mask — exclude pure black background
    valid = mask.reshape(-1)

    return points[valid], colors[valid]

crop_files = sorted([f for f in os.listdir("data/crops/") if f.endswith(".png")])
print(f"Processing {len(crop_files)} roofs...")

for filename in crop_files:
    img_path = f"data/crops/{filename}"
    depth_path = f"data/depths/{filename.replace('.png', '_depth.png')}"

    if not os.path.exists(depth_path):
        print(f"Skipping {filename} - no depth map found")
        continue

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    if img_rgb.shape[0] < 10 or img_rgb.shape[1] < 10:
        continue

    # Build mask from the crop:
    # segment.py sets background pixels to pure black (0,0,0)
    # So any pixel where ALL channels < 10 is background — exclude it
    # This works regardless of roof color
    mask = ~(
        (img_rgb[:, :, 0] < 10) &
        (img_rgb[:, :, 1] < 10) &
        (img_rgb[:, :, 2] < 10)
    )

    points, colors = depth_to_pointcloud(img_rgb, depth_map, mask)

    if len(points) < 100:
        print(f"Skipping {filename} - too few points")
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Remove outlier points aggressively
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    out_path = f"data/pointclouds/{filename.replace('.png', '.ply')}"
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved {out_path} ({len(pcd.points)} points)")

print("Done - all point clouds saved to data/pointclouds/")

# Visualize the first one
#first = sorted(os.listdir("data/pointclouds/"))[0]
#pcd_check = o3d.io.read_point_cloud(f"data/pointclouds/{first}")
#print(f"Visualizing {first}...")
#o3d.visualization.draw_geometries([pcd_check], window_name="Roof Point Cloud")