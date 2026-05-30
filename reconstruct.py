import cv2
import numpy as np
import open3d as o3d
import os

os.makedirs("outputs/meshes", exist_ok=True)

crop_dir = "data/crops"
depth_dir = "data/depths"
crop_files = sorted([f for f in os.listdir(crop_dir) if f.endswith(".png")])
print(f"Found {len(crop_files)} roof crops to reconstruct...")

success, skipped = 0, 0

for filename in crop_files:
    crop_path = os.path.join(crop_dir, filename)
    depth_path = os.path.join(depth_dir, filename.replace(".png", "_depth.png"))

    if not os.path.exists(depth_path):
        print(f"Skipping {filename} - no depth map found")
        skipped += 1
        continue

    crop = cv2.imread(crop_path)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    brightness = crop_rgb.mean(axis=2)
    valid_mask = brightness > 15
    # Erode mask edges to remove icicle artifacts at boundaries
    kernel = np.ones((5, 5), np.uint8)
    valid_mask = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    if valid_mask.sum() < 200:
        print(f"Skipping {filename} - too few valid pixels")
        skipped += 1
        continue

    step = 2
    h, w = depth_map.shape
    rows = np.arange(0, h - step, step)
    cols = np.arange(0, w - step, step)

    vertices, colors, triangles = [], [], []
    vertex_index = np.full((h, w), -1, dtype=int)

    idx = 0
    for r in rows:
        for c in cols:
            if valid_mask[r, c]:
                z = depth_map[r, c] / 255.0 * 80.0
                vertices.append([c, -r, z])
                colors.append(crop_rgb[r, c] / 255.0)
                vertex_index[r, c] = idx
                idx += 1

    for r in rows[:-1]:
        for c in cols[:-1]:
            r2, c2 = r + step, c + step
            v00 = vertex_index[r,  c ]
            v10 = vertex_index[r2, c ]
            v01 = vertex_index[r,  c2]
            v11 = vertex_index[r2, c2]
            if v00 >= 0 and v10 >= 0 and v01 >= 0:
                triangles.append([v00, v10, v01])
            if v10 >= 0 and v11 >= 0 and v01 >= 0:
                triangles.append([v10, v11, v01])

    if len(triangles) < 10:
        print(f"Skipping {filename} - not enough triangles")
        skipped += 1
        continue

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))
    mesh.compute_vertex_normals()

    out_path = f"outputs/meshes/{filename.replace('.png', '.obj')}"
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"Saved {out_path} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles)")
    success += 1

print(f"\nDone — {success} meshes saved to outputs/meshes/, {skipped} skipped.")
