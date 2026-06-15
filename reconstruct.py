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
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    # Median filter removes pixel-level noise before meshing.
    # Without this, noise spikes become physical bumps — especially visible on
    # sloped/triangle roofs where the depth gradient amplifies small errors.
    depth_map = cv2.medianBlur(depth_raw, 5).astype(np.float32)

    brightness = crop_rgb.mean(axis=2)
    valid_mask = (brightness > 15).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    # Close fills interior holes (shadows, AC units, equipment) before eroding.
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Erode pulls the boundary inward to remove icicle artifacts at edges.
    valid_mask = cv2.erode(valid_mask, kernel, iterations=5).astype(bool)

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

    # Visualization walls: find every boundary edge and drop it to Z=0.
    # Wall vertices are grey so they're visually distinct from the roof texture.
    wall_bottom_verts = {}

    def get_wall_v(top_idx):
        if top_idx not in wall_bottom_verts:
            x, y, _ = vertices[top_idx]
            new_idx = len(vertices)
            vertices.append([x, y, 0.0])
            colors.append([0.65, 0.65, 0.70])
            wall_bottom_verts[top_idx] = new_idx
        return wall_bottom_verts[top_idx]

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            v0 = vertex_index[r, c]
            if v0 < 0:
                continue

            # Horizontal edge to the right: (r,c)-(r,c2)
            if ci + 1 < len(cols):
                c2 = cols[ci + 1]
                v1 = vertex_index[r, c2]
                if v1 >= 0:
                    r_prev = rows[ri - 1] if ri > 0 else -1
                    r_next = rows[ri + 1] if ri + 1 < len(rows) else h
                    if r_prev < 0 or vertex_index[r_prev, c] < 0 or vertex_index[r_prev, c2] < 0:
                        b0, b1 = get_wall_v(v0), get_wall_v(v1)
                        triangles.extend([[v0, b0, v1], [b0, b1, v1]])
                    if r_next >= h or vertex_index[r_next, c] < 0 or vertex_index[r_next, c2] < 0:
                        b0, b1 = get_wall_v(v0), get_wall_v(v1)
                        triangles.extend([[v0, v1, b0], [b0, v1, b1]])

            # Vertical edge downward: (r,c)-(r2,c)
            if ri + 1 < len(rows):
                r2 = rows[ri + 1]
                v1 = vertex_index[r2, c]
                if v1 >= 0:
                    c_prev = cols[ci - 1] if ci > 0 else -1
                    c_next = cols[ci + 1] if ci + 1 < len(cols) else w
                    if c_prev < 0 or vertex_index[r, c_prev] < 0 or vertex_index[r2, c_prev] < 0:
                        b0, b1 = get_wall_v(v0), get_wall_v(v1)
                        triangles.extend([[v0, v1, b0], [b0, v1, b1]])
                    if c_next >= w or vertex_index[r, c_next] < 0 or vertex_index[r2, c_next] < 0:
                        b0, b1 = get_wall_v(v0), get_wall_v(v1)
                        triangles.extend([[v0, b0, v1], [b0, b1, v1]])

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