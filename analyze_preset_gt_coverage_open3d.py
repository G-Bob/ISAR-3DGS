#!/usr/bin/env python3
import argparse, csv, json, math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib import cm


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def basis(direction):
    d = unit(direction)
    ref = np.array([0, 0, 1], np.float32) if abs(float(d[2])) < .9 else np.array([0, 1, 0], np.float32)
    u = unit(np.cross(d, ref)); v = unit(np.cross(d, u))
    return u, v


def cast_visible(scene, direction, center, radius, resolution):
    # range_axis_cad points from the object toward the sensor, so rays travel -direction.
    d = unit(direction); u, v = basis(d)
    extent = radius * 1.08
    q = np.linspace(-extent, extent, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(q, q, indexing="xy")
    origins = center + d * (radius * 2.5) + xx[..., None] * u + yy[..., None] * v
    dirs = np.broadcast_to(-d, origins.shape)
    rays = np.concatenate([origins, dirs], axis=-1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    ids = ans["primitive_ids"].numpy().reshape(-1)
    invalid = np.iinfo(np.uint32).max
    return np.unique(ids[ids != invalid]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=192)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(args.gt), enable_post_processing=True)
    if mesh.is_empty(): raise RuntimeError(f"Cannot load GT mesh: {args.gt}")
    mesh.remove_duplicated_vertices(); mesh.remove_duplicated_triangles(); mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals(); mesh.compute_triangle_normals()
    verts = np.asarray(mesh.vertices); tri = np.asarray(mesh.triangles)
    p0, p1, p2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    areas = .5 * np.linalg.norm(np.cross(p1-p0, p2-p0), axis=1)
    centers = (p0+p1+p2)/3
    total_area = float(areas.sum())
    bbox = mesh.get_axis_aligned_bounding_box(); center = np.asarray(bbox.get_center(), np.float32)
    radius = float(np.linalg.norm(bbox.get_extent())/2)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    all_results = []
    fig = plt.figure(figsize=(18, 9))
    for panel, k in enumerate(range(2, 10), 1):
        preset_path = args.data_root / f"preset_batch_20260822_{k}.json"
        preset = json.loads(preset_path.read_text())
        encounter_masks, los_all = [], []
        for enc in preset["encounters"]:
            enc_dir = args.data_root / enc["folder_name"]
            los = unit(np.load(enc_dir / "range_axis_cad.npy"))
            los_all.append(los)
            mask = np.zeros(len(tri), dtype=bool)
            for direction in los:
                mask[cast_visible(scene, direction, center, radius, args.resolution)] = True
            encounter_masks.append(mask)
        counts = np.stack(encounter_masks).sum(axis=0)
        los_all = np.concatenate(los_all)
        any_area = float(areas[counts > 0].sum())
        a2 = float(areas[counts >= 2].sum())
        a3 = float(areas[counts >= 3].sum()) if k >= 3 else 0.0
        weighted = float((areas * counts).sum())
        # Pairwise angular separation between encounter central LOS directions.
        central = unit(np.stack([x.mean(axis=0) for x in np.split(los_all, k)]))
        dots = np.clip(central @ central.T, -1, 1)
        angles = np.degrees(np.arccos(dots[np.triu_indices(k, 1)]))
        result = {
            "k": k, "gt_total_area_m2": total_area,
            "visible_area_m2": any_area, "visible_fraction": any_area/total_area,
            "area_per_encounter_m2": any_area/k,
            "multi_encounter_area_ge2_m2": a2, "multi_encounter_fraction_ge2": a2/total_area,
            "multi_encounter_area_ge3_m2": a3, "multi_encounter_fraction_ge3": a3/total_area,
            "mean_encounter_observations_per_surface": weighted/total_area,
            "central_los_pair_angle_min_deg": float(angles.min()),
            "central_los_pair_angle_mean_deg": float(angles.mean()),
            "central_los_pair_angle_max_deg": float(angles.max()),
            "encounters": [x["folder_name"] for x in preset["encounters"]],
        }
        all_results.append(result)

        # Open3D point cloud: one point per GT triangle visible from at least one encounter.
        keep = counts > 0; cmax = max(1, counts.max())
        colors = cm.turbo((counts[keep]-1)/max(1, cmax-1))[:, :3]
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers[keep]))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(args.out/f"K{k:02d}_gt_los_hits.ply"), cloud)
        los_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(center + los_all*radius*1.5))
        los_cloud.colors = o3d.utility.Vector3dVector(np.tile([[1., .15, .05]], (len(los_all), 1)))
        o3d.io.write_point_cloud(str(args.out/f"K{k:02d}_los_sphere.ply"), los_cloud)

        ax = fig.add_subplot(2, 4, panel, projection="3d")
        ax.scatter(centers[keep,0], centers[keep,1], centers[keep,2], c=colors, s=1, alpha=.85)
        ax.scatter((center+central*radius*1.5)[:,0], (center+central*radius*1.5)[:,1],
                   (center+central*radius*1.5)[:,2], c="red", s=18)
        ax.set_title(f"K={k}  area={any_area/total_area:.1%}  >=2={a2/total_area:.1%}")
        ext=np.asarray(bbox.get_extent()); ax.set_box_aspect(np.maximum(ext, .01)); ax.set_axis_off()

    (args.out/"coverage_metrics.json").write_text(json.dumps(all_results, indent=2))
    keys=[x for x in all_results[0] if x != "encounters"]
    with (args.out/"coverage_metrics.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for row in all_results: w.writerow({k:row[k] for k in keys})
    fig.suptitle("GT surface hit by preset LOS (color = number of distinct encounters)")
    fig.tight_layout(); fig.savefig(args.out/"K02_K09_gt_los_coverage.png", dpi=220); plt.close(fig)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__": main()
