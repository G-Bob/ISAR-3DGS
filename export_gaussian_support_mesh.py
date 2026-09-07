from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from skimage.measure import marching_cubes

from radar_gs.ply_evaluation import (
    filter_by_support,
    load_reference_geometry,
    make_reference_voxel_grid,
    read_ascii_ply,
    voxelize_gaussian_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ply", required=True)
    parser.add_argument("--reference_geometry", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--q_threshold", type=float, default=0.3)
    parser.add_argument("--occupancy_threshold", type=float, default=0.5)
    parser.add_argument("--overlap_threshold", type=float, default=0.05)
    parser.add_argument("--sigma_extent", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--padding_fraction", type=float, default=0.05)
    parser.add_argument("--minimum_component_triangles", type=int, default=50)
    parser.add_argument("--minimum_component_area_ratio", type=float, default=1.0e-4)
    parser.add_argument("--taubin_iterations", type=int, default=5)
    return parser.parse_args()


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise RuntimeError(f"Failed to write mesh: {path}")


def export_glb(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    converted = trimesh.Trimesh(
        vertices=vertices,
        faces=triangles,
        vertex_normals=normals,
        process=False,
    )
    converted.visual.vertex_colors = np.tile(
        np.asarray([[190, 205, 220, 255]], dtype=np.uint8),
        (len(vertices), 1),
    )
    converted.export(path)


def mesh_summary(mesh: o3d.geometry.TriangleMesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "bounds_min_m": vertices.min(axis=0).tolist() if len(vertices) else None,
        "bounds_max_m": vertices.max(axis=0).tolist() if len(vertices) else None,
        "watertight": bool(mesh.is_watertight()),
        "edge_manifold": bool(mesh.is_edge_manifold(allow_boundary_edges=True)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = filter_by_support(read_ascii_ply(args.input_ply), args.q_threshold)
    points = np.column_stack([table.column(axis) for axis in ("x", "y", "z")])
    normals = np.column_stack([table.column(axis) for axis in ("nx", "ny", "nz")])
    scales = np.column_stack(
        [table.column(name) for name in ("scale_t1", "scale_t2", "scale_normal")]
    )
    support = table.column("target_support_probability")

    reference_points, _ = load_reference_geometry(
        args.reference_geometry, sample_count=100_000, seed=42
    )
    grid_spec = make_reference_voxel_grid(
        reference_points,
        resolution=args.resolution,
        padding_fraction=args.padding_fraction,
    )
    volume = voxelize_gaussian_support(
        points,
        normals,
        scales,
        support,
        grid_spec,
        sigma_extent=args.sigma_extent,
        occupancy_probability_threshold=args.occupancy_threshold,
        overlap_contribution_threshold=args.overlap_threshold,
        fill_interior=False,
    )
    probability = volume.probability_grid
    if probability.max(initial=0.0) < args.occupancy_threshold:
        raise RuntimeError("Probability field never reaches the requested isovalue")

    vertices, triangles, _, _ = marching_cubes(
        probability,
        level=args.occupancy_threshold,
        spacing=(grid_spec.voxel_size_m,) * 3,
        allow_degenerate=False,
    )
    vertices += grid_spec.origin_m + 0.5 * grid_spec.voxel_size_m
    raw = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles.astype(np.int32)),
    )
    raw.remove_duplicated_vertices()
    raw.remove_duplicated_triangles()
    raw.remove_degenerate_triangles()
    raw.remove_non_manifold_edges()
    raw.remove_unreferenced_vertices()
    save_mesh(raw, output_dir / "k8_support_mesh_raw.ply")

    clean = o3d.geometry.TriangleMesh(raw)
    triangle_clusters, cluster_counts, cluster_areas = clean.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_counts = np.asarray(cluster_counts)
    cluster_areas = np.asarray(cluster_areas)
    largest_area = float(cluster_areas.max(initial=0.0))
    keep_cluster = (
        (cluster_counts >= args.minimum_component_triangles)
        & (cluster_areas >= largest_area * args.minimum_component_area_ratio)
    )
    remove_mask = ~keep_cluster[triangle_clusters]
    clean.remove_triangles_by_mask(remove_mask)
    clean.remove_unreferenced_vertices()
    save_mesh(clean, output_dir / "k8_support_mesh_clean.ply")
    export_glb(clean, output_dir / "k8_support_mesh_clean.glb")

    smooth = clean.filter_smooth_taubin(
        number_of_iterations=args.taubin_iterations,
        lambda_filter=0.5,
        mu=-0.53,
    )
    save_mesh(smooth, output_dir / "k8_support_mesh_smooth.ply")
    export_glb(smooth, output_dir / "k8_support_mesh_smooth.glb")

    metadata = {
        "input_ply": str(Path(args.input_ply).resolve()),
        "reference_geometry": str(Path(args.reference_geometry).resolve()),
        "q_threshold": args.q_threshold,
        "occupancy_threshold": args.occupancy_threshold,
        "overlap_threshold": args.overlap_threshold,
        "sigma_extent": args.sigma_extent,
        "resolution": args.resolution,
        "voxel_size_m": grid_spec.voxel_size_m,
        "grid_origin_m": grid_spec.origin_m.tolist(),
        "retained_gaussians": int(len(points)),
        "support_volume": volume.summary(),
        "component_count_before_cleaning": int(len(cluster_counts)),
        "component_count_after_cleaning": int(np.count_nonzero(keep_cluster)),
        "raw_mesh": mesh_summary(raw),
        "clean_mesh": mesh_summary(clean),
        "smooth_mesh": mesh_summary(smooth),
    }
    (output_dir / "mesh_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
