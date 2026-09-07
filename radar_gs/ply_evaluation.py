from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree


@dataclass
class PlyTable:
    properties: list[str]
    values: np.ndarray

    def column(self, name: str) -> np.ndarray:
        if name not in self.properties:
            raise KeyError(f"PLY has no vertex property named {name!r}")
        return self.values[:, self.properties.index(name)]


@dataclass(frozen=True)
class VoxelGridSpec:
    origin_m: np.ndarray
    voxel_size_m: float
    resolution: int

    @property
    def maximum_m(self) -> np.ndarray:
        return self.origin_m + self.voxel_size_m * self.resolution


@dataclass
class GaussianSupportVolume:
    probability_grid: np.ndarray
    occupied_grid: np.ndarray
    overlap_count_grid: np.ndarray
    outside_occupied_count: int

    def summary(self) -> dict:
        directly_supported = self.probability_grid > 0.0
        occupied_from_probability = self.probability_grid[
            self.probability_grid > 0.0
        ]
        overlapping = directly_supported & (self.overlap_count_grid >= 2)
        return {
            "soft_support_voxel_count": int(np.count_nonzero(directly_supported)),
            "mean_nonzero_support_probability": (
                float(occupied_from_probability.mean())
                if occupied_from_probability.size
                else 0.0
            ),
            "maximum_support_probability": (
                float(occupied_from_probability.max())
                if occupied_from_probability.size
                else 0.0
            ),
            "multi_gaussian_overlap_voxel_count": int(np.count_nonzero(overlapping)),
            "multi_gaussian_overlap_fraction": float(
                np.count_nonzero(overlapping) / max(np.count_nonzero(directly_supported), 1)
            ),
            "maximum_overlap_count": int(self.overlap_count_grid.max(initial=0)),
        }


def make_reference_voxel_grid(
    reference_points: np.ndarray,
    resolution: int = 256,
    padding_fraction: float = 0.05,
) -> VoxelGridSpec:
    """Create a cubic, GT-anchored grid shared by every q threshold."""
    points = np.asarray(reference_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Reference points must have shape [N,3] and be non-empty")
    if resolution < 8:
        raise ValueError("voxel resolution must be at least 8")
    if padding_fraction < 0:
        raise ValueError("voxel padding fraction cannot be negative")
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    side = float(np.max(upper - lower))
    if side <= 0:
        raise ValueError("Reference geometry has zero spatial extent")
    side *= 1.0 + 2.0 * float(padding_fraction)
    center = 0.5 * (lower + upper)
    origin = center - 0.5 * side
    return VoxelGridSpec(origin, side / int(resolution), int(resolution))


def _points_to_voxel_grid(
    points: np.ndarray, spec: VoxelGridSpec, strict_bounds: bool = False
) -> np.ndarray:
    grid = np.zeros((spec.resolution,) * 3, dtype=bool)
    if not len(points):
        return grid
    indices = np.floor(
        (np.asarray(points, dtype=np.float64) - spec.origin_m) / spec.voxel_size_m
    ).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < spec.resolution), axis=1)
    if strict_bounds and not np.all(valid):
        raise ValueError(
            f"Reference voxelization placed {int(np.count_nonzero(~valid))} voxels "
            "outside the evaluation grid; increase --voxel_padding_fraction"
        )
    indices = indices[valid]
    if len(indices):
        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid


def _load_trimesh_geometry(path: Path):
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "Mesh voxel evaluation requires trimesh; run pip install -r requirements.txt"
        ) from error
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if isinstance(mesh, (list, tuple)):
        mesh = trimesh.util.concatenate(mesh)
    if not hasattr(mesh, "vertices") or not len(mesh.vertices):
        raise ValueError(f"Geometry contains no usable mesh: {path}")
    return mesh


def voxelize_reference_geometry(
    path: str | Path,
    spec: VoxelGridSpec,
    unit_scale: float = 1.0,
    fallback_points: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Voxelize GT as a solid mesh where possible.

    Point-only formats fall back to occupied sample voxels and are explicitly
    labelled so they cannot be confused with the solid-mesh G2-SDF protocol.
    """
    path = Path(path)
    if path.suffix.lower() in {".obj", ".glb", ".gltf"}:
        mesh = _load_trimesh_geometry(path).copy()
        mesh.apply_scale(float(unit_scale))
        voxelized = mesh.voxelized(spec.voxel_size_m).fill()
        grid = _points_to_voxel_grid(
            np.asarray(voxelized.points), spec, strict_bounds=True
        )
        return grid, "solid_mesh"
    if fallback_points is None:
        fallback_points, _ = load_reference_geometry(path, sample_count=0)
        fallback_points = fallback_points * float(unit_scale)
    return _points_to_voxel_grid(
        fallback_points, spec, strict_bounds=True
    ), "sampled_surface"


def _surface_bases_numpy(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    z_axis = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), normals.shape)
    y_axis = np.broadcast_to(np.asarray([0.0, 1.0, 0.0]), normals.shape)
    reference = np.where((np.abs(normals[:, 2]) > 0.9)[:, None], y_axis, z_axis)
    tangent_1 = np.cross(reference, normals)
    tangent_1 /= np.maximum(np.linalg.norm(tangent_1, axis=1, keepdims=True), 1.0e-12)
    tangent_2 = np.cross(normals, tangent_1)
    tangent_2 /= np.maximum(np.linalg.norm(tangent_2, axis=1, keepdims=True), 1.0e-12)
    return np.stack((tangent_1, tangent_2, normals), axis=-1)


def voxelize_gaussians(
    points: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    spec: VoxelGridSpec,
    sigma_extent: float = 2.0,
    fill_interior: bool = True,
    max_candidate_voxels_per_gaussian: int = 250000,
) -> tuple[np.ndarray, int]:
    """Rasterize oriented Gaussian ellipsoids and count voxels outside the GT grid."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if points.shape != normals.shape or scales.shape != points.shape:
        raise ValueError("points, normals and scales must all have shape [N,3]")
    if sigma_extent <= 0:
        raise ValueError("Gaussian sigma extent must be positive")
    grid = np.zeros((spec.resolution,) * 3, dtype=bool)
    outside: set[tuple[int, int, int]] = set()
    if not len(points):
        return grid, 0

    bases = _surface_bases_numpy(normals)
    scales = np.maximum(scales, spec.voxel_size_m * 0.25)
    for mean, basis, gaussian_scales in zip(points, bases, scales):
        covariance_diagonal = (basis**2) @ (gaussian_scales**2)
        half_extent = float(sigma_extent) * np.sqrt(covariance_diagonal)
        lower = np.floor((mean - half_extent - spec.origin_m) / spec.voxel_size_m).astype(int)
        upper = np.floor((mean + half_extent - spec.origin_m) / spec.voxel_size_m).astype(int)
        shape = upper - lower + 1
        candidate_count = int(np.prod(shape, dtype=np.int64))
        if candidate_count > max_candidate_voxels_per_gaussian:
            raise ValueError(
                "A Gaussian spans too many candidate voxels "
                f"({candidate_count:,}); check units/scales or increase "
                "max_candidate_voxels_per_gaussian"
            )
        ix, iy, iz = np.meshgrid(
            np.arange(lower[0], upper[0] + 1),
            np.arange(lower[1], upper[1] + 1),
            np.arange(lower[2], upper[2] + 1),
            indexing="ij",
        )
        indices = np.column_stack((ix.ravel(), iy.ravel(), iz.ravel()))
        centers = spec.origin_m + (indices + 0.5) * spec.voxel_size_m
        local = (centers - mean) @ basis
        inside_ellipsoid = np.sum((local / gaussian_scales) ** 2, axis=1) <= sigma_extent**2
        indices = indices[inside_ellipsoid]
        valid = np.all((indices >= 0) & (indices < spec.resolution), axis=1)
        valid_indices = indices[valid]
        if len(valid_indices):
            grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = True
        outside.update(map(tuple, indices[~valid].tolist()))

    if fill_interior and np.any(grid):
        grid = np.asarray(binary_fill_holes(grid), dtype=bool)
    return grid, len(outside)


def voxelize_gaussian_support(
    points: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    support: np.ndarray,
    spec: VoxelGridSpec,
    sigma_extent: float = 2.0,
    occupancy_probability_threshold: float = 0.5,
    overlap_contribution_threshold: float = 0.05,
    fill_interior: bool = True,
    max_candidate_voxels_per_gaussian: int = 250000,
) -> GaussianSupportVolume:
    """Accumulate 3D Gaussian support with the renderer's noisy-OR rule.

    Unlike binary ellipsoid union, a retained Gaussian is not treated as fully
    opaque.  Its learned target-support probability weights the Gaussian
    kernel.  A voxel is occupied only when the accumulated probability reaches
    ``occupancy_probability_threshold``.  The overlap-count grid records how
    many individual Gaussians contribute at least
    ``overlap_contribution_threshold`` to each voxel.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64).reshape(-1)
    if points.shape != normals.shape or scales.shape != points.shape:
        raise ValueError("points, normals and scales must all have shape [N,3]")
    if len(support) != len(points):
        raise ValueError("support must have shape [N]")
    if sigma_extent <= 0:
        raise ValueError("Gaussian sigma extent must be positive")
    if not 0.0 < occupancy_probability_threshold <= 1.0:
        raise ValueError("occupancy probability threshold must be in (0, 1]")
    if not 0.0 <= overlap_contribution_threshold <= 1.0:
        raise ValueError("overlap contribution threshold must be in [0, 1]")
    if np.any((support < 0.0) | (support > 1.0)):
        raise ValueError("support probabilities must be in [0, 1]")

    shape = (spec.resolution,) * 3
    log_background = np.zeros(shape, dtype=np.float32)
    overlap_count = np.zeros(shape, dtype=np.uint16)
    outside_log_background: dict[tuple[int, int, int], float] = {}
    if not len(points):
        return GaussianSupportVolume(
            probability_grid=np.zeros(shape, dtype=np.float32),
            occupied_grid=np.zeros(shape, dtype=bool),
            overlap_count_grid=overlap_count,
            outside_occupied_count=0,
        )

    bases = _surface_bases_numpy(normals)
    scales = np.maximum(scales, spec.voxel_size_m * 0.25)
    for mean, basis, gaussian_scales, probability in zip(
        points, bases, scales, support
    ):
        if probability <= 0.0:
            continue
        covariance_diagonal = (basis**2) @ (gaussian_scales**2)
        half_extent = float(sigma_extent) * np.sqrt(covariance_diagonal)
        lower = np.floor(
            (mean - half_extent - spec.origin_m) / spec.voxel_size_m
        ).astype(int)
        upper = np.floor(
            (mean + half_extent - spec.origin_m) / spec.voxel_size_m
        ).astype(int)
        candidate_shape = upper - lower + 1
        candidate_count = int(np.prod(candidate_shape, dtype=np.int64))
        if candidate_count > max_candidate_voxels_per_gaussian:
            raise ValueError(
                "A Gaussian spans too many candidate voxels "
                f"({candidate_count:,}); check units/scales or increase "
                "max_candidate_voxels_per_gaussian"
            )
        ix, iy, iz = np.meshgrid(
            np.arange(lower[0], upper[0] + 1),
            np.arange(lower[1], upper[1] + 1),
            np.arange(lower[2], upper[2] + 1),
            indexing="ij",
        )
        indices = np.column_stack((ix.ravel(), iy.ravel(), iz.ravel()))
        centers = spec.origin_m + (indices + 0.5) * spec.voxel_size_m
        local = (centers - mean) @ basis
        mahalanobis = np.sum((local / gaussian_scales) ** 2, axis=1)
        retained = mahalanobis <= sigma_extent**2
        indices = indices[retained]
        contribution = probability * np.exp(-0.5 * mahalanobis[retained])
        contribution = np.clip(contribution, 0.0, 1.0 - 1.0e-7)
        valid = np.all((indices >= 0) & (indices < spec.resolution), axis=1)
        valid_indices = indices[valid]
        valid_contribution = contribution[valid]
        if len(valid_indices):
            coordinates = tuple(valid_indices[:, axis] for axis in range(3))
            np.add.at(
                log_background,
                coordinates,
                np.log1p(-valid_contribution).astype(np.float32),
            )
            if overlap_contribution_threshold <= 0.0:
                overlap_valid = np.ones(len(valid_indices), dtype=bool)
            else:
                overlap_valid = valid_contribution >= overlap_contribution_threshold
            if np.any(overlap_valid):
                selected = valid_indices[overlap_valid]
                np.add.at(
                    overlap_count,
                    tuple(selected[:, axis] for axis in range(3)),
                    np.uint16(1),
                )

        for outside_index, outside_contribution in zip(
            indices[~valid], contribution[~valid]
        ):
            key = tuple(int(value) for value in outside_index)
            outside_log_background[key] = outside_log_background.get(key, 0.0) + float(
                np.log1p(-outside_contribution)
            )

    probability_grid = -np.expm1(log_background)
    occupied_grid = probability_grid >= float(occupancy_probability_threshold)
    if fill_interior and np.any(occupied_grid):
        occupied_grid = np.asarray(binary_fill_holes(occupied_grid), dtype=bool)
    outside_occupied_count = sum(
        -np.expm1(log_probability) >= occupancy_probability_threshold
        for log_probability in outside_log_background.values()
    )
    return GaussianSupportVolume(
        probability_grid=probability_grid.astype(np.float32, copy=False),
        occupied_grid=occupied_grid,
        overlap_count_grid=overlap_count,
        outside_occupied_count=int(outside_occupied_count),
    )


def write_voxel_surface_ply(
    path: str | Path,
    occupied_grid: np.ndarray,
    spec: VoxelGridSpec,
) -> None:
    """Export the boundary of a fused occupancy grid as an ASCII mesh PLY."""
    occupied = np.asarray(occupied_grid, dtype=bool)
    expected_shape = (spec.resolution,) * 3
    if occupied.shape != expected_shape:
        raise ValueError(f"occupied grid must have shape {expected_shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    directions = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )
    vertices: list[tuple[int, int, int]] = []
    vertex_lookup: dict[tuple[int, int, int], int] = {}
    faces: list[tuple[int, int, int, int]] = []
    resolution = spec.resolution
    occupied_indices = np.argwhere(occupied)
    for index_array in occupied_indices:
        index = tuple(int(value) for value in index_array)
        for direction, corners in directions:
            neighbor = tuple(index[axis] + direction[axis] for axis in range(3))
            neighbor_inside = all(0 <= neighbor[axis] < resolution for axis in range(3))
            if neighbor_inside and occupied[neighbor]:
                continue
            face = []
            for corner in corners:
                lattice_vertex = tuple(index[axis] + corner[axis] for axis in range(3))
                vertex_id = vertex_lookup.get(lattice_vertex)
                if vertex_id is None:
                    vertex_id = len(vertices)
                    vertex_lookup[lattice_vertex] = vertex_id
                    vertices.append(lattice_vertex)
                face.append(vertex_id)
            faces.append(tuple(face))

    header = [
        "ply",
        "format ascii 1.0",
        "comment fused noisy-OR Gaussian occupancy surface",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(header) + "\n")
        for lattice_vertex in vertices:
            coordinate = spec.origin_m + np.asarray(lattice_vertex) * spec.voxel_size_m
            stream.write(f"{coordinate[0]:.8g} {coordinate[1]:.8g} {coordinate[2]:.8g}\n")
        for face in faces:
            stream.write(f"4 {face[0]} {face[1]} {face[2]} {face[3]}\n")


def save_gaussian_support_preview(
    path: str | Path,
    volume: GaussianSupportVolume,
    q_threshold: float,
    occupancy_probability_threshold: float,
) -> None:
    """Save orthographic soft-support and overlap projections for inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projections = ((2, "XY"), (1, "XZ"), (0, "YZ"))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for column, (axis, label) in enumerate(projections):
        support_projection = volume.probability_grid.max(axis=axis).T
        overlap_projection = volume.overlap_count_grid.max(axis=axis).T
        occupied_projection = volume.occupied_grid.max(axis=axis).T
        axes[0, column].imshow(
            support_projection,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"{label}: accumulated support")
        overlap_image = axes[1, column].imshow(
            overlap_projection,
            origin="lower",
            cmap="viridis",
            vmin=0,
            interpolation="nearest",
        )
        if np.any(occupied_projection) and not np.all(occupied_projection):
            axes[1, column].contour(
                occupied_projection.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.7,
                origin="lower",
            )
        axes[1, column].set_title(f"{label}: contributor count")
        figure.colorbar(overlap_image, ax=axes[1, column], fraction=0.046, pad=0.04)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle(
        f"q >= {q_threshold:g}; fused occupancy P >= "
        f"{occupancy_probability_threshold:g} (white contour)"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def voxel_overlap_metrics(
    prediction_grid: np.ndarray,
    reference_grid: np.ndarray,
    spec: VoxelGridSpec,
    prediction_outside_count: int = 0,
    reference_mode: str = "solid_mesh",
    sigma_extent: float = 2.0,
    fill_prediction_interior: bool = True,
    prediction_voxelization_mode: str = "binary_ellipsoid_union",
    occupancy_probability_threshold: float | None = None,
) -> dict:
    prediction_grid = np.asarray(prediction_grid, dtype=bool)
    reference_grid = np.asarray(reference_grid, dtype=bool)
    if prediction_grid.shape != reference_grid.shape:
        raise ValueError("Prediction and reference voxel grids must have identical shapes")
    intersection = int(np.count_nonzero(prediction_grid & reference_grid))
    prediction_count = int(np.count_nonzero(prediction_grid)) + int(prediction_outside_count)
    reference_count = int(np.count_nonzero(reference_grid))
    union = prediction_count + reference_count - intersection
    precision = intersection / max(prediction_count, 1)
    recall = intersection / max(reference_count, 1)
    f_score = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "available": bool(prediction_count and reference_count),
        "iou_3d": float(intersection / max(union, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f_score": float(f_score),
        "intersection_voxel_count": intersection,
        "union_voxel_count": int(union),
        "prediction_voxel_count": prediction_count,
        "prediction_outside_grid_voxel_count": int(prediction_outside_count),
        "reference_voxel_count": reference_count,
        "reference_mode": reference_mode,
        "voxel_resolution": int(spec.resolution),
        "voxel_size_m": float(spec.voxel_size_m),
        "grid_origin_m": [float(value) for value in spec.origin_m],
        "grid_maximum_m": [float(value) for value in spec.maximum_m],
        "gaussian_sigma_extent": float(sigma_extent),
        "fill_prediction_interior": bool(fill_prediction_interior),
        "prediction_voxelization_mode": prediction_voxelization_mode,
        "occupancy_probability_threshold": (
            float(occupancy_probability_threshold)
            if occupancy_probability_threshold is not None
            else None
        ),
    }


def read_ascii_ply(path: str | Path) -> PlyTable:
    """Read the vertex table from the ASCII PLY files produced by this project."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        first = stream.readline().strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        vertex_count: int | None = None
        properties: list[str] = []
        reading_vertex_properties = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            text = line.strip()
            if text.startswith("format ") and "ascii" not in text:
                raise ValueError(f"Only ASCII PLY is supported: {path}")
            if text.startswith("element "):
                tokens = text.split()
                reading_vertex_properties = len(tokens) >= 3 and tokens[1] == "vertex"
                if reading_vertex_properties:
                    vertex_count = int(tokens[2])
            elif text.startswith("property ") and reading_vertex_properties:
                tokens = text.split()
                if len(tokens) != 3 or tokens[1] == "list":
                    raise ValueError(f"Unsupported vertex property in {path}: {text}")
                properties.append(tokens[2])
            elif text == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        if vertex_count == 0:
            values = np.empty((0, len(properties)), dtype=np.float64)
        else:
            rows = []
            for _ in range(vertex_count):
                line = stream.readline()
                if not line:
                    raise ValueError(f"PLY ended before {vertex_count} vertices: {path}")
                row = [float(value) for value in line.split()]
                if len(row) < len(properties):
                    raise ValueError(f"Short vertex row in {path}")
                rows.append(row[: len(properties)])
            values = np.asarray(rows, dtype=np.float64)
    return PlyTable(properties, values)


def write_ascii_ply(path: str | Path, table: PlyTable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(table.values)}",
        *[f"property float {name}" for name in table.properties],
        "end_header",
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(header) + "\n")
        if len(table.values):
            np.savetxt(stream, table.values, fmt="%.8g")


def filter_by_support(table: PlyTable, threshold: float) -> PlyTable:
    support = table.column("target_support_probability")
    return PlyTable(table.properties, table.values[support >= float(threshold)].copy())


class _UnionFind:
    def __init__(self, count: int):
        self.parent = np.arange(count, dtype=np.int64)
        self.size = np.ones(count, dtype=np.int64)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def intrinsic_point_metrics(
    points: np.ndarray,
    support: np.ndarray,
    graph_neighbors: int = 8,
    connectivity_radius: float = 0.0,
    outlier_multiplier: float = 3.0,
) -> dict:
    """Measure compactness without claiming ground-truth geometric accuracy.

    The kNN outlier ratio flags locally isolated points. Connectivity is built
    from a capped kNN graph, rather than all radius pairs, so dense clouds do
    not create an unbounded edge set.
    """
    points = np.asarray(points, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    count = len(points)
    result: dict[str, float | int | list[float] | None] = {
        "point_count": int(count),
        "support_mean": float(support.mean()) if count else None,
        "support_median": float(np.median(support)) if count else None,
    }
    if count == 0:
        result.update(
            {
                "bbox_extent_m": None,
                "bbox_diagonal_m": None,
                "bbox_volume_m3": None,
                "nearest_neighbor_median_m": None,
                "knn_distance_median_m": None,
                "knn_outlier_ratio": None,
                "connectivity_radius_m": None,
                "component_count": 0,
                "largest_component_ratio": None,
            }
        )
        return result

    extent = points.max(axis=0) - points.min(axis=0)
    result.update(
        {
            "bbox_extent_m": [float(value) for value in extent],
            "bbox_diagonal_m": float(np.linalg.norm(extent)),
            "bbox_volume_m3": float(np.prod(extent)),
        }
    )
    if count == 1:
        result.update(
            {
                "nearest_neighbor_median_m": None,
                "knn_distance_median_m": None,
                "knn_outlier_ratio": 0.0,
                "connectivity_radius_m": None,
                "component_count": 1,
                "largest_component_ratio": 1.0,
            }
        )
        return result

    k = min(max(1, int(graph_neighbors)), count - 1)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k + 1, workers=-1)
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    nearest = distances[:, 1]
    kth = distances[:, -1]
    median_nearest = float(np.median(nearest))
    median_kth = float(np.median(kth))
    isolation_cutoff = float(outlier_multiplier) * max(median_kth, 1.0e-12)
    radius = float(connectivity_radius)
    if radius <= 0:
        radius = float(outlier_multiplier) * max(median_nearest, 1.0e-12)

    union = _UnionFind(count)
    for row in range(count):
        for offset in range(1, k + 1):
            if distances[row, offset] <= radius:
                union.union(row, int(indices[row, offset]))
    component_sizes: dict[int, int] = {}
    for index in range(count):
        root = union.find(index)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    result.update(
        {
            "nearest_neighbor_median_m": median_nearest,
            "knn_distance_median_m": median_kth,
            "knn_distance_p95_m": float(np.percentile(kth, 95.0)),
            "knn_outlier_cutoff_m": isolation_cutoff,
            "knn_outlier_ratio": float(np.mean(kth > isolation_cutoff)),
            "connectivity_radius_m": radius,
            "component_count": int(len(component_sizes)),
            "largest_component_ratio": float(max(component_sizes.values()) / count),
        }
    )
    return result


def _sample_rows(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or len(values) <= count:
        return values
    generator = np.random.default_rng(seed)
    return values[generator.choice(len(values), size=count, replace=False)]


def _sample_triangles(
    triangles: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.float64)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 0
    triangles = triangles[valid]
    cross = cross[valid]
    areas = areas[valid]
    if not len(triangles):
        raise ValueError("Mesh contains no non-degenerate triangles")
    generator = np.random.default_rng(seed)
    chosen = generator.choice(
        len(triangles), size=sample_count, replace=True, p=areas / areas.sum()
    )
    selected = triangles[chosen]
    uv = generator.random((sample_count, 2))
    fold = uv.sum(axis=1) > 1.0
    uv[fold] = 1.0 - uv[fold]
    samples = selected[:, 0].copy()
    samples += uv[:, :1] * (selected[:, 1] - selected[:, 0])
    samples += uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    normals = cross[chosen]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    return samples, normals


def _load_obj_surface(
    path: Path, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray | None]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            polygon = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for offset in range(1, len(polygon) - 1):
                faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    vertex_array = np.asarray(vertices, dtype=np.float64)
    if not faces or sample_count <= 0:
        return _sample_rows(vertex_array, sample_count, seed), None

    triangles = vertex_array[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not np.any(np.linalg.norm(cross, axis=1) > 0):
        return _sample_rows(vertex_array, sample_count, seed), None
    return _sample_triangles(triangles, sample_count, seed)


def _load_glb_surface(
    path: Path, sample_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "GLB/GLTF evaluation requires trimesh; run pip install -r requirements.txt"
        ) from error
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded
    if isinstance(mesh, (list, tuple)):
        mesh = trimesh.util.concatenate(mesh)
    if not hasattr(mesh, "triangles") or len(mesh.triangles) == 0:
        raise ValueError(f"GLB/GLTF contains no usable triangle mesh: {path}")
    count = sample_count if sample_count > 0 else max(len(mesh.vertices), 50000)
    return _sample_triangles(np.asarray(mesh.triangles), count, seed)


def load_reference_geometry(
    path: str | Path,
    sample_count: int = 50000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _load_obj_surface(path, sample_count, seed)
    if suffix in {".glb", ".gltf"}:
        return _load_glb_surface(path, sample_count, seed)
    if suffix == ".ply":
        table = read_ascii_ply(path)
        points = np.column_stack([table.column(axis) for axis in ("x", "y", "z")])
        normals = None
        if all(name in table.properties for name in ("nx", "ny", "nz")):
            normals = np.column_stack([table.column(axis) for axis in ("nx", "ny", "nz")])
        if sample_count > 0 and len(points) > sample_count:
            generator = np.random.default_rng(seed)
            chosen = generator.choice(len(points), size=sample_count, replace=False)
            points = points[chosen]
            normals = normals[chosen] if normals is not None else None
        return points, normals
    if suffix == ".npy":
        values = np.asarray(np.load(path), dtype=np.float64)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive else archive.files[0]
        values = np.asarray(archive[key], dtype=np.float64)
    else:
        values = np.asarray(np.loadtxt(path), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Reference geometry must have shape [N,3+]: {path}")
    values = _sample_rows(values, sample_count, seed)
    normals = values[:, 3:6] if values.shape[1] >= 6 else None
    return values[:, :3], normals


def ground_truth_metrics(
    prediction_points: np.ndarray,
    reference_points: np.ndarray,
    thresholds_m: Iterable[float],
    prediction_normals: np.ndarray | None = None,
    reference_normals: np.ndarray | None = None,
) -> dict:
    prediction_points = np.asarray(prediction_points, dtype=np.float64)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    if not len(prediction_points) or not len(reference_points):
        return {"available": False, "reason": "empty prediction or reference geometry"}
    reference_tree = cKDTree(reference_points)
    prediction_tree = cKDTree(prediction_points)
    accuracy, nearest_reference = reference_tree.query(prediction_points, k=1, workers=-1)
    completeness, _ = prediction_tree.query(reference_points, k=1, workers=-1)
    reference_extent = reference_points.max(axis=0) - reference_points.min(axis=0)
    reference_diagonal = float(np.linalg.norm(reference_extent))
    chamfer_l1 = float(0.5 * (np.mean(accuracy) + np.mean(completeness)))
    chamfer_l2_m2 = float(0.5 * (np.mean(accuracy**2) + np.mean(completeness**2)))
    hausdorff_p95 = float(
        max(np.percentile(accuracy, 95.0), np.percentile(completeness, 95.0))
    )
    result: dict[str, object] = {
        "available": True,
        "accuracy_mean_m": float(np.mean(accuracy)),
        "accuracy_median_m": float(np.median(accuracy)),
        "accuracy_p95_m": float(np.percentile(accuracy, 95.0)),
        "completeness_mean_m": float(np.mean(completeness)),
        "completeness_median_m": float(np.median(completeness)),
        "completeness_p95_m": float(np.percentile(completeness, 95.0)),
        "chamfer_l1_m": chamfer_l1,
        "chamfer_l2_m2": chamfer_l2_m2,
        "chamfer_l2_rmse_m": float(np.sqrt(chamfer_l2_m2)),
        "hausdorff_p95_m": hausdorff_p95,
        "hausdorff_m": float(max(np.max(accuracy), np.max(completeness))),
        "reference_bbox_diagonal_m": reference_diagonal,
        "normalized_chamfer_l1": (
            chamfer_l1 / reference_diagonal if reference_diagonal > 0 else None
        ),
        "normalized_hausdorff_p95": (
            hausdorff_p95 / reference_diagonal if reference_diagonal > 0 else None
        ),
    }
    scores = {}
    for threshold in thresholds_m:
        threshold = float(threshold)
        precision = float(np.mean(accuracy <= threshold))
        recall = float(np.mean(completeness <= threshold))
        f_score = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        scores[f"{threshold:g}"] = {
            "threshold_m": threshold,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
        }
    result["distance_threshold_scores"] = scores

    if prediction_normals is not None and reference_normals is not None:
        predicted = np.asarray(prediction_normals, dtype=np.float64)
        reference = np.asarray(reference_normals, dtype=np.float64)[nearest_reference]
        predicted /= np.maximum(np.linalg.norm(predicted, axis=1, keepdims=True), 1.0e-12)
        reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1.0e-12)
        result["normal_consistency_abs_cosine"] = float(
            np.mean(np.abs(np.sum(predicted * reference, axis=1)))
        )
    return result
