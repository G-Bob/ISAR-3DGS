from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import ISARView
from .geometry import normalize


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _radial_normals(points: torch.Tensor) -> torch.Tensor:
    centered = points - points.mean(dim=0, keepdim=True)
    fallback = torch.zeros_like(centered)
    fallback[:, 2] = 1.0
    valid = centered.norm(dim=-1, keepdim=True) > 1.0e-7
    return normalize(torch.where(valid, centered, fallback))


def random_volume(count: int, bbox: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    points = (torch.rand((count, 3), generator=_generator(seed)) * 2.0 - 1.0) * bbox
    return points, _radial_normals(points)


def sphere_surface(count: int, bbox: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    directions = normalize(torch.randn((count, 3), generator=_generator(seed)))
    radius = bbox * (0.75 + 0.1 * torch.rand((count, 1), generator=_generator(seed + 1)))
    return directions * radius, directions


def box_surface(count: int, bbox: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = _generator(seed)
    points = (torch.rand((count, 3), generator=generator) * 2.0 - 1.0) * bbox
    faces = torch.randint(0, 6, (count,), generator=generator)
    axes = faces // 2
    signs = torch.where(faces % 2 == 0, -1.0, 1.0)
    normals = torch.zeros_like(points)
    rows = torch.arange(count)
    points[rows, axes] = signs * bbox * 0.8
    normals[rows, axes] = signs
    return points, normals


def backprojection_surface(
    count: int,
    bbox: float,
    seed: int,
    views: list[ISARView],
    intensity_power: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not views:
        raise ValueError("backprojection initialization requires loaded ISAR views")
    generator = _generator(seed)
    points: list[torch.Tensor] = []
    view_count = min(len(views), count)
    chosen_views = [views[index] for index in np.linspace(0, len(views) - 1, view_count).round().astype(int)]
    per_view = max(1, int(np.ceil(count / len(chosen_views))))
    for view in chosen_views:
        image = view.image.clamp_min(0)
        weights = image.flatten().pow(intensity_power)
        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)
        replacement = per_view > int((weights > 0).sum())
        indices = torch.multinomial(weights, per_view, replacement=replacement, generator=generator)
        height, width = image.shape
        row = torch.div(indices, width, rounding_mode="floor").float()
        col = (indices % width).float()
        range_coordinate = ((height - 1.0) / 2.0 - row) * view.dr
        doppler_coordinate = (col - (width - 1.0) / 2.0) * view.dd
        unresolved = (torch.rand(per_view, generator=generator) - 0.5) * 0.04 * bbox
        position = (
            range_coordinate[:, None] * view.kr[None]
            + doppler_coordinate[:, None] * view.kd[None]
            + unresolved[:, None] * view.normal[None]
        )
        points.append(position)
    merged = torch.cat(points, dim=0)[:count].clamp(-bbox, bbox)
    return merged, _radial_normals(merged)


def _load_ascii_ply(path: Path) -> torch.Tensor:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"Only ASCII PLY is supported: {path}")
    vertex_count = None
    end_header = None
    for index, line in enumerate(lines):
        if line.startswith("format") and "ascii" not in line:
            raise ValueError(f"Only ASCII PLY is supported: {path}")
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            end_header = index + 1
            break
    if vertex_count is None or end_header is None:
        raise ValueError(f"Invalid PLY header: {path}")
    values = [[float(item) for item in line.split()[:6]] for line in lines[end_header:end_header + vertex_count]]
    return torch.tensor(values, dtype=torch.float32)


def pointcloud_surface(
    path: str | Path, count: int, bbox: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        values = torch.from_numpy(np.load(path)).float()
    elif path.suffix.lower() == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive else archive.files[0]
        values = torch.from_numpy(archive[key]).float()
    elif path.suffix.lower() == ".ply":
        values = _load_ascii_ply(path)
    else:
        values = torch.from_numpy(np.loadtxt(path)).float()
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("Point cloud must have shape [N,3+] or [N,6+] with normals")
    points = values[:, :3]
    normals = normalize(values[:, 3:6]) if values.shape[1] >= 6 else _radial_normals(points)
    generator = _generator(seed)
    indices = torch.randint(0, len(points), (count,), generator=generator)
    points = points[indices].clamp(-bbox, bbox)
    normals = normals[indices]
    return points, normals


def mesh_surface(
    path: str | Path, count: int, bbox: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    vertices: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not vertices or not faces:
        raise ValueError(f"OBJ contains no usable vertices/faces: {path}")
    vertex_tensor = torch.tensor(vertices, dtype=torch.float32)
    face_tensor = torch.tensor(faces, dtype=torch.long)
    triangles = vertex_tensor[face_tensor]
    edges_1 = triangles[:, 1] - triangles[:, 0]
    edges_2 = triangles[:, 2] - triangles[:, 0]
    cross = torch.cross(edges_1, edges_2, dim=-1)
    areas = 0.5 * cross.norm(dim=-1)
    generator = _generator(seed)
    chosen = torch.multinomial(areas.clamp_min(1.0e-12), count, replacement=True, generator=generator)
    triangles = triangles[chosen]
    normals = normalize(cross[chosen])
    uv = torch.rand((count, 2), generator=generator)
    fold = uv.sum(dim=-1) > 1.0
    uv[fold] = 1.0 - uv[fold]
    points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
    points = points + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    maximum = points.abs().max().clamp_min(1.0e-8)
    if maximum > bbox:
        points = points * (bbox * 0.95 / maximum)
    return points, normals


def initialize_surface(
    method: str,
    count: int,
    bbox: float,
    seed: int,
    views: list[ISARView] | None = None,
    init_path: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    method = method.lower()
    if method == "random_volume":
        return random_volume(count, bbox, seed)
    if method == "sphere_surface":
        return sphere_surface(count, bbox, seed)
    if method == "box_surface":
        return box_surface(count, bbox, seed)
    if method == "backprojection_surface":
        return backprojection_surface(count, bbox, seed, views or [])
    if method == "pointcloud_surface":
        if init_path is None:
            raise ValueError("pointcloud_surface requires --init_path")
        return pointcloud_surface(init_path, count, bbox, seed)
    if method == "mesh_surface":
        if init_path is None:
            raise ValueError("mesh_surface requires --init_path")
        return mesh_surface(init_path, count, bbox, seed)
    raise ValueError(f"Unknown initialization method: {method}")
