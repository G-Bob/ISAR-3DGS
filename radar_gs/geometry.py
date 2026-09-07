from __future__ import annotations

import math
from typing import Tuple

import torch


def normalize(v: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def orthonormalize_view(
    kr: torch.Tensor, kd: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the ISAR basis using the established GEIST convention.

    kr is target-to-radar (the range observation direction), kd is the image
    horizontal/right direction, and n = kr x kd.
    """
    kr = normalize(kr)
    kd = kd - (kd * kr).sum(dim=-1, keepdim=True) * kr
    kd = normalize(kd)
    normal = normalize(torch.cross(kr, kd, dim=-1))
    return kr, kd, normal


def los_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = normalize(a)
    b = normalize(b)
    cosine = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def select_by_los_angle(
    kr: torch.Tensor,
    minimum_angle_deg: float,
    keep_last: bool = True,
) -> list[int]:
    """Greedily retain views separated by a minimum LOS angle.

    A larger ``minimum_angle_deg`` can only retain the same number or fewer
    frames for a given encounter. The first frame is always retained.
    """
    if kr.ndim != 2 or kr.shape[1] != 3:
        raise ValueError(f"Expected kr with shape [N,3], got {tuple(kr.shape)}")
    if len(kr) == 0:
        return []
    if minimum_angle_deg <= 0:
        return list(range(len(kr)))

    kr = normalize(kr)
    # Parameterize the encounter by cumulative LOS angular travel. Sampling
    # this one-dimensional coordinate guarantees that raising the threshold
    # reduces (or preserves) the number of retained frames.
    increments = los_angle_deg(kr[1:], kr[:-1])
    cumulative = torch.cat((torch.zeros(1, device=kr.device), increments.cumsum(0)))
    selected = [0]
    next_angle = minimum_angle_deg
    for index in range(1, len(kr)):
        if float(cumulative[index]) + 1.0e-7 >= next_angle:
            selected.append(index)
            next_angle += minimum_angle_deg

    if keep_last and selected[-1] != len(kr) - 1:
        selected.append(len(kr) - 1)
    return selected


def projection_coordinates(
    points: torch.Tensor,
    kr: torch.Tensor,
    kd: torch.Tensor,
    height: int,
    width: int,
    dr: float,
    dd: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D points using the project's range-Doppler sign convention.

    row = cy - dot(kr, P) / dr
    col = cx + dot(kd, P) / dd

    The returned depth is dot(kr, P); larger depth is closer to a monostatic
    radar located in the positive kr direction.
    """
    kr, kd, _ = orthonormalize_view(kr, kd)
    cy = (height - 1.0) / 2.0
    cx = (width - 1.0) / 2.0
    depth = points @ kr
    row = cy - depth / dr
    col = cx + (points @ kd) / dd
    return row, col, depth


def projection_jacobian(
    kr: torch.Tensor, kd: torch.Tensor, dr: float, dd: float
) -> torch.Tensor:
    kr, kd, _ = orthonormalize_view(kr, kd)
    return torch.stack((-kr / dr, kd / dd), dim=0)


def positional_encoding(x: torch.Tensor, levels: int) -> torch.Tensor:
    if levels <= 0:
        return x
    encoded = [x]
    for level in range(levels):
        frequency = (2.0**level) * math.pi
        encoded.extend((torch.sin(frequency * x), torch.cos(frequency * x)))
    return torch.cat(encoded, dim=-1)


def build_surface_basis(normals: torch.Tensor) -> torch.Tensor:
    """Construct right-handed [t1, t2, normal] bases."""
    normals = normalize(normals)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=normals.device, dtype=normals.dtype)
    z_axis = z_axis.expand_as(normals)
    y_axis = y_axis.expand_as(normals)
    use_y = normals[:, 2].abs() > 0.9
    reference = torch.where(use_y[:, None], y_axis, z_axis)
    tangent_1 = normalize(torch.cross(reference, normals, dim=-1))
    tangent_2 = normalize(torch.cross(normals, tangent_1, dim=-1))
    return torch.stack((tangent_1, tangent_2, normals), dim=-1)
