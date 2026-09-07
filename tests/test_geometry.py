import math

import torch

from radar_gs.geometry import projection_coordinates, select_by_los_angle


def test_projection_sign_convention():
    point = torch.tensor([[1.0, 2.0, 3.0]])
    kr = torch.tensor([0.0, 0.0, 1.0])
    kd = torch.tensor([1.0, 0.0, 0.0])
    row, col, depth = projection_coordinates(point, kr, kd, 101, 101, 0.5, 0.25)
    assert torch.allclose(row, torch.tensor([44.0]))
    assert torch.allclose(col, torch.tensor([54.0]))
    assert torch.allclose(depth, torch.tensor([3.0]))


def test_larger_los_threshold_retains_fewer_views():
    angles = torch.deg2rad(torch.arange(0.0, 31.0, 1.0))
    kr = torch.stack((torch.sin(angles), torch.zeros_like(angles), torch.cos(angles)), dim=-1)
    counts = [len(select_by_los_angle(kr, threshold)) for threshold in (2.0, 5.0, 10.0)]
    assert counts[0] >= counts[1] >= counts[2]
    assert counts == [16, 7, 4]

