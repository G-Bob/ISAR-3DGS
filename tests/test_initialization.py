import torch

from radar_gs.initialization import box_surface, random_volume, sphere_surface


def test_initializers_return_unit_normals():
    for initializer in (random_volume, sphere_surface, box_surface):
        points, normals = initializer(128, 2.0, 7)
        assert points.shape == (128, 3)
        assert normals.shape == (128, 3)
        assert float(points.abs().max()) <= 2.0 + 1.0e-6
        assert torch.allclose(normals.norm(dim=-1), torch.ones(128), atol=1.0e-5)

