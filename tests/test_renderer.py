import torch

from radar_gs.model import RadarGaussianScene
from radar_gs.renderer import RadarVisibilityRenderer


def test_near_gaussian_attenuates_far_gaussian_and_gradients_exist():
    # Both Gaussians share a LOS in the (kd, n) visibility plane but occupy
    # different range bins in the rendered ISAR image.
    means = torch.tensor([[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    scene = RadarGaussianScene(
        means,
        normals,
        initial_tangent_scale=0.10,
        initial_normal_scale=0.02,
        bbox=1.0,
        feature_dim=4,
        encoding_levels=1,
        hidden_dim=8,
    )
    scene.support_logits.data.fill_(4.0)
    renderer = RadarVisibilityRenderer(48, 48, depth_bins=16, kernel_radius=3)
    result = renderer(
        scene,
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        dr=0.04,
        dd=0.04,
        amplitudes=torch.ones(2),
    )
    assert result.gaussian_visibility[0] > result.gaussian_visibility[1]
    assert result.image.sum() < result.raw_image.sum()
    assert float(result.support_map.min()) >= 0.0
    assert float(result.support_map.max()) <= 1.0
    assert float(result.overlap_map.min()) >= 0.0
    assert float(result.overlap_map.max()) <= 1.0
    assert result.overlap_map.sum() > 0
    assert result.overlap_map.sum() < result.support_map.sum()
    result.image.sum().backward()
    assert scene.means.grad is not None
    assert torch.isfinite(scene.means.grad).all()
