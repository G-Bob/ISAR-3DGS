import pytest
import torch

from radar_gs.model import RadarGaussianScene


@pytest.mark.parametrize("mode", ["offset_udf", "offset_sdf"])
def test_offset_surface_field_has_finite_trainable_losses(mode):
    torch.manual_seed(4)
    means = torch.randn(32, 3) * 0.2
    normals = torch.nn.functional.normalize(torch.randn(32, 3), dim=-1)
    scene = RadarGaussianScene(
        means,
        normals,
        initial_tangent_scale=0.04,
        initial_normal_scale=0.012,
        bbox=2.0,
        surface_field_mode=mode,
    )
    losses = scene.surface_field_losses(
        support=torch.full((32,), 0.8),
        evidence=torch.full((32,), 0.5),
        minimum_support=0.3,
        minimum_evidence=1.0e-4,
        sample_count=16,
        eikonal_jitter=0.08,
        offset_distance=0.05,
    )
    total = losses["zero"] + losses["normal"] + losses["eikonal"] + losses["offset"]
    assert torch.isfinite(total)
    assert losses["selected"] == 16
    assert float(losses["offset"]) > 0.0
    total.backward()
    gradient_norm = sum(
        float(parameter.grad.norm())
        for parameter in scene.surface_field.parameters()
        if parameter.grad is not None
    )
    assert gradient_norm > 0.0


def test_legacy_udf_still_returns_offset_key():
    means = torch.randn(8, 3)
    normals = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    scene = RadarGaussianScene(
        means, normals, 0.04, 0.012, 2.0, surface_field_mode="udf"
    )
    losses = scene.surface_field_losses(
        torch.ones(8), torch.ones(8), 0.3, 1.0e-4, 4, 0.08
    )
    assert float(losses["offset"]) == 0.0


def test_geometry_normal_loss_updates_gaussian_orientation_only():
    torch.manual_seed(9)
    means = torch.randn(24, 3) * 0.2
    normals = torch.nn.functional.normalize(torch.randn(24, 3), dim=-1)
    scene = RadarGaussianScene(
        means, normals, 0.04, 0.012, 2.0, surface_field_mode="offset_sdf"
    )
    losses = scene.surface_field_losses(
        torch.ones(24), torch.ones(24), 0.3, 1.0e-4, 12, 0.08,
        offset_distance=0.05,
        offset_loss_mode="normalized_l1",
    )
    losses["geometry_normal"].backward()
    assert scene.normal_parameters.grad is not None
    assert float(scene.normal_parameters.grad.norm()) > 0.0
    assert scene.means.grad is None
    assert all(parameter.grad is None for parameter in scene.surface_field.parameters())


def test_normalized_l1_and_detached_geometry_train_only_field():
    means = torch.randn(16, 3) * 0.2
    normals = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1)
    scene = RadarGaussianScene(
        means, normals, 0.04, 0.012, 2.0, surface_field_mode="offset_sdf"
    )
    losses = scene.surface_field_losses(
        torch.ones(16), torch.ones(16), 0.3, 1.0e-4, 8, 0.08,
        offset_distance=0.05,
        offset_loss_mode="normalized_l1",
        detach_geometry=True,
    )
    total = losses["zero"] + losses["offset"] + losses["eikonal"]
    total.backward()
    assert scene.means.grad is None
    assert scene.normal_parameters.grad is None
    assert any(
        parameter.grad is not None and float(parameter.grad.norm()) > 0.0
        for parameter in scene.surface_field.parameters()
    )
