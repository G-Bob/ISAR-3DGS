import torch

from radar_gs.model import RadarGaussianScene


def test_prune_eligibility_requires_independent_negative_evidence_gate():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    with torch.no_grad():
        scene.support_logits.fill_(-8.0)
    result = scene.adaptive_control(
        evidence=torch.zeros(2),
        mean_gradient=torch.zeros(2),
        visible_fraction=torch.ones(2),
        support_temperature=1.0,
        prune_support_threshold=0.05,
        prune_evidence_threshold=1.0e-4,
        prune_min_visible_fraction=0.3,
        split_support_threshold=0.5,
        split_gradient_threshold=1.0,
        maximum_gaussians=2,
        minimum_gaussians=1,
        split_mode="none",
        prune_eligible=torch.tensor([False, True]),
    )
    assert result["pruned"] == 1
    assert torch.allclose(scene.means[0], torch.tensor([0.0, 0.0, 0.0]))


def _scene(means: torch.Tensor, tangent_scale: float = 0.1) -> RadarGaussianScene:
    normals = torch.zeros_like(means)
    normals[:, 2] = 1.0
    return RadarGaussianScene(
        means=means,
        normals=normals,
        initial_tangent_scale=tangent_scale,
        initial_normal_scale=0.01,
        bbox=2.0,
        initial_support_probability=0.9,
        graph_neighbors=1,
    )


def test_adaptive_split_is_symmetric_and_uses_both_tangent_axes():
    original = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    scene = _scene(original)
    result = scene.adaptive_control(
        evidence=torch.ones(2),
        mean_gradient=torch.ones(2),
        visible_fraction=torch.ones(2),
        support_temperature=1.0,
        prune_support_threshold=0.05,
        prune_evidence_threshold=1.0e-4,
        prune_min_visible_fraction=0.3,
        split_support_threshold=0.5,
        split_gradient_threshold=0.5,
        maximum_gaussians=4,
        minimum_gaussians=1,
        split_displacement_ratio=0.3,
        split_scale_shrink=0.8,
    )

    source_indices = result["source_indices"]
    assert result["split"] == 2
    assert result["split_axis_t1"] == 1
    assert result["split_axis_t2"] == 1
    assert result["symmetric_split"] is True

    for split_order, (child_row, original_index) in enumerate(
        zip(range(2, 4), source_indices[2:])
    ):
        parent_row = int(original_index)
        midpoint = 0.5 * (scene.means[parent_row] + scene.means[child_row])
        assert torch.allclose(midpoint, original[original_index], atol=1.0e-7)
        displacement = scene.means[child_row] - scene.means[parent_row]
        if split_order % 2 == 0:
            assert displacement[0].abs() > 0
            assert torch.allclose(displacement[1:], torch.zeros(2), atol=1.0e-7)
        else:
            assert displacement[1].abs() > 0
            assert torch.allclose(
                displacement[[0, 2]], torch.zeros(2), atol=1.0e-7
            )

    expected_scale = torch.full((4, 2), 0.08)
    assert torch.allclose(scene.scales()[:, :2], expected_scale, atol=1.0e-7)


def test_surface_gap_ignores_large_structural_separation():
    local_scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]), 0.05)
    local_terms = local_scene.surface_loss(
        support=torch.ones(2),
        gap_scale_ratio=1.5,
        gap_max_distance=0.25,
        gap_min_normal_cosine=0.8,
    )
    assert local_terms["surface_gap"] > 0

    separated_scene = _scene(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), 0.05
    )
    separated_terms = separated_scene.surface_loss(
        support=torch.ones(2),
        gap_scale_ratio=1.5,
        gap_max_distance=0.25,
        gap_min_normal_cosine=0.8,
    )
    assert torch.allclose(separated_terms["surface_gap"], torch.tensor(0.0))


def test_center_gradient_split_is_symmetric_and_tangent():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0]]))
    scene.adaptive_control(
        evidence=torch.ones(1), mean_gradient=torch.ones(1),
        visible_fraction=torch.ones(1), support_temperature=1.0,
        prune_support_threshold=0.05, prune_evidence_threshold=1.0e-4,
        prune_min_visible_fraction=0.3, split_support_threshold=0.5,
        split_gradient_threshold=0.5, maximum_gaussians=2,
        minimum_gaussians=1, split_mode="symmetric",
        split_direction_mode="center_gradient",
        split_direction_vector=torch.tensor([[1.0, 2.0, 5.0]]),
    )
    delta = scene.means[1] - scene.means[0]
    assert torch.allclose(delta[2], torch.tensor(0.0), atol=1.0e-7)
    assert delta[:2].norm() > 0
    assert torch.allclose(scene.means.mean(0), torch.zeros(3), atol=1.0e-7)


def test_tangent_repulsion_penalizes_close_neighbors():
    close = _scene(torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]))
    far = _scene(torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]))
    assert close.surface_loss(support=torch.ones(2))["tangent_repulsion"] > 0
    assert torch.allclose(
        far.surface_loss(support=torch.ones(2))["tangent_repulsion"],
        torch.tensor(0.0), atol=1.0e-8,
    )


def test_support_losses_can_ignore_unobserved_gaussians():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    support = torch.tensor([0.8, 0.2])
    terms = scene.support_losses(
        support,
        spatial_sigma=0.25,
        normal_sigma=0.25,
        observation_weight=torch.tensor([1.0, 0.0]),
    )
    assert torch.allclose(terms["support_sparsity"], torch.tensor(0.8))
    expected_entropy = -(0.8 * torch.log(torch.tensor(0.8)) + 0.2 * torch.log(torch.tensor(0.2)))
    assert torch.allclose(terms["support_entropy"], expected_entropy)


def test_tangent_isotropy_penalizes_only_unequal_tangent_scales():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    equal = scene.surface_loss()["tangent_isotropy"]
    assert torch.allclose(equal, torch.tensor(0.0))
    scene.log_scales.data[0, 0] += torch.log(torch.tensor(2.0))
    unequal = scene.surface_loss()["tangent_isotropy"]
    assert unequal > 0


def test_covariance_consistency_penalizes_neighbor_shape_mismatch():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    equal = scene.surface_loss(support=torch.ones(2))["covariance_consistency"]
    assert torch.allclose(equal, torch.tensor(0.0), atol=1.0e-8)
    scene.log_scales.data[0, 0] += torch.log(torch.tensor(2.0))
    unequal = scene.surface_loss(support=torch.ones(2))["covariance_consistency"]
    assert unequal > 0


def test_local_pca_split_uses_geometry_instead_of_alternating_axes():
    original = torch.tensor(
        [[-0.1, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    )
    scene = RadarGaussianScene(
        means=original,
        normals=torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1),
        initial_tangent_scale=0.1,
        initial_normal_scale=0.01,
        bbox=2.0,
        initial_support_probability=0.9,
        graph_neighbors=2,
    )
    result = scene.adaptive_control(
        evidence=torch.ones(3),
        mean_gradient=torch.ones(3),
        visible_fraction=torch.ones(3),
        support_temperature=1.0,
        prune_support_threshold=0.05,
        prune_evidence_threshold=1.0e-4,
        prune_min_visible_fraction=0.3,
        split_support_threshold=0.5,
        split_gradient_threshold=0.5,
        maximum_gaussians=6,
        minimum_gaussians=1,
        split_direction_mode="local_pca",
    )
    assert result["split"] == 3
    assert result["split_local_pca"] == 3
    for child_row, original_index in zip(range(3, 6), result["source_indices"][3:]):
        parent_row = int(original_index)
        displacement = scene.means[child_row] - scene.means[parent_row]
        assert displacement[0].abs() > 0
        assert torch.allclose(displacement[1:], torch.zeros(2), atol=1.0e-7)


def test_asymmetric_conservative_split_preserves_support_mass():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    temperature = 0.6
    before = scene.support_probability(temperature)[0]
    result = scene.adaptive_control(
        evidence=torch.ones(2),
        mean_gradient=torch.ones(2),
        visible_fraction=torch.ones(2),
        support_temperature=temperature,
        prune_support_threshold=0.05,
        prune_evidence_threshold=1.0e-4,
        prune_min_visible_fraction=0.3,
        split_support_threshold=0.5,
        split_gradient_threshold=0.5,
        maximum_gaussians=3,
        minimum_gaussians=1,
        split_mode="asymmetric_conservative",
    )
    assert result["split"] == 1
    parent = int(result["split_reset_positions"][0])
    child = int(result["split_reset_positions"][1])
    q = scene.support_probability(temperature)
    combined = 1.0 - (1.0 - q[parent]) * (1.0 - q[child])
    assert torch.allclose(combined, before, atol=1.0e-6)
    assert not torch.allclose(scene.means[parent], scene.means[child])


def test_split_eligibility_blocks_only_ineligible_candidates():
    scene = _scene(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    result = scene.adaptive_control(
        evidence=torch.ones(2),
        mean_gradient=torch.ones(2),
        visible_fraction=torch.ones(2),
        support_temperature=1.0,
        prune_support_threshold=0.05,
        prune_evidence_threshold=1.0e-4,
        prune_min_visible_fraction=0.3,
        split_support_threshold=0.5,
        split_gradient_threshold=0.5,
        maximum_gaussians=4,
        minimum_gaussians=1,
        split_mode="asymmetric",
        split_eligible=torch.tensor([False, True]),
    )
    assert result["split"] == 1
    assert result["source_indices"][-1].item() == 1
    assert result["split_reset_positions"].tolist() == [1, 2]


def test_pca_spectral_gate_accepts_plane_and_rejects_volume():
    plane_xy = torch.tensor(
        [[x, y, 0.0] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)]
    )
    plane = RadarGaussianScene(
        means=plane_xy,
        normals=torch.tensor([[1.0, 0.0, 0.0]]).repeat(9, 1),
        initial_tangent_scale=0.04,
        initial_normal_scale=0.01,
        bbox=2.0,
        initial_support_probability=0.9,
        graph_neighbors=8,
    )
    accepted = plane.refresh_normals_pca(
        torch.ones(9),
        hard_reset=True,
        maximum_surface_variation=0.05,
        minimum_planar_spread_ratio=0.15,
    )
    assert accepted["updated"] == 9
    assert torch.allclose(plane.normals()[:, 2].abs(), torch.ones(9), atol=1.0e-6)

    cube_xyz = torch.tensor(
        [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)]
    )
    volume = RadarGaussianScene(
        means=cube_xyz,
        normals=torch.tensor([[0.0, 0.0, 1.0]]).repeat(8, 1),
        initial_tangent_scale=0.04,
        initial_normal_scale=0.01,
        bbox=2.0,
        initial_support_probability=0.9,
        graph_neighbors=7,
    )
    rejected = volume.refresh_normals_pca(
        torch.ones(8), hard_reset=True, maximum_surface_variation=0.05
    )
    assert rejected["updated"] == 0


def test_pca_distance_gate_rejects_wide_neighborhood():
    means = torch.tensor(
        [[x, y, 0.0] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)]
    )
    scene = _scene(means)
    result = scene.refresh_normals_pca(
        torch.ones(9), hard_reset=True, maximum_neighbor_distance=0.05
    )
    assert result["updated"] == 0


def test_isotropic_geometric_covariance_removes_tangent_anisotropy():
    scene = RadarGaussianScene(
        means=torch.zeros((1, 3)),
        normals=torch.tensor([[0.2, 0.3, 0.9327379]]),
        initial_tangent_scale=0.1,
        initial_normal_scale=0.01,
        bbox=2.0,
        graph_neighbors=1,
        tangent_covariance_mode="isotropic_geometric",
    )
    with torch.no_grad():
        scene.log_scales[0] = torch.tensor([0.05, 0.20, 0.01]).log()
    eigenvalues = torch.linalg.eigvalsh(scene.covariance()[0])
    expected = torch.tensor([0.0001, 0.01, 0.01])
    assert torch.allclose(eigenvalues, expected, atol=1.0e-6)


def test_normal_covariance_floor_thickens_rendered_surface_only():
    scene = RadarGaussianScene(
        means=torch.zeros((1, 3)),
        normals=torch.tensor([[0.2, 0.3, 0.9327379]]),
        initial_tangent_scale=0.1,
        initial_normal_scale=0.01,
        bbox=2.0,
        graph_neighbors=1,
        tangent_covariance_mode="isotropic_geometric",
        normal_covariance_floor_ratio=0.5,
    )
    eigenvalues = torch.linalg.eigvalsh(scene.covariance()[0])
    expected = torch.tensor([0.0025, 0.01, 0.01])
    assert torch.allclose(eigenvalues, expected, atol=1.0e-6)
    assert torch.allclose(scene.scales()[0], torch.tensor([0.1, 0.1, 0.01]))
