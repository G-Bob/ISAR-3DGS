import torch

from radar_gs.model import RadarGaussianScene


device = torch.device("cuda")
means = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], device=device)
normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device)
scene = RadarGaussianScene(
    means, normals, 0.04, 0.01, 2.0,
    initial_support_probability=0.9, graph_neighbors=1,
).to(device)
terms = scene.surface_loss(
    support=torch.ones(2, device=device), tangent_repulsion_min_ratio=0.75
)
assert torch.isfinite(terms["tangent_repulsion"])
result = scene.adaptive_control(
    evidence=torch.ones(2, device=device),
    mean_gradient=torch.ones(2, device=device),
    visible_fraction=torch.ones(2, device=device),
    support_temperature=1.0,
    prune_support_threshold=0.05,
    prune_evidence_threshold=1.0e-4,
    prune_min_visible_fraction=0.3,
    split_support_threshold=0.5,
    split_gradient_threshold=0.5,
    maximum_gaussians=4,
    minimum_gaussians=1,
    split_mode="symmetric",
    split_direction_mode="center_gradient",
    split_direction_vector=torch.tensor(
        [[1.0, 1.0, 3.0], [1.0, -1.0, 3.0]], device=device
    ),
    adaptive_knn_tangent_scale_ratio=0.8,
)
assert result["split"] == 2 and scene.means.is_cuda
print("CUDA_SMOKE_V13_OK", scene.count)
