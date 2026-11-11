from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import build_surface_basis, normalize, positional_encoding
from .losses import support_entropy


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1.0e-6)
    return torch.log(torch.expm1(value).clamp_min(1.0e-12))


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-5), 1.0 - 1.0e-5)
    return math.log(value / (1.0 - value))


def build_knn_graph(points: torch.Tensor, neighbors: int, chunk: int = 512) -> torch.Tensor:
    """Build a directed kNN graph without allocating a full NxN matrix."""
    count = len(points)
    if neighbors <= 0 or count <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    neighbors = min(int(neighbors), count - 1)
    points = points.detach().float().cpu()
    sources = []
    destinations = []
    all_indices = torch.arange(count)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        distance = torch.cdist(points[start:stop], points)
        local = torch.arange(stop - start)
        distance[local, all_indices[start:stop]] = float("inf")
        nearest = torch.topk(distance, k=neighbors, largest=False).indices
        source = all_indices[start:stop, None].expand_as(nearest)
        sources.append(source.reshape(-1))
        destinations.append(nearest.reshape(-1))
    return torch.stack((torch.cat(sources), torch.cat(destinations)), dim=0)


class ScatteringNetwork(nn.Module):
    """Small from-scratch geometry-conditioned amplitude network."""

    def __init__(
        self,
        feature_dim: int = 16,
        encoding_levels: int = 4,
        hidden_dim: int = 64,
        maximum_amplitude: float = 1.0,
        direction_encoding_levels: int | None = None,
    ):
        super().__init__()
        self.encoding_levels = int(encoding_levels)
        self.direction_encoding_levels = (
            self.encoding_levels if direction_encoding_levels is None
            else int(direction_encoding_levels)
        )
        if self.direction_encoding_levels < 0:
            raise ValueError("direction_encoding_levels must be nonnegative")
        self.maximum_amplitude = float(maximum_amplitude)
        encoded_xyz = 3 * (1 + 2 * encoding_levels)
        encoded_direction = 3 * (1 + 2 * self.direction_encoding_levels)
        input_dim = encoded_xyz + 2 * encoded_direction + 3 + feature_dim + 4
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        xyz_normalized: torch.Tensor,
        normals: torch.Tensor,
        features: torch.Tensor,
        kr: torch.Tensor,
        kd: torch.Tensor,
    ) -> torch.Tensor:
        count = len(xyz_normalized)
        kr_expanded = kr.expand(count, -1)
        kd_expanded = kd.expand(count, -1)
        incident = -kr_expanded
        explicit = torch.cat(
            (
                (normals * incident).sum(-1, keepdim=True),
                (normals * kd_expanded).sum(-1, keepdim=True),
                (xyz_normalized * kr_expanded).sum(-1, keepdim=True),
                (xyz_normalized * kd_expanded).sum(-1, keepdim=True),
            ),
            dim=-1,
        )
        inputs = torch.cat(
            (
                positional_encoding(xyz_normalized, self.encoding_levels),
                positional_encoding(kr_expanded, self.direction_encoding_levels),
                positional_encoding(kd_expanded, self.direction_encoding_levels),
                normals,
                features,
                explicit,
            ),
            dim=-1,
        )
        return self.maximum_amplitude * torch.sigmoid(self.net(inputs).squeeze(-1))


class SurfaceFieldNetwork(nn.Module):
    """Low-capacity smooth implicit field used only as a geometry prior."""

    def __init__(self, hidden_dim: int = 32, layers: int = 3):
        super().__init__()
        if hidden_dim <= 0 or layers < 2:
            raise ValueError("surface field requires hidden_dim > 0 and layers >= 2")
        modules: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.Softplus(beta=10.0)]
        for _ in range(layers - 2):
            modules.extend((nn.Linear(hidden_dim, hidden_dim), nn.Softplus(beta=10.0)))
        final = nn.Linear(hidden_dim, 1)
        nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(final.bias)
        modules.append(final)
        self.net = nn.Sequential(*modules)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)

    def forward(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        return self.net(xyz_normalized).squeeze(-1)


class RadarGaussianScene(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        normals: torch.Tensor,
        initial_tangent_scale: float,
        initial_normal_scale: float,
        bbox: float,
        feature_dim: int = 16,
        encoding_levels: int = 4,
        hidden_dim: int = 64,
        maximum_scattering_amplitude: float = 1.0,
        initial_support_probability: float = 0.7,
        background_levels: torch.Tensor | None = None,
        background_mode: str = "constant",
        graph_neighbors: int = 8,
        tangent_covariance_mode: str = "anisotropic",
        normal_covariance_floor_ratio: float = 0.0,
        primitive_mode: str = "gaussian3d",
        surfel_normal_scale: float = 0.002,
        surfel_fixed_normals: bool = False,
        surface_field_mode: str = "none",
        surface_field_hidden_dim: int = 32,
        surface_field_layers: int = 3,
        direction_encoding_levels: int | None = None,
    ):
        super().__init__()
        if means.shape != normals.shape or means.shape[-1] != 3:
            raise ValueError("means and normals must both have shape [G,3]")
        if background_mode not in {"constant", "planar"}:
            raise ValueError("background_mode must be constant or planar")
        if tangent_covariance_mode not in {"anisotropic", "isotropic_geometric"}:
            raise ValueError(
                "tangent_covariance_mode must be anisotropic or isotropic_geometric"
            )
        if normal_covariance_floor_ratio < 0.0:
            raise ValueError("normal_covariance_floor_ratio must be non-negative")
        if primitive_mode not in {"gaussian3d", "surfel2d"}:
            raise ValueError("primitive_mode must be gaussian3d or surfel2d")
        if surfel_normal_scale <= 0.0:
            raise ValueError("surfel_normal_scale must be positive")
        if surface_field_mode not in {"none", "udf", "offset_udf", "offset_sdf"}:
            raise ValueError(
                "surface_field_mode must be none, udf, offset_udf, or offset_sdf"
            )
        self.bbox = float(bbox)
        self.background_mode = background_mode
        self.graph_neighbors = int(graph_neighbors)
        self.tangent_covariance_mode = tangent_covariance_mode
        self.normal_covariance_floor_ratio = float(normal_covariance_floor_ratio)
        self.primitive_mode = primitive_mode
        self.surfel_normal_scale = float(surfel_normal_scale)
        self.surfel_fixed_normals = bool(surfel_fixed_normals)
        self.surface_field_mode = surface_field_mode
        self.means = nn.Parameter(means.clone())
        self.normal_parameters = nn.Parameter(normalize(normals.clone()))
        tangent = torch.full((len(means), 2), float(initial_tangent_scale)).log()
        normal = torch.full((len(means), 1), float(initial_normal_scale)).log()
        self.log_scales = nn.Parameter(torch.cat((tangent, normal), dim=-1))
        self.support_logits = nn.Parameter(
            torch.full((len(means),), _logit(initial_support_probability))
        )
        self.features = nn.Parameter(torch.randn(len(means), feature_dim) * 0.01)
        self.register_buffer("reference_normals", normalize(normals.clone()))
        self.register_buffer("graph_edges", build_knn_graph(means, graph_neighbors))

        if background_levels is None:
            background_levels = torch.empty(0)
        background_levels = background_levels.float().clamp_min(1.0e-6)
        coefficient_count = 1 if background_mode == "constant" else 3
        coefficients = torch.zeros((len(background_levels), coefficient_count))
        if len(background_levels):
            coefficients[:, 0] = _inverse_softplus(background_levels)
        self.background_coefficients = nn.Parameter(coefficients)
        self.scattering = ScatteringNetwork(
            feature_dim,
            encoding_levels,
            hidden_dim,
            maximum_scattering_amplitude,
            direction_encoding_levels,
        )
        if surface_field_mode != "none":
            # Do not let enabling the ablation change the legacy scene/scattering
            # RNG stream. This keeps controlled runs identical before the field
            # becomes active.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1729)
                self.surface_field = SurfaceFieldNetwork(
                    surface_field_hidden_dim, surface_field_layers
                )
        else:
            self.surface_field = None

    @property
    def count(self) -> int:
        return len(self.means)

    @property
    def training_view_count(self) -> int:
        return len(self.background_coefficients)

    def normals(self) -> torch.Tensor:
        if self.primitive_mode == "surfel2d" and self.surfel_fixed_normals:
            return normalize(self.reference_normals)
        return normalize(self.normal_parameters)

    def scales(self) -> torch.Tensor:
        scales = self.log_scales.exp().clamp(1.0e-4, self.bbox)
        if self.primitive_mode == "surfel2d":
            radius = torch.exp(
                self.log_scales[:, :2].mean(dim=-1, keepdim=True)
            ).clamp(1.0e-4, self.bbox)
            thickness = torch.full_like(radius, self.surfel_normal_scale)
            return torch.cat((radius, radius, thickness), dim=-1)
        return scales

    def support_probability(self, temperature: float = 1.0) -> torch.Tensor:
        return torch.sigmoid(self.support_logits / max(float(temperature), 1.0e-3))

    def opacity(self) -> torch.Tensor:
        """Compatibility alias; opacity is now the target-support probability."""
        return self.support_probability()

    def covariance(self) -> torch.Tensor:
        basis = build_surface_basis(self.normals())
        scales = self.scales()
        tangent_geometric = torch.exp(
            self.log_scales[:, :2].mean(dim=-1, keepdim=True)
        ).clamp(1.0e-4, self.bbox)
        if self.primitive_mode == "surfel2d":
            scales = torch.cat(
                (
                    tangent_geometric,
                    tangent_geometric,
                    torch.full_like(tangent_geometric, self.surfel_normal_scale),
                ),
                dim=-1,
            )
        elif self.tangent_covariance_mode == "isotropic_geometric":
            scales = torch.cat(
                (tangent_geometric, tangent_geometric, scales[:, 2:3]), dim=-1
            )
        if self.primitive_mode == "gaussian3d" and self.normal_covariance_floor_ratio > 0.0:
            normal_floor = self.normal_covariance_floor_ratio * tangent_geometric
            scales = torch.cat(
                (scales[:, :2], torch.maximum(scales[:, 2:3], normal_floor)), dim=-1
            )
        scales_squared = torch.diag_embed(scales.square())
        return basis @ scales_squared @ basis.transpose(-1, -2)

    def scattering_amplitude(self, kr: torch.Tensor, kd: torch.Tensor) -> torch.Tensor:
        xyz = (self.means / self.bbox).clamp(-1.5, 1.5)
        return self.scattering(xyz, self.normals(), self.features, kr, kd)

    def surface_field_value(self, points: torch.Tensor) -> torch.Tensor:
        if self.surface_field is None:
            raise RuntimeError("surface field is disabled")
        return self.surface_field((points / self.bbox).clamp(-1.5, 1.5)) * self.bbox

    def surface_field_losses(
        self,
        support: torch.Tensor,
        evidence: torch.Tensor,
        minimum_support: float,
        minimum_evidence: float,
        sample_count: int,
        eikonal_jitter: float,
        offset_distance: float = 0.05,
        offset_loss_mode: str = "smooth_l1",
        detach_geometry: bool = False,
        epsilon: float = 1.0e-4,
    ) -> dict[str, torch.Tensor | int]:
        """Evidence-gated zero-set, normal, and local Eikonal constraints."""
        zero = self.means.new_zeros(())
        if self.surface_field is None:
            return {
                "zero": zero, "normal": zero, "eikonal": zero,
                "offset": zero, "geometry_normal": zero, "selected": 0,
            }
        if evidence.shape != (self.count,) or support.shape != (self.count,):
            raise ValueError("support and evidence must have shape [G]")
        eligible = (support.detach() >= float(minimum_support)) & (
            evidence.detach() >= float(minimum_evidence)
        )
        candidates = torch.where(eligible)[0]
        if not len(candidates):
            return {
                "zero": zero, "normal": zero, "eikonal": zero,
                "offset": zero, "geometry_normal": zero, "selected": 0,
            }
        count = min(max(1, int(sample_count)), len(candidates))
        weights_all = (
            support.detach()[candidates] * evidence.detach()[candidates].clamp_min(0.0)
        )
        chosen = candidates[
            torch.topk(weights_all, k=count, largest=True, sorted=False).indices
        ]
        weights = support.detach()[chosen] * evidence.detach()[chosen].clamp_min(0.0)
        weights = weights / weights.sum().clamp_min(1.0e-8)

        if offset_loss_mode not in {"smooth_l1", "l1", "normalized_l1"}:
            raise ValueError("unsupported surface-field offset loss mode")
        points = self.means[chosen]
        if detach_geometry:
            points = points.detach().requires_grad_(True)
        values = self.surface_field_value(points)
        field_gradient = torch.autograd.grad(
            values.sum(), points, create_graph=True, retain_graph=True
        )[0]
        zero_residual = torch.sqrt(values.square() + float(epsilon) ** 2)
        if offset_loss_mode == "normalized_l1":
            zero_residual = zero_residual / max(float(offset_distance), 1.0e-4)
        zero_loss = (weights * zero_residual).sum()
        field_normals = normalize(field_gradient)
        target_normals = self.normals()[chosen]
        geometry_normal_loss = (
            weights
            * (1.0 - (field_normals.detach() * target_normals).sum(dim=-1).abs())
        ).sum()
        if detach_geometry:
            target_normals = target_normals.detach()
        normal_loss = (
            weights
            * (1.0 - (field_normals * target_normals).sum(dim=-1).abs())
        ).sum()

        offset_loss = zero
        offset_points = None
        if self.surface_field_mode in {"offset_udf", "offset_sdf"}:
            distance = max(float(offset_distance), 1.0e-4)
            chosen_normals = self.normals()[chosen].detach()
            signed_offsets = torch.cat(
                (
                    torch.full((count,), distance, device=points.device, dtype=points.dtype),
                    torch.full((count,), -distance, device=points.device, dtype=points.dtype),
                )
            )
            offset_points = torch.cat((points.detach(), points.detach()), dim=0)
            offset_normals = torch.cat((chosen_normals, chosen_normals), dim=0)
            offset_points = (
                offset_points + signed_offsets[:, None] * offset_normals
            ).requires_grad_(True)
            offset_values = self.surface_field_value(offset_points)
            offset_targets = (
                signed_offsets
                if self.surface_field_mode == "offset_sdf"
                else signed_offsets.abs()
            )
            paired_weights = torch.cat((weights, weights), dim=0) * 0.5
            if offset_loss_mode == "smooth_l1":
                offset_residual = F.smooth_l1_loss(
                    offset_values, offset_targets, reduction="none"
                )
            else:
                offset_residual = (offset_values - offset_targets).abs()
                if offset_loss_mode == "normalized_l1":
                    offset_residual = offset_residual / distance
            offset_loss = (paired_weights * offset_residual).sum()
            offset_gradient = torch.autograd.grad(
                offset_values.sum(), offset_points, create_graph=True, retain_graph=True
            )[0]
            expected_normals = offset_normals
            if self.surface_field_mode == "offset_udf":
                expected_normals = expected_normals * signed_offsets.sign()[:, None]
            normal_loss = (
                paired_weights
                * (1.0 - (normalize(offset_gradient) * expected_normals).sum(dim=-1))
            ).sum()
            eikonal_loss = (offset_gradient.norm(dim=-1) - 1.0).square().mean()
        else:
            eikonal_loss = None

        if eikonal_loss is None:
            cuda_devices = []
            if points.is_cuda:
                cuda_devices = [
                    points.device.index
                    if points.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(2718)
                jitter = torch.randn_like(points)
            jittered = (
                points.detach() + float(eikonal_jitter) * jitter
            ).requires_grad_(True)
            jittered_values = self.surface_field_value(jittered)
            jittered_gradient = torch.autograd.grad(
                jittered_values.sum(), jittered, create_graph=True, retain_graph=True
            )[0]
            eikonal_loss = (jittered_gradient.norm(dim=-1) - 1.0).square().mean()
        return {
            "zero": zero_loss,
            "normal": normal_loss,
            "eikonal": eikonal_loss,
            "offset": offset_loss,
            "geometry_normal": geometry_normal_loss,
            "selected": int(count),
        }

    def tangent_project_mean_gradients(self, eligible: torch.Tensor) -> int:
        if self.means.grad is None:
            return 0
        if eligible.shape != (self.count,):
            raise ValueError("eligible must have shape [G]")
        normals = self.normals().detach()
        gradient = self.means.grad
        normal_component = (gradient * normals).sum(dim=-1, keepdim=True) * normals
        gradient[eligible] = gradient[eligible] - normal_component[eligible]
        return int(eligible.sum().item())

    def project_means_to_surface(
        self,
        eligible: torch.Tensor,
        step: float = 0.25,
        maximum_displacement: float = 0.02,
    ) -> dict[str, float | int]:
        if self.surface_field is None:
            return {"projected": 0, "mean_displacement": 0.0, "maximum_displacement": 0.0}
        if eligible.shape != (self.count,):
            raise ValueError("eligible must have shape [G]")
        indices = torch.where(eligible)[0]
        if not len(indices):
            return {"projected": 0, "mean_displacement": 0.0, "maximum_displacement": 0.0}
        with torch.enable_grad():
            points = self.means.detach()[indices].requires_grad_(True)
            values = self.surface_field_value(points)
            gradient = torch.autograd.grad(values.sum(), points)[0]
            displacement = (
                float(step)
                * values[:, None]
                * gradient
                / gradient.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            )
            lengths = displacement.norm(dim=-1, keepdim=True)
            if maximum_displacement > 0.0:
                displacement = displacement * torch.clamp(
                    float(maximum_displacement) / lengths.clamp_min(1.0e-8), max=1.0
                )
        with torch.no_grad():
            self.means[indices] -= displacement.detach()
        final_lengths = displacement.detach().norm(dim=-1)
        return {
            "projected": int(len(indices)),
            "mean_displacement": float(final_lengths.mean().item()),
            "maximum_displacement": float(final_lengths.max().item()),
        }

    def background(
        self,
        view_index: int,
        height: int,
        width: int,
        detach: bool = False,
    ) -> torch.Tensor:
        if self.training_view_count == 0:
            return torch.zeros((height, width), device=self.means.device, dtype=self.means.dtype)
        index = min(max(int(view_index), 0), self.training_view_count - 1)
        coefficients = self.background_coefficients[index]
        if detach:
            coefficients = coefficients.detach()
        if self.background_mode == "constant":
            return F.softplus(coefficients[0]).expand(height, width)
        y = torch.linspace(-1.0, 1.0, height, device=self.means.device, dtype=self.means.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=self.means.device, dtype=self.means.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return F.softplus(coefficients[0] + coefficients[1] * xx + coefficients[2] * yy)

    def background_loss(self) -> torch.Tensor:
        if self.training_view_count == 0:
            return self.means.new_zeros(())
        base = F.softplus(self.background_coefficients[:, 0]).mean()
        if self.background_coefficients.shape[1] == 1:
            return base
        return base + 0.1 * self.background_coefficients[:, 1:].square().mean()

    def _graph_geometry(
        self,
        spatial_sigma: float,
        normal_sigma: float,
        use_normal_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph_edges.numel() == 0:
            empty = self.means.new_empty(0)
            return empty, empty, empty, empty
        source, destination = self.graph_edges
        delta = self.means[destination] - self.means[source]
        distance_squared = delta.square().sum(-1)
        normal_dot = (
            self.normals()[source] * self.normals()[destination]
        ).sum(-1).abs().clamp(0.0, 1.0)
        weight = torch.exp(-distance_squared / (2.0 * float(spatial_sigma) ** 2 + 1.0e-8))
        if use_normal_gate:
            weight = weight * torch.exp(
                -(1.0 - normal_dot) / max(float(normal_sigma), 1.0e-6)
            )
        return source, destination, delta, weight.detach()

    @torch.no_grad()
    def refresh_normals_pca(
        self,
        confidence: torch.Tensor,
        minimum_confidence: float = 1.0e-4,
        hard_reset: bool = False,
        maximum_neighbor_distance: float = 0.0,
        maximum_surface_variation: float = 1.0,
        minimum_planar_spread_ratio: float = 0.0,
    ) -> dict[str, float | int]:
        """Estimate local surface normals from the current spatial kNN graph."""
        if confidence.shape != (self.count,):
            raise ValueError("confidence must have shape [G]")
        if self.graph_edges.numel() == 0:
            return {"updated": 0, "mean_abs_cosine_before": 1.0}
        if maximum_neighbor_distance < 0:
            raise ValueError("maximum_neighbor_distance cannot be negative")
        if not 0.0 <= maximum_surface_variation <= 1.0:
            raise ValueError("maximum_surface_variation must be in [0, 1]")
        if not 0.0 <= minimum_planar_spread_ratio <= 1.0:
            raise ValueError("minimum_planar_spread_ratio must be in [0, 1]")
        source, destination = self.graph_edges.to(self.means.device)
        counts = torch.bincount(source, minlength=self.count)
        if not torch.all(counts == counts[0]):
            raise RuntimeError("PCA refresh requires a regular kNN graph")
        neighbors = int(counts[0].item())
        destination = destination.view(self.count, neighbors)
        delta = self.means.detach()[destination] - self.means.detach()[:, None, :]
        weights = confidence.detach()[destination].clamp_min(0.0)
        neighbor_confidence = weights.sum(dim=1)
        weights = weights / neighbor_confidence[:, None].clamp_min(1.0e-8)
        centroid = (weights[..., None] * delta).sum(dim=1, keepdim=True)
        centered = delta - centroid
        covariance = torch.einsum(
            "gk,gki,gkj->gij", weights, centered, centered
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        pca_normals = normalize(eigenvectors[:, :, 0])
        current = self.normals().detach()
        sign = torch.sign((pca_normals * current).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        pca_normals = pca_normals * sign
        eigenvalue_sum = eigenvalues.sum(dim=-1).clamp_min(1.0e-12)
        surface_variation = eigenvalues[:, 0].clamp_min(0.0) / eigenvalue_sum
        planar_spread_ratio = eigenvalues[:, 1].clamp_min(0.0) / eigenvalues[
            :, 2
        ].clamp_min(1.0e-12)
        maximum_distance = delta.norm(dim=-1).amax(dim=1)
        valid = confidence.detach() >= float(minimum_confidence)
        valid = valid & (neighbor_confidence > 1.0e-8)
        if maximum_neighbor_distance > 0:
            valid = valid & (
                maximum_distance <= float(maximum_neighbor_distance)
            )
        valid = valid & (
            surface_variation <= float(maximum_surface_variation)
        )
        valid = valid & (
            planar_spread_ratio >= float(minimum_planar_spread_ratio)
        )
        before = (pca_normals[valid] * current[valid]).sum(dim=-1).abs()
        refreshed = self.reference_normals.detach().clone()
        refreshed[valid] = pca_normals[valid]
        self.reference_normals = refreshed
        if hard_reset:
            self.normal_parameters.data[valid] = pca_normals[valid]
        return {
            "updated": int(valid.sum().item()),
            "rejected": int((~valid).sum().item()),
            "mean_abs_cosine_before": float(before.mean().item()) if len(before) else 1.0,
            "mean_surface_variation": (
                float(surface_variation[valid].mean().item()) if valid.any() else 0.0
            ),
            "mean_planar_spread_ratio": (
                float(planar_spread_ratio[valid].mean().item()) if valid.any() else 0.0
            ),
        }

    def support_losses(
        self,
        support: torch.Tensor,
        spatial_sigma: float,
        normal_sigma: float,
        observation_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        source, destination, _, weight = self._graph_geometry(spatial_sigma, normal_sigma)
        if len(weight):
            graph = (weight * (support[source] - support[destination]).square()).sum()
            graph = graph / weight.sum().clamp_min(1.0e-8)
        else:
            graph = support.new_zeros(())
        if observation_weight is None:
            sparsity = support.mean()
            entropy = support_entropy(support)
        else:
            if observation_weight.shape != support.shape:
                raise ValueError("observation_weight must match support shape")
            weight = observation_weight.detach().clamp_min(0.0)
            denominator = weight.sum().clamp_min(1.0e-8)
            sparsity = (weight * support).sum() / denominator
            per_gaussian_entropy = -(
                support.clamp(1.0e-6, 1.0 - 1.0e-6) * support.clamp(1.0e-6, 1.0 - 1.0e-6).log()
                + (1.0 - support.clamp(1.0e-6, 1.0 - 1.0e-6))
                * (1.0 - support.clamp(1.0e-6, 1.0 - 1.0e-6)).log()
            )
            entropy = (weight * per_gaussian_entropy).sum() / denominator
        return {
            "support_sparsity": sparsity,
            "support_entropy": entropy,
            "support_graph": graph,
        }

    def surface_loss(
        self,
        maximum_normal_ratio: float = 0.2,
        spatial_sigma: float = 0.25,
        normal_sigma: float = 0.25,
        support: torch.Tensor | None = None,
        gap_scale_ratio: float = 1.5,
        gap_max_distance: float = 0.25,
        gap_min_normal_cosine: float = 0.8,
        coverage_min_ratio: float = 0.75,
        coverage_max_ratio: float = 1.25,
        use_normal_gate: bool = True,
        tangent_repulsion_min_ratio: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        if gap_scale_ratio <= 0:
            raise ValueError("gap_scale_ratio must be positive")
        if gap_max_distance <= 0:
            raise ValueError("gap_max_distance must be positive")
        if not 0.0 <= gap_min_normal_cosine <= 1.0:
            raise ValueError("gap_min_normal_cosine must be in [0, 1]")
        if coverage_min_ratio < 0 or coverage_max_ratio <= coverage_min_ratio:
            raise ValueError("coverage ratios must satisfy 0 <= min < max")
        scales = self.scales()
        tangent_mean = scales[:, :2].mean(dim=-1)
        thinness = F.relu(scales[:, 2] - maximum_normal_ratio * tangent_mean).mean()
        reference_alignment = 1.0 - (
            self.normals() * self.reference_normals
        ).sum(dim=-1).abs()
        bbox_penalty = F.relu(self.means.abs() - self.bbox).mean()
        source, destination, delta, weight = self._graph_geometry(
            spatial_sigma, normal_sigma, use_normal_gate=use_normal_gate
        )
        if len(weight):
            direction = normalize(delta)
            local_plane = (
                weight * (self.normals()[source] * direction).sum(-1).square()
            ).sum() / weight.sum().clamp_min(1.0e-8)
            normal_consistency = (
                weight
                * (1.0 - (self.normals()[source] * self.normals()[destination]).sum(-1).abs())
            ).sum() / weight.sum().clamp_min(1.0e-8)
            destination_plane = (
                self.normals()[destination] * direction
            ).sum(-1).square()
            bidirectional_plane = (
                weight
                * 0.5
                * (
                    (self.normals()[source] * direction).sum(-1).square()
                    + destination_plane
                )
            ).sum() / weight.sum().clamp_min(1.0e-8)
            scale_difference = (
                self.log_scales[source, :2] - self.log_scales[destination, :2]
            ).square().mean(dim=-1)
            scale_smoothness = (weight * scale_difference).sum()
            scale_smoothness = scale_smoothness / weight.sum().clamp_min(1.0e-8)

            covariance = self.covariance()
            covariance_trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
            covariance_shape = covariance / covariance_trace[:, None, None].clamp_min(
                1.0e-8
            )
            covariance_difference = (
                covariance_shape[source] - covariance_shape[destination]
            ).square().sum(dim=(-1, -2))
            covariance_normal_cosine = (
                self.normals()[source] * self.normals()[destination]
            ).sum(-1).abs()
            covariance_weight = weight * covariance_normal_cosine.detach().pow(4)
            if support is not None:
                covariance_weight = covariance_weight * (
                    support[source] * support[destination]
                ).detach()
            covariance_consistency = (
                covariance_weight * covariance_difference
            ).sum() / covariance_weight.sum().clamp_min(1.0e-8)

            # Penalize only small gaps between nearby, similarly oriented
            # surface elements. The hard distance gate deliberately prevents
            # this term from bridging distinct spacecraft structures.
            source_normals = self.normals()[source]
            destination_normals = self.normals()[destination]
            normal_cosine = (source_normals * destination_normals).sum(-1).abs()
            normal_offset = (delta * source_normals).sum(-1, keepdim=True)
            tangent_distance = (delta - normal_offset * source_normals).norm(dim=-1)
            allowed_distance = float(gap_scale_ratio) * (
                tangent_mean[source] + tangent_mean[destination]
            ).clamp_min(1.0e-6)
            # Use scales to define the desired coverage but do not let the
            # optimizer satisfy continuity merely by inflating every splat.
            allowed_distance = allowed_distance.detach()
            local_pair = normal_cosine >= float(gap_min_normal_cosine)
            local_pair = local_pair & (
                delta.norm(dim=-1) <= float(gap_max_distance)
            )
            gap_weight = weight * local_pair.to(weight.dtype)
            if support is not None:
                if support.shape != (self.count,):
                    raise ValueError("support must have shape [G]")
                gap_weight = gap_weight * (
                    support[source] * support[destination]
                ).detach()
            relative_excess = F.relu(tangent_distance / allowed_distance - 1.0)
            surface_gap = (gap_weight * relative_excess.square()).sum()
            surface_gap = surface_gap / gap_weight.sum().clamp_min(1.0e-8)
            coverage_weight = gap_weight
            scale_sum = (tangent_mean[source] + tangent_mean[destination]).clamp_min(
                1.0e-6
            )
            spacing_ratio = tangent_distance / scale_sum
            coverage_error = F.relu(float(coverage_min_ratio) - spacing_ratio).square()
            coverage_error = coverage_error + F.relu(
                spacing_ratio - float(coverage_max_ratio)
            ).square()
            coverage_band = (coverage_weight * coverage_error).sum()
            coverage_band = coverage_band / coverage_weight.sum().clamp_min(1.0e-8)
            minimum_spacing = (
                float(tangent_repulsion_min_ratio) * 0.5
                * (tangent_mean[source] + tangent_mean[destination])
            ).detach().clamp_min(1.0e-6)
            tangent_repulsion = (
                gap_weight * F.relu(1.0 - tangent_distance / minimum_spacing).square()
            ).sum() / gap_weight.sum().clamp_min(1.0e-8)
        else:
            local_plane = thinness.new_zeros(())
            normal_consistency = thinness.new_zeros(())
            surface_gap = thinness.new_zeros(())
            bidirectional_plane = thinness.new_zeros(())
            scale_smoothness = thinness.new_zeros(())
            covariance_consistency = thinness.new_zeros(())
            coverage_band = thinness.new_zeros(())
            tangent_repulsion = thinness.new_zeros(())
        tangent_isotropy = (
            self.log_scales[:, 0] - self.log_scales[:, 1]
        ).square().mean()
        return {
            "surface_thinness": thinness,
            "reference_alignment": reference_alignment.mean(),
            "local_plane": local_plane,
            "normal_consistency": normal_consistency,
            "surface_gap": surface_gap,
            "bidirectional_plane": bidirectional_plane,
            "scale_smoothness": scale_smoothness,
            "tangent_isotropy": tangent_isotropy,
            "covariance_consistency": covariance_consistency,
            "coverage_band": coverage_band,
            "tangent_repulsion": tangent_repulsion,
            "bbox": bbox_penalty,
        }

    def feature_loss(self) -> torch.Tensor:
        return self.features.square().mean()

    def parameter_groups(self, lr: float) -> list[dict]:
        groups = [
            {"params": [self.means], "lr": lr * 0.25},
            {"params": [self.log_scales, self.normal_parameters], "lr": lr * 0.1},
            {"params": [self.support_logits, self.features], "lr": lr},
            {"params": [self.background_coefficients], "lr": lr * 0.25},
            {"params": self.scattering.parameters(), "lr": lr},
        ]
        if self.surface_field is not None:
            groups.append({"params": self.surface_field.parameters(), "lr": lr * 0.25})
        return groups

    def adaptive_control(
        self,
        evidence: torch.Tensor,
        mean_gradient: torch.Tensor,
        visible_fraction: torch.Tensor,
        support_temperature: float,
        prune_support_threshold: float,
        prune_evidence_threshold: float,
        prune_min_visible_fraction: float,
        split_support_threshold: float,
        split_gradient_threshold: float,
        maximum_gaussians: int,
        minimum_gaussians: int,
        split_displacement_ratio: float = 0.3,
        split_scale_shrink: float = 0.8,
        split_mode: str = "symmetric",
        split_evidence_threshold: float = 0.0,
        split_eligible: torch.Tensor | None = None,
        split_direction_mode: str = "tangent_axes",
        split_direction_vector: torch.Tensor | None = None,
        relocation_noise_ratio: float = 0.5,
        relocation_maximum_fraction: float = 0.05,
        adaptive_knn_tangent_scale_ratio: float = 0.0,
        prune_eligible: torch.Tensor | None = None,
    ) -> dict:
        """Safely prune and optionally split from multi-view evidence."""
        if split_displacement_ratio <= 0:
            raise ValueError("split_displacement_ratio must be positive")
        if not 0.0 < split_scale_shrink <= 1.0:
            raise ValueError("split_scale_shrink must be in (0, 1]")
        if split_mode not in {
            "none",
            "asymmetric",
            "asymmetric_conservative",
            "symmetric",
            "symmetric_conservative",
            "relocate",
        }:
            raise ValueError(
                "split_mode must be none, asymmetric, asymmetric_conservative, "
                "symmetric, symmetric_conservative, or relocate"
            )
        if relocation_noise_ratio <= 0.0:
            raise ValueError("relocation_noise_ratio must be positive")
        if not 0.0 < relocation_maximum_fraction <= 1.0:
            raise ValueError("relocation_maximum_fraction must be in (0, 1]")
        if split_evidence_threshold < 0:
            raise ValueError("split_evidence_threshold cannot be negative")
        if split_direction_mode not in {"tangent_axes", "local_pca", "center_gradient"}:
            raise ValueError(
                "split_direction_mode must be tangent_axes, local_pca, or center_gradient"
            )
        if split_direction_vector is not None and split_direction_vector.shape != self.means.shape:
            raise ValueError("split_direction_vector must have shape [G,3]")
        if adaptive_knn_tangent_scale_ratio < 0.0:
            raise ValueError("adaptive_knn_tangent_scale_ratio cannot be negative")
        if split_eligible is not None and split_eligible.shape != (self.count,):
            raise ValueError("split_eligible must have shape [G]")
        if prune_eligible is not None and prune_eligible.shape != (self.count,):
            raise ValueError("prune_eligible must have shape [G]")
        with torch.no_grad():
            support = self.support_probability(support_temperature)
            prune = (
                (support < prune_support_threshold)
                & (evidence < prune_evidence_threshold)
                & (visible_fraction >= prune_min_visible_fraction)
            )
            if prune_eligible is not None:
                prune &= prune_eligible
            relocate = split_mode == "relocate"
            relocation_candidates = torch.where(prune)[0]
            if relocate:
                maximum_relocations = max(
                    1, int(round(self.count * float(relocation_maximum_fraction)))
                )
                if len(relocation_candidates) > maximum_relocations:
                    badness = support[relocation_candidates] + evidence[relocation_candidates]
                    relocation_candidates = relocation_candidates[
                        torch.topk(
                            badness,
                            k=maximum_relocations,
                            largest=False,
                        ).indices
                    ]
                prune[:] = False
            if self.count - int(prune.sum()) < minimum_gaussians:
                candidates = torch.where(prune)[0]
                allowed = max(0, self.count - minimum_gaussians)
                prune[:] = False
                if allowed and len(candidates):
                    score = support[candidates] + evidence[candidates]
                    chosen = candidates[torch.topk(score, k=min(allowed, len(score)), largest=False).indices]
                    prune[chosen] = True
            keep = ~prune
            kept_indices = torch.where(keep)[0]
            available = max(0, int(maximum_gaussians) - len(kept_indices))
            split_score = mean_gradient * evidence.clamp_min(1.0e-8)
            split_candidates = torch.where(
                keep
                & (support >= split_support_threshold)
                & (mean_gradient >= split_gradient_threshold)
                & (evidence >= split_evidence_threshold)
                & (
                    split_eligible.to(device=keep.device, dtype=torch.bool)
                    if split_eligible is not None
                    else torch.ones_like(keep)
                )
            )[0]
            if split_mode in {"none", "relocate"}:
                split_candidates = split_candidates.new_empty(0)
            if available and len(split_candidates):
                order = torch.topk(
                    split_score[split_candidates],
                    k=min(available, len(split_candidates)),
                ).indices
                split_indices = split_candidates[order]
            else:
                split_indices = kept_indices.new_empty(0)
            source_indices = torch.cat((kept_indices, split_indices))

            means = self.means.detach()[source_indices].clone()
            normals = self.normal_parameters.detach()[source_indices].clone()
            log_scales = self.log_scales.detach()[source_indices].clone()
            support_logits = self.support_logits.detach()[source_indices].clone()
            features = self.features.detach()[source_indices].clone()
            reference_normals = self.reference_normals.detach()[source_indices].clone()
            split_axis_t1 = 0
            split_axis_t2 = 0
            split_reset_positions = kept_indices.new_empty(0)
            relocated = 0
            if relocate and len(relocation_candidates):
                donor_mask = (
                    (support >= split_support_threshold)
                    & (evidence >= split_evidence_threshold)
                )
                donor_indices = torch.where(donor_mask)[0]
                if len(donor_indices):
                    donor_score = support[donor_indices] * evidence[donor_indices].clamp_min(1.0e-8)
                    donor_order = torch.topk(
                        donor_score,
                        k=len(donor_indices),
                        largest=True,
                    ).indices
                    donors = donor_indices[donor_order]
                    donors = donors[
                        torch.arange(len(relocation_candidates), device=means.device)
                        .remainder(len(donors))
                    ]
                    basis = build_surface_basis(self.normals().detach()[donors])
                    donor_radius = self.scales().detach()[donors, :2].mean(dim=-1)
                    tangent_noise = torch.randn(
                        len(donors), 2, device=means.device, dtype=means.dtype
                    )
                    tangent_noise = tangent_noise * donor_radius[:, None]
                    tangent_noise = tangent_noise * float(relocation_noise_ratio)
                    offsets = torch.einsum("gdi,gi->gd", basis[:, :, :2], tangent_noise)
                    means[relocation_candidates] = self.means.detach()[donors] + offsets
                    normals[relocation_candidates] = self.normal_parameters.detach()[donors]
                    log_scales[relocation_candidates] = self.log_scales.detach()[donors]
                    support_logits[relocation_candidates] = self.support_logits.detach()[donors]
                    features[relocation_candidates] = self.features.detach()[donors]
                    features[relocation_candidates] += 0.005 * torch.randn_like(
                        features[relocation_candidates]
                    )
                    reference_normals[relocation_candidates] = self.reference_normals.detach()[donors]
                    source_indices[relocation_candidates] = donors
                    split_reset_positions = relocation_candidates
                    relocated = len(relocation_candidates)
            if len(split_indices):
                split_start = len(kept_indices)
                basis = build_surface_basis(normalize(normals[split_start:]))
                split_tangent_scales = log_scales[split_start:, :2].exp()
                rows = torch.arange(len(split_indices), device=means.device)
                tangent_axis = split_tangent_scales.argmax(dim=-1)
                tied_axes = torch.isclose(
                    split_tangent_scales[:, 0],
                    split_tangent_scales[:, 1],
                    rtol=1.0e-5,
                    atol=1.0e-8,
                )
                tangent_axis = torch.where(
                    tied_axes,
                    rows.remainder(2),
                    tangent_axis,
                )
                tangent = basis[rows, :, tangent_axis]
                selected_scale = split_tangent_scales[rows, tangent_axis]
                split_local_pca = 0
                if split_direction_mode == "local_pca":
                    source_edges, destination_edges = self.graph_edges
                    counts = torch.bincount(source_edges, minlength=self.count)
                    if not torch.all(counts == counts[0]):
                        raise RuntimeError(
                            "local-PCA split requires a regular kNN graph"
                        )
                    neighbor_count = int(counts[0].item())
                    neighbors = destination_edges.view(self.count, neighbor_count)
                    local_delta = (
                        self.means.detach()[neighbors[split_indices]]
                        - self.means.detach()[split_indices, None]
                    )
                    split_normals = normalize(normals[split_start:])
                    normal_offset = (
                        local_delta * split_normals[:, None, :]
                    ).sum(dim=-1, keepdim=True)
                    projected = local_delta - normal_offset * split_normals[:, None, :]
                    covariance = torch.einsum(
                        "gki,gkj->gij", projected, projected
                    ) / max(neighbor_count, 1)
                    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                    pca_tangent = normalize(eigenvectors[:, :, -1])
                    reliable = eigenvalues[:, -1] > 1.0e-10
                    # A line direction has no inherent sign.  Canonicalize it
                    # by making its largest global component positive so that
                    # asymmetric splitting does not alternate by row order.
                    dominant = pca_tangent.abs().argmax(dim=-1, keepdim=True)
                    dominant_value = pca_tangent.gather(1, dominant)
                    sign = torch.where(
                        dominant_value < 0,
                        -torch.ones_like(dominant_value),
                        torch.ones_like(dominant_value),
                    )
                    pca_tangent = pca_tangent * sign
                    tangent = torch.where(reliable[:, None], pca_tangent, tangent)
                    tangent_coordinates = torch.einsum(
                        "gdi,gd->gi", basis, tangent
                    )[:, :2]
                    directional_variance = (
                        tangent_coordinates.square()
                        * split_tangent_scales.square()
                    ).sum(dim=-1)
                    pca_scale = directional_variance.clamp_min(1.0e-12).sqrt()
                    selected_scale = torch.where(reliable, pca_scale, selected_scale)
                    tangent_axis = tangent_coordinates.abs().argmax(dim=-1)
                    split_local_pca = int(reliable.sum().item())
                elif split_direction_mode == "center_gradient":
                    if split_direction_vector is None:
                        raise ValueError(
                            "center_gradient split requires split_direction_vector"
                        )
                    center_direction = split_direction_vector.detach()[split_indices]
                    split_normals = normalize(normals[split_start:])
                    center_direction = center_direction - (
                        center_direction * split_normals
                    ).sum(dim=-1, keepdim=True) * split_normals
                    direction_norm = center_direction.norm(dim=-1)
                    reliable = direction_norm > 1.0e-10
                    center_direction = normalize(center_direction)
                    tangent = torch.where(reliable[:, None], center_direction, tangent)
                    selected_scale = split_tangent_scales.mean(dim=-1)
                    tangent_coordinates = torch.einsum("gdi,gd->gi", basis, tangent)[:, :2]
                    tangent_axis = tangent_coordinates.abs().argmax(dim=-1)
                displacement = (
                    tangent
                    * selected_scale[:, None]
                    * float(split_displacement_ratio)
                )

                retained_position = torch.full(
                    (self.count,), -1, dtype=torch.long, device=means.device
                )
                retained_position[kept_indices] = torch.arange(
                    len(kept_indices), device=means.device
                )
                parent_rows = retained_position[split_indices]
                if split_mode in {"symmetric", "symmetric_conservative"}:
                    # Preserve the pre-split centroid with matched +/- offsets.
                    means[parent_rows] -= displacement
                    means[split_start:] += displacement
                else:
                    # Match the symmetric arm's parent-child separation while
                    # allowing the centroid to drift in one direction.
                    means[split_start:] += 2.0 * displacement

                shrink = math.log(float(split_scale_shrink))
                log_scales[parent_rows, :2] += shrink
                log_scales[split_start:, :2] += shrink
                if split_mode in {
                    "asymmetric_conservative",
                    "symmetric_conservative",
                }:
                    # Two children replace one occupancy mass.  For coincident
                    # Bernoulli supports, q_child = 1-sqrt(1-q_parent) keeps
                    # 1-(1-q_child)^2 equal to the parent's q.
                    temperature = max(float(support_temperature), 1.0e-3)
                    parent_q = torch.sigmoid(
                        support_logits[parent_rows] / temperature
                    )
                    child_q = 1.0 - torch.sqrt((1.0 - parent_q).clamp_min(1.0e-6))
                    child_logits = temperature * torch.logit(
                        child_q.clamp(1.0e-5, 1.0 - 1.0e-5)
                    )
                    support_logits[parent_rows] = child_logits
                    support_logits[split_start:] = child_logits
                features[split_start:] += 0.005 * torch.randn_like(features[split_start:])
                split_axis_t1 = int((tangent_axis == 0).sum().item())
                split_axis_t2 = int((tangent_axis == 1).sum().item())
                split_reset_positions = torch.cat(
                    (
                        parent_rows,
                        torch.arange(split_start, len(source_indices), device=means.device),
                    )
                )

            self.means = nn.Parameter(means)
            self.normal_parameters = nn.Parameter(normals)
            self.log_scales = nn.Parameter(log_scales)
            self.support_logits = nn.Parameter(support_logits)
            self.features = nn.Parameter(features)
            self.reference_normals = reference_normals
            self.graph_edges = build_knn_graph(means, self.graph_neighbors).to(means.device)
            if adaptive_knn_tangent_scale_ratio > 0.0 and self.count > 1:
                graph_source, graph_destination = self.graph_edges
                neighbor_distance = (
                    means[graph_destination] - means[graph_source]
                ).norm(dim=-1)
                counts = torch.bincount(graph_source, minlength=self.count)
                if torch.all(counts == counts[0]):
                    neighbor_distance = neighbor_distance.view(self.count, int(counts[0]))
                    local_spacing = neighbor_distance.median(dim=1).values
                    tangent_scale = (
                        float(adaptive_knn_tangent_scale_ratio) * local_spacing
                    ).clamp(1.0e-4, self.bbox)
                    self.log_scales.data[:, :2] = tangent_scale[:, None].log()
            return {
                "pruned": int(prune.sum()),
                "split": len(split_indices),
                "relocated": relocated,
                "split_axis_t1": split_axis_t1,
                "split_axis_t2": split_axis_t2,
                "split_local_pca": split_local_pca if len(split_indices) else 0,
                "split_mode": split_mode,
                "symmetric_split": split_mode == "symmetric",
                "count": self.count,
                "source_indices": source_indices,
                "split_reset_positions": split_reset_positions,
            }
