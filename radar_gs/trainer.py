from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from .data import ISARView
from .geometry import projection_coordinates
from .io import export_ply, save_checkpoint, save_preview, write_json
from .losses import compress_amplitude, reconstruction_loss
from .model import RadarGaussianScene
from .renderer import RadarVisibilityRenderer


def training_schedule(iteration: int, total: int, arguments: dict) -> dict:
    # Include both endpoints: the first update is always stage A and the final
    # update reaches the fully annealed stage D settings.
    fraction = (iteration - 1) / max(total - 1, 1)
    end_a = float(arguments["stage_a_fraction"])
    end_b = float(arguments["stage_b_fraction"])
    end_c = float(arguments["stage_c_fraction"])
    if not 0.0 < end_a < end_b < end_c < 1.0:
        raise ValueError("Require 0 < stage_a_fraction < stage_b_fraction < stage_c_fraction < 1")

    if fraction <= end_a:
        stage = "A"
        local = fraction / end_a
        occupancy_factor = 0.0
        binary_factor = 0.0
        graph_factor = 0.0
        background_trainable = False
    elif fraction <= end_b:
        stage = "B"
        local = (fraction - end_a) / (end_b - end_a)
        occupancy_factor = local
        binary_factor = 0.0
        graph_factor = local
        background_trainable = True
    elif fraction <= end_c:
        stage = "C"
        local = (fraction - end_b) / (end_c - end_b)
        occupancy_factor = 1.0
        binary_factor = 0.5 * local
        graph_factor = 1.0
        background_trainable = True
    else:
        stage = "D"
        local = (fraction - end_c) / (1.0 - end_c)
        occupancy_factor = 1.0
        binary_factor = 0.5 + 0.5 * local
        graph_factor = 1.0
        background_trainable = True

    post_a = max(0.0, min(1.0, (fraction - end_a) / (1.0 - end_a)))
    temperature = 1.0 + post_a * (float(arguments["final_support_temperature"]) - 1.0)
    visibility_gain = float(arguments["initial_visibility_gain"])
    visibility_gain += fraction * (1.0 - visibility_gain)
    surface_factor = 0.25 + 0.75 * fraction
    field_start = float(arguments.get("surface_field_start_fraction", 0.35))
    field_full = float(arguments.get("surface_field_full_fraction", 0.55))
    if not 0.0 <= field_start < field_full <= 1.0:
        raise ValueError("Require 0 <= surface_field_start_fraction < surface_field_full_fraction <= 1")
    field_factor = max(
        0.0, min(1.0, (fraction - field_start) / max(field_full - field_start, 1.0e-8))
    )
    return {
        "stage": stage,
        "stage_progress": local,
        "occupancy_factor": occupancy_factor,
        "binary_factor": binary_factor,
        "graph_factor": graph_factor,
        "surface_factor": surface_factor,
        "background_trainable": background_trainable,
        "temperature": temperature,
        "visibility_gain": visibility_gain,
        "surface_field_factor": field_factor,
    }


def _make_optimizer(scene: RadarGaussianScene, base_lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(scene.parameter_groups(base_lr))


def geometry_gradient_activity(fraction: float, arguments: dict) -> dict[str, bool]:
    """Return which geometry parameter families may receive optimizer updates."""
    fraction = float(fraction)
    activity = {}
    for name in ("means", "normals", "scales"):
        start = float(arguments.get(f"{name}_start_fraction", 0.0))
        if not 0.0 <= start <= 1.0:
            raise ValueError(f"{name}_start_fraction must be in [0, 1]")
        activity[name] = (
            not bool(arguments.get(f"freeze_{name}", False)) and fraction >= start
        )
    return activity


def _apply_geometry_gradient_schedule(
    scene: RadarGaussianScene, fraction: float, arguments: dict
) -> dict[str, bool]:
    activity = geometry_gradient_activity(fraction, arguments)
    if not activity["means"]:
        scene.means.grad = None
    if not activity["normals"]:
        scene.normal_parameters.grad = None
    if not activity["scales"]:
        scene.log_scales.grad = None
    return activity


def train(
    scene: RadarGaussianScene,
    renderer: RadarVisibilityRenderer,
    views: list[ISARView],
    output_dir: str | Path,
    arguments: dict,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(scene.parameters()).device
    current_base_lr = float(arguments["lr"])
    optimizer = _make_optimizer(scene, current_base_lr)
    randomizer = random.Random(arguments["seed"])
    history: list[dict] = []
    start_time = time.perf_counter()
    views_per_batch = max(1, int(arguments["views_per_batch"]))
    replay_fraction = float(arguments.get("continual_replay_fraction", 0.0))
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("continual_replay_fraction must be in [0, 1)")
    current_encounter = arguments.get("continual_current_encounter")
    current_views = [view for view in views if view.encounter == current_encounter]
    replay_views = [view for view in views if view.encounter != current_encounter]
    if current_encounter and not current_views:
        raise ValueError(f"continual_current_encounter is unavailable: {current_encounter}")
    evidence_ema = torch.zeros(scene.count, device=device)
    gradient_ema = torch.zeros(scene.count, device=device)
    direction_gradient_ema = torch.zeros_like(scene.means.detach())
    visible_ema = torch.zeros(scene.count, device=device)
    observation_ema = torch.zeros(scene.count, device=device)
    protection_threshold = float(
        arguments.get("protect_warm_start_support_threshold", 0.0)
    )
    if not 0.0 <= protection_threshold <= 1.0:
        raise ValueError("protect_warm_start_support_threshold must be in [0, 1]")
    protected_inherited = (
        scene.support_probability().detach() >= protection_threshold
        if arguments.get("warm_start_checkpoint") and protection_threshold > 0.0
        else torch.zeros(scene.count, dtype=torch.bool, device=device)
    )
    encounter_names = sorted({view.encounter for view in views})
    encounter_to_index = {name: index for index, name in enumerate(encounter_names)}
    encounter_observation_ema = torch.zeros(
        (len(encounter_names), scene.count), device=device
    )
    encounter_evidence_ema = torch.zeros_like(encounter_observation_ema)
    split_cooldown = max(0, int(arguments.get("split_cooldown_iterations", 0)))
    split_age = torch.full(
        (scene.count,), split_cooldown, dtype=torch.long, device=device
    )
    ema_decay = float(arguments["evidence_ema_decay"])
    direction_ema_decay = float(arguments.get("split_direction_ema_decay", 0.0))
    if not 0.0 <= direction_ema_decay < 1.0:
        raise ValueError("split_direction_ema_decay must be in [0, 1)")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scene.train()
    previous_stage = None
    pca_reference_ready = False
    for iteration in range(1, arguments["iters"] + 1):
        split_age.add_(1)
        schedule = training_schedule(iteration, arguments["iters"], arguments)
        if schedule["stage"] == "A":
            support = torch.full_like(
                scene.support_logits,
                float(arguments["initial_support_probability"]),
            )
        else:
            support = scene.support_probability(schedule["temperature"])

        pca_refresh = None
        pca_mode = arguments.get("pca_normal_refresh_mode", "none")
        if schedule["stage"] == "C" and previous_stage != "C" and pca_mode != "none":
            pca_refresh = scene.refresh_normals_pca(
                evidence_ema,
                arguments.get("pca_min_confidence", 1.0e-4),
                hard_reset=pca_mode in {"stage_c_hard", "adaptive_hard"},
                maximum_neighbor_distance=arguments.get(
                    "pca_max_neighbor_distance", 0.0
                ),
                maximum_surface_variation=arguments.get(
                    "pca_max_surface_variation", 1.0
                ),
                minimum_planar_spread_ratio=arguments.get(
                    "pca_min_planar_spread_ratio", 0.0
                ),
            )
            pca_reference_ready = True
            print(
                f"pca-normal iter={iteration:06d} updated={pca_refresh['updated']} "
                f"cos_before={pca_refresh['mean_abs_cosine_before']:.4f} "
                f"hard={pca_mode in {'stage_c_hard', 'adaptive_hard'}}"
            )

        if current_encounter and replay_views and replay_fraction > 0.0:
            replay_count = min(
                views_per_batch - 1,
                max(1, round(views_per_batch * replay_fraction)),
            )
            current_count = views_per_batch - replay_count
            def draw(pool, count):
                if count <= len(pool):
                    return randomizer.sample(pool, count)
                return [pool[randomizer.randrange(len(pool))] for _ in range(count)]
            batch = draw(current_views, current_count) + draw(replay_views, replay_count)
            randomizer.shuffle(batch)
        elif views_per_batch <= len(views):
            batch = randomizer.sample(views, views_per_batch)
        else:
            batch = [views[randomizer.randrange(len(views))] for _ in range(views_per_batch)]

        image_losses = []
        robust_values = []
        log_values = []
        ssim_values = []
        contribution_values = []
        visibility_values = []
        loss_gate_values = []
        observation_gate_values = []
        observed_visible_values = []
        amplitude_values = []
        preview_items = None
        for view in batch:
            target = view.image.to(device, non_blocking=True)
            output = renderer(
                scene,
                view.kr,
                view.kd,
                view.dr,
                view.dd,
                support_probability=support,
                visibility_gain=schedule["visibility_gain"],
            )
            signal = output.raw_image if arguments["disable_visibility"] else output.image
            background = scene.background(
                view.training_index,
                target.shape[0],
                target.shape[1],
                detach=not schedule["background_trainable"],
            )
            raw_prediction = compress_amplitude(
                output.raw_image + background,
                arguments["render_gamma"],
            )
            prediction = compress_amplitude(signal + background, arguments["render_gamma"])
            per_view_loss, image_metrics = reconstruction_loss(
                prediction,
                target,
                arguments["lambda_ssim"],
                arguments["lambda_log_amplitude"],
                arguments["charbonnier_epsilon"],
                arguments["log_amplitude_gain"],
                view.loss_weight.to(device, non_blocking=True)
                if view.loss_weight is not None else None,
            )
            image_losses.append(per_view_loss)
            robust_values.append(image_metrics["charbonnier"])
            log_values.append(image_metrics["log_charbonnier"])
            ssim_values.append(image_metrics["ssim"])
            row_float, col_float, _ = projection_coordinates(
                scene.means.detach(), view.kr.to(device), view.kd.to(device),
                target.shape[0], target.shape[1], view.dr, view.dd,
            )
            margin = float(arguments.get("observation_fov_margin_pixels", 0.0))
            in_frame = (
                (row_float >= margin)
                & (row_float <= target.shape[0] - 1 - margin)
                & (col_float >= margin)
                & (col_float <= target.shape[1] - 1 - margin)
            ).float()
            gaussian_gate = torch.ones_like(in_frame)
            if view.loss_weight is not None:
                loss_weight = view.loss_weight.to(device, non_blocking=True)
                row = row_float.round().long().clamp(0, target.shape[0] - 1)
                col = col_float.round().long().clamp(0, target.shape[1] - 1)
                gaussian_gate = loss_weight[row, col].detach()
            visible = (
                output.gaussian_visibility.detach() >= arguments["visible_threshold"]
            ).float()
            contribution_values.append(output.gaussian_contribution.detach())
            amplitude_values.append(output.gaussian_amplitude)
            visibility_values.append(visible)
            loss_gate_values.append(gaussian_gate)
            observation_gate = in_frame * gaussian_gate
            observation_gate_values.append(observation_gate)
            observed_visible_values.append(visible * observation_gate)
            preview_items = (
                view,
                target,
                raw_prediction,
                prediction,
                output.support_map,
                output.overlap_map,
            )

        image_loss = torch.stack(image_losses).mean()
        support_weight = None
        if arguments.get("observation_weighted_support", False):
            support_weight = observation_ema.clamp_min(
                float(arguments.get("observation_support_weight_floor", 0.0))
            )
        support_terms = scene.support_losses(
            support,
            arguments["graph_spatial_sigma"],
            arguments["graph_normal_sigma"],
            observation_weight=support_weight,
        )
        surface_terms = scene.surface_loss(
            arguments["maximum_normal_ratio"],
            arguments["graph_spatial_sigma"],
            arguments["graph_normal_sigma"],
            support=support,
            gap_scale_ratio=arguments["surface_gap_scale_ratio"],
            gap_max_distance=arguments["surface_gap_max_distance"],
            gap_min_normal_cosine=arguments["surface_gap_min_normal_cosine"],
            coverage_min_ratio=arguments.get("coverage_min_ratio", 0.75),
            coverage_max_ratio=arguments.get("coverage_max_ratio", 1.25),
            use_normal_gate=arguments.get("surface_graph_mode", "spatial_normal")
            == "spatial_normal",
            tangent_repulsion_min_ratio=arguments.get(
                "tangent_repulsion_min_ratio", 0.5
            ),
        )
        surface_loss = (
            surface_terms["surface_thinness"]
            + arguments["lambda_local_plane"] * surface_terms["local_plane"]
            + arguments["lambda_normal_consistency"] * surface_terms["normal_consistency"]
            + arguments["lambda_reference_normal"]
            * (
                1.0
                if pca_mode == "none" or pca_reference_ready
                else 0.0
            )
            * surface_terms["reference_alignment"]
            + arguments["lambda_surface_gap"] * surface_terms["surface_gap"]
            + arguments.get("lambda_bidirectional_plane", 0.0)
            * surface_terms["bidirectional_plane"]
            + arguments.get("lambda_scale_smoothness", 0.0)
            * surface_terms["scale_smoothness"]
            + arguments.get("lambda_tangent_isotropy", 0.0)
            * surface_terms["tangent_isotropy"]
            + arguments.get("lambda_covariance_consistency", 0.0)
            * surface_terms["covariance_consistency"]
            + arguments.get("lambda_coverage_band", 0.0)
            * surface_terms["coverage_band"]
            + arguments.get("lambda_tangent_repulsion", 0.0)
            * surface_terms["tangent_repulsion"]
            + arguments["lambda_bbox"] * surface_terms["bbox"]
        )
        field_terms = scene.surface_field_losses(
            support,
            evidence_ema,
            arguments.get("surface_field_minimum_support", 0.3),
            arguments.get("surface_field_minimum_evidence", 1.0e-4),
            arguments.get("surface_field_sample_count", 2048),
            arguments.get("surface_field_eikonal_jitter", 0.08),
            arguments.get("surface_field_offset_distance", 0.05),
            arguments.get("surface_field_offset_loss_mode", "smooth_l1"),
            arguments.get("surface_field_detach_geometry", False),
        ) if schedule["surface_field_factor"] > 0.0 else {
            "zero": scene.means.new_zeros(()),
            "normal": scene.means.new_zeros(()),
            "eikonal": scene.means.new_zeros(()),
            "offset": scene.means.new_zeros(()),
            "geometry_normal": scene.means.new_zeros(()),
            "selected": 0,
        }
        total = image_loss
        total = total + arguments["lambda_support_sparsity"] * schedule[
            "occupancy_factor"
        ] * support_terms["support_sparsity"]
        total = total + arguments["lambda_support_binary"] * schedule[
            "binary_factor"
        ] * support_terms["support_entropy"]
        total = total + arguments["lambda_support_graph"] * schedule[
            "graph_factor"
        ] * support_terms["support_graph"]
        total = total + arguments["lambda_surface"] * schedule[
            "surface_factor"
        ] * surface_loss
        if schedule["background_trainable"]:
            total = total + arguments["lambda_background"] * scene.background_loss()
        total = total + arguments["lambda_scattering_feature"] * scene.feature_loss()
        scattering_view_consistency = scene.means.new_zeros(())
        if float(arguments.get("lambda_scattering_view_consistency", 0.0)) > 0.0:
            amplitude_stack = torch.stack(amplitude_values)
            amplitude_mean = amplitude_stack.mean(0)
            relative_variance = amplitude_stack.var(0, unbiased=False) / (
                amplitude_mean.square() + 1.0e-4
            )
            scattering_weight = support.detach() * torch.stack(
                observation_gate_values
            ).mean(0)
            scattering_eligible = (
                support.detach()
                >= float(arguments.get("scattering_consistency_minimum_support", 0.0))
            ) & (
                observation_ema
                >= float(
                    arguments.get(
                        "scattering_consistency_minimum_observation", 0.0
                    )
                )
            )
            scattering_weight = scattering_weight * scattering_eligible.float()
            scattering_view_consistency = (
                relative_variance * scattering_weight
            ).sum() / scattering_weight.sum().clamp_min(1.0)
            total = total + float(
                arguments["lambda_scattering_view_consistency"]
            ) * scattering_view_consistency
        base_total = total
        field_loss = schedule["surface_field_factor"] * (
            arguments.get("lambda_surface_field", 0.0) * field_terms["zero"]
            + arguments.get("lambda_surface_field_normal", 0.0) * field_terms["normal"]
            + arguments.get("lambda_surface_field_eikonal", 0.0) * field_terms["eikonal"]
            + arguments.get("lambda_surface_field_offset", 0.0) * field_terms["offset"]
            + arguments.get("lambda_surface_field_geometry_normal", 0.0)
            * field_terms["geometry_normal"]
        )
        total = base_total + field_loss

        optimizer.zero_grad(set_to_none=True)
        adaptive_image_gradient = None
        adaptive_gradient_source = arguments.get("adaptive_gradient_source", "total")
        if adaptive_gradient_source == "image":
            adaptive_loss = image_loss
        elif adaptive_gradient_source == "base":
            adaptive_loss = base_total
        elif adaptive_gradient_source == "blended":
            field_weight = float(
                arguments.get("adaptive_surface_field_gradient_weight", 0.5)
            )
            if not 0.0 <= field_weight <= 1.0:
                raise ValueError(
                    "adaptive_surface_field_gradient_weight must be in [0, 1]"
                )
            adaptive_loss = base_total + field_weight * field_loss
        else:
            adaptive_loss = None
        if adaptive_loss is not None:
            adaptive_image_gradient = torch.autograd.grad(
                adaptive_loss,
                scene.means,
                retain_graph=True,
                allow_unused=True,
            )[0]
        total.backward()
        adaptive_direction_gradient = (
            adaptive_image_gradient.detach()
            if adaptive_image_gradient is not None
            else scene.means.grad.detach().clone()
        )
        if adaptive_image_gradient is None:
            mean_gradient = scene.means.grad.detach().norm(dim=-1)
        else:
            mean_gradient = adaptive_image_gradient.detach().norm(dim=-1)
        tangent_gradient_count = 0
        if (
            arguments.get("surface_field_tangent_means", False)
            and schedule["stage"] in {"C", "D"}
            and scene.surface_field is not None
        ):
            tangent_eligible = (
                (support.detach() >= arguments.get("surface_field_minimum_support", 0.3))
                & (
                    evidence_ema
                    >= arguments.get("surface_field_minimum_evidence", 1.0e-4)
                )
            )
            tangent_gradient_count = scene.tangent_project_mean_gradients(tangent_eligible)
        geometry_activity = _apply_geometry_gradient_schedule(
            scene,
            (iteration - 1) / max(arguments["iters"] - 1, 1),
            arguments,
        )
        if arguments["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(scene.parameters(), arguments["gradient_clip"])
        optimizer.step()
        projection_result = None
        projection_interval = int(arguments.get("surface_field_projection_interval", 0))
        projection_mode = arguments.get("surface_field_projection_mode", "none")
        projection_active = (
            (projection_mode == "stage_c" and schedule["stage"] in {"C", "D"})
            or (projection_mode == "stage_d" and schedule["stage"] == "D")
        )
        if (
            projection_active
            and projection_interval > 0
            and iteration % projection_interval == 0
        ):
            projection_eligible = (
                (support.detach() >= arguments.get("surface_field_minimum_support", 0.3))
                & (
                    evidence_ema
                    >= arguments.get("surface_field_minimum_evidence", 1.0e-4)
                )
            )
            projection_result = scene.project_means_to_surface(
                projection_eligible,
                arguments.get("surface_field_projection_step", 0.25),
                arguments.get("surface_field_projection_maximum_displacement", 0.02),
            )
        if iteration % arguments["lr_decay_interval"] == 0:
            current_base_lr *= float(arguments["lr_decay"])
            for group in optimizer.param_groups:
                group["lr"] *= float(arguments["lr_decay"])

        contribution_stack = torch.stack(contribution_values)
        visibility_stack = torch.stack(visibility_values)
        loss_gate_stack = torch.stack(loss_gate_values)
        observation_gate_stack = torch.stack(observation_gate_values)
        observed_visible_stack = torch.stack(observed_visible_values)
        batch_observation = observed_visible_stack.mean(0)
        observation_ema.mul_(ema_decay).add_(
            batch_observation, alpha=1.0 - ema_decay
        )
        if arguments.get("observation_aware_ema", False):
            observation_sum = observation_gate_stack.sum(0)
            was_in_frame = observation_sum > 1.0e-8
            batch_evidence = (
                contribution_stack * observation_gate_stack
            ).sum(0) / observation_sum.clamp_min(1.0e-8)
            batch_visible = observed_visible_stack.sum(0) / observation_sum.clamp_min(1.0e-8)
            evidence_ema[was_in_frame] = (
                ema_decay * evidence_ema[was_in_frame]
                + (1.0 - ema_decay) * batch_evidence[was_in_frame]
            )
            visible_ema[was_in_frame] = (
                ema_decay * visible_ema[was_in_frame]
                + (1.0 - ema_decay) * batch_visible[was_in_frame]
            )
            if (
                int(arguments.get("prune_min_negative_encounters", 0)) > 0
                or int(arguments.get("split_min_positive_encounters", 0)) > 0
            ):
                for encounter in {view.encounter for view in batch}:
                    view_indices = [
                        index for index, view in enumerate(batch)
                        if view.encounter == encounter
                    ]
                    encounter_index = encounter_to_index[encounter]
                    encounter_gate = observation_gate_stack[view_indices]
                    encounter_sum = encounter_gate.sum(0)
                    encounter_observed = encounter_sum > 1.0e-8
                    encounter_evidence = (
                        contribution_stack[view_indices] * encounter_gate
                    ).sum(0) / encounter_sum.clamp_min(1.0e-8)
                    encounter_observation_ema[encounter_index].mul_(ema_decay).add_(
                        encounter_observed.float(), alpha=1.0 - ema_decay
                    )
                    encounter_evidence_ema[encounter_index, encounter_observed] = (
                        ema_decay
                        * encounter_evidence_ema[encounter_index, encounter_observed]
                        + (1.0 - ema_decay)
                        * encounter_evidence[encounter_observed]
                    )
        else:
            batch_evidence = (contribution_stack * loss_gate_stack).mean(0)
            batch_visible = (visibility_stack * loss_gate_stack).mean(0)
            evidence_ema.mul_(ema_decay).add_(batch_evidence, alpha=1.0 - ema_decay)
            visible_ema.mul_(ema_decay).add_(batch_visible, alpha=1.0 - ema_decay)
        gradient_ema.mul_(ema_decay).add_(mean_gradient, alpha=1.0 - ema_decay)
        if direction_ema_decay > 0.0:
            direction_gradient_ema.mul_(direction_ema_decay).add_(
                adaptive_direction_gradient,
                alpha=1.0 - direction_ema_decay,
            )
        else:
            direction_gradient_ema.copy_(adaptive_direction_gradient)

        record = {
            "iteration": iteration,
            "stage": schedule["stage"],
            "loss": float(total.detach()),
            "image_loss": float(image_loss.detach()),
            "charbonnier": float(torch.stack(robust_values).mean().detach()),
            "log_charbonnier": float(torch.stack(log_values).mean().detach()),
            "ssim": float(torch.stack(ssim_values).mean().detach()),
            "support_mean": float(support_terms["support_sparsity"].detach()),
            "support_entropy": float(support_terms["support_entropy"].detach()),
            "support_graph": float(support_terms["support_graph"].detach()),
            "surface": float(surface_loss.detach()),
            "thinness": float(surface_terms["surface_thinness"].detach()),
            "local_plane": float(surface_terms["local_plane"].detach()),
            "normal_consistency": float(surface_terms["normal_consistency"].detach()),
            "surface_gap": float(surface_terms["surface_gap"].detach()),
            "bidirectional_plane": float(
                surface_terms["bidirectional_plane"].detach()
            ),
            "scale_smoothness": float(surface_terms["scale_smoothness"].detach()),
            "tangent_isotropy": float(surface_terms["tangent_isotropy"].detach()),
            "covariance_consistency": float(
                surface_terms["covariance_consistency"].detach()
            ),
            "coverage_band": float(surface_terms["coverage_band"].detach()),
            "surface_field_factor": schedule["surface_field_factor"],
            "surface_field_zero": float(field_terms["zero"].detach()),
            "surface_field_normal": float(field_terms["normal"].detach()),
            "surface_field_eikonal": float(field_terms["eikonal"].detach()),
            "surface_field_offset": float(field_terms["offset"].detach()),
            "surface_field_geometry_normal": float(
                field_terms["geometry_normal"].detach()
            ),
            "surface_field_selected": int(field_terms["selected"]),
            "surface_field_tangent_gradient_count": tangent_gradient_count,
            "background": float(scene.background_loss().detach()),
            "scattering_view_consistency": float(
                scattering_view_consistency.detach()
            ),
            "scattering_consistency_selected": int(
                scattering_eligible.sum().detach()
            ) if float(arguments.get("lambda_scattering_view_consistency", 0.0)) > 0.0
            else 0,
            "gaussians": scene.count,
            "means_trainable": geometry_activity["means"],
            "normals_trainable": geometry_activity["normals"],
            "scales_trainable": geometry_activity["scales"],
            "observation_mean": float(observation_ema.mean().detach()),
            "observation_coverage": float((observation_ema > 1.0e-4).float().mean().detach()),
            "views_per_batch": views_per_batch,
            "encounters": [view.encounter for view in batch],
            "frames": [view.frame_index for view in batch],
        }
        if pca_refresh is not None:
            record["pca_normal_refresh"] = pca_refresh
        if projection_result is not None:
            record["surface_field_projection"] = projection_result
        if device.type == "cuda":
            record["gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            record["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 2**30

        if iteration == 1 or iteration % arguments["log_interval"] == 0:
            memory_text = ""
            if device.type == "cuda":
                memory_text = (
                    f" gpu_peak={record['gpu_peak_allocated_gb']:.2f}/"
                    f"{record['gpu_peak_reserved_gb']:.2f}GB"
                )
            print(
                f"iter={iteration:06d} stage={schedule['stage']} "
                f"loss={record['loss']:.6f} ssim={record['ssim']:.5f} "
                f"q={record['support_mean']:.4f} G={scene.count}"
                f"{memory_text}"
            )

        if iteration % arguments["preview_interval"] == 0 or iteration == arguments["iters"]:
            view, target, raw_prediction, prediction, support_map, overlap_map = preview_items
            save_preview(
                output_dir / "render_preview" / f"iter_{iteration:06d}.png",
                target,
                raw_prediction,
                prediction,
                f"{view.encounter} frame {view.frame_index}",
                support=support_map,
                overlap=overlap_map,
            )

        adaptive_result = None
        if (
            schedule["stage"] == "C"
            and arguments["adaptive_interval"] > 0
            and iteration % arguments["adaptive_interval"] == 0
        ):
            minimum_negative_encounters = int(
                arguments.get("prune_min_negative_encounters", 0)
            )
            prune_eligible = None
            if minimum_negative_encounters > 0:
                persistent_observation = encounter_observation_ema >= float(
                    arguments.get("prune_encounter_observation_threshold", 0.10)
                )
                negative_evidence = encounter_evidence_ema < float(
                    arguments["prune_evidence_threshold"]
                )
                negative_encounter_count = (
                    persistent_observation & negative_evidence
                ).sum(0)
                prune_eligible = (
                    negative_encounter_count >= minimum_negative_encounters
                )
                record["negative_encounter_mean"] = float(
                    negative_encounter_count.float().mean().detach()
                )
                record["prune_encounter_eligible"] = int(prune_eligible.sum())
            if protected_inherited.any():
                if prune_eligible is None:
                    prune_eligible = torch.ones(
                        scene.count, dtype=torch.bool, device=device
                    )
                prune_eligible &= ~protected_inherited
                record["protected_inherited_gaussians"] = int(
                    protected_inherited.sum()
                )
            split_eligible = (
                split_age >= split_cooldown if split_cooldown > 0
                else torch.ones(scene.count, dtype=torch.bool, device=device)
            )
            minimum_positive_encounters = int(
                arguments.get("split_min_positive_encounters", 0)
            )
            if minimum_positive_encounters > 0:
                persistent_split_observation = encounter_observation_ema >= float(
                    arguments.get("split_encounter_observation_threshold", 0.10)
                )
                positive_evidence = encounter_evidence_ema >= float(
                    arguments["split_evidence_threshold"]
                )
                positive_encounter_count = (
                    persistent_split_observation & positive_evidence
                ).sum(0)
                split_eligible &= (
                    positive_encounter_count >= minimum_positive_encounters
                )
                record["positive_encounter_mean"] = float(
                    positive_encounter_count.float().mean().detach()
                )
                record["split_encounter_eligible"] = int(split_eligible.sum())
            adaptive_result = scene.adaptive_control(
                evidence_ema,
                gradient_ema,
                visible_ema,
                schedule["temperature"],
                arguments["prune_support_threshold"],
                arguments["prune_evidence_threshold"],
                arguments["prune_min_visible_fraction"],
                arguments["split_support_threshold"],
                arguments["split_gradient_threshold"],
                arguments["maximum_gaussians"],
                arguments["minimum_gaussians"],
                arguments["split_displacement_ratio"],
                arguments["split_scale_shrink"],
                arguments["split_mode"],
                arguments["split_evidence_threshold"],
                split_eligible,
                arguments.get("split_direction_mode", "tangent_axes"),
                direction_gradient_ema,
                arguments.get("relocation_noise_ratio", 0.5),
                arguments.get("relocation_maximum_fraction", 0.05),
                arguments.get("adaptive_knn_tangent_scale_ratio", 0.0),
                prune_eligible,
            )
            source = adaptive_result.pop("source_indices")
            split_reset_positions = adaptive_result.pop("split_reset_positions")
            evidence_ema = evidence_ema[source].clone()
            gradient_ema = gradient_ema[source].clone()
            direction_gradient_ema = direction_gradient_ema[source].clone()
            visible_ema = visible_ema[source].clone()
            observation_ema = observation_ema[source].clone()
            protected_inherited = protected_inherited[source].clone()
            encounter_observation_ema = encounter_observation_ema[:, source].clone()
            encounter_evidence_ema = encounter_evidence_ema[:, source].clone()
            split_age = split_age[source].clone()
            if len(split_reset_positions):
                split_age[split_reset_positions] = 0
            optimizer = _make_optimizer(scene, current_base_lr)
            if pca_mode == "adaptive_hard":
                adaptive_pca = scene.refresh_normals_pca(
                    evidence_ema,
                    arguments.get("pca_min_confidence", 1.0e-4),
                    hard_reset=True,
                    maximum_neighbor_distance=arguments.get(
                        "pca_max_neighbor_distance", 0.0
                    ),
                    maximum_surface_variation=arguments.get(
                        "pca_max_surface_variation", 1.0
                    ),
                    minimum_planar_spread_ratio=arguments.get(
                        "pca_min_planar_spread_ratio", 0.0
                    ),
                )
                record["adaptive_pca_normal_refresh"] = adaptive_pca
            if (
                projection_active
                and scene.surface_field is not None
            ):
                adaptive_support = scene.support_probability(schedule["temperature"])
                adaptive_projection_eligible = (
                    adaptive_support.detach()
                    >= arguments.get("surface_field_minimum_support", 0.3)
                ) & (
                    evidence_ema
                    >= arguments.get("surface_field_minimum_evidence", 1.0e-4)
                )
                newly_adapted = torch.zeros_like(adaptive_projection_eligible)
                newly_adapted[split_reset_positions] = True
                adaptive_projection_eligible = adaptive_projection_eligible & newly_adapted
                record["adaptive_surface_field_projection"] = (
                    scene.project_means_to_surface(
                        adaptive_projection_eligible,
                        arguments.get("surface_field_projection_step", 0.25),
                        arguments.get(
                            "surface_field_projection_maximum_displacement", 0.02
                        ),
                    )
                )
            record["adaptive_control"] = adaptive_result
            print(
                f"adaptive iter={iteration:06d} pruned={adaptive_result['pruned']} "
                f"split={adaptive_result['split']} "
                f"relocated={adaptive_result.get('relocated', 0)} "
                f"axes={adaptive_result['split_axis_t1']}/"
                f"{adaptive_result['split_axis_t2']} G={adaptive_result['count']}"
            )

        history.append(record)
        previous_stage = schedule["stage"]
        if iteration % arguments["checkpoint_interval"] == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"iter_{iteration:06d}.pt",
                scene,
                arguments,
                iteration,
            )

    elapsed = time.perf_counter() - start_time
    save_checkpoint(output_dir / "checkpoints" / "final.pt", scene, arguments, arguments["iters"])
    export_ply(
        output_dir / "gaussians_final.ply",
        scene,
        support_temperature=arguments["final_support_temperature"],
    )
    write_json(output_dir / "training_history.json", history)
    timing_report = {
        "device": str(device),
        "iterations": arguments["iters"],
        "views_per_batch": views_per_batch,
        "total_view_renders": arguments["iters"] * views_per_batch,
        "selected_views": len(views),
        "final_gaussians": scene.count,
        "total_seconds": elapsed,
        "seconds_per_iteration": elapsed / max(arguments["iters"], 1),
        "seconds_per_view_render": elapsed / max(arguments["iters"] * views_per_batch, 1),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        timing_report.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_gb": properties.total_memory / 2**30,
                "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
                "peak_allocated_fraction": torch.cuda.max_memory_allocated(device)
                / properties.total_memory,
            }
        )
    write_json(output_dir / "timing_report.json", timing_report)
    return history
