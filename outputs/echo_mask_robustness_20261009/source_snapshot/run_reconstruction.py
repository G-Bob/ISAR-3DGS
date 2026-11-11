from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from radar_gs.data import load_dataset, save_selection_report
from radar_gs.initialization import initialize_surface
from radar_gs.io import export_ply, load_scene, write_json
from radar_gs.losses import inverse_compress_amplitude
from radar_gs.model import RadarGaussianScene
from radar_gs.presets import load_encounter_batch_preset
from radar_gs.renderer import RadarVisibilityRenderer
from radar_gs.trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radar-aware 3D Gaussian splatting for PNG-only multiview ISAR"
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--method", default="radargs", choices=["radargs"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--native_dr", type=float, default=0.0149896)
    parser.add_argument("--native_dd", type=float, default=0.0148244)
    parser.add_argument("--background_threshold", type=float, default=0.0)
    parser.add_argument("--normalization_percentile", type=float, default=99.5)
    parser.add_argument(
        "--amplitude_processing_mode",
        choices=[
            "control", "legacy_rgb_max", "psf_masked", "clean",
            "inverse_hot_db", "inverse_hot_linear",
        ],
        default="control",
    )
    parser.add_argument("--psf_beta_range", type=float, default=3.0)
    parser.add_argument("--psf_beta_doppler", type=float, default=5.0)
    parser.add_argument("--psf_mask_minimum_weight", type=float, default=0.25)
    parser.add_argument("--clean_gain", type=float, default=0.1)
    parser.add_argument("--clean_maximum_removed_energy_fraction", type=float, default=0.1)
    parser.add_argument("--legacy_background_threshold_for_decode", type=float, default=0.1)
    parser.add_argument("--encounter_count", type=int)
    parser.add_argument(
        "--encounter_preset_json",
        help=(
            "GEIST preset_batch JSON. Its encounter folder_name entries define the exact "
            "training set and selected_k; stored folder_path values are ignored."
        ),
    )
    parser.add_argument(
        "--encounter_names_file",
        help="Text file containing one exact encounter directory name per line",
    )
    parser.add_argument("--frames_per_encounter", type=int)
    parser.add_argument(
        "--images_per_encounter",
        type=int,
        help="Select exactly this many aperture-spanning frames from every encounter",
    )
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument(
        "--los_angle_deg",
        type=float,
        default=5.0,
        help="Minimum LOS separation. Larger values retain fewer images per encounter.",
    )
    parser.add_argument("--num_gaussians", type=int, default=10000)
    parser.add_argument("--bbox", type=float, default=6.0)
    parser.add_argument(
        "--initialization",
        choices=[
            "random_volume",
            "sphere_surface",
            "box_surface",
            "backprojection_surface",
            "pointcloud_surface",
            "mesh_surface",
        ],
        default="backprojection_surface",
    )
    parser.add_argument("--init_path", type=str)
    parser.add_argument(
        "--warm_start_checkpoint",
        type=str,
        help=(
            "Initialize the complete Gaussian/scattering model from a previous "
            "checkpoint while resetting optimizer state, the training schedule, "
            "and view-specific background coefficients."
        ),
    )
    parser.add_argument(
        "--continual_current_encounter",
        type=str,
        help="Encounter receiving the non-replay share of each continual batch.",
    )
    parser.add_argument(
        "--continual_replay_fraction",
        type=float,
        default=0.0,
        help="Fraction of every view batch sampled uniformly from earlier encounters.",
    )
    parser.add_argument(
        "--protect_warm_start_support_threshold",
        type=float,
        default=0.0,
        help="Prevent adaptive pruning of inherited Gaussians whose initial q meets this threshold; 0 disables protection.",
    )
    parser.add_argument("--initial_tangent_scale", type=float, default=0.08)
    parser.add_argument("--initial_normal_scale", type=float, default=0.012)
    parser.add_argument(
        "--tangent_covariance_mode",
        choices=("anisotropic", "isotropic_geometric"),
        default="anisotropic",
        help="Use independent tangent scales or an equal-area circular tangent covariance.",
    )
    parser.add_argument(
        "--normal_covariance_floor_ratio",
        type=float,
        default=0.0,
        help="Minimum rendered normal scale as a fraction of geometric-mean tangent scale.",
    )
    parser.add_argument(
        "--primitive_mode",
        choices=("gaussian3d", "surfel2d"),
        default="gaussian3d",
        help="Use the legacy full 3D covariance or a circular rank-2 surface splat.",
    )
    parser.add_argument(
        "--surfel_normal_scale",
        type=float,
        default=0.002,
        help="Fixed metre-scale normal thickness used only by surfel2d.",
    )
    parser.add_argument(
        "--surfel_fixed_normals",
        action="store_true",
        help="Use PCA/reference normals rather than gradient-learned normals in surfel2d.",
    )
    parser.add_argument(
        "--surface_field_mode",
        choices=("none", "udf", "offset_udf", "offset_sdf"),
        default="none",
        help="Enable a low-capacity implicit zero-set geometry prior.",
    )
    parser.add_argument("--surface_field_hidden_dim", type=int, default=32)
    parser.add_argument("--surface_field_layers", type=int, default=3)
    parser.add_argument("--surface_field_start_fraction", type=float, default=0.35)
    parser.add_argument("--surface_field_full_fraction", type=float, default=0.55)
    parser.add_argument("--lambda_surface_field", type=float, default=0.0)
    parser.add_argument("--lambda_surface_field_normal", type=float, default=0.0)
    parser.add_argument(
        "--lambda_surface_field_geometry_normal",
        type=float,
        default=0.0,
        help=(
            "Align Gaussian covariance normals to detached SDF gradients; unlike "
            "lambda_surface_field_normal, this term updates Gaussian orientation."
        ),
    )
    parser.add_argument("--lambda_surface_field_eikonal", type=float, default=0.0)
    parser.add_argument("--lambda_surface_field_offset", type=float, default=0.0)
    parser.add_argument("--surface_field_minimum_support", type=float, default=0.3)
    parser.add_argument("--surface_field_minimum_evidence", type=float, default=0.0001)
    parser.add_argument("--surface_field_sample_count", type=int, default=2048)
    parser.add_argument("--surface_field_eikonal_jitter", type=float, default=0.08)
    parser.add_argument("--surface_field_offset_distance", type=float, default=0.05)
    parser.add_argument(
        "--surface_field_offset_loss_mode",
        choices=("smooth_l1", "l1", "normalized_l1"),
        default="smooth_l1",
    )
    parser.add_argument(
        "--surface_field_detach_geometry",
        action="store_true",
        help="Train the field on detached Gaussian geometry; geometry changes only by explicit projection.",
    )
    parser.add_argument(
        "--adaptive_gradient_source",
        choices=("total", "base", "blended", "image"),
        default="total",
        help=(
            "Gradient used by adaptive split statistics. 'total' preserves legacy "
            "behaviour; 'base' excludes only the implicit-field loss; 'blended' "
            "adds a configurable fraction of the implicit-field loss; 'image' "
            "excludes all regularizer and implicit-field gradients."
        ),
    )
    parser.add_argument(
        "--adaptive_surface_field_gradient_weight",
        type=float,
        default=0.5,
        help="Implicit-field fraction used only when adaptive_gradient_source=blended.",
    )
    parser.add_argument(
        "--surface_field_projection_mode",
        choices=("none", "stage_c", "stage_d"),
        default="none",
    )
    parser.add_argument("--surface_field_projection_interval", type=int, default=500)
    parser.add_argument("--surface_field_projection_step", type=float, default=0.25)
    parser.add_argument(
        "--surface_field_projection_maximum_displacement", type=float, default=0.02
    )
    parser.add_argument("--surface_field_tangent_means", action="store_true")
    parser.add_argument("--feature_dim", type=int, default=16)
    parser.add_argument("--encoding_levels", type=int, default=4)
    parser.add_argument("--direction_encoding_levels", type=int, default=None,
                        help="Direction-only Fourier encoding levels; omitted preserves legacy")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--maximum_scattering_amplitude", type=float, default=1.0)
    parser.add_argument("--initial_support_probability", type=float, default=0.7)
    parser.add_argument("--background_mode", choices=["constant", "planar"], default="constant")
    parser.add_argument("--graph_neighbors", type=int, default=8)
    parser.add_argument("--graph_spatial_sigma", type=float, default=0.25)
    parser.add_argument("--graph_normal_sigma", type=float, default=0.25)
    parser.add_argument(
        "--surface_graph_mode",
        choices=("spatial_normal", "spatial"),
        default="spatial_normal",
    )
    parser.add_argument("--depth_bins", type=int, default=32)
    parser.add_argument("--kernel_radius", type=int, default=4)
    parser.add_argument(
        "--support_composition",
        choices=("noisy_or", "optical_density"),
        default="noisy_or",
    )
    parser.add_argument("--minimum_pixel_variance", type=float, default=0.35)
    parser.add_argument("--opacity_gain", type=float, default=1.0)
    parser.add_argument(
        "--visibility_pixel_size",
        type=float,
        default=0.0,
        help="Cross-LOS visibility-plane spacing in metres; 0 uses sqrt(dr*dd).",
    )
    parser.add_argument(
        "--disable_visibility",
        action="store_true",
        help="Ablation: train from raw splats without LOS transmittance.",
    )
    parser.add_argument("--iters", type=int, default=30000)
    parser.add_argument(
        "--views_per_batch",
        type=int,
        default=4,
        help="Number of independently rendered views retained for one backward pass.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr_decay", type=float, default=0.8)
    parser.add_argument("--lr_decay_interval", type=int, default=5000)
    parser.add_argument("--render_gamma", type=float, default=1.0)
    parser.add_argument("--lambda_ssim", type=float, default=0.1)
    parser.add_argument("--lambda_log_amplitude", type=float, default=0.2)
    parser.add_argument("--charbonnier_epsilon", type=float, default=1.0e-3)
    parser.add_argument("--log_amplitude_gain", type=float, default=20.0)
    parser.add_argument("--lambda_surface", type=float, default=0.1)
    parser.add_argument("--lambda_local_plane", type=float, default=0.2)
    parser.add_argument("--lambda_normal_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_reference_normal", type=float, default=0.02)
    parser.add_argument("--lambda_surface_gap", type=float, default=0.02)
    parser.add_argument("--lambda_bidirectional_plane", type=float, default=0.0)
    parser.add_argument("--lambda_scale_smoothness", type=float, default=0.0)
    parser.add_argument("--lambda_tangent_isotropy", type=float, default=0.0)
    parser.add_argument("--lambda_covariance_consistency", type=float, default=0.0)
    parser.add_argument("--lambda_tangent_repulsion", type=float, default=0.0)
    parser.add_argument("--tangent_repulsion_min_ratio", type=float, default=0.5)
    parser.add_argument("--lambda_coverage_band", type=float, default=0.0)
    parser.add_argument("--coverage_min_ratio", type=float, default=0.75)
    parser.add_argument("--coverage_max_ratio", type=float, default=1.25)
    parser.add_argument(
        "--pca_normal_refresh_mode",
        choices=("none", "stage_c_soft", "stage_c_hard", "adaptive_hard"),
        default="none",
    )
    parser.add_argument("--pca_min_confidence", type=float, default=1.0e-4)
    parser.add_argument(
        "--pca_max_neighbor_distance",
        type=float,
        default=0.0,
        help="Reject PCA refresh when the furthest kNN neighbour exceeds this distance; zero disables.",
    )
    parser.add_argument(
        "--pca_max_surface_variation",
        type=float,
        default=1.0,
        help="Maximum smallest-eigenvalue / eigenvalue-sum accepted by PCA refresh.",
    )
    parser.add_argument(
        "--pca_min_planar_spread_ratio",
        type=float,
        default=0.0,
        help="Minimum middle / largest covariance eigenvalue ratio accepted by PCA refresh.",
    )
    parser.add_argument("--surface_gap_scale_ratio", type=float, default=1.5)
    parser.add_argument(
        "--surface_gap_max_distance",
        type=float,
        default=0.25,
        help=(
            "Maximum neighbour distance affected by the surface-gap loss; "
            "larger structural separations are left unchanged"
        ),
    )
    parser.add_argument("--surface_gap_min_normal_cosine", type=float, default=0.8)
    parser.add_argument("--lambda_bbox", type=float, default=1.0)
    parser.add_argument("--lambda_support_sparsity", type=float, default=1.0e-3)
    parser.add_argument("--lambda_support_binary", type=float, default=1.0e-3)
    parser.add_argument("--lambda_support_graph", type=float, default=1.0e-2)
    parser.add_argument("--lambda_background", type=float, default=1.0e-2)
    parser.add_argument("--lambda_scattering_feature", type=float, default=1.0e-4)
    parser.add_argument(
        "--lambda_scattering_view_consistency",
        type=float,
        default=0.0,
        help="Penalize unsupported view-to-view scattering amplitude variation.",
    )
    parser.add_argument(
        "--scattering_consistency_minimum_support", type=float, default=0.0
    )
    parser.add_argument(
        "--scattering_consistency_minimum_observation", type=float, default=0.0
    )
    parser.add_argument("--maximum_normal_ratio", type=float, default=0.2)
    parser.add_argument("--stage_a_fraction", type=float, default=0.15)
    parser.add_argument("--stage_b_fraction", type=float, default=0.50)
    parser.add_argument("--stage_c_fraction", type=float, default=0.80)
    parser.add_argument("--final_support_temperature", type=float, default=0.35)
    parser.add_argument("--initial_visibility_gain", type=float, default=0.25)
    parser.add_argument("--evidence_ema_decay", type=float, default=0.98)
    parser.add_argument(
        "--split_direction_ema_decay",
        type=float,
        default=0.0,
        help=(
            "EMA decay for the vector used by center-gradient split direction; "
            "zero preserves the current-iteration behavior."
        ),
    )
    parser.add_argument("--visible_threshold", type=float, default=0.2)
    parser.add_argument(
        "--observation_aware_ema",
        action="store_true",
        help="Update evidence/visibility EMA only when a Gaussian projects inside the view.",
    )
    parser.add_argument(
        "--observation_weighted_support",
        action="store_true",
        help="Weight support sparsity/binary losses by accumulated observation confidence.",
    )
    parser.add_argument("--observation_fov_margin_pixels", type=float, default=0.0)
    parser.add_argument("--observation_support_weight_floor", type=float, default=0.0)
    parser.add_argument("--adaptive_interval", type=int, default=1000)
    parser.add_argument("--prune_support_threshold", type=float, default=0.05)
    parser.add_argument("--prune_evidence_threshold", type=float, default=1.0e-4)
    parser.add_argument("--prune_min_visible_fraction", type=float, default=0.3)
    parser.add_argument(
        "--prune_min_negative_encounters",
        type=int,
        default=0,
        help=(
            "Require persistent low evidence from at least this many independent "
            "encounters before pruning; zero preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--prune_encounter_observation_threshold",
        type=float,
        default=0.10,
        help="Minimum per-encounter observation EMA required to count negative evidence.",
    )
    parser.add_argument("--split_support_threshold", type=float, default=0.6)
    parser.add_argument("--split_gradient_threshold", type=float, default=1.0e-4)
    parser.add_argument("--split_evidence_threshold", type=float, default=0.0)
    parser.add_argument("--split_min_positive_encounters", type=int, default=0)
    parser.add_argument(
        "--split_encounter_observation_threshold", type=float, default=0.10
    )
    parser.add_argument("--split_displacement_ratio", type=float, default=0.3)
    parser.add_argument("--split_scale_shrink", type=float, default=0.8)
    parser.add_argument(
        "--split_mode",
        choices=(
            "none",
            "asymmetric",
            "asymmetric_conservative",
            "symmetric",
            "symmetric_conservative",
            "relocate",
        ),
        default="symmetric",
        help="Adaptive split placement; pruning remains enabled for all modes.",
    )
    parser.add_argument(
        "--split_cooldown_iterations",
        type=int,
        default=0,
        help=(
            "Minimum training iterations before either child of a split may "
            "be split again; zero disables the cooldown."
        ),
    )
    parser.add_argument(
        "--split_direction_mode",
        choices=("tangent_axes", "local_pca", "center_gradient"),
        default="tangent_axes",
        help=(
            "Choose split displacement from the arbitrary tangent basis, local "
            "projected kNN geometry, or the projected center-loss gradient."
        ),
    )
    parser.add_argument(
        "--adaptive_knn_tangent_scale_ratio",
        type=float,
        default=0.0,
        help="After adaptive control, set tangent scale from median kNN spacing; zero disables.",
    )
    parser.add_argument("--maximum_gaussians", type=int, default=20000)
    parser.add_argument("--minimum_gaussians", type=int, default=2000)
    parser.add_argument("--relocation_noise_ratio", type=float, default=0.5)
    parser.add_argument("--relocation_maximum_fraction", type=float, default=0.05)
    parser.add_argument("--gradient_clip", type=float, default=5.0)
    parser.add_argument("--means_start_fraction", type=float, default=0.0)
    parser.add_argument("--normals_start_fraction", type=float, default=0.0)
    parser.add_argument("--scales_start_fraction", type=float, default=0.0)
    parser.add_argument("--freeze_means", action="store_true")
    parser.add_argument("--freeze_normals", action="store_true")
    parser.add_argument("--freeze_scales", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=5000)
    parser.add_argument("--preview_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate preset, encounter folders, frames and view geometry without training",
    )
    # Accepted for compatibility with the previous run_reconstruction.py CLI.
    parser.add_argument("--no_complex", action="store_true", default=True)
    parser.add_argument("--num_rays", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sample_count", type=int, help=argparse.SUPPRESS)
    return parser


def parse_arguments() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    preliminary, _ = config_parser.parse_known_args()
    parser = build_parser()
    if preliminary.config:
        values = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        parser.set_defaults(**values)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.lambda_surface_gap < 0:
        raise ValueError("--lambda_surface_gap cannot be negative")
    for name in (
        "lambda_bidirectional_plane",
        "lambda_scale_smoothness",
        "lambda_coverage_band",
    ):
        if getattr(arguments, name) < 0:
            raise ValueError(f"--{name} cannot be negative")
    if not 0 <= arguments.coverage_min_ratio < arguments.coverage_max_ratio:
        raise ValueError("coverage ratios must satisfy 0 <= min < max")
    if arguments.pca_min_confidence < 0:
        raise ValueError("--pca_min_confidence cannot be negative")
    if arguments.surface_gap_scale_ratio <= 0:
        raise ValueError("--surface_gap_scale_ratio must be positive")
    if arguments.surface_gap_max_distance <= 0:
        raise ValueError("--surface_gap_max_distance must be positive")
    if not 0.0 <= arguments.surface_gap_min_normal_cosine <= 1.0:
        raise ValueError("--surface_gap_min_normal_cosine must be in [0, 1]")
    if arguments.split_displacement_ratio <= 0:
        raise ValueError("--split_displacement_ratio must be positive")
    if arguments.split_evidence_threshold < 0:
        raise ValueError("--split_evidence_threshold cannot be negative")
    if not 0.0 < arguments.split_scale_shrink <= 1.0:
        raise ValueError("--split_scale_shrink must be in (0, 1]")
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU")
        arguments.device = "cpu"
    device = torch.device(arguments.device)

    encounter_names = None
    preset_metadata = None
    if arguments.encounter_preset_json:
        if arguments.encounter_names_file:
            raise ValueError(
                "Use either --encounter_preset_json or --encounter_names_file, not both"
            )
        preset_metadata = load_encounter_batch_preset(arguments.encounter_preset_json)
        encounter_names = preset_metadata["encounter_names"]
        selected_k = int(preset_metadata["selected_k"])
        if arguments.encounter_count is not None and arguments.encounter_count != selected_k:
            raise ValueError(
                f"--encounter_count={arguments.encounter_count} conflicts with preset "
                f"selected_k={selected_k}"
            )
        arguments.encounter_count = selected_k
        declared_frames = preset_metadata.get("frames_per_encounter")
        if declared_frames is not None:
            if arguments.frames_per_encounter is None:
                arguments.frames_per_encounter = int(declared_frames)
            elif arguments.frames_per_encounter > int(declared_frames):
                raise ValueError(
                    f"--frames_per_encounter={arguments.frames_per_encounter} exceeds preset "
                    f"frame_count={declared_frames}"
                )
        print(
            f"Preset selected K={selected_k}: {', '.join(encounter_names)}",
            flush=True,
        )
    if arguments.encounter_names_file:
        encounter_names = [
            line.strip()
            for line in Path(arguments.encounter_names_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not encounter_names:
            raise ValueError("encounter_names_file contains no encounter names")
    arguments.encounter_names = encounter_names
    arguments.encounter_preset_metadata = preset_metadata

    views, selection = load_dataset(
        arguments.data_root,
        image_size=arguments.image_size,
        native_dr=arguments.native_dr,
        native_dd=arguments.native_dd,
        los_angle_deg=arguments.los_angle_deg,
        encounter_count=arguments.encounter_count,
        encounter_names=encounter_names,
        frames_per_encounter=arguments.frames_per_encounter,
        images_per_encounter=arguments.images_per_encounter,
        frame_start=arguments.frame_start,
        background_threshold=arguments.background_threshold,
        normalization_percentile=arguments.normalization_percentile,
        amplitude_processing_mode=arguments.amplitude_processing_mode,
        psf_beta_range=arguments.psf_beta_range,
        psf_beta_doppler=arguments.psf_beta_doppler,
        psf_mask_minimum_weight=arguments.psf_mask_minimum_weight,
        clean_gain=arguments.clean_gain,
        clean_maximum_removed_energy_fraction=arguments.clean_maximum_removed_energy_fraction,
        legacy_background_threshold_for_decode=arguments.legacy_background_threshold_for_decode,
    )
    if preset_metadata is not None:
        selection["encounter_preset"] = preset_metadata
        declared_frames = preset_metadata.get("frames_per_encounter")
        if declared_frames is not None and arguments.frames_per_encounter == declared_frames:
            incomplete = {
                name: information["available_after_range_filter"]
                for name, information in selection["selection"].items()
                if information["available_after_range_filter"] != declared_frames
            }
            if incomplete:
                raise ValueError(
                    "Preset/data frame-count mismatch; expected "
                    f"{declared_frames} frames per encounter, got {incomplete}"
                )
    output_dir = Path(arguments.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_selection_report(selection, output_dir / "selected_views.json")
    write_json(output_dir / "run_config.json", vars(arguments))
    print(f"Loaded {len(views)} selected PNG views")
    if arguments.validate_only:
        print(f"Validation complete; reports written to {output_dir}")
        return

    background_levels = torch.tensor([view.background_init for view in views])
    background_levels = inverse_compress_amplitude(
        background_levels, arguments.render_gamma
    )
    if arguments.warm_start_checkpoint:
        checkpoint_path = Path(arguments.warm_start_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")
        scene, inherited_checkpoint = load_scene(checkpoint_path, device)
        coefficient_count = 1 if scene.background_mode == "constant" else 3
        coefficients = torch.zeros(
            (len(background_levels), coefficient_count),
            dtype=background_levels.dtype,
            device=device,
        )
        if len(background_levels):
            levels = background_levels.to(device).clamp_min(1.0e-6)
            coefficients[:, 0] = torch.log(torch.expm1(levels))
        scene.background_coefficients = torch.nn.Parameter(coefficients)
        print(
            f"Warm-started {scene.count} Gaussians from {checkpoint_path} "
            f"(source iteration {inherited_checkpoint.get('iteration', 'unknown')}); "
            f"reset {len(background_levels)} view backgrounds"
        )
    else:
        means, normals = initialize_surface(
            arguments.initialization,
            arguments.num_gaussians,
            arguments.bbox,
            arguments.seed,
            views=views,
            init_path=arguments.init_path,
        )
        scene = RadarGaussianScene(
            means,
            normals,
            arguments.initial_tangent_scale,
            arguments.initial_normal_scale,
            arguments.bbox,
            arguments.feature_dim,
            arguments.encoding_levels,
            arguments.hidden_dim,
            arguments.maximum_scattering_amplitude,
            arguments.initial_support_probability,
            background_levels,
            arguments.background_mode,
            arguments.graph_neighbors,
            arguments.tangent_covariance_mode,
            arguments.normal_covariance_floor_ratio,
            arguments.primitive_mode,
            arguments.surfel_normal_scale,
            arguments.surfel_fixed_normals,
            arguments.surface_field_mode,
            arguments.surface_field_hidden_dim,
            arguments.surface_field_layers,
            direction_encoding_levels=arguments.direction_encoding_levels,
        ).to(device)
    export_ply(output_dir / "gaussians_initial.ply", scene)
    renderer = RadarVisibilityRenderer(
        arguments.image_size,
        arguments.image_size,
        arguments.depth_bins,
        arguments.kernel_radius,
        arguments.minimum_pixel_variance,
        arguments.opacity_gain,
        arguments.visibility_pixel_size,
        arguments.support_composition,
    ).to(device)
    train(scene, renderer, views, output_dir, vars(arguments))


if __name__ == "__main__":
    main()
