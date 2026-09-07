from __future__ import annotations

import argparse
from pathlib import Path

import torch

from radar_gs.data import load_dataset
from radar_gs.io import load_scene, save_preview, write_json
from radar_gs.losses import (
    compress_amplitude,
    inverse_compress_amplitude,
    reconstruction_loss,
)
from radar_gs.renderer import RadarVisibilityRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and evaluate a trained radar-GS checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_views",
        type=int,
        default=0,
        help="Maximum frames to render; 0 renders every available frame.",
    )
    parser.add_argument("--include_estimated_background", action="store_true")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    scene, checkpoint = load_scene(args.checkpoint, device)
    config = checkpoint["arguments"]
    views, _ = load_dataset(
        args.data_root,
        image_size=config["image_size"],
        native_dr=config["native_dr"],
        native_dd=config["native_dd"],
        los_angle_deg=0.0,
        encounter_count=config.get("encounter_count"),
        encounter_names=config.get("encounter_names"),
        frames_per_encounter=config.get("frames_per_encounter"),
        frame_start=config.get("frame_start", 0),
        background_threshold=config.get("background_threshold", 0.0),
        normalization_percentile=config.get("normalization_percentile", 99.5),
    )
    renderer = RadarVisibilityRenderer(
        config["image_size"],
        config["image_size"],
        config["depth_bins"],
        config["kernel_radius"],
        config["minimum_pixel_variance"],
        config["opacity_gain"],
        config.get("visibility_pixel_size", 0.0),
    ).to(device)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.eval()
    scores = []
    evaluation_views = views if args.max_views <= 0 else views[: args.max_views]
    with torch.no_grad():
        for index, view in enumerate(evaluation_views):
            result = renderer(
                scene,
                view.kr,
                view.kd,
                view.dr,
                view.dd,
                support_temperature=config.get("final_support_temperature", 1.0),
            )
            gamma = config.get("render_gamma", 1.0)
            if args.include_estimated_background:
                level = torch.tensor(view.background_init, device=device)
                background = inverse_compress_amplitude(level, gamma)
            else:
                background = 0.0
            raw_prediction = compress_amplitude(
                result.raw_image + background,
                gamma,
            )
            prediction = compress_amplitude(
                result.image + background,
                gamma,
            )
            target = view.image.to(device)
            loss, metrics = reconstruction_loss(
                prediction,
                target,
                config.get("lambda_ssim", 0.1),
                config.get("lambda_log_amplitude", 0.2),
                config.get("charbonnier_epsilon", 1.0e-3),
                config.get("log_amplitude_gain", 20.0),
            )
            scores.append((float(loss), float(metrics["ssim"])))
            save_preview(
                output_dir / f"view_{index:04d}.png",
                target,
                raw_prediction,
                prediction,
                f"{view.encounter} frame {view.frame_index}",
                support=result.support_map,
                overlap=result.overlap_map,
            )
    mean_loss = sum(value[0] for value in scores) / len(scores)
    mean_ssim = sum(value[1] for value in scores) / len(scores)
    write_json(
        output_dir / "evaluation_metrics.json",
        {"views": len(scores), "mean_loss": mean_loss, "mean_ssim": mean_ssim},
    )
    print(f"views={len(scores)} mean_loss={mean_loss:.6f} mean_ssim={mean_ssim:.6f}")


if __name__ == "__main__":
    main()
