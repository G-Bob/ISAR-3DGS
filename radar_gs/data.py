from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from .geometry import orthonormalize_view, select_by_los_angle
from .radar_psf import (
    amplitude_psf,
    conservative_sidelobe_clean,
    decode_hot_db_rgb,
    decode_hot_index_rgb,
    decode_hot_normalized_db_rgb,
    encode_loader_amplitude,
    sidelobe_confidence_weight,
)


@dataclass
class ISARView:
    image: torch.Tensor
    kr: torch.Tensor
    kd: torch.Tensor
    normal: torch.Tensor
    dr: float
    dd: float
    encounter: str
    frame_index: int
    source_path: str
    training_index: int
    background_init: float
    loss_weight: torch.Tensor | None = None


def _numeric_key(path: Path) -> tuple:
    numbers = re.findall(r"\d+", path.name)
    return (0, *(int(value) for value in numbers)) if numbers else (1, path.name)


def _find_encounters(root: Path) -> list[Path]:
    if (root / "isar_images_noaxis").is_dir():
        return [root]
    encounters = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "isar_images_noaxis").is_dir()
    ]
    return sorted(encounters, key=_numeric_key)


def _select_fixed_count(indices: list[int], count: int | None) -> list[int]:
    """Select exactly ``count`` aperture-spanning indices, including both ends."""
    if count is None or count <= 0:
        return list(indices)
    if count > len(indices):
        raise ValueError(
            f"Requested {count} images from only {len(indices)} available frames"
        )
    if count == len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    selected = [indices[int(position)] for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError("Fixed-count view selection produced duplicate frame indices")
    return selected


def _find_view_angles(encounter: Path) -> Path:
    candidates = [encounter / "view_angle.npy", encounter / "Es" / "view_angle.npy"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No view_angle.npy at {encounter}/view_angle.npy or {encounter}/Es/view_angle.npy"
    )


def _read_amplitude_png_uint8(path: Path) -> np.ndarray:
    """Read the existing no-axis PNG without per-frame min-max scaling."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return rgb.max(axis=2)


def _read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_input_domain(path: Path, mode: str) -> np.ndarray:
    rgb = _read_rgb(path)
    if mode in {"control", "legacy_rgb_max", "psf_masked", "clean"}:
        return rgb.max(axis=2).astype(np.float32) / 255.0
    if mode == "inverse_hot_db":
        return decode_hot_normalized_db_rgb(rgb)
    if mode == "inverse_hot_linear":
        return decode_hot_db_rgb(rgb)
    raise ValueError(f"Unsupported amplitude processing mode: {mode}")


def _domain_scale(paths: list[Path], percentile: float, mode: str) -> float:
    """Exact percentile scale for monotonic 256-level hot-map domains."""
    if mode in {"control", "legacy_rgb_max", "psf_masked", "clean"}:
        return _encounter_scale(paths, percentile)
    histogram = np.zeros(256, dtype=np.int64)
    for path in paths:
        index = decode_hot_index_rgb(_read_rgb(path))
        histogram += np.bincount(index.reshape(-1), minlength=256)
    positive = histogram.copy()
    positive[0] = 0
    total = int(positive.sum())
    if total == 0:
        return 1.0
    rank = int(np.ceil(np.clip(percentile, 0.0, 100.0) / 100.0 * total))
    index = int(np.searchsorted(np.cumsum(positive), max(rank, 1)))
    if mode == "inverse_hot_db":
        return max(index / 255.0, 1.0 / 255.0)
    normalized_db = index / 255.0
    return max(10.0 ** ((-60.0 * (1.0 - normalized_db)) / 20.0), 1.0e-6)


def _encounter_scale(paths: list[Path], percentile: float) -> float:
    """Compute one robust scale shared by every frame in an encounter."""
    histogram = np.zeros(256, dtype=np.int64)
    for path in paths:
        image = _read_amplitude_png_uint8(path)
        histogram += np.bincount(image.reshape(-1), minlength=256)
    # Ignore exact black pixels when estimating the signal scale. They remain
    # in the training image and are not treated as a mask.
    positive = histogram.copy()
    positive[0] = 0
    total = int(positive.sum())
    if total == 0:
        return 1.0
    rank = int(np.ceil(np.clip(percentile, 0.0, 100.0) / 100.0 * total))
    value = int(np.searchsorted(np.cumsum(positive), max(rank, 1)))
    return max(value / 255.0, 1.0 / 255.0)


def _border_level(image: np.ndarray) -> float:
    border = max(1, min(image.shape) // 32)
    values = np.concatenate(
        (
            image[:border].reshape(-1),
            image[-border:].reshape(-1),
            image[:, :border].reshape(-1),
            image[:, -border:].reshape(-1),
        )
    )
    return float(np.median(values))


def _resize_amplitude(amplitude: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0 or amplitude.shape == (image_size, image_size):
        return amplitude
    image = Image.fromarray(np.uint8(np.clip(amplitude, 0.0, 1.0) * 255.0), mode="L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_dataset(
    data_root: str | Path,
    image_size: int = 256,
    native_dr: float = 0.0149896,
    native_dd: float = 0.0148244,
    los_angle_deg: float = 0.0,
    encounter_count: int | None = None,
    encounter_names: list[str] | None = None,
    frames_per_encounter: int | None = None,
    images_per_encounter: int | None = None,
    frame_start: int = 0,
    background_threshold: float = 0.0,
    normalization_percentile: float = 99.5,
    amplitude_processing_mode: str = "control",
    psf_beta_range: float = 3.0,
    psf_beta_doppler: float = 5.0,
    psf_mask_minimum_weight: float = 0.25,
    clean_gain: float = 0.1,
    clean_maximum_removed_energy_fraction: float = 0.1,
    legacy_background_threshold_for_decode: float = 0.1,
) -> tuple[list[ISARView], dict]:
    """Load PNG-only GEIST views and apply per-encounter LOS subsampling."""
    root = Path(data_root)
    encounters = _find_encounters(root)
    if encounter_names:
        by_name = {path.name: path for path in encounters}
        missing = [name for name in encounter_names if name not in by_name]
        if missing:
            raise FileNotFoundError(f"Requested encounters not found under {root}: {missing}")
        encounters = [by_name[name] for name in encounter_names]
    elif encounter_count is not None and encounter_count > 0:
        encounters = encounters[:encounter_count]
    if not encounters:
        raise FileNotFoundError(f"No encounter with isar_images_noaxis found under {root}")

    views: list[ISARView] = []
    selection_report: dict[str, dict] = {}
    for encounter in encounters:
        images = sorted((encounter / "isar_images_noaxis").glob("*.png"), key=_numeric_key)
        geometry = np.load(_find_view_angles(encounter))
        if geometry.ndim != 2 or geometry.shape[1] < 6:
            raise ValueError(f"{encounter}: view_angle.npy must have shape [N,6+]")
        count = min(len(images), len(geometry))
        stop = count
        if frames_per_encounter is not None and frames_per_encounter > 0:
            stop = min(stop, frame_start + frames_per_encounter)
        raw_indices = list(range(frame_start, stop))
        if not raw_indices:
            continue

        if images_per_encounter is not None and images_per_encounter > 0:
            if los_angle_deg > 0:
                raise ValueError(
                    "Use either images_per_encounter for equal-cost selection or "
                    "los_angle_deg for angular subsampling, not both"
                )
            chosen = _select_fixed_count(raw_indices, images_per_encounter)
            selection_method = "fixed_image_count"
        else:
            kr_all = torch.from_numpy(geometry[raw_indices, :3]).float()
            local_indices = select_by_los_angle(kr_all, los_angle_deg)
            chosen = [raw_indices[index] for index in local_indices]
            selection_method = "los_angle"
        selection_report[encounter.name] = {
            "available_after_range_filter": len(raw_indices),
            "selected": len(chosen),
            "los_angle_deg": los_angle_deg,
            "images_per_encounter": images_per_encounter,
            "selection_method": selection_method,
            "indices": chosen,
        }

        # The calibration set is independent of LOS subsampling. Angle
        # ablations and full-frame evaluation therefore retain the same
        # amplitude scale for the same encounter/range filter.
        calibration_paths = [images[index] for index in raw_indices]
        encounter_scale = _domain_scale(
            calibration_paths, normalization_percentile, amplitude_processing_mode
        )
        legacy_scale = _encounter_scale(calibration_paths, normalization_percentile)
        selection_report[encounter.name]["normalization_scale"] = encounter_scale
        selection_report[encounter.name]["normalization_percentile"] = normalization_percentile
        selection_report[encounter.name]["normalization_frame_count"] = len(raw_indices)
        if amplitude_processing_mode not in {
            "control", "legacy_rgb_max", "psf_masked", "clean",
            "inverse_hot_db", "inverse_hot_linear",
        }:
            raise ValueError("unsupported amplitude_processing_mode")
        raw_images = [_read_input_domain(images[index], amplitude_processing_mode) for index in chosen]
        processing_reports = []

        for index, raw_image in zip(chosen, raw_images):
            loss_weight = None
            processing_report = {"mode": amplitude_processing_mode}
            if amplitude_processing_mode in {"psf_masked", "clean"}:
                rgb = _read_rgb(images[index])
                linear_amplitude = decode_hot_db_rgb(rgb)
                psf = amplitude_psf(
                    linear_amplitude.shape[0], linear_amplitude.shape[1],
                    float(psf_beta_range), float(psf_beta_doppler),
                )
                if amplitude_processing_mode == "psf_masked":
                    weight_native, details = sidelobe_confidence_weight(
                        linear_amplitude, psf,
                        minimum_weight=float(psf_mask_minimum_weight),
                    )
                    loss_weight = _resize_amplitude(weight_native, image_size)
                elif amplitude_processing_mode == "clean":
                    cleaned, details = conservative_sidelobe_clean(
                        linear_amplitude, psf,
                        gain=float(clean_gain),
                        maximum_removed_energy_fraction=float(
                            clean_maximum_removed_energy_fraction
                        ),
                    )
                    raw_image = encode_loader_amplitude(cleaned)
                processing_report.update(details)
            processing_reports.append(processing_report)
            original = np.clip(raw_image.astype(np.float32) / encounter_scale, 0.0, 1.0)
            if amplitude_processing_mode in {"inverse_hot_db", "inverse_hot_linear"}:
                legacy = _read_amplitude_png_uint8(images[index]).astype(np.float32) / 255.0
                legacy = np.clip(legacy / legacy_scale, 0.0, 1.0)
                original[legacy < float(legacy_background_threshold_for_decode)] = 0.0
            elif background_threshold > 0:
                original[original < background_threshold] = 0.0
            original_height, original_width = original.shape
            resized = _resize_amplitude(original, image_size)
            height, width = resized.shape
            dr = native_dr * original_height / height
            dd = native_dd * original_width / width
            kr, kd, normal = orthonormalize_view(
                torch.from_numpy(geometry[index, :3]).float(),
                torch.from_numpy(geometry[index, 3:6]).float(),
            )
            views.append(
                ISARView(
                    image=torch.from_numpy(resized).float(),
                    kr=kr,
                    kd=kd,
                    normal=normal,
                    dr=float(dr),
                    dd=float(dd),
                    encounter=encounter.name,
                    frame_index=index,
                    source_path=str(images[index]),
                    training_index=len(views),
                    background_init=_border_level(resized),
                    loss_weight=(
                        torch.from_numpy(loss_weight).float()
                        if loss_weight is not None else None
                    ),
                )
            )
        selection_report[encounter.name]["amplitude_processing"] = {
            "mode": amplitude_processing_mode,
            "psf_beta_range": float(psf_beta_range),
            "psf_beta_doppler": float(psf_beta_doppler),
            "normalization_scale": float(encounter_scale),
            "legacy_normalization_scale": float(legacy_scale),
            "legacy_background_threshold_for_decode": float(
                legacy_background_threshold_for_decode
            ),
            "mean_masked_fraction": float(np.mean([
                report.get("masked_fraction", 0.0) for report in processing_reports
            ])),
            "mean_removed_energy_fraction": float(np.mean([
                report.get("removed_energy_fraction", 0.0) for report in processing_reports
            ])),
        }

    if not views:
        raise RuntimeError("No ISAR views remained after filtering")
    report = {
        "data_root": str(root),
        "total_selected_views": len(views),
        "selection": selection_report,
    }
    return views, report


def save_selection_report(report: dict, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
