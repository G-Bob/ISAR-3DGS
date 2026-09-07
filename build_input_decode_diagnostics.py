from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from radar_gs.data import _domain_scale, _encounter_scale, _read_amplitude_png_uint8
from radar_gs.radar_psf import (
    _hot_lut,
    decode_hot_db_rgb,
    decode_hot_index_rgb,
    decode_hot_normalized_db_rgb,
)


ROOT = Path("../datasets/skynet5d")
OUT = Path("outputs/radar_gs/k5_input_decode_20260828/diagnostics")
PRESET = ROOT / "preset_batch_20260822_5.json"


def numeric_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in path.stem.replace("Es_", "").split("_") if part.isdigit())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preset = json.loads(PRESET.read_text())
    names = [entry["folder_name"] for entry in preset["encounters"]]
    paths: list[Path] = []
    per_encounter = {}
    for name in names:
        images = sorted((ROOT / name / "isar_images_noaxis").glob("*.png"), key=numeric_key)[:60]
        paths.extend(images)
        per_encounter[name] = {
            "legacy_scale": _encounter_scale(images, 99.5),
            "inverse_hot_db_scale": _domain_scale(images, 99.5, "inverse_hot_db"),
            "inverse_hot_linear_scale": _domain_scale(images, 99.5, "inverse_hot_linear"),
        }

    lut = _hot_lut()
    with (OUT / "hot_lut.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("index", "r", "g", "b", "normalized_db", "db", "linear_amplitude", "legacy_rgb_max"))
        for index, colour in enumerate(lut):
            normalized = index / 255.0
            db = -60.0 * (1.0 - normalized)
            writer.writerow((index, *map(int, colour), normalized, db, 10.0 ** (db / 20.0), int(colour.max()) / 255.0))

    sample_paths = paths[::20]
    values = {"legacy_rgb_max": [], "inverse_hot_db": [], "inverse_hot_linear": []}
    exact = []
    saturated = []
    for path in sample_paths:
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        index = decode_hot_index_rgb(rgb)
        exact.append(float(np.mean(lut[index] == rgb)))
        legacy = rgb.max(axis=2).astype(np.float32) / 255.0
        values["legacy_rgb_max"].append(legacy.reshape(-1))
        values["inverse_hot_db"].append(decode_hot_normalized_db_rgb(rgb).reshape(-1))
        values["inverse_hot_linear"].append(decode_hot_db_rgb(rgb).reshape(-1))
        saturated.append(float(np.mean(legacy >= 1.0)))

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for axis, (name, arrays) in zip(axes, values.items()):
        value = np.concatenate(arrays)
        axis.hist(value[value > 0], bins=100, log=True, color="#3877d6")
        axis.set_title(name)
        axis.set_xlabel("normalized input value")
        axis.set_ylabel("pixel count")
    figure.tight_layout()
    figure.savefig(OUT / "input_decode_histograms.png", dpi=180)
    plt.close(figure)

    rgb = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.uint8)
    panels = (
        ("legacy RGB max", rgb.max(axis=2).astype(np.float32) / 255.0),
        ("inverse hot dB", decode_hot_normalized_db_rgb(rgb)),
        ("inverse hot linear", decode_hot_db_rgb(rgb)),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for axis, (title, value) in zip(axes, panels):
        axis.imshow(value, cmap="gray", vmin=0.0, vmax=float(value.max()))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUT / "representative_frame_decode.png", dpi=180)
    plt.close(figure)

    report = {
        "sample_frame_count": len(sample_paths),
        "roundtrip_rgb_channel_exact_fraction": float(np.mean(exact)),
        "legacy_rgb_max_saturated_pixel_fraction": float(np.mean(saturated)),
        "per_encounter_scales": per_encounter,
        "background_mask_policy": "decoded modes reuse the exact legacy normalized RGB-max >= 0.1 spatial mask",
    }
    (OUT / "input_decode_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
