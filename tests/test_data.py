from pathlib import Path

import numpy as np
from PIL import Image

from radar_gs.data import load_dataset
from radar_gs.presets import load_encounter_batch_preset


def _write_encounter(root: Path) -> None:
    image_dir = root / "isar_images_noaxis"
    image_dir.mkdir(parents=True)
    angles = np.deg2rad(np.arange(0.0, 21.0, 5.0))
    kr = np.stack(
        (np.sin(angles), np.zeros_like(angles), np.cos(angles)), axis=-1
    )
    kd = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(angles), 1))
    np.save(root / "view_angle.npy", np.concatenate((kr, kd), axis=-1))
    for index in range(len(angles)):
        image = np.zeros((16, 16), dtype=np.uint8)
        image[4:12, 4:12] = 40 + index * 30
        Image.fromarray(image, mode="L").save(image_dir / f"isar_{index:05d}.png")


def test_encounter_normalization_is_independent_of_los_sampling(tmp_path):
    encounter = tmp_path / "encounter_0001"
    _write_encounter(encounter)
    dense, dense_report = load_dataset(
        encounter, image_size=16, los_angle_deg=0.0, normalization_percentile=95.0
    )
    sparse, sparse_report = load_dataset(
        encounter, image_size=16, los_angle_deg=10.0, normalization_percentile=95.0
    )
    dense_info = dense_report["selection"][encounter.name]
    sparse_info = sparse_report["selection"][encounter.name]
    assert dense_info["normalization_scale"] == sparse_info["normalization_scale"]
    assert len(sparse) < len(dense)


def test_fixed_image_count_spans_aperture(tmp_path):
    encounter = tmp_path / "encounter_0001"
    _write_encounter(encounter)
    views, report = load_dataset(
        encounter,
        image_size=16,
        los_angle_deg=0.0,
        images_per_encounter=3,
    )
    assert [view.frame_index for view in views] == [0, 2, 4]
    information = report["selection"][encounter.name]
    assert information["selection_method"] == "fixed_image_count"
    assert information["selected"] == 3


def test_fixed_image_count_rejects_simultaneous_angle_sampling(tmp_path):
    encounter = tmp_path / "encounter_0001"
    _write_encounter(encounter)
    try:
        load_dataset(
            encounter,
            image_size=16,
            los_angle_deg=5.0,
            images_per_encounter=3,
        )
    except ValueError as error:
        assert "either images_per_encounter" in str(error)
    else:
        raise AssertionError("Expected incompatible fixed-count/angle selection to fail")


def test_fixed_image_count_rejects_insufficient_frames(tmp_path):
    encounter = tmp_path / "encounter_0001"
    _write_encounter(encounter)
    try:
        load_dataset(
            encounter,
            image_size=16,
            los_angle_deg=0.0,
            images_per_encounter=6,
        )
    except ValueError as error:
        assert "only 5 available frames" in str(error)
    else:
        raise AssertionError("Expected insufficient fixed-count selection to fail")


def test_batch_preset_uses_folder_names_and_selected_k(tmp_path):
    preset = tmp_path / "preset_batch.json"
    preset.write_text(
        """{
  "schema": "geist_independent_preset_batch_v2",
  "selected_k": 2,
  "frames_per_encounter": 60,
  "pulses_per_frame": 751,
  "cross_range_bins": 751,
  "encounters": [
    {"set_k": 2, "folder_name": "run_a", "folder_path": "/remote/run_a", "frame_count": 60},
    {"set_k": 2, "folder_name": "run_b", "folder_path": "/remote/run_b", "frame_count": 60}
  ]
}""",
        encoding="utf-8",
    )
    report = load_encounter_batch_preset(preset)
    assert report["selected_k"] == 2
    assert report["encounter_names"] == ["run_a", "run_b"]
    assert report["frames_per_encounter"] == 60
    assert report["cross_range_bins"] == 751
