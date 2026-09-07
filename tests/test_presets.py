import json

import pytest

from radar_gs.presets import load_encounter_batch_preset


def _write_preset(path, selected_k, names):
    path.write_text(
        json.dumps(
            {
                "selected_k": selected_k,
                "frames_per_encounter": 60,
                "encounters": [
                    {"folder_name": name, "frame_count": 60, "set_k": selected_k}
                    for name in names
                ],
            }
        ),
        encoding="utf-8",
    )


def test_incomplete_preset_is_rejected_for_training(tmp_path):
    path = tmp_path / "testset.json"
    _write_preset(path, 5, ["a", "b", "c"])
    with pytest.raises(ValueError, match="selected_k=5"):
        load_encounter_batch_preset(path)


def test_incomplete_preset_can_be_loaded_for_explicit_evaluation(tmp_path):
    path = tmp_path / "testset.json"
    _write_preset(path, 5, ["a", "b", "c"])
    result = load_encounter_batch_preset(path, allow_incomplete=True)
    assert result["selected_k"] == 5
    assert result["available_k"] == 3
    assert result["is_incomplete"] is True
    assert result["encounter_names"] == ["a", "b", "c"]
