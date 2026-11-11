from __future__ import annotations

import json
from pathlib import Path


def load_encounter_batch_preset(
    path: str | Path, *, allow_incomplete: bool = False
) -> dict:
    """Read a generated GEIST batch manifest and return validated metadata.

    Incomplete manifests are rejected by default so they cannot accidentally be
    used for training.  ``allow_incomplete`` is reserved for explicit robustness
    evaluation, where every listed encounter is still validated and loaded.
    """
    preset_path = Path(path).expanduser().resolve()
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{preset_path}: preset JSON must contain an object")
    encounters = payload.get("encounters")
    if not isinstance(encounters, list) or not encounters:
        raise ValueError(f"{preset_path}: expected a non-empty 'encounters' list")

    names: list[str] = []
    frame_counts: list[int] = []
    for position, encounter in enumerate(encounters, start=1):
        if not isinstance(encounter, dict):
            raise ValueError(f"{preset_path}: encounters[{position - 1}] must be an object")
        name = encounter.get("folder_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{preset_path}: encounters[{position - 1}] has no usable folder_name"
            )
        names.append(name.strip())
        if encounter.get("frame_count") is not None:
            frame_counts.append(int(encounter["frame_count"]))
    if len(set(names)) != len(names):
        raise ValueError(f"{preset_path}: encounter folder_name values must be unique")

    selected_k = int(payload.get("selected_k", len(names)))
    if selected_k != len(names) and not allow_incomplete:
        raise ValueError(
            f"{preset_path}: selected_k={selected_k}, but {len(names)} encounters are listed"
        )
    member_k = {
        int(item["set_k"])
        for item in encounters
        if isinstance(item, dict) and item.get("set_k") is not None
    }
    if member_k and member_k != {selected_k}:
        raise ValueError(
            f"{preset_path}: encounter set_k values {sorted(member_k)} do not match "
            f"selected_k={selected_k}"
        )

    declared_frames = None
    if frame_counts:
        if len(frame_counts) != len(names) or len(set(frame_counts)) != 1:
            raise ValueError(
                f"{preset_path}: every encounter must declare the same frame_count"
            )
        declared_frames = frame_counts[0]
    top_frames = payload.get("frames_per_encounter")
    if top_frames is not None:
        top_frames = int(top_frames)
        if declared_frames is not None and top_frames != declared_frames:
            raise ValueError(
                f"{preset_path}: frames_per_encounter={top_frames} conflicts with "
                f"encounter frame_count={declared_frames}"
            )
        declared_frames = top_frames

    return {
        "path": str(preset_path),
        "schema": payload.get("schema"),
        "batch_timestamp": payload.get("batch_timestamp"),
        "design_mode": payload.get("design_mode"),
        "selected_k": selected_k,
        "available_k": len(names),
        "is_incomplete": selected_k != len(names),
        "encounter_names": names,
        "frames_per_encounter": declared_frames,
        "pulses_per_frame": payload.get("pulses_per_frame"),
        "cross_range_bins": payload.get("cross_range_bins"),
    }
