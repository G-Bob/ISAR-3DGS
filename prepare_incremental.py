"""Prepare an eight-stage continual encounter-training chain for skynet5d-4."""
import json
from pathlib import Path


BASE = Path("/root/autodl-fs/radar_gs_isar_enc")
SOURCE = BASE / "outputs/radar_gs/skynet5d4_k1_k8_v22_20260909"
ROOT = BASE / "outputs/radar_gs/skynet5d4_incremental_v23_20260909"
DATA = Path("/root/Yao/datasets/skynet5d-4")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "jobs.json").exists():
        raise RuntimeError("Existing incremental manifest: refuse to overwrite")
    audit = json.loads((SOURCE / "acquisition_audit.json").read_text())
    assert len(audit) == 8
    baseline = json.loads((SOURCE / "configs/CPA50_k01_angular_low.json").read_text())
    gt = SOURCE / "gt_native_cad_recovered.glb"
    jobs = []
    for k, encounter in enumerate(audit, start=1):
        current_names = ROOT / "presets" / f"stage_k{k:02d}_current.txt"
        current_names.parent.mkdir(parents=True, exist_ok=True)
        current_names.write_text(encounter["folder_name"] + "\n")
        cumulative = ROOT / "presets" / f"stage_k{k:02d}_cumulative.json"
        write_json(cumulative, {
            "schema": "incremental_evaluation_prefix_v1",
            "selected_k": k,
            "encounters": [
                {"folder_name": item["folder_name"]} for item in audit[:k]
            ],
        })
        name = f"stage_k{k:02d}_encounter_{k:02d}"
        run_dir = ROOT / "training" / name
        previous = None if k == 1 else ROOT / "training" / f"stage_k{k-1:02d}_encounter_{k-1:02d}" / "checkpoints/final.pt"
        config = dict(baseline)
        config.update(
            data_root=str(DATA),
            out_dir=str(run_dir),
            config=None,
            encounter_preset_json=None,
            encounter_preset_metadata=None,
            encounter_names=None,
            encounter_names_file=str(current_names),
            encounter_count=1,
            frames_per_encounter=None,
            images_per_encounter=None,
            frame_start=0,
            los_angle_deg=0,
            direction_encoding_levels=1,
            encoding_levels=4,
            split_min_positive_encounters=1,
            seed=42,
            iters=6000,
            checkpoint_interval=100,
            warm_start_checkpoint=str(previous) if previous else None,
            validate_only=False,
        )
        config_path = ROOT / "configs" / f"{name}.json"
        write_json(config_path, config)
        jobs.append({
            "name": name,
            "k": k,
            "current_encounter": encounter["folder_name"],
            "current_frames": encounter["frame_count"],
            "config": str(config_path),
            "run_dir": str(run_dir),
            "warm_start": str(previous) if previous else None,
            "cumulative_preset": str(cumulative),
            "gt": str(gt),
        })
    write_json(ROOT / "jobs.json", jobs)
    write_json(ROOT / "protocol.json", {
        "mode": "continual sequential encounter adaptation",
        "semantics": "stage K inherits stage K-1 model and trains only on encounter K",
        "optimizer": "reset at every stage",
        "four_stage_schedule": "reset at every stage",
        "iterations_per_stage": 6000,
        "total_iterations": 48000,
        "checkpoint_interval": 100,
        "per_view_exposure": "approximately fixed because every encounter contains 61-64 frames",
        "evaluation": ["current encounter image fit", "cumulative encounters 1..K image fit", "native-CAD geometry"],
        "seed": 42,
        "source_code_policy": "isolated v23 copy; v21/v22 unchanged",
    })
    print(json.dumps({"jobs": len(jobs), "frames_per_stage": [j["current_frames"] for j in jobs], "root": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
