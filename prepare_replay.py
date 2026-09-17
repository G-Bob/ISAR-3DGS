"""Prepare two replay chains against the completed pure-sequential control."""
import json
from pathlib import Path


BASE = Path("/root/autodl-fs/radar_gs_isar_enc")
CONTROL = BASE / "outputs/radar_gs/skynet5d4_incremental_v23_20260909"
ROOT = BASE / "outputs/radar_gs/skynet5d4_replay_v24_20260910"
DATA = Path("/root/Yao/datasets/skynet5d-4")


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "chains.json").exists():
        raise RuntimeError("Existing replay manifest: refuse to overwrite")
    audit = json.loads((CONTROL / "acquisition_audit.json").read_text())
    baseline = json.loads((CONTROL / "configs/stage_k02_encounter_02.json").read_text())
    gt = Path(json.loads((CONTROL / "jobs.json").read_text())[0]["gt"])
    variants = {
        "replay25": {"replay_fraction": 0.25, "protect_threshold": 0.0},
        "replay50_protected": {"replay_fraction": 0.50, "protect_threshold": 0.50},
    }
    chains = {}
    for variant, settings in variants.items():
        jobs = []
        variant_root = ROOT / variant
        for k in range(2, 9):
            current = audit[k - 1]
            names = variant_root / "presets" / f"stage_k{k:02d}_training.txt"
            names.parent.mkdir(parents=True, exist_ok=True)
            names.write_text("\n".join(e["folder_name"] for e in audit[:k]) + "\n")
            current_preset = variant_root / "presets" / f"stage_k{k:02d}_current.json"
            cumulative_preset = variant_root / "presets" / f"stage_k{k:02d}_cumulative.json"
            write(current_preset, {"selected_k": 1, "encounters": [{"folder_name": current["folder_name"]}]})
            write(cumulative_preset, {"selected_k": k, "encounters": [{"folder_name": e["folder_name"]} for e in audit[:k]]})
            name = f"stage_k{k:02d}_{variant}"
            run = variant_root / "training" / name
            previous = (
                CONTROL / "training/stage_k01_encounter_01/checkpoints/final.pt"
                if k == 2 else
                variant_root / "training" / f"stage_k{k-1:02d}_{variant}" / "checkpoints/final.pt"
            )
            config = dict(baseline)
            config.update(
                data_root=str(DATA), out_dir=str(run), config=None,
                encounter_preset_json=None, encounter_preset_metadata=None,
                encounter_names=None, encounter_names_file=str(names),
                encounter_count=k, frames_per_encounter=None,
                images_per_encounter=None, frame_start=0, los_angle_deg=0,
                iters=6000, checkpoint_interval=100, preview_interval=100,
                seed=42, warm_start_checkpoint=str(previous),
                continual_current_encounter=current["folder_name"],
                continual_replay_fraction=settings["replay_fraction"],
                protect_warm_start_support_threshold=settings["protect_threshold"],
                validate_only=False,
            )
            config_path = variant_root / "configs" / f"{name}.json"
            write(config_path, config)
            jobs.append({"name": name, "variant": variant, "k": k, "config": str(config_path), "run_dir": str(run), "current_preset": str(current_preset), "cumulative_preset": str(cumulative_preset), "gt": str(gt)})
        write(variant_root / "jobs.json", jobs)
        write(variant_root / "protocol.json", {"shared_k1_control": str(CONTROL), "iterations_per_stage": 6000, "checkpoint_interval": 100, **settings})
        chains[variant] = jobs
    write(ROOT / "chains.json", chains)
    print(json.dumps({k: len(v) for k, v in chains.items()}, indent=2))


if __name__ == "__main__":
    main()
