"""Align future incremental render previews with 100-iteration checkpoints."""
import json
from pathlib import Path


ROOT = Path("/root/autodl-fs/radar_gs_isar_enc/outputs/radar_gs/skynet5d4_incremental_v23_20260909")
for path in sorted((ROOT / "configs").glob("stage_k*_encounter_*.json")):
    k = int(path.stem.split("_")[1][1:])
    run = ROOT / "training" / path.stem
    if k <= 2 or run.exists():
        continue
    config = json.loads(path.read_text())
    config["preview_interval"] = 100
    path.write_text(json.dumps(config, indent=2))
    print(f"K={k}: preview_interval=100")
