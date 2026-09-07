"""Read-only live dashboard for RadarGS checkpoints and training previews."""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
from pathlib import Path

import numpy as np
import torch
from flask import Flask, abort, jsonify, render_template, send_file


ITERATION_RE = re.compile(
    r"iter=(\d+)\s+stage=([A-D])\s+loss=([0-9.eE+-]+)\s+ssim=([0-9.eE+-]+)"
    r"\s+q=([0-9.eE+-]+)\s+G=(\d+)"
)
CHECKPOINT_RE = re.compile(r"iter_(\d+)\.pt$")


def stage_for_iteration(iteration: int, arguments: dict) -> str:
    total = max(int(arguments.get("iters", 1)), 1)
    fraction = iteration / total
    if fraction < float(arguments.get("stage_a_fraction", 0.15)):
        return "A"
    if fraction < float(arguments.get("stage_b_fraction", 0.50)):
        return "B"
    if fraction < float(arguments.get("stage_c_fraction", 0.80)):
        return "C"
    return "D"


class CheckpointCache:
    def __init__(self, output_root: Path, maximum_points: int):
        self.root = output_root.resolve()
        self.maximum_points = maximum_points
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[str, dict]] = {}

    @property
    def training_root(self) -> Path:
        return self.root / "training"

    def safe_run(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            abort(404)
        path = (self.training_root / name).resolve()
        if path.parent != self.training_root.resolve():
            abort(404)
        return path

    @staticmethod
    def latest_checkpoint(run: Path) -> tuple[Path | None, int]:
        choices: list[tuple[int, Path]] = []
        for path in (run / "checkpoints").glob("iter_*.pt"):
            match = CHECKPOINT_RE.search(path.name)
            if match:
                choices.append((int(match.group(1)), path))
        final = run / "checkpoints" / "final.pt"
        if final.exists():
            try:
                payload = torch.load(final, map_location="cpu", weights_only=False)
                choices.append((int(payload.get("iteration", 0)), final))
            except Exception:
                pass
        return (max(choices, default=(0, None), key=lambda item: item[0])[1],
                max(choices, default=(0, None), key=lambda item: item[0])[0])

    @staticmethod
    def latest_log_record(run: Path) -> dict:
        log = run / "train.log"
        record: dict = {}
        if not log.exists():
            return record
        try:
            tail = log.read_bytes()[-128_000:].decode("utf-8", errors="ignore")
            matches = list(ITERATION_RE.finditer(tail))
            if matches:
                match = matches[-1]
                record = {
                    "logged_iteration": int(match.group(1)),
                    "logged_stage": match.group(2),
                    "loss": float(match.group(3)),
                    "ssim": float(match.group(4)),
                    "mean_support": float(match.group(5)),
                    "logged_gaussians": int(match.group(6)),
                }
            record["log_updated_unix"] = log.stat().st_mtime
        except OSError:
            pass
        return record

    def list_runs(self) -> list[dict]:
        states = {}
        for status in (self.root / "status").glob("*.json"):
            try:
                states[status.stem] = json.loads(status.read_text())
            except (OSError, json.JSONDecodeError):
                continue
        names = set(states)
        if self.training_root.exists():
            names.update(path.name for path in self.training_root.iterdir() if path.is_dir())
        rows = []
        for name in sorted(names):
            run = self.safe_run(name)
            checkpoint, checkpoint_iteration = self.latest_checkpoint(run)
            status = states.get(name, {})
            row = {
                "name": name,
                "scenario": status.get("scenario", name.split("_", 1)[0]),
                "k": status.get("k"),
                "variant": status.get("variant"),
                "state": status.get("state", "training" if run.exists() else "queued"),
                "checkpoint_iteration": checkpoint_iteration,
                "has_checkpoint": checkpoint is not None,
            }
            row.update(self.latest_log_record(run))
            rows.append(row)
        return rows

    def load_cloud(self, name: str) -> dict:
        run = self.safe_run(name)
        checkpoint, iteration = self.latest_checkpoint(run)
        if checkpoint is None:
            abort(404, "No checkpoint is available yet")
        fingerprint = f"{checkpoint}:{checkpoint.stat().st_mtime_ns}:{checkpoint.stat().st_size}"
        with self._lock:
            cached = self._cache.get(name)
            if cached and cached[0] == fingerprint:
                return cached[1]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload["model"]
        means = state["means"].detach().float().numpy()
        support_key = "support_logits" if "support_logits" in state else "opacity_logits"
        support = torch.sigmoid(state[support_key].detach().float()).numpy()
        scales = state.get("log_scales")
        scales = scales.detach().float().exp().numpy() if scales is not None else np.zeros_like(means)
        count = len(means)
        if count > self.maximum_points:
            generator = np.random.default_rng(42)
            keep = np.sort(generator.choice(count, self.maximum_points, replace=False))
            means, support, scales = means[keep], support[keep], scales[keep]
        arguments = payload.get("arguments", {})
        result = {
            "name": name,
            "checkpoint": checkpoint.name,
            "iteration": int(payload.get("iteration", iteration)),
            "stage": stage_for_iteration(int(payload.get("iteration", iteration)), arguments),
            "gaussian_count": count,
            "displayed_count": len(means),
            "support_ge_03": int(np.count_nonzero(support >= 0.3)),
            "support_mean": float(np.mean(support)),
            "xyz": means.round(5).tolist(),
            "support": support.round(5).tolist(),
            "scale": np.mean(scales, axis=1).round(5).tolist(),
        }
        result.update(self.latest_log_record(run))
        with self._lock:
            self._cache[name] = (fingerprint, result)
        return result

    def preview(self, name: str) -> Path:
        run = self.safe_run(name)
        checkpoint, iteration = self.latest_checkpoint(run)
        previews = []
        for path in (run / "render_preview").glob("iter_*.png"):
            match = re.search(r"iter_(\d+)\.png$", path.name)
            if match and int(match.group(1)) <= iteration:
                previews.append((int(match.group(1)), path))
        if not previews:
            abort(404, "No preview is available yet")
        return max(previews, key=lambda item: item[0])[1]


def build_app(output_root: Path, maximum_points: int) -> Flask:
    app = Flask(__name__, template_folder="live_dashboard/templates")
    cache = CheckpointCache(output_root, maximum_points)

    @app.get("/")
    def index():
        return render_template("index.html", output_root=str(output_root))

    @app.get("/api/runs")
    def runs():
        return jsonify(cache.list_runs())

    @app.get("/api/run/<name>")
    def cloud(name: str):
        return jsonify(cache.load_cloud(name))

    @app.get("/api/run/<name>/preview")
    def preview(name: str):
        return send_file(cache.preview(name), mimetype="image/png", conditional=True)

    @app.get("/health")
    def health():
        return {"status": "ok", "output_root": str(output_root)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--maximum-points", type=int, default=8000)
    args = parser.parse_args()
    build_app(args.output_root, args.maximum_points).run(
        host=args.host, port=args.port, debug=False, threaded=True
    )


if __name__ == "__main__":
    main()
