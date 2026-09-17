"""Static regression checks for continual-training wiring."""
import ast
from pathlib import Path


def test_warm_start_parser_and_background_reset_present():
    text = Path("run_reconstruction.py").read_text()
    ast.parse(text)
    assert "--warm_start_checkpoint" in text
    assert "scene.background_coefficients = torch.nn.Parameter(coefficients)" in text
    assert "load_scene(checkpoint_path, device)" in text


def test_chain_is_serial_and_uses_current_encounter_only():
    text = Path("prepare_incremental.py").read_text()
    ast.parse(text)
    assert "encounter_count=1" in text
    assert "iters=6000" in text
    assert "checkpoint_interval=100" in text
    assert "warm_start_checkpoint=str(previous)" in text
