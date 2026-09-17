import ast
from pathlib import Path

def test_sampler_and_protection_are_wired():
    text=Path('radar_gs/trainer.py').read_text();ast.parse(text)
    assert 'continual_replay_fraction' in text
    assert 'draw(current_views, current_count)' in text
    assert 'prune_eligible &= ~protected_inherited' in text
    assert 'protected_inherited = protected_inherited[source].clone()' in text

def test_two_serial_chains():
    text=Path('prepare_replay.py').read_text();ast.parse(text)
    assert '"replay25"' in text and '"replay50_protected"' in text
    assert 'range(2, 9)' in text
    assert 'iters=6000' in text and 'checkpoint_interval=100' in text
