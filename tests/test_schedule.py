import pytest

from radar_gs.trainer import geometry_gradient_activity, training_schedule


def _arguments():
    return {
        "stage_a_fraction": 0.15,
        "stage_b_fraction": 0.50,
        "stage_c_fraction": 0.80,
        "final_support_temperature": 0.35,
        "initial_visibility_gain": 0.25,
    }


def test_schedule_includes_coarse_and_final_stages():
    arguments = _arguments()
    first = training_schedule(1, 100, arguments)
    last = training_schedule(100, 100, arguments)
    assert first["stage"] == "A"
    assert first["background_trainable"] is False
    assert last["stage"] == "D"
    assert last["temperature"] == arguments["final_support_temperature"]
    assert last["visibility_gain"] == 1.0


def test_geometry_gradient_activity_supports_delay_and_freeze():
    arguments = {
        "means_start_fraction": 0.15,
        "normals_start_fraction": 0.50,
        "scales_start_fraction": 0.0,
        "freeze_scales": True,
    }
    assert geometry_gradient_activity(0.10, arguments) == {
        "means": False, "normals": False, "scales": False
    }
    assert geometry_gradient_activity(0.20, arguments) == {
        "means": True, "normals": False, "scales": False
    }
    assert geometry_gradient_activity(0.50, arguments) == {
        "means": True, "normals": True, "scales": False
    }


def test_geometry_gradient_activity_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        geometry_gradient_activity(0.5, {"means_start_fraction": 1.1})
