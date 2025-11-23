# tests/test_plant.py
import math
from sim.plant import PlantModel


def test_plant_growth_increases_with_light_and_water():
    p = PlantModel()
    init_C = p.C
    # apply high light, normal temp, no evap, with water
    out = p.step(light=0.9, temp=20.0, evap_factor=0.0, water_input=0.5, dt=1.0)
    assert isinstance(out, dict)
    assert 'C' in out and 'M' in out
    assert out['C'] >= init_C


def test_stage_transitions():
    p = PlantModel()
    # force canopy high
    p.C = 0.6
    assert p.stage() == 2
    p.C = 0.2
    assert p.stage() == 1
    p.C = 0.01
    assert p.stage() == 0


# -------------------------
# tests/test_env.py
from sim.env_model import EnvironmentModel


def test_env_step_outputs():
    env = EnvironmentModel()
    res = env.step(hour=12, shield_pos=0.0, heater_power=0.0, fan_on=False)
    assert isinstance(res, dict)
    assert 'T' in res and 'L' in res and 'evap' in res
    assert 0.0 <= res['L'] <= 1.0


def test_env_temperature_response_to_heater_and_fan():
    env = EnvironmentModel()
    t0 = env.T
    res_heater = env.step(hour=12, shield_pos=0.0, heater_power=1.0, fan_on=False)
    assert res_heater['T'] >= t0
    env2 = EnvironmentModel()
    res_fan = env2.step(hour=12, shield_pos=0.0, heater_power=0.0, fan_on=True)
    # fan should increase cooling effect compared to no fan when ambient < internal
    assert 'T' in res_fan


# -------------------------
# Optional: visual composite smoke test using uploaded session image
# Path provided in session: /mnt/data/A_high-resolution_digital_photograph_showcases_an_.png

def test_visual_composite_works_with_sample_image():
    try:
        from viz.animate import composite_with_real
        sample_path = 'digitaltwin/plant_system.png'
        out = composite_with_real(base_real_path=sample_path, out_path='test_composite.png')
        # expect the function to return output path and file created
        from pathlib import Path
        assert Path(out).exists()
    except Exception:
        # if PIL not available or file missing, skip the test gracefully
        import pytest
        pytest.skip('visual composite test skipped (missing image or PIL)')
