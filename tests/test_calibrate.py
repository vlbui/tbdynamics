from pathlib import Path
from tbdynamics.calibration.runner import calibrate

def test_calibrate_runs():
    covid_effects = {'detection_reduction': True, 'contact_reduction': False}
    params = {
        "start_population_size": 30000.0,
        "seed_time": 1805.0,
        "seed_num": 1.0,
        "seed_duration": 1.0,
        "screening_scaleup_shape": 0.3,
        "screening_inflection_time": 1993,
        "acf_sensitivity": 0.90,
        "prop_mixing_same_stratum": 0.5,
    }
    out_path = Path("test_outputs")
    out_path.mkdir(exist_ok=True)

    try:
        calibrate(
            out_path=out_path,
            params=params,
            covid_effects=covid_effects,
            budget=100,
            n_chains=4,
            draws=50,
            tune=25,
            save_results=False
        )
    except Exception as e:
        assert False, f"calibrate() raised an exception: {e}"
