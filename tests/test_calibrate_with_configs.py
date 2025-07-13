from pathlib import Path
from tbdynamics.calibration.runner import calibrate_with_configs

def test_calibrate_with_configs_runs():
    covid_configs = {
        'no_covid': {
            "detection_reduction": False,
            "contact_reduction": False
        },
        'contact_only': {
            "detection_reduction": False,
            "contact_reduction": True
        }
    }

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
        calibrate_with_configs(
            out_path=out_path,
            params=params,
            covid_configs=covid_configs,
            budget=100,
            n_chains=4,
            draws=100,
            tune=50
        )
    except Exception as e:
        assert False, f"calibrate_with_configs() raised an exception: {e}"