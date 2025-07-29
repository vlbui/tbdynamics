from summer2 import Stratification, Overwrite, Multiply
from summer2.functions.time import get_linear_interpolation_function
from summer2.parameters import Parameter
from tbdynamics.constants import AGE_STRATA, ORGAN_STRATA
from tbdynamics.camau.constants import ACT3_STRATA
from tbdynamics.tools.utils import get_mix_from_strat_props
from tbdynamics.tools.detect import get_interpolation_rates_from_annual
from typing import Dict, List


def get_act3_strat(
    compartments: List[str],
    fixed_params: Dict[str, any],
    scenario_future: Dict[str, bool] = None
) -> Stratification:
    """
    Generates a stratification for the ACT3 trial, defining population groups and their
    mixing behavior, and applying ACF adjustments based on trial participation.

    Args:
        compartments: A list of compartment names in the model.
        fixed_params: Dictionary containing model parameters.
        scenario_future: Dictionary indicating whether future ACF is applied for each arm and strategy.
                         Keys are in the format "<arm>_<frequency>_<coverage>", e.g., "trial_2_80".
    
    Returns:
        Stratification: A configured `Stratification` object representing the ACT3 trial arms.
    """
    
    strat = Stratification("act3", ACT3_STRATA, compartments)

    # Set population split
    props = fixed_params["act3_stratification"]["proportions"]
    assert abs(sum(props.values()) - 1.0) < 1e-6, "Proportions do not sum to 1.0"
    strat.set_population_split(props)
    strat.set_flow_adjustments("birth", {k: Multiply(v) for k, v in props.items()})

    # Set mixing matrix
    prop_same_strat = fixed_params["act3_stratification"]["prop_mixing_same_stratum"]
    props_list = [props[act3_stratum] for act3_stratum in ACT3_STRATA]
    mixing_matrix = get_mix_from_strat_props(prop_same_strat, props_list)
    strat.set_mixing_matrix(mixing_matrix)

    # Historical screening rates
    act_trial_screening_rate = {
        2014.0: 0.0,
        2015.0: 1.51,
        2016.0: 1.15,
        2017.0: 0.97,
        2018.0: 0.80,
        2018.1: 0.00,
    }
    act_control_screening_rate = {
        2017.0: 0.0,
        2018.0: 1.14,
        2018.1: 0.0,
    }

    # Convert to interpolation functions
    trial_func = get_linear_interpolation_function(
        list(act_trial_screening_rate.keys()),
        list(act_trial_screening_rate.values())
    )
    control_func = get_linear_interpolation_function(
        list(act_control_screening_rate.keys()),
        list(act_control_screening_rate.values())
    )
    def make_future_acf_scenarios() -> Dict[str, Dict[float, float]]:
        """
        Generate future ACF screening rate scenarios based on frequency and coverage.

        Returns:
            Dict[str, Dict[float, float]]: A dictionary of scenarios, where keys are
            strings like "2_80", and values are dictionaries mapping years to rates.
        """
        scenarios = {}
        frequencies = {
            "2": [2027.0, 2029.0, 2031.0, 2033.0, 2035.0], #every 2 years
            "4": [2027.0, 2031.0, 2035.0] #every 4 years
        }
        coverages = {
            "80": 1.6, #(coverage 80%, rate =-ln(1-0.8)=1.6)
            "50": 0.7 #(coverage 50%, rate =-ln(1-0.8)=0.7)
        }
        for freq_label, years in frequencies.items():
            for cov_label, rate in coverages.items():
                scenario_key = f"{freq_label}_{cov_label}"
                year_rates = {2026.0: 0.0}
                year_rates.update({year: rate for year in years})
                year_rates[2035.1] = 0.0
                scenarios[scenario_key] = year_rates
        return scenarios

    future_acf_scenarios = make_future_acf_scenarios()

    def combine_screen_func(hist_func, future_dict, apply_future):
        if not apply_future or not future_dict:
            return hist_func
        combined = {**{k: hist_func(k) for k in hist_func.x}, **future_dict}
        times = sorted(combined.keys())
        values = [combined[t] for t in times]
        return get_linear_interpolation_function(times, values)

    if scenario_future is None:
        scenario_future = {}

    acf_sens = Parameter("acf_sensitivity")
    act3_adjs = {"other": None}

    for arm in ["trial", "control"]:
        selected_future = None
        for key, apply in scenario_future.items():
            if not apply:
                continue
            if key.startswith(arm):
                _, freq, cov = key.split("_")
                selected_future = future_acf_scenarios.get(f"{freq}_{cov}")
                break
        hist_func = trial_func if arm == "trial" else control_func
        screen_func = combine_screen_func(hist_func, selected_future, selected_future is not None)
        act3_adjs[arm] = Overwrite(acf_sens * screen_func)

    for age_stratum in AGE_STRATA[2:]:
        source = {"age": str(age_stratum)}
        strat.set_flow_adjustments("acf_detection", act3_adjs, source_strata=source)

    return strat