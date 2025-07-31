from summer2 import Stratification, Overwrite, Multiply
from summer2.functions.time import get_linear_interpolation_function
from summer2.parameters import Parameter
from tbdynamics.constants import AGE_STRATA
from tbdynamics.camau.constants import ACT3_STRATA
from tbdynamics.tools.utils import get_mix_from_strat_props
from tbdynamics.tools.detect import get_interpolation_rates_from_annual, calculate_screening_rate
from typing import Dict, List, Any
from tbdynamics.tools.inputs import load_targets
from tbdynamics.settings import CM_PATH


def get_act3_strat(
    compartments: List[str],
    fixed_params: Dict[str, Any],
    future_acf_scenarios: Dict[str, List] = None,
) -> Stratification:
    """
    Generates a stratification for the ACT3 trial, defining population groups and their
    mixing behavior, and applying ACF adjustments based on trial participation.

    Args:
        compartments: A list of compartment names in the model.
        fixed_params: Dictionary containing model parameters.

    Returns:
        Stratification: A configured `Stratification` object representing the ACT3 trial arms.
    """
    strat = Stratification("act3", ACT3_STRATA, compartments)

    # Set the population proportions for each stratum
    props = fixed_params["act3_stratification"]["proportions"]
    assert abs(sum(props.values()) - 1.0) < 1e-6, "Proportions do not sum to 1.0"
    strat.set_population_split(props)

    # Apply the same adjustments to the birth flow
    strat.set_flow_adjustments("birth", {k: Multiply(v) for k, v in props.items()})

    # Set the mixing matrix in the stratification object
    prop_same_strat = fixed_params["act3_stratification"]["prop_mixing_same_stratum"]
    props_list = [props[act3_stratum] for act3_stratum in ACT3_STRATA]
    mixing_matrix = get_mix_from_strat_props(prop_same_strat, props_list)
    strat.set_mixing_matrix(mixing_matrix)
    targets = load_targets(CM_PATH / "targets.yml")
    # Incorporate the screening rates
    act_trial_screening_rate = calculate_screening_rate(targets["act3_trial_adults_pop"].to_dict(), targets["act3_trial_sputum_collected"].to_dict())
    act_trial_screening_rate = get_interpolation_rates_from_annual(act_trial_screening_rate)
    act_control_screening_rate = calculate_screening_rate(targets["act3_control_adults_pop"].to_dict(), targets["act3_control_sputum_collected"].to_dict())
    act_control_screening_rate = get_interpolation_rates_from_annual(act_control_screening_rate)

    def combine_screen_func(hist_acf, future_acf):
        # Case 1: Both hist_acf and future_acf are empty
        if not hist_acf and not future_acf:
            return None

        # Case 2: hist_acf is not empty, but future_acf is empty
        if hist_acf and not future_acf:
            combined_acf_times = list(hist_acf.keys())
            combined_acf_values = list(hist_acf.values())
            return get_linear_interpolation_function(
                combined_acf_times, combined_acf_values
            )

        # Case 3: Both hist_acf and future_acf are not empty
        combined_acf = hist_acf.copy()
        if future_acf:
            combined_acf.update(future_acf)

        combined_acf_times = list(combined_acf.keys())
        combined_acf_values = list(combined_acf.values())
        return get_linear_interpolation_function(
            combined_acf_times, combined_acf_values
        )

    acf_sens = Parameter("acf_sensitivity")
    act3_adjs = {"other": None}
    future_rate = {}
    if future_acf_scenarios:
        for _, v in future_acf_scenarios.items():
            future_rate = get_interpolation_rates_from_annual(v)
    act3_adjs["trial"] = Overwrite(
        acf_sens * combine_screen_func(act_trial_screening_rate, future_rate)
    )
    act3_adjs["control"] = Overwrite(
        acf_sens * combine_screen_func(act_control_screening_rate, future_rate)
    )
    # act3_adjs["other"] = Overwrite(acf_sens * combine_screen_func({}, future_rate))
    print(f"ACT3 ACF adjustments: {act3_adjs}")
    for age_stratum in AGE_STRATA[2:]:
        source = {"age": str(age_stratum)}
        strat.set_flow_adjustments("acf_detection", act3_adjs, source_strata=source)

    return strat
