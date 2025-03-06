from summer2 import Stratification, Overwrite, Multiply
from typing import Dict, List
from summer2.functions.time import get_linear_interpolation_function
from summer2.parameters import Parameter
from tbdynamics.constants import AGE_STRATA
from tbdynamics.camau.constants import ACT3_STRATA
from tbdynamics.tools.utils import get_mix_from_strat_props


def get_act3_strat(
    compartments: List[str],
    fixed_params: Dict[str, any],
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
    strat.set_population_split(props)

    # Apply the same adjustments to the birth flow
    strat.set_flow_adjustments("birth", {k: Multiply(v) for k, v in props.items()})

    # Set the mixing matrix in the stratification object
    prop_same_strat = Parameter("prop_mixing_same_stratum")
    props_list = [props[act3_stratum] for act3_stratum in ACT3_STRATA]
    mixing_matrix = get_mix_from_strat_props(prop_same_strat, props_list)
    strat.set_mixing_matrix(mixing_matrix)

    # Incorporate the screening rates
    act_trial_screening_rate = {
        2014.0: 0.0,
        2015.0: 0.63,
        2016.0: 0.60,
        2017.0: 0.58,
        2018.0: 0.55,
        2018.1: 0.0,
    }
    act_control_screening_rate = {
        2017.0: 0.0,
        2018.0: 0.6,
        2018.1: 0.0,
    }

    trial_screen_times = list(act_trial_screening_rate.keys())
    trial_screen_rates = list(act_trial_screening_rate.values())
    trial_screen_func = get_linear_interpolation_function(
        trial_screen_times, trial_screen_rates
    )
    control_screen_times = list(act_control_screening_rate.keys())
    control_screen_rates = list(act_control_screening_rate.values())
    control_screen_func = get_linear_interpolation_function(
        control_screen_times, control_screen_rates
    )

    acf_sens = Parameter("acf_sensitivity")
    act3_adjs = {"other": None}
    act3_adjs["trial"] = Overwrite(acf_sens * trial_screen_func)
    act3_adjs["control"] = Overwrite(acf_sens * control_screen_func)

    for age_stratum in AGE_STRATA[2:]:
        source = {"age": str(age_stratum)}
        strat.set_flow_adjustments("acf_detection", act3_adjs, source_strata=source)

    return strat