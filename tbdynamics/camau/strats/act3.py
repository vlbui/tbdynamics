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
        compartments (list): A list of compartment names in the model.
        fixed_params (dict): Dictionary containing model parameters.
    Returns:
        Stratification: A configured `Stratification` object representing the ACT3 trial arms.
    """

    # Extract the requested strata
    # act3_strata = fixed_params["act3_stratification"]["strata"]
    proportions = fixed_params["act3_stratification"]["proportions"]
    # prop_mixing_same_stratum = fixed_params["act3_stratification"][
    #     "prop_mixing_same_stratum"
    # ]
    prop_mixing_same_stratum = Parameter("prop_mixing_same_stratum")
    # Create the stratification object
    strat = Stratification("act3", ACT3_STRATA, compartments)
    # Set the population proportions for each stratum
    strat.set_population_split(proportions)
    mixing_matrix = get_mix_from_strat_props(
        prop_mixing_same_stratum, [proportions[act3_stratum] for act3_stratum in ACT3_STRATA]
    )

    # Set the mixing matrix in the stratification object
    strat.set_mixing_matrix(mixing_matrix)
    # Apply the adjustments to birth flow
    strat.set_flow_adjustments("birth", {stratum: Multiply(value) for stratum, value in proportions.items()})

    act_trial_screening_rate = {
        2014.0: 0.0,  # Value for 2014
        2015.0: 0.81,  # Value for 2015
        2016.0: 0.70,  # Value for 2016
        2017.0: 0.66,  # Value for 2017
        2018.0: 0.60,  # Value for 2018
        2018.1: 0.0,  # Value for 2019
    }

    act_control_screening_rate = {
        2017.0: 0.0,  # Value for 2017
        2018.0: 0.86,  # Value for 2018
        2018.1: 0.0,  # Value for 2019
    }

    for age_stratum in AGE_STRATA:
        # Initialize adjustment for each age and strata
        act3_adjs = {stratum: 0.0 for stratum in ACT3_STRATA}
        if age_stratum in AGE_STRATA[2:]:
            # Define intervention parameters. Screesning rates were calculated by the formula: screening_rate = -log(1-coverage); coverage = number of persons consented/total population.
            # Generate interpolation function and set the adjustment
            act3_adjs["trial"] = Parameter(
                "acf_sensitivity"
            ) * get_linear_interpolation_function(
                list(act_trial_screening_rate.keys()),
                list(act_trial_screening_rate.values()),
            )
            act3_adjs["control"] = Parameter(
                "acf_sensitivity"
            ) * get_linear_interpolation_function(
                list(act_control_screening_rate.keys()),
                list(act_control_screening_rate.values()),
            )
            # Set the flow adjustments without needing to loop over interventions
        strat.set_flow_adjustments(
            "acf_detection",
            {k: Overwrite(v) for k, v in act3_adjs.items()},
            source_strata={"age": str(age_stratum)},
        )
    return strat
