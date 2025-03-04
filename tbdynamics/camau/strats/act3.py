from summer2 import CompartmentalModel, Stratification, Overwrite, Multiply
from typing import Dict
import numpy as np
from summer2.functions.time import (
    get_linear_interpolation_function,
    get_sigmoidal_interpolation_function,
)
from summer2.parameters import Parameter
from tbdynamics.constants import age_strata
from tbdynamics.camau.constants import act3_strata
from tbdynamics.tools.utils import get_mix_from_strat_props


def get_act3_strat(
    compartments: CompartmentalModel,
    fixed_params: Dict,
) -> Stratification:
    """
    Generates a stratification for the ACT3 trial, defining population groups and their
    mixing behavior, and applying ACF adjustments based on trial participation.

    This function:
    - Defines the ACT3 trial stratification (`trial`, `control`, `other`).
    - Sets population proportions for each stratum.
    - Computes and applies a mixing matrix for stratified population interactions.
    - Adjusts ACF detection flow for the trial and control arms using time-dependent
      linear interpolation functions.

    Args:
        compartments (list):
            A list of compartment names in the model.

        fixed_params (dict):
            Dictionary containing model parameters, including:
            - `"act3_stratification"`: A nested dictionary with:
                - `"strata"` (list): Names of the trial arms (`trial`, `control`, `other`).
                - `"proportions"` (dict): Proportion of the population in each stratum.
                - `"prop_mixing_same_stratum"` (float): Proportion of mixing within the same stratum.
                - `"adjustments"` (dict): Dictionary for flow adjustments.

    Returns:
        Stratification:
            A configured `Stratification` object representing the ACT3 trial arms, including:
            - Population proportions.
            - A mixing matrix defining within- and between-stratum interactions.
            - Adjusted detection flows for active case-finding based on time-dependent sensitivity.
    """

    # Extract the requested strata
    # act3_strata = fixed_params["act3_stratification"]["strata"]
    proportions = fixed_params["act3_stratification"]["proportions"]
    prop_mixing_same_stratum = fixed_params["act3_stratification"][
        "prop_mixing_same_stratum"
    ]
    # prop_mixing_same_stratum = Parameter("prop_mixing_same_stratum")
    # Create the stratification object
    strat = Stratification("act3", act3_strata, compartments)
    # Set the population proportions for each stratum
    strat.set_population_split(proportions)
    mixing_matrix = get_mix_from_strat_props(
        prop_mixing_same_stratum, [proportions[stratum] for stratum in act3_strata]
    )

    # Set the mixing matrix in the stratification object
    strat.set_mixing_matrix(mixing_matrix)
    # Apply the adjustments to birth flow
    adjustments = {}
    adjustments["birth"] = proportions
    for flow_name, adjustment in adjustments.items():
        adj = {stratum: Multiply(value) for stratum, value in adjustment.items()}
        strat.set_flow_adjustments(flow_name, adj)

    act_trial_screening_rate = {
        2014.0: 0.0,  # Value for 2014
        2015.0: 1.00,  # Value for 2015
        2016.0: 0.92,  # Value for 2016
        2017.0: 0.88,  # Value for 2017
        2018.0: 0.85,  # Value for 2018
        2018.9: 0.0,  # Value for 2019
    }

    act_control_screening_rate = {
        2017.0: 0.0,  # Value for 2017
        2018.0: 1.00,  # Value for 2018
        2018.1: 0.0,  # Value for 2019
    }

    for age_stratum in age_strata:
        # Initialize adjustment for each age and strata
        act3_adjs = {stratum: 0.0 for stratum in act3_strata}
        if age_stratum in age_strata[2:]:
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
