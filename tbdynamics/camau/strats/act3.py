from summer2 import Stratification, Overwrite, Multiply
import numpy as np
from summer2.functions.time import (
    get_linear_interpolation_function,
    get_sigmoidal_interpolation_function,
)
from summer2.parameters import Parameter
from tbdynamics.constants import age_strata
from tbdynamics.tools.utils import get_mix_from_strat_props


def get_act3_strat(
    compartments,
    fixed_params,
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
    act3_strata = fixed_params["act3_stratification"]["strata"]
    proportions = fixed_params["act3_stratification"]["proportions"]
    prop_mixing_same_stratum = fixed_params["act3_stratification"][
        "prop_mixing_same_stratum"
    ]
    # Create the stratification object
    strat = Stratification("act3", act3_strata, compartments)
    # Set the population proportions for each stratum
    strat.set_population_split(proportions)
    mixing_matrix = get_mix_from_strat_props(
        prop_mixing_same_stratum, [proportions[stratum] for stratum in act3_strata]
    )

    # Set the mixing matrix in the stratification object
    strat.set_mixing_matrix(mixing_matrix)
    adjustments = fixed_params["act3_stratification"]["adjustments"]
    adjustments["birth"] = proportions
    # adjust detection flow for act3 with active case finding, only for trial
    for age_stratum in age_strata:
        # Initialize adjustment for each age and strata
        act3_adjs = {stratum: 0.0 for stratum in act3_strata}

        if age_stratum not in age_strata[:2]:
            # Define intervention parameters. Screesning rates were calculated by the formula: screening_rate = -log(1-coverage); coverage = number of persons consented/total population.
            times = [2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2019.0]
            vals = [
                0.0,  # Value for 2014
                0.6,  # Value for 2015 1.14 0.97
                0.52,
                0.43,
                0.40,  # Value for 2018
                0.0,  # Value for 2018.2
            ]

            # Generate interpolation function and set the adjustment
            act3_adjs["trial"] = Parameter(
                "acf_sensitivity"
            ) * get_linear_interpolation_function(times, vals)
            times = [2017.0, 2018.0, 2019.0]
            vals = [0.0, 0.53, 0.0]  # Value for 2018  # Value for 208.2
            act3_adjs["control"] = Parameter(
                "acf_sensitivity"
            ) * get_linear_interpolation_function(times, vals)
            # Set the flow adjustments without needing to loop over interventions
        strat.set_flow_adjustments(
            "acf_detection",
            {k: Overwrite(v) for k, v in act3_adjs.items()},
            source_strata={"age": str(age_stratum)},
        )

    # organ_adjs = {
    #     "smear_positive": Multiply(1.0),
    #     "smear_negative": Multiply(1.0),
    #     "extrapulmonary": Multiply(0.0),
    # }
    # strat.set_flow_adjustments("acf_detection", organ_adjs)

    # detection_adjs = {
    #     act3_stratum: (
    #         1.0
    #         if act3_stratum == "trial"
    #         else get_sigmoidal_interpolation_function(
    #             [2014.0, 2016.0, 2018.0],
    #             [1.0, Parameter("detection_spill_over_effect"), 1.0],
    #         )
    #     )
    #     for act3_stratum in act3_strata
    # }

    # strat.set_flow_adjustments(
    #     "detection", {stratum: Multiply(adj) for stratum, adj in detection_adjs.items()}
    # )

    return strat
