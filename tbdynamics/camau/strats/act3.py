from summer2 import Stratification, Multiply
import numpy as np
from summer2.functions.time import get_linear_interpolation_function
from summer2.parameters import Parameter
from tbdynamics.constants import age_strata
from tbdynamics.tools.utils import get_mix_from_strat_props


def get_act3_strat(
    compartments,
    fixed_params,
) -> Stratification:

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
    # print(proportions)
    mixing_matrix = get_mix_from_strat_props(prop_mixing_same_stratum, [proportions[stratum] for stratum in act3_strata])
    

    # Set the mixing matrix in the stratification object
    strat.set_mixing_matrix(mixing_matrix)
    adjustments = fixed_params["act3_stratification"]["adjustments"]
    adjustments["birth"] = proportions
    # if "infection" in adjustments:
    #     for stage in ["susceptible", "late_latent", "recovered"]:
    #         flow_name = f"infection_from_{stage}"
    #         if flow_name not in adjustments:
    #             adjustments[flow_name] = adjustments["infection"]
    # Apply the adjustments to flows (e.g., infection and detection)
    # for flow_name, adjustment in adjustments.items():
    #     # Create the adjustment dictionary for each stratum
    #     if flow_name != "infection":
    #         adj = {stratum: Multiply(value) for stratum, value in adjustment.items()}
    #         strat.set_flow_adjustments(flow_name, adj)

    # adjust detection flow for act3 with active case finding, only for trial
    for age_stratum in age_strata:
        # Initialize adjustment for each age and strata
        act3_adjs = {stratum: 0.0 for stratum in act3_strata}

        if age_stratum not in age_strata[:2]:
            # Define intervention parameters
            times = [2014.0, 2015.0, 2016.0, 2017.0, 2018.0, 2018.1]
            vals = [
                        0.0,  # Value for 2014
                        0.6,  # Value for 2015 1.14 0.97
                        0.52,
                        0.43,
                        0.40,  # Value for 2018
                        0.0,  # Value for 2018.2
                    ]

                # Generate interpolation function and set the adjustment
            act3_adjs["trial"] = Parameter("acf_sensitivity") * get_linear_interpolation_function(
                        times, vals
                    )
            times = [2017.0, 2018.0, 2018.1]
            vals = [
                        0.0,  # Value for 2018
                        0.53,  # Value for 208.2
                        0.0
                    ]
            act3_adjs["control"] = Parameter("acf_sensitivity") * get_linear_interpolation_function(
                        times, vals
                    )
            # Set the flow adjustments without needing to loop over interventions
        strat.set_flow_adjustments(
                    "acf_detection",
                    {k: Multiply(v) for k, v in act3_adjs.items()},
                    source_strata={"age": str(age_stratum)},
                )

    # detection_adjs = {
    #     act3_stratum: (
    #         1.0
    #         if act3_stratum == "trial"
    #         else get_sigmoidal_interpolation_function(
    #             [2014.0, 2016.0, 2018.0],
    #             [1.0, Parameter("act3_spill_over_effects"), 1.0],
    #         )
    #     )
    #     for act3_stratum in act3_strata
    # }

    # strat.set_flow_adjustments(
    #     "detection", {stratum: Multiply(adj) for stratum, adj in detection_adjs.items()}
    # )

    return strat
