from typing import Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter, Function, Time
from summer2.functions.time import (
    get_sigmoidal_interpolation_function,
    get_linear_interpolation_function,
)
from tbdynamics.tools.utils import tanh_based_scaleup
from tbdynamics.constants import (
    infectious_compartments,
    organ_strata,
)


def get_organ_strat(
    fixed_params: Dict[str, any],
    detection_reduction,
    improved_detection_multiplier=None,
) -> Stratification:
    """
    Creates and configures an organ stratification for the model, defining adjustments
    for infectiousness, detection rates, self-recovery, and infection-related death rates
    based on organ involvement. It also modifies progression rates based on requested
    incidence proportions.

    This stratification differentiates tuberculosis (TB) by organ involvement, typically:
    - `smear_positive` (high transmissibility)
    - `smear_negative` (lower transmissibility)
    - `extrapulmonary` (non-transmissible)

    Args:
        fixed_params (Dict[str, any]):
            A dictionary containing fixed parameters for the model, including:
            - `*_infect_multiplier`: Multiplier for infectiousness by organ.
            - `*_self_recovery`: Self-recovery rate by organ.
            - `passive_screening_sensitivity_*`: Sensitivity of passive case detection by organ.
            - `*_death_rate`: Infection-related death rate by organ.
            - `incidence_props_pulmonary`: Proportion of TB cases that are pulmonary.
            - `incidence_props_smear_positive_among_pulmonary`: Proportion of pulmonary cases that are smear-positive.

        detection_reduction (bool):
            Whether COVID-19 reduced TB case detection. If `True`, modifies the detection rate.

        improved_detection_multiplier (float, optional):
            A positive multiplier (>1) representing improved case detection from future interventions.
            If provided, modifies the detection function over time (2025-2035).

    Returns: A `Stratification` object configured with:

    """
    strat = Stratification("organ", organ_strata, infectious_compartments)

    # Define different detection rates by organ status
    detection_adjs = {}
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            1.0 / Parameter("time_to_screening_end_asymp"),
        ],
    )
    # detection_func*= (get_sigmoidal_interpolation_function([2014.0, 2018.0,2018.1], [1.0, Parameter("detection_spill_over_effect"),1.0])) 
    covid_reduction = {
        2020.0: 1.0,
        2021.0: 1.0 - Parameter("detection_reduction"),
        2022.0: 1.0}
    detection_func*= (get_sigmoidal_interpolation_function(list(covid_reduction.keys()), list(covid_reduction.values()), curvature=8) if detection_reduction else 1.0)
    if improved_detection_multiplier is not None:
        assert (
            isinstance(improved_detection_multiplier, float)
            and improved_detection_multiplier > 0
        ), "improved_detection_multiplier must be a positive float."
        detection_func *= get_linear_interpolation_function(
            [2025.0, 2035.0], [1.0, improved_detection_multiplier]
        )

    # Detection, self-recovery and infect death
    inf_adj, detection_adjs, infect_death_adjs, self_recovery_adjustments = (
        {},
        {},
        {},
        {},
    )
    for organ_stratum in organ_strata:
        # Define infectiousness adjustment by organ status
        inf_adj_param = fixed_params[f"{organ_stratum}_infect_multiplier"]
        inf_adj[organ_stratum] = Multiply(inf_adj_param)

        # Define different natural history (self-recovery) by organ status
        param_strat = (
            "smear_negative" if organ_stratum == "extrapulmonary" else organ_stratum
        )
        self_recovery_adjustments[organ_stratum] = Overwrite(
            Parameter(f"{param_strat}_self_recovery")
        )

        # Adjust detection by organ status
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = fixed_params[param_name] * detection_func

        # Calculate infection death adjustment using detection adjustments
        infect_death_adjs[organ_stratum] = Parameter(f"{param_strat}_death_rate")

    # Apply the Multiply function to the detection adjustments
    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    infect_death_adjs = {k: Overwrite(v) for k, v in infect_death_adjs.items()}

    # Set flow and infectiousness adjustments
    strat.set_flow_adjustments("detection", detection_adjs)
    strat.set_flow_adjustments("self_recovery", self_recovery_adjustments)
    strat.set_flow_adjustments("infect_death", infect_death_adjs)
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    splitting_proportions = {
        "smear_positive": fixed_params["incidence_props_pulmonary"]
        * fixed_params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": fixed_params["incidence_props_pulmonary"]
        * (1.0 - fixed_params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - fixed_params["incidence_props_pulmonary"],
    }
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)

    # organ_adjs = {
    #     "smear_positive": Multiply(1.0),
    #     "smear_negative": Multiply(1.0),
    #     "extrapulmonary": Multiply(0.0),
    # }
    # strat.set_flow_adjustments("acf_detection", organ_adjs)
    return strat
