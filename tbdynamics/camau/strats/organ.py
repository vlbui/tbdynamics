from typing import Dict
from summer2 import Stratification
from summer2 import Overwrite, Multiply
from summer2.parameters import Parameter
from tbdynamics.constants import INFECTIOUS_COMPARTMENTS, ORGAN_STRATA


def get_organ_strat(
    fixed_params: Dict[str, any],
    detection_func,
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
        fixed_params (Dict[str, any]): A dictionary containing fixed parameters for the model.

        detection_function: A callable object

    Returns: A `Stratification` object configured with:

    """
    strat = Stratification("organ", ORGAN_STRATA, INFECTIOUS_COMPARTMENTS)

    # Detection, self-recovery and infect death
    inf_adj = {}
    detection_adjs = {}
    infect_death_adjs = {}
    self_recovery_adjustments = {}

    for organ_stratum in ORGAN_STRATA:
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
        # Calculate infection death adjustment using detection adjustments
        death_adj_val = Parameter(f"{param_strat}_death_rate")
        infect_death_adjs[organ_stratum] = Overwrite(death_adj_val)

        # Adjust detection by organ status
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        screen_adj_val = fixed_params[param_name] * detection_func
        detection_adjs[organ_stratum] = Multiply(screen_adj_val)

    # Set flow and infectiousness adjustments
    strat.set_flow_adjustments("detection", detection_adjs)
    strat.set_flow_adjustments("self_recovery", self_recovery_adjustments)
    strat.set_flow_adjustments("infect_death", infect_death_adjs)
    for comp in INFECTIOUS_COMPARTMENTS:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    prop_pul = fixed_params["incidence_props_pulmonary"]
    prop_smearpos_in_pul = fixed_params["incidence_props_smear_positive_among_pulmonary"]
    smear_pos_prop = prop_pul * prop_smearpos_in_pul
    smear_neg_prop = prop_pul * (1.0 - prop_smearpos_in_pul)
    extrapul_prop = 1.0 - prop_pul

    splitting_adjs = {
        "smear_positive": Multiply(smear_pos_prop),
        "smear_negative": Multiply(smear_neg_prop),
        "extrapulmonary": Multiply(extrapul_prop),
    }
    for flow_name in ["early_activation", "late_activation"]:
        strat.set_flow_adjustments(flow_name, splitting_adjs)

    return strat
