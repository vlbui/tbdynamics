from typing import Dict, Any
import numpy as np
from summer2 import CompartmentalModel
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2.parameters import Parameter
from tbdynamics.tools.inputs import get_birth_rate, get_death_rate, process_death_rate
from tbdynamics.constants import COMPARTMENTS, INFECTIOUS_COMPARTMENTS, AGE_STRATA
from tbdynamics.camau.outputs import request_model_outputs
from tbdynamics.camau.strats import get_organ_strat, get_act3_strat, get_age_strat
from tbdynamics.tools.detect import get_detection_func
from tbdynamics.tools.model_utils import (
    add_treatment_related_outcomes,
    seed_infectious,
    PLACEHOLDER_PARAM,
)

def build_model(
    fixed_params: Dict[str, Any],
    matrix: np.ndarray,
    covid_effects: Dict[str, bool],
    implement_act3: bool = True,
    clearance_mode: bool = False,
    future_acf_scenarios: Dict[str, Dict[float, float]] = None,
) -> CompartmentalModel:
    """
    Builds the Ca Mau compartmental TB model.

    Args:
        fixed_params: Fixed parameter dictionary (time range, step, etc.).
        matrix: Age-mixing contact matrix.
        covid_effects: COVID-19 effects on detection/contact rates.
        implement_act3: If True, stratifies by ACT3 trial arms (trial/control/other).
        clearance_mode: If True, adds IGRA clearance flow (late_latent → cleared).
        future_acf_scenarios: Optional ACF scenario dicts for post-trial projections.

    Returns:
        A configured CompartmentalModel instance.
    """
    model = CompartmentalModel(
        times=(fixed_params["time_start"], fixed_params["time_end"]),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
        timestep=fixed_params["time_step"],
    )
    birth_rates = get_birth_rate()
    death_rates = get_death_rate()
    death_df = process_death_rate(death_rates, AGE_STRATA, birth_rates.index)
    model.set_initial_population({"susceptible": Parameter("start_population_size")})
    seed_infectious(model, target_compartment="early_latent")
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates.values
    )
    model.add_crude_birth_flow("birth", crude_birth_rate, "susceptible")
    model.add_universal_death_flows("universal_death", PLACEHOLDER_PARAM)
    add_infection_flows(model, covid_effects["contact_reduction"])
    add_latency_flows(model, clearance_mode)
    model.add_transition_flow("self_recovery", PLACEHOLDER_PARAM, "infectious", "recovered")
    model.add_transition_flow("detection", PLACEHOLDER_PARAM, "infectious", "on_treatment")
    add_treatment_related_outcomes(model)
    model.add_death_flow("infect_death", PLACEHOLDER_PARAM, "infectious")
    model.add_transition_flow("acf_detection", 0.0, "infectious", "on_treatment")
    age_strat = get_age_strat(death_df, fixed_params, matrix, clearance_mode)
    model.stratify_with(age_strat)
    detection_func = get_detection_func(covid_effects["detection_reduction"])
    organ_strat = get_organ_strat(fixed_params, detection_func)
    model.stratify_with(organ_strat)
    if implement_act3:
        act3_strat = get_act3_strat(COMPARTMENTS, fixed_params, future_acf_scenarios)
        model.stratify_with(act3_strat)
    request_model_outputs(model, covid_effects["detection_reduction"], implement_act3)
    return model


def add_infection_flows(
    model: CompartmentalModel,
    contact_reduction: bool,
):
    """
    Adds infection flows to the model, transitioning individuals from
    each compartment that can be infected (e.g., susceptible, late latent, recovered)
    to the early latent state.
    Transitions are modified by parameters that adjust the base contact
    rate, which represents the frequency of infection transmission.

    Args:
        model: The compartmental model to which the infection flows are to be added.

    Each flow is defined by a pair (origin, modifier):
        - `origin`: The name of the compartment from which individuals will transition.
        - `modifier`: A parameter name that modifies the base contact rate for the specific flow.
        - If `None`, the contact rate is used without modification.
    """
    infection_flows = [
        ("susceptible", PLACEHOLDER_PARAM),
        ("late_latent", Parameter("rr_infection_latent")),
        ("recovered", Parameter("rr_infection_recovered")),
        ("cleared", Parameter("rr_infection_latent")),
    ]
    contact_vals = {
        2020.0: 1.0,
        2021.0: 1.0 - Parameter("contact_reduction"),
        2022.0: 1.0,
    }
    contact_rate_func = get_sigmoidal_interpolation_function(
        list(contact_vals.keys()),
        list(contact_vals.values()),
        curvature=8.0,
    )
    is_reduce_contact = contact_rate_func if contact_reduction else 1.0
    contact_rate = Parameter("contact_rate") * is_reduce_contact

    for origin, modifier in infection_flows:
        process = f"infection_from_{origin}"
        flow_rate = contact_rate * modifier
        model.add_infection_frequency_flow(process, flow_rate, origin, "early_latent")

def add_latency_flows(model: CompartmentalModel, clearance_mode):
    """
    Adds latency flows to the compartmental model, representing disease progression
    through different latency stages.

    - Stabilisation: Transition from 'early_latent' to 'late_latent' (disease remains latent).
    - Early activation: Transition from 'early_latent' to 'infectious' (rapid progression).
    - Late activation: Transition from 'late_latent' to 'infectious' (delayed progression).
    - Clearance: Transition from 'early_latent' to 'clearance' (immune clearance of infection).

    Args:
        model: The compartmental model to which latency flows are to be added.
    """
    if clearance_mode:
        clearance_rate = Parameter("clearance_rate")
    else:
        clearance_rate = 0.0
    latency_flows = [
        ("stabilisation", PLACEHOLDER_PARAM, "early_latent", "late_latent"),
        ("early_activation", PLACEHOLDER_PARAM, "early_latent", "infectious"),
        ("late_activation", PLACEHOLDER_PARAM, "late_latent", "infectious"),
        ("clearance", clearance_rate, "late_latent", "cleared"),
    ]
    for latency_flow in latency_flows:
        model.add_transition_flow(*latency_flow)

