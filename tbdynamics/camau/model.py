from typing import Dict, Any
import numpy as np
from summer2 import CompartmentalModel
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2.parameters import Parameter, Function, Time
from tbdynamics.tools.utils import triangle_wave_func
from tbdynamics.tools.inputs import get_birth_rate, get_death_rate, process_death_rate
from tbdynamics.constants import COMPARTMENTS, INFECTIOUS_COMPARTMENTS, AGE_STRATA
from tbdynamics.camau.outputs import request_model_outputs
from tbdynamics.camau.strats import get_organ_strat, get_act3_strat, get_age_strat
from tbdynamics.tools.detect import get_detection_func

PLACEHOLDER_PARAM = 1.0

def build_model(
    fixed_params: Dict[str, Any],
    matrix: np.ndarray,
    covid_effects: Dict[str, bool],
    improved_detection_multiplier: float = None,
    implement_act3: bool = True
) -> CompartmentalModel:
    """
    Builds a compartmental model for TB transmission, incorporating infection dynamics,
    treatment, and stratifications for age, organ status, and ACT3 trial arms.

    Args:
        fixed_params: Fixed parameter dictionary (e.g., time range, population size).
        matrix: Age-mixing matrix for contact patterns.
        covid_effects: Effects of COVID-19 on TB detection and transmission.
        improved_detection_multiplier: Multiplier for improved case detection.
        implement_act3: Whether to include ACT3 trial stratification in the model, enabling
                        differentiation by trial arm and incorporation of ACF adjustments.

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
    seed_infectious(model)
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates.values
    )
    model.add_crude_birth_flow("birth", crude_birth_rate, "susceptible")
    model.add_universal_death_flows(
        "universal_death", PLACEHOLDER_PARAM
    )  # Adjust later in age strat
    add_infection_flows(model, covid_effects["contact_reduction"])
    add_latency_flows(model)
    model.add_transition_flow(
        "self_recovery", PLACEHOLDER_PARAM, "infectious", "recovered"
    )  # Adjust later in organ strat
    model.add_transition_flow(
        "detection", PLACEHOLDER_PARAM, "infectious", "on_treatment"
    )
    add_treatment_related_outcomes(model)
    model.add_death_flow(
        "infect_death", PLACEHOLDER_PARAM, "infectious"
    )  # Adjust later organ strat
    model.add_transition_flow("acf_detection", 0.0, "infectious", "on_treatment")  # ** This function is so short, you can probably just change it to plain code here **
    age_strat = get_age_strat(death_df, fixed_params, matrix)
    model.stratify_with(age_strat)
    detection_func = get_detection_func(covid_effects["detection_reduction"], improved_detection_multiplier)
    organ_strat = get_organ_strat(fixed_params, detection_func)
    model.stratify_with(organ_strat)
    if implement_act3:
        act3_strat = get_act3_strat(COMPARTMENTS, fixed_params)
        model.stratify_with(act3_strat)
    request_model_outputs(model, covid_effects["detection_reduction"])
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
        ("susceptible", None),
        ("late_latent", "rr_infection_latent"),
        ("recovered", "rr_infection_recovered"),
    ]
    contact_vals = {
        2020.0: 1.0,
        2021.0: 1.0 - Parameter("contact_reduction"),  # ** Would a better name for this parameter be "covid_reduction"? **
        2022.0: 1.0,
    }
    contact_rate_func = get_sigmoidal_interpolation_function(
        list(contact_vals.keys()),
        list(contact_vals.values()),
        curvature=8,
    )
    is_reduce_contact = contact_rate_func if contact_reduction else 1.0
    contact_rate = Parameter("contact_rate") * is_reduce_contact

    for origin, modifier in infection_flows:
        process = f"infection_from_{origin}"
        modifier = Parameter(modifier) if modifier else PLACEHOLDER_PARAM
        flow_rate = contact_rate * modifier
        model.add_infection_frequency_flow(process, flow_rate, origin, "early_latent")

def add_latency_flows(model: CompartmentalModel):
    """
    Adds latency flows to the compartmental model, representing disease progression
    through different latency stages.

    - Stabilisation: Transition from 'early_latent' to 'late_latent' (disease remains latent).
    - Early activation: Transition from 'early_latent' to 'infectious' (rapid progression).
    - Late activation: Transition from 'late_latent' to 'infectious' (delayed progression).

    Args:
        model: The compartmental model to which latency flows are to be added.
    """
    latency_flows = [
        ("stabilisation", PLACEHOLDER_PARAM, "early_latent", "late_latent"),
        ("early_activation", PLACEHOLDER_PARAM, "early_latent", "infectious"),
        ("late_activation", PLACEHOLDER_PARAM, "late_latent", "infectious"),
    ]
    for latency_flow in latency_flows:
        model.add_transition_flow(*latency_flow)

def add_treatment_related_outcomes(model: CompartmentalModel):
    """
    Adds treatment-related outcome flows to the compartmental model. This includes flows for treatment recovery,
    treatment-related death, and relapse. Initial rates are set as placeholders, with the expectation that
    they may be adjusted later based on specific factors such as organ involved or patient age.

    Args:
        model: The model object to which the treatment flow is to be added.
    """

    treatment_outcomes_flows = [
        ("treatment_recovery", PLACEHOLDER_PARAM, "recovered"),  # Later adjusted by age
        ("relapse", PLACEHOLDER_PARAM, "infectious"),
    ]

    # Add each transition flow defined in treatment_flows
    for flow_name, rate, to_compartment in treatment_outcomes_flows:
        model.add_transition_flow(flow_name, rate, "on_treatment", to_compartment)

    # Define and add treatment death flow separately since it uses a different method
    model.add_death_flow("treatment_death", PLACEHOLDER_PARAM, "on_treatment")


def seed_infectious(model: CompartmentalModel):
    """
    Adds an importation flow to the model to simulate the initial seeding of infectious individuals.
    This is used to introduce the disease into the population at any time of the simulation.

    Args:
        model: The compartmental model to which the infectious seed is to be added.
    """
    seed_func = Function(
        triangle_wave_func,
        [
            Time,
            Parameter("seed_time"),
            Parameter("seed_duration"),
            Parameter("seed_num"),
        ],
    )
    model.add_importation_flow(
        "seed_infectious", seed_func, "infectious", split_imports=True
    )
