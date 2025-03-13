from jax import numpy as jnp
from typing import List, Dict
from pandas import DataFrame
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2.parameters import Parameter, Function
from summer2 import AgeStratification
from summer2 import Overwrite, Multiply
from tbdynamics.tools.utils import (
    get_average_sigmoid,
    calculate_treatment_outcomes,
    calculate_bcg_adjustment,
    interpolate_age_strata_values,
    calculate_latency_rates,
)
from tbdynamics.constants import (
    COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    AGE_STRATA,
    bcg_multiplier_dict,
)


def get_age_strat(
    death_df: DataFrame,
    fixed_params: Dict[str, any],
    matrix: List[List[float]],
) -> AgeStratification:
    """
    Creates and configures an age stratification for a compartmental model. This includes setting up
    mixing matrices, adjusting death rates and latency rates by age, applying BCG vaccination effects,
    and configuring treatment outcome adjustments based on age.

    Args:
        compartments: A list of the names of all compartments in the model.
        infectious: A list of the names of infectious compartments in the model.
        age_strata: A list of age strata (as integers) for the model.
        death_df: A DataFrame containing death rates by age.
        fixed_params: A dictionary of fixed parameters for the model, which includes
                      keys for age-specific latency adjustments, infectiousness switch ages,
                      and parameters for BCG effects and treatment outcomes.
        matrix: A mixing matrix

    Returns:
        AgeStratification: An object representing the configured age stratification for the model.
    """
    strat = AgeStratification("age", AGE_STRATA, COMPARTMENTS)
    strat.set_mixing_matrix(matrix)

    # Set universal death rates
    universal_death_funcs, death_adjs = {}, {}
    for age in AGE_STRATA:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(
            death_df.index, death_df[age]
        )
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)

    early_sojourn_time = interpolate_age_strata_values(
        fixed_params["early_sojourn_time"]
    )
    # props_early = {0: Parameter("early_prop_0"), 5: Parameter("early_prop_5"), 15: Parameter("early_prop_15")}
    early_activation_rates = interpolate_age_strata_values(
        fixed_params["age_latency"]["early_activation"]
    )
    stabilisation_rates = interpolate_age_strata_values(
        fixed_params["age_latency"]["stabilisation"]
    )
    late_activation_rates = interpolate_age_strata_values(
        fixed_params["age_latency"]["late_activation"]
    )
    flow_adjs = {'early_activation': {}, 'stabilisation': {}, 'late_activation': {}}
    for age in AGE_STRATA:
        age_latency = calculate_latency_rates(
            early_sojourn_time[age],
            early_activation_rates[age],
            stabilisation_rates[age],
            late_activation_rates[age],
            universal_death_funcs[age],
            Parameter("early_prop_adjuster")
        )
        for flow_name, latency_param in age_latency.items():
            adj = (
                Parameter("late_reactivation_multiplier") * latency_param
                if flow_name == "late_activation"
                else latency_param
            )

            flow_adjs[flow_name][str(age)] = Overwrite(adj)

# Set flow adjustments clearly separated by flow name
    for flow_name, adjs in flow_adjs.items():
        strat.set_flow_adjustments(flow_name, adjs)

    # print(age_latency)
    # Set age-specific latency rate
    # age_latency = fixed_params["age_latency"]
    # for flow_name, latency_params in age_latency.items():
    #     adjs = {}
    #     for age in AGE_STRATA:
    #         # Apply the progression mutiplier to activation flow
    #         adj = (
    #             Parameter("late_reactivation_multiplier") *  latency_params[age]
    #             if flow_name == "late_activation"
    #             else latency_params[age]
    #         )
    #         adjs[str(age)] = adj
    #     adjs = {k: Overwrite(v) for k, v in adjs.items()}
    #     strat.set_flow_adjustments(flow_name, adjs)

    # Infectiousness
    inf_switch_age = fixed_params["age_infectiousness_switch"]
    for comp in INFECTIOUS_COMPARTMENTS:
        inf_adjs = {}
        for i, age_low in enumerate(AGE_STRATA):
            if age_low == AGE_STRATA[-1]:
                average_infectiousness = 1.0
            else:
                age_high = AGE_STRATA[i + 1]
                average_infectiousness = get_average_sigmoid(
                    age_low, age_high, inf_switch_age
                )
            # Adjust infectiousness for the "on_treatment" compartment, the on_treatment_infect_multiplier = 0.08 based on the assumption that the individuals remain infectious on the first 2 weeks of treatment
            if comp == "on_treatment":
                average_infectiousness *= fixed_params["on_treatment_infect_multiplier"]
            # Update the adjustments dictionary for the current age group.
            inf_adjs[str(age_low)] = Multiply(average_infectiousness)

        # Apply infectiousness adjustments to the current compartment.
        strat.add_infectiousness_adjustments(comp, inf_adjs)

    # Add BCG effect without stratifying for BCG
    bcg_adjs = {}  # Initialize dictionary to hold BCG adjustments
    for age, multiplier in bcg_multiplier_dict.items():
        bcg_adjs[age] = calculate_bcg_adjustment(
            age,
            multiplier,
            list(fixed_params["time_variant_bcg_perc"].keys()),
            list(fixed_params["time_variant_bcg_perc"].values()),
        )
    strat.set_flow_adjustments("infection_from_susceptible", bcg_adjs)

    # Get the treatment outcomes, using the get_treatment_outcomes function above and apply to model
    # Initialize dictionaries to hold treatment outcome functions by age strata
    time_variant_tsr = get_sigmoidal_interpolation_function(
        list(fixed_params["time_variant_tsr"].keys()),
        list(fixed_params["time_variant_tsr"].values()),
    )
    treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = (
        {},
        {},
        {},
    )
    for age in AGE_STRATA:
        natural_death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            calculate_treatment_outcomes,
            [
                fixed_params["treatment_duration"],
                fixed_params["prop_death_among_negative_tx_outcome"],
                natural_death_rate,
                time_variant_tsr,
            ],
        )
        treatment_recovery_funcs[str(age)] = Multiply(treatment_outcomes[0])
        treatment_death_funcs[str(age)] = Multiply(treatment_outcomes[1])
        treatment_relapse_funcs[str(age)] = Multiply(treatment_outcomes[2])
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_funcs)
    strat.set_flow_adjustments("treatment_death", treatment_death_funcs)
    strat.set_flow_adjustments("relapse", treatment_relapse_funcs)
    return strat
