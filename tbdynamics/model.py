from pathlib import Path
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Function, Time, DerivedOutput
from summer2 import AgeStratification, Stratification, Overwrite, Multiply
from .utils import triangle_wave_func, get_average_sigmoid, tanh_based_scaleup
from .inputs import get_birth_rate, process_death_rate

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


def build_model(
    compartments,
    latent_compartments,
    infectious_compartments,
    age_strata,
    time_start,
    time_end,
    time_step,
    fixed_params,
    matrix,
    add_triangular=True,
):
    model = CompartmentalModel(
        times=(time_start, time_end),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=time_step,
    )
    initialize_model_conditions(model,add_triangular)
    add_entry_flow(model)
    add_natural_death_flow(model)
    add_infection_flow(model)
    add_latency_flow(model)
    add_infect_death_flow(model)
    add_self_recovery_flow(model)
    stratify_model_by_age(
        model, compartments, infectious_compartments, age_strata, fixed_params, matrix
    )
    request_model_outputs(
        model, compartments, latent_compartments, infectious_compartments, age_strata
    )
    return model

def initialize_model_conditions(model, add_triangular):
    # Set the initial population with either 0 or 1 infectious individual(s)
    start_pop = Parameter("start_population_size")
    if add_triangular:
        model.set_initial_population({
            "infectious": 0,
            "susceptible": start_pop - 0,
        })
        seed_infectious(model)
    else:
        model.set_initial_population({
            "infectious": 1,
            "susceptible": start_pop - 1,
        })

def add_entry_flow(model):
    process = "birth"
    birth_rates = get_birth_rate()
    crude_birth_rate = get_sigmoidal_interpolation_function(
        birth_rates.index, birth_rates.values
    )
    model.add_crude_birth_flow(process, crude_birth_rate, "susceptible")


def add_natural_death_flow(model):
    model.add_universal_death_flows(
        "universal_death", death_rate=1.0
    )  # Adjusted later by age stratification


def add_infection_flow(model):
    infection_flows = [
        ("infection", "susceptible", "early_latent", "contact_rate"),
        (
            "infection_from_latent",
            "late_latent",
            "early_latent",
            "contact_rate",
            "rr_infection_latent",
        ),
        (
            "infection_from_recovered",
            "recovered",
            "early_latent",
            "contact_rate",
            "rr_infection_recovered",
        ),
    ]
    for flow in infection_flows:
        process, origin, destination, rate_param = flow[:4]  # Always present
        modifier_param = flow[4] if len(flow) > 4 else None  # Optional parameter
        rate = (
            Parameter(rate_param) * Parameter(modifier_param)
            if modifier_param
            else Parameter(rate_param)
        )
        model.add_infection_frequency_flow(process, rate, origin, destination)


def add_latency_flow(model):
    latency_flows = [
        ("stabilisation", "early_latent", "late_latent", 1),
        ("early_activation", "early_latent", "infectious", 1),
        (
            "late_activation",
            "late_latent",
            "infectious",
            Parameter("progression_multiplier"),
        ),
    ]
    for process, origin, destination, rate in latency_flows:
        model.add_transition_flow(process, rate, origin, destination)


def add_self_recovery_flow(model):
    model.add_transition_flow("self_recovery", 0.2, "infectious", "recovered")


def add_infect_death_flow(model):
    model.add_death_flow("infect_death", 0.2, "infectious")


def stratify_model_by_age(
    model, compartments, infectious_compartments, age_strata, fixed_params, matrix
):
    age_strat = get_age_strat(
        compartments, infectious_compartments, age_strata, fixed_params, matrix
    )
    model.stratify_with(age_strat)


def get_age_strat(compartments, infectious, age_strata, fixed_params, matrix):
    strat = AgeStratification("age", age_strata, compartments)
    strat.set_mixing_matrix(matrix)
    universal_death_funcs, death_adjs = {}, {}
    death_df = process_death_rate(age_strata)
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(
            death_df.index, death_df[age]
        )
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)
    # Set age-specific latency rate
    for flow_name, latency_params in fixed_params["age_latency"].items():
        adjs = {
            str(t): Multiply(latency_params[max([k for k in latency_params if k <= t])])
            for t in age_strata
        }
        strat.set_flow_adjustments(flow_name, adjs)

    inf_switch_age = fixed_params["age_infectiousness_switch"]
    for comp in infectious:
        inf_adjs = {}
        for i, age_low in enumerate(age_strata):
            if comp != "on_treatment":
                infectiousness = (
                    1.0
                    if age_low == age_strata[-1]
                    else get_average_sigmoid(age_low, age_strata[i + 1], inf_switch_age)
                )
            else:
                infectiousness *= 1
            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    return strat


def get_organ_strat(
    fixed_params: dict,
    infectious_compartments: list,
):
    ORGAN_STRATA = [
        "smear_positive",
        "smear_negative",
        "extrapulmonary",
    ]
    strat = Stratification("organ", ORGAN_STRATA, infectious_compartments)

    # Define infectiousness adjustment by organ status
    inf_adj = {
        organ: Multiply(fixed_params.get(f"{organ}_infect_multiplier", 1))
        for organ in ORGAN_STRATA
    }

    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    # Define different natural history (infection death) by organ status
    infect_death_adjs = {
        "smear_positive": Overwrite(Parameter("smear_positive_death_rate")),
        "smear_negative": Overwrite(Parameter("smear_negative_death_rate")),
        "extrapulmonary": Overwrite(Parameter("smear_negative_death_rate")),  
    }
    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    # Define different natural history (self recovery) by organ status
    self_recovery_adjs = {
        organ: Overwrite(Parameter(f"{organ}_self_recovery"))
        for organ in ORGAN_STRATA[:-1]  # Excluding 'extrapulmonary' for custom handling
    }
    # Handle 'extrapulmonary' case if its adjustment is different from the others
    self_recovery_adjs["extrapulmonary"] = Overwrite(Parameter("smear_negative_self_recovery"))
    strat.set_flow_adjustments("self_recovery", self_recovery_adjs)

    # Adjust the progression rates by organ using the requested incidence proportions
    splitting_proportions = {
        "smear_positive": fixed_params["incidence_props_pulmonary"]
        * fixed_params["incidence_props_smear_positive_among_pulmonary"],
        "smear_negative": fixed_params["incidence_props_pulmonary"]
        * (1.0 - fixed_params["incidence_props_smear_positive_among_pulmonary"]),
        "extrapulmonary": 1.0 - fixed_params["incidence_props_pulmonary"],
    }
    print(splitting_proportions)
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)
    return strat


def seed_infectious(model: CompartmentalModel):
    """Seed infectious.
    Args:
        model: The summer epidemiological model
        latent_compartments: The names of the latent compartments
    """
    seed_args = [Time, Parameter("seed_time"), Parameter("seed_duration"), 1]
    voc_seed_func = Function(triangle_wave_func, seed_args)
    model.add_importation_flow(
        "seed_infectious",
        voc_seed_func,
        "infectious",
        split_imports=False,
    )


def request_model_outputs(
    model, compartments, latent_compartments, infectious_compartments, age_strata
):
    model.request_output_for_compartments("total_population", compartments)
    model.request_output_for_compartments("latent_population_size", latent_compartments)
    model.request_function_output(
        "percentage_latent",
        100.0
        * DerivedOutput("latent_population_size")
        / DerivedOutput("total_population"),
    )
    model.request_output_for_compartments(
        "infectious_population_size", infectious_compartments
    )
    model.request_function_output(
        "prevalence_infectious",
        1e5
        * DerivedOutput("infectious_population_size")
        / DerivedOutput("total_population"),
    )
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
