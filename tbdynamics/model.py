
from summer2 import AgeStratification, Overwrite, Multiply, Stratification
from jax import numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

from summer2.functions.time import get_sigmoidal_interpolation_function, get_linear_interpolation_function
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput, Function, Time

from .inputs import *
from .utils import *
from .outputs import *

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"
DATA_PATH = BASE_PATH / "data"


"""
Model is constructed through sequentially calling the following functions.
Rather than docstrings for each, the text string to be included 
in the documentation is best description of the code's function.
"""

def build_base_model(
    compartments: list,
    infectious_compartments,
    start_date,
    end_date,
    step
) -> tuple:
    model = CompartmentalModel(
        times=(start_date, end_date),
        compartments=compartments,
        infectious_compartments=infectious_compartments,
        timestep=step
    )

    description = f"The base model consists of {len(compartments)} states, " \
        f"representing the following states: {', '.join(compartments)}. " \
        f"Only the {infectious_compartments} compartment contributes to the force of infection. " \
        f"The model is run from {str(start_date)} to {str(end_date)}. "
    
    return model, description

def get_pop_data():
    pop_data = load_pop_data()
    des = f"For demographics estimates of the Camau" 
    return des, pop_data

def set_starting_conditions(
    model,
) -> str:
    start_pop = Parameter("start_population_size") 
    init_pop = {
        "infectious": 1,
        "susceptible": start_pop - 1,
    }

    # Assign to the model
    model.set_initial_population(init_pop)
    return f"The simulation starts with {start_pop} million fully susceptible persons, " \
        "with infectious persons introduced later through strain seeding as described below. "

def add_entry_flow(
    model: CompartmentalModel
) -> str:
    process = "birth"
    birth_rates = load_pop_data()[1]
    destination =  "susceptible"
    crude_birth_rate = get_sigmoidal_interpolation_function(birth_rates.iloc[:,0], birth_rates.iloc[:,1])
    model.add_crude_birth_flow(
        process,
        crude_birth_rate,
        destination,
    )
    return f"The {process} process add newborns to the {destination} compartment of the model"

def add_natural_death_flow(
    model: CompartmentalModel 
) -> str:
    process = "universal_death"
    universal_death_rate = 1.0
    model.add_universal_death_flows("universal_death", death_rate=universal_death_rate)
    return f"The {process} process add universal death to the model"

def add_infection(
    model: CompartmentalModel,
) -> tuple:
    """
    Args:
        model: Working compartmental model

    Returns:
        Description of process added
    """
    process = "infection"
    origin = "susceptible"
    destination = "early_latent"
    model.add_infection_frequency_flow(process, Parameter("contact_rate"), origin, destination)
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "


    process= "infection_from_latent"
    origin = "late_latent"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_latent") ,
        "late_latent",
        "early_latent",
    )
    des2 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    process = "infection_from_recovered"
    origin = "recovered"
    destination = "early_latent"
    model.add_infection_frequency_flow(
        process,
        Parameter("contact_rate") * Parameter("rr_infection_recovered"),
        origin,
        destination,
    )
    des3 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    return des1, des2, des3

def add_latency(
    model: CompartmentalModel,
) -> tuple:
    # add stabilization process 
    stabilisation_rate = 1.0 # later adjusted by age group
    process =  "stabilisation"
    origin = "early_latent"
    destination = "late_latent"
    model.add_transition_flow(
        process,
        stabilisation_rate,
        origin,
        destination,
    )
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "
    # Add the early activattion process 
    early_activation_rate = 1.0 # later adjusted by age group
    process = "early_activation"
    origin = "early_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        early_activation_rate,
        origin,
        destination,
    )
    des2 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    process = "late_activation"
    origin = "late_latent"
    destination = "infectious"
    model.add_transition_flow(
        process,
        Parameter("progression_multiplier"),  # Set to the adjuster, rather than one
        origin,
        destination,
    )
    des3 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    return des1, des2, des3

def add_detection(
    model: CompartmentalModel,
) -> str:
    detection_rate = 1.0 # later adjusted by organ
    process = "detection"
    origin = "infectious"
    destination = "on_treatment"
    model.add_transition_flow(
        process,
        detection_rate,
        origin,
        destination,
    )
    return f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

def add_treatment_related_outcomes(
    model : CompartmentalModel
) -> tuple:
    #Treatment recovery, releapse, death flows.
    treatment_recovery_rate = 1.0 #  later adjusted by organ
    process = "treatment_recovery"
    origin = "on_treatment"
    destination = "recovered"
    model.add_transition_flow(
        process,
        treatment_recovery_rate,
        origin,
        destination,
    )
    des1 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "

    treatment_death_rate = 1.0  #  later adjusted by age
    process = "treatment_death"
    origin = "on_treatment"
    model.add_death_flow(
        process,
        treatment_death_rate,
        "on_treatment",
    )
    des2 = f"The {process} process moves people from the {origin} " \
        f"compartment to the death, " \
        "under the frequency-dependent transmission assumption. "
    
    relapse_rate = 1.0 #  later adjusted by age
    process = "early_activation"
    origin = "on_treatment"
    destination = "infectious"
    model.add_transition_flow(
        "relapse",
        relapse_rate,
        "on_treatment",
        "infectious",
    )
    des3 = f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination} compartment, " \
        "under the frequency-dependent transmission assumption. "
    
    return des1, des2, des3

def add_self_recovery(
        model: CompartmentalModel
) -> str: 
    process = "self_recovery"
    origin = "on_treatment"
    destination = "recovered"
    model.add_transition_flow(
        "self_recovery",
        0.2,
        origin,
        destination,
    )
    return f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination}, " \
        "under the frequency-dependent transmission assumption. "
     
        
def add_infect_death(
    model: CompartmentalModel 
):
    process = "infect_death"
    origin = "infectious"
    model.add_death_flow(
        "infect_death",
        Parameter("infect_death_rate_unstratified"),
        "infectious",
    )
    return f"The {process} process moves people from the {origin}"

def add_acf(
    model: CompartmentalModel,
    fixed_params
) -> str:
            # Universal active case funding is applied
    times = list(fixed_params["time_variant_screening_rate"])
    vals = [
                v * fixed_params["acf_screening_sensitivity"]
                for v in fixed_params["time_variant_screening_rate"].values()
            ]
    print(times)
    print(vals)
    acf_detection_rate = get_sigmoidal_interpolation_function(times, vals)

    process = "acf_detection"
    origin = "infectious"
    destination = "on_treatment"
    model.add_transition_flow(
        process,
        acf_detection_rate,
        origin,
        destination,
    )

    return f"The {process} process moves people from the {origin} " \
        f"compartment to the {destination}, " \
        "under the frequency-dependent transmission assumption. "
     
def add_age_strat(
    compartments,
    infectious,
    age_strata,
    matrix,
    fixed_params
):
    strat = AgeStratification("age", age_strata, compartments)
    strat.set_mixing_matrix(matrix)
    universal_death_funcs, death_adjs = {}, {}
    for age in age_strata:
        universal_death_funcs[age] = get_sigmoidal_interpolation_function(death_rate_years, death_rates_by_age[age])
        death_adjs[str(age)] = Overwrite(universal_death_funcs[age])
    strat.set_flow_adjustments("universal_death", death_adjs)
    # Set age-specific late activation rate
    for flow_name, latency_params in fixed_params['age_stratification'].items():
        #is_activation_flow = flow_name in ["late_activation"]
        #if is_activation_flow:
        adjs = {str(t): Multiply(latency_params[max([k for k in latency_params if k <= t])]) for t in age_strata}
        strat.set_flow_adjustments(flow_name, adjs)

    #inflate for diabetes
        is_activation_flow = flow_name in ["early_activation", "late_activation"]
        if fixed_params['inflate_reactivation_for_diabetes'] and is_activation_flow:
                # Inflate reactivation rate to account for diabetes.
                for age in age_strata:
                    adjs[age] = Function(get_latency_with_diabetes,[Time,
                        fixed_params["prop_diabetes"][age],
                        adjs[str(age)],
                        Parameter("rr_progression_diabetes")]
                    )
    inf_switch_age = fixed_params['age_infectiousness_switch']
    for comp in infectious:
        inf_adjs = {}
        for i, age_low in enumerate(age_strata):
            infectiousness = 1.0 if age_low == age_strata[-1] else get_average_sigmoid(age_low, age_strata[i + 1], inf_switch_age)

            # Infectiousness multiplier for treatment
            if comp == 'on_treatment':
                infectiousness *= Parameter("on_treatment_infect_multiplier")

            inf_adjs[str(age_low)] = Multiply(infectiousness)

        strat.add_infectiousness_adjustments(comp, inf_adjs)

    # Get the time-varying treatment success proportions
    time_variant_tsr = get_sigmoidal_interpolation_function(
            list(fixed_params['time_variant_tsr'].keys()), list(fixed_params['time_variant_tsr'].values())
        )
     

    # Get the treatment outcomes, using the get_treatment_outcomes function above and apply to model
    treatment_recovery_funcs, treatment_death_funcs, treatment_relapse_funcs = {}, {}, {}
    for age in age_strata:
        death_rate = universal_death_funcs[age]
        treatment_outcomes = Function(
            get_treatment_outcomes,
            [
                fixed_params['treatment_duration'],
                fixed_params['prop_death_among_negative_tx_outcome'],
                death_rate,
                time_variant_tsr,
            ],
        )
        treatment_recovery_funcs[str(age)] = Multiply(treatment_outcomes[0])
        treatment_death_funcs[str(age)] = Multiply(treatment_outcomes[1])
        treatment_relapse_funcs[str(age)] = Multiply(treatment_outcomes[2])
    strat.set_flow_adjustments("treatment_recovery", treatment_recovery_funcs)
    strat.set_flow_adjustments("treatment_death", treatment_death_funcs)
    strat.set_flow_adjustments("relapse", treatment_relapse_funcs)

    #Add BCG effect without stratifying for BCG
    bcg_multiplier_dict = {'0': 0.3, '5': 0.3, '15': 0.7375, '35': 1.0, '50': 1.0} # Ragonnet et al. (IJE, 2020)
    bcg_adjs = {}
    for age, multiplier in bcg_multiplier_dict.items():
        if multiplier < 1.0:
            bcg_adjs[str(age)] = Multiply(
                Function(
                bcg_multiplier_func, 
                [
                get_sigmoidal_interpolation_function(
                                                    list(fixed_params['time_variant_bcg_perc'].keys()),
                                                    list(fixed_params['time_variant_bcg_perc'].values()), 
                                                    Time - get_average_age_for_bcg(age, age_strata)
                                                ),
                multiplier
                ])
            )
        else:
            bcg_adjs[str(age)] = None


    strat.set_flow_adjustments("infection", bcg_adjs)
    des = "The age stratification adjusts following process: (1) The universal death by age group. The data was taken from autumn's database." \
         " (2) The early and late activation" \
         "(3) Age infectioness switched at age of 15" \
         "(4) Infectiousness multiplier for treatment" \
         "(5) Treatment outcomes: relapse, recovery and death"

    return strat, des


def add_organ_strat(
        fixed_params: dict,
        infectious_compartments: list, 
    )-> Stratification:
    
    ORGAN_STRATA = [
        "smear_positive",
        "smear_negative",
        "extrapulmonary",
    ]
    strat = Stratification("organ", ORGAN_STRATA, infectious_compartments)
    # Better if do in loop
    # Define infectiousness adjustment by organ status
    inf_adj = {}
    inf_adj["smear_positive"] = Multiply(1)
    inf_adj["smear_negative"] = Multiply(fixed_params["smear_negative_infect_multiplier"])
    inf_adj["extrapulmonary"] = Multiply(fixed_params["extrapulmonary_infect_multiplier"])
    for comp in infectious_compartments:
        strat.add_infectiousness_adjustments(comp, inf_adj)

    #Define different natural history (infection death) by organ status
    infect_death_adjs = {}
    infect_death_adjs["smear_positive"] = Overwrite(Parameter("smear_positive_death_rate"))
    infect_death_adjs["smear_negative"] = Overwrite(Parameter("smear_negative_death_rate"))
    infect_death_adjs["extrapulmonary"] = Overwrite(Parameter("smear_negative_death_rate"))
    strat.set_flow_adjustments("infect_death", infect_death_adjs)

    #Define different natural history (self recovery) by organ status
    self_recovery_adjs = {}
    self_recovery_adjs["smear_positive"] = Overwrite(Parameter("smear_positive_self_recovery"))
    self_recovery_adjs["smear_negative"] = Overwrite(Parameter("smear_negative_self_recovery"))
    self_recovery_adjs["extrapulmonary"] = Overwrite(Parameter("smear_negative_self_recovery"))
    strat.set_flow_adjustments("self_recovery", self_recovery_adjs)

    # Define different detection rates by organ status.
    detection_adjs = {}
    for organ_stratum in ORGAN_STRATA:
        #adj_vals = sensitivity[organ_stratum]
        param_name = f"passive_screening_sensitivity_{organ_stratum}"
        detection_adjs[organ_stratum] = Parameter("cdr_adjustment") * get_sigmoidal_interpolation_function(list(fixed_params["time_variant_tb_screening_rate"].keys()), 
                                                                                list(fixed_params["time_variant_tb_screening_rate"].values())) * fixed_params[param_name]

    detection_adjs = {k: Multiply(v) for k, v in detection_adjs.items()}
    strat.set_flow_adjustments("detection", detection_adjs)        

    # Adjust the progression rates by organ using the requested incidence proportions
    splitting_proportions = {
        "smear_positive": fixed_params['incidence_props_pulmonary']
        * fixed_params['incidence_props_smear_positive_among_pulmonary'],
        "smear_negative": fixed_params['incidence_props_pulmonary']
        * (1.0 - fixed_params['incidence_props_smear_positive_among_pulmonary']),
        "extrapulmonary": 1.0 - fixed_params['incidence_props_pulmonary'],
    }
    for flow_name in ["early_activation", "late_activation"]:
        flow_adjs = {k: Multiply(v) for k, v in splitting_proportions.items()}
        strat.set_flow_adjustments(flow_name, flow_adjs)
    
    des = "The age stratification adjusts following:" \
            "(1) Infectiousness adjustment by organ status" \
            "(2) Different natural history (infection death) by organ status" \
            "(3) Different detection rates by organ status" \
            "(4) The progression rates by organ using the requested incidence proportions"

    return strat, des


def request_output(
        model: CompartmentalModel,
        compartments,
        latent_compartments,
        infectious_compartments,
        implement_acf = True
):
    """
    Get the applicable outputs
    """
    model.request_output_for_compartments(
        "total_population", compartments, save_results=True
    )
    model.request_output_for_compartments(
        "latent_population_size", latent_compartments, save_results=True
    )
    # latency
    model.request_function_output("percentage_latent", 100.0 * DerivedOutput("latent_population_size") / DerivedOutput("total_population"))

    # Prevalence
    model.request_output_for_compartments("infectious_population_size", infectious_compartments, save_results=True)
    model.request_function_output("prevalence_infectious", 1e5 * DerivedOutput("infectious_population_size") / DerivedOutput("total_population"),)

    # Death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death", save_results=True)

    sources = ["mortality_infectious_raw"]
    request_aggregation_output(model, "mortality_raw", sources, save_results=False)
    model.request_cumulative_output("cumulative_deaths", "mortality_raw", start_time=2000)

    # Disease incidence
    request_flow_output(model,"incidence_early_raw", "early_activation", save_results=False)
    request_flow_output(model,"incidence_late_raw", "late_activation", save_results=False)
    sources = ["incidence_early_raw", "incidence_late_raw"]
    request_aggregation_output(model,"incidence_raw", sources, save_results=False)
    model.request_cumulative_output("cumulative_diseased", "incidence_raw", start_time=2000)

    # Normalise incidence so that it is per unit time (year), not per timestep
    request_normalise_flow_output(model, "incidence_early", "incidence_early_raw")
    request_normalise_flow_output(model, "incidence_late", "incidence_late_raw")
    request_normalise_flow_output(model, "incidence_norm", "incidence_raw", save_results=False)
    model.request_function_output(
        "incidence", 1e5 * DerivedOutput("incidence_norm") / DerivedOutput("total_population")
    )
    request_flow_output(model, "passive_notifications_raw", "detection", save_results=False)
    if implement_acf:
        request_flow_output(model,"active_notifications_raw", "acf_detection", save_results=False)
        sources = ["passive_notifications_raw", "active_notifications_raw"]
        # request_aggregation_output(model,"notifications_raw", sources, save_results=False)
    else:
        sources = ["passive_notifications_raw"] 
    request_aggregation_output(model, "notifications_raw", sources, save_results=False)
    # request notifications
    request_normalise_flow_output(model, "notifications", "notifications_raw")



