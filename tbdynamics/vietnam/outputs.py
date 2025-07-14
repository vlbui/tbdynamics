from summer2 import CompartmentalModel
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2.parameters import Function, Parameter, Time, DerivedOutput
from tbdynamics.tools.utils import tanh_based_scaleup
from tbdynamics.constants import (
    COMPARTMENTS,
    LATENT_COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    AGE_STRATA,
    ORGAN_STRATA,
)
from tbdynamics.tools.detect import get_detection_func
import numpy as np


def request_model_outputs(model: CompartmentalModel, detection_reduction: bool):
    """
    Requests various model outputs

    Args:
        model: The compartmental model from which outputs are requested.
    """
    # Request total population size
    total_population = model.request_output_for_compartments(
        "total_population", COMPARTMENTS
    )

    # Calculate and request percentage of latent population
    latent_population_size = model.request_output_for_compartments(
        "latent_population_size", LATENT_COMPARTMENTS
    )
    model.request_function_output(
        "percentage_latent",
        100.0 * latent_population_size / total_population,
    )
    # Death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death")
    model.request_output_for_flow("mortality_on_treatment_raw", "treatment_death")
    mortality_raw = model.request_aggregate_output(
        "mortality_raw", ["mortality_infectious_raw", "mortality_on_treatment_raw"]
    )
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=2016.0,
    )
    model.request_function_output(
        "mortality",
        1e5 * mortality_raw / total_population,
    )

    # Calculate and request prevalence of pulmonary
    for organ_stratum in ORGAN_STRATA:
        model.request_output_for_compartments(
            f"infectious_sizeXorgan_{organ_stratum}",
            INFECTIOUS_COMPARTMENTS,
            strata={"organ": organ_stratum},
            save_results=False,
        )
    pulmonary_outputs = [
        f"infectious_sizeXorgan_{organ_stratum}"
        for organ_stratum in ["smear_positive", "smear_negative"]
    ]
    pulmonary_pop_size = model.request_aggregate_output(
        "pulmonary_pop_size", pulmonary_outputs
    )
    model.request_function_output(
        "prevalence_pulmonary",
        1e5 * pulmonary_pop_size / total_population,
    )
    # total prevalence
    infectious_population_size = model.request_output_for_compartments(
        "infectious_population_size", INFECTIOUS_COMPARTMENTS
    )
    model.request_function_output(
        "prevalence_infectious",
        1e5 * infectious_population_size / total_population,
    )

    # incidence
    incidence_early_raw = model.request_output_for_flow(
        "incidence_early_raw", "early_activation"
    )
    model.request_output_for_flow("incidence_late_raw", "late_activation")

    incidence_raw = model.request_aggregate_output(
        "incidence_raw",
        ["incidence_early_raw", "incidence_late_raw"],
        save_results=True,
    )
    incidence_early_prop = model.request_function_output(
        "incidence_early_prop", incidence_early_raw / incidence_raw * 100
    )
    model.request_function_output("incidence_late_prop", 100 - incidence_early_prop)
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=2016.0,
    )
    model.request_function_output("incidence", 1e5 * incidence_raw / total_population)

    # notification
    notif = model.request_output_for_flow("notification", "detection")
    model.request_function_output("log_notification", np.log(notif))
    model.request_function_output("case_notification_rate", notif / incidence_raw * 100)
    extra_notif = model.request_output_for_flow(
        name="extra_notification",
        flow_name="detection",
        source_strata={"organ": "extrapulmonary"},
    )
    pul_notif = model.request_function_output("pulmonary_notif", notif - extra_notif)
    # extra_prop= model.request_function_output("extra_prop", extra_notif / notification * 100)
    model.request_function_output("pulmonary_prop",  pul_notif / notif * 100)

    # Request proportion of each compartment in the total population
    for compartment in COMPARTMENTS:
        model.request_output_for_compartments(f"number_{compartment}", compartment)
        model.request_function_output(
            f"prop_{compartment}",
            DerivedOutput(f"number_{compartment}") / total_population,
        )

    # Request total population by age stratum
    for age_stratum in AGE_STRATA:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            COMPARTMENTS,
            strata={"age": str(age_stratum)},
        )
        model.request_output_for_flow(
                f"early_activationXage_{age_stratum}_raw",
                "early_activation",
                source_strata={"age": str(age_stratum)},
            )
        model.request_output_for_flow(
                f"late_activationXage_{age_stratum}_raw",
                "late_activation",
                source_strata={"age": str(age_stratum)},
            )
    # request adults poppulation
    adults_pop = [
        f"total_populationXage_{adults_stratum}" for adults_stratum in AGE_STRATA[2:]
    ]
    adults_pop = model.request_aggregate_output("adults_pop", adults_pop)
    children_pop = model.request_function_output("children_pop", total_population - adults_pop)
    for organ_stratum in ORGAN_STRATA:
        model.request_output_for_compartments(
            f"total_infectiousXorgan_{organ_stratum}",
            INFECTIOUS_COMPARTMENTS,
            strata={"organ": str(organ_stratum)},
        )
        for age_stratum in AGE_STRATA:
            model.request_output_for_compartments(
                f"total_infectiousXorgan_{organ_stratum}Xage_{age_stratum}",
                INFECTIOUS_COMPARTMENTS,
                strata={"organ": str(organ_stratum), "age": str(age_stratum)},
            )
            
        model.request_function_output(
            f"prop_{organ_stratum}",
            DerivedOutput(f"total_infectiousXorgan_{organ_stratum}")
            / DerivedOutput("infectious_population_size"),
        )

    # Request adults smear_positive
    adults_smear_positive = [
        f"total_infectiousXorgan_smear_positiveXage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
    ]
    model.request_aggregate_output("adults_smear_positive", adults_smear_positive)
    model.request_function_output(
        "prevalence_smear_positive",
        1e5 * DerivedOutput("adults_smear_positive") / adults_pop,
    )
    # request adults pulmonary (smear postive + smear neagative)
    adults_pulmonary = [
        f"total_infectiousXorgan_{smear_status}Xage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
        for smear_status in ORGAN_STRATA[:2]
    ]
    model.request_aggregate_output("adults_pulmonary", adults_pulmonary)
    model.request_function_output(
        "adults_prevalence_pulmonary",
        1e5 * DerivedOutput("adults_pulmonary") / adults_pop,
    )
    # Request output for children
    children_pulmonary = [
        f"total_infectiousXorgan_{smear_status}Xage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[:2]
        for smear_status in ORGAN_STRATA[:2]
    ]
    model.request_aggregate_output("children_pulmonary", children_pulmonary)
    model.request_function_output(
        "children_prevalence_pulmonary",
        1e5 * DerivedOutput("children_pulmonary") / children_pop
    )
    children_early_activation = [
        f"early_activationXage_{children_stratum}_raw"
        for children_stratum in AGE_STRATA[:2]
    ]
    model.request_aggregate_output("children_early_activation", children_early_activation)
    children_late_activation = [
        f"late_activationXage_{children_stratum}_raw"
        for children_stratum in AGE_STRATA[:2]
    ]
    model.request_aggregate_output("children_late_activation", children_late_activation)
    children_incidence_raw = model.request_aggregate_output("children_incidence_raw", [DerivedOutput("children_early_activation"), DerivedOutput("children_late_activation")])
    model.request_function_output("children_incidence", 1e5 *  children_incidence_raw / children_pop)
    # Request detection rate
    detection_func = get_detection_func(detection_reduction)
    model.add_computed_value_func("detection_rate", detection_func)
    model.request_computed_value_output("detection_rate")
