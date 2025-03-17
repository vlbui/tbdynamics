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
from tbdynamics.camau.constants import ACT3_STRATA
from tbdynamics.tools.detect import get_detection_func


def request_model_outputs(
    model: CompartmentalModel,
    detection_reduction,
):
    """
    Requests various model outputs from the compartmental model for M.tb transmission,
    including prevalence, incidence, mortality, notification, and stratified outputs by organ status,
    age, and ACT3 trial participation.

    This function aggregates and computes key epidemiological indicators using model outputs
    for compartment sizes, flows, and derived functions.

    Args:
        model (CompartmentalModel):
            The compartmental model from which outputs are requested.

        detection_reduction (bool):
            Whether COVID-19 reduced TB case detection. If `True`, modifies the detection rate.

    The function ensures that key epidemiological indicators are computed for further analysis and visualization.
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
    # Request death
    model.request_output_for_flow("mortality_infectious_raw", "infect_death")
    model.request_output_for_flow("mortality_on_treatment_raw", "treatment_death")
    mortality_raw = model.request_aggregate_output(
        "mortality_raw", ["mortality_infectious_raw", "mortality_on_treatment_raw"]
    )
    model.request_cumulative_output(
        "cumulative_deaths",
        "mortality_raw",
        start_time=2014.0,
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

    # Request incidence
    model.request_output_for_flow("incidence_early_raw", "early_activation")
    model.request_output_for_flow("incidence_late_raw", "late_activation")

    incidence_raw = model.request_aggregate_output(
        "incidence_raw",
        ["incidence_early_raw", "incidence_late_raw"],
        save_results=True,
    )
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=2014.0,
    )
    model.request_function_output("incidence", 1e5 * incidence_raw / total_population)

    # Request notification
    model.request_output_for_flow("passive_notification", "detection")
    model.request_output_for_flow("acf_notification", "acf_detection")
    model.request_aggregate_output(
        "notification", ["passive_notification", "acf_notification"]
    )
    for organ_stratum in ORGAN_STRATA:
        model.request_output_for_flow(
            f"passive_notification_{organ_stratum}",
            "detection",
            {"organ": str(organ_stratum)},
        )

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
        model.request_output_for_compartments(
            f"latent_population_sizeXage_{age_stratum}",
            LATENT_COMPARTMENTS,
            strata={"age": str(age_stratum)},
        )
    # Request adults population
    adults_pop = [
        f"total_populationXage_{adults_stratum}" for adults_stratum in AGE_STRATA[2:]
    ]
    adults_pop = model.request_aggregate_output("adults_pop", adults_pop)

    # Request latent among adults
    latent_pop = [
        f"latent_population_sizeXage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
    ]
    latent_pop = model.request_aggregate_output("latent_adults", latent_pop)

    model.request_function_output(
        "percentage_latent_adults", latent_pop / total_population * 100
    )
    # Request prop for each organ stratum
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
            / infectious_population_size,
        )

    # Request adults SPTB
    adults_smear_positive = [
        f"total_infectiousXorgan_smear_positiveXage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
    ]
    adults_smear_positive = model.request_aggregate_output(
        "adults_smear_positive", adults_smear_positive
    )
    model.request_function_output(
        "adults_prevalence_smear_positive", 1e5 * adults_smear_positive / adults_pop
    )
    # request adults pulmonary (smear postive + smear neagative)
    adults_pulmonary = [
        f"total_infectiousXorgan_{smear_status}Xage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
        for smear_status in ["smear_positive", "smear_negative"]
    ]
    adults_pulmonary = model.request_aggregate_output(
        "adults_pulmonary", adults_pulmonary
    )
    model.request_function_output(
        "adults_prevalence_pulmonary",
        1e5 * adults_pulmonary / adults_pop,
    )

    # Request outputs for ACT3
    for act3_stratum in ACT3_STRATA:
        # Request flow output for ACT3 stratum
        model.request_output_for_flow(
            f"acf_detectionXact3_{act3_stratum}",
            "acf_detection",
            source_strata={"act3": str(act3_stratum)},
        )
        for organ_stratum in ORGAN_STRATA[:2]:  # Only reuwest SPTB and ANTB
            model.request_output_for_flow(
                f"acf_detectionXact3_{act3_stratum}Xorgan_{organ_stratum}",
                "acf_detection",
                source_strata={
                    "act3": str(act3_stratum),
                    "organ": str(organ_stratum),
                },
            )
        for age_stratum in AGE_STRATA:
            # Request population output for each ACT3 and age stratum combination
            model.request_output_for_compartments(
                f"total_populationXact3_{act3_stratum}Xage_{age_stratum}",
                COMPARTMENTS,
                strata={"act3": str(act3_stratum), "age": str(age_stratum)},
            )
            # Request infectious compartments output for each ACT3, organ, and age stratum combination
            for organ_stratum in ORGAN_STRATA:
                if organ_stratum in ORGAN_STRATA[:2]:
                    model.request_output_for_compartments(
                        f"total_infectiousXact3_{act3_stratum}Xorgan_{organ_stratum}Xage_{age_stratum}",
                        INFECTIOUS_COMPARTMENTS,
                        strata={
                            "act3": str(act3_stratum),
                            "organ": str(organ_stratum),
                            "age": str(age_stratum),
                        },
                    )
                    if age_stratum in AGE_STRATA[2:]:
                        model.request_output_for_flow(
                            f"acf_detectionXact3_{act3_stratum}Xorgan_{organ_stratum}Xage_{age_stratum}",
                            "acf_detection",
                            dest_strata={
                                "act3": str(act3_stratum),
                                "organ": str(organ_stratum),
                                "age": str(age_stratum),
                            },
                        )
        # Request pop for each arm
        act3_total_pop = [
            f"total_populationXact3_{act3_stratum}Xage_{age_stratum}"
            for age_stratum in AGE_STRATA
        ]
        model.request_aggregate_output(
            f"total_populationXact3_{act3_stratum}", act3_total_pop
        )
        act3_adults_pulmonary = [
            f"total_infectiousXact3_{act3_stratum}Xorgan_{smear_status}Xage_{adults_stratum}"
            for adults_stratum in AGE_STRATA[2:]
            for smear_status in ORGAN_STRATA[:2]
        ]

        model.request_aggregate_output(
            f"act3_{act3_stratum}_adults_pulmonary", act3_adults_pulmonary
        )
        act3_adults_pop = [
            f"total_populationXact3_{act3_stratum}Xage_{age_stratum}"
            for age_stratum in AGE_STRATA[2:]
        ]
        model.request_aggregate_output(
            f"act3_{act3_stratum}_adults_pop", act3_adults_pop
        )
        model.request_function_output(
            f"act3_{act3_stratum}_adults_prevalence",
            1e5
            * DerivedOutput(f"act3_{act3_stratum}_adults_pulmonary")
            / DerivedOutput(f"act3_{act3_stratum}_adults_pop"),
        )
        act3_pulmonary = [
            f"acf_detectionXact3_{act3_stratum}Xorgan_{organ_stratum}"
            for organ_stratum in ORGAN_STRATA[:2]
        ]
        model.request_aggregate_output(
            f"acf_detectionXact3_{act3_stratum}Xorgan_pulmonary", act3_pulmonary
        )
        model.request_function_output(
            f"act3_{act3_stratum}_screened",
            DerivedOutput(f"act3_{act3_stratum}_adults_pop") * 0.80,
        )
        model.request_function_output(
            f"acf_detectionXact3_{act3_stratum}Xorgan_pulmonary_prop",
            DerivedOutput(f"acf_detectionXact3_{act3_stratum}Xorgan_pulmonary")
            / (
                DerivedOutput(f"act3_{act3_stratum}_screened")
            ),  # adjust for screened population (about 80% of adult)
        )

    # request screening profile
    detection_func = get_detection_func(detection_reduction)
    model.add_computed_value_func("detection_rate", detection_func)
    model.request_computed_value_output("detection_rate")
