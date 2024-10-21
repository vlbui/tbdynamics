from summer2 import CompartmentalModel
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2.parameters import Function, Parameter, Time, DerivedOutput
from tbdynamics.tools.utils import tanh_based_scaleup
from tbdynamics.constants import (
    compartments,
    latent_compartments,
    infectious_compartments,
    age_strata,
    organ_strata,
)


def request_model_outputs(
    model: CompartmentalModel,
    detection_reduction,
):
    """
    Requests various model outputs

    Args:
        model: The compartmental model from which outputs are requested.
        compartments: A list of all compartment names in the model.
        latent_compartments: A list of latent compartment names.
        infectious_compartments: A list of infectious compartment names.
        age_strata: A list of age groups used for stratification.
        organ_strata: A list of organ strata used for stratification.
    """
    # Request total population size
    total_population = model.request_output_for_compartments(
        "total_population", compartments
    )

    # Calculate and request percentage of latent population
    latent_population_size = model.request_output_for_compartments(
        "latent_population_size", latent_compartments
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
        start_time=2014.0,
    )
    model.request_function_output(
        "mortality",
        1e5 * mortality_raw / total_population,
    )

    # Calculate and request prevalence of pulmonary
    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"infectious_sizeXorgan_{organ_stratum}",
            infectious_compartments,
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
        "infectious_population_size", infectious_compartments
    )
    model.request_function_output(
        "prevalence_infectious",
        1e5 * infectious_population_size / total_population,
    )

    # incidence
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
        start_time=2016.0,
    )
    model.request_function_output("incidence", 1e5 * incidence_raw / total_population)

    # notification
    model.request_output_for_flow("passive_notification", "detection")
    model.request_output_for_flow("acf_notification", "acf_detection")
    model.request_aggregate_output(
        "notification", ["passive_notification", "acf_notification"]
    )
    for organ_stratum in organ_strata:
        model.request_output_for_flow(
            f"passive_notification_{organ_stratum}",
            "detection",
            {"organ": str(organ_stratum)},
        )
    # model.request_function_output("extra_notif_prop", extra_notif / notif * 100)
    # case notification rate:
    # model.request_function_output("case_notification_rate", notif / incidence_raw * 100)

    # Request proportion of each compartment in the total population
    for compartment in compartments:
        model.request_output_for_compartments(f"number_{compartment}", compartment)
        model.request_function_output(
            f"prop_{compartment}",
            DerivedOutput(f"number_{compartment}") / total_population,
        )

    # Request total population by age stratum
    for age_stratum in age_strata:
        model.request_output_for_compartments(
            f"total_populationXage_{age_stratum}",
            compartments,
            strata={"age": str(age_stratum)},
        )
        model.request_output_for_compartments(
            f"latent_population_sizeXage_{age_stratum}",
            latent_compartments,
            strata={"age": str(age_stratum)},
        )
    # request adults poppulation
    adults_pop = [
        f"total_populationXage_{adults_stratum}" for adults_stratum in age_strata[2:]
    ]
    latent_pop = [
        f"latent_population_sizeXage_{adults_stratum}"
        for adults_stratum in age_strata[2:]
    ]
    adults_pop = model.request_aggregate_output("adults_pop", adults_pop)
    # request latent among adults
    latent_pop = model.request_aggregate_output("latent_adults", latent_pop)
    model.request_function_output(
        "percentage_latent_adults", latent_pop / adults_pop * 100
    )
    for organ_stratum in organ_strata:
        model.request_output_for_compartments(
            f"total_infectiousXorgan_{organ_stratum}",
            infectious_compartments,
            strata={"organ": str(organ_stratum)},
        )
        for age_stratum in age_strata:
            model.request_output_for_compartments(
                f"total_infectiousXorgan_{organ_stratum}Xage_{age_stratum}",
                infectious_compartments,
                strata={"organ": str(organ_stratum), "age": str(age_stratum)},
            )
        model.request_function_output(
            f"prop_{organ_stratum}",
            DerivedOutput(f"total_infectiousXorgan_{organ_stratum}")
            / infectious_population_size,
        )

    # Request adults smear_positive
    adults_smear_positive = [
        f"total_infectiousXorgan_smear_positiveXage_{adults_stratum}"
        for adults_stratum in age_strata[2:]
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
        for adults_stratum in age_strata[2:]
        for smear_status in ["smear_positive", "smear_negative"]
    ]
    adults_pulmonary = model.request_aggregate_output(
        "adults_pulmonary", adults_pulmonary
    )
    model.request_function_output(
        "adults_prevalence_pulmonary",
        1e5 * adults_pulmonary / adults_pop,
    )

    # request outputs for act3
    for act3_stratum in ["trial", "control", "other"]:
        # Request flow output for ACT3 stratum
        model.request_output_for_flow(
            f"acf_detectionXact3_{act3_stratum}",
            "acf_detection",
            source_strata={"act3": str(act3_stratum)},
        )
        for organ_stratum in organ_strata:
            if organ_stratum in organ_strata[:2]:
                model.request_output_for_flow(
                    f"acf_detectionXact3_{act3_stratum}Xorgan_{organ_stratum}",
                    "acf_detection",
                    source_strata={
                        "act3": str(act3_stratum),
                        "organ": str(organ_stratum),
                    },
                )
        for age_stratum in age_strata:
            # Request population output for each ACT3 and age stratum combination
            model.request_output_for_compartments(
                f"total_populationXact3_{act3_stratum}Xage_{age_stratum}",
                compartments,
                strata={"act3": str(act3_stratum), "age": str(age_stratum)},
            )

            # Request infectious compartments output for each ACT3, organ, and age stratum combination
            for organ_stratum in organ_strata:
                if organ_stratum in organ_strata[:2]:
                    model.request_output_for_compartments(
                        f"total_infectiousXact3_{act3_stratum}Xorgan_{organ_stratum}Xage_{age_stratum}",
                        infectious_compartments,
                        strata={
                            "act3": str(act3_stratum),
                            "organ": str(organ_stratum),
                            "age": str(age_stratum),
                        },
                    )
                    if age_stratum not in [0, 5]:
                        model.request_output_for_flow(
                            f"acf_detectionXact3_{act3_stratum}Xorgan_{organ_stratum}Xage_{age_stratum}",
                            "acf_detection",
                            dest_strata={
                                "act3": str(act3_stratum),
                                "organ": str(organ_stratum),
                                "age": str(age_stratum),
                            },
                        )
        act3_adults_pulmonary = [
            f"total_infectiousXact3_{act3_stratum}Xorgan_{smear_status}Xage_{adults_stratum}"
            for adults_stratum in age_strata[2:]
            for smear_status in organ_strata[:2]
        ]
        act3_adults_pop = [
            f"total_populationXact3_{act3_stratum}Xage_{adults_stratum}"
            for adults_stratum in age_strata[2:]
        ]
        model.request_aggregate_output(
            f"act3_{act3_stratum}_adults_pulmonary", act3_adults_pulmonary
        )
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
            for organ_stratum in organ_strata[:2]
        ]
        model.request_aggregate_output(
            f"acf_detectionXact3_{act3_stratum}Xorgan_pulmonary", act3_pulmonary
        )
    # model.request_output_for_flow("acf_detection", "acf_detection")

    # request screening profile
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
    detection_func *= (
        get_sigmoidal_interpolation_function(
            [2020.0, 2021.0, 2022.0],
            [1.0, 1.0 - Parameter("detection_reduction"), 1.0],
            curvature=8,
        )
        if detection_reduction
        else 1.0
    )

    model.add_computed_value_func("detection_rate", detection_func)
    model.request_computed_value_output("detection_rate")
