from summer2 import CompartmentalModel
from summer2.parameters import DerivedOutput, Parameter
from tbdynamics.constants import (
    COMPARTMENTS,
    LATENT_COMPARTMENTS,
    INFECTIOUS_COMPARTMENTS,
    AGE_STRATA,
    ORGAN_STRATA,
)
from tbdynamics.camau.constants import ACT3_STRATA
from tbdynamics.tools.detect import get_detection_func
import numpy as np

time_start = 2014.0


def request_model_outputs(
    model: CompartmentalModel,
    detection_reduction: bool = False,
    implement_act3: bool = True,
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
    early_and_late_latent = model.request_output_for_compartments(
        "early_and_late_latent", ['early_latent', 'late_latent']
    )
    clearance_population_size = model.request_output_for_compartments(
        "clearance_population_size", ['cleared']
    )
    clearance_positive = model.request_function_output("clearance_positive", clearance_population_size  * Parameter("cleared_proportion"))
    latent_population_size = model.request_aggregate_output("latent_population_size", [early_and_late_latent, clearance_positive])
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
        start_time=time_start,
    )
    model.request_function_output(
        "mortality",
        1e5 * mortality_raw / total_population,
    )

    # Request prevalence of pulmonary
    for organ_stratum in ORGAN_STRATA:
        model.request_output_for_compartments(
            f"total_infectiousXorgan_{organ_stratum}",
            INFECTIOUS_COMPARTMENTS,
            strata={"organ": organ_stratum},
            save_results=False,
        )
        model.request_output_for_flow(
            f"passive_notification_{organ_stratum}",
            "detection",
            {"organ": str(organ_stratum)},
        )

    total_pulmonary = [
        f"total_infectiousXorgan_{organ_stratum}" for organ_stratum in ORGAN_STRATA[:2]
    ]
    total_pulmonary = model.request_aggregate_output(
        "pulmonary_pop_size", total_pulmonary
    )
    model.request_function_output(
        "prevalence_pulmonary",
        1e5 * total_pulmonary / total_population,
    )
    # total prevalence
    total_infectious = model.request_output_for_compartments(
        "total_infectious", INFECTIOUS_COMPARTMENTS
    )
    undetected_infectious = model.request_output_for_compartments(
        "undetected_infectious", "infectious"
    )
    model.request_function_output(
        "prevalence",
        1e5 * total_infectious / total_population,
    )
    model.request_function_output(
        "undetected_prevalence", 1e5 * undetected_infectious / total_population
    )

    # Request incidence
    incidence_early_raw = model.request_output_for_flow(
        "incidence_early_raw", "early_activation"
    )
    model.request_output_for_flow("incidence_late_raw", "late_activation")

    incidence_raw = model.request_aggregate_output(
        "incidence_raw",
        ["incidence_early_raw", "incidence_late_raw"],
    )
    incidence_early_perc = model.request_function_output(
        "incidence_early_perc", 100 * incidence_early_raw / incidence_raw
    )
    model.request_function_output("incidence_late_perc", 100.0 - incidence_early_perc)
    model.request_cumulative_output(
        "cumulative_diseased",
        "incidence_raw",
        start_time=time_start,
    )
    model.request_function_output("incidence", 1e5 * incidence_raw / total_population)

    # Request notification
    model.request_output_for_flow("passive_notification", "detection")
    model.request_output_for_flow("acf_notification", "acf_detection")
    notification = model.request_aggregate_output(
        "notification", ["passive_notification", "acf_notification"]
    )
    model.request_function_output("log_notification", np.log(notification))

    # Request proportion of each compartment in the total population
    for compartment in COMPARTMENTS:
        model.request_output_for_compartments(f"number_{compartment}", compartment)
        model.request_function_output(
            f"prop_{compartment}",
            DerivedOutput(f"number_{compartment}") / total_population,
        )

    # Request total population size and size of latent compartments by age stratum
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
        # --- ADDED: cleared-by-age, cleared-positive-by-age, and latent-like-by-age ---
        model.request_output_for_compartments(
            f"cleared_population_sizeXage_{age_stratum}",
            "cleared",
            strata={"age": str(age_stratum)},
        )
        model.request_function_output(
            f"cleared_positiveXage_{age_stratum}",
            DerivedOutput(f"cleared_population_sizeXage_{age_stratum}") * Parameter("cleared_proportion"),
        )
        model.request_aggregate_output(
            f"latent_likeXage_{age_stratum}",
            [f"latent_population_sizeXage_{age_stratum}", f"cleared_positiveXage_{age_stratum}"],
        )
        # ------------------------------------------------------------------------------

        model.request_output_for_flow(
            f"early_activationXage_{age_stratum}",
            "early_activation",
            source_strata={"age": str(age_stratum)},
        )
        model.request_output_for_flow(
            f"late_activationXage_{age_stratum}",
            "late_activation",
            source_strata={"age": str(age_stratum)},
        )
    # Request adult population
    adults_pop = [
        f"total_populationXage_{adults_stratum}" for adults_stratum in AGE_STRATA[2:]
    ]
    adults_pop = model.request_aggregate_output("adults_pop", adults_pop)
    children_pop = total_population - adults_pop
    early_activation_adults = [
        f"early_activationXage_{age_stratum}" for age_stratum in AGE_STRATA[2:]
    ]
    late_reactivation_adults = [
        f"late_activationXage_{age_stratum}" for age_stratum in AGE_STRATA[2:]
    ]
    early_activation_adults = model.request_aggregate_output(
        "early_activation_adults", early_activation_adults
    )
    late_reactivation_adults = model.request_aggregate_output(
        "late_reactivation_adults", late_reactivation_adults
    )
    incidence_adults_raw = model.request_aggregate_output(
        "incidence_adults_raw", [early_activation_adults, late_reactivation_adults]
    )
    model.request_function_output(
        "incidence_adults", incidence_adults_raw / adults_pop * 1e5
    )  # incidence per 100,000 adults

    # Request latent among adults
    latent_pop = [
        f"latent_population_sizeXage_{adults_stratum}"
        for adults_stratum in AGE_STRATA[2:]
    ]
    latent_pop = model.request_aggregate_output("latent_adults", latent_pop)

    model.request_function_output(
        "percentage_latent_adults", latent_pop / total_population * 100.0
    )

    # Request latent among children
    children_latent = [
        f"latent_population_sizeXage_{children_stratum}"
        for children_stratum in AGE_STRATA[:2]
    ]

    children_latent = model.request_aggregate_output("children_latent", children_latent)
    model.request_function_output(
        "percentage_latent_children", children_latent / children_pop * 100.0
    )
    # --- CHANGED: use latent_likeXage_5 to include cleared-positive in school-aged latent ---
    model.request_function_output(
        "school_aged_latent",
        DerivedOutput("latent_likeXage_5")
        / DerivedOutput("total_populationXage_5")
        * 100.0,
    )
    # ---------------------------------------------------------------------------------------

    # Request prop for each organ stratum
    for organ_stratum in ORGAN_STRATA:
        for age_stratum in AGE_STRATA:
            model.request_output_for_compartments(
                f"total_infectiousXorgan_{organ_stratum}Xage_{age_stratum}",
                INFECTIOUS_COMPARTMENTS,
                strata={"organ": str(organ_stratum), "age": str(age_stratum)},
            )
        model.request_function_output(
            f"prop_{organ_stratum}",
            DerivedOutput(f"total_infectiousXorgan_{organ_stratum}")
            / total_infectious,
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
        for smear_status in ORGAN_STRATA[:2]
    ]
    adults_pulmonary = model.request_aggregate_output(
        "adults_pulmonary", adults_pulmonary
    )
    model.request_function_output(
        "adults_prevalence_pulmonary",
        1e5 * adults_pulmonary / adults_pop,
    )

    # Request outputs for ACT3
    if implement_act3:
        for act3_stratum in ACT3_STRATA:
            # Request flow output for ACT3 stratum
            model.request_output_for_flow(
                f"passive_notificationXact3_{act3_stratum}",
                "detection",
                source_strata={"act3": str(act3_stratum)},
            )
            acf_detection = model.request_output_for_flow(
                f"acf_detectionXact3_{act3_stratum}",
                "acf_detection",
                source_strata={"act3": str(act3_stratum)},
            )

            for age_stratum in AGE_STRATA:
                # Request population output for each ACT3 and age stratum combination
                model.request_output_for_compartments(
                    f"total_populationXact3_{act3_stratum}Xage_{age_stratum}",
                    COMPARTMENTS,
                    strata={"act3": str(act3_stratum), "age": str(age_stratum)},
                )
                # Request passive notification:
                model.request_output_for_flow(
                    f"passive_notificationXact3_{act3_stratum}Xage_{age_stratum}",
                    "detection",
                    source_strata={"act3": str(act3_stratum), "age": str(age_stratum)},
                )
                # Request incidence
                model.request_output_for_flow(
                    f"incidence_earlyXact3_{act3_stratum}Xage_{age_stratum}",
                    "early_activation",
                    source_strata={"act3": str(act3_stratum), "age": str(age_stratum)},
                )
                model.request_output_for_flow(
                    f"incidence_lateXact3_{act3_stratum}Xage_{age_stratum}",
                    "late_activation",
                    source_strata={"act3": str(act3_stratum), "age": str(age_stratum)},
                )
                # Request acf
                model.request_output_for_flow(
                    f"acf_detectionXact3_{act3_stratum}Xage_{age_stratum}",
                    "acf_detection",
                    source_strata={"act3": str(act3_stratum), "age": str(age_stratum)},
                )
                # Request infectious compartments output for each ACT3, organ, and age stratum combination
                for organ_stratum in ORGAN_STRATA:
                    model.request_output_for_compartments(
                        f"total_infectiousXact3_{act3_stratum}Xorgan_{organ_stratum}Xage_{age_stratum}",
                        INFECTIOUS_COMPARTMENTS,
                        strata={
                            "act3": str(act3_stratum),
                            "organ": str(organ_stratum),
                            "age": str(age_stratum),
                        },
                    )
                    model.request_output_for_compartments(
                        f"undetected_infectiousXact3_{act3_stratum}Xorgan_{organ_stratum}Xage_{age_stratum}",
                        "infectious",
                        strata={
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
            act3_total_pop = model.request_aggregate_output(
                f"total_populationXact3_{act3_stratum}", act3_total_pop
            )
            # Request mortality rate
            model.request_output_for_flow(
                f"mortality_infectious_rawXact3_{act3_stratum}",
                "infect_death",
                source_strata={"act3": str(act3_stratum)},
            )
            model.request_output_for_flow(
                f"mortality_on_treatment_rawXact3_{act3_stratum}",
                "treatment_death",
                source_strata={"act3": str(act3_stratum)},
            )
            mortality_raw = model.request_aggregate_output(
                f"mortality_rawXact3_{act3_stratum}",
                [
                    f"mortality_infectious_rawXact3_{act3_stratum}",
                    f"mortality_on_treatment_rawXact3_{act3_stratum}",
                ],
            )
            model.request_cumulative_output(
                f"cumulative_deathsXact3_{act3_stratum}",
                f"mortality_rawXact3_{act3_stratum}",
                start_time=time_start,
            )
            model.request_function_output(
                f"mortality_rateXact3_{act3_stratum}",
                1e5 * mortality_raw / act3_total_pop,
            )
            # Request infectious compartments
            model.request_output_for_compartments(
                f"total_infectiousXact3_{act3_stratum}",
                INFECTIOUS_COMPARTMENTS,
                strata={"act3": str(act3_stratum)},
            )
            for organ_stratum in ORGAN_STRATA:
                model.request_output_for_compartments(
                    f"undetected_infectiousXact3_{act3_stratum}Xorgan_{organ_stratum}",
                    "infectious",
                    strata={"act3": str(act3_stratum), "organ": organ_stratum},
                )
            # Request notification
            passive_notification = [
                f"passive_notificationXact3_{act3_stratum}Xage_{age_stratum}"
                for age_stratum in AGE_STRATA[2:]
            ]
            passive_notification = model.request_aggregate_output(
                f"passive_notification_adultsXact3_{act3_stratum}",
                passive_notification,
            )
            # Request prevalence for pulmonary TB
            undetected_infectious = [f"undetected_infectiousXact3_{act3_stratum}Xorgan_{organ_stratum}" for organ_stratum in ORGAN_STRATA]
            model.request_aggregate_output(
                f"undetected_infectiousXact3_{act3_stratum}",
                undetected_infectious,
            )
            model.request_function_output(
                f"undetected_prevalenceXact3_{act3_stratum}",
                1e5
                * DerivedOutput(f"undetected_infectiousXact3_{act3_stratum}")
                / act3_total_pop,
            )
            model.request_function_output(
                f"total_prevalence_infectiousXact3_{act3_stratum}",
                1e5
                * DerivedOutput(f"total_infectiousXact3_{act3_stratum}")
                / act3_total_pop,
            )
            # Request total adults pulmonary
            act3_adults_pulmonary = [
                f"total_infectiousXact3_{act3_stratum}Xorgan_{smear_status}Xage_{adults_stratum}"
                for adults_stratum in AGE_STRATA[2:]
                for smear_status in ORGAN_STRATA[:2]
            ]

            act3_adults_pulmonary = model.request_aggregate_output(
                f"adults_pulmonaryXact3_{act3_stratum}", act3_adults_pulmonary
            )
            act3_adults_pop = [
                f"total_populationXact3_{act3_stratum}Xage_{age_stratum}"
                for age_stratum in AGE_STRATA[2:]
            ]
            act3_adults_pop = model.request_aggregate_output(
                f"adults_popXact3_{act3_stratum}", act3_adults_pop
            )
            # Request prevalence for pulmonary TB among adults
            model.request_function_output(
                f"adults_prevalence_pulmonaryXact3_{act3_stratum}",
                act3_adults_pulmonary / act3_adults_pop * 1e5,
            )

            model.request_function_output(
                f"acf_detectionXact3_{act3_stratum}Xrate1",
                acf_detection / act3_adults_pop * 1e5,
            )  # request detection rate among general adult population per 100,000 population (denominator is total adults population in each patch)

            # Request for incidence for ACT3 stratum
            incidence_early_raw = model.request_output_for_flow(
                f"incidence_early_rawXact3_{act3_stratum}",
                "early_activation",
                source_strata={"act3": str(act3_stratum)},
                dest_strata={"organ": 'smear_positive',"organ":'smear_negative'},
            )
            incidence_late_raw = model.request_output_for_flow(
                f"incidence_late_rawXact3_{act3_stratum}",
                "late_activation",
                source_strata={"act3": str(act3_stratum)},
                dest_strata={"organ": 'smear_positive',"organ":'smear_negative'}
            )
            act3_incidence_raw = model.request_aggregate_output(
                f"incidence_rawXact3_{act3_stratum}",
                [
                    incidence_early_raw,
                    incidence_late_raw,
                ],
            )
            model.request_function_output(
                f"incidence_pulmonaryXact3_{act3_stratum}",
                act3_incidence_raw / act3_adults_pop * 1e5,
            )
            model.request_function_output(
                f"recent_infection_percXact3_{act3_stratum}",
                incidence_early_raw / act3_incidence_raw * 100,
            )
            model.request_cumulative_output(
                f"cumulative_diseasedXact3_{act3_stratum}",
                f"incidence_rawXact3_{act3_stratum}",
                start_time=time_start, 
            )
            # Request for incidence among adults for ACT3 stratum
            adults_incidence_early_raw = [
                f"incidence_earlyXact3_{act3_stratum}Xage_{adults_stratum}"
                for adults_stratum in AGE_STRATA[2:]
            ]
            adults_incidence_late_raw = [
                f"incidence_lateXact3_{act3_stratum}Xage_{adults_stratum}"
                for adults_stratum in AGE_STRATA[2:]
            ]
            adults_incidence_early_raw = model.request_aggregate_output(
                f"incidence_earlyXact3_{act3_stratum}", adults_incidence_early_raw
            )
            adults_incidence_late_raw = model.request_aggregate_output(
                f"incidence_lateXact3_{act3_stratum}", adults_incidence_late_raw
            )
            adults_incidence_raw = model.request_aggregate_output(
                f"adults_incidence_rawXact3_{act3_stratum}",
                [
                    adults_incidence_early_raw,
                    adults_incidence_late_raw,
                ],
            )
            model.request_function_output(
                f"adults_incidence_pulmonaryXact3_{act3_stratum}",
                1e5 * adults_incidence_raw / act3_adults_pop,
            )
            # --- CHANGED: use latent_likeXage_5 in ACT3 school-aged latent for consistency ---
            model.request_function_output(
                f"school_aged_latentXact3_{act3_stratum}",
                DerivedOutput("latent_likeXage_5")
                / DerivedOutput("total_populationXage_5")
                * 100.0,
            )
            # -------------------------------------------------------------------------------

    # request screening profile
    detection_func = get_detection_func(detection_reduction)
    model.add_computed_value_func("detection_rate", detection_func)
    model.request_computed_value_output("detection_rate")
