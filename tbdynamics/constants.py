from pathlib import Path

organ_strata = [
    "smear_positive",
    "smear_negative",
    "extrapulmonary",
]
age_strata = [0, 5, 15, 35, 50, 70]

compartments = [
    "susceptible",
    "early_latent",
    "late_latent",
    "infectious",
    "on_treatment",
    "recovered",
]

latent_compartments = compartments[1:3]
infectious_compartments = compartments[3:5]

bcg_multiplier_dict = {
    "0": 0.3,
    "5": 0.3,
    "15": 0.7375,
    "35": 1.0,
    "50": 1.0,
    "70": 1.0,
}

PLOT_START_DATE = 1800
PLOT_END_DATE = 2035

indicator_names = {
    "total_population": "Total population",
    "notification": "TB notifications",
    "incidence": "TB incidence (per 100,000)",
    "percentage_latent": "Percentage of latent TB infection (%)",
    "prevalence_pulmonary": "Prevalence of pulmonary TB (per 100,000)",
    "case_detection_rate": "Case detection rate",
    "mortality_raw": "TB deaths",
    "adults_prevalence_pulmonary": "Prevalence of adults pulmonary TB (per 100,000)",
    "prevalence_smear_positive": "Prevalence of adults smear positive pulmonary TB (per 100,000)",
    "detection_rate": "Screening profile",
}

indicator_legends = {
        "prevalence_smear_positive": "National prevalence survey",
        "adults_prevalence_pulmonary": "National prevalence survey ",
        "incidence": "WHO's estimates",
        "mortality_raw": "WHO's estimates",
        "notification": "Reported to WHO",
        "percentage_latent": "Ding et al. (2022)",
        "total_population": "National census"
    }

params_name = {
    "contact_rate": "Transmission scaling factor",
    "rr_infection_latent": "Relative risk of infection for individuals with latent infection",
    "rr_infection_recovered": "Relative risk of infection for individuals with history of infection",
    "progression_multiplier": "Uncertainty multiplier for the rates of TB progression",
    "smear_positive_death_rate": "Smear-positive TB death rate",
    "smear_negative_death_rate": "Smear-negative TB death rate",
    "smear_positive_self_recovery": "Smear-positive TB self-recovery rate",
    "smear_negative_self_recovery": "Smear-negative TB self-recovery rate",
    "screening_scaleup_shape": "Screening shape",
    "screening_inflection_time": "Screening inflection time",
    "time_to_screening_end_asymp": "Time from active TB to be diagnosed",
    "detection_reduction": "Relative reduction of screening rate during COVID-19",
    "contact_reduction": "Relative reduction of contact rate during COVID-19",
}

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]


scenario_names = {
    'base_scenario': 'Baseline scenario',
    'increase_case_detection_by_2_0': 'Scenario 1',
    'increase_case_detection_by_5_0': 'Scenario 2',
    'increase_case_detection_by_12_0': 'Scenario 3'
}

covid_configs = {
        'no_covid': {
            "detection_reduction": False,
            "contact_reduction": False
        },  # No reduction
        'detection_and_contact_reduction': {
            "detection_reduction": True,
            "contact_reduction": True
        },  # With detection + contact reduction
        'case_detection_reduction_only': {
            "detection_reduction": True,
            "contact_reduction": False
        },  # No contact reduction
        'contact_reduction_only': {
            "detection_reduction": False,
            "contact_reduction": True
        }  # Only contact reduction
    }

