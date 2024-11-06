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
    "total_population": "Population size",
    "notification": "Number of TB notifications",
    "incidence": "TB incidence (/100,000/y)",
    "percentage_latent": "LTBI prevalence (%)",
    "prevalence_pulmonary": "Prevalence of pulmonary TB (/100,000/y)",
    "case_detection_rate": "Case detection rate",
    "mortality_raw": "Number of TB deaths",
    "adults_prevalence_pulmonary": "Prevalence of pulmonary TB<br>among adult (/100,000/y)",
    "prevalence_smear_positive": "Prevalence of SPTB<br>among adult (/100,000/y)",
    "detection_rate": "Rate of presentation to care<br>with active TB (/y)",
    "case_notification_rate": "Case detection proportion (%)",
    "incidence_early_prop": "Progression from early latent (%)"
}

indicator_legends = {
    "prevalence_smear_positive": "National prevalence survey",
    "adults_prevalence_pulmonary": "National prevalence survey ",
    "incidence": "WHO's estimates",
    "mortality_raw": "WHO's estimates",
    "notification": "Reported to WHO",
    "percentage_latent": "Ding et al. (2022)",
    "total_population": "National census",
    "case_notification_rate": "Reported to WHO"
}

params_name = {
    "contact_rate": "Contact rate",
    "rr_infection_latent": "Relative risk of infection for individuals with latent infection",
    "rr_infection_recovered": "Relative risk of infection for individuals with history of TB disease",
    "progression_multiplier": "Uncertainty multiplier for the rates of TB progression",
    "smear_positive_death_rate": "SPTB death rate",
    "smear_negative_death_rate": "SNTB TB death rate",
    "smear_positive_self_recovery": "SPTB self-recovery rate",
    "smear_negative_self_recovery": "SNTB TB self-recovery rate",
    "screening_scaleup_shape": "Screening shape",
    "screening_inflection_time": "Screening inflection time",
    "time_to_screening_end_asymp": "Time from active TB to be diagnosed",
    "detection_reduction": "Relative reduction of screening rate during COVID-19",
    "contact_reduction": "Relative reduction of contact rate during COVID-19",
    "duration_positive": "Disease duration of SPTB",
}

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]


scenario_names = {
    "base_scenario": "Baseline scenario",
    "increase_case_detection_by_2_0": "Scenario 1",
    "increase_case_detection_by_5_0": "Scenario 2",
    "increase_case_detection_by_12_0": "Scenario 3",
}

covid_configs = {
    "no_covid": {"detection_reduction": False, "contact_reduction": False},
    "detection": {"detection_reduction": True, "contact_reduction": False},
    "contact": {"detection_reduction": False, "contact_reduction": True},
    "detection_and_contact": {"detection_reduction": True, "contact_reduction": True},
}