PLOT_START_DATE = 1800
PLOT_END_DATE = 2035

indicator_names = {
    "total_population": "Population size",
    "notification": "Number of TB notifications",
    "incidence": "TB incidence (/100,000/y)",
    "percentage_latent": "LTBI prevalence (%)",
    "prevalence_pulmonary": "Prevalence of pulmonary TB (/100,000)",
    "case_detection_rate": "Case detection rate",
    "mortality_raw": "Number of TB deaths",
    "mortality": "TB mortality (/100,000/y)",
    "adults_prevalence_pulmonary": "Prevalence of pulmonary TB<br>among adult (/100,000)",
    "prevalence_smear_positive": "Prevalence of SPTB<br>among adult (/100,000)",
    "detection_rate": "Rate of presentation to care<br>with active TB (/y)",
    "case_notification_rate": "Case detection proportion (%)",
    "incidence_early_prop": "Proportion (%)",
    "children_incidence_raw": "Proportion (%)",
    "pulmonary_prop": "Pulmonary TB notification proportion (%)",
}

indicator_legends = {
    "prevalence_smear_positive": "National prevalence survey",
    "adults_prevalence_pulmonary": "National prevalence survey ",
    "incidence": "WHO's estimate",
    "mortality_raw": "WHO's estimate",
    "notification": "Reported to WHO",
    "percentage_latent": "Ding et al. (2022)",
    "total_population": "National census",
    "case_notification_rate": "Reported to WHO",
    "pulmonary_prop": "Reported to WHO",
}

params_name = {
    "contact_rate": "Contact rate",
    "rr_infection_latent": "Relative risk of infection for individuals with latent infection",
    "rr_infection_recovered": "Relative risk of infection for individuals with history of TB disease",
    "progression_multiplier": "Uncertainty multiplier for the rate of latent TB progression",
    "early_progression_multiplier": "Uncertainty multiplier for the rate of early reactivation",
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
    "incidence_props_pulmonary": "Proportion of pulmonary TB among incidence",
    "incidence_props_smear_positive_among_pulmonary": "Proportion of SNTP among pulmonary incidence"
}

