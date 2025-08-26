
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
    "acf_detectionXact3_control": "Number of detected cases<br>in the control arm", 
    "acf_detectionXact3_trial": "Number of detected cases<br>in the trial arm",
    "act3_trial_adults_pop": "Adult population in the trial arm",
    "act3_control_adults_pop": "Adult population in the trial arm",
    "percentage_latent_adults": "LTBI prevalence among adults (%)",
    "prevalence_infectiousXact3_trial": "Prevalence of TB<br>in the trial arm (/100,000)",
    "prevalence_infectiousXact3_control": "Prevalence of TB<br>in the control arm (/100,000)",
    "prevalence_infectiousXact3_other": "Prevalence of TB<br>in other area (/100,000)",
    "incidenceXact3_trial": "Incidence of TB<br>in the trial arm (/100,000/y)",
    "incidenceXact3_control": "Incidence of TB<br>in the control arm (/100,000/y)",
    "incidenceXact3_other": "Incidence of TB<br>in other area (/100,000/y)",
    "acf_detectionXact3_trialXorgan_pulmonary_rate1": "TB active case finding detection rate<br> in the trial arm (/100,000)",
    "acf_detectionXact3_controlXorgan_pulmonary_rate1": "TB active case finding detection rate<br> in the control arm (/100,000)",
    "adults_incidence_pulmonaryXact3_trial": "Incidence of pulmonary TB<br>among adults in the trial arm (/100,000/y)",
    "adults_incidence_pulmonaryXact3_control": "Incidence of pulmonary TB<br>among adults in the control arm (/100,000/y)",
    "adults_prevalence_pulmonaryXact3_trial": "Prevalence of pulmonary TB<br>among adults in the trial arm (/100,000)",
    "adults_prevalence_pulmonaryXact3_control": "Prevalence of pulmonary TB<br>among adults in the control arm (/100,000)",
    "school_aged_latentXact3_trial": "LTBI prevalence among children aged 5-14<br> in the trial arm (%)",
    "school_aged_latentXact3_control": "LTBI prevalence among children aged 5-14<br> in the control arm (%)",
    "undetected_prevalenceXact3_trial": "Undetected prevalence of TB<br> in the trial arm (/100,000)",
    "undetected_prevalenceXact3_control": "Undetected prevalence of TB<br> in the control arm (/100,000)",
}

indicator_legends = {
    "notification": "Reported to NTP",
    "percentage_latent_adults": "Marks et al. (2018)",
    "school_aged_latentXact3_trial": "Marks et al. (2019)",
    "school_aged_latentXact3_control": "Marks et al. (2019)",
    "total_population": "National census",
}

params_name = {
    "contact_rate": "Contact rate",
    "rr_infection_latent": "Relative risk of infection for individuals with latent infection",
    "rr_infection_recovered": "Relative risk of infection for individuals with history of TB disease",
    "prop_mixing_same_stratum": "Proportion of interactions are confined to the same ACT3 stratum",
    "late_reactivation_adjuster": "Uncertainty multiplier for the rate of late LTBI reactivation",
    "early_prop_adjuster": "Adjuster for the proportion of active TB from early LTBI",
    "early_progression_multiplier": "Uncertainty multiplier for the rates of early reactivation",
    "smear_positive_death_rate": "SPTB death rate",
    "smear_negative_death_rate": "SNTB death rate",
    "smear_positive_self_recovery": "SPTB self-recovery rate",
    "smear_negative_self_recovery": "SNTB TB self-recovery rate",
    "screening_scaleup_shape": "Screening shape",
    "screening_inflection_time": "Screening inflection time",
    "time_to_screening_end_asymp": "Time from active TB to be diagnosed",
    "detection_reduction": "Relative reduction of screening rate during COVID-19",
    "contact_reduction": "Relative reduction of contact rate during COVID-19",
    "duration_positive": "Disease duration of SPTB",
    "prop_mixing_same_stratum":"Proportion of within-stratum mixing",
    "incidence_props_pulmonary": "Proportion of pulmonary TB among all TB incidence",
    "incidence_props_smear_positive_among_pulmonary": "Proportion of SPTB among pulmonary TB incidence"
}

ACT3_STRATA = ["trial", "control", "other"]
