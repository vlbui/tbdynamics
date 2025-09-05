ORGAN_STRATA = [
    "smear_positive",
    "smear_negative",
    "extrapulmonary",
]
AGE_STRATA = [0, 5, 15, 35, 50, 70]

COMPARTMENTS = [
    "susceptible",
    "early_latent",
    "late_latent",
    "cleared",
    "infectious",
    "on_treatment",
    "recovered",
]

LATENT_COMPARTMENTS = COMPARTMENTS[1:4]
INFECTIOUS_COMPARTMENTS = COMPARTMENTS[4:6]

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

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]


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
