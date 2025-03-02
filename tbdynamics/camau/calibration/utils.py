from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
import numpy as np
from typing import List, Dict, Optional

from tbdynamics.camau.model import build_model
from tbdynamics.tools.inputs import load_params, load_targets, matrix
from tbdynamics.settings import CM_PATH


def get_bcm(
    params: Dict[str, float],
    covid_effects: Optional[Dict[str, bool]] = None,
    improved_detection_multiplier: Optional[float] = None,
    homo_mixing: bool = False,
) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model (BCM) for tuberculosis (TB) transmission.

    This function:
    - Loads model parameters.
    - Defines mixing matrices based on homogeneous or heterogeneous mixing assumptions.
    - Specifies Bayesian priors for inference.
    - Retrieves calibration targets.
    - Builds the compartmental model and wraps it within a Bayesian framework.

    Args:
        params (Dict[str, float]): Fixed parameters for the model.
        covid_effects (Optional[Dict[str, bool]]): A dictionary specifying COVID-19 effects on TB transmission.
        improved_detection_multiplier (Optional[float]): Multiplier for improved case detection, if applicable.
        homo_mixing (bool): If True, uses a homogeneous mixing matrix; otherwise, uses an age-structured matrix.

    Returns:
        BayesianCompartmentalModel: A Bayesian model object for inference and analysis.
    """
    params = params or {}
    fixed_params = load_params(CM_PATH / "params.yml")
    #create a mixing matrix
    mixing_matrix = np.ones((6, 6)) if homo_mixing else matrix

    priors = get_all_priors(covid_effects)
    priors.insert(0, esp.UniformPrior("contact_rate", (1.0, 50.0) if homo_mixing else (0.001, 0.05)))

    targets = get_targets()
    tb_model = build_model(fixed_params, mixing_matrix, covid_effects, improved_detection_multiplier)

    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors(covid_effects: Optional[Dict[str, bool]]) -> List:
    """
    Defines the set of prior distributions used in Bayesian inference.

    Args:
        covid_effects (Optional[Dict[str, bool]]): A dictionary indicating whether COVID-19 affected
            TB detection and contact rates.

    Returns:
        List[esp.Prior]: A list of prior distributions for model parameters.
    """
    priors = [
        esp.BetaPrior("rr_infection_latent", 3.0, 8.0),
        esp.BetaPrior("rr_infection_recovered", 3.0, 8.0),
        esp.GammaPrior.from_mode("progression_multiplier", 1.0, 2.0),
        # esp.UniformPrior("detection_spill_over_effect", (1.0, 5.0)),
        # esp.UniformPrior("seed_time", (1800.0, 1840.0)),
        # esp.UniformPrior("seed_num", (1.0, 100.0)),
        # esp.UniformPrior("seed_duration", (1.0, 20.0)),
        esp.TruncNormalPrior("smear_positive_death_rate", 0.389, 0.0276, (0.335, 0.449)),
        esp.TruncNormalPrior("smear_negative_death_rate", 0.025, 0.0041, (0.017, 0.035)),
        esp.TruncNormalPrior("smear_positive_self_recovery", 0.231, 0.0276, (0.177, 0.288)),
        esp.TruncNormalPrior("smear_negative_self_recovery", 0.130, 0.0291, (0.073, 0.209)),
        esp.UniformPrior("screening_scaleup_shape", (0.05, 0.5)),
        esp.TruncNormalPrior("screening_inflection_time", 1998, 6.0, (1986, 2010)),
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 2.0, 5.0),
        esp.UniformPrior("acf_sensitivity", (0.7, 0.99)),
        esp.UniformPrior("prop_mixing_same_stratum", (0.1, 0.99)),
        esp.UniformPrior("incidence_props_pulmonary",(0.2, 0.9)),
        esp.UniformPrior("incidence_props_smear_positive_among_pulmonary",(0.2, 0.9))
    ]

    if covid_effects:
        if covid_effects.get("contact_reduction"):
            priors.append(esp.UniformPrior("contact_reduction", (0.01, 0.9)))
        if covid_effects.get("detection_reduction"):
            priors.append(esp.UniformPrior("detection_reduction", (0.01, 0.9)))

    for prior in priors:
        prior._pymc_transform_eps_scale = 0.1  # Stability scaling for PyMC transformations

    return priors


def get_targets() -> List[est.NormalTarget]:
    """
    Loads target data for Bayesian model calibration.

    Returns:
        List[est.NormalTarget]: A list of target distributions used for calibration.
    """
    target_data = load_targets(CM_PATH / "targets.yml")

    return [
        est.NormalTarget("total_population", target_data["total_population"], 1000),
        est.NormalTarget("notification", target_data["notification"], esp.UniformPrior("notif_dispersion", (10.0, 150.0))),
        est.NormalTarget("total_populationXact3_trial", target_data["total_populationXact3_trial"], 500),
        est.NormalTarget("total_populationXact3_control", target_data["total_populationXact3_coltrol"], 500),
        est.NormalTarget("percentage_latent_adults", target_data["percentage_latent_adults_target"], esp.UniformPrior("latent_dispersion", (2.0, 10.0))),
        # est.NormalTarget("passive_notification_smear_positive", target_data["passive_notification_smear_positive"], esp.UniformPrior("passive_notification_smear_positive_dispersion", (10.0, 100.0))),
        # est.NormalTarget("acf_detectionXact3_trialXorgan_pulmonary", target_data["acf_detectionXact3_trialXorgan_pulmonary"], esp.UniformPrior("acf_detectionXact3_trial_dispersion", (10.0, 50.0))),
        # est.NormalTarget("acf_detectionXact3_controlXorgan_pulmonary", target_data["acf_detectionXact3_controlXorgan_pulmonary"], esp.UniformPrior("acf_detectionXact3_control_dispersion", (10.0, 30.0))),
        est.BinomialTarget("acf_detectionXact3_trialXorgan_pulmonary_prop", target_data["acf_detectionXact3_trialXorgan_pulmonary_prop"],target_data["acf_detectionXact3_trialXsample"]),
        est.BinomialTarget("acf_detectionXact3_controlXorgan_pulmonary_prop", target_data["acf_detectionXact3_controlXorgan_pulmonary_prop"],target_data["acf_detectionXact3_controlXsample"])
    ]
