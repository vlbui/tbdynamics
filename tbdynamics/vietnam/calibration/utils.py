from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
from estival.sampling import tools as esamp
import arviz as az
import pandas as pd
from typing import List, Dict
from tbdynamics.vietnam.model import build_model
from tbdynamics.tools.inputs import load_params, load_targets, matrix
from tbdynamics.constants import QUANTILES, COMPARTMENTS
from tbdynamics.settings import VN_PATH
from tbdynamics.calibration.utils import (
    load_extracted_idata,
    build_diff_quantile_df,
    convert_ll_to_idata,
    calculate_waic_comparison,
    run_model_for_covid as _run_model_for_covid,
    calculate_covid_cum_results as _calculate_covid_cum_results,
)
import numpy as np


def get_bcm(
    params, covid_effects=None, improved_detection_multiplier=None, remove_prior = None
) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model.
    Parameters:
    - params (dict): A dictionary containing fixed parameters for the model.

    Returns:
    - BayesianCompartmentalModel: An instance of the BayesianCompartmentalModel class, ready for
      simulation and analysis. This model encapsulates the TB compartmental model, the dynamic
      and fixed parameters, prior distributions for Bayesian inference, and target data for model
      validation or calibration.
    """
    params = params or {}
    fixed_params = load_params(VN_PATH / "params.yml")
    tb_model = build_model(
        fixed_params, matrix, covid_effects, improved_detection_multiplier
    )
    priors = get_all_priors(covid_effects)
    targets = get_targets()
    return BayesianCompartmentalModel(tb_model, params, priors, targets)



def get_all_priors(covid_effects: Dict) -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    priors = [
        esp.UniformPrior("contact_rate", (0.001, 0.05)),
        esp.BetaPrior("rr_infection_latent", 3.0, 8.0),
        esp.BetaPrior("rr_infection_recovered", 3.0, 8.0),
        esp.GammaPrior.from_mode("progression_multiplier", 1.0, 2.0),
        # esp.UniformPrior("incidence_props_smear_positive_among_pulmonary", (0.1, 0.7) ),
        # esp.UniformPrior("incidence_props_pulmonary", (0.5, 0.95)),
        esp.TruncNormalPrior("smear_positive_death_rate", 0.389, 0.0276, (0.335, 0.449)),
        esp.TruncNormalPrior("smear_negative_death_rate", 0.025, 0.0041, (0.017, 0.035)),
        esp.TruncNormalPrior("smear_positive_self_recovery", 0.231, 0.0276, (0.177, 0.288)),
        esp.TruncNormalPrior("smear_negative_self_recovery", 0.130, 0.0291, (0.073, 0.209)),
        esp.UniformPrior("screening_scaleup_shape", (0.05, 0.5)),
        esp.TruncNormalPrior("screening_inflection_time", 2000, 3.5, (1986, 2010)),
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 2.0, 5.0),
        esp.UniformPrior("incidence_props_smear_positive_among_pulmonary", (0.1, 0.7) ),
        esp.UniformPrior("incidence_props_pulmonary", (0.5, 0.95))
    ]
    if covid_effects["contact_reduction"]:
        priors.append(esp.UniformPrior("contact_reduction", (0.01, 0.9)))
    if covid_effects["detection_reduction"]:
        priors.append(esp.UniformPrior("detection_reduction", (0.01, 0.9)))
        # priors.append(esp.BetaPrior("detection_reduction", 10.0, 10.0))
    for prior in priors:
        prior._pymc_transform_eps_scale = 0.1
    return priors


def get_targets() -> List:
    """
    Loads target data for a model and constructs a list of NormalTarget instances.

    This function is designed to load external target data, presumably for the purpose of
    model calibration or validation. It then constructs and returns a list of NormalTarget
    instances, each representing a specific target metric with associated observed values
    and standard deviations. These targets are essential for fitting the model to observed
    data, allowing for the estimation of model parameters that best align with real-world
    observations.

    Returns:
    - list: A list of Target instances.
    """
    target_data = load_targets(VN_PATH / "targets.yml")
    notif_dispersion = esp.TruncNormalPrior("notif_dispersion",0.0,0.1, (0.0, np.inf))
    # prev_dispersion = esp.UniformPrior("prev_dispersion", (20.0, 70.0))
    # sptb_dispersion = esp.UniformPrior("sptb_dispersion", (5.0,30.0))
    # ptb_dispersion = esp.UniformPrior("ptb_dispersion", (1.0,10.0))
    return [
        est.NormalTarget(
            "total_population", target_data["total_population"], stdev=100000.0
        ),
        est.NormalTarget("log_notification", np.log(target_data["notification"]), notif_dispersion),
        # est.NormalTarget(
        #     "adults_prevalence_pulmonary",
        #     target_data["adults_prevalence_pulmonary_target"],
        #     prev_dispersion,
        # ),
        # est.NormalTarget("prevalence_smear_positive", target_data["prevalence_smear_positive_target"], sptb_dispersion),
        # est.NormalTarget("pulmonary_prop", target_data["pulmonary_prop"], ptb_dispersion),
    ]


def calculate_covid_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: float = 2020.0,
    years: List[float] = [2021.0, 2022.0, 2025.0, 2030.0, 2035.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    covid_configs = [
        {"detection_reduction": False, "contact_reduction": False},
        {"detection_reduction": True, "contact_reduction": True},
    ]

    covid_results = []
    for covid_effects in covid_configs:
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time) & (spaghetti_res.index % 1 == 0)
        ]
        covid_results.append({
            "cumulative_diseased": yearly_data["incidence_raw"].cumsum(),
            "cumulative_deaths": yearly_data["mortality_raw"].cumsum(),
            "children_cumulative_diseased": yearly_data["children_incidence_raw"].cumsum(),
        })

    indicators = ["cumulative_diseased", "cumulative_deaths", "children_cumulative_diseased"]
    abs_diff = {ind: covid_results[1][ind] - covid_results[0][ind] for ind in indicators}
    rel_diff = {ind: abs_diff[ind] / covid_results[0][ind] for ind in indicators}

    return {
        "abs": {ind: build_diff_quantile_df(abs_diff[ind], years, QUANTILES) for ind in indicators},
        "rel": {ind: build_diff_quantile_df(rel_diff[ind], years, QUANTILES) for ind in indicators},
    }

def calculate_diff_cum_detection_reduction(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    detection_reduction_values: List[float],
    cumulative_start_time: float = 2020.0,
    year: float = 2035.0,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate absolute and relative differences in cumulative TB incidence and mortality
    by a target year for various detection reduction values, compared to a baseline.

    Args:
        params: Dictionary of model parameters.
        idata_extract: InferenceData object.
        detection_reduction_values: List of detection reduction values to test.
        cumulative_start_time: Year to start cumulative calculations.
        year: Target year to extract cumulative outcomes.

    Returns:
        A dictionary with absolute and relative quantile differences for each scenario.
    """
    # Baseline: no detection or contact reduction
    covid_effects = {"detection_reduction": False, "contact_reduction": False}
    bcm_base = get_bcm(params, covid_effects)
    spaghetti_base = esamp.model_results_for_samples(idata_extract, bcm_base).results
    yearly_base = spaghetti_base.loc[
        (spaghetti_base.index >= cumulative_start_time) & (spaghetti_base.index % 1 == 0)
    ]
    base_cum_diseased = yearly_base["incidence_raw"].cumsum()
    base_cum_deaths = yearly_base["mortality_raw"].cumsum()

    output = {"abs": {}, "rel": {}}

    for val in detection_reduction_values:
        covid_effects = {"detection_reduction": True, "contact_reduction": False}
        scenario_params = {**params, "detection_reduction": val}
        bcm = get_bcm(scenario_params, covid_effects)
        spaghetti = esamp.model_results_for_samples(idata_extract, bcm).results
        yearly = spaghetti.loc[
            (spaghetti.index >= cumulative_start_time) & (spaghetti.index % 1 == 0)
        ]
        cum_diseased = yearly["incidence_raw"].cumsum()
        cum_deaths = yearly["mortality_raw"].cumsum()

        abs_diff_diseased = cum_diseased.loc[year] - base_cum_diseased.loc[year]
        abs_diff_deaths = cum_deaths.loc[year] - base_cum_deaths.loc[year]

        scenario_key = f"detection_reduction_{val}"
        output["abs"][scenario_key] = pd.DataFrame({
            "cumulative_diseased": abs_diff_diseased.quantile(QUANTILES),
            "cumulative_deaths": abs_diff_deaths.quantile(QUANTILES),
        }).T
        output["rel"][scenario_key] = pd.DataFrame({
            "cumulative_diseased": (abs_diff_diseased / base_cum_diseased.loc[year]).quantile(QUANTILES),
            "cumulative_deaths": (abs_diff_deaths / base_cum_deaths.loc[year]).quantile(QUANTILES),
        }).T

    return output


def calculate_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ["incidence", "mortality"],
    detection_multipliers: List[float] = [2.0, 5.0, 12.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate the model results for each scenario with different detection multipliers
    and return the baseline and scenario outputs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to return for the other scenarios (default: ['incidence', 'mortality_raw']).
        detection_multipliers: List of multipliers for improved detection to loop through (default: [2.0, 5.0, 12.0]).

    Returns:
        A dictionary containing results for the baseline and each scenario.
    """

    # Fixed scenario configuration
    scenario_config = {"detection_reduction": True, "contact_reduction": False}

    # Base scenario (calculate outputs for all indicators)
    bcm = get_bcm(params, scenario_config, None)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, QUANTILES)
 
    baseline_indicators = [
        "total_population",
        "notification",
        "adults_prevalence_pulmonary",
        "children_prevalence_pulmonary",
        "children_pulmonary",
        "children_incidence_raw",
        "children_incidence",
        "incidence",
        "case_notification_rate",
        "incidence_early_prop",
        "incidence_late_prop",
        "mortality_raw",
        "prevalence_smear_positive",
        "percentage_latent",
        "detection_rate",
        "mortality",
        *[f"prop_{compartment}" for compartment in COMPARTMENTS],
    ]

    # Filter the baseline results and quantiles
    base_results = base_results[baseline_indicators]
    base_quantiles = base_quantiles[baseline_indicators]
    # Store results for the baseline scenario, including base_results
    scenario_outputs = {
        "base_scenario": {
            "results": base_results,
            "quantiles": base_quantiles,
        }
    }

    # Add no-transmission scenario
    no_transmission_bcm = get_bcm(params, scenario_config, None)
    no_transmission_results = esamp.model_results_for_samples(idata_extract, no_transmission_bcm).results
    no_transmission_quantiles = esamp.quantiles_for_results(no_transmission_results, QUANTILES)

    # Store the results for the no-transmission scenario
    scenario_outputs["no_transmission"] = no_transmission_quantiles


    # Calculate quantiles for each detection multiplier scenario
    for multiplier in detection_multipliers:
        bcm = get_bcm(params, scenario_config, multiplier, False)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)
        scenario_quantiles['mortality'] = scenario_quantiles['mortality'] *0.9

        # Store the results for this scenario
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    # Extract only the relevant indicators for each scenario
    for scenario_key in scenario_outputs:
        if scenario_key != "base_scenario":
            scenario_outputs[scenario_key] = scenario_outputs[scenario_key][indicators]

    return scenario_outputs


def calculate_scenario_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    detection_multipliers: List[float],
    cumulative_start_time: int = 2020,
    years: List[int] = [2021, 2022, 2025, 2030, 2035],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calculate the cumulative incidence and deaths for each scenario with different detection multipliers,
    compute the differences compared to a base scenario, and return quantiles for absolute and relative differences.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        detection_multipliers: List of multipliers for improved detection to loop through.
        cumulative_start_time: Year to start calculating the cumulative values.
        scenario_choice: Integer specifying which scenario to use (1 or 2).
        years: List of years for which to calculate the quantiles.

    Returns:
        A dictionary containing the quantiles for absolute and relative differences between scenarios.
    """

    # Set scenario configuration based on scenario_choice

    covid_config = {"detection_reduction": True, "contact_reduction": False}

    # Base scenario (without improved detection)
    bcm = get_bcm(params, covid_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results

    # Calculate cumulative sums for the base scenario
    yearly_data_base = base_results.loc[
        (base_results.index >= cumulative_start_time) & (base_results.index % 1 == 0)
    ]
    cumulative_diseased_base = yearly_data_base["incidence_raw"].cumsum()
    cumulative_deaths_base = yearly_data_base["mortality_raw"].cumsum()

    detection_diff_results = {}
    indicators = ["cumulative_diseased", "cumulative_deaths"]

    for multiplier in detection_multipliers:
        bcm = get_bcm(params, covid_config, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time) & (scenario_result.index % 1 == 0)
        ]
        abs_diff = {
            "cumulative_diseased": yearly_data["incidence_raw"].cumsum() - cumulative_diseased_base,
            "cumulative_deaths": yearly_data["mortality_raw"].cumsum() - cumulative_deaths_base,
        }
        rel_diff = {ind: abs_diff[ind] / cumulative_diseased_base * 100 for ind in indicators}

        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        detection_diff_results[scenario_key] = {
            "abs": {ind: build_diff_quantile_df(abs_diff[ind], years, QUANTILES) for ind in indicators},
            "rel": {ind: build_diff_quantile_df(rel_diff[ind], years, QUANTILES) for ind in indicators},
        }

    return detection_diff_results


def run_model_for_covid(params, output_dir, covid_configs, quantiles):
    return _run_model_for_covid(params, output_dir, covid_configs, quantiles, get_bcm)


def calculate_covid_cum_results(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: float = 2020.0,
    years: List[float] = [2021.0, 2022.0, 2025.0, 2030.0, 2035.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    return _calculate_covid_cum_results(params, idata_extract, get_bcm, cumulative_start_time, years)
