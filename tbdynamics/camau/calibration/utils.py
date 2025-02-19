from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
from estival.sampling import tools as esamp
import arviz as az
import pandas as pd
from typing import List, Dict
from tbdynamics.camau.model import build_model
from tbdynamics.tools.inputs import load_params, load_targets, matrix
from tbdynamics.constants import quantiles, covid_configs
from tbdynamics.settings import CM_PATH
from pathlib import Path
import xarray as xr
import numpy as np

def get_bcm(params, covid_effects = None, improved_detection_multiplier = None, homo_mixing=True) -> BayesianCompartmentalModel:
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
    fixed_params = load_params(CM_PATH / "params.yml")
    matrix_homo = np.ones((6, 6))
    mixing_matrix = matrix_homo if homo_mixing else matrix
    priors = get_all_priors(covid_effects)
    # contact_prior = esp.UniformPrior("contact_rate", (0.06, 300.0) if homo_mixing else (0.001, 0.05))
    priors.insert(0, esp.UniformPrior("contact_rate", (1.0, 50.0) if homo_mixing else (0.001, 0.05)))# Inserts at the first position in the list
    targets = get_targets()
    tb_model = build_model(
        fixed_params,
        mixing_matrix,
        covid_effects,
        improved_detection_multiplier
    )
    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors(covid_effects) -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    priors = [
        # esp.UniformPrior("contact_rate", (0.001, 0.05)),
        # esp.TruncNormalPrior("contact_rate", 0.0255, 0.00817,  (0.001, 0.05)),
        # esp.UniformPrior("start_population_size", (1.0, 300000.0)),
        esp.BetaPrior("rr_infection_latent", 3.0, 8.0), #2508
        esp.BetaPrior("rr_infection_recovered", 3.0, 8.0),
        # esp.UniformPrior("rr_infection_latent", (0.2, 0.5)), 
        # esp.UniformPrior("rr_infection_recovered", (0.2, 1.0)),
        # esp.TruncNormalPrior("rr_infection_latent", 0.35, 0.1, (0.2, 0.5)), #2608
        # esp.TruncNormalPrior("rr_infection_recovered", 0.6, 0.2, (0.2, 1.0)),
        esp.UniformPrior("progression_multiplier", (0.5, 5.0)),
        # esp.UniformPrior("early_progression_multiplier", (0.5, 5.0)),
        esp.UniformPrior("seed_time", (1800.0, 1840.0)),
        esp.UniformPrior("seed_num", (1.0, 100.00)),
        esp.UniformPrior("seed_duration", (1.0, 20.0)),
        esp.TruncNormalPrior(
            "smear_positive_death_rate", 0.389, 0.0276, (0.335, 0.449)
        ),
        esp.TruncNormalPrior(
            "smear_negative_death_rate", 0.025, 0.0041, (0.017, 0.035)
        ),
        esp.TruncNormalPrior(
            "smear_positive_self_recovery", 0.231, 0.0276, (0.177, 0.288)
        ),
        esp.TruncNormalPrior(
            "smear_negative_self_recovery", 0.130, 0.0291, (0.073, 0.209)
        ),
        esp.UniformPrior("screening_scaleup_shape", (0.05, 0.5)),
        esp.TruncNormalPrior("screening_inflection_time", 1998, 6.0, (1986, 2010)),
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 2.0, 5.0),
        esp.UniformPrior("acf_sensitivity", (0.7,0.99)),
        # esp.UniformPrior("act3_spill_over_effects", (1.0, 2.0))
        # esp.UniformPrior("time_to_screening_end_asymp", (0.1, 2.0)),
        # esp.UniformPrior("intervention_multiplier", (1.0, 50.0)),
    ]
    if covid_effects["contact_reduction"]:
        priors.append(esp.UniformPrior("contact_reduction", (0.01, 0.8)))
    if covid_effects["detection_reduction"]:
        priors.append(esp.UniformPrior("detection_reduction", (0.01, 0.8)))
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
    target_data = load_targets(CM_PATH / "targets.yml")
    notif_dispersion = esp.UniformPrior("notif_dispersion", (10.0, 150.0))
    latent_dispersion = esp.UniformPrior("latent_dispersion", (2.0,10.0))
    passive_notification_smear_positive_dispersion = esp.UniformPrior("passive_notification_smear_positive_dispersion", (10.0,50.0))
    # acf_detectionXact3_trail_dispersion = esp.UniformPrior("acf_detectionXact3_trail_dispersion", (10.0,30.0))
    # acf_detectionXact3_control_dispersion = esp.UniformPrior("acf_detectionXact3_control_dispersion", (10.0,30.0))
    return [
        est.NormalTarget(
            "total_population", target_data["total_population"], stdev=1000
        ),
        est.NormalTarget("notification", target_data["notification"], notif_dispersion),
        est.NormalTarget("percentage_latent_adults", target_data["percentage_latent_adults_target"], latent_dispersion),
        est.NormalTarget("passive_notification_smear_positive", target_data["passive_notification_smear_positive"], passive_notification_smear_positive_dispersion),
        # est.NormalTarget("acf_detectionXact3_trialXorgan_pulmonary", target_data["acf_detectionXact3_trialXorgan_pulmonary"], acf_detectionXact3_trail_dispersion),
        # est.NormalTarget("acf_detectionXact3_controlXorgan_pulmonary", target_data["acf_detectionXact3_trialXorgan_pulmonary"], acf_detectionXact3_control_dispersion)
        est.BinomialTarget("acf_detectionXact3_trialXorgan_pulmonary", target_data["acf_detectionXact3_trialXorgan_pulmonary"],target_data["acf_detectionXact3_trialXsample"]),
        est.BinomialTarget("acf_detectionXact3_controlXorgan_pulmonary", target_data["acf_detectionXact3_controlXorgan_pulmonary"],target_data["acf_detectionXact3_controlXsample"])
        
    ]


def calculate_covid_diff_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str],
    years: List[int],
    covid_analysis: int = 1,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the models for the specified scenarios and calculate the absolute and relative differences.
    Store the quantiles in DataFrames with years as the index and quantiles as columns.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to calculate differences for.
        years: List of years for which to calculate the differences.
        covid_analysis: Integer specifying which analysis to run (1 or 2).
            - 1: Compare scenario 1 and scenario 0.
            - 2: Compare scenario 2 and scenario 0.

    Returns:
        A dictionary containing two dictionaries:
        - "abs": Stores DataFrames for absolute differences (keyed by indicator name).
        - "rel": Stores DataFrames for relative differences (keyed by indicator name).
    """

    # Validate that covid_analysis is either 1 or 2
    if covid_analysis not in [1, 2]:
        raise ValueError("Invalid value for covid_analysis. Must be 1 or 2.")
    covid_scenarios = [
        {"detection_reduction": False, "contact_reduction": False},  # No reduction
        {
            "detection_reduction": True,
            "contact_reduction": True,
        },  # With detection + contact reduction
        {
            "detection_reduction": True,
            "contact_reduction": False,
        },  # No contact reduction
    ]
    # Run models for the specified scenarios
    scenario_results = []
    for covid_effects in covid_scenarios:
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm)
        scenario_results.append(spaghetti_res.results)

    # Calculate the differences based on the covid_analysis value
    abs_diff = (
        scenario_results[covid_analysis][indicators] - scenario_results[0][indicators]
    )
    rel_diff = abs_diff / scenario_results[0][indicators]

    # Calculate the differences for each indicator and store them in DataFrames
    diff_quantiles_abs = {}
    diff_quantiles_rel = {}
    for ind in indicators:
        diff_quantiles_df_abs = pd.DataFrame(
            {
                quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        diff_quantiles_df_rel = pd.DataFrame(
            {
                quantile: [rel_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        diff_quantiles_abs[ind] = diff_quantiles_df_abs
        diff_quantiles_rel[ind] = diff_quantiles_df_rel

    return {"abs": diff_quantiles_abs, "rel": diff_quantiles_rel}

def calculate_covid_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: int = 2020,
    covid_analysis: int = 2,
    years: List[float] = [2021.0, 2022.0, 2025.0, 2030.0, 2035.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the models for the specified scenarios, calculate cumulative diseased and death values,
    and return quantiles for absolute and relative differences between scenarios.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        cumulative_start_time: Year to start calculating the cumulative values.
        covid_analysis: Integer specifying which analysis to run (default is 2).
        years: List of years for which to calculate the differences.

    Returns:
        A dictionary containing quantiles for absolute and relative differences between scenarios.
    """

    # Validate that covid_analysis is either 1 or 2
    if covid_analysis not in [1, 2]:
        raise ValueError("Invalid value for covid_analysis. Must be 1 or 2.")

    # Define the scenarios
    covid_scenarios = [
        {"detection_reduction": False, "contact_reduction": False},  # No reduction
        {
            "detection_reduction": True,
            "contact_reduction": True,
        },  # With detection + contact reduction
        {
            "detection_reduction": True,
            "contact_reduction": False,
        },  # No contact reduction
    ]

    scenario_results = []
    for covid_effects in covid_scenarios:
        # Get the model results
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results

        # Filter the results to include only the rows where the index (year) is an integer
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time) & 
            (spaghetti_res.index % 1 == 0)
        ]

        # Calculate cumulative sums for each sample
        cumulative_diseased_yearly = yearly_data['incidence_raw'].cumsum()
        cumulative_deaths_yearly = yearly_data['mortality_raw'].cumsum()

        # Store the cumulative results in the list
        scenario_results.append({
            "cumulative_diseased": cumulative_diseased_yearly,
            "cumulative_deaths": cumulative_deaths_yearly,
        })

    # Calculate the differences based on the covid_analysis value
    abs_diff = {
        "cumulative_diseased": scenario_results[covid_analysis]["cumulative_diseased"] - scenario_results[0]["cumulative_diseased"],
        "cumulative_deaths": scenario_results[covid_analysis]["cumulative_deaths"] - scenario_results[0]["cumulative_deaths"]
    }
    rel_diff = {
        "cumulative_diseased": abs_diff["cumulative_diseased"] / scenario_results[0]["cumulative_diseased"],
        "cumulative_deaths": abs_diff["cumulative_deaths"] / scenario_results[0]["cumulative_deaths"]
    }

    # Calculate quantiles for absolute and relative differences
    diff_quantiles_abs = {}
    diff_quantiles_rel = {}

    for ind in ["cumulative_diseased", "cumulative_deaths"]:
        # Calculate absolute difference quantiles
        diff_quantiles_df_abs = pd.DataFrame(
            {
                quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        # Calculate relative difference quantiles
        diff_quantiles_df_rel = pd.DataFrame(
            {
                quantile: [rel_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in quantiles
            },
            index=years,
        )

        diff_quantiles_abs[ind] = diff_quantiles_df_abs
        diff_quantiles_rel[ind] = diff_quantiles_df_rel

    return {"abs": diff_quantiles_abs, "rel": diff_quantiles_rel}


def calculate_notifications_for_covid(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate model outputs for each scenario defined in covid_configs and store the results
    in a dictionary where the keys correspond to the keys in covid_configs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to calculate outputs for.

    Returns:
        A dictionary where each key corresponds to a scenario in covid_configs and the value is 
        another dictionary containing DataFrames with outputs for the given indicators.
    """

    # Define the covid_configs inside the function
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

    scenario_outputs = {}

    # Loop through each scenario in covid_configs
    for scenario_name, covid_effects in covid_configs.items():
        # Run the model for the current scenario
        bcm = get_bcm(params, covid_effects)
        model_results = esamp.model_results_for_samples(idata_extract, bcm)
        spaghetti_res = model_results.results
        ll_res = model_results.extras  # Extract additional results (e.g., log-likelihoods)
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        # Initialize a dictionary to store indicator-specific outputs
        indicator_outputs = {}

         # Extract the results only for the "notification" indicator
        notification_indicator = "notification"  # Replace with the exact name of the notification indicator
        if notification_indicator in scenario_quantiles:
            indicator_outputs[notification_indicator] = scenario_quantiles[notification_indicator]

        # Store the outputs and ll_res in the dictionary with the scenario name as the key
        scenario_outputs[scenario_name] = {
            "indicator_outputs": indicator_outputs,
            "ll_res": ll_res
        }

    return scenario_outputs


def calculate_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'mortality_raw'],
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
    bcm = get_bcm(params, scenario_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, quantiles)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each detection multiplier scenario
    for multiplier in detection_multipliers:
        bcm = get_bcm(params, scenario_config, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, quantiles)

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
    scenario_choice: int = 2,
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
    if scenario_choice == 1:
        scenario_config = {"detection_reduction": True, "contact_reduction": True}
    elif scenario_choice == 2:
        scenario_config = {"detection_reduction": True, "contact_reduction": False}
    else:
        raise ValueError("Invalid scenario_choice. Choose 1 or 2.")

    # Base scenario (without improved detection)
    bcm = get_bcm(params, scenario_config)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results

    # Calculate cumulative sums for the base scenario
    yearly_data_base = base_results.loc[
        (base_results.index >= cumulative_start_time) & 
        (base_results.index % 1 == 0)
    ]
    cumulative_diseased_base = yearly_data_base['incidence_raw'].cumsum()
    cumulative_deaths_base = yearly_data_base['mortality_raw'].cumsum()

    # Store results for each detection multiplier
    detection_diff_results = {}

    for multiplier in detection_multipliers:
        # Improved detection scenario
        bcm = get_bcm(params, scenario_config, multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results

        # Calculate cumulative sums for each scenario
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time) & 
            (scenario_result.index % 1 == 0)
        ]
        cumulative_diseased = yearly_data['incidence_raw'].cumsum()
        cumulative_deaths = yearly_data['mortality_raw'].cumsum()

        # Calculate differences compared to the base scenario
        abs_diff = {
            "cumulative_diseased": cumulative_diseased - cumulative_diseased_base,
            "cumulative_deaths": cumulative_deaths - cumulative_deaths_base,
        }
        rel_diff = {
            "cumulative_diseased": abs_diff["cumulative_diseased"] / cumulative_diseased_base,
            "cumulative_deaths": abs_diff["cumulative_deaths"] / cumulative_deaths_base,
        }

        # Calculate quantiles for absolute and relative differences
        diff_quantiles_abs = {}
        diff_quantiles_rel = {}

        for ind in ["cumulative_diseased", "cumulative_deaths"]:
            diff_quantiles_df_abs = pd.DataFrame(
                {
                    quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                    for quantile in quantiles
                },
                index=years,
            )

            diff_quantiles_df_rel = pd.DataFrame(
                {
                    quantile: [rel_diff[ind].loc[year].quantile(quantile) for year in years]
                    for quantile in quantiles
                },
                index=years,
            )

            diff_quantiles_abs[ind] = diff_quantiles_df_abs
            diff_quantiles_rel[ind] = diff_quantiles_df_rel

        # Store the quantile results
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        detection_diff_results[scenario_key] = {
            "abs": diff_quantiles_abs,
            "rel": diff_quantiles_rel,
        }

    # Return the quantiles for absolute and relative differences
    return detection_diff_results