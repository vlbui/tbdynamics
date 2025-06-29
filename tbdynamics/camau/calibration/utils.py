from estival.model import BayesianCompartmentalModel
from estival.sampling import tools as esamp
import estival.priors as esp
import estival.targets as est
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
import arviz as az
import xarray as xr
from tbdynamics.camau.model import build_model
from tbdynamics.tools.inputs import load_params, load_targets, matrix
from tbdynamics.calibration.utils import load_extracted_idata
from tbdynamics.settings import CM_PATH
from tbdynamics.constants import COMPARTMENTS, QUANTILES


def get_bcm(
    params: Dict[str, float],
    covid_effects: Optional[Dict[str, bool]] = None,
    improved_detection_multiplier: Optional[float] = None,
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
    # create a mixing matrix
    # mixing_matrix = np.ones((6, 6)) if homo_mixing else matrix

    priors = get_all_priors(covid_effects)
    # priors.insert(
    #     0,
    #     esp.UniformPrior("contact_rate", (0.001, 0.03)),
    #     # esp.UniformPrior("contact_rate", (1.0, 50.0) if homo_mixing else (0.001, 0.05)),
    # )
    # for prior in priors:
    #     prior._pymc_transform_eps_scale = 0.1

    targets = get_targets()
    tb_model = build_model(
        fixed_params, matrix, covid_effects, improved_detection_multiplier
    )

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
        esp.TruncNormalPrior("contact_rate", 0.02, 0.05, (0.001, 0.04)),
        esp.BetaPrior("rr_infection_latent", 3.0, 5.0),
        esp.BetaPrior("rr_infection_recovered", 2.5, 4.5),
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
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 2.0, 5.0),
        esp.TruncNormalPrior("early_prop_adjuster", 0, 0.05, (-2.0, 2.0)),
        esp.GammaPrior.from_mode("late_reactivation_adjuster", 1.0, 2.0),
    ]

    if covid_effects:
        if covid_effects.get("contact_reduction"):
            priors.append(esp.UniformPrior("contact_reduction", (0.01, 0.9)))
        if covid_effects.get("detection_reduction"):
            priors.append(esp.UniformPrior("detection_reduction", (0.01, 0.9)))

    return priors


def get_targets() -> List[est.NormalTarget]:
    """
    Loads target data for Bayesian model calibration.

    Returns:
        List[est.NormalTarget]: A list of target distributions used for calibration.
    """
    target_data = load_targets(CM_PATH / "targets.yml")

    return [
        est.NormalTarget(
            "total_population",
            target_data["total_population"],
            esp.UniformPrior("total_population_dispersion", (500.0, 5000.0)),
        ),
        est.NormalTarget(
            "log_notification",
            np.log(target_data["notification"]),
            esp.TruncNormalPrior("notif_dispersion",0.0,0.1, (0.0, np.inf))
        ),
        # est.NormalTarget(
        #     "act3_trial_adults_pop",
        #     target_data["act3_trial_adults_pop"],
        #     esp.UniformPrior("adults_populationXact3_trial_dispersion", (10.0, 3000.0)),
        # ),
        # est.NormalTarget(
        #     "act3_control_adults_pop",
        #     target_data["act3_control_adults_pop"],
        #     esp.UniformPrior(
        #         "adults_populationXact3_control_dispersion", (10.0, 3000.0)
        #     ),
        # ),
        est.NormalTarget(
            "percentage_latent_adults",
            target_data["percentage_latent_adults_target"],
            esp.UniformPrior("latent_dispersion", (1.0, 15.0)),
        ),
        # est.BinomialTarget(
        #     "acf_detectionXact3_trialXorgan_pulmonary_prop",
        #     target_data["acf_detectionXact3_trialXorgan_pulmonary_prop"],
        #     target_data["acf_detectionXact3_trialXsample"],
        # ),
        est.BinomialTarget(
            "acf_detectionXact3_controlXorgan_pulmonary_prop",
            target_data["acf_detectionXact3_controlXorgan_pulmonary_prop"],
            target_data["acf_detectionXact3_controlXsample"],
        ),
        # est.NormalTarget(
        #     "acf_detectionXact3_trialXorgan_pulmonary",
        #     target_data["acf_detectionXact3_trialXorgan_pulmonary"],
        #     esp.UniformPrior("act3_trial_dispersion", (1.0, 30.0))
        # ),
        # est.NormalTarget(
        #     "acf_detectionXact3_controlXorgan_pulmonary",
        #     target_data["acf_detectionXact3_controlXorgan_pulmonary"],
        #     esp.UniformPrior("act3_control_dispersion", (1.0, 30.0))
        # ),
    ]

def calculate_covid_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: float = 2020.0,
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

    # Define the scenarios
    covid_configs = [
        {"detection_reduction": False, "contact_reduction": False},  # No reduction
        {
            "detection_reduction": True,
            "contact_reduction": False,
        },  # No contact reduction
    ]

    covid_results = []
    for covid_effects in covid_configs:
        # Get the model results
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results

        # Filter the results to include only the rows where the index (year) is an integer
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time)
            & (spaghetti_res.index % 1 == 0)
        ]

        # Calculate cumulative sums for each sample
        cumulative_diseased_yearly = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths_yearly = yearly_data["mortality_raw"].cumsum()

        # Store the cumulative results in the list
        covid_results.append(
            {
                "cumulative_diseased": cumulative_diseased_yearly,
                "cumulative_deaths": cumulative_deaths_yearly,
            }
        )

    # Calculate the differences based on the covid_analysis value
    abs_diff = {
        "cumulative_diseased": covid_results[1]["cumulative_diseased"]
        - covid_results[0]["cumulative_diseased"],
        "cumulative_deaths": covid_results[1]["cumulative_deaths"]
        - covid_results[0]["cumulative_deaths"],
    }
    rel_diff = {
        "cumulative_diseased": abs_diff["cumulative_diseased"]
        / covid_results[0]["cumulative_diseased"],
        "cumulative_deaths": abs_diff["cumulative_deaths"]
        / covid_results[0]["cumulative_deaths"],
    }

    # Calculate quantiles for absolute and relative differences
    diff_quantiles_abs = {}
    diff_quantiles_rel = {}

    for ind in ["cumulative_diseased", "cumulative_deaths"]:
        # Calculate absolute difference quantiles
        diff_quantiles_df_abs = pd.DataFrame(
            {
                quantile: [abs_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in QUANTILES
            },
            index=years,
        )

        # Calculate relative difference quantiles
        diff_quantiles_df_rel = pd.DataFrame(
            {
                quantile: [rel_diff[ind].loc[year].quantile(quantile) for year in years]
                for quantile in QUANTILES
            },
            index=years,
        )

        diff_quantiles_abs[ind] = diff_quantiles_df_abs
        diff_quantiles_rel[ind] = diff_quantiles_df_rel

    return {"abs": diff_quantiles_abs, "rel": diff_quantiles_rel}


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
    bcm = get_bcm(params, scenario_config, None, False)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, QUANTILES)
 
    baseline_indicators = [
        "total_population",
        "notification",
        "adults_prevalence_pulmonary",
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
    no_transmission_bcm = get_bcm(params, scenario_config, None, True)
    no_transmission_results = esamp.model_results_for_samples(idata_extract, no_transmission_bcm).results
    no_transmission_quantiles = esamp.quantiles_for_results(no_transmission_results, QUANTILES)

    # Store the results for the no-transmission scenario
    scenario_outputs["no_transmission"] = no_transmission_quantiles


    # Calculate quantiles for each detection multiplier scenario
    for multiplier in detection_multipliers:
        bcm = get_bcm(params, scenario_config, multiplier, False)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

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
    extreme_transmission: bool = False,
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

    # Store results for each detection multiplier
    detection_diff_results = {}

    for multiplier in detection_multipliers:
        # Improved detection scenario
        bcm = get_bcm(params, covid_config, multiplier, extreme_transmission)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results

        # Calculate cumulative sums for each scenario
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time)
            & (scenario_result.index % 1 == 0)
        ]
        cumulative_diseased = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths = yearly_data["mortality_raw"].cumsum()

        # Calculate differences compared to the base scenario
        abs_diff = {
            "cumulative_diseased": cumulative_diseased - cumulative_diseased_base,
            "cumulative_deaths": cumulative_deaths - cumulative_deaths_base,
        }
        rel_diff = {
            "cumulative_diseased": abs_diff["cumulative_diseased"]
            / cumulative_diseased_base * 100,
            "cumulative_deaths": abs_diff["cumulative_deaths"] / cumulative_deaths_base * 100,
        }

        # Calculate quantiles for absolute and relative differences
        diff_quantiles_abs = {}
        diff_quantiles_rel = {}

        for ind in ["cumulative_diseased", "cumulative_deaths"]:
            diff_quantiles_df_abs = pd.DataFrame(
                {
                    quantile: [
                        abs_diff[ind].loc[year].quantile(quantile) for year in years
                    ]
                    for quantile in QUANTILES
                },
                index=years,
            )

            diff_quantiles_df_rel = pd.DataFrame(
                {
                    quantile: [
                        rel_diff[ind].loc[year].quantile(quantile) for year in years
                    ]
                    for quantile in QUANTILES
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


def run_model_for_covid(params, output_dir, covid_configs, quantiles):
    covid_outputs = {}

    # Load the extracted InferenceData
    inference_data_dict = load_extracted_idata(output_dir, covid_configs)

    for covid_name, covid_effects in covid_configs.items():
        # Load the inference data for this specific scenario
        if covid_name not in inference_data_dict:
            print(f"Skipping {covid_name} as no inference data was loaded.")
            continue

        idata_extract = inference_data_dict[covid_name]

        # Run the model for the current scenario
        bcm = get_bcm(params, covid_effects)  # Adjust this function as needed
        model_results = esamp.model_results_for_samples(idata_extract, bcm)

        # Extract results from the model output
        spaghetti_res = model_results.results
        ll_res = (
            model_results.extras
        )  # Extract additional results (e.g., log-likelihoods)
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        # Define the indicators you're interested in
        indicators = ["notification", "total_population", "adults_prevalence_pulmonary"]

        missing_indicators = [
            indicator
            for indicator in indicators
            if indicator not in scenario_quantiles.columns
        ]
        if missing_indicators:
            print(
                f"Missing indicators {missing_indicators} in scenario {covid_name}. Skipping this scenario."
            )
            continue

        # Store the DataFrame of quantiles directly for the defined indicators
        indicator_outputs = scenario_quantiles[indicators]

        # Store the outputs and log-likelihoods in the dictionary with the scenario name as the key
        covid_outputs[covid_name] = {
            "indicator_outputs": indicator_outputs,
            "ll_res": ll_res,
        }

    return covid_outputs


def convert_ll_to_idata(ll_res):
    # Convert log-likelihoods into a DataFrame
    df = pd.DataFrame(ll_res)

    # Convert the DataFrame into an xarray.Dataset
    ds = xr.Dataset.from_dataframe(df)

    # Create an InferenceData object
    idata = az.from_dict(
        posterior={"logposterior": ds["logposterior"]},
        prior={"logprior": ds["logprior"]},
        log_likelihood={"total_loglikelihood": ds["loglikelihood"]},
    )

    return idata


def calculate_waic_comparison(covid_outputs):
    waic_dict = {}

    for covid_name, output in covid_outputs.items():
        # Extract the log-likelihoods (ll_res) for the current scenario
        ll_res = output["ll_res"]

        # Convert ll_res to InferenceData
        idata = convert_ll_to_idata(ll_res)

        # Store InferenceData in the dictionary for WAIC comparison
        waic_dict[covid_name] = idata

    # Compare the WAIC across all scenarios
    waic_results = {
        config_name: az.waic(idata) for config_name, idata in waic_dict.items()
    }

    # Compare using az.compare
    waic_comparison = az.compare(
        waic_results, ic="waic"
    )  # Using WAIC for information criterion

    return waic_comparison


def calculate_covid_cum_results(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    cumulative_start_time: float = 2020.0,
    years: List[float] = [2021.0, 2022.0, 2025.0, 2030.0, 2035.0],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the models for the specified scenarios, calculate cumulative diseased and death values,
    and return the results for each scenario.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        cumulative_start_time: Year to start calculating the cumulative values.
        years: List of years for which to calculate the results.

    Returns:
        A dictionary containing cumulative diseased and deaths results for each scenario.
    """

    # Define the scenarios with scenario names as keys
    covid_configs = {
        "no_covid": {
            "detection_reduction": False,
            "contact_reduction": False,
        },  # No reduction
        "detection_reduction_only": {
            "detection_reduction": True,
            "contact_reduction": False,
        },  # Detection reduction only
    }

    scenario_results = {}

    for scenario_name, covid_effects in covid_configs.items():
        # Get the model results
        bcm = get_bcm(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results

        # Filter the results to include only the rows where the index (year) is an integer
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time)
            & (spaghetti_res.index % 1 == 0)
        ]

        # Calculate cumulative sums for each sample
        cumulative_diseased_yearly = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths_yearly = yearly_data["mortality_raw"].cumsum()

        # Extract results for specified years
        cumulative_diseased_results = cumulative_diseased_yearly.loc[years]
        cumulative_deaths_results = cumulative_deaths_yearly.loc[years]

        # Store the results in the dictionary
        scenario_results[scenario_name] = {
            "cumulative_diseased": cumulative_diseased_results,
            "cumulative_deaths": cumulative_deaths_results,
        }

    return scenario_results

