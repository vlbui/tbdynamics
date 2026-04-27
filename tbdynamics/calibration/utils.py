import arviz as az
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Callable, Dict, List


def load_idata(out_path: str, covid_configs: Dict) -> dict:
    """
    Load inference data for different COVID-19 configurations from NetCDF files.

    Args:
        out_path (str): The directory containing inference data files.
        covid_configs (dict): Dictionary of COVID-19 configuration names.

    Returns:
        dict: A dictionary mapping configuration names to their corresponding InferenceData objects.
    """
    inference_data_dict = {}
    for config_name in covid_configs.keys():
        calib_file = Path(out_path) / f"calib_full_out_{config_name}.nc"
        if calib_file.exists():
            idata_raw = az.from_netcdf(calib_file)
            inference_data_dict[config_name] = idata_raw
        else:
            print(f"File {calib_file} does not exist.")
    return inference_data_dict


def extract_and_save_idata(idata_dict: Dict, output_dir: str,tune_draws = 50000, num_samples: int = 1000) -> None:
    """
    Extract and save inference data for each COVID-19 configuration.

    Args:
        idata_dict (dict): Dictionary mapping configuration names to InferenceData objects.
        output_dir (str): Directory to save the extracted inference data.
        num_samples (int, optional): Number of samples to extract. Defaults to 1000.

    Returns:
        None
    """
    for config_name, idata in idata_dict.items():
        # Extract samples (you might adjust the number of samples as needed)
        burnt_idata = idata.sel(draw=slice(tune_draws, None))
        idata_extract = az.extract(burnt_idata, num_samples=num_samples)

        # Convert extracted data into InferenceData object
        inference_data = az.convert_to_inference_data(
            idata_extract.reset_index("sample")
        )

        # Save the extracted InferenceData object to a NetCDF file
        output_file = Path(output_dir) / f"idata_{config_name}.nc"
        az.to_netcdf(inference_data, output_file)
        print(f"Saved extracted inference data for {config_name} to {output_file}")


def build_diff_quantile_df(
    diff_series: pd.Series,
    years: List,
    quantiles: List[float],
) -> pd.DataFrame:
    """Build a quantile DataFrame from a cross-sample diff series at specified years."""
    return pd.DataFrame(
        {q: [diff_series.loc[year].quantile(q) for year in years] for q in quantiles},
        index=years,
    )


def convert_ll_to_idata(ll_res) -> az.InferenceData:
    df = pd.DataFrame(ll_res)
    ds = xr.Dataset.from_dataframe(df)
    return az.from_dict(
        posterior={"logposterior": ds["logposterior"]},
        prior={"logprior": ds["logprior"]},
        log_likelihood={"total_loglikelihood": ds["loglikelihood"]},
    )


def calculate_waic_comparison(covid_outputs: Dict) -> pd.DataFrame:
    waic_dict = {
        covid_name: convert_ll_to_idata(output["ll_res"])
        for covid_name, output in covid_outputs.items()
    }
    waic_results = {name: az.waic(idata) for name, idata in waic_dict.items()}
    return az.compare(waic_results, ic="waic")


def run_model_for_covid(
    params: Dict,
    output_dir,
    covid_configs: Dict,
    quantiles: List[float],
    get_bcm_func: Callable,
) -> Dict:
    """Run model for each COVID scenario and return quantile outputs + log-likelihoods."""
    from estival.sampling import tools as esamp

    covid_outputs = {}
    inference_data_dict = load_extracted_idata(output_dir, covid_configs)

    for covid_name, covid_effects in covid_configs.items():
        if covid_name not in inference_data_dict:
            print(f"Skipping {covid_name} as no inference data was loaded.")
            continue

        idata_extract = inference_data_dict[covid_name]
        bcm = get_bcm_func(params, covid_effects)
        model_results = esamp.model_results_for_samples(idata_extract, bcm)
        spaghetti_res = model_results.results
        ll_res = model_results.extras
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        indicators = ["notification", "total_population", "adults_prevalence_pulmonary"]
        missing = [i for i in indicators if i not in scenario_quantiles.columns]
        if missing:
            print(f"Missing indicators {missing} in scenario {covid_name}. Skipping.")
            continue

        covid_outputs[covid_name] = {
            "indicator_outputs": scenario_quantiles[indicators],
            "ll_res": ll_res,
        }

    return covid_outputs


def calculate_covid_cum_results(
    params: Dict,
    idata_extract,
    get_bcm_func: Callable,
    cumulative_start_time: float = 2020.0,
    years: List[float] = [2021.0, 2022.0, 2025.0, 2030.0, 2035.0],
) -> Dict:
    """Calculate cumulative incidence and deaths for no-COVID vs detection-reduction scenarios."""
    from estival.sampling import tools as esamp

    covid_configs = {
        "no_covid": {"detection_reduction": False, "contact_reduction": False},
        "detection_reduction_only": {"detection_reduction": True, "contact_reduction": False},
    }

    scenario_results = {}
    for scenario_name, covid_effects in covid_configs.items():
        bcm = get_bcm_func(params, covid_effects)
        spaghetti_res = esamp.model_results_for_samples(idata_extract, bcm).results
        yearly_data = spaghetti_res.loc[
            (spaghetti_res.index >= cumulative_start_time) & (spaghetti_res.index % 1 == 0)
        ]
        scenario_results[scenario_name] = {
            "cumulative_diseased": yearly_data["incidence_raw"].cumsum().loc[years],
            "cumulative_deaths": yearly_data["mortality_raw"].cumsum().loc[years],
        }

    return scenario_results


def load_extracted_idata(out_path: str, covid_configs: Dict) -> Dict:
    """
    Load extracted inference data from NetCDF files for different COVID-19 configurations.

    Args:
        out_path (str): Directory containing extracted inference data files.
        covid_configs (dict): Dictionary of COVID-19 configuration names.

    Returns:
        Dict: A dictionary mapping configuration names to their corresponding InferenceData objects.
    """
    inference_data_dict = {}
    for config_name in covid_configs.keys():
        input_file = Path(out_path) / f"idata_{config_name}.nc"
        if input_file.exists():
            idata = az.from_netcdf(input_file)
            inference_data_dict[config_name] = idata
        else:
            print(f"File {input_file} does not exist.")
    return inference_data_dict

