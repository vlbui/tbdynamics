import arviz as az
from pathlib import Path
from typing import Dict


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


def extract_and_save_idata(idata_dict: Dict, output_dir: str, num_samples: int = 1000) -> None:
    """
    Extract and save inference data for each COVID-19 configuration.

    Args:
        idata_dict (dict): Dictionary mapping configuration names to InferenceData objects.
        output_dir (str): Directory to save the extracted inference data.
        num_samples (int, optional): Number of samples to extract. Defaults to 1000.

    Returns:
        None
    """
    for config_name, burnt_idata in idata_dict.items():
        # Extract samples (you might adjust the number of samples as needed)
        idata_extract = az.extract(burnt_idata, num_samples=num_samples)

        # Convert extracted data into InferenceData object
        inference_data = az.convert_to_inference_data(
            idata_extract.reset_index("sample")
        )

        # Save the extracted InferenceData object to a NetCDF file
        output_file = Path(output_dir) / f"idata_{config_name}.nc"
        az.to_netcdf(inference_data, output_file)
        print(f"Saved extracted inference data for {config_name} to {output_file}")


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
