from summer2.parameters import Parameter, Function, Time
from summer2.functions.time import (
    get_sigmoidal_interpolation_function,
    get_linear_interpolation_function,
)
from typing import Dict
from tbdynamics.tools.utils import tanh_based_scaleup


def get_detection_func(
    detection_reduction: bool, improved_detection_multiplier: float=None
) -> Function:
    """
    Creates a detection function that scales over time based on various conditions.

    Args:
        detection_reduction (bool): Whether detection reduction due to COVID-19 should be applied.
        improved_detection_multiplier (float): A positive multiplier indicating an improvement in detection.

    Returns:
        Function: A function representing the detection rate over time.
    """
    # Define different detection rates by organ status
    detection_func = Function(
        tanh_based_scaleup,
        [
            Time,
            Parameter("screening_scaleup_shape"),
            Parameter("screening_inflection_time"),
            0.0,
            1.0 / Parameter("time_to_screening_end_asymp"),
        ],
    )

    if detection_reduction:
        detection_func = adjust_detection_for_covid(detection_func)

    if improved_detection_multiplier:
        assert (
            isinstance(improved_detection_multiplier, float)
            and improved_detection_multiplier > 0.0
        ), "improved_detection_multiplier must be a positive float."
        detection_func = adjust_detection_for_improvement(
            detection_func, improved_detection_multiplier
        )

    return detection_func


def adjust_detection_for_improvement(
    detection_func: Function, improved_detection_multiplier: float
) -> Function:
    """
    Adjusts the detection function to account for improvements in case detection over time.

    Args:
        detection_func (Function): The original detection function.
        improved_detection_multiplier (float): A multiplier indicating the improvement in detection.

    Returns:
        Function: The adjusted detection function incorporating the improvement factor.
    """
    improve_detect_vals = {
        2025.0: 1.0,
        2035.0: improved_detection_multiplier,
    }
    improve_detect_func = get_linear_interpolation_function(
        list(improve_detect_vals.keys()),
        list(improve_detect_vals.values()),
    )
    return detection_func * improve_detect_func


def adjust_detection_for_covid(detection_func: Function) -> Function:
    """
    Adjusts the detection function to account for reductions in detection due to COVID-19.

    Args:
        detection_func (Function): The original detection function.

    Returns:
        Function: The adjusted detection function incorporating COVID-19 impact.
    """
    covid_reduction = {
        2020.0: 1.0,
        2021.0: 1.0 - Parameter("detection_reduction"),
        2022.0: 1.0,
    }
    covid_detect_func = get_sigmoidal_interpolation_function(
        list(covid_reduction.keys()),
        list(covid_reduction.values()),
        curvature=8,
    )
    return detection_func * covid_detect_func

def get_interpolation_rates_from_annual(rates: Dict[float, float], most_of_year: float = 0.9) -> Dict[float, float]:
    """
    Generates interpolation points for rates given on an annual basis by adding an additional point
    at a specified fraction of the year.

    Parameters:
    - rates (Dict[float, float]): A dictionary where the key is the year (as a float) and the value is the rate.
    - most_of_year (float, optional): The fraction of the year after which to add another point for interpolation. Default is 0.9.

    Returns:
    - Dict[float, float]: A dictionary with the original and interpolated rate points sorted by year.
    """
    # Convert year keys to float and create starting rates
    start_rates = {float(k): v for k, v in rates.items()}
    # Calculate end rates by adding the fraction of the year to the year keys
    end_rates = {float(k) + most_of_year: v for k, v in rates.items()}
    # Merge dictionaries and sort by year
    interp_rates = start_rates | end_rates
    return dict(sorted(interp_rates.items()))