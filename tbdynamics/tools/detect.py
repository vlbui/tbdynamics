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

def get_interpolation_rates_from_annual(rates, most_of_year=0.9):
    # Convert keys to float and create initial rates at the start of the period
    start_rates = {float(k): v for k, v in rates.items()}
    
    # Create rates towards the end of the period based on most_of_year
    end_rates = {}
    keys = sorted(rates.keys())  # Sort keys to manage sequence properly
    last_key_index = len(keys) - 1
    
    for i, k in enumerate(keys):
        current_key = float(k)
        current_value = rates[k]
        
        # Determine if we are at the last key or if the next key is a fractional continuation
        if i != last_key_index:
            next_key = keys[i + 1]
            
            # Check if the next key is a fractional continuation of the current year
            if next_key == current_key + 0.1:
                start_rates[next_key] = rates[next_key]
            else:
                # If not, then extend the current rate to the most_of_year point
                end_rate_time = current_key + most_of_year
                if end_rate_time < next_key:
                    end_rates[end_rate_time] = current_value
        else:
            # For the last key, extend the rate to the end of the year and add an extra point
            end_rates[current_key + most_of_year] = current_value
            # This also means maintaining the last rate up to one year later if it's a special non-full-year key
            if current_key != int(current_key):
                end_rates[current_key + 1] = current_value

    # Combine the start and end rates and ensure no duplicates with different values
    interp_rates = {**start_rates, **end_rates}

    # Sort and return the dictionary
    return dict(sorted(interp_rates.items()))
