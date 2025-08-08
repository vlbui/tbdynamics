from summer2.parameters import Parameter, Function, Time
from summer2.functions.time import (
    get_sigmoidal_interpolation_function,
    get_linear_interpolation_function,
)
from typing import Dict, List
from tbdynamics.tools.utils import tanh_based_scaleup
from tbdynamics.camau.constants import ACT3_STRATA
import math


def get_detection_func(
    detection_reduction: bool,
    improved_detection_multiplier: float = None,
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


def adjust_detection_for_act3(
    detection_func: Function
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
        2018.0: 1.0,
        2020.0: Parameter("detection_spill_over_effect"),
        2020.1: 1.0
    }
    improve_detect_func = get_linear_interpolation_function(
        list(improve_detect_vals.keys()),
        list(improve_detect_vals.values()),
    )
    return detection_func * improve_detect_func


def get_interpolation_rates_from_annual(rates: Dict[float, float]):
    if not rates:
        return {}
    # Ensure keys are sorted floats
    years = sorted(float(k) for k in rates.keys())
    interp_rates = {}

    for i in range(len(years)):
        y = years[i]
        v = rates[y]
        interp_rates[y] = v

        if i < len(years) - 1:
            next_y = years[i + 1]
            next_v = rates[next_y]

            # Interpolate next year's value at y + 0.1
            interp_rates[y + 0.1] = next_v

    return dict(sorted(interp_rates.items()))

def calculate_screening_rate(
    adults_pop: Dict[float, float], sputum_collected: Dict[float, float]
) -> Dict[float, float]:
    """
    Calculates the screening rate for each year as -ln(1 - sputum_collected / adults_pop).

    Args:
        adults_pop: Dictionary with year as keys and adult population as values.
        sputum_collected: Dictionary with year as keys and sputum collected count as values.

    Returns:
        Dict: A dictionary with year as keys and calculated screening rate as values.
    """
    screening_rates = {}

    for year in adults_pop:
        if year in sputum_collected:
            # Calculate the screening rate: -ln(1 - sputum_collected/adults_pop)
            rate = -math.log(1 - (sputum_collected[year] / adults_pop[year]))
            screening_rates[year] = rate

    # Add the last year + 0.1 with value 0
    if screening_rates:
        first_year = min(screening_rates.keys())
        screening_rates[first_year - 1.0] = 0.0
        last_year = max(screening_rates.keys())
        screening_rates[last_year + 0.1] = 0.0
    return dict(sorted(screening_rates.items()))


def make_future_acf_scenarios(
    config: Dict[str, List] = {
        "arm": ACT3_STRATA,
        "every": [2, 4],
        "coverage": [0.5, 0.8],
    },
    start_year: int = 2026,
    horizon_end: float = 2035.0,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    scenarios = {}
    arms = config.get("arm", ACT3_STRATA)
    every_list = config.get("every", [2, 4])
    coverages = config.get("coverage", [0.5, 0.8])

    for freq in every_list:
        assert freq in [2, 4], f"Unsupported frequency: {freq}"
        starts = list(range(start_year, int(horizon_end) + 1, freq))

        for cov in coverages:
            assert 0 < cov <= 1.0, f"Coverage must be in (0, 1], got {cov}"
            rate = -math.log(1.0 - cov)

            d: Dict[float, float] = {}
            d[round(start_year - 0.1, 1)] = 0.0  # initial pre-start anchor

            for start in starts:
                if start > horizon_end:
                    break

                # For every=4, add an off point before ON starts
                if freq == 4 and start != start_year:
                    d[round(start - 0.1, 1)] = 0.0

                # Two-year ON: start and start+1 (clamped to horizon)
                d[float(start)] = rate
                if start + 1.0 <= horizon_end:
                    d[float(start + 1.0)] = rate

                # OFF markers after ON
                off_01 = round(min(start + 1.1, horizon_end + 0.1), 1)
                d[off_01] = 0.0
                off_09 = round(min(start + 1.9, horizon_end + 0.9), 1)
                d[off_09] = 0.0

            # Post-horizon anchors
            d[round(horizon_end + 0.1, 1)] = 0.0
            d[round(horizon_end + 0.9, 1)] = 0.0

            rate_dict = dict(sorted(d.items()))

            arm_label = "all" if set(arms) == set(ACT3_STRATA) else "_".join(arms)
            key = f"{arm_label}_{freq}_{int(cov * 100)}"
            scenarios[key] = {arm: rate_dict.copy() for arm in arms}

    return scenarios


