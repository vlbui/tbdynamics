from summer2.parameters import Parameter, Function, Time
from summer2.functions.time import (
    get_sigmoidal_interpolation_function,
    get_linear_interpolation_function,
)
from tbdynamics.tools.utils import tanh_based_scaleup


def get_detection_func(detection_reduction, improved_detection_multiplier):

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
        detection_func = adjust_detection_for_improvement()

    return detection_func


def adjust_detection_for_improvement(detection_func, improved_detection_multiplier):

    improve_detect_vals = {
        2025.0: 1.0,
        2035.0: improved_detection_multiplier,
    }
    improve_detect_func = get_linear_interpolation_function(
        list(improve_detect_vals.keys()),
        list(improve_detect_vals.values()),
    )
    return detection_func * improve_detect_func


def adjust_detection_for_covid(detection_func):

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