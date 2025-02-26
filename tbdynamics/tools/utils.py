from math import log, exp
from jax import numpy as jnp
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List
from summer2.parameters import Function, Time
from summer2.functions.time import get_sigmoidal_interpolation_function
from summer2 import Multiply


def triangle_wave_func(
    time: float,
    start: float,
    duration: float,
    peak: float,
) -> float:
    """
    Computes a smooth transition using a tanh-based scaling function.

    Args:
        t (float): Input time.
        shape (float): Controls the steepness of the transition.
        inflection_time (float): Time at which the transition is centered.
        start_asymptote (float): Lower asymptotic value.
        end_asymptote (float, optional): Upper asymptotic value.

    Returns:
        float: Scaled value at time t.
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(
        time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0
    )


def get_average_sigmoid(low_val: float, upper_val: float, inflection: float) -> float:
    """
    Computes the average value of a sigmoidal function over a given range.

    Uses a logistic function to model a progressive increase with age, as applied
    in Ragonnet et al. (BMC Medicine, 2019).

    Args:
        low_val (float): Lower bound of the range.
        upper_val (float): Upper bound of the range.
        inflection (float): Inflection point of the sigmoid function.

    Returns:
        float: Average value of the sigmoid function over the given range.
    """
    return (
        log(1.0 + exp(upper_val - inflection)) - log(1.0 + exp(low_val - inflection))
    ) / (upper_val - low_val)


def tanh_based_scaleup(t, shape, inflection_time, start_asymptote, end_asymptote=1.0):
    """
    return the function t: (1 - sigma) / 2 * tanh(b * (a - c)) + (1 + sigma) / 2
    :param shape: shape parameter
    :param inflection_time: inflection point
    :param start_asymptote: lowest asymptotic value
    :param end_asymptote: highest asymptotic value
    :return: a function
    """
    rng = end_asymptote - start_asymptote
    return (jnp.tanh(shape * (t - inflection_time)) / 2.0 + 0.5) * rng + start_asymptote


def get_average_age_for_bcg(agegroup, age_breakpoints):
    """
    Computes the average age for a given age group based on age breakpoints.

    Assumes age groups are defined by their starting ages and calculates the midpoint
    to the next group. If the last group is selected, it returns its starting age.

    Args:
        agegroup (str): Starting age of the age group.
        age_breakpoints (list): Sorted list of age group starting points.

    Returns:
        Average age of the group, or its starting age if it is the last group.
    """
    agegroup_idx = age_breakpoints.index(int(agegroup))
    if agegroup_idx == len(age_breakpoints) - 1:
        # We should normally never be in this situation because the last agegroup is not affected by BCG anyway.
        print(
            "Warning: the agegroup name is being used to represent the average age of the group"
        )
        return float(agegroup)
    else:
        return 0.5 * (age_breakpoints[agegroup_idx] + age_breakpoints[agegroup_idx + 1])


def bcg_multiplier_func(tfunc, fmultiplier):
    """
    Calculates the BCG vaccination effect multiplier based on a transmission function
    and a fractional multiplier.

    This function is used to adjust the impact of BCG vaccination on disease transmission
    by calculating a multiplier based on the reduction in transmission (as a percentage)
    and a fractional multiplier that represents the remaining susceptibility after vaccination.

    Parameters:
    - tfunc (float): The sigmoidal function, reflecting the percentage reduction in transmission due to BCG vaccination.
    - fmultiplier (float): The fractional multiplier representing the remaining susceptibility
      or the efficacy of the BCG vaccine in preventing disease transmission.

    Returns:
    - float: A multiplier representing the adjusted impact of BCG vaccination on disease
      transmission.

    """
    return 1.0 - tfunc / 100.0 * (1.0 - fmultiplier)


def calculate_treatment_outcomes(
    duration, prop_death_among_non_success, natural_death_rate, tsr
):
    """
    Computes adjusted treatment outcome proportions over a given duration.

    Args:
        duration (float): Treatment duration in the same time units as natural_death_rate.
        prop_death_among_non_success (float): Proportion of non-successful cases resulting in death (excluding natural deaths).
        natural_death_rate (float): Annual natural death rate for those under treatment.
        tsr (float): Treatment success rate.

    Returns:
        tuple of float: Adjusted proportions of treatment success, deaths from treatment (≥0), and relapse.

    Notes:
        - Accounts for natural deaths using exponential decay.
        - Ensures treatment-related deaths are non-negative.
    """
    # Calculate the proportion of people dying from natural causes while on treatment
    prop_natural_death_while_on_treatment = 1.0 - jnp.exp(
        -duration * natural_death_rate
    )

    # Calculate the target proportion of treatment outcomes resulting in death based on requests
    requested_prop_death_on_treatment = (1.0 - tsr) * prop_death_among_non_success

    # Calculate the actual rate of deaths on treatment, with floor of zero
    prop_death_from_treatment = jnp.max(
        jnp.array(
            (
                requested_prop_death_on_treatment
                - prop_natural_death_while_on_treatment,
                0.0,
            )
        )
    )

    # Calculate the proportion of treatment episodes resulting in relapse
    relapse_prop = (
        1.0 - tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
    )

    return tuple(
        [param * duration for param in [tsr, prop_death_from_treatment, relapse_prop]]
    )


def round_sigfig(value: float, sig_figs: int) -> float:
    """
    Round a number to a certain number of significant figures,
    rather than decimal places.

    Args:
        value: Number to round
        sig_figs: Number of significant figures to round to
    """
    if np.isinf(value):
        return "infinity"
    else:
        return (
            round(value, -int(np.floor(np.log10(value))) + (sig_figs - 1))
            if value != 0.0
            else 0.0
        )


def get_target_from_name(
    targets: list,
    name: str,
) -> pd.Series:
    """Get the data for a specific target from a set of targets from its name.

    Args:
        targets: All the targets
        name: The name of the desired target

    Returns:
        Single target to identify
    """
    return next((t.data for t in targets if t.name == name), None)


def get_row_col_for_subplots(i_panel, n_cols):
    return int(np.floor(i_panel / n_cols)) + 1, i_panel % n_cols + 1


def get_standard_subplot_fig(
    n_rows: int,
    n_cols: int,
    titles: List[str],
    share_y: bool = False,
) -> go.Figure:
    """Start a plotly figure with subplots off from standard formatting.

    Args:
        n_rows: Argument to pass through to make_subplots
        n_cols: Pass through
        titles: Pass through

    Returns:
        Figure with nothing plotted
    """
    heights = [320, 600, 680]
    height = 680 if n_rows > 3 else heights[n_rows - 1]
    fig = make_subplots(
        n_rows,
        n_cols,
        subplot_titles=titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
        shared_yaxes=share_y,
    )
    # return fig.update_layout(margin={i: 25 for i in ['t', 'b', 'l', 'r']}, height=height)
    return fig.update_layout(height=height)


# def calculate_cdr_adjustments(case_detection_rate, infect_death, self_recovery):
#     return case_detection_rate * (infect_death + self_recovery) / (1 - case_detection_rate)


def get_mix_from_strat_props(within_strat, props):
    """
    Generates a mixing matrix based on stratification proportions.

    Args:
        within_strat (float): Proportion of mixing within the same stratum.
        props (List[float]): Proportions of each stratum in the population.

    Returns:
        np.ndarray: A mixing matrix where within-stratum mixing is scaled by `within_strat`,
        and between-stratum mixing is distributed based on `props`.
    """
    return np.eye(len(props)) * within_strat + np.stack(
        [np.array(props)] * len(props)
    ) * (1.0 - within_strat)


def calculate_bcg_adjustment(
    age: float,
    multiplier: float,
    age_strata: List[int],
    bcg_time_keys: List[float],
    bcg_time_values: List[float],
):
    """
    Calculates an age-adjusted BCG vaccine efficacy multiplier for individuals based on
    their age and the provided BCG time keys and values. If the given multiplier is less
    than 1.0, indicating some vaccine efficacy, the function calculates an age-adjusted
    multiplier using a sigmoidal interpolation function.

    Args:
        age: The age of the individual for which the adjustment is being calculated.
        multiplier: The baseline efficacy multiplier of the BCG vaccine.
        age_strata: A list of age groups used in the model for stratification.
        bcg_time_keys: A list of time points (usually in years) for the sigmoidal interpolation function.
        bcg_time_values: A list of efficacy multipliers corresponding to the bcg_time_keys.
    """
    if multiplier < 1.0:
        # Calculate age-adjusted multiplier using a sigmoidal interpolation function
        age_adjusted_time = Time - get_average_age_for_bcg(age, age_strata)
        interpolation_func = get_sigmoidal_interpolation_function(
            bcg_time_keys,
            bcg_time_values,
            age_adjusted_time,
        )
        return Multiply(Function(bcg_multiplier_func, [interpolation_func, multiplier]))
    else:
        # No adjustment needed for multipliers of 1.0
        return None
