from math import log, exp
from jax import numpy as jnp
import numpy as np
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_PATH / "data"


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

def triangle_wave_func(
    time: float,
    start: float,
    duration: float,
    peak: float,
) -> float:
    """Generate a peaked triangular wave function
    that starts from and returns to zero.

    Args:
        time: Model time
        start: Time at which wave starts
        duration: Duration of wave
        peak: Peak flow rate for wave

    Returns:
        The wave function
    """
    gradient = peak / (duration * 0.5)
    peak_time = start + duration * 0.5
    time_from_peak = jnp.abs(peak_time - time)
    return jnp.where(
        time_from_peak < duration * 0.5, peak - time_from_peak * gradient, 0.0
    )

def get_average_sigmoid(low_val, upper_val, inflection):
    """
    A sigmoidal function (x -> 1 / (1 + exp(-(x-alpha)))) is used to model a progressive increase with age.
    This is the approach used in Ragonnet et al. (BMC Medicine, 2019)
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

