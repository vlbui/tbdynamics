from estival.model import BayesianCompartmentalModel
import estival.priors as esp
import estival.targets as est
import arviz as az
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from tbdynamics.model import build_model
from tbdynamics.inputs import load_params, load_targets, matrix
from tbdynamics.constants import (
    age_strata,
    compartments,
    latent_compartments,
    infectious_compartments,
    indicator_names,
)
from tbdynamics.utils import get_row_col_for_subplots, get_standard_subplot_fig
from tbdynamics.constants import indicator_names

pio.templates.default = "simple_white"


def get_bcm(params=None) -> BayesianCompartmentalModel:
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
    fixed_params = load_params()
    tb_model = build_model(
        compartments,
        latent_compartments,
        infectious_compartments,
        age_strata,
        fixed_params,
        matrix,
    )
    priors = get_all_priors()
    targets = get_targets()
    # prev_dispersion = esp.UniformPrior("prev_dispersion", (10, 50))
    # notif_dispersion = esp.UniformPrior("notif_dispersion", (2000, 10000))
    target_data = load_targets()

    targets.extend(
        [
            # est.NormalTarget(
            #     "adults_prevalence_pulmonary",
            #     target_data["adults_prevalence_pulmonary"],
            #     stdev=36.0,
            # ),
            est.NormalTarget("notification", target_data["notification"], stdev=4000.0),
            # est.NormalTarget(
            #     "prevalence_smear_positive",
            #     target_data["prevalence_smear_positive"],
            #     20.0,
            # ),
        ]
    )
    return BayesianCompartmentalModel(tb_model, params, priors, targets)


def get_all_priors() -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    return [
        esp.UniformPrior("contact_rate", (0.001, 0.05)),
        # esp.UniformPrior("start_population_size", (2000000.0, 2500000.0)),
        esp.BetaPrior("rr_infection_latent", 3.0, 8.0), #The weighted adjusted risk ratio was 0.21 (95% CI: .14–.30)
        esp.BetaPrior("rr_infection_recovered", 2.0, 2.0),
        # esp.GammaPrior("progression_multiplier", 2.0, 1.0),
        # esp.UniformPrior("seed_time", (1800.0, 1840.0)),
        # esp.UniformPrior("seed_num", (1.0, 100.00)),
        # esp.UniformPrior("seed_duration", (1.0, 20.0)),
        # esp.UniformPrior("smear_positive_death_rate", (0.335, 0.449)),
        # esp.UniformPrior("smear_negative_death_rate", (0.017, 0.035)),
        # esp.UniformPrior("smear_positive_self_recovery", (0.177, 0.288)),
        # esp.UniformPrior("smear_negative_self_recovery", (0.073, 0.209)),
        # esp.UniformPrior("screening_scaleup_shape", (0.05, 0.5)),
        # esp.UniformPrior("screening_inflection_time", (1990, 2010)),
        # esp.GammaPrior.from_mode("time_to_screening_end_asymp", 1.0, 2.0),
        # esp.TruncNormalPrior("time_to_screening_end_asymp", 1.3, 0.077, (0.0, 12.8)),
        # esp.UniformPrior("detection_reduction", (0.01, 0.5)),
        # esp.UniformPrior("contact_reduction", (0.01, 0.8)),
        # esp.UniformPrior("incidence_props_smear_positive_among_pulmonary", (0.1, 0.8)),
    ]


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
    target_data = load_targets()
    return [
        est.NormalTarget(
            "total_population", target_data["total_population"], stdev=100000.0
        ),
        # est.NormalTarget("notification", target_data["notification"], stdev=6000.0),
        # est.NormalTarget(
        #     "adults_prevalence_pulmonary",
        #     target_data["adults_prevalence_pulmonary"],
        #     stdev=36.0
        # ),
        # est.NormalTarget("prevalence_smear_positive", target_data["prevalence_smear_positive"], 15.0),
        # est.NormalTarget("case_detection_rate", target_data["case_detection_rate"], 5.0),
    ]


def plot_spaghetti(
    spaghetti: pd.DataFrame,
    indicators: List[str],
    n_cols: int,
    target_data: Dict,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
) -> go.Figure:
    """Generate a spaghetti plot to compare any number of requested outputs to targets.

    Args:
        spaghetti: The values from the sampled runs
        indicators: The names of the indicators to look at
        n_cols: Number of columns for the figure
        targets: The calibration targets

    Returns:
        The spaghetti plot figure object
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [
            (indicator_names[ind] if ind in indicator_names else ind.replace("_", " "))
            for ind in indicators
        ],
    )
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)

        # Model outputs
        ind_spagh = spaghetti[ind]
        ind_spagh.columns = ind_spagh.columns.map(lambda col: f"{col[0]}, {col[1]}")
        fig.add_traces(px.line(ind_spagh).data, rows=row, cols=col)

        # Targets
        if ind in target_data.keys():
            target = target_data[ind]
            target_marker_config = dict(
                size=5.0, line=dict(width=0.5, color="DarkSlateGrey")
            )
            lines = go.Scatter(
                x=target.index,
                y=target,
                marker=target_marker_config,
                name="targets",
                mode="markers",
            )
            fig.add_trace(lines, row=row, col=col)
        fig.update_layout(yaxis4=dict(range=[0, 2.2]))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=None)
    return fig.update_xaxes(range=[plot_start_date, plot_end_date])


def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict,
    indicators: List[str],
    quantiles: List[float],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
    max_alpha: float = 0.7,
) -> go.Figure:
    """Plot the credible intervals with subplots for each output,
    for a single run of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type
        targets: Calibration targets
        output: User-requested output of interest
        analysis: The key for the analysis type
        quantiles: User-requested quantiles for the patches to be plotted over
        max_alpha: Maximum alpha value to use in patches

    Returns:
        The interactive figure
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [
            (indicator_names[ind] if ind in indicator_names else ind.replace("_", " "))
            for ind in indicators
        ],
    )
    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]
        for q, quant in enumerate(quantiles):
            alpha = (
                min((q, len(quantiles) - q)) / np.floor(len(quantiles) / 2) * max_alpha
            )
            fill_colour = f"rgba(0,30,180,{str(alpha)})"
            fig.add_traces(
                go.Scatter(
                    x=data.index,
                    y=data[quant],
                    fill="tonexty",
                    fillcolor=fill_colour,
                    line={"width": 0},
                    name=quant,
                ),
                rows=row,
                cols=col,
            )
        fig.add_traces(
            go.Scatter(
                x=data.index, y=data[0.5], line={"color": "black"}, name="median"
            ),
            rows=row,
            cols=col,
        )
        if ind in target_data.keys():
            target = target_data[ind]
            marker_format = {
                "size": 5.0,
                "color": "rgba(250, 135, 206, 0.2)",
                "line": {"width": 1.0},
            }
            fig.add_traces(
                go.Scatter(
                    x=target.index, y=target, mode="markers", marker=marker_format
                ),
                rows=row,
                cols=col,
            )

    fig.update_xaxes(range=[plot_start_date, plot_end_date])
    return fig.update_layout(yaxis4={"range": [0.0, 2.5]}, showlegend=False)
