import plotly.graph_objects as go
from typing import List
from pandas import DataFrame, Series
import numpy as np
import matplotlib as plt

def plot_model_vs_actual(
    modeled_df: DataFrame,
    actual_series: Series,
    modeled_column: str,
    y_axis_title: str,
    plot_title: str,
    actual_color: str = "red",
) -> None:
    """
    Plot modeled values alongside observed data with a fixed ``Year`` X-axis.

    Args:
        modeled_df: Modeled values to plot.
        actual_series: Observed data indexed by year.
        modeled_column: Column in ``modeled_df`` to plot.
        y_axis_title: Label for the Y-axis.
        plot_title: Title of the plot.
        actual_color: Marker color for observed data.
    """
    # Create a line trace for the modeled data
    line_trace = go.Scatter(
        x=modeled_df.index,
        y=modeled_df[modeled_column],
        mode="lines",
        name="Modeled Data",
    )

    # Create a scatter plot for the actual data
    scatter_trace = go.Scatter(
        x=actual_series.index,
        y=actual_series.values,
        mode="markers",
        marker=dict(color=actual_color),
        name="Actual Data",
    )

    # Combine the traces into one figure
    fig = go.Figure(data=[line_trace, scatter_trace])

    # Update the layout for the combined figure
    fig.update_layout(
        title=plot_title,
        title_x=0.5,
        xaxis_title="Year",  # X-axis title fixed as 'Year'
        yaxis_title=y_axis_title,
    )

    # Show the figure
    fig.show()

def get_mix_from_strat_props(
    within_strat: float,
    props: List[float],
) -> np.ndarray:
    """
    Generate a mixing matrix from stratification proportions and a
    within-stratum mixing parameter.

    Args:
        within_strat: Fraction of contacts occurring within the same stratum.
        props: Population share for each stratum.

    Returns:
        Mixing matrix with shape ``(n, n)`` where ``n`` is the number of strata.
    """
    n_strata = len(props)
    within_strat_component = np.eye(n_strata) * within_strat
    all_pop_component = np.stack([np.array(props)] * len(props)) * (1.0 - within_strat)
    return within_strat_component + all_pop_component





