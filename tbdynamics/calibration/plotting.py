import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from typing import List, Dict

from tbdynamics.constants import (
    QUANTILES,
    scenario_names,
)
from tbdynamics.tools.utils import get_row_col_for_subplots, get_standard_subplot_fig

# Define the custom template for Plotly
extended_layout = pio.templates["simple_white"].layout
# Update the layout with custom settings
extended_layout.update(
    xaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        title_font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=12,
            color="black",
        ),
        tickfont=dict(
            family="Arial", size=10, color="black"  # Set x-axis tick font to Arial
        ),
    ),
    yaxis=dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        ticks="outside",
        title_font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=10,
            color="black",
        ),
        tickfont=dict(
            family="Arial", size=10, color="black"  # Set y-axis tick font to Arial
        ),
    ),
    title=dict(
        font=dict(
            family="Arial",  # Use Arial Black for bold font
            size=12,
            color="black",
        )
    ),
    font=dict(family="Arial", size=12),  # General font settings for the figure
    legend=dict(
        font=dict(family="Arial", size=12, color="black")  # Set legend font to Arial
    ),
)
# Create a new template using the updated layout
custom_template = go.layout.Template(layout=extended_layout)
# Register the custom template
pio.templates["custom_template"] = custom_template
pio.templates.default = "custom_template"


def plot_spaghetti(
    spaghetti: pd.DataFrame,
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    indicator_names: Dict[str, str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
) -> go.Figure:
    """
    Generate a spaghetti plot to compare multiple sampled model outputs with calibration target points.

    Args:
        spaghetti (pd.DataFrame): The values from the sampled model runs.
        target_data (Dict[str, pd.Series]): The calibration targets for each indicator.
        indicators (List[str]): The list of indicator names to include in the plot.
        indicator_names (Dict[str, str]): Mapping of indicator codes to display names.
        n_cols (int): Number of columns for subplot arrangement.
        plot_start_date (int, optional): Start year for the plot. Defaults to 1800.
        plot_end_date (int, optional): End year for the plot. Defaults to 2035.

    Returns:
        go.Figure: The generated spaghetti plot figure object.
    """
    rows = int(np.ceil(len(indicators) / n_cols))

    # Apply conditional titles with bold formatting
    fig = get_standard_subplot_fig(
        rows,
        n_cols,
        [""] * len(indicators),  # Remove individual titles
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)  # Set font size for titles

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)

        # Filter data by date range
        ind_spagh = spaghetti[ind]
        filtered_data = ind_spagh[
            (ind_spagh.index >= plot_start_date) & (ind_spagh.index <= plot_end_date)
        ]
        point_color = (
            "red"
            if ind in ["total_population", "adults_prevalence_pulmonary"]
            else "purple"
        )

        # Plot each line with grey color and line width of 0.5
        for col_name in filtered_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[col_name],
                    line=dict(color="grey", width=0.5),
                    mode="lines",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # Plot target points for each indicator
        if ind in [
            "prevalence_smear_positive",
            "adults_prevalence_pulmonary",
            "incidence",
        ]:
            # Special case with target bounds
            target_series = target_data[f"{ind}_target"]
            filtered_target = target_series[
                (target_series.index >= plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 6.0, "color": point_color},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        else:
            # Plot point estimates for other indicators if available
            target_series = target_data[ind]
            filtered_target = target_series[
                (target_series.index >= plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target,
                    mode="markers",
                    marker={"size": 6.0, "color": point_color},
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), plot_start_date)
        x_max = filtered_data.index.max()
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

        # Update y-axis range dynamically for each subplot
        if ind in [
            "prevalence_smear_positive",
            "adults_prevalence_pulmonary",
            "incidence",
        ]:
            y_max = max(
                filtered_data.max().max(),
                target_data[f"{ind}_target"].max(),
                target_data[f"{ind}_lower_bound"].max(),
                target_data[f"{ind}_upper_bound"].max(),
            )
        else:
            y_max = max(filtered_data.max().max(), target_data[ind].max())

        y_min = 0
        y_range = y_max - y_min
        padding = 0.05 * y_range  # Consistent padding for all scenarios
        fig.update_yaxes(
            range=[y_min - padding, y_max + padding],
            title=dict(
                text=f"<b>{indicator_names.get(ind, ind.replace('_', ' ').capitalize())}</b>",
                font=dict(size=12),  # Adjust font size for better visibility
            ),
            row=row,
            col=col,
            title_standoff=0,  # Adds space between axis and title for better visibility
        )

    fig.update_layout(showlegend=False, margin=dict(l=10, r=5, t=30, b=40))

    return fig


def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    indicator_names: Dict[str, str],
    indicator_legends: Dict[str, str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
    history: bool = False,  # New argument
    max_alpha: float = 0.7,
    option: str = "vietnam",
) -> go.Figure:
    """
    Plot the credible intervals with subplots for each output, comparing model outputs with calibration targets.

    Args:
        quantile_outputs (Dict[str, pd.DataFrame]): DataFrames containing quantile-based outputs of interest.
        target_data (Dict[str, pd.Series]): The calibration targets for each indicator.
        indicators (List[str]): List of indicators to be plotted.
        indicator_names (Dict[str, str]): Mapping of indicator codes to display names.
        indicator_legends (Dict[str, str]): Mapping of indicator codes to legend labels.
        n_cols (int): Number of columns for the subplot arrangement.
        plot_start_date (int, optional): Start year for the plot. Defaults to 1800.
        plot_end_date (int, optional): End year for the plot. Defaults to 2035.
        history (bool, optional): If True, sets tick intervals to 50 years. Defaults to False.
        max_alpha (float, optional): Maximum alpha (transparency) value for credible interval shading. Defaults to 0.7.

    Returns:
        go.Figure: The generated interactive Plotly figure with credible intervals.
    """

    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [""] * len(indicators),  # Remove individual titles
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)  # Set font size for titles

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]

        # Set plot_start_date to 2005 if the indicator is "prevalence_smear_positive"
        current_plot_start_date = (
            2005 if ind == "prevalence_smear_positive" else plot_start_date
        )

        # Filter data by date range
        filtered_data = data[
            (data.index >= current_plot_start_date) & (data.index <= plot_end_date)
        ]

        for q, quant in enumerate(QUANTILES):
            if quant not in filtered_data.columns:
                continue

            alpha = (
                min((QUANTILES.index(quant), len(QUANTILES) - QUANTILES.index(quant)))
                / (len(QUANTILES) / 2)
                * max_alpha
            )
            fill_color = f"rgba(0,30,180,{alpha})"

            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[quant],
                    fill="tonexty",
                    fillcolor=fill_color,
                    line={"width": 0},
                    name=f"{quant}",
                    showlegend=False,  # Hide legend for quantile traces
                ),
                row=row,
                col=col,
            )

        # Plot the median line
        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[0.5],
                line={"color": "black"},
                name="Median",
                showlegend=False,  # Hide legend for median line
            ),
            row=row,
            col=col,
        )

        # Define point color based on the indicator type
        point_color = (
            "red"
            if ind in ["total_population", "notification", "percentage_latent_adults", "act3_trial_adults_pop", "act3_control_adults_pop"]
            else "purple"
        )

        # Plot the point estimates with error bars for indicators with uncertainty bounds
        if option == "camau":
            indi_with_range = ["percentage_latent_adults"]
        else:
            indi_with_range = [
                "prevalence_smear_positive",
                "adults_prevalence_pulmonary",
                "incidence",
            ]
        if ind in indi_with_range:
            target_series = target_data[f"{ind}_target"]
            lower_bound_series = target_data[f"{ind}_lower_bound"]
            upper_bound_series = target_data[f"{ind}_upper_bound"]

            filtered_target = target_series[
                (target_series.index >= current_plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            filtered_lower_bound = lower_bound_series[
                (lower_bound_series.index >= current_plot_start_date)
                & (lower_bound_series.index <= plot_end_date)
            ]
            filtered_upper_bound = upper_bound_series[
                (upper_bound_series.index >= current_plot_start_date)
                & (upper_bound_series.index <= plot_end_date)
            ]

            # Plot the point estimates with error bars
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 6.0, "color": point_color},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=filtered_upper_bound - filtered_target,
                        arrayminus=filtered_target - filtered_lower_bound,
                        color=point_color,
                        thickness=1,
                        width=2,
                    ),
                    name="",  # No name for legend
                    showlegend=False,  # Hide legend for point estimates
                ),
                row=row,
                col=col,
            )
        else:
            # For other indicators, just plot the point estimate if available
            if ind in target_data.keys():
                target = target_data[ind]
                filtered_target = target[
                    (target.index >= current_plot_start_date)
                    & (target.index <= plot_end_date)
                ]

                # Plot the target point estimates
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 6.0, "color": point_color},
                        name="",  # No name for legend
                        showlegend=False,  # Hide legend for point estimates
                    ),
                    row=row,
                    col=col,
                )

        # Add indicator legend as annotation at the bottom right of each subplot
        legend_text = indicator_legends.get(ind, "")
        if legend_text and not history:
            # Compute axis ID for the subplot
            axis_id = (row - 1) * n_cols + col
            # Determine xref and yref for the annotation
            if axis_id == 1:
                xref = "x domain"
                yref = "y domain"
            else:
                xref = f"x{axis_id} domain"
                yref = f"y{axis_id} domain"

            # Add the annotation with a red point before the legend text
            fig.add_annotation(
                text=f'<span style="color:{point_color}; font-size:12px">&#9679;</span> <span style="font-size:12px">{legend_text}</span>',
                x=0.98,  # Right end of the x-axis domain
                y=0.05,  # Bottom of the y-axis domain
                xref=xref,
                yref=yref,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                # font=dict(size=10),
                bordercolor="black",
                borderwidth=1,
            )

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), current_plot_start_date)
        x_max = filtered_data.index.max()
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

        # Update y-axis range dynamically for each subplot
        y_min = 0
        y_max = max(
            filtered_data.max().max(),
            (
                max(
                    [
                        filtered_target.max()
                        for filtered_target in [
                            filtered_target,
                            filtered_lower_bound,
                            filtered_upper_bound,
                        ]
                    ]
                )
                if ind
                in [
                    "prevalence_smear_positive",
                    # "adults_prevalence_pulmonary",
                    # "incidence",
                ]
                else (
                    filtered_target.max()
                    if ind in target_data.keys()
                    else float("-inf")
                )
            ),
        )
        y_range = y_max - y_min
        padding = 0.05 * y_range  # Consistent padding for all scenarios
        fig.update_yaxes(
            range=[y_min - padding, y_max + padding],
            title=dict(
                text=f"<b>{indicator_names.get(ind, ind.replace('_', ' ').capitalize())}</b>",
                font=dict(size=12),  # Adjust font size for better visibility
            ),
            row=row,
            col=col,
            title_standoff=0,  # Adds space between axis and title for better visibility
        )
        if option == "camau":
            for year in [2014, 2018]:
                fig.add_vline(
                    x=year,
                    line=dict(color="gray", width=1, dash="dash"),
                    row=row,
                    col=col,
                )

    tick_interval = 50 if history else 2  # Set tick interval based on history
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=tick_interval,  # Adjust tick increment
    )

    # Update layout for the whole figure
    fig.update_layout(
        xaxis_title="",
        # yaxis_title="",
        showlegend=False,
        margin=dict(l=10, r=5, t=5, b=40),
    )

    return fig


def plot_outputs_for_covid(
    covid_outputs: Dict[str, Dict[str, pd.DataFrame]],
    target_data: Dict[str, pd.Series],
    indicator: str = "notification",
    plot_start_date: int = 2011,
    plot_end_date: int = 2024,
    max_alpha: float = 0.7,
) -> go.Figure:
    """
    Plot the specified indicator (default: "notification") for each COVID-19 scenario in a 2x2 grid.

    Args:
        covid_outputs (Dict[str, Dict[str, pd.DataFrame]]): Dictionary containing model outputs for each scenario.
        target_data (Dict[str, pd.Series]): The calibration targets for each indicator.
        indicator (str, optional): The indicator to be plotted. Defaults to "notification".
        plot_start_date (int, optional): Start year for the plot. Defaults to 2011.
        plot_end_date (int, optional): End year for the plot. Defaults to 2024.
        max_alpha (float, optional): Maximum alpha (transparency) value for credible interval shading. Defaults to 0.7.

    Returns:
        go.Figure: A Plotly figure displaying all scenarios in a 2x2 grid.
    """

    # Custom titles for each subplot
    covid_titles = {
        "no_covid": "Assumption 1",
        "detection": "Assumption 2",
        "contact": "Assumption 3",
        "detection_and_contact": "Assumption 4",
    }

    # Calculate the LOO-IC for each scenario
    # waic_results = calculate_waic_comparison(covid_outputs)
    # Define the 2x2 grid
    n_cols = 2
    n_rows = int(np.ceil(len(covid_titles) / n_cols))

    # Create the subplot figure
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=True,
        vertical_spacing=0.1,
        horizontal_spacing=0.07,
        subplot_titles=[
            f"<b>{covid_titles.get(scenario_name, scenario_name.replace('_', ' ').capitalize())}</b>"
            for scenario_name in covid_titles.keys()
        ],
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)  # Set font size for titles

    # Loop through each scenario and plot it on the grid
    for i, (scenario_name, title) in enumerate(covid_titles.items()):
        row = i // n_cols + 1
        col = i % n_cols + 1
        quantile_outputs = covid_outputs[scenario_name]["indicator_outputs"]
        data = quantile_outputs[indicator]

        # Filter data by date range
        filtered_data = data[
            (data.index >= plot_start_date) & (data.index <= plot_end_date)
        ]

        for q, quant in enumerate(QUANTILES):
            if quant not in filtered_data.columns:
                continue

            alpha = (
                min((QUANTILES.index(quant), len(QUANTILES) - QUANTILES.index(quant)))
                / (len(QUANTILES) / 2)
                * max_alpha
            )
            fill_color = f"rgba(0,30,180,{alpha})"

            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[quant],
                    fill="tonexty",
                    fillcolor=fill_color,
                    line={"width": 0},
                    name=f"{scenario_name} {quant}",
                    showlegend=False,  # Disable legend to avoid clutter
                ),
                row=row,
                col=col,
            )

        # Plot the median line

        fig.add_trace(
            go.Scatter(
                x=filtered_data.index,
                y=filtered_data[0.5],
                line={"color": "black"},
                name=f"{scenario_name} median",
                showlegend=False,  # Disable legend to avoid clutter
            ),
            row=row,
            col=col,
        )

        # Add target points if available

        targets = target_data[indicator]
        fig.add_trace(
            go.Scatter(
                x=targets.index,
                y=targets,
                mode="markers",
                marker=dict(size=6.0, color="red"),
                name="Target",
                showlegend=False,  # Hide legend for targets
            ),
            row=row,
            col=col,
        )

        # Add WAIC annotation to the bottom left of the subplot
        # elpd_waic_value = (
        #     waic_results.loc[scenario_name, "elpd_waic"]
        #     if scenario_name in waic_results.index
        #     else "N/A"
        # )
        # fig.add_annotation(
        #     text=(
        #         f"ELPD-WAIC: {elpd_waic_value:.3f}"
        #         if elpd_waic_value != "N/A"
        #         else "ELPD-WAIC: N/A"
        #     ),
        #     xref=f"x{i+1}",  # Refers to the x-axis of the current subplot
        #     yref=f"y{i+1}",  # Refers to the y-axis of the current subplot
        #     x=plot_start_date + 0.5,  # Align the annotation with the start date
        #     y=3000,  # Place it near the bottom left
        #     showarrow=False,
        #     font=dict(size=12, color="black"),
        #     xanchor="left",
        #     yanchor="bottom",
        #     bordercolor="black",  # Set the border color
        #     borderwidth=1,  # Set the border width (1px here)
        # )

    # Update layout for the whole figure
    fig.update_layout(
        title="",
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        height=600,  # Set the figure height to 600 pixels
        margin=dict(l=10, r=5, t=30, b=40),
        font=dict(size=8),
    )

    # Update all x-axes to have the same range based on plot_start_date and plot_end_date
    fig.update_xaxes(
        range=[plot_start_date, plot_end_date],  # Set the x-axis range
        tickmode="linear",  # Set tick mode to linear
        dtick=2,  # Set the tick interval to 2 years
    )
    if indicator == "notification":
        fig.update_yaxes(
            range=[0, 150000],
            showticklabels=True,
            # ticks="outside"
        )

    return fig


def plot_covid_configs_comparison_box(
    diff_quantiles: Dict[str, Dict[str, pd.DataFrame]],
    plot_type: str = "abs",
    log_scale: bool = False,
) -> go.Figure:
    """
    Plot the median differences with error bars indicating the range from 0.025 to 0.975 quantiles
    for given indicators across multiple years in a single plot.

    Args:
        diff_quantiles: A dictionary containing the calculated quantile differences (output from `calculate_diff_quantiles`).
        plot_type: "abs" for absolute differences, "rel" for relative differences.
        log_scale: If True, use scatter points instead of bars, and adjust y-positions for better visibility.

    Returns:
        A Plotly figure with all indicators plotted together, each containing horizontal bars for multiple years.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    indicators = list(diff_quantiles[plot_type].keys())
    years = list(reversed(diff_quantiles[plot_type][indicators[0]].index))

    # Assign unique colors to indicators
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    # Create y-position mapping for spacing indicators within each year
    year_positions = {year: i for i, year in enumerate(years)}

    for i, ind in enumerate(indicators):
        color = indicator_colors.get(
            ind, "rgba(0, 123, 255)"
        )  # Default to blue if not specified

        median_diffs, lower_diffs, upper_diffs, y_positions = [], [], [], []

        for year in years:
            quantile_data = diff_quantiles[plot_type][ind].loc[year]
            median_val = quantile_data[0.5]
            lower_val = quantile_data[0.025]
            upper_val = quantile_data[0.975]

            median_diffs.append(median_val)
            lower_diffs.append(median_val - lower_val)
            upper_diffs.append(upper_val - median_val)

            # Adjust y-position for indicator separation within each year
            y_positions.append(year_positions[year] + (i * 0.2) - 0.1)


            # Use bar plot otherwise
        fig.add_trace(
                go.Bar(
                    x=median_diffs,
                    y=y_positions,
                    orientation="h",
                    name=ind.replace("_", " ").capitalize(),
                    marker=dict(color=color),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=upper_diffs,
                        arrayminus=lower_diffs,
                        color="black",
                        thickness=1,
                        width=2,
                    ),
                )
            )

    # Ensure proper year labeling while keeping original order
    fig.update_layout(
        title={
            "text": "<i>Reference: COVID-19 had no effect on TB notifications</i>",
            "x": 0.5,
            "xanchor": "right",
            "yanchor": "top",
        },
        yaxis_title="",
        xaxis=dict(
            title="",
            type="log" if log_scale else "linear",  # Apply log scale if log_scale is True
            tickmode="array" if not log_scale else "auto",
        ),
        height=320,
        barmode="group" if not log_scale else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=20, r=5, t=30, b=40),
    )

    # Set y-axis ticks and labels (single label per year)
    fig.update_yaxes(
        tickvals=list(year_positions.values()),
        ticktext=[f"<b>{int(year)}</b>" for year in years],
        tickformat="d",
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        categoryorder="array",
    )

    return fig


# The function now retains the original year order and properly spaces indicators within each year. 🚀


def hex_to_rgb(hex_color):
    """
    Convert hex color (e.g., '#636EFA') to an rgb color tuple (e.g., (99, 110, 250)).
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def plot_scenario_output_ranges_by_col(
    scenario_outputs,
    plot_start_date: float = 2025.0,
    plot_end_date: float = 2036.0,
    max_alpha: float = 0.7,
    plot_scenario_mode: int = 2,  # Controls which scenarios to plot
    **kwargs,
) -> go.Figure:
    """
    Plot the credible intervals for incidence and mortality_raw with scenarios as rows.
    Add the 0.5 quantile of the baseline scenario to all scenario plots.

    Args:
        scenario_outputs: Dictionary containing scenario outputs with scenario names as keys.
        plot_start_date: Start year for the plot as float.
        plot_end_date: End year for the plot as float.
        max_alpha: Maximum alpha value to use in patches.
        plot_scenario_mode: Controls plotting behavior:
            - 1: Plot baseline and the 3 increase case detection scenarios.
            - 2: Plot baseline and the increase case detection by 12 scenario.
            - 3: Plot only scenarios 1 and 2.
            - 4: Plot only the no-transmission scenario.

    Returns:
        The interactive Plotly figure.
    """
    indicators = ["incidence", "mortality"]
    baseline_key = "base_scenario"
    last_scenario_key = "increase_case_detection_by_12_0"
    no_transmission_key = "no_transmission"

    # Determine which scenarios to plot
    if plot_scenario_mode == 1:
        scenario_keys = [
            baseline_key,
            "increase_case_detection_by_2_0",
            "increase_case_detection_by_5_0",
            # last_scenario_key,
        ]
        y_axis_titles = [
            "<i>'Status-quo'</i> scenario",
            "Scenario 1",
            "Scenario 2",
            "Scenario 3",
        ]
        plot_height = 680
    elif plot_scenario_mode == 2:
        scenario_keys = [baseline_key, last_scenario_key]
        y_axis_titles = ["<i>'Status-quo'</i> scenario", "Scenario 3"]
        plot_height = 600
    elif plot_scenario_mode == 3:
        scenario_keys = [
            "increase_case_detection_by_2_0",
            "increase_case_detection_by_5_0",
        ]
        y_axis_titles = ["Scenario 1", "Scenario 2"]
        plot_height = 600
    elif plot_scenario_mode == 4:
        scenario_keys = [no_transmission_key]
        y_axis_titles = ["No Transmission"]
        plot_height = 360

    n_scenarios = len(scenario_keys)
    n_cols = 2
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    fig = make_subplots(
        rows=n_scenarios,
        cols=n_cols,
        shared_yaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        column_titles=[
            "<b>TB incidence (/100,000/y)</b>",
            "<b>TB mortality (/100,000/y)</b>",
        ],
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)

    for scenario_idx, scenario_key in enumerate(scenario_keys):
        quantile_outputs = (
            scenario_outputs[scenario_key]["quantiles"]
            if scenario_key == baseline_key
            else scenario_outputs[scenario_key]
        )

        baseline_outputs = scenario_outputs[baseline_key]["quantiles"]
        row = scenario_idx + 1
        display_name = y_axis_titles[scenario_idx]

        for col_idx, indicator_name in enumerate(indicators):
            col = col_idx + 1
            color = indicator_colors[indicator_name]
            data = quantile_outputs[indicator_name]
            baseline_data = baseline_outputs[indicator_name]

            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]
            baseline_filtered_data = baseline_data[
                (baseline_data.index >= plot_start_date)
                & (baseline_data.index <= plot_end_date)
            ]

            # Add quantile ranges
            for quant in QUANTILES:
                if quant not in filtered_data.columns:
                    continue
                alpha = (
                    min(
                        (
                            QUANTILES.index(quant),
                            len(QUANTILES) - QUANTILES.index(quant),
                        )
                    )
                    / (len(QUANTILES) / 2)
                    * max_alpha
                )
                fill_color = f"rgba({hex_to_rgb(color)[0]}, {hex_to_rgb(color)[1]}, {hex_to_rgb(color)[2]}, {alpha})"
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[quant],
                        fill="tonexty",
                        fillcolor=fill_color,
                        line={"width": 0},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Add median line for the scenario
            if 0.5 in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[0.5],
                        line={"color": color},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Add dashed line for baseline median
            if scenario_key != baseline_key and 0.5 in baseline_filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=baseline_filtered_data.index,
                        y=baseline_filtered_data[0.5],
                        mode="lines",
                        line=dict(dash="dash", color="gray", width=1.5),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            fig.update_yaxes(
                title_text=f"<b>{display_name}</b>",
                title_font=dict(size=12),
                row=row,
                col=1,
            )

            if row < n_scenarios:
                fig.update_xaxes(showticklabels=False, row=row, col=col)

    legend_defaults = dict(
        title="",
        orientation="v",
        yanchor="top",
        y=0.13,
        xanchor="right",
        x=0.9,
        bordercolor="black",
        borderwidth=1,
    )

    # Update legend with kwargs
    legend_config = {**legend_defaults, **kwargs}

    fig.update_layout(
        height=plot_height,
        title="",
        xaxis_title="",
        showlegend=True,
        legend=legend_config,
        margin=dict(l=20, r=5, t=30, b=40),
    )

    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=2,
    )

    return fig


def plot_detection_scenarios_comparison_box(
    diff_quantiles: dict,
    plot_type: str = "abs",
    log_scale: bool = False,
) -> go.Figure:
    """
    Plot the quantile differences for the fixed indicators across multiple scenarios.

    Args:
        diff_quantiles (dict): The quantile difference data structured as a dictionary.
        plot_type (str): "abs" for absolute differences, "rel" for relative differences.
        log_scale (bool): If True, use scatter points instead of bars.

    Returns:
        fig: A Plotly figure object.
    """
    # Fixed indicators
    indicators = ["cumulative_diseased", "cumulative_deaths"]
    colors = px.colors.qualitative.Plotly
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    fig = go.Figure()

    # Extract all scenario names and reverse their order
    scenarios = list(
        reversed(
            [
                scenario
                for scenario in diff_quantiles.keys()
                if scenario != "increase_case_detection_by_12_0"
            ]
        )
    )

    # Mapping scenarios to y-axis positions (ensuring only one label per scenario)
    scenario_positions = {scenario: i for i, scenario in enumerate(scenarios)}

    for i, indicator in enumerate(indicators):
        color = indicator_colors.get(indicator, "rgba(0, 123, 255)")

        medians, lower_errors, upper_errors, y_positions = [], [], [], []

        for scenario in scenarios:
            median_val = -diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.500
            ]
            lower_val = -diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.025
            ]
            upper_val = -diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.975
            ]
            medians.append(median_val)
            lower_errors.append(median_val - lower_val)
            upper_errors.append(upper_val - median_val)

            # Adjust y-position for indicator separation within each scenario
            y_positions.append(scenario_positions[scenario] + (i * 0.2) - 0.1)

       
        fig.add_trace(
                go.Bar(
                    x=medians,
                    y=y_positions,
                    orientation="h",
                    marker=dict(color=color),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=upper_errors,
                        arrayminus=lower_errors,
                        color="black",
                        thickness=1,
                        width=2,
                    ),
                    name=indicator.replace("_", " ").capitalize(),
                )
            )

    # Define y-axis labels (only one label per scenario)
    y_labels = [
        scenario_names.get(scenario, scenario.replace("_", " ").capitalize())
        for scenario in scenarios
    ]

    fig.update_layout(
        title={
            "text": "Reference: <i>Status-quo</i> scenario",
            "x": 0.36,
            "xanchor": "right",
            "yanchor": "top",
        },
        xaxis=dict(
            title="",
            type="log" if log_scale else "linear",  # Apply log scale if log_scale is True
            tickmode="array" if not log_scale else "auto",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(scenario_positions.values()),  # One label per scenario
            ticktext=y_labels,
            tickangle=-45,
            categoryorder="array",
            tickfont=dict(size=12, family="Arial", color="black", weight="bold"),
        ),
        height=200,
        margin=dict(l=50, r=5, t=30, b=40),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            itemsizing="constant",
            traceorder="normal",
        ),
    )
    return fig


def plot_trial_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    indicator_names: Dict[str, str],
    n_cols: int,
    share_y = True,
) -> go.Figure:
    """
    Plot the credible intervals with subplots for each output, comparing model outputs with calibration targets.
    The control arm is now represented using a box plot.

    Args:
        quantile_outputs (Dict[str, pd.DataFrame]): DataFrames containing quantile-based outputs.
        target_data (Dict[str, pd.Series]): The calibration targets for each indicator.
        indicators (List[str]): List of indicators to be plotted.
        indicator_names (Dict[str, str]): Mapping of indicator codes to display names.
        n_cols (int): Number of columns for the subplot arrangement.
        max_alpha (float, optional): Maximum alpha (transparency) value for credible interval shading. Defaults to 0.7.

    Returns:
        go.Figure: The generated interactive Plotly figure with credible intervals.
    """

    # Only plot these two indicators
    valid_indicators = [
        "acf_detectionXact3_trialXorgan_pulmonary_rate1",
        "acf_detectionXact3_controlXorgan_pulmonary_rate1",
    ]
    indicators = [ind for ind in indicators if ind in valid_indicators]

    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [""] * len(indicators),
        share_y
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)

        # Define the year range for extracting data
        if ind == "acf_detectionXact3_trialXorgan_pulmonary_rate1":
            year_start, year_end = 2014.5, 2018  # Years 1 to 4
            x_axis_range = [2014.5, 2018.5]  # X-axis from 2014 to 2019
        elif ind == "acf_detectionXact3_controlXorgan_pulmonary_rate1":
            year_start, year_end = 2017.5, 2018.0  # Extract data, but only plot 2018
            x_axis_range = [2017.5, 2018.5]  # X-axis from 2017 to 2019

        # Extract target data for the indicator within the full range
        y_max = 0  # Initialize maximum Y value

        if ind in target_data:
            target_series = target_data[ind]

            # Extract all points within the range
            filtered_target = target_series[
                (target_series.index >= year_start) & (target_series.index <= year_end)
            ]

            # Update max y-value for scaling
            if not filtered_target.empty:
                y_max = max(y_max, filtered_target.max())

            # Plot point estimates
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 6.0, "color": "red"},
                    name=f"{ind} target",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # Extract quantile output data
        if ind in quantile_outputs:
            data = quantile_outputs[ind]

            # Extract all points within the full range
            if ind == "acf_detectionXact3_trialXorgan_pulmonary_rate1":
                filtered_data = data.loc[data.index.isin([2015, 2016, 2017, 2018])]
            elif ind == "acf_detectionXact3_controlXorgan_pulmonary_rate1":
                filtered_data = data.loc[data.index == 2018]

            # if ind == "acf_detectionXact3_controlXorgan_pulmonary":
            # Box plot for control arm
            fig.add_trace(
                go.Box(
                    x=filtered_data.index,
                    lowerfence=filtered_data[0.025],
                    q1=filtered_data[0.25],
                    median=filtered_data[0.5],
                    q3=filtered_data[0.75],
                    upperfence=filtered_data[0.975],
                    boxpoints="all",  # Show all points within the whiskers
                    # jitter=0.3,  # Add slight horizontal jitter to separate points
                    marker={"color": "rgba(0,30,180,0.5)"},
                    name="Box Plot Control Arm",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Update max y-value for scaling
            if not filtered_data.empty:
                y_max = max(y_max, filtered_data.max().max())

        # Set x-axis range based on indicator type
        fig.update_xaxes(
            range=x_axis_range,  # Set correct x-axis range
            tickmode="linear",  # Keep actual year ticks (2015, 2016, etc.)
            dtick=1,
            row=row,
            col=col,
        )

        # Set y-axis range from 0 to max value with padding
        y_range = [0, y_max * 1.1] if y_max > 0 else [0, 1]
        fig.update_yaxes(
            range=y_range,
            title=dict(
                text=f"<b>{indicator_names.get(ind, ind.replace('_', ' ').capitalize())}</b>",
                font=dict(size=12),
            ),
            row=row,
            col=col,
            # title_standoff=0,
        )

    fig.update_layout(
        xaxis_title="",
        showlegend=False,
        margin=dict(l=10, r=5, t=5, b=40),
    )

    return fig

def plot_output_ranges_box(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    indicator_names: Dict[str, str],
    indicator_legends: Dict[str, str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
    history: bool = False,
    option: str = "vietnam",
) -> go.Figure:
    """
    Plot annual box plots of model outputs and compare them with calibration targets.

    Args:
        quantile_outputs, target_data, indicators, indicator_names, indicator_legends, n_cols, etc.

    Returns:
        go.Figure: Plotly figure with box plots and target values.
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        [""] * len(indicators),
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=12)

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]

        current_plot_start_date = (
            2005 if ind == "prevalence_smear_positive" else plot_start_date
        )
        filtered_data = data[
            (data.index >= current_plot_start_date) & (data.index <= plot_end_date)
        ]

        # Generate annual box plots
        for year in range(current_plot_start_date, plot_end_date + 1):
            year_data = filtered_data[filtered_data.index.year == year]
            if year_data.empty:
                continue

            box_values = year_data[[q for q in QUANTILES if q in year_data.columns]].values.flatten()
            box_values = box_values[~np.isnan(box_values)]

            fig.add_trace(
                go.Box(
                    y=box_values,
                    x=[year] * len(box_values),
                    name=str(year),
                    boxpoints=False,
                    line_color="rgba(0,30,180,1)",
                    fillcolor="rgba(0,30,180,0.5)",
                    marker_color="rgba(0,30,180,0.7)",
                    showlegend=False
                ),
                row=row,
                col=col,
            )

        point_color = (
            "red"
            if ind in ["total_population", "notification", "percentage_latent_adults", "act3_trial_adults_pop", "act3_control_adults_pop"]
            else "purple"
        )

        if option == "camau":
            indi_with_range = ["percentage_latent_adults"]
        else:
            indi_with_range = [
                "prevalence_smear_positive",
                "adults_prevalence_pulmonary",
                "incidence",
            ]

        if ind in indi_with_range:
            target_series = target_data[f"{ind}_target"]
            lower_bound_series = target_data[f"{ind}_lower_bound"]
            upper_bound_series = target_data[f"{ind}_upper_bound"]

            filtered_target = target_series[
                (target_series.index >= current_plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            filtered_lower_bound = lower_bound_series[
                (lower_bound_series.index >= current_plot_start_date)
                & (lower_bound_series.index <= plot_end_date)
            ]
            filtered_upper_bound = upper_bound_series[
                (upper_bound_series.index >= current_plot_start_date)
                & (upper_bound_series.index <= plot_end_date)
            ]

            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 6.0, "color": point_color},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=filtered_upper_bound - filtered_target,
                        arrayminus=filtered_target - filtered_lower_bound,
                        color=point_color,
                        thickness=1,
                        width=2,
                    ),
                    name="",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        else:
            if ind in target_data.keys():
                target = target_data[ind]
                filtered_target = target[
                    (target.index >= current_plot_start_date)
                    & (target.index <= plot_end_date)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 6.0, "color": point_color},
                        name="",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        legend_text = indicator_legends.get(ind, "")
        if legend_text and not history:
            axis_id = (row - 1) * n_cols + col
            xref = "x domain" if axis_id == 1 else f"x{axis_id} domain"
            yref = "y domain" if axis_id == 1 else f"y{axis_id} domain"

            fig.add_annotation(
                text=f'<span style="color:{point_color}; font-size:12px">&#9679;</span> <span style="font-size:12px">{legend_text}</span>',
                x=0.98,
                y=0.05,
                xref=xref,
                yref=yref,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
            )

        # Update axes
        x_min = current_plot_start_date
        x_max = plot_end_date
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

        y_min = 0
        y_max = max(
            filtered_data.max().max(),
            (
                max(
                    [
                        filtered_target.max()
                        for filtered_target in [
                            filtered_target,
                            filtered_lower_bound,
                            filtered_upper_bound,
                        ]
                    ]
                )
                if ind in indi_with_range
                else (
                    filtered_target.max()
                    if ind in target_data.keys()
                    else float("-inf")
                )
            ),
        )
        y_range = y_max - y_min
        padding = 0.05 * y_range
        fig.update_yaxes(
            range=[y_min - padding, y_max + padding],
            title=dict(
                text=f"<b>{indicator_names.get(ind, ind.replace('_', ' ').capitalize())}</b>",
                font=dict(size=12),
            ),
            row=row,
            col=col,
            title_standoff=0,
        )

    tick_interval = 50 if history else 2
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=tick_interval,
    )
    fig.update_layout(
        xaxis_title="",
        showlegend=False,
        margin=dict(l=10, r=5, t=5, b=40),
    )
    return fig