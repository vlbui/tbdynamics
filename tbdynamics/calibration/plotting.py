import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from typing import List, Dict

from tbdynamics.constants import (
    indicator_names,
    indicator_legends,
    quantiles,
    scenario_names,
)
from tbdynamics.tools.utils import get_row_col_for_subplots, get_standard_subplot_fig

# from tbdynamics.vietnam.calibration.utils import calculate_waic_comparison


# Define the custom template
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
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
) -> go.Figure:
    """Generate a spaghetti plot to compare any number of requested outputs with target points.

    Args:
        spaghetti: The values from the sampled runs.
        target_data: The calibration targets for each indicator.
        indicators: The names of the indicators to look at.
        n_cols: Number of columns for the figure.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.

    Returns:
        The spaghetti plot figure object.
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
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
    history: bool = False,  # New argument
    max_alpha: float = 0.7,
) -> go.Figure:
    """Plot the credible intervals with subplots for each output,
    for a single run of interest.

    Args:
        quantile_outputs: DataFrames containing derived outputs of interest for each analysis type.
        target_data: Calibration targets.
        indicators: List of indicators to plot.
        n_cols: Number of columns for the subplots.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.
        history: If True, set tick intervals to 50 years.

    Returns:
        The interactive Plotly figure.
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

        for q, quant in enumerate(quantiles):
            if quant not in filtered_data.columns:
                continue

            alpha = (
                min((quantiles.index(quant), len(quantiles) - quantiles.index(quant)))
                / (len(quantiles) / 2)
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
            if ind in ["total_population", "adults_prevalence_pulmonary"]
            else "purple"
        )

        # Plot the point estimates with error bars for indicators with uncertainty bounds
        if ind in [
            "prevalence_smear_positive",
            "adults_prevalence_pulmonary",
            "incidence",
        ]:
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
                    "adults_prevalence_pulmonary",
                    "incidence",
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
    indicator: str ='notification',
    plot_start_date: int = 2011,
    plot_end_date: int = 2024,
    max_alpha: float = 0.7,
) -> go.Figure:
    """
    Plot the "notification" indicator for each scenario in a 2x2 grid with subplot titles
    based on configuration keys, include target points, and show LOO-IC in the bottom left.

    Args:
        covid_outputs: Dictionary containing outputs for each scenario.
        target_data: Calibration targets.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        A Plotly figure with all scenarios plotted in a 2x2 grid, with LOO-IC values annotated.
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

        for q, quant in enumerate(quantiles):
            if quant not in filtered_data.columns:
                continue

            alpha = (
                min((quantiles.index(quant), len(quantiles) - quantiles.index(quant)))
                / (len(quantiles) / 2)
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
    if indicator=='notification':
        fig.update_yaxes(
            range=[0, 150000],
            showticklabels=True,
            # ticks="outside"
        )

    return fig


def plot_covid_configs_comparison_box(
    diff_quantiles: Dict[str, Dict[str, pd.DataFrame]], plot_type: str = "abs"
):
    """
    Plot the median differences with error bars indicating the range from 0.025 to 0.975 quantiles
    for given indicators across multiple years in a single plot.

    Args:
        diff_quantiles: A dictionary containing the calculated quantile differences (output from `calculate_diff_quantiles`).
        plot_type: "abs" for absolute differences, "rel" for relative differences.

    Returns:
        A Plotly figure with all indicators plotted together, each containing horizontal bars for multiple years.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    indicators = diff_quantiles[plot_type].keys()
    years = diff_quantiles[plot_type][list(indicators)[0]].index

    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    for ind in indicators:
        color = indicator_colors.get(
            ind, "rgba(0, 123, 255)"
        )  # Default to blue if not specified

        median_diffs = []
        lower_diffs = []
        upper_diffs = []
        for year in years:
            quantile_data = diff_quantiles[plot_type][ind].loc[year]
            median_diffs.append(round(quantile_data[0.5]))
            lower_diffs.append(round(quantile_data[0.025]))
            upper_diffs.append(round(quantile_data[0.975]))

        fig.add_trace(
            go.Bar(
                y=[str(int(year)) for year in years],  # Convert years to strings
                x=median_diffs,  # Median differences
                orientation="h",
                name=ind.replace("_", " ").capitalize(),
                marker=dict(color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        upper - median
                        for upper, median in zip(upper_diffs, median_diffs)
                    ],
                    arrayminus=[
                        median - lower
                        for median, lower in zip(median_diffs, lower_diffs)
                    ],
                    color="black",
                    thickness=1,
                    width=2,
                ),
            )
        )

    fig.update_layout(
        title={
            "text": "Reference: COVID-19 had no effect on TB notifications",
            "x": 0.08,
            "xanchor": "left",
            "yanchor": "top",
        },
        yaxis_title="",
        xaxis_title="",
        height=320,
        barmode="group",
        legend=dict(
            orientation="h",  # Horizontal orientation for the legend
            yanchor="bottom",  # Anchor the legend at the bottom
            y=-0.3,  # Move the legend below the x-axis
            xanchor="center",  # Center the legend horizontally
            x=0.5,
            font=dict(size=12),
            itemsizing="constant",
            traceorder="normal",
        ),
        margin=dict(l=20, r=5, t=30, b=40),
    )

    # Ensure the y-axis is visible by adjusting its properties
    fig.update_yaxes(
        tickvals=[str(int(year)) for year in reversed(years)],
        tickformat="d",
        showline=True,  # Ensure the line is shown
        linecolor="black",  # Set the color of the y-axis line
        linewidth=1,  # Adjust the width of the y-axis line
        mirror=True,  # Ensure the axis line is mirrored
        ticks="outside",  # Show ticks outside the plot
        categoryorder="array",
        categoryarray=[str(int(year)) for year in reversed(years)],
    )
    fig.update_xaxes(range=[0, None])  # Ensure x-axes start at zero for clarity

    return fig


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
    indicators = ["incidence", "mortality_raw"]
    baseline_key = "base_scenario"
    last_scenario_key = "increase_case_detection_by_12_0"
    no_transmission_key = "no_transmission"

    # Determine which scenarios to plot
    if plot_scenario_mode == 1:
        scenario_keys = [
            baseline_key,
            "increase_case_detection_by_2_0",
            "increase_case_detection_by_5_0",
            last_scenario_key,
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
    indicator_colors = {ind: colors[i % len(colors)] for i, ind in enumerate(indicators)}
    sdg_target_color = "purple"
    end_tb_target_color = "red"

    fig = make_subplots(
        rows=n_scenarios,
        cols=n_cols,
        shared_yaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        column_titles=[
            "<b>TB incidence (per 100,000 populations)</b>",
            "<b>TB deaths</b>",
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
            for quant in quantiles:
                if quant not in filtered_data.columns:
                    continue
                alpha = (
                    min((quantiles.index(quant), len(quantiles) - quantiles.index(quant)))
                    / (len(quantiles) / 2)
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

            # Add targets
            if indicator_name == "incidence":
                # Incidence targets
                fig.add_trace(
                    go.Scatter(
                        x=[2030.0],
                        y=[31.0],
                        mode="markers",
                        marker=dict(size=6.0, color=sdg_target_color),
                        name="2030 SDGs target",
                        showlegend=(scenario_idx == 0 and col_idx == 0),
                        legendgroup="Targets",
                        legendrank=2,
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[2035.0],
                        y=[15.0],
                        mode="markers",
                        marker=dict(size=6.0, color=end_tb_target_color),
                        name="2035 End TB target",
                        showlegend=(scenario_idx == 0 and col_idx == 0),
                        legendgroup="Targets",
                        legendrank=1,
                    ),
                    row=row,
                    col=col,
                )

            if indicator_name == "mortality_raw":
                # Mortality targets
                fig.add_trace(
                    go.Scatter(
                        x=[2030.0],
                        y=[1913],
                        mode="markers",
                        marker=dict(size=6.0, color=sdg_target_color),
                        name="2030 SDGs target",
                        showlegend=(scenario_idx == 0 and col_idx == 0),
                        legendgroup="Targets",
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[2035.0],
                        y=[957.0],
                        mode="markers",
                        marker=dict(size=6.0, color=end_tb_target_color),
                        name="2035 End TB target",
                        showlegend=(scenario_idx == 0 and col_idx == 0),
                        legendgroup="Targets",
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
    diff_quantiles: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    plot_type: str = "abs",
) -> go.Figure:
    """
    Plot the quantile differences for the fixed indicators across multiple scenarios.

    Args:
        diff_quantiles (dict): The quantile difference data structured as a dictionary.
        plot_type (str): "abs" for absolute differences, "rel" for relative differences.

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

    for i, indicator in enumerate(indicators):
        color = indicator_colors.get(indicator, "rgba(0, 123, 255)")

        # Extract data for the given indicator and plot_type
        scenarios = list(diff_quantiles.keys())  # Extract scenario names
        medians = []
        lower_errors = []
        upper_errors = []

        for scenario in scenarios:
            median_val = diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.500
            ]
            lower_val = diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.025
            ]
            upper_val = diff_quantiles[scenario][plot_type][indicator].loc[
                2035.0, 0.975
            ]

            medians.append(median_val)
            lower_errors.append(median_val - lower_val)
            upper_errors.append(upper_val - median_val)

        # Add trace for this indicator in the specified order
        fig.add_trace(
            go.Bar(
                y=[
                    scenario_names.get(
                        scenario, scenario.replace("_", " ").capitalize()
                    )
                    for scenario in scenarios
                ],  # Descriptive scenario names
                x=medians,  # Median values on x-axis
                orientation="h",
                marker=dict(color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=upper_errors,  # Upper bound error
                    arrayminus=lower_errors,  # Lower bound error
                    color="black",  # Black color for error bars
                    thickness=1,  # Thicker error bars
                    width=2,  # Wider error bars
                ),
                name=indicator.replace(
                    "_", " "
                ).capitalize(),  # Use indicator name for legend
            )
        )

    # Ensure traces are ordered according to indicators list
    fig.data = sorted(
        fig.data,
        key=lambda trace: indicators.index(trace.name.lower().replace(" ", "_")),
    )

    # Update layout with tight margins and ordered legend
    fig.update_layout(
        title={
            "text": "Reference: <i>'Status-quo'</i> scenario",
            "x": 0.99,
            "xanchor": "right",
            "yanchor": "top",
        },
        xaxis_title="",
        yaxis_title="",
        barmode="group",
        height=320,  # Adjust height based on the number of rows
        margin=dict(
            l=20, r=5, t=30, b=40
        ),  # Tight layout with more bottom margin for legend
        yaxis=dict(
            tickangle=-45,  # Rotate y-axis labels by 45 degrees
            categoryorder="array",  # Ensure the order follows the scenarios
            categoryarray=[
                scenario_names.get(scenario, scenario.replace("_", " ").capitalize())
                for scenario in reversed(scenarios)
            ],
            tickfont=dict(
                size=12,
                family="Arial",  # You can specify a font family if desired
                color="black",
                weight="bold",
            ),
        ),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Position the legend below the plot
            xanchor="center",
            x=0.5,
            itemsizing="constant",  # Consistent item sizing
            traceorder="normal",  # Keep the legend order as per the traces added
        ),
    )

    return fig
