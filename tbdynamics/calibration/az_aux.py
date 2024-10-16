import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import estival.priors as esp
from numpyro import distributions as dist
from scipy.stats import truncnorm, gaussian_kde


def convert_prior_to_numpyro(prior):
    """
    Converts a given custom prior to a corresponding Numpyro distribution and its bounds based on its type.

    Args:
        prior: A custom prior object.

    Returns:
        A tuple of (Numpyro distribution, bounds).
    """
    if isinstance(prior, esp.UniformPrior):
        return dist.Uniform(low=prior.start, high=prior.end), (prior.start, prior.end)
    elif isinstance(prior, esp.TruncNormalPrior):
        return dist.TruncatedNormal(
            loc=prior.mean,
            scale=prior.stdev,
            low=prior.trunc_range[0],
            high=prior.trunc_range[1],
        ), (prior.trunc_range[0], prior.trunc_range[1])
    elif isinstance(prior, esp.GammaPrior):
        rate = 1.0 / prior.scale
        return dist.Gamma(concentration=prior.shape, rate=rate), None
    elif isinstance(prior, esp.BetaPrior):
        return dist.Beta(concentration1=prior.a, concentration0=prior.b), (0, 1)
    else:
        raise TypeError(f"Unsupported prior type: {type(prior).__name__}")


def convert_all_priors_to_numpyro(priors):
    """
    Converts a dictionary of custom priors to a dictionary of corresponding Numpyro distributions.

    Args:
        priors: Dictionary of custom prior objects.

    Returns:
        Dictionary of Numpyro distributions.
    """
    numpyro_priors = {}
    for key, prior in priors.items():
        numpyro_prior, _ = convert_prior_to_numpyro(prior)
        numpyro_priors[key] = numpyro_prior
    return numpyro_priors


def tabulate_calib_results(idata: az.InferenceData, params_name) -> pd.DataFrame:
    """
    Get tabular outputs from calibration inference object,
    except for the dispersion parameters, and standardize formatting.

    Args:
        idata: InferenceData object from ArviZ containing calibration outputs.
        priors: List of parameter names as strings.

    Returns:
        Calibration results table in standard format.
    """
    # Generate summary table
    table = az.summary(idata)

    # Filter out dispersion parameters
    table = table[
        ~(
            table.index.str.contains("_dispersion")
            | (table.index == "contact_reduction")
        )
    ]

    # Round and format the relevant columns
    for col_to_round in [
        "mean",
        "sd",
        "hdi_3%",
        "hdi_97%",
        "ess_bulk",
        "ess_tail",
        "r_hat",
    ]:
        table[col_to_round] = table.apply(
            lambda x: str(round(x[col_to_round], 3)), axis=1
        )

    # Create the HDI column
    table["hdi"] = table.apply(lambda x: f'{x["hdi_3%"]} to {x["hdi_97%"]}', axis=1)

    # Drop unnecessary columns
    table = table.drop(["mcse_mean", "mcse_sd", "hdi_3%", "hdi_97%"], axis=1)

    # Rename columns for standardized format
    table.columns = [
        "Mean",
        "Standard deviation",
        "ESS bulk",
        "ESS tail",
        "\\textit{\^{R}}",
        "High-density interval",
    ]
    table.index = table.index.map(lambda x: params_name.get(x, x))
    table.index.name = "Parameter"
    return table


def plot_post_prior_comparison(idata, priors, params_name):
    """
    Plot comparison of model posterior outputs against priors.

    Args:
        idata: Arviz inference data from calibration.
        priors: Dictionary of custom prior objects.
        params_name: Dictionary mapping parameter names to descriptive titles.

    Returns:
        The figure object.
    """
    # Filter priors to exclude those containing '_dispersion'
    req_vars = [
        var
        for var in priors.keys()
        if "_dispersion" not in var and var != "contact_reduction"
    ]
    num_vars = len(req_vars)
    num_rows = (num_vars + 1) // 2  # Ensure even distribution across two columns

    # Set figure size to match A4 page width (8.27 inches) in portrait mode and adjust height based on rows
    fig, axs = plt.subplots(
        num_rows, 2, figsize=(28, 6.2 * num_rows)
    )  # A4 width in portrait mode
    axs = axs.ravel()

    for i_ax, ax in enumerate(axs):
        if i_ax < num_vars:
            var_name = req_vars[i_ax]
            posterior_samples = idata.posterior[var_name].values.flatten()
            low_post = np.min(posterior_samples)
            high_post = np.max(posterior_samples)
            x_vals_posterior = np.linspace(low_post, high_post, 100)

            # Use gaussian_kde to estimate the posterior density
            post_kde = gaussian_kde(posterior_samples)
            posterior_density = post_kde(x_vals_posterior)

            # Convert the prior to a Numpyro distribution
            numpyro_prior, prior_bounds = convert_prior_to_numpyro(priors[var_name])
            if prior_bounds:
                low_prior, high_prior = prior_bounds
                x_vals_prior = np.linspace(low_prior, high_prior, 100)
            else:
                x_vals_prior = (
                    x_vals_posterior  # Fallback if no specific prior bounds are given
                )

            # Compute the prior density using Numpyro
            prior_density = np.exp(numpyro_prior.log_prob(x_vals_prior))

            # Plot the prior density
            ax.fill_between(
                x_vals_prior,
                prior_density,
                color="k",
                alpha=0.2,
                linewidth=2,
                label="Prior",
            )

            # Plot the posterior density (smooth with KDE)
            ax.plot(
                x_vals_posterior,
                posterior_density,
                color="b",
                linewidth=1,
                linestyle="solid",
                label="Posterior",
            )

            # Set the title using the descriptive name from params_name
            title = params_name.get(
                var_name, var_name
            )  # Use var_name if not in params_name
            ax.set_title(title, fontsize=30, fontname="Arial")  # Set title to Arial 30
            ax.tick_params(axis="both", labelsize=24)

            # Add legend to the first subplot
            if i_ax == 0:
                ax.legend(fontsize=24)
        else:
            ax.axis("off")  # Turn off empty subplots if the number of req_vars is odd

    # Adjust padding and spacing
    plt.tight_layout(
        h_pad=1.0, w_pad=5
    )  # Increase padding between plots for better fit
    return fig


def plot_trace(idata: az.InferenceData, params_name: dict):
    """
    Plot trace plots for the InferenceData object, excluding parameters containing '_dispersion'.
    Adds descriptive titles from `params_name`.

    Args:
        idata: InferenceData object from ArviZ containing calibration outputs.
        params_name: Dictionary mapping parameter names to descriptive titles.

    Returns:
        A Matplotlib figure object containing the trace plots.
    """
    # Filter out parameters containing '_dispersion' and 'contact_reduction'
    filtered_posterior = idata.posterior.drop_vars(
        [
            var
            for var in idata.posterior.data_vars
            if "_dispersion" in var or var == "contact_reduction"
        ]
    )

    # Plot trace plots with the filtered parameters
    trace_fig = az.plot_trace(
        filtered_posterior, figsize=(28, 3.1 * len(filtered_posterior.data_vars))
    )

    # Set titles for each row of plots
    var_names = list(
        filtered_posterior.data_vars.keys()
    )  # Get the list of variable names
    for i, var_name in enumerate(var_names):
        for ax in trace_fig[i]:
            title = params_name.get(
                var_name, var_name
            )  # Get the title from params_name or default to var_name
            ax.set_title(title, fontsize=30, loc="center")  # Set title for each axis

    plt.tight_layout()

    fig = plt.gcf()  # Get the current figure
    plt.close(fig)  # Close the figure to free memory but do not save it here

    return fig  # Return the figure object


def calculate_derived_metrics(
    death_rate, recovery_rate, natural_death_rate, time_period
):
    """Calculate derived disease duration and CFR."""
    disease_duration = 1 / (death_rate + recovery_rate + natural_death_rate)
    term1 = (recovery_rate / (recovery_rate + death_rate)) * np.exp(
        -natural_death_rate * time_period
    )
    term2 = (death_rate / (recovery_rate + death_rate)) * np.exp(
        -(recovery_rate + death_rate + natural_death_rate) * time_period
    )
    cfr = 1 - term1 - term2
    return disease_duration, cfr


def process_idata_for_derived_metrics(idata, universal_death):
    """
    Extract the necessary posterior samples from idata and calculate derived metrics.

    Args:
        idata: ArviZ InferenceData containing the posterior samples.

    Returns:
        A dictionary containing the derived metrics for both smear-positive and smear-negative cases.
    """
    # Extract posterior samples
    death_rate_pos = idata.posterior["smear_positive_death_rate"].values
    recovery_rate_pos = idata.posterior["smear_positive_self_recovery"].values
    death_rate_neg = idata.posterior["smear_negative_death_rate"].values
    recovery_rate_neg = idata.posterior["smear_negative_self_recovery"].values

    # Calculate derived metrics for smear-positive and smear-negative cases
    post_duration_pos, post_cfr_pos = calculate_derived_metrics(
        death_rate_pos, recovery_rate_pos, universal_death, 1
    )
    post_duration_neg, post_cfr_neg = calculate_derived_metrics(
        death_rate_neg, recovery_rate_neg, universal_death, 1
    )

    # Return dictionary of derived posterior metrics
    return {
        "post_duration_positive": post_duration_pos.flatten(),
        "post_cfr_positive": post_cfr_pos.flatten(),
        "post_duration_negative": post_duration_neg.flatten(),
        "post_cfr_negative": post_cfr_neg.flatten(),
    }


def sample_truncated_normal(mean, stdev, trunc_range, num_samples=100000):
    """Sample from a truncated normal distribution."""
    a, b = (trunc_range[0] - mean) / stdev, (trunc_range[1] - mean) / stdev
    return truncnorm(a, b, loc=mean, scale=stdev).rvs(num_samples)


# Integrated function to sample priors, calculate derived metrics, and plot both prior and posterior
def plot_derived_comparison(priors, posterior_metrics, params_name, num_samples=10000):
    """
    Plot comparison of derived outputs (disease duration and CFR) between priors and posteriors.

    Args:
        priors: Dictionary of prior distributions for the derived metrics.
        posterior_metrics: Dictionary of posterior samples for the derived metrics.
        params_name: Dictionary mapping parameter names to descriptive titles.
        num_samples: Number of samples to draw from each prior.

    Returns:
        The figure object and prints the derived metrics table (mean, 2.5% and 97.5% quantiles).
    """
    # Derived parameters to compare
    derived_vars = [
        "duration_positive",
        "cfr_positive",
        "duration_negative",
        "cfr_negative",
    ]
    num_vars = len(derived_vars)
    num_rows = (num_vars + 1) // 2  # Even distribution across two columns

    # Set up plot for derived metrics
    fig, axs = plt.subplots(num_rows, 2, figsize=(28, 6.2 * num_rows))
    axs = axs.ravel()

    # Titles for the plots
    plot_titles = [
        "SPTB disease duration (year)",
        "One-year SPTB case fatality rate (%)",
        "SNTB disease duration (year)",
        "One-year SPTB case fatality rate (%)",
    ]

    results = []

    # Define universal death rate (assumed)
    universal_death = 0.01  # You need to adjust this value according to your model

    for i_ax, ax in enumerate(axs):
        if i_ax < num_vars:
            var_name = derived_vars[i_ax]

            # Posterior samples
            posterior_samples = posterior_metrics[f"post_{var_name}"].flatten()
            low_post = np.min(posterior_samples)
            high_post = np.max(posterior_samples)
            x_vals_posterior = np.linspace(low_post, high_post, 100)
            post_kde = gaussian_kde(posterior_samples)
            posterior_density = post_kde(x_vals_posterior)

            # Sample from the priors and calculate derived metrics
            case = "positive" if "positive" in var_name else "negative"
            death_rate_key = f"smear_{case}_death_rate"
            recovery_rate_key = f"smear_{case}_self_recovery"

            # Sample death rate and recovery rate
            samples_death_rate = sample_truncated_normal(
                priors[death_rate_key].mean,
                priors[death_rate_key].stdev,
                priors[death_rate_key].trunc_range,
                num_samples,
            )
            recovery_rate = sample_truncated_normal(
                priors[recovery_rate_key].mean,
                priors[recovery_rate_key].stdev,
                priors[recovery_rate_key].trunc_range,
                num_samples,
            )

            # Calculate derived metrics (duration and CFR) for each sample
            derived_metrics = []
            for death_rate, rec_rate in zip(samples_death_rate, recovery_rate):
                duration, cfr = calculate_derived_metrics(
                    death_rate, rec_rate, universal_death, 1
                )
                derived_metrics.append(duration if "duration" in var_name else cfr)

            derived_metrics = np.array(derived_metrics)

            # Prior density calculation
            prior_kde = gaussian_kde(derived_metrics)
            low_prior = np.min(derived_metrics)
            high_prior = np.max(derived_metrics)
            x_vals_prior = np.linspace(low_prior, high_prior, 100)
            prior_density = prior_kde(x_vals_prior)

            # Plot prior and posterior distributions
            ax.fill_between(
                x_vals_prior,
                prior_density,
                color="k",
                alpha=0.2,
                linewidth=2,
                label="Prior",
            )
            ax.plot(
                x_vals_posterior,
                posterior_density,
                color="b",
                linewidth=1,
                linestyle="solid",
                label="Posterior",
            )

            # Set the title using the descriptive name from params_name
            ax.set_title(plot_titles[i_ax], fontsize=30, fontname="Arial")
            ax.tick_params(axis="both", labelsize=24)

            # Calculate the mean and 95% CI (0.025 and 0.975 quantiles)
            mean_val = np.mean(derived_metrics)
            quantiles = np.percentile(derived_metrics, [2.5, 97.5])

            # Append the results to the table list
            results.append(
                {
                    "Metric": plot_titles[i_ax],
                    "Mean": f"{mean_val:.3f}",
                    "2.5% Quantile": f"{quantiles[0]:.3f}",
                    "97.5% Quantile": f"{quantiles[1]:.3f}",
                }
            )

            # Add legend to the first subplot
            if i_ax == 0:
                ax.legend(fontsize=24)

        else:
            ax.axis("off")  # Turn off empty subplots if there are extra axes

    # Adjust padding and spacing
    plt.tight_layout()
    plt.show()

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)
    print("\nDerived Metrics with Mean and 95% CI:")
    print(results_df)
