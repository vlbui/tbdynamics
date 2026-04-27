import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import estival.priors as esp
from numpyro import distributions as dist
from scipy.stats import truncnorm, gaussian_kde
from jax import numpy as jnp
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers used by the manuscript-revision diagnostics notebook
# (Phase 3: per-parameter ESS/R-hat, corner plot, correlations, PPC,
#  recent-transmission-share metrics)
# ---------------------------------------------------------------------------


def _structural_param_names(idata: az.InferenceData) -> List[str]:
    """Return calibrated structural parameters (drop dispersions, etc.)."""
    return [
        v for v in idata.posterior.data_vars
        if "_dispersion" not in v
    ]


def report_calibration_diagnostics(
    idata: az.InferenceData,
    params_name: Optional[Dict[str, str]] = None,
    ess_threshold: int = 400,
    rhat_threshold: float = 1.05,
    exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a per-parameter diagnostics table (mean, 95% CrI, ESS, R-hat)
    and flag parameters with ESS < threshold or R-hat > threshold.

    Returns
    -------
    table : pd.DataFrame
        Indexed by descriptive name (or raw if no params_name provided);
        columns: mean, hdi_2.5%, hdi_97.5%, ess_bulk, ess_tail, r_hat,
        ess_flag (True if ess_bulk < threshold), rhat_flag.
    flagged : list
        Parameter names that fail either threshold.
    """
    summary = az.summary(idata, hdi_prob=0.95)
    drop_mask = summary.index.str.contains("_dispersion")
    if exclude:
        drop_mask = drop_mask | summary.index.isin(exclude)
    summary = summary[~drop_mask].copy()

    summary = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%",
                       "ess_bulk", "ess_tail", "r_hat"]]
    summary["ess_flag"] = summary["ess_bulk"] < ess_threshold
    summary["rhat_flag"] = summary["r_hat"] > rhat_threshold

    flagged = summary.index[summary["ess_flag"] | summary["rhat_flag"]].tolist()

    if params_name:
        summary.index = [params_name.get(v, v) for v in summary.index]
    summary.index.name = "Parameter"
    return summary, flagged


def plot_posterior_corner(
    idata: az.InferenceData,
    exclude: Optional[List[str]] = None,
    params_name: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (22, 22),
) -> plt.Figure:
    """Full posterior corner (pair) plot for calibrated structural parameters."""
    var_names = _structural_param_names(idata)
    if exclude:
        var_names = [v for v in var_names if v not in exclude]

    axes = az.plot_pair(
        idata,
        var_names=var_names,
        kind="kde",
        marginals=True,
        figsize=figsize,
        textsize=12,
    )
    if params_name:
        # Rewrite axis labels with descriptive names
        labels = [params_name.get(v, v) for v in var_names]
        n = len(var_names)
        for i in range(n):
            axes[n - 1, i].set_xlabel(labels[i], fontsize=11)
            axes[i, 0].set_ylabel(labels[i], fontsize=11)
    fig = plt.gcf()
    return fig


def compute_posterior_correlations(
    idata: az.InferenceData,
    exclude: Optional[List[str]] = None,
    threshold: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation matrix from posterior samples.

    Returns
    -------
    corr : pd.DataFrame
        Full correlation matrix.
    high_pairs : pd.DataFrame
        Pairs with |rho| > threshold, sorted by absolute correlation.
    """
    var_names = _structural_param_names(idata)
    if exclude:
        var_names = [v for v in var_names if v not in exclude]

    samples = {v: idata.posterior[v].values.flatten() for v in var_names}
    df = pd.DataFrame(samples)
    corr = df.corr()

    pairs = []
    for i, a in enumerate(var_names):
        for b in var_names[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > threshold:
                pairs.append({"param_a": a, "param_b": b, "rho": r})
    high_pairs = (
        pd.DataFrame(pairs)
        .assign(abs_rho=lambda d: d["rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .drop(columns="abs_rho")
        .reset_index(drop=True)
        if pairs else pd.DataFrame(columns=["param_a", "param_b", "rho"])
    )
    return corr, high_pairs


def run_ppc(
    idata_extract: az.InferenceData,
    bcm,
    target_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Run model on posterior samples and return predicted values at each
    target time-point for each named target.

    Returns
    -------
    dict mapping target_name -> DataFrame indexed by target time, columns = sample.
    """
    from estival.sampling import tools as esamp

    results = esamp.model_results_for_samples(idata_extract, bcm).results
    out = {}
    for tname in target_names:
        if tname == "log_notification":
            obs_name = "log_notification" if "log_notification" in results else "notification"
            df = results[obs_name]
            if obs_name == "notification":
                df = np.log(df)
        else:
            df = results[tname]
        # Restrict to observed time-points if target attached to bcm
        target_obj = next((t for t in bcm.targets if t.name == tname), None)
        if target_obj is not None:
            df = df.loc[df.index.intersection(target_obj.data.index)]
        out[tname] = df
    return out


def plot_ppc_panel(
    predicted: Dict[str, pd.DataFrame],
    bcm,
    n_cols: int = 2,
    figsize_per_panel: Tuple[float, float] = (7.0, 4.5),
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Plot PPC for each target: posterior interval + observed points.
    Returns figure plus a summary DataFrame with Bayesian p-values
    (P(simulated > observed)) at each observed time-point.
    """
    targets = {t.name: t for t in bcm.targets if t.name in predicted}
    n = len(targets)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    rows = []
    for i, (tname, target) in enumerate(targets.items()):
        ax = axes_flat[i]
        df = predicted[tname]                       # rows = time, cols = sample
        obs = target.data
        med = df.median(axis=1)
        lo = df.quantile(0.025, axis=1)
        hi = df.quantile(0.975, axis=1)

        ax.fill_between(med.index, lo, hi, alpha=0.25, label="95% CrI")
        ax.plot(med.index, med, lw=1.2, label="Posterior median")
        ax.scatter(obs.index, obs.values, color="red", zorder=5, label="Observed")
        ax.set_title(tname)
        ax.legend(fontsize=8)

        for t, obs_val in obs.items():
            if t in df.index:
                sims = df.loc[t]
                p = float((sims > obs_val).mean())
                rows.append({
                    "target": tname, "time": t,
                    "observed": float(obs_val),
                    "predicted_median": float(sims.median()),
                    "predicted_2.5%": float(sims.quantile(0.025)),
                    "predicted_97.5%": float(sims.quantile(0.975)),
                    "bayesian_p_value": p,
                })

    # Hide unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    plt.tight_layout()

    summary = pd.DataFrame(rows)
    return fig, summary


def compute_recent_transmission_metrics(
    spaghetti_results: pd.DataFrame,
    indicator: str = "incidence_early_perc",
    baseline_year: float = 2013.0,
    act3_window: Tuple[float, float] = (2014.0, 2018.0),
    rebound_threshold_pp: float = 5.0,
    horizon_end: float = 2035.0,
) -> Dict[str, pd.Series]:
    """
    Summary metrics for the recent-transmission share time-series.

    Parameters
    ----------
    spaghetti_results : DataFrame indexed by time, columns = posterior sample.
    indicator : output variable holding the recent-transmission share (%).
    baseline_year : pre-ACT3 reference year.
    act3_window : (start, end) of the ACT3 trial.
    rebound_threshold_pp : "rebound" defined as recovery to within N percentage
        points of baseline.

    Returns
    -------
    dict with per-sample Series of:
      baseline, min_during_act3, value_2yr_post, value_5yr_post, time_to_rebound.
      time_to_rebound is np.nan if rebound never occurs before horizon_end.
    """
    df = spaghetti_results
    if df.index.name is None:
        df.index.name = "time"

    def at(t):
        # nearest available index value (handles 0.1 step grid)
        idx = df.index.get_indexer([t], method="nearest")[0]
        return df.iloc[idx]

    baseline = at(baseline_year)
    act3_mask = (df.index >= act3_window[0]) & (df.index <= act3_window[1])
    min_during = df.loc[act3_mask].min(axis=0)
    val_2yr = at(act3_window[1] + 2.0)
    val_5yr = at(act3_window[1] + 5.0)

    # time-to-rebound: first time after act3_window[1] where recent share
    # returns within `rebound_threshold_pp` points of baseline.
    post = df.loc[(df.index > act3_window[1]) & (df.index <= horizon_end)]
    rebound_times = []
    for col in df.columns:
        s = post[col]
        b = baseline[col]
        target = b - rebound_threshold_pp
        rec = s[s >= target]
        rebound_times.append(rec.index[0] - act3_window[1] if len(rec) else np.nan)
    time_to_rebound = pd.Series(rebound_times, index=df.columns)

    return {
        "baseline": baseline,
        "min_during_act3": min_during,
        "value_2yr_post": val_2yr,
        "value_5yr_post": val_5yr,
        "time_to_rebound_years": time_to_rebound,
    }


def quantile_summary(series: pd.Series, qs=(0.025, 0.5, 0.975)) -> Dict[str, float]:
    """Median and 95% CrI of a sample-distributed scalar."""
    out = {f"q{int(q * 1000) / 10:g}": float(series.quantile(q)) for q in qs}
    out["mean"] = float(series.mean())
    return out


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
    elif isinstance(prior, esp.NormalPrior):  # Adding support for Normal priors
        return dist.Normal(loc=prior.mean, scale=prior.stdev), None
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
     # Force index order according to params_name values
    desired_order = [params_name.get(k, k) for k in params_name.keys()]
    table = table.reindex(desired_order).dropna(how="all")
    table.index.name = "Parameter"
    return table


def plot_post_prior_comparison(idata, priors, params_name, display_option=1):
    """
    Plot comparison of model posterior outputs against priors.

    Args:
        idata: Arviz inference data from calibration.
        priors: Dictionary of custom prior objects.
        params_name: Dictionary mapping parameter names to descriptive titles.
        display_option: 1=all, 2=only early_prop & late_reactivation, 3=all others

    Returns:
        The figure object.
    """
    # Filter priors to exclude those containing '_dispersion'
    req_vars = [
        var
        for var in priors.keys()
        if "_dispersion" not in var and var != "contact_reduction"
    ]

    if display_option == 2:
        req_vars = [v for v in req_vars if v in ["early_prop_adjuster", "late_reactivation_adjuster"]]
    elif display_option == 3:
        req_vars = [v for v in req_vars if v not in ["early_prop_adjuster", "late_reactivation_adjuster"]]

    num_vars = len(req_vars)
    num_rows = (num_vars + 1) // 2  # two columns

    fig, axs = plt.subplots(num_rows, 2, figsize=(28, 6.2 * num_rows))
    axs = axs.ravel()

    for i_ax, ax in enumerate(axs):
        if i_ax >= num_vars:
            ax.axis("off")
            continue

        var_name = req_vars[i_ax]
        posterior_samples = idata.posterior[var_name].values.flatten()

        # Handle log-transformed parameter (early_prop_adjuster only)
        transform_prior = False
        if var_name == "early_prop_adjuster":
            posterior_samples = np.exp(posterior_samples)
            transform_prior = True

        low_post = float(np.min(posterior_samples))
        high_post = float(np.max(posterior_samples))
        x_vals_posterior = np.linspace(low_post, high_post, 400)

        # Posterior KDE
        post_kde = gaussian_kde(posterior_samples)
        posterior_density = post_kde(x_vals_posterior)

        # Prior -> NumPyro
        numpyro_prior, prior_bounds = convert_prior_to_numpyro(priors[var_name])

        # --- Build a wide prior grid to avoid truncation ---
        prior_low = None
        prior_high = None
        N_PRIOR = 2000  # higher resolution for smooth tails

        if transform_prior:
            # Prior is defined in log-space; sample log-domain widely, then transform.
            lb, ub = prior_bounds if prior_bounds is not None else (-5.0, 5.0)
            x_vals_prior_log = np.linspace(lb, ub, max(N_PRIOR, 4000))
            x_vals_prior = np.exp(x_vals_prior_log)
            prior_density_log = np.exp(numpyro_prior.log_prob(x_vals_prior_log))
            prior_density = prior_density_log / x_vals_prior  # Jacobian
            prior_low, prior_high = float(x_vals_prior.min()), float(x_vals_prior.max())
        else:
            # If explicit bounds exist, use them.
            if prior_bounds is not None:
                prior_low, prior_high = map(float, prior_bounds)
                x_vals_prior = np.linspace(prior_low, prior_high, N_PRIOR)
            else:
                # Try distribution-specific wide support
                try:
                    # Prefer Normal handling (±6σ)
                    if isinstance(numpyro_prior, dist.Normal):
                        mu = float(numpyro_prior.loc)
                        sd = float(numpyro_prior.scale)
                        prior_low = mu - 6 * sd
                        prior_high = mu + 6 * sd
                    else:
                        # Use icdf if available to capture extreme tails robustly
                        if hasattr(numpyro_prior, "icdf"):
                            eps = 1e-4
                            prior_low = float(numpyro_prior.icdf(jnp.array([eps]))[0])
                            prior_high = float(numpyro_prior.icdf(jnp.array([1 - eps]))[0])
                        else:
                            raise AttributeError("icdf not available")
                except Exception:
                    # Fallback: expand posterior range by 20% on both sides
                    span = high_post - low_post if high_post > low_post else 1.0
                    prior_low = low_post - 0.2 * span
                    prior_high = high_post + 0.2 * span

                x_vals_prior = np.linspace(prior_low, prior_high, N_PRIOR)

            # Evaluate prior density over x_vals_prior
            prior_density = np.exp(numpyro_prior.log_prob(x_vals_prior))

        # --- Plot prior and posterior ---
        ax.fill_between(x_vals_prior, prior_density, color="k", alpha=0.2, linewidth=2, label="Prior")
        ax.fill_between(x_vals_posterior, 0, posterior_density, color="b", alpha=0.3, label="Posterior")

        # Title & ticks
        ax.set_title(params_name.get(var_name, var_name), fontsize=34, fontname="Arial")
        ax.tick_params(axis="both", labelsize=30, length=18)

        # Legend once
        if i_ax == 0:
            ax.legend(fontsize=24)

        # --- Axis limits: span BOTH prior and posterior ---
        # Force late_reactivation_adjuster to start at 0
        if var_name == "early_prop_adjuster":
            ax.set_xlim(0.8, 1.2)
        else:
            left_union = min(low_post, prior_low) if prior_low is not None else low_post
            right_union = max(high_post, prior_high) if prior_high is not None else high_post

            if var_name == "late_reactivation_adjuster":
                left_union = 0.5  # always start at 0

            # Guard against degenerate ranges
            if not np.isfinite(left_union): left_union = 0.0
            if not np.isfinite(right_union) or right_union <= left_union:
                right_union = left_union + 1.0

            ax.set_xlim(left_union, right_union)

    plt.tight_layout(h_pad=1.0, w_pad=5)
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
            if "_dispersion" in var or var == "contact_reduction" or var=="acf_sensitivity"
        ]
    )
    # Plot trace plots with the filtered parameters
    trace_fig = az.plot_trace(
        filtered_posterior, figsize=(28, 3.2 * len(filtered_posterior.data_vars))
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
            ax.set_title(title, fontsize=28, loc="center")  # Set title for each axis
            ax.tick_params(axis="both", labelsize=30, length=18)  # Increase tick label size

    plt.tight_layout()

    fig = plt.gcf()  # Get the current figure
    plt.close(fig)  # Close the figure to free memory but do not save it here

    return fig  # Return the figure object


def calculate_derived_metrics(death_rate, recovery_rate, natural_death_rate):
    """Calculate derived disease duration and CFR."""
    disease_duration = 1 / (death_rate + recovery_rate + natural_death_rate)

    cfr = (death_rate + natural_death_rate) / (
        recovery_rate + death_rate + natural_death_rate
    )
    return disease_duration, cfr


def sample_truncated_normal(mean, stdev, trunc_range, num_samples=1000000):
    """Sample from a truncated normal distribution."""
    a, b = (trunc_range[0] - mean) / stdev, (trunc_range[1] - mean) / stdev
    return truncnorm(a, b, loc=mean, scale=stdev).rvs(num_samples)


def process_idata_for_derived_metrics(idata, natural_death_rate):
    """
    Extract the necessary posterior samples from idata and calculate derived metrics.

    Args:
        idata: ArviZ InferenceData containing the posterior samples.
        natural_death_rate: all-cause mortality death rate at time point
        time_period: length of time to calculate CFR

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
        death_rate_pos, recovery_rate_pos, natural_death_rate
    )
    post_duration_neg, post_cfr_neg = calculate_derived_metrics(
        death_rate_neg, recovery_rate_neg, natural_death_rate
    )

    # Return dictionary of derived posterior metrics
    return {
        "post_duration_positive": post_duration_pos.flatten(),
        "post_cfr_positive": post_cfr_pos.flatten(),
        "post_duration_negative": post_duration_neg.flatten(),
        "post_cfr_negative": post_cfr_neg.flatten(),
    }


def process_priors_for_derived_metrics(priors, universal_death):
    """
    Process priors for both smear-positive and smear-negative TB cases and calculate derived metrics based on sampled death and recovery rates.

    Args:
        priors: Dictionary with prior mean, standard deviation, and truncation range for death and recovery rates for both positive and negative cases.
        universal_death: Universal death rate applicable to all cases.
        num_samples: Number of samples to generate from the distribution for each metric.

    Returns:
        Dictionary of numpy arrays containing derived metrics for each type of TB (positive and negative) for both duration and CFR.
    """
    prior_metrics = {}
    for key in [
        "duration_positive",
        "cfr_positive",
        "duration_negative",
        "cfr_negative",
    ]:
        case = "positive" if "positive" in key else "negative"
        death_rate_key = f"smear_{case}_death_rate"
        recovery_rate_key = f"smear_{case}_self_recovery"

        # Sample death rate and recovery rate
        samples_death_rate = sample_truncated_normal(
            priors[death_rate_key].mean,
            priors[death_rate_key].stdev,
            priors[death_rate_key].trunc_range,
        )
        samples_recovery_rate = sample_truncated_normal(
            priors[recovery_rate_key].mean,
            priors[recovery_rate_key].stdev,
            priors[recovery_rate_key].trunc_range,
        )

        # Calculate derived metrics (duration and CFR) for each sample
        metrics = []
        for death_rate, recovery_rate in zip(samples_death_rate, samples_recovery_rate):
            duration, cfr = calculate_derived_metrics(
                death_rate, recovery_rate, universal_death
            )
            metrics.append(duration if "duration" in key else cfr)

        prior_metrics[key] = np.array(metrics)

    return prior_metrics


# Integrated function to sample priors, calculate derived metrics, and plot both prior and posterior
def plot_derived_comparison(prior_metrics, posterior_metrics):
    """
    Plot comparison of derived outputs (disease duration and CFR) between priors and posteriors.

    Args:
        prior_metrics: Dictionary containing arrays of derived metrics from priors.
        posterior_metrics: Dictionary containing arrays of derived metrics from posteriors.

    Returns:
        Displays the figure and prints the derived metrics table (mean, 2.5% and 97.5% quantiles).
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
        "SPTB case fatality rate (%)",
        "SNTB disease duration (year)",
        "SNTB case fatality rate (%)",
    ]

    results = []

    for i_ax, ax in enumerate(axs):
        if i_ax < num_vars:
            var_name = derived_vars[i_ax]

            # Posterior samples
            posterior_samples = posterior_metrics[f"post_{var_name}"].flatten()
            low_post = np.min(posterior_samples)
            high_post = np.max(posterior_samples)
            x_vals_posterior = np.linspace(low_post, high_post, 100)
            post_kde = gaussian_kde(posterior_samples, bw_method="silverman")
            posterior_density = post_kde(x_vals_posterior)

            # Prior samples
            prior_samples = prior_metrics[var_name]
            low_prior = np.min(prior_samples)
            high_prior = np.max(prior_samples)
            x_vals_prior = np.linspace(low_prior, high_prior, 100)
            prior_kde = gaussian_kde(prior_samples)
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
            ax.fill_between(
                x_vals_posterior, 0, posterior_density, color="b", alpha=0.3, label="Posterior"
            )  # Fill under posterior

            # Set the title
            ax.set_title(plot_titles[i_ax], fontsize=34, fontname="Arial")
            ax.tick_params(axis="both", labelsize=24, length=18)

            # Calculate the mean and 95% CI from posterior samples
            mean_val = np.mean(posterior_samples)
            quantiles = np.percentile(posterior_samples, [2.5, 97.5])

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

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)
    print("\nDerived Metrics with Mean and 95% CI:")
    print(results_df)
    plt.show()
    return fig
