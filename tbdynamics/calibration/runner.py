import pymc as pm
from pathlib import Path
import multiprocessing
from estival.wrappers import pymc as epm
from estival.sampling import tools as esamp
from estival.wrappers import nevergrad as eng
from estival.utils.parallel import map_parallel
import nevergrad as ng
from tbdynamics.camau.calibration.utils import get_bcm
from estival.utils.sample import SampleTypes
from typing import Optional, Dict, Any, Union
import warnings


def calibrate(
    out_path: Path,
    params: Dict[str, Union[float, int]],
    covid_effects: Dict[str, bool],
    budget:int = 1000,
    n_chains: Optional[int] = None,
    draws: int = 100000,
    tune: int = 50000,
    save_results: bool = True
) -> Optional[Any]:
    """
    Run Bayesian calibration using a specified epidemiological model.

    Args:
        out_path (Path): Directory to save output files.
        params (Dict[str, Union[float, int]]): Initial parameter values for the model.
        covid_effects (Dict[str, bool]): Effects of COVID-19 on detection/contact.
        n_chains (Optional[int]): Number of MCMC chains. Defaults to half CPU cores.
        draws (int): Total number of MCMC samples per chain.
        tune (int): Number of tuning steps for each chain.
        save_results (bool): If True, saves results as NetCDF to 'calib_full_out.nc'. If False, returns the result.

    Returns:
        Optional[Any]: Returns InferenceData if save_results=False; otherwise None.
    """
    cpu_count = multiprocessing.cpu_count()

    if n_chains is None:
        n_chains = max(1, cpu_count // 2)
    elif n_chains > cpu_count:
        warnings.warn(f"Requested n_chains={n_chains} exceeds available CPUs ({cpu_count}). Setting n_chains={cpu_count}.")
        n_chains = cpu_count

    bcm = get_bcm(params, covid_effects)

    def optimize_ng_with_idx(item):
        idx, sample = item
        opt = eng.optimize_model(
            bcm,
            budget=budget,
            opt_class=ng.optimizers.TwoPointsDE,
            suggested=sample,
            num_workers=cpu_count
        )
        rec = opt.minimize(budget)
        return idx, rec.value[1]

    lhs_samples = bcm.sample.lhs(n_chains * 2)
    lhs_lle = esamp.likelihood_extras_for_samples(lhs_samples, bcm)
    lhs_sorted = lhs_lle.sort_values("loglikelihood", ascending=False)
    opt_samples_idx = map_parallel(optimize_ng_with_idx, lhs_sorted.iterrows())
    best_opt_samps = bcm.sample.convert(opt_samples_idx)
    init_samps = best_opt_samps.convert(SampleTypes.LIST_OF_DICTS)[0:n_chains]

    with pm.Model() as pm_model:
        variables = epm.use_model(bcm)
        idata_raw = pm.sample(
            step=[pm.DEMetropolisZ(variables, proposal_dist=pm.NormalProposal)],
            draws=draws,
            cores=n_chains,
            tune=tune,
            discard_tuned_samples=False,
            chains=n_chains,
            progressbar=True,
            initvals=init_samps,
        )

    if save_results:
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / "calib_full_out.nc"
        idata_raw.to_netcdf(str(out_file))
    else:
        return idata_raw


def calibrate_with_configs(
    out_path: Path,
    params: Dict[str, Union[float, int]],
    covid_configs: Dict[str, Dict[str, bool]],
    budget: int = 1000,
    n_chains: int = 8,
    draws: int = 100000,
    tune: int = 50000
) -> None:
    """
    Run calibration across multiple COVID effect configurations and save each result.

    Args:
        out_path (Path): Directory to save output files.
        params (Dict[str, Union[float, int]]): Model parameter dictionary.
        covid_configs (Dict[str, Dict[str, bool]]): Mapping of configuration names to COVID effect settings.
        budget (int): Optimization budget passed internally (placeholder, currently unused).
        n_chains (int): Number of MCMC chains.
        draws (int): Number of posterior draws.
        tune (int): Number of tuning steps.
    """
    for config_name, covid_effects in covid_configs.items():
        idata_raw = calibrate(
            out_path=out_path,
            params=params,
            covid_effects=covid_effects,
            budget=budget,
            n_chains=n_chains,
            draws=draws,
            tune=tune,
            save_results=False
        )
        idata_raw.to_netcdf(str(out_path / f"calib_full_out_{config_name}.nc"))
