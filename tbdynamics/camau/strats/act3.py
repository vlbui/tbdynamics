from summer2 import Stratification, Overwrite, Multiply
from summer2.functions.time import get_linear_interpolation_function, get_sigmoidal_interpolation_function
from summer2.parameters import Parameter
from tbdynamics.constants import AGE_STRATA, ORGAN_STRATA
from tbdynamics.camau.constants import ACT3_STRATA
from tbdynamics.tools.utils import get_mix_from_strat_props
from tbdynamics.tools.detect import (
    get_interpolation_rates_from_annual,
    calculate_screening_rate,
)
from typing import Dict, List, Any, Optional
from tbdynamics.tools.inputs import load_targets
from tbdynamics.settings import CM_PATH


def get_act3_strat(
    compartments: List[str],
    fixed_params: Dict[str, Any],
    future_acf_scenarios: Optional[Dict[str, Dict[str, Dict[float, float]]]] = None,
) -> Stratification:
    strat = Stratification("act3", ACT3_STRATA, compartments)

    # Population split and mixing matrix
    props = fixed_params["act3_stratification"]["proportions"]
    assert abs(sum(props.values()) - 1.0) < 1e-6, "Proportions do not sum to 1.0"
    strat.set_population_split(props)
    strat.set_flow_adjustments("birth", {k: Multiply(v) for k, v in props.items()})

    prop_same_strat = fixed_params["act3_stratification"]["prop_mixing_same_stratum"]
    props_list = [props[arm] for arm in ACT3_STRATA]
    mixing_matrix = get_mix_from_strat_props(prop_same_strat, props_list)
    strat.set_mixing_matrix(mixing_matrix)

    # Historical screening rates
    targets = load_targets(CM_PATH / "targets.yml")
    trial_acf_rates = calculate_screening_rate(
        targets["act3_trial_adults_pop"].to_dict(),
        targets["act3_trial_sputum_collected"].to_dict(),
    )


    control_acf_rates = calculate_screening_rate(
        targets["act3_control_adults_pop"].to_dict(),
        targets["act3_control_sputum_collected"].to_dict(),
    )

    control_acf_rates = get_interpolation_rates_from_annual(control_acf_rates)

    def combine_screen_func(hist_acf, future_acf):
        if not hist_acf and not future_acf:
            return None
        if hist_acf and not future_acf:
            return get_linear_interpolation_function(
                list(hist_acf.keys()), list(hist_acf.values())
            )
        combined = hist_acf.copy() if hist_acf else {}
        combined.update(future_acf)
        return get_sigmoidal_interpolation_function(
            list(combined.keys()), list(combined.values())
        )

    # acf_sens = Parameter("acf_sensitivity")
    acf_sens = fixed_params["act3_stratification"]['acf_sensitivity']
    act3_adjs = {}
    base_rates = {
        "trial": trial_acf_rates,
        "control": control_acf_rates,
        "other": None,
    }
    for arm in ACT3_STRATA:
        hist = base_rates[arm]
        future_merged = {}

        # Merge future ACF rates from all scenarios that include this arm
        if future_acf_scenarios:
            for scenario_dict in future_acf_scenarios.values():
                if arm in scenario_dict:
                    future_merged.update(scenario_dict[arm])

        future_rate = (
            get_interpolation_rates_from_annual(future_merged)
            if future_merged
            else None
        )

        combined = combine_screen_func(hist, future_rate)
        act3_adjs[arm] = Overwrite(acf_sens * combined) if combined else None

    # Only apply flow adjustments once per age stratum
    for age_stratum in AGE_STRATA[2:]:
        source = {"age": str(age_stratum)}
        strat.set_flow_adjustments("acf_detection", act3_adjs, source_strata=source)

    return strat