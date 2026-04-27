from summer2 import CompartmentalModel
from summer2.parameters import Parameter, Function, Time
from tbdynamics.tools.utils import triangle_wave_func

PLACEHOLDER_PARAM = 1.0


def add_treatment_related_outcomes(model: CompartmentalModel):
    for flow_name, rate, to_compartment in [
        ("treatment_recovery", PLACEHOLDER_PARAM, "recovered"),
        ("relapse", PLACEHOLDER_PARAM, "infectious"),
    ]:
        model.add_transition_flow(flow_name, rate, "on_treatment", to_compartment)
    model.add_death_flow("treatment_death", PLACEHOLDER_PARAM, "on_treatment")


def seed_infectious(model: CompartmentalModel, target_compartment: str = "infectious"):
    seed_func = Function(
        triangle_wave_func,
        [Time, Parameter("seed_time"), Parameter("seed_duration"), Parameter("seed_num")],
    )
    model.add_importation_flow("seed_infectious", seed_func, target_compartment, split_imports=True)
