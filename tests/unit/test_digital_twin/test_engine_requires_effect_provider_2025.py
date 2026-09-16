"""``SimulationEngine`` has no default effect provider (#2025).

The engine defaulted to ``SyntheticEffectDataProvider()``, whose planted ``true_ate`` is
0.15: a caller that forgot the provider got that planted effect back as a simulation
result (a plausible uplift, the live cohort estimate for Kisqali email is 0.135). Every
caller now states where its effect data comes from; a test that wants the synthetic DGP
passes it explicitly.
"""

from __future__ import annotations

import inspect

import pytest

from src.digital_twin.models.twin_models import Brand, TwinPopulation, TwinType
from src.digital_twin.simulation_engine import SimulationEngine


def _empty_population() -> TwinPopulation:
    return TwinPopulation(twin_type=TwinType.HCP, brand=Brand.KISQALI, twins=[], size=0)


def test_engine_refuses_construction_without_an_effect_provider() -> None:
    with pytest.raises(TypeError, match="effect_provider"):
        SimulationEngine(population=_empty_population())  # type: ignore[call-arg]


def test_effect_provider_parameter_has_no_default() -> None:
    param = inspect.signature(SimulationEngine.__init__).parameters["effect_provider"]
    assert param.default is inspect.Parameter.empty
