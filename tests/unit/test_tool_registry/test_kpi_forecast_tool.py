"""The composable ``kpi_forecaster`` the tool composer plans with (#2115, demo 6.5).

6.5 is a two-part ask — forecast, then the biggest risk to that forecast — so it routes
to the tool composer, which plans a forecast step plus risk steps from the gap and
causal tools. The composer can only plan a step for a tool that is IN THE REGISTRY, so
this is the registration and the contract the planner reads.
"""

from __future__ import annotations

import asyncio

import pytest

from src.tool_registry import get_registry
from src.tool_registry.tools import kpi_forecast as kf

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(autouse=True)
def registered():
    kf.register_kpi_forecast_tool()
    return get_registry()


def test_the_tool_is_in_the_registry_under_the_name_the_planner_maps_to(registered):
    assert registered.validate_tool_exists(kf.TOOL_NAME)
    assert kf.TOOL_NAME == "kpi_forecaster"


def test_registering_twice_is_harmless(registered):
    kf.register_kpi_forecast_tool()
    assert registered.validate_tool_exists(kf.TOOL_NAME)


def test_the_schema_declares_the_arguments_a_planner_has_to_fill(registered):
    schema = registered.get_schema(kf.TOOL_NAME)
    params = {p.name: p for p in schema.input_parameters}
    assert set(params) == {"kpi_name", "brand", "region", "horizon_months"}
    assert params["kpi_name"].required is True
    assert params["brand"].required is False
    assert params["horizon_months"].required is False


def test_the_description_tells_the_planner_it_forecasts_and_what_it_cannot_see(registered):
    """The planner picks tools from these descriptions. If this one does not say the
    forecast is univariate, the planner has no reason to add the risk steps 6.5 needs."""
    description = registered.get_schema(kf.TOOL_NAME).description.lower()
    assert "forecast" in description
    assert "univariate" in description
    assert "risk" in description


def test_the_source_agent_is_the_predictive_one_the_intent_map_routes_to(registered):
    assert registered.get_schema(kf.TOOL_NAME).source_agent == "prediction_synthesizer"


def test_the_registered_callable_is_the_one_the_chat_tool_uses_not_a_second_copy():
    """Two implementations of one forecast would drift and disagree in the same answer."""
    import inspect

    assert "run_forecast" in inspect.getsource(kf.kpi_forecaster)


def test_it_returns_the_same_payload_shape_the_chat_tool_returns():
    out = asyncio.run(kf.kpi_forecaster(kpi_name="conversion_rate", brand="Kisqali"))
    assert out["query_type"] == "kpi_forecast"
    assert out["success"] is False
    assert "error" in out
