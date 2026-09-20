"""Demo 6.5 routes to the tool composer, so the forecaster must be composable (#2115).

Three separate things have to be true before a composed plan can contain a forecast
step, and each has failed silently in this codebase before:

* the tool is registered in the LIVE registry WITH an output model (the composer's
  schema builder raises ``LookupError`` without one);
* it is named in ``TOOL_METADATA``, which is what decides it exists for the planner;
* the planner's PREDICTIVE fallback actually reaches it for a forecast-shaped ask —
  the map sent every PREDICTIVE question to ``risk_scorer``, which is why 6.5 came back
  with an entity-level risk score instead of a forecast.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.timeout(600)


def test_the_forecaster_is_named_in_the_metadata_the_planner_reads():
    from src.agents.tool_composer.schemas import ToolCategory
    from src.agents.tool_composer.tool_registry import TOOL_METADATA
    from src.tool_registry.tools.kpi_forecast import TOOL_NAME

    assert TOOL_NAME in TOOL_METADATA
    category, consumes = TOOL_METADATA[TOOL_NAME]
    assert category is ToolCategory.PREDICTION
    assert consumes == [], "a forecast reads the stored series, not another tool's output"


def test_the_composer_schema_builder_accepts_it_which_needs_an_output_model():
    from src.agents.tool_composer.tool_registry import create_default_tools
    from src.tool_registry.tools.kpi_forecast import TOOL_NAME

    schemas = {s.name: s for s in create_default_tools()}
    assert TOOL_NAME in schemas, "the forecaster is missing from the composer's schema set"
    schema = schemas[TOOL_NAME]
    assert schema.output_schema, "no output schema means the planner cannot bind the step"
    assert "forecast" in schema.output_schema.get("properties", {})


def test_the_composer_bootstrap_registers_it_so_a_live_plan_can_use_it():
    from src.agents.tool_composer import composer
    from src.tool_registry.registry import get_registry
    from src.tool_registry.tools.kpi_forecast import TOOL_NAME

    composer._ensure_tools_registered()
    assert get_registry().validate_tool_exists(TOOL_NAME)


# --------------------------------------------------------------------------- planning
class _Registry:
    """The planner only asks the registry whether a tool exists."""

    def __init__(self, known):
        self.known = set(known)

    def validate_tool_exists(self, name):
        return name in self.known

    def get_schema(self, name):
        return None


class _SubQuestion:
    def __init__(self, question, intent="PREDICTIVE", sq_id="sq1"):
        self.question = question
        self.intent = intent
        self.id = sq_id


def _map(question, intent="PREDICTIVE", known=("kpi_forecaster", "risk_scorer")):
    from src.agents.tool_composer.planner import ToolPlanner

    planner = ToolPlanner.__new__(ToolPlanner)
    planner.registry = _Registry(known)
    return planner._get_fallback_mapping(_SubQuestion(question, intent))


@pytest.mark.parametrize(
    "question",
    [
        "Forecast Kisqali TRx volume for the next two quarters",
        "What will NRx be by the end of the year?",
        "Project NBRx through Q2",
        "Where is TRx headed over the next six months?",
        "Give me the TRx outlook for Fabhalta",
    ],
)
def test_a_forecast_shaped_predictive_ask_maps_to_the_forecaster(question):
    """This is the defect 6.5 hit: every PREDICTIVE ask went to risk_scorer."""
    mapping = _map(question)
    assert mapping is not None
    assert mapping.tool_name == "kpi_forecaster", f"{question!r} -> {mapping.tool_name}"


@pytest.mark.parametrize(
    "question",
    [
        "Which HCP segments are highest risk of churn?",
        "Score the propensity of these prescribers to switch",
    ],
)
def test_a_scoring_shaped_predictive_ask_still_maps_to_the_risk_scorer(question):
    """The fix must not swallow the asks the old mapping was RIGHT about.

    A gate that only proves the new route would also pass if every PREDICTIVE ask now
    went to the forecaster, which would break the entity-scoring questions this map has
    served all along.
    """
    mapping = _map(question)
    assert mapping is not None
    assert mapping.tool_name == "risk_scorer", f"{question!r} -> {mapping.tool_name}"


def test_the_forecast_route_falls_back_when_the_forecaster_is_not_registered():
    """An unregistered forecaster must degrade to the old behaviour, not return None."""
    mapping = _map("Forecast Kisqali TRx for two quarters", known=("risk_scorer",))
    assert mapping is not None
    assert mapping.tool_name == "risk_scorer"
