"""Unit tests for orchestrator's per-agent dispatch / response field map.

The map is the single source of truth for how Tier 1-5 agents are invoked
by the orchestrator's dispatcher (method name, async vs sync, kwargs splat,
optional Pydantic input wrapper) and where the synthesizer finds the
narrative in each agent's output. Schema drift here is the most common cause
of "orchestrator returns empty response" bugs, so we pin the surface.
"""

from __future__ import annotations

import pytest

from src.agents.orchestrator._agent_method_map import (
    AGENT_METHOD_MAP,
    AGENT_RESPONSE_FIELDS,
    AgentMethodSpec,
    extract_narrative,
    get_method_spec,
)


def test_method_map_covers_all_tier1_5_agents() -> None:
    """Every agent named in the factory's Tier 1-5 must have a method spec."""
    expected = {
        "orchestrator",
        "tool_composer",
        "causal_impact",
        "gap_analyzer",
        "heterogeneous_optimizer",
        "drift_monitor",
        "experiment_designer",
        "experiment_monitor",
        "health_score",
        "prediction_synthesizer",
        "resource_optimizer",
        "explainer",
        "feedback_learner",
    }
    missing = expected - set(AGENT_METHOD_MAP.keys())
    assert not missing, f"AGENT_METHOD_MAP missing entries: {missing}"


def test_get_method_spec_fallback_for_unknown_agent() -> None:
    """Unknown agents get the legacy ``.analyze`` contract so old code keeps working."""
    spec = get_method_spec("never-heard-of-this-agent")
    assert spec.method == "analyze"
    assert spec.is_async is True
    assert spec.uses_kwargs is False


def test_method_specs_match_real_agent_entry_points() -> None:
    """Lock in the actual entry-point method names so refactors must update both.

    causal_impact, gap_analyzer, heterogeneous_optimizer all implement BOTH
    ``.run()`` AND ``.analyze()``. Per #252, both production and harness use
    ``.run()`` (the newer primary entry point returning the Pydantic Output
    contract). Updating this set without updating AGENT_METHOD_MAP must fail.
    """
    cases = {
        "orchestrator": ("run", True, False),
        "tool_composer": ("run", True, False),
        "causal_impact": ("run", True, False),
        "gap_analyzer": ("run", True, False),
        "heterogeneous_optimizer": ("run", True, False),
        "drift_monitor": ("run", True, False),
        "experiment_designer": ("run", False, False),
        "experiment_monitor": ("run_async", True, False),
        "health_score": ("check_health", True, True),
        "prediction_synthesizer": ("synthesize", True, True),
        "resource_optimizer": ("optimize", True, True),
        "explainer": ("explain", True, True),
        "feedback_learner": ("learn", True, True),
    }
    for agent_name, (method, is_async, uses_kwargs) in cases.items():
        spec = get_method_spec(agent_name)
        assert spec.method == method, f"{agent_name} method"
        assert spec.is_async is is_async, f"{agent_name} async"
        assert spec.uses_kwargs is uses_kwargs, f"{agent_name} kwargs"


def test_pydantic_input_wrappers_point_at_real_modules() -> None:
    """Agents that wrap input in a model must declare both module and class."""
    for name, spec in AGENT_METHOD_MAP.items():
        if spec.input_model is None and spec.input_module is None:
            continue
        assert spec.input_model and spec.input_module, (
            f"{name}: input_model and input_module must be set together "
            f"(model={spec.input_model!r}, module={spec.input_module!r})"
        )


def test_extract_narrative_per_agent_fields() -> None:
    """Each per-agent narrative key wins over generic ``narrative``/``response``."""
    assert (
        extract_narrative("causal_impact", {"executive_summary": "ATE 12%", "narrative": "ignored"})
        == "ATE 12%"
    )
    assert extract_narrative("health_score", {"health_summary": "all green"}) == "all green"
    assert (
        extract_narrative("experiment_monitor", {"monitor_summary": "0 critical, 2 warn"})
        == "0 critical, 2 warn"
    )


def test_extract_narrative_falls_back_to_legacy_fields() -> None:
    """Agents not in the per-agent map still get narrative/response defaults."""
    assert extract_narrative("legacy_agent", {"narrative": "n"}) == "n"
    assert extract_narrative("legacy_agent", {"response": "r"}) == "r"


def test_extract_narrative_handles_list_first_element() -> None:
    """``key_insights`` is a list[str]; the first string is the narrative."""
    assert (
        extract_narrative("gap_analyzer", {"key_insights": ["first insight", "second"]})
        == "first insight"
    )


def test_extract_narrative_empty_returns_empty_string() -> None:
    """Empty / missing fields don't crash and don't fabricate text."""
    assert extract_narrative("orchestrator", {}) == ""
    assert extract_narrative("orchestrator", {"narrative": "   "}) == ""


def test_response_fields_aligned_with_method_map() -> None:
    """Every agent in the method map should have at least one narrative field."""
    for agent_name in AGENT_METHOD_MAP:
        assert agent_name in AGENT_RESPONSE_FIELDS, (
            f"{agent_name} has no AGENT_RESPONSE_FIELDS entry — synthesizer "
            f"will fall through to str(output) which is rarely useful."
        )


def test_agent_method_spec_is_frozen() -> None:
    """Spec is a frozen dataclass; accidental mutation must raise."""
    spec = AgentMethodSpec(method="run")
    with pytest.raises(Exception):
        spec.method = "analyze"  # type: ignore[misc]
