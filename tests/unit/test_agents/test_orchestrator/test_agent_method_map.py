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
    """Lock in the actual entry-point method names so refactors must update both."""
    cases = {
        "orchestrator": ("run", True, False),
        "tool_composer": ("run", True, False),
        "causal_impact": ("analyze", True, False),
        "gap_analyzer": ("analyze", True, False),
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


# ---------------------------------------------------------------------------
# Issue #252: harness <-> orchestrator dispatch-spec parity
#
# ``scripts/run_tier1_5_test.py`` must NOT carry its own copy of per-agent
# dispatch metadata (method name, async flag, kwargs splat, input model).
# Phase 1 of the Tier 1-5 plan promoted that data into AGENT_METHOD_MAP so
# both the live dispatcher and the integration harness import from one place.
# These tests enforce that contract structurally: silent drift here is the
# single most common cause of "harness PASS, prod AttributeError" bugs.
# ---------------------------------------------------------------------------


def _load_harness_module():
    """Import ``scripts/run_tier1_5_test.py`` as a module without executing main.

    The harness uses ``from __future__`` + dataclasses + dotenv but does not
    execute side effects at import time beyond ``load_dotenv``, so a plain
    import is safe for parity assertions.
    """
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    harness_path = repo_root / "scripts" / "run_tier1_5_test.py"
    assert harness_path.exists(), f"harness script missing at {harness_path}"

    spec = importlib.util.spec_from_file_location("run_tier1_5_test_harness", harness_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Ensure the harness can resolve ``src.*`` imports relative to repo root.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec.loader.exec_module(module)
    return module


def test_harness_imports_method_map_as_single_source_of_truth() -> None:
    """AC #252 / A1: harness pulls AGENT_METHOD_MAP from orchestrator."""
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    harness_path = repo_root / "scripts" / "run_tier1_5_test.py"
    source = harness_path.read_text()
    tree = ast.parse(source)

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("orchestrator._agent_method_map"):
                imported_names.update(a.name for a in node.names)

    assert "AGENT_METHOD_MAP" in imported_names, (
        "scripts/run_tier1_5_test.py must `from src.agents.orchestrator."
        "_agent_method_map import AGENT_METHOD_MAP, AGENT_RESPONSE_FIELDS` — "
        "harness still owns its own copy of the dispatch map (issue #252)."
    )
    assert "AGENT_RESPONSE_FIELDS" in imported_names, (
        "harness must also import AGENT_RESPONSE_FIELDS so the synthesis "
        "narrative key list is shared with the orchestrator (issue #252)."
    )


def test_harness_agent_configs_derived_from_method_map() -> None:
    """AC #252 / A3: every harness-known agent has an AGENT_METHOD_MAP entry."""
    harness = _load_harness_module()
    harness_agents = set(harness.AGENT_CONFIGS.keys())
    missing = harness_agents - set(AGENT_METHOD_MAP.keys())
    assert not missing, (
        f"harness AGENT_CONFIGS references agents missing from "
        f"AGENT_METHOD_MAP: {sorted(missing)} — single-source-of-truth "
        f"broken (issue #252)."
    )


def test_harness_dispatch_fields_match_method_map() -> None:
    """AC #252 / A3: spec.method + spec.uses_kwargs + spec.is_async + input
    model fields agree between AGENT_METHOD_MAP and the harness's per-agent
    config dict. If any value diverges the test trips loudly."""
    harness = _load_harness_module()
    for agent_name, harness_cfg in harness.AGENT_CONFIGS.items():
        spec = AGENT_METHOD_MAP[agent_name]
        assert harness_cfg.get("method") == spec.method, (
            f"{agent_name}: harness method "
            f"{harness_cfg.get('method')!r} != spec.method {spec.method!r}"
        )
        # ``is_async`` and ``uses_kwargs`` are bools; both sides should be
        # explicit when the spec differs from the default.
        harness_async = harness_cfg.get("is_async", True)
        harness_kwargs = harness_cfg.get("uses_kwargs", False)
        assert harness_async is spec.is_async, (
            f"{agent_name}: harness is_async={harness_async} != spec.is_async={spec.is_async}"
        )
        assert harness_kwargs is spec.uses_kwargs, (
            f"{agent_name}: harness uses_kwargs={harness_kwargs} != "
            f"spec.uses_kwargs={spec.uses_kwargs}"
        )
        # input_model / input_module are optional. When the spec sets them
        # the harness MUST surface the same string (since it actually
        # imports the model class). When the spec does not set them the
        # harness must also leave them unset.
        assert harness_cfg.get("input_model") == spec.input_model, (
            f"{agent_name}: harness input_model "
            f"{harness_cfg.get('input_model')!r} != "
            f"spec.input_model {spec.input_model!r}"
        )
        assert harness_cfg.get("input_module") == spec.input_module, (
            f"{agent_name}: harness input_module "
            f"{harness_cfg.get('input_module')!r} != "
            f"spec.input_module {spec.input_module!r}"
        )


def test_harness_preserves_per_harness_extras() -> None:
    """AC #252 / A2: harness-only fields (tier, state_class, agent_class,
    state_module, agent_module) MUST still be present on every harness
    config — they layer on top of the shared dispatch spec but are not
    promoted to the orchestrator-side map (different concern)."""
    harness = _load_harness_module()
    required_extras = {
        "tier",
        "state_module",
        "state_class",
        "agent_module",
        "agent_class",
    }
    for agent_name, cfg in harness.AGENT_CONFIGS.items():
        missing = required_extras - set(cfg.keys())
        assert not missing, (
            f"{agent_name}: harness extras {sorted(missing)} missing — "
            f"the harness needs these to load + validate output contracts "
            f"even though they are not part of AGENT_METHOD_MAP."
        )


def test_harness_analysis_config_narrative_fields_overlap_response_map() -> None:
    """AC #252 / A2: AGENT_ANALYSIS_CONFIG.key_fields must include at least
    one narrative key from AGENT_RESPONSE_FIELDS for each agent that has a
    narrative source. (The rest of insights_template is harness-local and
    explicitly out of scope per the issue body.)"""
    harness = _load_harness_module()
    analysis = harness.AGENT_ANALYSIS_CONFIG
    for agent_name, narrative_keys in AGENT_RESPONSE_FIELDS.items():
        if agent_name not in analysis:
            # Tier-specific analysis is optional; only enforce overlap when
            # the harness has an analysis entry for the agent.
            continue
        key_fields = set(analysis[agent_name].get("key_fields", []))
        overlap = key_fields & set(narrative_keys)
        # Some agents (orchestrator, tool_composer) have analysis-only
        # fields and do not necessarily surface AGENT_RESPONSE_FIELDS;
        # only enforce overlap when the agent has a domain narrative
        # (executive_summary / *_summary / *_interpretation pattern).
        narrative_kinds = {
            k
            for k in narrative_keys
            if k.endswith("_summary") or k == "executive_summary" or k.endswith("_interpretation")
        }
        if not narrative_kinds:
            continue
        assert overlap, (
            f"{agent_name}: AGENT_ANALYSIS_CONFIG.key_fields {sorted(key_fields)} "
            f"shares no narrative key with AGENT_RESPONSE_FIELDS "
            f"{narrative_keys} — harness analysis is reading fields the "
            f"orchestrator synthesizer does not even pull (issue #252)."
        )
