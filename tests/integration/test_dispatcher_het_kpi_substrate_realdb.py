"""Faithful (real Supabase) proof that the heterogeneous_optimizer input
resolver BUILDS a real, leakage-free causal spec from the live KPI substrate
(audit F12 — "build substrate from real data").

Gated behind ``E2I_DB_INTEGRATION=1`` because it queries the real docker-Supabase
``triggers ⋈ treatment_events`` conversion substrate. It exercises ONLY the
resolver (not the full CATE pipeline), so it is cheap and OOM-safe while still
proving the premise against real rows: a "conversion" chat query yields a real
treatment (``accepted``) + outcome (``converted``) + real effect modifiers drawn
from the frame's columns, with the treatment's raw source (``acceptance_status``)
excluded, over the real per-trigger frame threaded via ``tier0_data``.
"""

from __future__ import annotations

import os

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import NeedsStructuredInput

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="requires real docker-Supabase (set E2I_DB_INTEGRATION=1)",
)


def _dispatch(params=None):
    return {
        "agent_name": "heterogeneous_optimizer",
        "priority": "high",
        "parameters": params or {},
        "timeout_ms": 30000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def test_het_resolver_builds_real_spec_from_live_conversion_substrate() -> None:
    # include_synthetic opt-in (#872): the 2026-06-11 synthetic-gold cleanup left the
    # live DB with ZERO untagged trigger/treatment_event rows, so a real-mode build
    # now correctly fails closed (that is the designed provenance isolation, proven
    # by test_het_resolver_real_mode_fails_closed_on_clean_substrate below). The
    # substrate-binding proof therefore opts in explicitly, exactly like gate 11.
    agent_input = {
        "query": "which segments respond best on conversion rate?",
        "session_id": "itest-het",
        "user_context": {"include_synthetic": True},
        "parsed_query": {"entities": []},
    }
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](agent_input, _dispatch())

    # The live conversion frame is large (thousands of real rows) → real spec.
    assert not isinstance(resolved, NeedsStructuredInput), getattr(resolved, "reason", "")
    assert resolved["outcome_var"] == "converted"
    assert resolved["treatment_var"] == "accepted"
    # Real frame threaded via the het_opt tier0 passthrough, above the 100-row floor.
    assert resolved["tier0_data"] is not None
    assert len(resolved["tier0_data"]) >= disp._HET_MIN_ROWS
    # Effect modifiers are REAL columns present in the frame, and LEAKAGE-FREE:
    # neither the treatment nor its raw source may appear.
    cols = set(resolved["tier0_data"].columns)
    assert resolved["effect_modifiers"], "expected ≥1 real effect modifier"
    for m in resolved["effect_modifiers"]:
        assert m in cols, f"modifier {m} not a real frame column"
    assert "accepted" not in resolved["effect_modifiers"]
    assert "acceptance_status" not in resolved["effect_modifiers"], "treatment-source leakage!"
    assert "kpi_substrate" in resolved["data_source"]


def test_het_resolver_real_mode_fails_closed_on_clean_substrate() -> None:
    """Real mode (no opt-in) on the post-cleanup substrate (zero untagged rows):
    the resolver must fail closed — synthetic rows are NEVER silently read (#872)."""
    agent_input = {
        "query": "which segments respond best on conversion rate?",
        "session_id": "itest-het",
        "user_context": {},
        "parsed_query": {"entities": []},
    }
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](agent_input, _dispatch())
    assert isinstance(resolved, NeedsStructuredInput), "real-mode must not read synthetic rows"


def test_het_resolver_string_false_opt_out_fails_closed_on_clean_substrate() -> None:
    """Issue #880: ``include_synthetic="false"`` (string) is an explicit opt-OUT.
    The loose ``bool()`` coercion read it as truthy and silently bound the
    SYNTHETIC substrate (wrong-provenance read); strict parsing keeps the
    dispatch real-mode, which on the clean post-cleanup substrate fails closed
    exactly like no opt-in above."""
    agent_input = {
        "query": "which segments respond best on conversion rate?",
        "session_id": "itest-het",
        "user_context": {"include_synthetic": "false"},
        "parsed_query": {"entities": []},
    }
    resolved = disp.INPUT_RESOLVERS["heterogeneous_optimizer"](agent_input, _dispatch())
    assert isinstance(resolved, NeedsStructuredInput), (
        "string 'false' must NOT opt into the synthetic substrate"
    )
