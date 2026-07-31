"""Faithful (real Supabase) proof that the causal_impact input resolver BUILDS a
real, leakage-free causal spec from the live KPI substrate (#1351).

Mirrors ``test_dispatcher_het_kpi_substrate_realdb.py`` — the causal resolver's
branch (2) is deliberately the same build (same KpiFrame, same min-rows floor,
same treatment-source leak exclusion), so the same live proof applies: a
"conversion" chat ask yields the real treatment (``accepted``) + outcome
(``converted``) + real confounders drawn from the frame's driver columns with
``acceptance_status`` excluded, over the real per-trigger frame threaded as
``data`` (→ ``data_cache['estimation_data']``, the #606 channel).

Gated behind ``E2I_DB_INTEGRATION=1``; exercises ONLY the resolver (never the
DoWhy pipeline), so it is cheap and OOM-safe.
"""

from __future__ import annotations

import os
import time

import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import NeedsStructuredInput

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="requires real docker-Supabase (set E2I_DB_INTEGRATION=1)",
)


def _dispatch(params=None):
    return {
        "agent_name": "causal_impact",
        "priority": "critical",
        "parameters": params or {},
        "timeout_ms": 150000,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def test_causal_resolver_builds_real_spec_from_live_conversion_substrate() -> None:
    # include_synthetic opt-in (#872): the synthetic-gold cleanup left the live
    # DB with ZERO untagged trigger/treatment_event rows; the substrate-binding
    # proof opts in explicitly, exactly like the het integration test.
    agent_input = {
        "query": "what is the causal impact on Kisqali conversion in the west region?",
        "session_id": "itest-causal",
        "user_context": {"include_synthetic": True},
        "parsed_query": {"entities": []},
    }
    before = time.monotonic()
    resolved = disp.INPUT_RESOLVERS["causal_impact"](agent_input, _dispatch())

    assert not isinstance(resolved, NeedsStructuredInput), getattr(resolved, "reason", "")
    assert resolved["treatment_var"] == "accepted"
    assert resolved["outcome_var"] == "converted"
    # Leak guard on real columns.
    assert "accepted" not in resolved["confounders"]
    assert "acceptance_status" not in resolved["confounders"]
    assert resolved["confounders"], "real driver columns must remain as confounders"
    # Real frame threaded via the estimation-data channel, above the row floor.
    assert resolved["data"] is not None
    assert len(resolved["data"]) >= disp._CAUSAL_MIN_ROWS
    assert resolved["data_source"].startswith("kpi_substrate:")
    # Query-text grounding (no parsed_query producer exists on the chat path).
    assert resolved["brand"] == "Kisqali"
    # Cooperative refutation deadline strictly inside the dispatch budget.
    assert before < resolved["compute_deadline"] <= before + 150.0


def test_causal_resolver_unbindable_ask_fails_closed_with_kg_candidates() -> None:
    """A TRx-style ask (no KPI substrate builder) fails closed GRACEFULLY —
    never the pre-#1351 hard raise — and seeds candidates from the curated
    causal KG when the graph service is reachable."""
    agent_input = {
        "query": "What is the causal impact of rep visits on TRx for Kisqali?",
        "session_id": "itest-causal-2",
        "user_context": {"include_synthetic": True},
        "parsed_query": {"entities": []},
    }
    resolved = disp.INPUT_RESOLVERS["causal_impact"](agent_input, _dispatch())
    assert isinstance(resolved, NeedsStructuredInput)
    err = resolved.to_error()
    assert "Missing required field(s)" not in err
    assert "treatment_var" in err
    assert "no values were fabricated" in err.lower()
