"""Tests for cohort_profiler — the chat companion that answers cohort/segment
queries with REAL per-segment counts instead of the cohort_constructor dead-end.

Covers three surfaces of the fix:
  1. the agent computes a real severity + line-of-therapy breakdown and fails
     closed (no fabricated table) when a brand has no prescribing population;
  2. its input resolver grounds an optional brand and NEVER fails closed;
  3. the classifier routes COHORT_DEFINITION queries to cohort_profiler.

The KPI calculator is faked so the test needs no DB — the real calculation path
(get_kpi_calculator().calculate) is exercised faithfully by the container replay.
"""

import pytest

from src.agents.cohort_profiler import CohortProfilerAgent

# Real Remibrutinib NRx numbers (mig-105 breakdown; verified live in PR #1208).
_REMI_NRX = {
    None: 3256.0,
    "low_severity": 855.0,
    "medium_severity": 1752.0,
    "high_severity": 649.0,
    "0": 822.0,
    "1": 825.0,
    "2": 831.0,
    "3": 778.0,
}


class _FakeCalc:
    """Stands in for KPICalculator: returns the mig-105 counts for Remibrutinib,
    and None (no population) for any other brand."""

    def __init__(self, table):
        self._table = table

    def calculate(self, kpi_id, context=None):
        context = context or {}
        if context.get("brand") != "Remibrutinib":
            return {"value": None}
        key = context.get("segment") or context.get("therapy_line") or None
        return {"value": self._table.get(key)}


def _agent_with(table):
    agent = CohortProfilerAgent()
    agent._get_calculator = lambda: _FakeCalc(table)  # type: ignore[method-assign]
    return agent


@pytest.mark.asyncio
async def test_analyze_returns_real_severity_and_line_breakdown():
    agent = _agent_with(_REMI_NRX)
    out = await agent.analyze({"brand": "Remibrutinib", "query": "build a cohort"})

    assert out["status"] == "completed"
    narrative = out["narrative"]
    # Real per-tier + per-line counts appear (comma-formatted), summing to headline.
    for token in ("Remibrutinib", "855", "1,752", "649", "822", "778", "3,256"):
        assert token in narrative, f"missing {token!r} in narrative"
    # Honest hand-off: population size, not a materialized patient list.
    assert "scope_definer" in narrative and "cohort_constructor" in narrative


@pytest.mark.asyncio
async def test_analyze_canonicalizes_brand_casing():
    # brand predicate is case-SENSITIVE; a lowercase chat mention must still work.
    agent = _agent_with(_REMI_NRX)
    out = await agent.analyze({"brand": "remibrutinib", "query": "cohort"})
    assert out["status"] == "completed"
    assert "855" in out["narrative"]


@pytest.mark.asyncio
async def test_analyze_fails_closed_when_no_population():
    # Calculator returns None for every context (empty table) → honest fail-closed,
    # NOT an empty/zero table laundered as success.
    agent = _agent_with({})
    out = await agent.analyze({"brand": "Remibrutinib", "query": "cohort"})
    assert out["status"] == "failed"
    assert out["narrative"] == ""
    assert out["errors"]


def test_resolver_grounds_brand_and_never_fails_closed():
    from src.agents.orchestrator.nodes.dispatcher import (
        NeedsStructuredInput,
        _resolve_cohort_profiler_input,
    )

    dispatch = {"agent_name": "cohort_profiler", "parameters": {}}
    resolved = _resolve_cohort_profiler_input(
        {"user_context": {"brand": "Remibrutinib"}, "query": "define a cohort"}, dispatch
    )
    assert not isinstance(resolved, NeedsStructuredInput)
    assert resolved["brand"] == "Remibrutinib"

    # No brand named → still returns inputs (profiles all brands), never fails closed.
    resolved_nobrand = _resolve_cohort_profiler_input({"query": "define a cohort"}, dispatch)
    assert not isinstance(resolved_nobrand, NeedsStructuredInput)
    assert "brand" not in resolved_nobrand


def test_classifier_routes_cohort_definition_to_profiler():
    from src.agents.orchestrator.classifier.pattern_selector import DOMAIN_TO_AGENT
    from src.agents.orchestrator.classifier.schemas import Domain

    assert DOMAIN_TO_AGENT[Domain.COHORT_DEFINITION] == "cohort_profiler"
