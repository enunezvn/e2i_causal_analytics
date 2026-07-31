"""Legacy weak-class rule tuning against the #1337 337-query gold.

Step 0 (PR #1362) chose the incumbent legacy classifier over the 4-stage
pipeline and both LLM candidates (0.757 vs 0.626/0.623, non-overlapping CIs).
The scope ruling (issue #1337, 2026-07-31 comment) directs improvement to the
measured winner. Legacy per-pattern recall was SA 0.913 / PARALLEL 0.200 /
TOOL_COMPOSER 0.071 / CLARIFICATION 0.000; the standout defects are:

  * **PARALLEL over-trigger (precision 0.028).** ``requires_multi_agent`` fired
    on ANY second strong-intent keyword co-match with no multi-clause check, so
    30 gold-SINGLE rows were split into two agents by an *incidental* keyword
    (``prediction_synthesizer`` appended to drift/health/experiment asks that
    merely say "model"/"forecast"; ``heterogeneous_optimizer`` appended to
    cohort "break down by segment" asks; ``explainer`` appended via the #1366
    KPI regex). Fix: a genuine second facet requires a genuine second *clause*.

  * **TOOL_COMPOSER under-detection (recall 0.071).** Dependency-linked
    multi-step pipelines with >=2 mapped strong intents were routed as PARALLEL
    (dependency undetected) instead of ``tool_composer``. Fix: promote to
    ``tool_composer`` on a broadened dependency signal (referential/conditional
    back-references) OR >=3 intent-bearing clauses, keeping the existing
    ">=2 distinct MAPPED strong intents + not-a-parallel-pair" gate.

Both changes are guarded by the deterministic ``pattern_diff.py`` scorer over
the full 337 gold (every flip justified vs gold; SA recall must not regress).
Queries below are verbatim gold rows (bench-NNNN) — real traffic + authored
probes, not synthetic phrasings — so these are behavioural pins on the tuning,
not phrase overfitting.

No mocks: the deterministic pattern + routing layers run directly.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode
from src.agents.orchestrator.nodes.router import RouterNode


def _classify(query: str) -> dict:
    node = IntentClassifierNode.__new__(IntentClassifierNode)  # skip LLM ctor
    return dict(node._pattern_classify(query.lower()))


def _route(intent: dict) -> list[str]:
    router = RouterNode()
    routed = asyncio.run(router.execute({"query": "", "intent": intent}))
    return sorted(d["agent_name"] for d in routed.get("dispatch_plan", []))


def _classify_and_route(query: str) -> list[str]:
    return _route(_classify(query))


# ---------------------------------------------------------------------------
# Change 1a — incidental second-intent co-matches must NOT split into 2 agents.
# These verbatim gold-SINGLE rows were mis-split to PARALLEL_DELEGATION because a
# single incidental keyword ("predict"/"segment"/a KPI verb) matched a second
# intent inside the SAME clause. The gate collapses them to one dispatch. (Which
# single agent is correct is a SEPARATE tie-break concern — see 1b for the rows
# where the primary intent already equals gold; the rest still stop the
# over-dispatch, the direct PARALLEL-precision win.)
# ---------------------------------------------------------------------------
class TestIncidentalCoMatchCollapsesToSingle:
    @pytest.mark.parametrize(
        "query",
        [
            "How well is our Kisqali predictive model performing in terms of "
            "ROC-AUC and calibration metrics?",
            "Predict which HCP segments are most likely to increase Fabhalta "
            "prescriptions next quarter",
            "Has the churn prediction model degraded since deployment — does it need retraining?",
            "did the TRx forecast model drift after the last reseed?",
            "Are the prediction distributions shifting for the Kisqali model?",
            "Break down Remibrutinib NRx by biologic-naive vs biologic-experienced "
            "patients and by IgE level.",
            "Design an experiment to test whether increasing rep visits improves Fabhalta adoption",
        ],
    )
    def test_single_clause_two_intents_not_split(self, query):
        agents = _classify_and_route(query)
        assert len(agents) == 1, (
            f"{query!r} over-split to {agents}; a second intent keyword inside one "
            f"clause is an incidental co-match, not an independent facet"
        )


# ---------------------------------------------------------------------------
# Change 1b — where the primary intent already equals the gold agent, collapsing
# the over-split recovers agent-exact (gold-SINGLE rows verbatim).
# ---------------------------------------------------------------------------
class TestGoldSingleAgentRecovered:
    @pytest.mark.parametrize(
        "query,expected_agent",
        [
            (
                "Has the churn prediction model degraded since deployment — does it "
                "need retraining?",
                "drift_monitor",
            ),
            ("did the TRx forecast model drift after the last reseed?", "drift_monitor"),
            ("Are the prediction distributions shifting for the Kisqali model?", "drift_monitor"),
            (
                "Have we detected any data distribution shifts or model performance "
                "degradation in our Kisqali predictive analytics?",
                "drift_monitor",
            ),
            (
                "Break down Remibrutinib NRx by biologic-naive vs biologic-experienced "
                "patients and by IgE level.",
                "cohort_profiler",
            ),
            (
                "Design an experiment to test whether increasing rep visits improves "
                "Fabhalta adoption",
                "experiment_designer",
            ),
        ],
    )
    def test_gold_single_row_recovers_agent(self, query, expected_agent):
        assert _classify_and_route(query) == [expected_agent]


# ---------------------------------------------------------------------------
# Change 1 — genuine two-clause parallels MUST survive the gate.
# ---------------------------------------------------------------------------
class TestGenuineParallelSurvives:
    def test_two_wh_clauses_parallel(self):
        # bench-0143 (gold PARALLEL): two independent asks joined by "and".
        agents = _classify_and_route(
            "What is the current total TRx and which region has the largest gap opportunity?"
        )
        assert agents == ["explainer", "gap_analyzer"]

    def test_locked_reversed_pair_still_parallel(self):
        # Locked in test_multipart_tool_composer_routing.py — must not regress.
        agents = _classify_and_route(
            "Which segments responded best and what was the campaign impact?"
        )
        assert agents == ["causal_impact", "heterogeneous_optimizer"]


# ---------------------------------------------------------------------------
# Change 2 — dependency-linked multi-step pipelines route to tool_composer
# (verbatim gold-TOOL_COMPOSER rows previously mis-routed to PARALLEL).
# ---------------------------------------------------------------------------
class TestDependentPipelineRoutesToolComposer:
    @pytest.mark.parametrize(
        "query",
        [
            # 3 dependent steps, referential "its root cause" + "to close it".
            "Find the biggest performance gap for Remibrutinib, explain its root "
            "cause, and propose a resource reallocation to close it",
            # 3 dependent steps, "the worst one" + "the top fix" back-references.
            "Compare persistence across the three brands, run causal attribution on "
            "the worst one, and design a test for the top fix",
            # conditional dependency: "if it has, re-run …".
            "Check whether the Kisqali adoption model has drifted, and if it has, "
            "re-run the segment analysis and tell me which HCP targets change",
            # 5-step dependent investigation (>=3 intent-bearing clauses).
            "Our Kisqali TRx dropped in the northeast last quarter while conversion "
            "rates for Remibrutinib stayed flat, and I need to understand several "
            "things: what actually caused the Kisqali decline, whether "
            "biologic-experienced patient segments were disproportionately affected "
            "compared to biologic-naive ones, what the models predict for both "
            "brands next quarter, whether any data drift could be confounding these "
            "reads, and finally what experiment we should run to test whether adding "
            "rep capacity in the northeast would recover the trend.",
        ],
    )
    def test_dependent_pipeline_tool_composer(self, query):
        assert _classify_and_route(query) == ["tool_composer"]


# ---------------------------------------------------------------------------
# Change 2 negative guards — must NOT over-promote to the 180s tool_composer.
# ---------------------------------------------------------------------------
class TestDependencyPromotionNegatives:
    @pytest.mark.parametrize(
        "query",
        [
            # bench-0143 again: a superlative "the largest gap" is NOT a back-ref to
            # a computed result — genuine parallel, must stay 2-agent.
            "What is the current total TRx and which region has the largest gap opportunity?",
            # locked single-ask-with-marker cases (test_multipart_tool_composer_routing).
            "If the data refresh completes, then forecast next quarter TRx",
            "Compare the causal impact and segment response, then recommend an experiment",
        ],
    )
    def test_not_tool_composer(self, query):
        assert "tool_composer" not in _classify_and_route(query)
