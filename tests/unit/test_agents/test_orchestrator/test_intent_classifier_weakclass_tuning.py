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


# ---------------------------------------------------------------------------
# Codex HIGH (2026-07-31): the dependency markers must be ANAPHORIC, not generic
# preambles. "Given the budget constraints, …" / "If it is possible, …" are NOT
# back-references — they must not promote a 2-intent query to the 180s
# tool_composer. (Red on the pre-fix broad "given the <word>" / "if it is/was"
# markers, which matched these and over-promoted.)
# ---------------------------------------------------------------------------
class TestDependencyMarkersAreAnaphoricNotPreambles:
    @pytest.mark.parametrize(
        "query",
        [
            "Given the budget constraints, design an experiment and explain the result",
            "If it is possible, run an experiment and explain expected impact",
            "Given the current quarter, forecast TRx and explain the variance",
        ],
    )
    def test_preamble_does_not_promote_to_tool_composer(self, query):
        assert "tool_composer" not in _classify_and_route(query)

    def test_genuine_anaphor_still_promotes(self):
        # "if it has" (state back-ref) + 2 mapped intents still routes tool_composer.
        assert _classify_and_route(
            "Check whether the Kisqali adoption model has drifted, and if it has, "
            "re-run the segment analysis and tell me which HCP targets change"
        ) == ["tool_composer"]


# ---------------------------------------------------------------------------
# Codex MED (2026-07-31): a SENTENCE boundary ("." + whitespace) is a real clause
# boundary. Two intent-bearing sentences must not be suppressed to a single agent
# by the clause gate. (Red on the pre-fix splitter, which excluded "." entirely.)
# ---------------------------------------------------------------------------
class TestSentenceBoundarySplitsMultipart:
    def test_two_sentences_two_intents_not_single(self):
        agents = _classify_and_route("What caused the drop. Design an experiment to verify.")
        assert agents == ["causal_impact", "experiment_designer"]

    def test_decimal_not_split(self):
        # "15.5%" must NOT split — the period has no following whitespace, so this
        # single cohort ask stays one dispatch.
        agents = _classify_and_route(
            "Break down Remibrutinib NRx by patient segment where conversion is 15.5%."
        )
        assert len(agents) == 1


# ---------------------------------------------------------------------------
# #1408 (partial) — predictive-model adjective exclusion (Lever A only).
#
# SCOPE (2026-08-01 decision). Of the three post-#1400 residual levers, only the
# bench-0221 fix ships: the `predict` lexeme must not fire on the "predictive
# model" / "predictive analytics" ADJECTIVE (a model-MONITORING subject: ROC-AUC,
# calibration, drift), so system_health / drift_check win those asks. The one
# residual — "predictive model" + a genuine forecast — fails OPEN to the LLM
# fallback, never a confident wrong route.
#
# The bench-0016 "likely"-family broadening (Lever B) and the #1409
# compound-object collapse (Lever C) were EVALUATED, prototyped, and REVERTED:
# adversarial review found both regress phrasings that `main` routes correctly
# today into CONFIDENT misroutes that bypass the LLM safety net —
#   * "likely to have caused/driven X" -> forecaster (should stay causal), and
#   * a terse independent "forecast the uplift, and design a sample-size plan"
#     -> a silently dropped forecast task.
# bench-0016 and all of #1409 defer to the #1406 semantic classifier; both
# issues stay OPEN. The two "guard" tests below pin the currently-correct `main`
# behaviour so neither reverted lever can silently return.
#
# Guarded end-to-end by pattern_diff.py over the full 337 gold: +1 agent-exact
# (bench-0221), 0 losses, 0 escalate-boundary crossings.
# ---------------------------------------------------------------------------
class TestIntentTieBreakResiduals1408:
    def test_bench0221_predictive_model_routes_health_score(self):
        # bench-0221: model-MONITORING subject -> health_score, not prediction.
        assert _classify_and_route(
            "How well is our Kisqali predictive model performing in terms of "
            "ROC-AUC and calibration metrics?"
        ) == ["health_score"]

    def test_predictive_model_adjective_suppressed_but_forecast_verb_fires(self):
        # The `predict` lexeme must NOT match the "predictive model" adjective
        # phrase (model monitoring), but MUST still fire on the forecast verb.
        assert (
            _classify("How is the predictive model performing?")["primary_intent"] != "prediction"
        )
        assert _classify("Predict next quarter TRx for Kisqali")["primary_intent"] == "prediction"

    def test_prediction_model_noun_stays_a_forecast(self):
        # Lever A is scoped to the "predictive" ADJECTIVE only — the "prediction
        # model" NOUN is a live forecast subject and must still route prediction,
        # not fail open. (A broader `ion model` exclusion silently dropped this
        # genuine forecast to the LLM fallback.)
        assert _classify_and_route(
            "What does the prediction model say about Kisqali TRx for next quarter?"
        ) == ["prediction_synthesizer"]

    def test_predictive_model_plus_forecast_fails_open_not_confidently_wrong(self):
        # ACCEPTED residual: "predictive model" (adjective) + a genuine forecast
        # cannot be disambiguated lexically from the bench-0221 MONITORING ask, so
        # prediction is suppressed and the query falls to the LLM fallback (primary
        # "general", confidence below the 0.8 pattern-trust floor) — fails OPEN to
        # the safety net, never a confident wrong route. The monitoring-vs-forecast
        # split here is #1406's semantic job.
        intent = _classify("What does the predictive model say about Kisqali TRx next quarter?")
        assert intent["primary_intent"] == "general"
        assert intent["confidence"] < 0.8


class TestRevertedLeverGuards:
    """Pin the currently-correct `main` behaviour that Levers B and C would break,
    so neither reverted lever can silently return without tripping a test."""

    def test_likely_cause_is_causal_not_prediction(self):
        # Lever B guard: prediction outranks causal_effect in INTENT_PRIORITY, so a
        # "likely"-family broadening would let a causal-ATTRIBUTION ask win the tie
        # and confidently misroute to prediction_synthesizer. On `main` (no such
        # broadening) "likely cause" stays causal_effect -> causal_impact.
        assert _classify_and_route(
            "What is the likely cause of the Kisqali TRx decline in the northeast?"
        ) == ["causal_impact"]
        assert _classify_and_route(
            "What is the most likely cause of the drop in conversion rate?"
        ) == ["causal_impact"]
        # The adversarial case that specifically broke the `likely to` anchor.
        assert _classify_and_route("What is likely to have caused the Kisqali TRx decline?") == [
            "causal_impact"
        ]

    @pytest.mark.parametrize(
        "query",
        [
            # Independent forecast + design pairs must dispatch BOTH agents — a
            # compound-object collapse would silently drop the forecast task.
            "Forecast next quarter Kisqali TRx, and separately, design an A/B test "
            "for the new copay assistance program.",
            "Predict Q4 TRx for Kisqali. Also, design an experiment testing the new rep messaging.",
            # The terse phrasing that broke Lever C's tight-window anchor (effect
            # + sample-size within 30 chars, but two genuinely independent tasks).
            "Forecast the expected uplift, and design a sample size plan for the trial.",
        ],
    )
    def test_independent_forecast_and_design_not_collapsed(self, query):
        # Lever C guard: without the compound-object collapse these stay parallel.
        agents = _classify_and_route(query)
        assert "prediction_synthesizer" in agents and "experiment_designer" in agents, (
            f"{query!r} wrongly collapsed to {agents}; an independent forecast task "
            f"must not be dropped by a compound-object rule"
        )


# ---------------------------------------------------------------------------
# #1409 — digital-twin simulation READOUT collapses to a single experiment_designer.
#
# A query about a digital-twin simulation's outputs asks about the {expected
# effect size, required sample size} pair as a compound OBJECT. The effect-size
# term ("expected lift"/"projected gains") co-fires `prediction`, so the clause
# gate counts two facets and the pre-fix router split these into
# [experiment_designer, prediction_synthesizer] — silently dispatching a forecast
# agent. But a digital-twin pre-screen AND its power-analysis sample-size output
# are BOTH explicit experiment_designer covers: one object, one task. The
# collapse is gated on the READOUT SHAPE (see _DIGITAL_TWIN_READOUT_RE — "digital
# twin" + a nearby readout verb say/report/results/show/indicate/tell/reveal),
# NOT the bare "digital twin" object: an imperative DIRECTIVE that merely USES a
# twin ("forecast NRx using the digital twin, and separately design ...") names
# no readout verb by the twin and stays parallel. Rows are verbatim gold
# (bench-0018/0199/0200), all gold_agents==experiment_designer.
# ---------------------------------------------------------------------------
class TestDigitalTwinReadoutCollapses1409:
    @pytest.mark.parametrize(
        "query",
        [
            # bench-0018
            "What did the digital twin simulation say about expected lift and sample size?",
            # bench-0199
            "According to the digital twin simulation results, what were the projected "
            "performance gains and the required sample size for statistical validity?",
            # bench-0200 (typo variant)
            "what did the digital twin sim say about expected lift and sample size??",
        ],
    )
    def test_digital_twin_readout_single_experiment_designer(self, query):
        assert _classify_and_route(query) == ["experiment_designer"], (
            f"{query!r} should be one experiment_designer task (digital-twin readout), "
            f"not split with a spurious prediction_synthesizer"
        )

    def test_collapse_anchored_on_readout_shape_not_bare_object(self):
        # NEGATIVE guards: the collapse must fire ONLY on a digital-twin READOUT,
        # never on a bare {experiment_design, prediction} pair. Both a pair with no
        # twin AND — the reviewer's counterexample — an imperative DIRECTIVE that
        # merely USES a digital twin for a forecast while separately directing a
        # design must stay parallel so the forecast task is never dropped.
        for query in (
            # No digital-twin object at all.
            "Forecast the expected lift, and design a sample size plan for the trial.",
            # Digital twin PRESENT but as a forecasting TOOL in an independent
            # directive pair — no readout verb next to the twin (reviewer MEDIUM).
            "Forecast Kisqali NRx next quarter using the digital twin, and separately "
            "design an A/B test for the speaker program.",
        ):
            agents = _classify_and_route(query)
            assert "prediction_synthesizer" in agents and "experiment_designer" in agents, (
                f"{query!r} wrongly collapsed to {agents}; only a digital-twin readout "
                f"(twin + a nearby readout verb) may collapse, never an independent directive"
            )


# ---------------------------------------------------------------------------
# #1408 — single-agent tie-break recoveries (subset shippable deterministically).
#
# Two rows where the incumbent picked the wrong single agent and a PRINCIPLED,
# single-row-blast-radius keyword recovers gold without regressing any other gold
# row or crossing the escalate boundary (proven end-to-end by pattern_diff.py):
#   * bench-0177: "create a patient segment ... restricted to [age/diagnosis]" is
#     cohort CONSTRUCTION -> cohort_profiler, not the segment_analysis CATE ask.
#   * bench-0282: an "interim readout" is a report on an IN-PROGRESS experiment ->
#     experiment_monitor, not experiment_designer (whose "run(ning)...test"
#     co-match otherwise wins).
# The larger segment_analysis->heterogeneous_optimizer cluster and the
# experiment_monitor rows whose only clean anchor crosses the escalate boundary
# (bench-0283/0285/0289) are deferred to the #1406 semantic classifier.
# ---------------------------------------------------------------------------
class TestSingleAgentTieBreakRecovered1408:
    def test_create_patient_segment_is_cohort_construction(self):
        # bench-0177
        assert _classify_and_route(
            "I need to create a patient segment for Remibrutinib indicated for chronic "
            "spontaneous urticaria, restricted to individuals aged 18 and above who "
            "received their initial diagnosis during the 2024 calendar year."
        ) == ["cohort_profiler"]

    def test_construction_verb_gate_keeps_analysis_asks_out(self):
        # NEGATIVE guard: a bare "which patient segments ..." analysis ask carries
        # no construction verb, so it must NOT be pulled into cohort_definition by
        # the new "patient segment" object — it stays segment_analysis.
        assert _classify_and_route(
            "Which patient segments have the worst adherence for Fabhalta?"
        ) == ["heterogeneous_optimizer"]

    @pytest.mark.parametrize(
        "query",
        [
            # Reviewer probe: a construction verb EARLIER in the sentence (bound to
            # its own object) must not reach across >3 words to "patient segments"
            # and misroute a segment_analysis ask to cohort.
            "As we define our GTM strategy, which patient segments have the lowest engagement?",
            "Can you build a report showing how the patient segment responded to the "
            "new copay program?",
        ],
    )
    def test_far_construction_verb_does_not_pull_segment_to_cohort(self, query):
        assert "cohort_profiler" not in _classify_and_route(query), (
            f"{query!r} is a segment_analysis ask; a distant construction verb must "
            f"not bound-jump to 'patient segment' and route it to cohort"
        )

    def test_interim_readout_routes_experiment_monitor(self):
        # bench-0282
        assert _classify_and_route(
            "Give me an interim readout on the running Fabhalta speaker-program test"
        ) == ["experiment_monitor"]

    @pytest.mark.parametrize(
        "query",
        [
            # Reviewer HIGH: bare "interim" over-fired experiment_monitor (highest
            # priority intent, so it won the tie and skipped the LLM) on
            # non-experiment asks. Domain-gated now — no experiment noun co-occurs,
            # so these must NOT route experiment_monitor.
            "What's our interim CFO's view on Q3 pharma spend?",
            "summarize the interim guidance from FDA on labeling for Fabhalta",
            "Who is the interim head of oncology commercial?",
        ],
    )
    def test_interim_without_experiment_noun_not_monitor(self, query):
        assert "experiment_monitor" not in _classify_and_route(query), (
            f"{query!r} names no experiment; 'interim' alone must not route to experiment_monitor"
        )

    def test_in_the_interim_causal_stays_causal(self):
        # Reviewer HIGH: a real regression of a previously-correct row — the
        # "in the interim" preamble must not steal a causal-attribution ask.
        assert _classify_and_route(
            "In the interim, can you tell me what caused the Kisqali TRx drop?"
        ) == ["causal_impact"]

    def test_design_ask_still_routes_experiment_designer(self):
        # NEGATIVE guard: a genuine design ask (no "interim"/monitoring signal)
        # still routes experiment_designer — the interim broadening did not steal
        # the design class.
        assert _classify_and_route(
            "Design an experiment to measure whether speaker programs increase Fabhalta NRx"
        ) == ["experiment_designer"]
