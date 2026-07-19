"""Unit tests for the DSPy-lane provider A/B harness logic.

Pure-logic coverage only (scoring, aggregation, gates, bundle emission).
The real LLM calls happen when the emitted bundle runs in the prod container;
nothing here mocks a model, and no test performs a network call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.optimization.dspy_lane_ab import (
    GATE_ACCURACY_MARGIN,
    GATE_E2E_LATENCY_FACTOR,
    GATE_RAGAS_MARGIN,
    emit_container_script,
    evaluate_gates,
    is_correct,
    load_golden_set,
    production_chatbot_intent,
    summarize_e2e_runs,
    summarize_signature_runs,
)

FIXTURE_PATH = Path(__file__).parents[3] / "tests" / "fixtures" / "dspy_lane_golden_queries.json"


# ---------------------------------------------------------------------------
# is_correct
# ---------------------------------------------------------------------------


def test_is_correct_exact_match_case_sensitive():
    # Production code compares intent strings exactly (cognitive_rag_dspy.py:576,
    # chatbot_dspy.py:2062), so strict scoring must be case-sensitive.
    assert is_correct("CAUSAL_ANALYSIS", ["CAUSAL_ANALYSIS"]) is True
    assert is_correct("causal_analysis", ["CAUSAL_ANALYSIS"]) is False
    assert is_correct(" CAUSAL_ANALYSIS ", ["CAUSAL_ANALYSIS"]) is True  # whitespace stripped


def test_is_correct_acceptable_set():
    assert is_correct("GENERAL", ["GENERAL", "EXPLANATION"]) is True
    assert is_correct("PREDICTION", ["GENERAL", "EXPLANATION"]) is False


def test_is_correct_excluded_and_missing():
    assert is_correct("anything", None) is None  # query excluded from this taxonomy
    assert is_correct(None, ["GENERAL"]) is False  # parse failure -> wrong


def test_is_correct_lenient_mode():
    assert is_correct("Causal_Analysis", ["CAUSAL_ANALYSIS"], lenient=True) is True
    assert is_correct('"kpi_query".', ["kpi_query"], lenient=True) is True
    assert is_correct("kpi query", ["kpi_query"], lenient=False) is False


# ---------------------------------------------------------------------------
# summarize_signature_runs
# ---------------------------------------------------------------------------


def _rec(
    model="m",
    taxonomy="cognitive_rag",
    qid="q1",
    predicted="GENERAL",
    acceptable=("GENERAL",),
    latency=1.0,
    error=None,
):
    return {
        "model": model,
        "taxonomy": taxonomy,
        "query_id": qid,
        "predicted": predicted,
        "acceptable": list(acceptable) if acceptable is not None else None,
        "latency_s": latency,
        "error": error,
    }


def test_summarize_signature_runs_basic():
    records = [
        _rec(qid="q1", predicted="GENERAL", acceptable=["GENERAL"], latency=1.0),
        _rec(qid="q2", predicted="PREDICTION", acceptable=["GENERAL"], latency=2.0),
        _rec(qid="q3", predicted="general", acceptable=["GENERAL"], latency=3.0),
        _rec(qid="q4", predicted="x", acceptable=None, latency=4.0),  # excluded
        _rec(
            qid="q5",
            predicted=None,
            acceptable=["GENERAL"],
            latency=5.0,
            error="AuthenticationError: boom",
        ),
    ]
    s = summarize_signature_runs(records)["m"]["cognitive_rag"]
    assert s["n_scored"] == 4
    assert s["n_excluded"] == 1
    assert s["accuracy_strict"] == pytest.approx(1 / 4)
    assert s["accuracy_lenient"] == pytest.approx(2 / 4)
    assert s["parse_failure_rate"] == pytest.approx(1 / 4)
    assert s["error_classes"] == {"AuthenticationError": 1}
    # excluded record still contributes latency (the call really happened)
    assert s["latency_p50"] == pytest.approx(3.0)


def test_summarize_signature_runs_groups_by_model_and_taxonomy():
    records = [
        _rec(model="a", taxonomy="cognitive_rag"),
        _rec(model="a", taxonomy="chatbot", predicted="kpi_query", acceptable=["kpi_query"]),
        _rec(model="b", taxonomy="cognitive_rag"),
    ]
    s = summarize_signature_runs(records)
    assert set(s) == {"a", "b"}
    assert set(s["a"]) == {"cognitive_rag", "chatbot"}


# ---------------------------------------------------------------------------
# summarize_e2e_runs
# ---------------------------------------------------------------------------


def test_summarize_e2e_runs():
    records = [
        {
            "model": "m",
            "query_id": "q1",
            "latency_s": 10.0,
            "hop_count": 2,
            "evidence_count": 5,
            "answer_chars": 900,
            "error": None,
        },
        {
            "model": "m",
            "query_id": "q2",
            "latency_s": 30.0,
            "hop_count": 3,
            "evidence_count": 7,
            "answer_chars": 1100,
            "error": None,
        },
        {
            "model": "m",
            "query_id": "q3",
            "latency_s": 5.0,
            "hop_count": 0,
            "evidence_count": 0,
            "answer_chars": 0,
            "error": "TimeoutError: slow",
        },
    ]
    s = summarize_e2e_runs(records)["m"]
    assert s["n"] == 3
    assert s["n_errors"] == 1
    assert s["latency_p50"] == pytest.approx(10.0)
    assert s["error_classes"] == {"TimeoutError": 1}
    assert s["mean_hops"] == pytest.approx((2 + 3) / 2)  # errored runs excluded


# ---------------------------------------------------------------------------
# evaluate_gates
# ---------------------------------------------------------------------------


def _ragas_per_sample(faith, rel=0.80, n=10, n_ctx=10):
    # First n_ctx replays carry retrieved contexts (and thus a faithfulness
    # score); the rest mirror the judge's no-context rows (score None).
    return [
        {
            "query_id": f"q{i}",
            "n_contexts": 1 if i < n_ctx else 0,
            "faithfulness": faith if i < n_ctx else None,
            "answer_relevancy": rel,
        }
        for i in range(n)
    ]


def _ragas_block(faith=0.85, rel=0.80, n_ctx=10):
    # Aggregates mirror what the judge would compute from these rows, so the
    # block is internally consistent (tests that need inconsistency corrupt it
    # explicitly).
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "n_samples": 10,
        "n_faithfulness": n_ctx,
        "per_sample": _ragas_per_sample(faith, rel=rel, n_ctx=n_ctx),
    }


# Query-id multisets shared by baseline and candidate fixtures - the
# signature_query_set gate requires the two sides to have been measured on
# exactly the same golden queries (codex iter-6).
_COG_SCORED_IDS = [f"g{i}" for i in range(40)]
_CHAT_SCORED_IDS = [f"g{i}" for i in range(38)]
_CHAT_EXCLUDED_IDS = ["g38", "g39"]

# What the golden set says SHOULD have been measured - the anchor the
# signature_golden_anchor gate holds both sides against (codex iter-7: two
# sides sharing the same truncated subset pass every side-to-side gate).
EXPECTED_SIG_SETS = {
    "cognitive_rag": {
        "scored_query_ids": sorted(_COG_SCORED_IDS),
        "excluded_query_ids": [],
    },
    "chatbot": {
        "scored_query_ids": sorted(_CHAT_SCORED_IDS),
        "excluded_query_ids": sorted(_CHAT_EXCLUDED_IDS),
    },
}

BASELINE = {
    "signature": {
        "cognitive_rag": {
            "accuracy_strict": 0.80,
            "parse_failure_rate": 0.0,
            "error_classes": {},
            "n_scored": 40,
            "n_excluded": 0,
            "scored_query_ids": list(_COG_SCORED_IDS),
            "excluded_query_ids": [],
        },
        "chatbot": {
            "accuracy_strict": 0.90,
            "parse_failure_rate": 0.0,
            "error_classes": {},
            "n_scored": 38,
            "n_excluded": 2,
            "scored_query_ids": list(_CHAT_SCORED_IDS),
            "excluded_query_ids": list(_CHAT_EXCLUDED_IDS),
        },
    },
    "ragas": _ragas_block(),
    "e2e": {"latency_p50": 40.0, "error_classes": {}, "n": 10},
}


def _candidate(
    cog_acc=0.80,
    chat_acc=0.90,
    parse_fail=0.0,
    faith=0.85,
    rel=0.80,
    p50=40.0,
    sig_errors=None,
    e2e_errors=None,
    n_ctx=10,
):
    return {
        "signature": {
            "cognitive_rag": {
                "accuracy_strict": cog_acc,
                "parse_failure_rate": parse_fail,
                "error_classes": sig_errors or {},
                "n_scored": 40,
                "n_excluded": 0,
                "scored_query_ids": list(_COG_SCORED_IDS),
                "excluded_query_ids": [],
            },
            "chatbot": {
                "accuracy_strict": chat_acc,
                "parse_failure_rate": parse_fail,
                "error_classes": sig_errors or {},
                "n_scored": 38,
                "n_excluded": 2,
                "scored_query_ids": list(_CHAT_SCORED_IDS),
                "excluded_query_ids": list(_CHAT_EXCLUDED_IDS),
            },
        },
        "ragas": _ragas_block(faith=faith, rel=rel, n_ctx=n_ctx),
        "e2e": {"latency_p50": p50, "error_classes": e2e_errors or {}, "n": 10},
    }


def test_gates_all_pass_at_parity():
    result = evaluate_gates(BASELINE, _candidate(), EXPECTED_SIG_SETS)
    assert result["all_passed"] is True
    assert all(g["passed"] for g in result["gates"])


def test_gate_accuracy_margin():
    # 5pp below baseline is allowed, more is not
    ok = evaluate_gates(
        BASELINE, _candidate(cog_acc=0.80 - GATE_ACCURACY_MARGIN), EXPECTED_SIG_SETS
    )
    assert ok["all_passed"] is True
    bad = evaluate_gates(
        BASELINE, _candidate(cog_acc=0.80 - GATE_ACCURACY_MARGIN - 0.01), EXPECTED_SIG_SETS
    )
    assert bad["all_passed"] is False
    failed = [g for g in bad["gates"] if not g["passed"]]
    assert any("accuracy" in g["name"] for g in failed)


def test_gate_parse_failure_must_not_exceed_baseline():
    bad = evaluate_gates(BASELINE, _candidate(parse_fail=0.05), EXPECTED_SIG_SETS)
    assert bad["all_passed"] is False


def test_gate_ragas_margin():
    ok = evaluate_gates(BASELINE, _candidate(faith=0.85 - GATE_RAGAS_MARGIN), EXPECTED_SIG_SETS)
    assert ok["all_passed"] is True
    bad = evaluate_gates(
        BASELINE, _candidate(rel=0.80 - GATE_RAGAS_MARGIN - 0.01), EXPECTED_SIG_SETS
    )
    assert bad["all_passed"] is False


def test_gate_no_new_error_class():
    # error classes already seen in baseline do not fail the candidate
    baseline = json.loads(json.dumps(BASELINE))
    baseline["e2e"]["error_classes"] = {"TimeoutError": 1}
    ok = evaluate_gates(baseline, _candidate(e2e_errors={"TimeoutError": 2}), EXPECTED_SIG_SETS)
    assert ok["all_passed"] is True
    bad = evaluate_gates(
        baseline, _candidate(sig_errors={"AuthenticationError": 1}), EXPECTED_SIG_SETS
    )
    assert bad["all_passed"] is False


def test_gate_e2e_latency_factor():
    ok = evaluate_gates(BASELINE, _candidate(p50=40.0 * GATE_E2E_LATENCY_FACTOR), EXPECTED_SIG_SETS)
    assert ok["all_passed"] is True
    bad = evaluate_gates(
        BASELINE, _candidate(p50=40.0 * GATE_E2E_LATENCY_FACTOR + 0.1), EXPECTED_SIG_SETS
    )
    assert bad["all_passed"] is False


def test_gate_missing_ragas_fails_closed():
    candidate = _candidate()
    candidate["ragas"] = None
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False


def test_gate_ragas_per_metric_none_fails_closed():
    # The judge emits {"faithfulness": None, ...} when zero judged samples had
    # retrieved contexts - a truthy dict that must fail the gate, not TypeError.
    candidate = _candidate()
    candidate["ragas"]["faithfulness"] = None
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" in failed


def test_gate_ragas_per_metric_none_baseline_fails_closed():
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"]["answer_relevancy"] = None
    result = evaluate_gates(baseline, _candidate(), EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[answer_relevancy]" in failed


def test_gate_ragas_completeness_requires_all_replays_judged():
    # Errored/empty-answer replays are excluded from judging by
    # build_ragas_samples; the gate must surface that as a failure instead of
    # letting RAGAS score only the candidate's easier successful subset.
    candidate = _candidate()
    candidate["ragas"]["n_samples"] = 8
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[completeness]" in failed


def test_gate_ragas_completeness_missing_counts_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["n_samples"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[completeness]" in failed


def test_gate_ragas_faithfulness_coverage_candidate_below_baseline():
    # Faithfulness averages only context-bearing replays; a candidate judged on
    # fewer of them than the baseline is being scored on a different (easier)
    # denominator even when n_samples matches (codex iter-2).
    candidate = _candidate(n_ctx=8)
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" in failed


def test_gate_ragas_faithfulness_coverage_min_n_floor():
    # A faithfulness mean over 1-2 replays is noise, not signal - both sides
    # need at least GATE_RAGAS_MIN_FAITHFULNESS_N context-bearing replays.
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"] = _ragas_block(n_ctx=2)
    result = evaluate_gates(baseline, _candidate(n_ctx=2), EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" in failed


def test_gate_ragas_faithfulness_coverage_missing_count_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["n_faithfulness"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" in failed


def test_gate_ragas_codex_iter2_scenario_tiny_context_subset():
    # Codex iter-2 HIGH: all 10 replays answered (completeness PASSES) but the
    # candidate retrieved contexts on only 1, so its faithfulness mean is a
    # single easy sample that clears the margin. Must still fail.
    candidate = _candidate(faith=0.90, n_ctx=1)
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[completeness]" not in failed  # the hole iter-2 identified
    assert "ragas[faithfulness]" not in failed  # margin cleared on the tiny mean
    assert "ragas[faithfulness_coverage]" in failed
    assert "ragas[faithfulness_common_subset]" in failed


def test_gate_ragas_common_subset_disjoint_contexts_fails_closed():
    # Equal coverage counts can still mean DIFFERENT judged subsets; with no
    # overlap there is no apples-to-apples comparison at all.
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"] = _ragas_block(n_ctx=5)
    candidate = _candidate(n_ctx=5)
    for row in candidate["ragas"]["per_sample"]:
        row["n_contexts"] = 0 if row["n_contexts"] else 1
        row["faithfulness"] = None if row["n_contexts"] == 0 else 0.85
    result = evaluate_gates(baseline, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_common_subset]" in failed


def test_gate_ragas_common_subset_recomputes_from_per_sample():
    # The common-subset gate must trust per-sample rows over the aggregate: a
    # candidate whose aggregate clears the margin but whose per-sample scores
    # on the shared replays do not must fail.
    candidate = _candidate(faith=0.85)
    for row in candidate["ragas"]["per_sample"]:
        row["faithfulness"] = 0.60
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" not in failed  # aggregate said 0.85
    assert "ragas[faithfulness_common_subset]" in failed


def test_gate_ragas_common_subset_missing_per_sample_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["per_sample"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_common_subset]" in failed


def test_gate_ragas_consistency_codex_iter3_fabricated_aggregate():
    # Codex iter-3 HIGH: every other RAGAS gate reads the reported aggregate
    # fields; a stale/hand-edited block claiming n_samples=10, n_faithfulness=10,
    # faithfulness=0.80 while its actual per_sample holds only 3 decent rows on
    # the baseline's shared replays passes value, completeness, coverage AND
    # common-subset. Only aggregate-vs-rows reconciliation catches it.
    candidate = _candidate(faith=0.80)
    candidate["ragas"]["per_sample"] = _ragas_per_sample(0.90, n=3, n_ctx=3)
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" not in failed  # fabricated aggregate clears margin
    assert "ragas[completeness]" not in failed  # fabricated n_samples matches e2e.n
    assert "ragas[faithfulness_coverage]" not in failed  # fabricated n_faithfulness
    assert "ragas[faithfulness_common_subset]" not in failed  # 3 real rows look fine
    assert "ragas[consistency]" in failed
    assert result["all_passed"] is False


def test_gate_ragas_consistency_n_faithfulness_mismatch():
    # Reported context-bearing count must equal what the rows actually show.
    candidate = _candidate(n_ctx=8)
    candidate["ragas"]["n_faithfulness"] = 10
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" not in failed  # fabricated count passes it
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_aggregate_value_mismatch():
    # A reported mean that differs from what the rows recompute to - subtly
    # enough that both the value gate and the common-subset gate still pass -
    # must fail reconciliation.
    candidate = _candidate(faith=0.85)
    for row in candidate["ragas"]["per_sample"]:
        if row["n_contexts"]:
            row["faithfulness"] = 0.84
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" not in failed
    assert "ragas[faithfulness_common_subset]" not in failed
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_baseline_side_checked():
    # A corrupted-low baseline aggregate weakens every floor; reconciliation
    # must validate the baseline block too, not just the candidate.
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"]["faithfulness"] = 0.30
    result = evaluate_gates(baseline, _candidate(), EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_relevancy_mismatch():
    candidate = _candidate()
    for row in candidate["ragas"]["per_sample"]:
        row["answer_relevancy"] = 0.70
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[answer_relevancy]" not in failed  # aggregate still says 0.80
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_missing_per_sample_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["per_sample"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[consistency]" in failed


def test_gate_ragas_fully_scored_codex_iter4_scoreless_ctx_rows():
    # Codex iter-4 HIGH: n_faithfulness counts context-bearing rows while the
    # faithfulness mean (and the consistency recompute that mirrors it) skips
    # None scores. A candidate whose judge covered 10 rows but scored only the
    # 3 shared with the baseline stays internally consistent and clears value,
    # completeness, coverage AND common-subset - a 3-row measurement wearing
    # 10-row coverage. Only a scored-row requirement catches it.
    candidate = _candidate(faith=0.85)
    for row in candidate["ragas"]["per_sample"][3:]:
        row["faithfulness"] = None  # covered (n_contexts=1) but never scored
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[consistency]" not in failed  # judge-mirror recompute skips None
    assert "ragas[faithfulness]" not in failed
    assert "ragas[completeness]" not in failed
    assert "ragas[faithfulness_coverage]" not in failed  # counts covered, not scored
    assert "ragas[faithfulness_common_subset]" not in failed  # q0-q2 still align
    assert "ragas[fully_scored]" in failed
    assert result["all_passed"] is False


def test_gate_ragas_fully_scored_baseline_side_checked():
    # A baseline whose covered rows are partly scoreless understates its own
    # denominator the same way; both sides must be fully scored.
    baseline = json.loads(json.dumps(BASELINE))
    for row in baseline["ragas"]["per_sample"][5:]:
        row["faithfulness"] = None
    result = evaluate_gates(baseline, _candidate(), EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[fully_scored]" in failed


def test_gate_ragas_fully_scored_relevancy_scoreless():
    # answer_relevancy averages over all rows with the same None-skip, so a
    # scoreless row hides there identically.
    candidate = _candidate()
    candidate["ragas"]["per_sample"][0]["answer_relevancy"] = None
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[answer_relevancy]" not in failed  # mean over the other 9 = 0.80
    assert "ragas[consistency]" not in failed
    assert "ragas[fully_scored]" in failed


def test_gate_ragas_fully_scored_ignores_uncovered_rows():
    # Rows the judge never retrieved contexts for (n_contexts=0, score None)
    # are legitimately scoreless - only covered-but-unscored rows fail.
    result = evaluate_gates(BASELINE, _candidate(n_ctx=7), EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[fully_scored]" not in failed


def test_gate_ragas_fully_scored_missing_per_sample_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["per_sample"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[fully_scored]" in failed


def test_gate_signature_denominator_partial_candidate_codex_iter5():
    # Codex iter-5 HIGH: the parse/accuracy gates compare rates but never the
    # denominator, so a truncated candidate summary measured over a single
    # query (n_scored=1, accuracy 1.0) beat a full-set baseline on every
    # signature gate before the denominator gate existed.
    candidate = _candidate(chat_acc=1.0)
    candidate["signature"]["chatbot"]["n_scored"] = 1
    candidate["signature"]["chatbot"]["n_excluded"] = 0
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "accuracy[chatbot]" not in failed  # 1.0 clears the margin
    assert "parse_failure[chatbot]" not in failed
    assert "signature_denominator[chatbot]" in failed
    assert "signature_denominator[cognitive_rag]" not in failed
    assert result["all_passed"] is False


def test_gate_signature_denominator_missing_counts_fails_closed():
    candidate = _candidate()
    del candidate["signature"]["cognitive_rag"]["n_scored"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_denominator[cognitive_rag]" in failed
    assert result["all_passed"] is False


def test_gate_signature_rates_none_fails_closed():
    # summarize_signature_runs emits None rates when n_scored=0; the rate
    # gates must fail closed on them, not raise on a None comparison.
    candidate = _candidate()
    candidate["signature"]["chatbot"].update(
        {"n_scored": 0, "n_excluded": 40, "accuracy_strict": None, "parse_failure_rate": None}
    )
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "accuracy[chatbot]" in failed
    assert "parse_failure[chatbot]" in failed
    assert "signature_denominator[chatbot]" in failed
    assert result["all_passed"] is False


def test_production_chatbot_intent_matches_prod_normalizer():
    # Codex iter-5 MED: production routes on _normalize_intent(pred.intent),
    # so the harness must score the value production consumes. Compared
    # against the real normalizer - casing and aliases map exactly.
    from src.api.routes.chatbot_dspy import _normalize_intent

    assert production_chatbot_intent(None) is None
    for raw in ["KPI_QUERY", " kpi ", "kpi query", "causal_analysis", "Greetings"]:
        assert production_chatbot_intent(raw) == _normalize_intent(raw)
    assert production_chatbot_intent("KPI_QUERY") == "kpi_query"


def test_gate_signature_query_set_same_count_different_ids_codex_iter6():
    # Codex iter-6 HIGH: matching (n_scored, n_excluded) cannot prove the two
    # sides measured the same queries - a merged file after a partial run can
    # swap a hard query for an easy one while preserving the counts.
    candidate = _candidate()
    ids = candidate["signature"]["chatbot"]["scored_query_ids"]
    ids[0] = "g99"  # same count, different membership
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_denominator[chatbot]" not in failed  # counts still match
    assert "signature_query_set[chatbot]" in failed
    assert "signature_query_set[cognitive_rag]" not in failed
    assert result["all_passed"] is False


def test_gate_signature_query_set_duplicate_ids_fail():
    # A duplicated easy query replacing a dropped hard one keeps the count
    # but changes the multiset.
    candidate = _candidate()
    ids = candidate["signature"]["cognitive_rag"]["scored_query_ids"]
    ids[1] = ids[0]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_query_set[cognitive_rag]" in failed
    assert result["all_passed"] is False


def test_gate_signature_query_set_missing_ids_fails_closed():
    candidate = _candidate()
    del candidate["signature"]["cognitive_rag"]["scored_query_ids"]
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_query_set[cognitive_rag]" in failed
    assert result["all_passed"] is False


def test_gate_signature_query_set_count_id_mismatch_fails_closed():
    # A block whose counts disagree with its own id lists is malformed.
    candidate = _candidate()
    candidate["signature"]["chatbot"]["n_scored"] = 37
    candidate["signature"]["chatbot"]["n_excluded"] = 3
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_query_set[chatbot]" in failed
    assert result["all_passed"] is False


def test_gate_signature_missing_baseline_block_fails_closed_codex_iter6():
    # Codex iter-6 HIGH: iterating baseline keys let an empty baseline
    # signature block emit ZERO signature gates - all_passed was True with no
    # signature comparison at all in the red run.
    import copy

    baseline = copy.deepcopy(BASELINE)
    baseline["signature"] = {}
    result = evaluate_gates(baseline, _candidate(), EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature[cognitive_rag]" in failed
    assert "signature[chatbot]" in failed
    assert result["all_passed"] is False


def test_gate_signature_missing_baseline_taxonomy_fails_closed():
    import copy

    baseline = copy.deepcopy(BASELINE)
    del baseline["signature"]["chatbot"]
    result = evaluate_gates(baseline, _candidate(), EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature[chatbot]" in failed
    assert "signature[cognitive_rag]" not in failed
    assert result["all_passed"] is False


def test_gate_e2e_latency_non_finite_fails_closed_codex_iter6():
    # Codex iter-6 MED: an inf baseline made the limit inf (any candidate
    # passed) and a string baseline raised TypeError instead of failing.
    import copy

    for bad in [float("inf"), float("nan"), "40.0"]:
        baseline = copy.deepcopy(BASELINE)
        baseline["e2e"]["latency_p50"] = bad
        result = evaluate_gates(baseline, _candidate(p50=1e9), EXPECTED_SIG_SETS)
        lat = [g for g in result["gates"] if g["name"] == "e2e_latency_p50"]
        assert len(lat) == 1 and lat[0]["passed"] is False
        assert result["all_passed"] is False
    candidate = _candidate()
    candidate["e2e"]["latency_p50"] = float("nan")
    result = evaluate_gates(BASELINE, candidate, EXPECTED_SIG_SETS)
    lat = [g for g in result["gates"] if g["name"] == "e2e_latency_p50"]
    assert len(lat) == 1 and lat[0]["passed"] is False


def test_gate_signature_golden_anchor_shared_truncated_subset_codex_iter7():
    # Codex iter-7 MED: every earlier signature gate is side-to-side, so a
    # results file where baseline AND candidate carry the same truncated
    # easy subset passes them all - nothing anchors either side to what the
    # golden set says SHOULD have been measured.
    import copy

    baseline = copy.deepcopy(BASELINE)
    candidate = _candidate()
    for side in (baseline, candidate):
        chat = side["signature"]["chatbot"]
        chat["scored_query_ids"] = sorted(_CHAT_SCORED_IDS[:20])
        chat["n_scored"] = 20
    result = evaluate_gates(baseline, candidate, EXPECTED_SIG_SETS)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_query_set[chatbot]" not in failed  # sides match each other
    assert "signature_denominator[chatbot]" not in failed  # counts match too
    assert "signature_golden_anchor[chatbot]" in failed
    assert "signature_golden_anchor[cognitive_rag]" not in failed
    assert result["all_passed"] is False


def test_gate_signature_golden_anchor_missing_expected_fails_closed():
    # A caller that forgets the golden anchor gets failed gates, not a pass.
    result = evaluate_gates(BASELINE, _candidate())
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "signature_golden_anchor[cognitive_rag]" in failed
    assert "signature_golden_anchor[chatbot]" in failed
    assert result["all_passed"] is False


def test_expected_signature_sets_from_real_fixture():
    # The anchor must derive from the shipped golden set: every query id is
    # either scored or excluded per taxonomy, matching the recorded run's
    # (40, 0) cognitive and (38, 2) chatbot shape.
    from src.optimization.dspy_lane_ab import expected_signature_sets

    golden = load_golden_set(FIXTURE_PATH)
    expected = expected_signature_sets(golden)
    all_ids = sorted(item["id"] for item in golden["queries"])
    assert set(expected) == {"cognitive_rag", "chatbot"}
    for taxonomy in expected:
        scored = expected[taxonomy]["scored_query_ids"]
        excluded = expected[taxonomy]["excluded_query_ids"]
        assert sorted(scored + excluded) == all_ids
    assert len(expected["cognitive_rag"]["scored_query_ids"]) == 40
    assert expected["cognitive_rag"]["excluded_query_ids"] == []
    assert len(expected["chatbot"]["scored_query_ids"]) == 38
    assert len(expected["chatbot"]["excluded_query_ids"]) == 2


_MINI_GOLDEN = {
    "queries": [
        {"id": "q1", "query": "x", "expected_cognitive": ["A"], "expected_chatbot": ["kpi_query"]},
        {"id": "q2", "query": "y", "expected_cognitive": ["B"], "expected_chatbot": None},
    ]
}


def test_rebind_acceptable_labels_restores_golden_truth_codex_iter8():
    # Codex iter-8 HIGH: the anchor validates query-id membership but scoring
    # trusted each record's own acceptable label - widening one hard query's
    # set to include its wrong prediction passed every gate. Rebinding from
    # the golden fixture makes the tampered/stale label irrelevant.
    from src.optimization.dspy_lane_ab import rebind_acceptable_labels

    records = [
        {
            "model": "m",
            "taxonomy": "cognitive_rag",
            "query_id": "q1",
            "predicted": "WRONG",
            "acceptable": ["WRONG"],  # tampered: golden says ["A"]
            "latency_s": 0.1,
            "error": None,
        }
    ]
    tampered = summarize_signature_runs(records)["m"]["cognitive_rag"]
    assert tampered["accuracy_strict"] == 1.0  # the exploit
    rebound = rebind_acceptable_labels(records, _MINI_GOLDEN)
    honest = summarize_signature_runs(rebound)["m"]["cognitive_rag"]
    assert honest["accuracy_strict"] == 0.0
    assert records[0]["acceptable"] == ["WRONG"]  # input not mutated


def test_rebind_acceptable_labels_preserves_exclusions():
    from src.optimization.dspy_lane_ab import rebind_acceptable_labels

    records = [
        {
            "model": "m",
            "taxonomy": "chatbot",
            "query_id": "q2",
            "predicted": "kpi_query",
            "acceptable": ["kpi_query"],  # stale: golden excludes q2 from chatbot
            "latency_s": 0.1,
            "error": None,
        }
    ]
    rebound = rebind_acceptable_labels(records, _MINI_GOLDEN)
    assert rebound[0]["acceptable"] is None
    s = summarize_signature_runs(rebound)["m"]["chatbot"]
    assert s["n_scored"] == 0 and s["n_excluded"] == 1


def test_rebind_acceptable_labels_unknown_query_raises():
    from src.optimization.dspy_lane_ab import rebind_acceptable_labels

    records = [
        {
            "model": "m",
            "taxonomy": "cognitive_rag",
            "query_id": "q99",
            "predicted": "A",
            "acceptable": ["A"],
            "latency_s": 0.1,
            "error": None,
        }
    ]
    with pytest.raises(ValueError, match="q99"):
        rebind_acceptable_labels(records, _MINI_GOLDEN)


def test_stored_summary_divergences_codex_iter8():
    from src.optimization.dspy_lane_ab import stored_summary_divergences

    recomputed = {"m": {"chatbot": {"n_scored": 2, "accuracy_strict": 0.5, "new_field": [1]}}}
    # identical stored keys -> no divergence; additive recomputed-only fields
    # never trigger on old files
    assert stored_summary_divergences({"m": {"chatbot": {"n_scored": 2}}}, recomputed) == []
    # a stored value contradicting the recompute is a divergence
    diverging = stored_summary_divergences(
        {"m": {"chatbot": {"n_scored": 2, "accuracy_strict": 1.0}}}, recomputed
    )
    assert len(diverging) == 1 and "accuracy_strict" in diverging[0]
    # absent/malformed stored block: nothing to cross-check, no divergence
    assert stored_summary_divergences(None, recomputed) == []
    assert stored_summary_divergences("junk", recomputed) == []


def test_summarize_emits_query_id_multisets():
    records = [
        {
            "model": "m",
            "taxonomy": "chatbot",
            "query_id": "a",
            "predicted": "kpi_query",
            "acceptable": ["kpi_query"],
            "latency_s": 0.1,
            "error": None,
        },
        {
            "model": "m",
            "taxonomy": "chatbot",
            "query_id": "b",
            "predicted": "general",
            "acceptable": ["kpi_query"],
            "latency_s": 0.1,
            "error": None,
        },
        {
            "model": "m",
            "taxonomy": "chatbot",
            "query_id": "c",
            "predicted": "greeting",
            "acceptable": None,
            "latency_s": 0.1,
            "error": None,
        },
    ]
    s = summarize_signature_runs(records)["m"]["chatbot"]
    assert s["scored_query_ids"] == ["a", "b"]
    assert s["excluded_query_ids"] == ["c"]


def test_run_e2e_replays_expected_model_mismatch_fails_fast(monkeypatch):
    from src.optimization.dspy_lane_ab import run_e2e_replays

    monkeypatch.setenv("DSPY_LM_MODEL", "openai/gpt-5.6-terra")
    golden = load_golden_set(FIXTURE_PATH)
    with pytest.raises(RuntimeError, match="expects"):
        run_e2e_replays(
            golden,
            ["ts-12"],
            "dspy-ab-test",
            expected_model="anthropic/claude-haiku-4-5-20251001",
        )


# ---------------------------------------------------------------------------
# golden set loading + bundle emission
# ---------------------------------------------------------------------------


def test_load_golden_set_real_fixture():
    golden = load_golden_set(FIXTURE_PATH)
    assert len(golden["queries"]) >= 35


def test_emit_container_script_is_self_contained():
    golden = load_golden_set(FIXTURE_PATH)
    script = emit_container_script(
        golden, models=["openai/gpt-5.6-terra", "anthropic/claude-haiku-4-5-20251001"]
    )
    # must compile standalone
    compile(script, "<bundle>", "exec")
    # carries the golden set, the models, and the runner entrypoint
    assert "ts-12" in script
    assert "openai/gpt-5.6-terra" in script
    assert "anthropic/claude-haiku-4-5-20251001" in script
    assert "def run_signature_ab" in script
    assert "RESULTS_JSON_BEGIN" in script
    # no dependency on the repo being present in the container
    assert "src.optimization.dspy_lane_ab" not in script


def test_emit_container_script_e2e_mode():
    golden = load_golden_set(FIXTURE_PATH)
    script = emit_container_script(
        golden,
        models=["anthropic/claude-haiku-4-5-20251001"],
        mode="e2e",
        e2e_query_ids=["ts-12", "ts-9", "dp-1"],
        conversation_prefix="dspy-ab-20260718",
    )
    compile(script, "<bundle>", "exec")
    assert "def run_e2e_replays" in script
    assert '"dp-1"' in script
    assert "dspy-ab-20260718" in script
    assert "RESULTS_JSON_BEGIN" in script
    # e2e mode takes its model from the per-process DSPY_LM_MODEL env, but the
    # bundle pins the intended candidate and fails fast on a mismatch
    assert "DSPY_LM_MODEL" in script
    assert "BUNDLE_EXPECTED_MODEL = 'anthropic/claude-haiku-4-5-20251001'" in script


def test_emit_container_script_e2e_requires_exactly_one_model():
    golden = load_golden_set(FIXTURE_PATH)
    for models in ([], ["a/x", "a/y"]):
        with pytest.raises(ValueError, match="exactly one model"):
            emit_container_script(golden, models=models, mode="e2e", e2e_query_ids=["ts-12"])


def test_emit_container_script_e2e_rejects_unknown_ids():
    golden = load_golden_set(FIXTURE_PATH)
    with pytest.raises(ValueError, match="unknown"):
        emit_container_script(golden, models=[], mode="e2e", e2e_query_ids=["nope-99"])


def test_emit_container_script_rejects_unknown_mode():
    golden = load_golden_set(FIXTURE_PATH)
    with pytest.raises(ValueError, match="mode"):
        emit_container_script(golden, models=[], mode="bogus")


# ---------------------------------------------------------------------------
# RAGAS sample construction from e2e replays
# ---------------------------------------------------------------------------


def test_build_ragas_samples_from_e2e_records():
    from src.optimization.dspy_lane_ab import build_ragas_samples

    e2e_results = {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "records": [
            {
                "model": "anthropic/claude-haiku-4-5-20251001",
                "query_id": "ts-9",
                "latency_s": 20.0,
                "response_text": "TRx dropped because of X.",
                "contexts": ["evidence one", "evidence two"],
                "error": None,
            },
            {
                "model": "anthropic/claude-haiku-4-5-20251001",
                "query_id": "dp-1",
                "latency_s": 5.0,
                "response_text": "",
                "contexts": [],
                "error": "TimeoutError: slow",
            },
        ],
    }
    golden = load_golden_set(FIXTURE_PATH)
    samples = build_ragas_samples(e2e_results, golden)
    # errored/empty runs are dropped: judging an error string is meaningless
    assert len(samples) == 1
    sample = samples[0]
    assert sample["query"] == "Why did Kisqali TRx drop in Q1?"
    assert sample["answer"] == "TRx dropped because of X."
    assert sample["retrieved_contexts"] == ["evidence one", "evidence two"]
    # ground_truth unused by faithfulness/answer_relevancy; kept explicit-empty
    assert sample["ground_truth"] == ""
    assert sample["metadata"]["query_id"] == "ts-9"
