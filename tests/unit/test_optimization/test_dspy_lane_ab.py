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


BASELINE = {
    "signature": {
        "cognitive_rag": {"accuracy_strict": 0.80, "parse_failure_rate": 0.0, "error_classes": {}},
        "chatbot": {"accuracy_strict": 0.90, "parse_failure_rate": 0.0, "error_classes": {}},
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
            },
            "chatbot": {
                "accuracy_strict": chat_acc,
                "parse_failure_rate": parse_fail,
                "error_classes": sig_errors or {},
            },
        },
        "ragas": _ragas_block(faith=faith, rel=rel, n_ctx=n_ctx),
        "e2e": {"latency_p50": p50, "error_classes": e2e_errors or {}, "n": 10},
    }


def test_gates_all_pass_at_parity():
    result = evaluate_gates(BASELINE, _candidate())
    assert result["all_passed"] is True
    assert all(g["passed"] for g in result["gates"])


def test_gate_accuracy_margin():
    # 5pp below baseline is allowed, more is not
    ok = evaluate_gates(BASELINE, _candidate(cog_acc=0.80 - GATE_ACCURACY_MARGIN))
    assert ok["all_passed"] is True
    bad = evaluate_gates(BASELINE, _candidate(cog_acc=0.80 - GATE_ACCURACY_MARGIN - 0.01))
    assert bad["all_passed"] is False
    failed = [g for g in bad["gates"] if not g["passed"]]
    assert any("accuracy" in g["name"] for g in failed)


def test_gate_parse_failure_must_not_exceed_baseline():
    bad = evaluate_gates(BASELINE, _candidate(parse_fail=0.05))
    assert bad["all_passed"] is False


def test_gate_ragas_margin():
    ok = evaluate_gates(BASELINE, _candidate(faith=0.85 - GATE_RAGAS_MARGIN))
    assert ok["all_passed"] is True
    bad = evaluate_gates(BASELINE, _candidate(rel=0.80 - GATE_RAGAS_MARGIN - 0.01))
    assert bad["all_passed"] is False


def test_gate_no_new_error_class():
    # error classes already seen in baseline do not fail the candidate
    baseline = json.loads(json.dumps(BASELINE))
    baseline["e2e"]["error_classes"] = {"TimeoutError": 1}
    ok = evaluate_gates(baseline, _candidate(e2e_errors={"TimeoutError": 2}))
    assert ok["all_passed"] is True
    bad = evaluate_gates(baseline, _candidate(sig_errors={"AuthenticationError": 1}))
    assert bad["all_passed"] is False


def test_gate_e2e_latency_factor():
    ok = evaluate_gates(BASELINE, _candidate(p50=40.0 * GATE_E2E_LATENCY_FACTOR))
    assert ok["all_passed"] is True
    bad = evaluate_gates(BASELINE, _candidate(p50=40.0 * GATE_E2E_LATENCY_FACTOR + 0.1))
    assert bad["all_passed"] is False


def test_gate_missing_ragas_fails_closed():
    candidate = _candidate()
    candidate["ragas"] = None
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False


def test_gate_ragas_per_metric_none_fails_closed():
    # The judge emits {"faithfulness": None, ...} when zero judged samples had
    # retrieved contexts - a truthy dict that must fail the gate, not TypeError.
    candidate = _candidate()
    candidate["ragas"]["faithfulness"] = None
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" in failed


def test_gate_ragas_per_metric_none_baseline_fails_closed():
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"]["answer_relevancy"] = None
    result = evaluate_gates(baseline, _candidate())
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[answer_relevancy]" in failed


def test_gate_ragas_completeness_requires_all_replays_judged():
    # Errored/empty-answer replays are excluded from judging by
    # build_ragas_samples; the gate must surface that as a failure instead of
    # letting RAGAS score only the candidate's easier successful subset.
    candidate = _candidate()
    candidate["ragas"]["n_samples"] = 8
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[completeness]" in failed


def test_gate_ragas_completeness_missing_counts_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["n_samples"]
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[completeness]" in failed


def test_gate_ragas_faithfulness_coverage_candidate_below_baseline():
    # Faithfulness averages only context-bearing replays; a candidate judged on
    # fewer of them than the baseline is being scored on a different (easier)
    # denominator even when n_samples matches (codex iter-2).
    candidate = _candidate(n_ctx=8)
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" in failed


def test_gate_ragas_faithfulness_coverage_min_n_floor():
    # A faithfulness mean over 1-2 replays is noise, not signal - both sides
    # need at least GATE_RAGAS_MIN_FAITHFULNESS_N context-bearing replays.
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"] = _ragas_block(n_ctx=2)
    result = evaluate_gates(baseline, _candidate(n_ctx=2))
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" in failed


def test_gate_ragas_faithfulness_coverage_missing_count_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["n_faithfulness"]
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness_coverage]" in failed


def test_gate_ragas_codex_iter2_scenario_tiny_context_subset():
    # Codex iter-2 HIGH: all 10 replays answered (completeness PASSES) but the
    # candidate retrieved contexts on only 1, so its faithfulness mean is a
    # single easy sample that clears the margin. Must still fail.
    candidate = _candidate(faith=0.90, n_ctx=1)
    result = evaluate_gates(BASELINE, candidate)
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
    result = evaluate_gates(baseline, candidate)
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
    result = evaluate_gates(BASELINE, candidate)
    assert result["all_passed"] is False
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" not in failed  # aggregate said 0.85
    assert "ragas[faithfulness_common_subset]" in failed


def test_gate_ragas_common_subset_missing_per_sample_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["per_sample"]
    result = evaluate_gates(BASELINE, candidate)
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
    result = evaluate_gates(BASELINE, candidate)
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
    result = evaluate_gates(BASELINE, candidate)
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
    result = evaluate_gates(BASELINE, candidate)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[faithfulness]" not in failed
    assert "ragas[faithfulness_common_subset]" not in failed
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_baseline_side_checked():
    # A corrupted-low baseline aggregate weakens every floor; reconciliation
    # must validate the baseline block too, not just the candidate.
    baseline = json.loads(json.dumps(BASELINE))
    baseline["ragas"]["faithfulness"] = 0.30
    result = evaluate_gates(baseline, _candidate())
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_relevancy_mismatch():
    candidate = _candidate()
    for row in candidate["ragas"]["per_sample"]:
        row["answer_relevancy"] = 0.70
    result = evaluate_gates(BASELINE, candidate)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[answer_relevancy]" not in failed  # aggregate still says 0.80
    assert "ragas[consistency]" in failed


def test_gate_ragas_consistency_missing_per_sample_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["per_sample"]
    result = evaluate_gates(BASELINE, candidate)
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
    result = evaluate_gates(BASELINE, candidate)
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
    result = evaluate_gates(baseline, _candidate())
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[fully_scored]" in failed


def test_gate_ragas_fully_scored_relevancy_scoreless():
    # answer_relevancy averages over all rows with the same None-skip, so a
    # scoreless row hides there identically.
    candidate = _candidate()
    candidate["ragas"]["per_sample"][0]["answer_relevancy"] = None
    result = evaluate_gates(BASELINE, candidate)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[answer_relevancy]" not in failed  # mean over the other 9 = 0.80
    assert "ragas[consistency]" not in failed
    assert "ragas[fully_scored]" in failed


def test_gate_ragas_fully_scored_ignores_uncovered_rows():
    # Rows the judge never retrieved contexts for (n_contexts=0, score None)
    # are legitimately scoreless - only covered-but-unscored rows fail.
    result = evaluate_gates(BASELINE, _candidate(n_ctx=7))
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[fully_scored]" not in failed


def test_gate_ragas_fully_scored_missing_per_sample_fails_closed():
    candidate = _candidate()
    del candidate["ragas"]["per_sample"]
    result = evaluate_gates(BASELINE, candidate)
    failed = {g["name"] for g in result["gates"] if not g["passed"]}
    assert "ragas[fully_scored]" in failed


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
