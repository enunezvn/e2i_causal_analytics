"""Contract tests for the real-pipeline RAGAS driver (#1485).

The driver wires ``replay_golden_set.py --record-out`` to the existing judge
(``scripts/run_dspy_lane_ragas_judge.py``, invoked unchanged) and gates the
verdict. Everything here is the fail-closed half: a judge that crashed, a
truncated block, or an environment that would quietly produce heuristic
scores instead of gpt-4o judgments must BLOCK, never report a number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_real_pipeline_ragas as driver  # noqa: E402

_BLOCK = {
    "model": "real-pipeline:cognitive",
    "n_samples": 10,
    "n_faithfulness": 4,
    "faithfulness": 0.61,
    "answer_relevancy": 0.42,
    "per_sample": [{"query_id": "q01", "n_contexts": 2, "faithfulness": 0.61}],
}


def test_module_under_test_is_the_worktree_copy():
    assert str(REPO_ROOT) in driver.__file__, f"imported {driver.__file__}"


# ---------------------------------------------------------------------------
# parse_judge_output — fail-closed
# ---------------------------------------------------------------------------


def test_parse_judge_output_extracts_the_marked_block():
    stdout = f"noise\nRESULTS_JSON_BEGIN\n{json.dumps(_BLOCK)}\nRESULTS_JSON_END\ntrailing"
    assert driver.parse_judge_output(stdout) == _BLOCK


def test_parse_judge_output_fails_closed_without_markers():
    """A judge that died before emitting results must not read as an empty pass."""
    with pytest.raises(driver.JudgeOutputError):
        driver.parse_judge_output("Traceback (most recent call last): RuntimeError: boom")


def test_parse_judge_output_fails_closed_on_truncated_block():
    with pytest.raises(driver.JudgeOutputError):
        driver.parse_judge_output('RESULTS_JSON_BEGIN\n{"n_samples": 10')


def test_parse_judge_output_fails_closed_on_malformed_json():
    with pytest.raises(driver.JudgeOutputError):
        driver.parse_judge_output("RESULTS_JSON_BEGIN\nnot json\nRESULTS_JSON_END")


# ---------------------------------------------------------------------------
# Judge environment guard
# ---------------------------------------------------------------------------


def test_judge_env_blocks_when_openai_key_absent():
    """Without a key RAGASEvaluator silently emits word-overlap heuristics.

    ``_evaluate_with_fallback`` (src/rag/evaluation.py:1191) returns
    plausible-looking floats that are NOT gpt-4o judgments, and the judge's
    output shape cannot distinguish them. So the absence must block up front.
    """
    failure = driver.judge_env_failure(key_present=False)
    assert failure and "OPENAI_API_KEY" in failure


def test_judge_env_passes_when_key_present():
    assert driver.judge_env_failure(key_present=True) is None


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------


def test_load_records_accepts_the_replay_wrapper(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"target": "cognitive", "records": [{"query_id": "q01"}]}))
    records, meta = driver.load_records(path)
    assert records == [{"query_id": "q01"}]
    assert meta["target"] == "cognitive"


def test_load_records_accepts_a_bare_list(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps([{"query_id": "q01"}]))
    records, meta = driver.load_records(path)
    assert records == [{"query_id": "q01"}]
    assert meta == {}


def test_load_records_rejects_an_empty_file(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"records": []}))
    with pytest.raises(ValueError, match="no records"):
        driver.load_records(path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_carries_measurement_thresholds_and_verdict():
    report = driver.build_report(
        block=_BLOCK,
        retrieval={
            "n_records": 12,
            "n_errors": 0,
            "n_with_contexts": 4,
            "retrieval_hit_rate": 1 / 3,
        },
        thresholds={"faithfulness": 0.50, "answer_relevancy": 0.30},
        passed=True,
        failures=[],
        meta={"target": "cognitive", "generated_at": "2026-08-05T00:00:00+00:00"},
    )
    assert report["passed"] is True
    assert report["metrics"]["faithfulness"] == 0.61
    assert report["metrics"]["answer_relevancy"] == 0.42
    assert report["n_samples"] == 10
    assert report["n_faithfulness"] == 4
    assert report["retrieval"]["retrieval_hit_rate"] == pytest.approx(1 / 3)
    assert report["thresholds"]["faithfulness"] == 0.50
    assert report["judge_model"] == "gpt-4o"
    assert report["failures"] == []


def test_report_never_claims_context_metrics():
    """context_precision/recall need a ground truth the replay does not fabricate."""
    report = driver.build_report(
        block=dict(_BLOCK, context_precision=1.0, context_recall=1.0),
        retrieval={"n_records": 10, "n_errors": 0, "n_with_contexts": 4, "retrieval_hit_rate": 0.4},
        thresholds={"faithfulness": 0.50, "answer_relevancy": 0.30},
        passed=True,
        failures=[],
        meta={},
    )
    # #1489 added answer_relevancy_hit_conditioned — the other factor of the
    # aggregate, derived from the block's own rows. Asserted explicitly rather
    # than by a key count, and paired with a rule that states the actual
    # intent: no context metric may ever appear, whatever else does.
    assert set(report["metrics"]) == {
        "faithfulness",
        "answer_relevancy",
        "answer_relevancy_hit_conditioned",
    }
    assert not [m for m in report["metrics"] if m.startswith("context_")]
    assert "context_precision" not in report["thresholds"]


def test_report_records_failures_when_gates_block():
    report = driver.build_report(
        block=_BLOCK,
        retrieval={"n_records": 10, "n_errors": 0, "n_with_contexts": 4, "retrieval_hit_rate": 0.4},
        thresholds={"faithfulness": 0.90, "answer_relevancy": 0.30},
        passed=False,
        failures=["faithfulness=0.610 < threshold 0.9"],
        meta={},
    )
    assert report["passed"] is False
    assert report["failures"] == ["faithfulness=0.610 < threshold 0.9"]


# ---------------------------------------------------------------------------
# Judge-script output contract (codex iter-1 HIGH)
# ---------------------------------------------------------------------------


def test_judge_script_emits_the_evaluation_method_stamp():
    """The gate can only refuse heuristic rows if the judge carries the stamp.

    ``scripts/run_dspy_lane_ragas_judge.py`` is SHARED with the DSPy A/B lane,
    so this is an add-only key. Without it, a mid-run degradation to
    ``fallback_heuristic`` (src/rag/evaluation.py:1188) is invisible in the
    block and the gate would pass on numbers no judge produced.
    """
    source = driver.JUDGE_SCRIPT.read_text()
    assert '"evaluation_method"' in source
    assert "res.metadata" in source


def test_driver_gates_retrieval_as_well_as_metrics():
    """The driver must use the composed verdict, not block gates alone."""
    source = driver.__file__ and Path(driver.__file__).read_text()
    assert "check_run_gates" in source, "driver must gate retrieval alongside the judge block"
    assert "check_real_pipeline_gates(" not in source, (
        "driver calls the block-only gate; a retrieval collapse would pass green"
    )


# ---------------------------------------------------------------------------
# Hit-conditioned relevancy surfaced by the driver (#1489 deferral 7)
# ---------------------------------------------------------------------------


def test_report_carries_the_hit_conditioned_relevancy_and_its_denominator():
    """The aggregate is the product of the hit rate and this number; a report
    that shows only the product cannot tell a reader which factor moved."""
    block = dict(
        _BLOCK,
        per_sample=[
            {"query_id": "q01", "n_contexts": 2, "faithfulness": 0.6, "answer_relevancy": 0.9},
            {"query_id": "q02", "n_contexts": 2, "faithfulness": 0.6, "answer_relevancy": 0.5},
            {"query_id": "q03", "n_contexts": 0, "faithfulness": None, "answer_relevancy": 0.0},
        ],
    )
    report = driver.build_report(
        block=block,
        retrieval={
            "n_records": 3,
            "n_errors": 0,
            "n_with_contexts": 2,
            "retrieval_hit_rate": 2 / 3,
        },
        thresholds={"faithfulness": 0.50, "answer_relevancy": 0.04},
        passed=True,
        failures=[],
        meta={},
    )
    assert report["metrics"]["answer_relevancy_hit_conditioned"] == pytest.approx(0.70)
    assert report["n_hit_conditioned"] == 2


def test_hit_conditioned_floor_is_a_cli_flag_defaulting_to_the_measured_constant():
    from src.rag.real_pipeline_eval import MIN_HIT_CONDITIONED_RELEVANCY

    args = driver.parse_args(["--records", "r.json"])
    assert args.hit_conditioned_relevancy == MIN_HIT_CONDITIONED_RELEVANCY

    args = driver.parse_args(["--records", "r.json", "--hit-conditioned-relevancy", "0.4"])
    assert args.hit_conditioned_relevancy == 0.4
