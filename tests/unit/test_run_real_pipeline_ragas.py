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
    assert set(report["metrics"]) == {"faithfulness", "answer_relevancy"}
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
