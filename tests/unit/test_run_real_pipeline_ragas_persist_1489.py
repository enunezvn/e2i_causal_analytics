"""The real-pipeline driver persists what it judged (#1489 deferral 1).

Before this, the driver printed a report and exited: the judged numbers went
to stdout and a JSON file, and the schema #1487 built to hold them stayed
empty (``evaluation_results`` 0 rows, measured 2026-08-06). The judge run is
the expensive part and it had already happened.

Persistence is ON by default with ``--no-persist`` to skip. A default-off flag
would reproduce the exact failure #1487 filed — a writer that exists and is
never called — one level up, in a script nobody runs twice.

Ordering is load-bearing: the report is written and printed BEFORE any DB
write, so a persistence failure can never destroy the record of a judge run
that costs several minutes of gpt-4o time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_real_pipeline_ragas as driver  # noqa: E402

_RECORDS = [
    {
        "query_id": f"q{i:02d}",
        "query": f"question {i}",
        "response_text": f"answer {i}",
        "contexts": ["ctx"] if i < 4 else [],
        "conversation_id": f"goldset-replay-20260806-q{i:02d}",
        "error": None,
    }
    for i in range(1, 11)
]

_BLOCK = {
    "model": "real-pipeline:cognitive",
    "n_samples": 10,
    "n_faithfulness": 3,
    "faithfulness": 0.61,
    "answer_relevancy": 0.42,
    "per_sample": [
        {
            "query_id": f"q{i:02d}",
            "n_contexts": 1 if i < 4 else 0,
            "faithfulness": 0.61 if i < 4 else None,
            "answer_relevancy": 0.42,
            "evaluation_method": None,
        }
        for i in range(1, 11)
    ],
}


def test_module_under_test_is_the_worktree_copy():
    assert str(REPO_ROOT) in driver.__file__, f"imported {driver.__file__}"


async def test_build_writers_resolves_its_real_imports(monkeypatch):
    """Execute the writer factory for real, with only the client stubbed.

    Every other test patches this factory out, so a wrong module path or a
    missing ``await`` inside it would never run and the driver would fail only
    on the live box. RED when first written: the factory reached for
    ``src.database.supabase_client.get_async_supabase_client``, which does not
    exist — the real symbol is in ``src.memory.services.factories`` and is a
    COROUTINE, so even the right path needed awaiting."""
    from src.memory.services import factories

    async def _fake_client():
        return MagicMock()

    monkeypatch.setattr(factories, "get_async_supabase_client", _fake_client)

    eval_repo, rubric_node = await driver._build_writers(persist_signals=True)

    assert eval_repo.table_name == "evaluation_results"
    assert rubric_node is not None
    assert rubric_node.db_client is not None

    eval_repo_only, no_node = await driver._build_writers(persist_signals=False)
    assert eval_repo_only.table_name == "evaluation_results"
    assert no_node is None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Drive main() with the judge and the DB stubbed, everything else real."""
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps({"target": "cognitive", "records": _RECORDS}))

    calls: Dict[str, Any] = {"persisted": [], "order": []}

    monkeypatch.setattr(driver, "_openai_key_present", lambda mode, container: True)

    def _fake_judge(samples, mode, container, model_label, timeout):
        calls["order"].append("judge")
        return _BLOCK

    monkeypatch.setattr(driver, "run_judge", _fake_judge)

    repo = MagicMock()

    async def _record_evaluation(**kwargs):
        calls["order"].append("persist")
        calls["persisted"].append(kwargs)
        return {"evaluation_id": "eval-1"}

    repo.record_evaluation = AsyncMock(side_effect=_record_evaluation)

    # calls["node"] lets a test inject a rubric node; the driver decides
    # whether to ASK for one, which is what --persist-signals controls.
    async def _fake_writers(persist_signals: bool):
        calls["persist_signals"] = persist_signals
        return repo, (calls.get("node") if persist_signals else None)

    monkeypatch.setattr(driver, "_build_writers", _fake_writers)

    return records_path, calls, repo


def test_persists_every_scored_sample_by_default(wired, tmp_path):
    records_path, calls, repo = wired
    out = tmp_path / "report.json"

    rc = driver.main(
        ["--records", str(records_path), "--judge-mode", "local", "--output", str(out)]
    )

    assert rc == 0
    # 10 judged, all 10 scored answer_relevancy, so all 10 have a measured half.
    assert repo.record_evaluation.await_count == 10
    first = calls["persisted"][0]
    assert first["query"] == "question 1"
    assert first["response"] == "answer 1"
    assert first["retrieved_contexts"] == ["ctx"]


def test_the_report_is_written_before_anything_is_persisted(wired, tmp_path):
    """A DB failure must never cost the judge run. The file on disk is the
    durable record; the DB write is the derived one."""
    records_path, calls, _ = wired
    out = tmp_path / "report.json"

    driver.main(["--records", str(records_path), "--judge-mode", "local", "--output", str(out)])

    assert out.exists()
    assert calls["order"][0] == "judge"
    assert calls["order"][1] == "persist"
    report = json.loads(out.read_text())
    assert report["n_samples"] == 10


def test_no_persist_skips_the_db_entirely(wired, tmp_path):
    records_path, _, repo = wired

    rc = driver.main(["--records", str(records_path), "--judge-mode", "local", "--no-persist"])

    assert rc == 0
    repo.record_evaluation.assert_not_awaited()


def test_persistence_failure_is_reported_and_exits_nonzero(wired, tmp_path):
    """Fail-loud: a silently swallowed insert would let a run report success
    while persisting nothing — the failure #1487 was filed about."""
    records_path, _, repo = wired
    repo.record_evaluation.side_effect = RuntimeError("db down")
    out = tmp_path / "report.json"

    rc = driver.main(
        ["--records", str(records_path), "--judge-mode", "local", "--output", str(out)]
    )

    assert rc == 1
    # The judge run survives the DB failure.
    assert out.exists()


def test_persistence_failure_does_not_mask_a_passing_gate(wired, tmp_path, capsys):
    records_path, _, repo = wired
    repo.record_evaluation.side_effect = RuntimeError("db down")

    driver.main(["--records", str(records_path), "--judge-mode", "local"])

    printed = capsys.readouterr().out
    assert "GATES PASSED" in printed
    assert "PERSISTENCE FAILED" in printed


def test_the_report_records_what_was_persisted(wired, tmp_path):
    """A report claiming a passing gate over rows that never landed is the
    reporting half of the same problem."""
    records_path, _, _ = wired
    out = tmp_path / "report.json"

    driver.main(["--records", str(records_path), "--judge-mode", "local", "--output", str(out)])

    persistence = json.loads(out.read_text())["persistence"]
    assert persistence["evaluation_results_written"] == 10
    assert persistence["skipped_unscored"] == []


def test_a_threshold_failure_is_still_persisted(wired, tmp_path, capsys):
    """A regression must reach the trend view, not vanish from it.

    Persistence was originally gated on the run's verdict, on the reasoning
    that a blocked run's numbers should not enter v_ragas_performance_trends
    "as if they were healthy measurements". That was wrong, and codex iter-2
    named why: a view that only ever contains PASSING runs is
    survivorship-biased and structurally incapable of showing a decline —
    which is the one thing "daily RAGAS metric trends for monitoring" exists
    to show. A low faithfulness IS the measurement.

    The verdict still governs the exit code and the report; it no longer
    filters the database. Rows whose scores are not trustworthy are refused
    row-wise instead, by judged_turns (heuristic contamination, unjoinable
    provenance, missing metric keys) and by the no-measured-half skip.
    """
    records_path, _, repo = wired
    out = tmp_path / "report.json"

    rc = driver.main(
        [
            "--records",
            str(records_path),
            "--judge-mode",
            "local",
            "--faithfulness",
            "0.99",
            "--output",
            str(out),
        ]
    )

    printed = capsys.readouterr().out
    assert "GATES BLOCKED" in printed
    assert repo.record_evaluation.await_count == 10
    report = json.loads(out.read_text())
    assert report["passed"] is False
    assert report["persistence"]["evaluation_results_written"] == 10
    assert rc == 0  # no --fail-on-threshold


def test_a_blocked_run_still_exits_1_with_fail_on_threshold(wired, tmp_path):
    """Persisting a regression must not soften the gate that reports it."""
    records_path, _, repo = wired

    rc = driver.main(
        [
            "--records",
            str(records_path),
            "--judge-mode",
            "local",
            "--faithfulness",
            "0.99",
            "--fail-on-threshold",
        ]
    )

    assert rc == 1
    assert repo.record_evaluation.await_count == 10


def test_an_inconsistent_block_is_refused_at_persist_time(wired, tmp_path, monkeypatch):
    """A block whose aggregates no longer describe its own rows is stale,
    hand-edited or partially merged — structurally untrustworthy rather than
    merely low-scoring. Persisting a bad measurement is right; persisting one
    whose provenance cannot be reconciled is not, and evaluation_results has
    no run-status column that could mark it afterwards. (codex iter-3.)"""
    records_path, _, repo = wired
    block = json.loads(json.dumps(_BLOCK))
    # Aggregates kept, rows halved: the judge was cut off mid-run.
    block["per_sample"] = block["per_sample"][:5]
    monkeypatch.setattr(
        driver, "run_judge", lambda samples, mode, container, model_label, timeout: block
    )

    rc = driver.main(["--records", str(records_path), "--judge-mode", "local"])

    assert rc == 1
    repo.record_evaluation.assert_not_awaited()


def test_heuristic_contamination_is_still_refused_at_persist_time(wired, tmp_path, monkeypatch):
    """Persisting regardless of the verdict must not let word-overlap scores
    into the table — evaluation_results has no column that could mark them."""
    records_path, _, repo = wired
    block = json.loads(json.dumps(_BLOCK))
    block["per_sample"][0]["evaluation_method"] = "fallback_heuristic"
    monkeypatch.setattr(
        driver, "run_judge", lambda samples, mode, container, model_label, timeout: block
    )

    rc = driver.main(["--records", str(records_path), "--judge-mode", "local"])

    assert rc == 1
    repo.record_evaluation.assert_not_awaited()


def test_a_scoreless_row_blocks_the_gate_but_the_scored_rows_persist(wired, tmp_path, monkeypatch):
    """#1488's fail-closed guard stays intact, and the valid rows still land.

    A row the judge covered but never scored BLOCKS the run
    (``_ragas_scoreless_error``) — unchanged, that guard is deliberate shipped
    behaviour. What changed is that blocking the VERDICT no longer blocks the
    DATABASE: the nine rows the judge did score are real measurements and are
    recorded, while the scoreless one is skipped and NAMED in the summary
    rather than quietly contributing nothing.

    Persisting it would be the actual harm — ``record_evaluation`` refuses a
    row with no measured half because ``v_ragas_performance_trends
    .evaluation_count`` is COUNT(*), so such a row inflates the denominator a
    reader compares the averages against while contributing to none of them.
    """
    records_path, _, repo = wired
    block = json.loads(json.dumps(_BLOCK))
    block["per_sample"][9]["answer_relevancy"] = None
    monkeypatch.setattr(
        driver, "run_judge", lambda samples, mode, container, model_label, timeout: block
    )
    out = tmp_path / "report.json"

    driver.main(["--records", str(records_path), "--judge-mode", "local", "--output", str(out)])

    report = json.loads(out.read_text())
    assert report["passed"] is False
    assert any("scoreless" in f for f in report["failures"])

    assert repo.record_evaluation.await_count == 9
    assert report["persistence"]["evaluation_results_written"] == 9
    assert report["persistence"]["skipped_unscored"] == ["q10"]


def test_persist_does_not_run_the_rubric_judge_by_default(wired, tmp_path):
    """The rubric half is a second LLM (Anthropic) per sample. Default-on for
    the free half, opt-in for the paid one."""
    records_path, calls, _ = wired
    node = MagicMock()
    node.evaluate_and_store = AsyncMock(return_value="sig-7")
    calls["node"] = node

    driver.main(["--records", str(records_path), "--judge-mode", "local"])

    assert calls["persist_signals"] is False
    node.evaluate_and_store.assert_not_awaited()
    assert all(c["learning_signal_id"] is None for c in calls["persisted"])


def test_persist_signals_flag_wires_the_rubric_node(wired, tmp_path):
    records_path, calls, _ = wired
    node = MagicMock()
    node.evaluate_and_store = AsyncMock(return_value="sig-7")
    calls["node"] = node

    driver.main(["--records", str(records_path), "--judge-mode", "local", "--persist-signals"])

    assert calls["persist_signals"] is True
    assert node.evaluate_and_store.await_count == 10
    assert all(c["learning_signal_id"] == "sig-7" for c in calls["persisted"])
