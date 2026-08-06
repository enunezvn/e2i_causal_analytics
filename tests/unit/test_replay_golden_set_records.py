"""Contract tests for replay record capture (#1485).

``scripts/replay_golden_set.py`` was written to generate learning signals, so
it checked each response for truthiness and threw the body away —
``send_cognitive`` returned only ``(ok, "agents=... hops=... latency_ms=...")``.
The real-pipeline RAGAS gate needs what that body carried: the genuinely
generated answer and the genuinely retrieved contexts. These tests pin the
capture, including the case that matters most — a turn that retrieved nothing
must record an EMPTY context list, never a reference stand-in.

SIBLING SUITE: ``tests/unit/test_scripts/test_replay_golden_set.py`` owns the
script's pre-existing contracts (dry-run discipline, 401 re-minting, the
never-raises guarantees) and pins the ``(ok, detail, body)`` sender contract
this file depends on. Run BOTH when touching scripts/replay_golden_set.py —
changing the senders' return arity breaks that file, not this one, and it was
missed for four commits because it sits under a different test directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import replay_golden_set as rgs  # noqa: E402


def _body(**overrides: Any) -> Dict[str, Any]:
    """A ``CognitiveRAGResponse`` body as the endpoint returns it."""
    body = {
        "response": "Kisqali TRx rose 12% in the Northeast on oncologist engagement.",
        "evidence": [
            {"content": "Northeast TRx up 12% QoQ.", "source": "agent_activities"},
            {"content": "Speaker programs up 30%.", "source": "kg"},
        ],
        "hop_count": 2,
        "routed_agents": ["causal_analysis"],
        "intent": "causal",
        "latency_ms": 17400.0,
        "error": None,
    }
    body.update(overrides)
    return body


def test_module_under_test_is_the_worktree_copy():
    """A stale installed copy would silently invalidate every assertion here."""
    assert str(REPO_ROOT) in rgs.__file__, f"imported {rgs.__file__}, expected under {REPO_ROOT}"


def test_cognitive_record_captures_answer_and_contexts():
    record = rgs.build_cognitive_record(
        query_id="q01",
        query="Why did Kisqali TRx move in the Northeast?",
        conversation_id="goldset-replay-20260805-q01",
        body=_body(),
        latency_s=17.4,
    )
    assert record["query_id"] == "q01"
    assert record["query"] == "Why did Kisqali TRx move in the Northeast?"
    assert record["response_text"].startswith("Kisqali TRx rose 12%")
    assert record["contexts"] == ["Northeast TRx up 12% QoQ.", "Speaker programs up 30%."]
    assert record["evidence_count"] == 2
    assert record["hop_count"] == 2
    assert record["answer_chars"] == len(record["response_text"])
    assert record["error"] is None
    assert record["latency_s"] == pytest.approx(17.4)


def test_cognitive_record_zero_evidence_records_empty_contexts():
    """A zero-retrieval turn is the signal, not a gap to paper over.

    3 of 10 replays retrieved any context on 2026-07-18. If this ever recorded
    a stand-in, RAGASEvaluator.evaluate_sample would score the turn against it
    (src/rag/evaluation.py:982) and the gate would go green on no evidence.
    """
    record = rgs.build_cognitive_record(
        query_id="q02",
        query="anything",
        conversation_id="c",
        body=_body(evidence=[]),
        latency_s=1.0,
    )
    assert record["contexts"] == []
    assert record["evidence_count"] == 0
    assert record["error"] is None, "no evidence is not an error — it is the measurement"


def test_cognitive_record_marks_in_band_error():
    """CognitiveRAGResponse reports workflow failure in-band via ``error``."""
    record = rgs.build_cognitive_record(
        query_id="q03",
        query="anything",
        conversation_id="c",
        body=_body(error="workflow exploded", response=""),
        latency_s=2.0,
    )
    assert record["error"] and "workflow exploded" in record["error"]
    assert record["response_text"] == ""


def test_transport_failure_record_is_recorded_not_dropped():
    """An HTTP/transport failure still produces a record so n stays honest."""
    record = rgs.build_failed_record(
        query_id="q04",
        query="anything",
        conversation_id="c",
        error="HTTP 502: bad gateway",
        latency_s=3.0,
    )
    assert record["error"] == "HTTP 502: bad gateway"
    assert record["response_text"] == ""
    assert record["contexts"] == []


def test_records_feed_the_judge_adapter_unchanged():
    """The record-shape contract: replay output must construct judge samples.

    This is the wiring #1489 called 'mostly plumbing' — pinned so a change to
    either side breaks here rather than in a paid live run.
    """
    from src.rag.evaluation import EvaluationSample
    from src.rag.real_pipeline_eval import build_samples_from_replay

    records = [
        rgs.build_cognitive_record(
            query_id="q01", query="q one", conversation_id="c1", body=_body(), latency_s=1.0
        ),
        rgs.build_cognitive_record(
            query_id="q02",
            query="q two",
            conversation_id="c2",
            body=_body(error="boom", response=""),
            latency_s=1.0,
        ),
    ]
    samples = build_samples_from_replay(records)

    assert [s["metadata"]["query_id"] for s in samples] == ["q01"], "errored replay must be dropped"
    sample = EvaluationSample(**samples[0])
    assert sample.answer.startswith("Kisqali TRx rose 12%")
    assert sample.retrieved_contexts == ["Northeast TRx up 12% QoQ.", "Speaker programs up 30%."]
    assert sample.contexts == [], "reference contexts must never reach a real-pipeline sample"


def test_record_out_requires_the_cognitive_target():
    """--target chat returns no evidence, so it cannot produce judgeable records."""
    with pytest.raises(SystemExit):
        rgs.main(["--target", "chat", "--record-out", "/tmp/nope.json", "--dry-run"])


def test_dry_run_with_record_out_writes_nothing(tmp_path):
    out = tmp_path / "records.json"
    rc = rgs.main(["--target", "cognitive", "--record-out", str(out), "--dry-run", "--limit", "1"])
    assert rc == 0
    assert not out.exists(), "a dry run must not write a records file"
