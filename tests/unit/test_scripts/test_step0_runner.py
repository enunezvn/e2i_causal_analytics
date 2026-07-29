"""Unit tests for the Step 0 runner-support functions (#1337, codex iter-1).

Pins the four iteration-1 audit findings:
- checkpoint resume must survive a truncated final JSONL line (interrupted
  append) and must physically drop it so later appends don't bury it mid-file;
- corrupt NON-final checkpoint lines are real corruption and must raise;
- duplicate query_ids in the gold slice must fail fast (they would race in
  run_candidate and make the last-write-wins reload nondeterministic);
- the decision readout must enforce the protocol's "comparable latency/cost"
  clause, not accuracy alone;
- legacy llm_used detection must be exact (the confidence<0.8 heuristic
  undercounted: _llm_classify can return >=0.8, e.g. its 0.85 parse-degradation
  default, and the node never emits a `method` field).
"""

import asyncio
import json

import pytest

from scripts.benchmarks.routing.step0_candidates import (
    install_llm_call_tracker,
    llm_call_seen,
    reset_llm_call_tracking,
)
from scripts.benchmarks.routing.step0_scoring import (
    assert_unique_query_ids,
    decision_verdict,
    load_checkpoint,
)

# =============================================================================
# load_checkpoint
# =============================================================================


def test_load_checkpoint_missing_file_is_empty(tmp_path):
    assert load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_load_checkpoint_tolerates_and_removes_truncated_tail(tmp_path):
    p = tmp_path / "pred.jsonl"
    p.write_text(
        json.dumps({"query_id": "q1", "pred_pattern": "SINGLE_AGENT"})
        + "\n"
        + '{"query_id": "q2", "pred_pa'  # interrupted append, no newline
    )
    done = load_checkpoint(p)
    assert set(done) == {"q1"}
    # The bad tail must be physically truncated: a subsequent append followed
    # by a resume would otherwise see a corrupt line in the MIDDLE and crash.
    with p.open("a") as f:
        f.write(json.dumps({"query_id": "q3", "pred_pattern": "TOOL_COMPOSER"}) + "\n")
    assert set(load_checkpoint(p)) == {"q1", "q3"}


def test_load_checkpoint_raises_on_corrupt_middle_line(tmp_path):
    p = tmp_path / "pred.jsonl"
    p.write_text('{"query_id": "q1"}\nGARBAGE not json\n{"query_id": "q2"}\n')
    with pytest.raises(ValueError, match="corrupt"):
        load_checkpoint(p)


def test_load_checkpoint_duplicate_query_id_keeps_last_deterministically(tmp_path):
    p = tmp_path / "pred.jsonl"
    p.write_text('{"query_id": "q1", "v": 1}\n{"query_id": "q1", "v": 2}\n')
    assert load_checkpoint(p)["q1"]["v"] == 2


# =============================================================================
# gold-slice uniqueness
# =============================================================================


def test_assert_unique_query_ids_passes_on_unique():
    assert_unique_query_ids([{"query_id": "a"}, {"query_id": "b"}])


def test_assert_unique_query_ids_raises_on_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        assert_unique_query_ids([{"query_id": "a"}, {"query_id": "a"}])


# =============================================================================
# decision readout gate ("accuracy at comparable latency/cost")
# =============================================================================


def _summary(acc: float, p95: float, share: float) -> dict:
    return {"pattern_accuracy": acc, "latency_ms_p95": p95, "llm_share": share}


def test_verdict_replace_when_parity_and_comparable():
    v = decision_verdict(_summary(0.62, 4800.0, 0.77), _summary(0.63, 4900.0, 1.0))
    assert "replace rather than extend" in v


def test_verdict_extend_when_pipeline_more_accurate():
    v = decision_verdict(_summary(0.70, 4800.0, 0.77), _summary(0.63, 4900.0, 1.0))
    assert "earns its keep" in v


def test_verdict_abstains_when_latency_not_comparable():
    # single_llm wins on accuracy but is 5x slower: no automatic replace.
    v = decision_verdict(_summary(0.62, 1000.0, 0.77), _summary(0.63, 5000.0, 1.0))
    assert "human review" in v
    assert "replace rather than extend" not in v


def test_verdict_abstains_when_llm_share_not_comparable():
    # single_llm wins on accuracy but doubles the LLM call share (cost).
    v = decision_verdict(_summary(0.62, 4800.0, 0.5), _summary(0.63, 4900.0, 1.0))
    assert "human review" in v


# =============================================================================
# legacy llm_used tracking (exact, per-asyncio-task)
# =============================================================================


class _StubClassifier:
    """High-confidence LLM result: invisible to the old <0.8 heuristic."""

    def __init__(self):
        self.calls = 0

    async def _llm_classify(self, *args, **kwargs):
        self.calls += 1
        return {"primary_intent": "general", "confidence": 0.85}


def test_tracker_detects_high_confidence_llm_call():
    stub = _StubClassifier()
    install_llm_call_tracker(stub)

    async def main():
        reset_llm_call_tracking()
        assert llm_call_seen() is False
        result = await stub._llm_classify("query")
        assert result["confidence"] == 0.85  # wrapper is transparent
        assert llm_call_seen() is True

    asyncio.run(main())
    assert stub.calls == 1


def test_tracker_target_exists_on_real_node():
    """Integration sentinel (codex iter-2 LOW): install_llm_call_tracker
    patches the node's private ``_llm_classify``. If the node renames or
    de-asyncs it, fail HERE loudly instead of silently measuring llm_share=0.
    """
    import inspect

    from src.agents.orchestrator.nodes.intent_classifier import IntentClassifierNode

    assert inspect.iscoroutinefunction(IntentClassifierNode._llm_classify)


def test_tracker_is_isolated_across_concurrent_tasks():
    stub = _StubClassifier()
    install_llm_call_tracker(stub)
    seen: dict = {}

    async def with_llm():
        reset_llm_call_tracking()
        await stub._llm_classify("q")
        await asyncio.sleep(0)  # interleave with the other task
        seen["with"] = llm_call_seen()

    async def without_llm():
        reset_llm_call_tracking()
        await asyncio.sleep(0)
        seen["without"] = llm_call_seen()

    async def main():
        await asyncio.gather(with_llm(), without_llm())

    asyncio.run(main())
    assert seen == {"with": True, "without": False}
