"""Shard 05 integration: a real GEPA/MIPRO optimization run on real signals.

CHEAPEST FAITHFUL DISPROOF that the loop's optimize step actually runs end to
end on dspy 3.1.0 with a live LM. Skipped without ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os

import pytest

# Gated behind an explicit opt-in: a real GEPA "light" run blocks for minutes in a
# thread pool pytest-timeout cannot interrupt, hanging CI's --timeout=60 shard to
# the job limit (#504 lesson). CI sets ANTHROPIC_API_KEY, so require
# E2I_RUN_REAL_LLM_E2E=1. Run manually with a long/no per-test timeout.
pytestmark = pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_LLM_E2E") != "1" or not os.getenv("ANTHROPIC_API_KEY"),
    reason="requires E2I_RUN_REAL_LLM_E2E=1 + live Anthropic LM (slow real GEPA run)",
)


def _rows(n: int = 8):
    """n/2 defect cycles + n/2 healthy cycles.

    #1668: the seed used to be n identical POSITIVE cycles. The trainset builder
    now refuses a single-class pool (an all-positive trainset is exactly the
    sampling bias that made the loop teach over-reporting), so a fixture that
    seeds only positives would produce zero examples and this test would assert
    against a phase that never ran.
    """
    from src.agents.feedback_learner.dspy_integration import FeedbackLearnerTrainingSignal

    feedback = [{"feedback_id": "f1", "feedback_type": "rating", "user_feedback": 2}]

    def _signal(tag: str, *, patterns, recommendations) -> FeedbackLearnerTrainingSignal:
        return FeedbackLearnerTrainingSignal(
            batch_id=tag,
            feedback_count=10,
            time_range_start="t0",
            time_range_end="t1",
            patterns_detected=len(patterns),
            recommendations_generated=len(recommendations),
            updates_applied=0,
            recommendation_actionability=0.8 if recommendations else 0.0,
            update_effectiveness=0.9,
            total_latency_ms=1000.0,
            feedback_batch=list(feedback),
            patterns=list(patterns),
            recommendations=list(recommendations),
            learning_summary="accuracy issue found" if patterns else "no patterns detected",
        )

    detected = [
        {
            "pattern_type": "accuracy_issue",
            "severity": "high",
            "affected_agents": ["causal_impact"],
            "root_cause_hypothesis": "gap",
        }
    ]
    recs = [{"category": "prompt_update", "expected_impact": "higher accuracy"}]

    rows = []
    for i in range(n):
        positive = i % 2 == 0
        sig = _signal(
            f"opt{i}",
            patterns=detected if positive else [],
            recommendations=recs if positive else [],
        )
        d = sig.to_dict()
        d["reward"] = 0.9 if positive else 0.1
        rows.append(d)
    return rows


class _Client:
    def __init__(self, rows):
        self._rows = rows

    def table(self, *_):
        return self

    def select(self, *_):
        return self

    def eq(self, *_):
        return self

    def gte(self, *_):
        return self

    def limit(self, *_):
        return self

    def execute(self):
        return type("R", (), {"data": self._rows})()


@pytest.mark.asyncio
async def test_optimize_pattern_phase_produces_artifact(tmp_path, monkeypatch):
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )

    monkeypatch.chdir(tmp_path)  # artifacts land under tmp ./optimized_modules
    result = await run_feedback_learner_optimization(
        phases=("pattern",), budget="light", client=_Client(_rows(8))
    )
    assert result["signals_used"] == 8
    assert result["phases"]["pattern"]["status"] == "optimized", result["phases"]["pattern"]
    assert os.path.exists(result["phases"]["pattern"]["path"])
