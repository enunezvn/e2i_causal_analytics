"""
Unit tests for the GEPA optimization trigger threshold.

Originally A3 (commit 4ce164f4f, 2026-06-08), which lowered ``min_signals``
100 -> 20 on the argument that "~1 signal/cycle; 20 ≈ reachable in normal
operation". Both halves of that were later falsified: #1668/#1677 re-pointed the
gate at the scarcer LABEL CLASS without moving the number, so 20 quietly became
"a 40-example trainset"; and the reachability claim measures false (8 rows at
``reward >= 0.5`` over 68.8 days, 0 positives in the last 8 recorded days).

The threshold is now stated in the unit it gates — TRAINSET EXAMPLES — at 40,
which is the bar that has been in force since #1677. The assertions below moved
with the unit; the strictness did not.

Asserts:
- GEPAOptimizationTrigger() default min_trainset_examples == 40
- should_trigger fires at 40 examples, does NOT fire at 38 (no escape)
- _decide_trigger reads the DSPY_MIN_TRAINSET_EXAMPLES env override
- Reward-delta escape and forced-after-hours escape still behave correctly
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger


class TestTriggerDefaultThreshold:
    """The default threshold, and the unit it is stated in."""

    def test_default_is_40_trainset_examples(self):
        trigger = GEPAOptimizationTrigger()
        assert trigger.min_trainset_examples == 40

    def test_the_old_name_is_gone(self):
        """A constant called ``min_signals`` gating examples is the defect itself."""
        assert not hasattr(GEPAOptimizationTrigger(), "min_signals")

    def test_fires_at_exactly_the_threshold(self):
        trigger = GEPAOptimizationTrigger()
        should, _reason = trigger.should_trigger(
            trainset_examples=40,
            current_reward=0.8,
            baseline_reward=0.0,
            last_optimization=None,
        )
        assert should is True

    def test_does_not_fire_one_class_pair_below(self):
        """38 examples — the builder emits pairs, so 38 is the real next step down."""
        trigger = GEPAOptimizationTrigger()
        should, reason = trigger.should_trigger(
            trainset_examples=38,
            current_reward=0.7,
            baseline_reward=0.65,  # small delta, no reward escape
            last_optimization=None,
        )
        assert should is False
        assert "Insufficient trainset" in reason
        assert "examples" in reason


class TestTriggerEnvOverride:
    """``_decide_trigger`` reads DSPY_MIN_TRAINSET_EXAMPLES."""

    def _make_signal(self, reward: float = 0.9, *, patterns: bool = True) -> dict:
        return {
            "source_agent": "feedback_learner",
            "reward": reward,
            "input_context": {"feedback_batch": [{"x": 1}]},
            "output": {"patterns": [{"severity": "high"}] if patterns else []},
        }

    def _pool(self, k: int) -> list:
        """``k`` positives + ``k`` negatives -> a ``2k``-example trainset.

        #1668: the trigger counts what the BUILDER produces, not rows. ``k``
        identical pattern-bearing rows are a single-class pool the builder
        refuses outright, so their trainset is 0 — these tests are about the
        THRESHOLD, so the pool must actually carry the trainset they name.
        """
        return [self._make_signal(0.9)] * k + [self._make_signal(0.0, patterns=False)] * k

    def test_env_override_triggers_at_a_lower_trainset(self, monkeypatch):
        """DSPY_MIN_TRAINSET_EXAMPLES=10 -> a 10-example trainset fires."""
        monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
        monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "10")
        from src.tasks.dspy_optimization_tasks import _decide_trigger

        should, _reason = _decide_trigger(self._pool(5), state={})
        assert should is True

    def test_env_override_blocks_below_the_overridden_threshold(self, monkeypatch):
        monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
        monkeypatch.setenv("DSPY_MIN_TRAINSET_EXAMPLES", "20")
        from src.tasks.dspy_optimization_tasks import _decide_trigger

        should, reason = _decide_trigger(self._pool(5), state={})
        assert should is False
        assert reason  # has a reason string

    def test_no_env_override_uses_the_default(self, monkeypatch):
        monkeypatch.delenv("DSPY_MIN_TRAINSET_EXAMPLES", raising=False)
        monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
        from src.tasks.dspy_optimization_tasks import _decide_trigger

        # 38 examples, no escape -> blocked
        should, _reason = _decide_trigger(self._pool(19), state={})
        assert should is False

        # 40 examples with mean reward > 0 baseline -> triggers
        should, _reason = _decide_trigger(self._pool(20), state={})
        assert should is True


class TestTriggerEscapesUnchanged:
    """Reward-delta and forced-after-hours escapes still work."""

    def test_reward_delta_escape_still_works(self):
        trigger = GEPAOptimizationTrigger()
        last_opt = datetime.now(timezone.utc) - timedelta(hours=30)
        should, reason = trigger.should_trigger(
            trainset_examples=50,
            current_reward=0.50,
            baseline_reward=0.75,  # degraded by 0.25, well above 0.05 threshold
            last_optimization=last_opt,
        )
        assert should is True
        assert "Reward degraded" in reason

    def test_forced_after_hours_escape_still_works(self):
        trigger = GEPAOptimizationTrigger()
        last_opt = datetime.now(timezone.utc) - timedelta(hours=200)
        should, reason = trigger.should_trigger(
            trainset_examples=50,
            current_reward=0.70,
            baseline_reward=0.70,  # no delta
            last_optimization=last_opt,
        )
        assert should is True
        assert "Forced" in reason
