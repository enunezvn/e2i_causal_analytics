"""
Unit tests for A3 — right-sized GEPA optimization trigger threshold.

Asserts:
- GEPAOptimizationTrigger() default min_signals == 20
- should_trigger fires at signal_count=20, does NOT fire at 19 (no escape)
- _decide_trigger reads DSPY_MIN_SIGNALS env override
- Reward-delta escape and forced-after-hours escape still behave correctly
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger


class TestGEPATriggerA3DefaultThreshold:
    """A3: default min_signals is 20, not 100."""

    def test_default_min_signals_is_20(self):
        """GEPAOptimizationTrigger() must default to min_signals=20."""
        trigger = GEPAOptimizationTrigger()
        assert trigger.min_signals == 20

    def test_fires_at_exactly_20_signals(self):
        """should_trigger(signal_count=20, reward improvement) -> True."""
        trigger = GEPAOptimizationTrigger()
        should, reason = trigger.should_trigger(
            signal_count=20,
            current_reward=0.8,
            baseline_reward=0.0,
            last_optimization=None,
        )
        assert should is True

    def test_does_not_fire_at_19_signals_no_escape(self):
        """should_trigger(signal_count=19, no escape conditions) -> (False, 'Insufficient signals...')."""
        trigger = GEPAOptimizationTrigger()
        should, reason = trigger.should_trigger(
            signal_count=19,
            current_reward=0.7,
            baseline_reward=0.65,  # small delta, no reward escape
            last_optimization=None,
        )
        assert should is False
        assert "Insufficient signals" in reason


class TestGEPATriggerA3EnvOverride:
    """A3: _decide_trigger reads DSPY_MIN_SIGNALS env var."""

    def _make_signal(self, reward: float = 0.9) -> dict:
        return {
            "source_agent": "feedback_learner",
            "reward": reward,
            "input_context": {"feedback_batch": [{"x": 1}]},
            "output": {"patterns": [{"severity": "high"}]},
        }

    def test_env_override_triggers_at_lower_count(self, monkeypatch):
        """When DSPY_MIN_SIGNALS=5, 5 signals with high reward should fire."""
        monkeypatch.setenv("DSPY_MIN_SIGNALS", "5")
        from src.tasks.dspy_optimization_tasks import _decide_trigger

        signals = [self._make_signal(0.9)] * 5
        should, reason = _decide_trigger(signals, state={})
        assert should is True

    def test_env_override_blocks_below_overridden_threshold(self, monkeypatch):
        """When DSPY_MIN_SIGNALS=10, 5 signals should NOT fire."""
        monkeypatch.setenv("DSPY_MIN_SIGNALS", "10")
        from src.tasks.dspy_optimization_tasks import _decide_trigger

        signals = [self._make_signal(0.9)] * 5
        should, reason = _decide_trigger(signals, state={})
        assert should is False
        assert reason  # has a reason string

    def test_no_env_override_uses_default_20(self, monkeypatch):
        """Without DSPY_MIN_SIGNALS set, default=20 is used."""
        monkeypatch.delenv("DSPY_MIN_SIGNALS", raising=False)
        from src.tasks.dspy_optimization_tasks import _decide_trigger

        # 19 signals with high reward but no escape -> blocked
        signals = [self._make_signal(0.9)] * 19
        should, reason = _decide_trigger(signals, state={})
        assert should is False

        # 20 signals with reward > 0 baseline -> triggers
        signals = [self._make_signal(0.9)] * 20
        should, reason = _decide_trigger(signals, state={})
        assert should is True


class TestGEPATriggerA3EscapesUnchanged:
    """A3: reward-delta and forced-after-hours escapes still work."""

    def test_reward_delta_escape_still_works(self):
        """Reward degradation should still trigger even at min_signals=20."""
        trigger = GEPAOptimizationTrigger()
        last_opt = datetime.now(timezone.utc) - timedelta(hours=30)
        should, reason = trigger.should_trigger(
            signal_count=25,
            current_reward=0.50,
            baseline_reward=0.75,  # degraded by 0.25, well above 0.05 threshold
            last_optimization=last_opt,
        )
        assert should is True
        assert "Reward degraded" in reason

    def test_forced_after_hours_escape_still_works(self):
        """Forced optimization after max_hours_without_optimization still fires."""
        trigger = GEPAOptimizationTrigger()
        last_opt = datetime.now(timezone.utc) - timedelta(hours=200)
        should, reason = trigger.should_trigger(
            signal_count=25,
            current_reward=0.70,
            baseline_reward=0.70,  # no delta
            last_optimization=last_opt,
        )
        assert should is True
        assert "Forced" in reason
