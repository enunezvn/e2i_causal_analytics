"""
Unit tests for GEPA Optimization Trigger.
Version: 4.5  (#1668 follow-up: the threshold is stated in TRAINSET EXAMPLES.
It was ``min_signals`` gating a per-class count; the builder emits two examples
per class-pair, so the effective bar was already 40 examples while the constant
read 20. The numbers below moved because the UNIT moved, not the strictness.)

Tests the trigger conditions for GEPA prompt optimization.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger


class TestGEPAOptimizationTrigger:
    """Tests for GEPAOptimizationTrigger class."""

    @pytest.fixture
    def trigger(self):
        """Create default trigger."""
        return GEPAOptimizationTrigger()

    @pytest.fixture
    def custom_trigger(self):
        """Create trigger with custom settings."""
        return GEPAOptimizationTrigger(
            min_trainset_examples=50,
            min_reward_delta=0.03,
            cooldown_hours=12,
            max_hours_without_optimization=72,
        )

    # Default configuration tests
    def test_default_config(self, trigger):
        """Test default configuration values."""
        # 40 examples == the 20-per-class bar that has been in force since #1677,
        # restated in the unit the gate actually compares against.
        assert trigger.min_trainset_examples == 40
        assert trigger.min_reward_delta == 0.05
        assert trigger.cooldown_hours == 24
        assert trigger.max_hours_without_optimization == 168
        assert trigger.critical_pattern_triggers is True

    def test_custom_config(self, custom_trigger):
        """Test custom configuration values."""
        assert custom_trigger.min_trainset_examples == 50
        assert custom_trigger.min_reward_delta == 0.03
        assert custom_trigger.cooldown_hours == 12

    # Cooldown tests
    def test_cooldown_blocks_trigger(self, trigger):
        """Trigger should be blocked during cooldown period."""
        last_opt = datetime.now(timezone.utc) - timedelta(hours=12)

        should_trigger, reason = trigger.should_trigger(
            trainset_examples=200,
            current_reward=0.8,
            baseline_reward=0.5,  # Big improvement
            last_optimization=last_opt,
        )

        assert should_trigger is False
        assert "Cooldown active" in reason

    def test_cooldown_expired_allows_trigger(self, trigger):
        """Trigger should work after cooldown expires."""
        last_opt = datetime.now(timezone.utc) - timedelta(hours=30)

        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.75,
            baseline_reward=0.65,
            last_optimization=last_opt,
        )

        assert should_trigger is True
        assert "Reward improved" in reason

    def test_no_last_optimization_no_cooldown(self, trigger):
        """First optimization should not be blocked by cooldown."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.75,
            baseline_reward=0.65,
            last_optimization=None,
        )

        assert should_trigger is True

    # Signal count tests
    def test_insufficient_signals_blocks_trigger(self, trigger):
        """Trigger should be blocked without enough signals (below new default of 20)."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=15,  # below the 40-example default
            current_reward=0.9,
            baseline_reward=0.5,
            last_optimization=None,
        )

        assert should_trigger is False
        assert "Insufficient trainset" in reason
        assert "examples" in reason

    def test_sufficient_signals_allows_trigger(self, trigger):
        """Trigger should work with sufficient signals."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.75,
            baseline_reward=0.65,
            last_optimization=None,
        )

        assert should_trigger is True

    # Reward delta tests
    def test_positive_reward_delta_triggers(self, trigger):
        """Improvement in reward should trigger optimization."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.75,
            baseline_reward=0.65,
            last_optimization=None,
        )

        assert should_trigger is True
        assert "Reward improved" in reason

    def test_negative_reward_delta_triggers(self, trigger):
        """Degradation in reward should trigger optimization."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.55,
            baseline_reward=0.75,
            last_optimization=None,
        )

        assert should_trigger is True
        assert "Reward degraded" in reason

    def test_small_reward_delta_no_trigger(self, trigger):
        """Small reward change should not trigger optimization."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.72,
            baseline_reward=0.70,
            last_optimization=None,
        )

        assert should_trigger is False
        assert "No trigger" in reason

    # Forced optimization tests
    def test_forced_optimization_after_max_hours(self, trigger):
        """Optimization should be forced after max hours without run."""
        last_opt = datetime.now(timezone.utc) - timedelta(hours=200)

        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.70,
            baseline_reward=0.70,  # No delta
            last_optimization=last_opt,
        )

        assert should_trigger is True
        assert "Forced" in reason

    # Critical pattern tests
    def test_critical_patterns_trigger_with_half_signals(self, trigger):
        """Critical patterns should trigger with half the required signals."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=10,  # A3: exactly half of new default min_trainset_examples=20
            current_reward=0.70,
            baseline_reward=0.70,  # No delta
            last_optimization=None,
            has_critical_patterns=True,
        )

        assert should_trigger is True
        assert "Critical patterns" in reason

    def test_critical_patterns_need_minimum_signals(self, trigger):
        """Critical patterns still need minimum signals (half of new default=20 → need >=10)."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=5,  # A3: less than half of 20 (i.e. < 10)
            current_reward=0.70,
            baseline_reward=0.70,
            last_optimization=None,
            has_critical_patterns=True,
        )

        # Should fall through to regular checks
        assert should_trigger is False

    def test_critical_patterns_disabled(self):
        """Critical patterns can be disabled."""
        trigger = GEPAOptimizationTrigger(critical_pattern_triggers=False)

        should_trigger, reason = trigger.should_trigger(
            trainset_examples=15,  # below BOTH the 40-example bar and nothing else
            current_reward=0.70,
            baseline_reward=0.70,
            last_optimization=None,
            has_critical_patterns=True,
        )

        # Should fall through and fail on the trainset size
        assert should_trigger is False
        assert "Insufficient trainset" in reason

    # Budget recommendation tests
    def test_budget_heavy_for_critical_patterns(self, trigger):
        """Critical patterns should recommend heavy budget."""
        budget = trigger.get_recommended_budget(
            trainset_examples=100,
            hours_since_last=24,
            has_critical_patterns=True,
        )

        assert budget == "heavy"

    def test_budget_heavy_for_long_delay(self, trigger):
        """Long time since optimization should recommend heavy budget."""
        budget = trigger.get_recommended_budget(
            trainset_examples=100,
            hours_since_last=200,
            has_critical_patterns=False,
        )

        assert budget == "heavy"

    def test_budget_heavy_for_many_signals(self, trigger):
        """Many signals should recommend heavy budget."""
        budget = trigger.get_recommended_budget(
            trainset_examples=82,  # BUDGET_MIN_TRAINSET_EXAMPLES["heavy"] → heavy
            hours_since_last=24,
            has_critical_patterns=False,
        )

        assert budget == "heavy"

    def test_budget_medium_for_moderate_signals(self, trigger):
        """Moderate signals should recommend medium budget."""
        budget = trigger.get_recommended_budget(
            trainset_examples=52,  # BUDGET_MIN_TRAINSET_EXAMPLES["medium"], below heavy → medium
            hours_since_last=24,
            has_critical_patterns=False,
        )

        assert budget == "medium"

    def test_budget_light_for_few_signals(self, trigger):
        """Few signals should recommend light budget."""
        budget = trigger.get_recommended_budget(
            trainset_examples=25,  # below medium's 52 → light
            hours_since_last=24,
            has_critical_patterns=False,
        )

        assert budget == "light"


class TestGEPAOptimizationTriggerEdgeCases:
    """Edge case tests for GEPA trigger."""

    @pytest.fixture
    def trigger(self):
        """Create default trigger."""
        return GEPAOptimizationTrigger()

    def test_zero_signals(self, trigger):
        """Zero signals should not trigger."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=0,
            current_reward=0.9,
            baseline_reward=0.5,
            last_optimization=None,
        )

        assert should_trigger is False

    def test_zero_reward(self, trigger):
        """Zero rewards should be handled correctly."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.0,
            baseline_reward=0.0,
            last_optimization=None,
        )

        assert should_trigger is False
        assert "No trigger" in reason

    def test_negative_baseline_reward(self, trigger):
        """Negative baseline should still work."""
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.5,
            baseline_reward=-0.1,  # Unusual but possible
            last_optimization=None,
        )

        # Delta is 0.6, which exceeds threshold
        assert should_trigger is True

    def test_future_last_optimization(self, trigger):
        """Future last optimization timestamp should block (cooldown)."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)

        should_trigger, reason = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.9,
            baseline_reward=0.5,
            last_optimization=future_time,
        )

        assert should_trigger is False
        assert "Cooldown" in reason

    def test_exactly_at_thresholds(self, trigger):
        """Test behavior at exact threshold values."""
        # Exactly at the trainset threshold
        should_trigger1, _ = trigger.should_trigger(
            trainset_examples=40,  # exactly at the 40-example default
            current_reward=0.75,
            baseline_reward=0.65,
            last_optimization=None,
        )
        assert should_trigger1 is True

        # At reward delta threshold (use values that avoid floating point issues)
        # Note: 0.70 - 0.65 = 0.04999... due to floating point, so use 0.80 - 0.75
        should_trigger2, _ = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.80,
            baseline_reward=0.75,  # Delta is 0.05 (clears threshold in floating point)
            last_optimization=None,
        )
        assert should_trigger2 is True


class TestGEPAOptimizationTriggerIntegration:
    """Integration tests simulating real usage patterns."""

    def test_typical_optimization_cycle(self):
        """Test a typical optimization cycle scenario."""
        trigger = GEPAOptimizationTrigger()
        now = datetime.now(timezone.utc)

        # Day 1: trainset too small (below the 40-example default)
        should_trigger, _ = trigger.should_trigger(
            trainset_examples=15,
            current_reward=0.65,
            baseline_reward=0.60,
            last_optimization=None,
        )
        assert should_trigger is False

        # Day 2: trainset clears the bar, improvement detected
        should_trigger, _ = trigger.should_trigger(
            trainset_examples=44,
            current_reward=0.72,
            baseline_reward=0.60,
            last_optimization=None,
        )
        assert should_trigger is True

        # Same day: Cooldown blocks
        last_opt = now
        should_trigger, _ = trigger.should_trigger(
            trainset_examples=150,
            current_reward=0.80,
            baseline_reward=0.72,
            last_optimization=last_opt,
        )
        assert should_trigger is False

        # Day 3: Cooldown expired, stable performance
        last_opt = now - timedelta(hours=30)
        should_trigger, _ = trigger.should_trigger(
            trainset_examples=180,
            current_reward=0.73,
            baseline_reward=0.72,  # Only 0.01 improvement
            last_optimization=last_opt,
        )
        assert should_trigger is False

        # Day 10: Forced optimization
        last_opt = now - timedelta(hours=200)
        should_trigger, _ = trigger.should_trigger(
            trainset_examples=300,
            current_reward=0.73,
            baseline_reward=0.72,
            last_optimization=last_opt,
        )
        assert should_trigger is True

    def test_emergency_optimization_scenario(self):
        """Test emergency optimization with critical patterns."""
        trigger = GEPAOptimizationTrigger()
        now = datetime.now(timezone.utc)

        # Recent optimization but critical patterns detected
        last_opt = now - timedelta(hours=6)  # Only 6 hours ago

        # Normally blocked by cooldown
        should_trigger, _ = trigger.should_trigger(
            trainset_examples=60,
            current_reward=0.50,
            baseline_reward=0.70,
            last_optimization=last_opt,
            has_critical_patterns=False,
        )
        assert should_trigger is False

        # But after cooldown with critical patterns and half signals
        last_opt = now - timedelta(hours=30)
        should_trigger, reason = trigger.should_trigger(
            trainset_examples=60,
            current_reward=0.50,
            baseline_reward=0.70,
            last_optimization=last_opt,
            has_critical_patterns=True,
        )
        assert should_trigger is True
        assert "Critical" in reason
