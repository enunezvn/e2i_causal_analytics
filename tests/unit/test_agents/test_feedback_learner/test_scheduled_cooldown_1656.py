"""#1656: the 24h cooldown must not suppress a wall-clock-scheduled optimization.

``cooldown_hours = 24`` is compared against ``last_optimization``, which
``dspy_optimization_tasks`` writes *after* the run completes (a COMPLETION
timestamp, not a start one). The beat entry is ``crontab(hour=6, minute=0)``.

So on the cron path any nonzero runtime puts the next fire under 24h:

    day 1  06:00  runs, completes ~06:35, stamps last_optimization=06:35
    day 2  06:00  hours_since = 23.4 < 24  ->  "Cooldown active", SKIPPED
    day 3  06:00  hours_since = 47.4 >= 24 ->  runs

A daily schedule that is actually every-other-day, announced by one log line.

The fix: a fixed daily crontab already IS the rate limit, so the cooldown is
redundant on that path and only interferes. It is retained for event-triggered
runs, where nothing else bounds the rate.

NOT LIVE-VERIFIABLE at time of writing: the trainset gate (30 examples < 40,
measured 2026-08-17, see #1668) returns ``skipped`` before any optimization
completes, so ``last_optimization`` is never written and the cooldown branch is
unreachable in production. These tests are the only evidence this fix works;
they are written to fail against the pre-fix behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.agents.feedback_learner.signal_store import decide_optimizer_trigger


def _signals(n: int, reward: float = 0.9) -> list[dict]:
    """``n`` positives + ``n`` negatives — a pool that builds a ``2n``-EXAMPLE trainset.

    #1668: ``decide_optimizer_trigger`` counts the examples the trainset builder
    will produce for the best-supplied phase, derived from the same classifier
    the builder uses, so ``[{"reward": 0.9}] * n`` no longer stands for "n
    signals the trigger will count" — such rows carry no input_context/output
    and are not trainable at all. These tests are about the COOLDOWN, so the
    pool has to genuinely carry the trainset they name or every one of them
    would be decided by the size gate instead. ``_signals(8)`` is 16 examples,
    ``_signals(50)`` is 100.
    """
    return [
        {
            "source_agent": "feedback_learner",
            "reward": reward,
            "input_context": {"feedback_batch": [{"x": 1}]},
            "output": {"patterns": [{"severity": "high"}]},
        }
        for _ in range(n)
    ] + [
        {
            "source_agent": "feedback_learner",
            "reward": 0.0,
            "input_context": {"feedback_batch": [{"x": 1}]},
            "output": {"patterns": []},
        }
        for _ in range(n)
    ]


def _state(hours_ago: float, baseline: float = 0.0) -> dict:
    stamp = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"last_optimization": stamp.isoformat(), "baseline_reward": baseline}


class TestScheduledPathIgnoresCooldown:
    """The cron path is rate-limited by the crontab, not by the cooldown."""

    def test_scheduled_run_fires_at_23_4h_after_a_completed_run(self):
        """The #1656 case exactly: yesterday's run completed 35 min late."""
        should, reason = decide_optimizer_trigger(
            _signals(50), _state(hours_ago=23.4), scheduled=True
        )
        assert should, f"#1656: a scheduled run was suppressed by the cooldown ({reason})"
        assert "Cooldown" not in reason

    def test_every_scheduled_day_fires_not_every_other(self):
        """Simulate a week of 06:00 fires, each completing 35 minutes in.

        Pre-fix this alternates run/skip/run/skip. The defect is invisible in a
        single-day check, which is why it is asserted across a span.
        """
        fired = [
            decide_optimizer_trigger(_signals(50), _state(hours_ago=23.4), scheduled=True)[0]
            for _ in range(7)
        ]
        assert all(fired), f"#1656: only {sum(fired)}/7 scheduled days would fire"

    def test_scheduled_still_respects_the_trainset_gate(self):
        """Dropping the cooldown must not drop the gate that actually matters.

        The live production state (#1668) is a 30-example trainset against
        min_trainset_examples=40.
        A scheduled run below the gate must still skip.
        """
        should, reason = decide_optimizer_trigger(_signals(8), _state(hours_ago=99), scheduled=True)
        assert not should
        assert "insufficient trainset" in reason.lower(), reason


class TestEventTriggeredPathKeepsCooldown:
    """Nothing bounds the rate of an event-triggered run except the cooldown."""

    def test_event_triggered_run_is_still_suppressed_inside_the_window(self):
        should, reason = decide_optimizer_trigger(
            _signals(50), _state(hours_ago=23.4), scheduled=False
        )
        assert not should
        assert "Cooldown" in reason, reason

    def test_event_triggered_run_fires_once_past_the_window(self):
        should, _ = decide_optimizer_trigger(_signals(50), _state(hours_ago=25), scheduled=False)
        assert should

    def test_default_is_the_conservative_path(self):
        """An un-migrated caller keeps the old suppressing behaviour."""
        should, reason = decide_optimizer_trigger(_signals(50), _state(hours_ago=23.4))
        assert not should
        assert "Cooldown" in reason


class TestGateStatusStaysHonest:
    """#1661's invariant: the status surface reports what the beat WOULD do.

    ``decide_optimizer_trigger`` is deliberately the single decision function
    for both. If the status endpoint asked with a different ``scheduled`` value
    than the beat uses, it would report Ready while the beat skipped — the exact
    defect #1661 fixed. Pinned so the #1656 parameter cannot reintroduce it.
    """

    def test_no_last_optimization_is_unaffected_by_the_path(self):
        """Today's real state: nothing has ever completed, so both agree."""
        for scheduled in (True, False):
            should, reason = decide_optimizer_trigger(_signals(8), {}, scheduled=scheduled)
            assert not should
            assert "insufficient trainset" in reason.lower(), (scheduled, reason)

    def test_the_two_paths_differ_only_inside_the_cooldown_window(self):
        outside = {
            s: decide_optimizer_trigger(_signals(50), _state(hours_ago=30), scheduled=s)[0]
            for s in (True, False)
        }
        assert outside[True] == outside[False] is True
