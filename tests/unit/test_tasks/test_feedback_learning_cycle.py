"""Unit tests for run_feedback_learning_cycle Celery beat task (A2).

Offline, no real LM/DB: FeedbackLearnerAgent.learn is patched to an
AsyncMock so the task runs end-to-end in pure Python.

Coverage:
- Window computation: start < end, ISO-8601 strings, learn() called once
- Return dict shape: {"status": "completed", "feedback_count", "training_reward", "task_id"}
- DSPY_LEARN_WINDOW_HOURS env override changes window size
- src.tasks.__all__ exports the symbol
- beat_schedule contains the task
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_output(feedback_count: int = 7, training_reward: float = 0.82) -> MagicMock:
    """Return a MagicMock shaped like FeedbackLearnerOutput."""
    out = MagicMock()
    out.status = "completed"
    out.feedback_count = feedback_count
    out.training_reward = training_reward
    return out


# Patch target: the FeedbackLearnerAgent class at its source module so that the
# deferred `from src.agents.feedback_learner.agent import FeedbackLearnerAgent`
# inside _run_learning_cycle sees the mock.
_AGENT_PATCH = "src.agents.feedback_learner.agent.FeedbackLearnerAgent"


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


def test_task_calls_learn_once_with_window():
    """Task constructs a (start, end) window and calls learn() exactly once."""
    fake_output = _make_fake_output()

    with patch(_AGENT_PATCH) as MockAgent:
        mock_instance = MagicMock()
        mock_instance.learn = AsyncMock(return_value=fake_output)
        MockAgent.return_value = mock_instance

        from src.tasks.dspy_optimization_tasks import run_feedback_learning_cycle

        before = datetime.now(timezone.utc)
        run_feedback_learning_cycle.run()
        after = datetime.now(timezone.utc)

        # learn() called exactly once
        mock_instance.learn.assert_called_once()
        call_kwargs = mock_instance.learn.call_args

        # Positional or keyword — normalise
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()

        # Resolve start / end from either kwargs or positional args (0, 1)
        start_iso = kwargs.get("time_range_start") or (args[0] if args else None)
        end_iso = kwargs.get("time_range_end") or (args[1] if len(args) > 1 else None)

        assert start_iso is not None, "time_range_start not passed to learn()"
        assert end_iso is not None, "time_range_end not passed to learn()"

        # Both must be parseable ISO timestamps
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = datetime.fromisoformat(end_iso)
        assert start_dt < end_dt, "window start must precede end"

        # end_dt should be within the test's wall-clock window (generous ±5s)
        assert before <= end_dt <= after or abs((end_dt - after).total_seconds()) < 5


def test_task_returns_completed_dict():
    """Return dict has status=completed and expected keys."""
    fake_output = _make_fake_output(feedback_count=13, training_reward=0.75)

    with patch(_AGENT_PATCH) as MockAgent:
        mock_instance = MagicMock()
        mock_instance.learn = AsyncMock(return_value=fake_output)
        MockAgent.return_value = mock_instance

        from src.tasks.dspy_optimization_tasks import run_feedback_learning_cycle

        result = run_feedback_learning_cycle.run()

    assert result["status"] == "completed"
    assert result["feedback_count"] == 13
    assert result["training_reward"] == 0.75
    # task_id is self.request.id — may be None in direct .run() calls
    assert "task_id" in result


def test_task_returns_failed_on_exception():
    """When learn() raises, task returns status=failed (never propagates)."""
    with patch(_AGENT_PATCH) as MockAgent:
        mock_instance = MagicMock()
        mock_instance.learn = AsyncMock(side_effect=RuntimeError("DB unreachable"))
        MockAgent.return_value = mock_instance

        from src.tasks.dspy_optimization_tasks import run_feedback_learning_cycle

        result = run_feedback_learning_cycle.run()

    assert result["status"] == "failed"
    assert "task_id" in result
    assert "error" in result


def test_window_hours_env_override():
    """DSPY_LEARN_WINDOW_HOURS=48 doubles the window."""
    fake_output = _make_fake_output()

    with patch.dict(os.environ, {"DSPY_LEARN_WINDOW_HOURS": "48"}):
        with patch(_AGENT_PATCH) as MockAgent:
            mock_instance = MagicMock()
            mock_instance.learn = AsyncMock(return_value=fake_output)
            MockAgent.return_value = mock_instance

            from src.tasks.dspy_optimization_tasks import run_feedback_learning_cycle

            run_feedback_learning_cycle.run()

            call_kwargs = mock_instance.learn.call_args
            kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
            args = call_kwargs.args if call_kwargs.args else ()
            start_iso = kwargs.get("time_range_start") or (args[0] if args else None)
            end_iso = kwargs.get("time_range_end") or (args[1] if len(args) > 1 else None)

            start_dt = datetime.fromisoformat(start_iso)
            end_dt = datetime.fromisoformat(end_iso)
            window_hours = (end_dt - start_dt).total_seconds() / 3600

    assert abs(window_hours - 48.0) < 0.1, f"Expected ~48h window, got {window_hours}h"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_run_feedback_learning_cycle_in_all():
    """The symbol must be in src.tasks.__all__ for Celery discovery."""
    import src.tasks

    assert "run_feedback_learning_cycle" in src.tasks.__all__


def test_run_feedback_learning_cycle_importable():
    """Direct import from src.tasks must succeed."""
    from src.tasks import run_feedback_learning_cycle  # noqa: F401


def test_beat_schedule_contains_task():
    """beat_schedule must contain an entry whose task is the learning cycle."""
    import src.tasks  # noqa: F401 — triggers task registration
    from src.workers.celery_app import celery_app

    scheduled_tasks = {entry["task"] for entry in celery_app.conf.beat_schedule.values()}
    assert "src.tasks.run_feedback_learning_cycle" in scheduled_tasks
