"""Shard 08: DSPy prompt-optimization Celery task + trigger gating."""

from __future__ import annotations


def _signal(reward: float) -> dict:
    return {
        "source_agent": "feedback_learner",
        "reward": reward,
        "input_context": {"feedback_batch": [{"x": 1}]},
        "output": {"patterns": [{"severity": "high"}]},
    }


def test_decide_trigger_skips_below_min_signals():
    from src.tasks.dspy_optimization_tasks import _decide_trigger

    should, reason = _decide_trigger([_signal(0.9)] * 3, state={})  # < min_signals (100 default)
    assert should is False
    assert reason


def test_decide_trigger_fires_with_enough_high_reward_signals_first_run():
    from src.tasks.dspy_optimization_tasks import _decide_trigger

    # 120 high-reward signals, no prior optimization -> reward delta vs 0 triggers.
    should, reason = _decide_trigger([_signal(0.9)] * 120, state={})
    assert should is True
    assert reason


def test_task_is_registered():
    # Registration happens when the src.tasks package __init__ imports the task
    # module (the project's task-discovery mechanism); importing celery_app alone
    # does not finalize autodiscover.
    import src.tasks  # noqa: F401
    from src.workers.celery_app import celery_app

    assert "src.tasks.run_dspy_prompt_optimization" in celery_app.tasks


def test_beat_schedule_has_dspy_entry():
    from src.workers.celery_app import celery_app

    beat = celery_app.conf.beat_schedule
    assert "dspy-prompt-optimization-daily" in beat
    assert (
        beat["dspy-prompt-optimization-daily"]["task"] == "src.tasks.run_dspy_prompt_optimization"
    )
