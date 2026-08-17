"""Shard 08: DSPy prompt-optimization Celery task + trigger gating."""

from __future__ import annotations


def _signal(reward: float, *, patterns: bool = True) -> dict:
    """One persisted signal row.

    #1668: ``patterns`` is the LABEL, and the trigger now counts the scarcer
    label class rather than the row count — so a pool of N identical
    pattern-bearing rows has a trainable supply of ZERO, not N. That is not a
    fixture quirk: ``_signals_to_examples`` refuses a single-class pool, so such
    a beat would fire and compile nothing.
    """
    return {
        "source_agent": "feedback_learner",
        "reward": reward,
        "input_context": {"feedback_batch": [{"x": 1}]},
        "output": {"patterns": [{"severity": "high"}] if patterns else []},
    }


def _pool(k: int, reward: float = 0.9) -> list:
    """``k`` positives + ``k`` negatives -> trainable supply ``k``."""
    return [_signal(reward)] * k + [_signal(0.0, patterns=False)] * k


def test_decide_trigger_skips_below_min_signals():
    from src.tasks.dspy_optimization_tasks import _decide_trigger

    should, reason = _decide_trigger(_pool(3), state={})  # supply 3 < min_signals
    assert should is False
    assert reason


def test_decide_trigger_fires_with_enough_high_reward_signals_first_run():
    from src.tasks.dspy_optimization_tasks import _decide_trigger

    # 120 trainable pairs, no prior optimization -> reward delta vs 0 triggers.
    should, reason = _decide_trigger(_pool(120), state={})
    assert should is True
    assert reason


def test_decide_trigger_does_not_fire_on_a_single_class_pool():
    """240 high-reward rows the trainset builder would refuse must not open it.

    This is #1668's own defect, stated as a test: before the fix the gate
    counted rows, so this pool read as 240 >= 20 and fired — then built a
    zero-example trainset.
    """
    from src.tasks.dspy_optimization_tasks import _decide_trigger

    should, reason = _decide_trigger([_signal(0.9)] * 240, state={})
    assert should is False
    assert reason == "Insufficient signals: 0 < 20"


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
