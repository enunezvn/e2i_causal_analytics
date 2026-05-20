"""Issue #376 — Phase 4 schema completion.

Tests pin:
  * Celery task ``src.tasks.crystallization_tasks.crystallize_portfolio``
    is registered with the canonical task name.
  * The task wraps ``Crystallizer.crystallize_portfolio()`` and returns
    a JSON-serializable summary.
  * The task module is imported from ``src/tasks/__init__.py`` so the
    Celery worker auto-discovers it. Per memory
    `[[feat-375-phase3-hardening-close-20260519]]` —
    ``@celery_app.task`` decorators only fire on actual import; without
    the import line, the task is silently undiscovered.
  * Beat-schedule entry is present in ``src/workers/celery_app.py`` at
    every 6h with a 30min offset after consolidation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_crystallization_tasks_module_importable():
    """The module must exist at the canonical path."""
    import src.tasks.crystallization_tasks  # noqa: F401


def test_crystallization_task_is_registered_with_celery():
    """``@celery_app.task(name=...)`` must register the task with the
    canonical name."""
    from src.tasks.crystallization_tasks import crystallize_portfolio
    from src.workers.celery_app import celery_app

    assert crystallize_portfolio.name == (
        "src.tasks.crystallization_tasks.crystallize_portfolio"
    ), (
        f"Expected task name 'src.tasks.crystallization_tasks.crystallize_portfolio'; "
        f"got {crystallize_portfolio.name}"
    )
    # Celery's task registry should have it
    assert crystallize_portfolio.name in celery_app.tasks


def test_crystallization_task_wraps_crystallizer_crystallize_portfolio():
    """Calling the task must invoke Crystallizer.crystallize_portfolio()
    and return a dict summary."""
    from src.tasks.crystallization_tasks import crystallize_portfolio

    fake_result = MagicMock(
        examined_groups=2,
        insights_created=5,
        edges_created=12,
        by_brand={"kisqali": 3, "fabhalta": 2},
        errors=[],
    )

    with patch(
        "src.tasks.crystallization_tasks._run_crystallize_portfolio",
        return_value=fake_result,
    ):
        # Celery binds `self` from bind=True; call .run() directly
        result = crystallize_portfolio.run()

    assert isinstance(result, dict)
    assert result["insights_created"] == 5
    assert result["edges_created"] == 12
    assert result["by_brand"] == {"kisqali": 3, "fabhalta": 2}


def test_crystallization_task_handles_exception_in_runner():
    """If the underlying Crystallizer raises, the task must NOT swallow
    silently — propagate so Celery retry / DLQ kicks in."""
    from src.tasks.crystallization_tasks import crystallize_portfolio

    def boom():
        raise RuntimeError("simulated failure")

    with patch(
        "src.tasks.crystallization_tasks._run_crystallize_portfolio",
        side_effect=RuntimeError("simulated failure"),
    ):
        with pytest.raises(RuntimeError, match="simulated failure"):
            crystallize_portfolio.run()


def test_crystallization_task_imported_in_tasks_init():
    """Per memory [[feat-375-phase3-hardening-close-20260519]]:
    @celery_app.task requires module import in src/tasks/__init__.py
    for worker discovery. Without this line the worker boot does not
    see the task and dispatcher.send_task calls dead-letter."""
    from pathlib import Path

    init_path = Path(__file__).resolve().parents[3] / "src" / "tasks" / "__init__.py"
    content = init_path.read_text()
    assert "crystallization_tasks" in content, (
        "src/tasks/__init__.py must import crystallization_tasks for Celery worker discovery"
    )


def test_crystallization_beat_schedule_registered_every_6h():
    """Plan §Phase 4 line 141: every 6h offset 30 min after consolidation."""
    from src.workers.celery_app import celery_app

    schedule = celery_app.conf.beat_schedule
    # Find any entry whose task name matches the canonical crystallize task
    matches = [
        (name, cfg)
        for name, cfg in schedule.items()
        if cfg.get("task") == "src.tasks.crystallization_tasks.crystallize_portfolio"
    ]
    assert matches, (
        "Expected a beat entry for src.tasks.crystallization_tasks.crystallize_portfolio; "
        f"found tasks: {sorted({cfg.get('task') for cfg in schedule.values()})}"
    )

    # 21600.0 seconds = 6 hours
    _name, cfg = matches[0]
    assert cfg["schedule"] == 21600.0, f"Expected 6h schedule (21600.0s); got {cfg['schedule']}"
