"""Verify Celery worker discovery imports register sentinel action tasks (#375 iter-1 H2).

Codex iter-0 H2: ``src/tasks/__init__.py`` imports ``insight_lifecycle_tasks``
(consolidator + dispatcher) but NEVER imports ``src.tasks.sentinel_actions``.
As a result, ``celery worker`` boot, which imports ``src.tasks`` to discover
tasks, would NOT register the 4 plan-specced action handlers
(``rerun_all_active_cohorts`` et al.). The tasks would be findable only by
tests that explicitly import ``src.tasks.sentinel_actions`` — production
workers would silently fail to enqueue them.

This test imports ``src.tasks`` (no submodule) and asserts the 4 task names
are present in ``celery_app.tasks``.
"""

from __future__ import annotations


def test_celery_worker_discovery_registers_sentinel_action_tasks():
    """Plain ``import src.tasks`` must surface the 4 sentinel action handlers
    to the Celery task registry.

    Pre-fix: this test fails because ``src/tasks/__init__.py`` does not
    transitively import ``src.tasks.sentinel_actions``.

    Post-fix: a single ``from . import sentinel_actions`` in __init__.py
    causes the @celery_app.task decorators to register all four.
    """
    # Import the package; deliberately NOT the submodule.
    import src.tasks  # noqa: F401
    from src.workers.celery_app import celery_app

    expected = {
        "src.tasks.sentinel_actions.rerun_all_active_cohorts",
        "src.tasks.sentinel_actions.notify_and_queue_reanalysis",
        "src.tasks.sentinel_actions.flag_for_review",
        "src.tasks.sentinel_actions.run_full_consolidation",
    }
    registered = set(celery_app.tasks.keys())
    missing = expected - registered
    assert not missing, (
        f"Celery worker boot would NOT see plan-specced sentinel actions: "
        f"{sorted(missing)}. Add `from . import sentinel_actions` to "
        f"src/tasks/__init__.py."
    )
