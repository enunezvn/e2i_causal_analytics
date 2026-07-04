"""CI guard: every beat_schedule entry must reference a registered task (#897).

This is the third occurrence of the same failure class:

1. ``monitor-drift`` -> ``src.tasks.monitor_model_drift`` (dangling, repointed
   to ``check_all_production_models``; per-entry guard in
   test_celery_retraining_wiring.py).
2. ``health-check`` -> ``src.tasks.health_check`` (dangling since the initial
   scaffold commit; worker KeyError every hour).
3. ``cache-cleanup`` -> ``src.tasks.cleanup_old_cache`` (dangling since the
   initial scaffold commit; worker KeyError every day).

A beat entry whose task name is registered nowhere does NOT fail at import or
deploy time: the scheduler happily enqueues the message every cycle, and the
worker that picks it up raises ``KeyError: '<task name>'`` ("Received
unregistered task of type ..."). The message is dropped, beat reports "sent",
and nothing ever runs — operational noise plus a false sense of coverage.

This guard kills the whole class instead of playing per-entry whack-a-mole.

Faithfulness note: production worker boot (``celery -A src.workers.celery_app
worker``) registers tasks by importing package ``__init__`` modules via the
deferred ``autodiscover_tasks([...])`` hook (none of the listed packages has a
``tasks.py`` submodule, so the package inits ARE the registration path).
Replaying the full hook here (``loader.import_default_modules()``) drags the
entire ``src.agents``/``src.mlops`` import tree (sentence_transformers,
transformers, ...) through a 30s-per-test CI budget, so we import the bounded
subset that defines every beat-referenced task today:

- ``src.workers.celery_app``  (debug task, DLQ monitor, monitoring tasks)
- ``src.tasks``               (package init eagerly imports all task modules)
- ``src.etl``                 (package init eagerly imports the rollup ETLs)

This is fail-closed, never false-green: a future beat entry pointing at a task
registered only under another autodiscovered package (src.mlops, src.causal,
src.digital_twin, src.agents) will FAIL this test even though production would
register it — in that case, add the defining module to the imports below.
"""

from __future__ import annotations

import src.etl  # noqa: F401 — registers the src.etl.* rollup tasks
import src.tasks  # noqa: F401 — registers all src.tasks.* task modules
from src.workers.celery_app import celery_app


def test_every_beat_entry_references_a_registered_task() -> None:
    registered = set(celery_app.tasks.keys())
    beat_schedule = celery_app.conf.beat_schedule

    # Sanity floor: schedule and registry are both non-trivially populated, so
    # a green run can't come from an empty-vs-empty comparison (e.g. the
    # schedule moving out of conf.beat_schedule, or task imports going dead).
    assert len(beat_schedule) >= 20, (
        f"beat_schedule unexpectedly small ({len(beat_schedule)} entries) — "
        "did the schedule move out of celery_app.conf.beat_schedule?"
    )
    assert len(registered) >= 40, (
        f"task registry unexpectedly small ({len(registered)} tasks) — "
        "did the src.tasks/src.etl registration imports stop working?"
    )

    dangling = {
        beat_key: entry["task"]
        for beat_key, entry in beat_schedule.items()
        if entry["task"] not in registered
    }
    assert not dangling, (
        "beat_schedule entries reference task names that no worker registers. "
        "Each tick enqueues a message the worker rejects with KeyError "
        "('Received unregistered task of type ...'); the task silently never "
        f"runs. Dangling entries: {dangling}. "
        "Fix by repointing to a registered task, implementing the task, or "
        "removing the entry (intent first — see issue #897). If the task IS "
        "registered in production via another autodiscovered package, add "
        "that module to this file's imports instead."
    )


def test_no_task_is_beat_scheduled_twice() -> None:
    """Regression (2026-07-04 alert storm): check_all_production_models was
    scheduled under TWO entries — "monitor-drift" in conf.beat_schedule and
    "drift-detection-sweep" via an on_after_finalize/add_periodic_task hook in
    drift_monitoring_tasks.py — so every cycle ran the full drift sweep twice
    (720 monitoring runs for 360 models, 10,080 duplicate alerts in one
    morning). Guard the whole class: no task name may appear in more than one
    beat entry, and hook-added entries land in conf.beat_schedule too once the
    app is finalized, so this catches a reintroduced add_periodic_task."""
    celery_app.finalize()  # flush any pending add_periodic_task registrations

    by_task: dict[str, list[str]] = {}
    for beat_key, entry in celery_app.conf.beat_schedule.items():
        by_task.setdefault(entry["task"], []).append(beat_key)

    duplicated = {task: keys for task, keys in by_task.items() if len(keys) > 1}
    assert not duplicated, (
        f"tasks scheduled by more than one beat entry (double-fire): {duplicated}"
    )

    # The two drift-monitoring schedules live in celery_app.py's beat_schedule
    # (single source of truth) — pin their presence so a refactor that moves
    # them back to a runtime hook (invisible to this dict before finalize)
    # fails loudly.
    assert by_task.get("src.tasks.check_all_production_models") == ["monitor-drift"]
    assert by_task.get("src.tasks.cleanup_old_drift_history") == ["drift-history-cleanup"]
