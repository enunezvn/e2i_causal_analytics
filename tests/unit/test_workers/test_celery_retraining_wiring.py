"""Phase D (D4): Celery wiring for the live retraining trigger.

  - execute_model_retraining runs a full training pipeline → must be routed to
    the heavy `ml` queue (worker_heavy), not the default queue.
  - the drift-monitoring beat entry must reference a task that actually exists
    (the prior `src.tasks.monitor_model_drift` was a dangling ref that would
    crash beat at fire time).
"""

from __future__ import annotations

import src.tasks.drift_monitoring_tasks  # noqa: F401 — register tasks on the app
from src.workers.celery_app import celery_app


def test_execute_model_retraining_routes_to_ml_queue() -> None:
    routes = celery_app.conf.task_routes
    assert routes.get("src.tasks.execute_model_retraining") == {"queue": "ml"}


def test_monitor_drift_beat_references_a_registered_task() -> None:
    entry = celery_app.conf.beat_schedule["monitor-drift"]
    task_name = entry["task"]
    assert task_name in celery_app.tasks, (
        f"beat 'monitor-drift' references task {task_name!r} which is not registered "
        f"on the Celery app — it would crash the scheduler at fire time."
    )
