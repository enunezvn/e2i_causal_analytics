"""Phase D (D4) + #2207: Celery wiring for the live retraining trigger.

- the drift-monitoring beat entry must reference a task that actually exists
  (the prior `src.tasks.monitor_model_drift` was a dangling ref that would
  crash beat at fire time).
- execute_model_retraining runs a full MLFoundationPipeline. Phase D routed it to
  worker_heavy's `ml` queue; #2207 measured that queue dark (worker_heavy
  replicas: 0, owner decision #705) and the pipeline's footprint inside
  worker_medium (peak 838 MB tree RSS / 126 s for the tier-0 harness at n=4000
  with 2 HPO trials; the four model-comparison fits at the prod cohort shape
  15,209x77 add ~75 MB over a 326 MB import floor, RandomForest the slowest at
  19 s per fit), so it now rides worker_medium's `analytics` queue (4G cgroup) —
  never worker_light's 1.5G one.
"""

from __future__ import annotations

import re
from pathlib import Path

from celery.schedules import crontab

import src.tasks.drift_monitoring_tasks  # noqa: F401 — register tasks on the app
from src.workers.celery_app import celery_app

REPO = Path(__file__).resolve().parents[3]


def _compose_queues(service: str) -> set[str]:
    text = (REPO / "docker" / "docker-compose.yml").read_text()
    block = text.split(f"\n  {service}:\n", 1)[1]
    m = re.search(r"--queues=([a-z_,]+)", block)
    assert m, f"{service} has no --queues in docker/docker-compose.yml"
    return set(m.group(1).split(","))


def test_execute_model_retraining_routes_to_a_queue_worker_medium_consumes() -> None:
    routes = celery_app.conf.task_routes
    queue = routes.get("src.tasks.execute_model_retraining", {}).get("queue")
    assert queue in _compose_queues("worker_medium"), queue
    assert queue not in _compose_queues("worker_light"), (
        "a ~1 GB training pipeline must not land inside worker_light's 1.5G cgroup"
    )
    assert queue != "ml", "the ml queue has no consumer on this box (#705)"


def test_retraining_evaluation_tasks_route_to_a_light_consumed_queue() -> None:
    routes = celery_app.conf.task_routes
    light = _compose_queues("worker_light")
    for task in ("src.tasks.check_retraining_for_all_models", "src.tasks.evaluate_retraining_need"):
        assert routes.get(task, {}).get("queue") in light, task


def test_retraining_evaluation_is_beat_scheduled_daily_on_a_consumed_queue() -> None:
    """#2207: check_retraining_for_all_models was implemented, route-reachable and never
    fired — it was in no beat entry. Now it is, on a wall-clock slot (#1645)."""
    entry = celery_app.conf.beat_schedule["retraining-evaluation-daily"]
    assert entry["task"] == "src.tasks.check_retraining_for_all_models"
    assert entry["task"] in celery_app.tasks
    assert isinstance(entry["schedule"], crontab)
    assert entry["options"]["queue"] in _compose_queues("worker_light")
    # approval semantics unchanged: the sweep evaluates; only critical drift
    # (>= auto_approve_threshold) triggers without a human
    assert not entry.get("kwargs", {}).get("auto_approve", False)


def test_monitor_drift_beat_references_a_registered_task() -> None:
    entry = celery_app.conf.beat_schedule["monitor-drift"]
    task_name = entry["task"]
    assert task_name in celery_app.tasks, (
        f"beat 'monitor-drift' references task {task_name!r} which is not registered "
        f"on the Celery app — it would crash the scheduler at fire time."
    )
