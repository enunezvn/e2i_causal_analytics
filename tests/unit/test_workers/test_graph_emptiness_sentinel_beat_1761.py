"""CI guard: the graph-emptiness sentinel must be beat-scheduled on a consumed queue (#1761).

#1758 wiped the FalkorDB knowledge graph and ``/knowledge-graph`` stayed empty for
four days because nothing in the platform notices, let alone heals, an empty graph.
#1759 removed the known wipe vector (``FALKORDB_DATA_PATH=/data``); this entry is the
recovery half — a 30-minute tick that probes the curated core and reseeds when it is
gone.

Two ways this could ship dead, both guarded here:

* wrong/absent beat entry  -> the task never fires at all;
* a queue no worker consumes -> beat reports "sent" every tick and the message rots.
  ``quick`` is consumed by ``worker_light`` (``--queues=default,quick,api`` in
  docker/docker-compose.yml), which is also where the memory budget for the reseed
  subprocess lives (#1761's ``src.rag`` import shed).

``test_beat_schedule_registration.py`` separately enforces that the referenced task
name is actually registered, and ``test_beat_daily_wallclock_1645.py`` that no entry
regresses to a bare >=24h interval.
"""

from __future__ import annotations

import src.tasks  # noqa: F401 — registers all src.tasks.* task modules
from src.workers.celery_app import celery_app

BEAT_KEY = "graph-emptiness-sentinel"
TASK_NAME = "src.tasks.graph_emptiness_sentinel"

# worker_light consumes default,quick,api (docker/docker-compose.yml command).
LIGHT_WORKER_QUEUES = {"default", "quick", "api"}


def test_graph_emptiness_sentinel_is_beat_scheduled() -> None:
    entry = celery_app.conf.beat_schedule.get(BEAT_KEY)
    assert entry is not None, (
        f"beat entry {BEAT_KEY!r} is missing — the knowledge graph has no "
        "emptiness sentinel and an empty graph heals only when a human notices "
        "(#1758 took four days). See #1761."
    )
    assert entry["task"] == TASK_NAME, (
        f"{BEAT_KEY!r} must target {TASK_NAME!r}, got {entry['task']!r}"
    )


def test_graph_emptiness_sentinel_runs_every_30_minutes() -> None:
    entry = celery_app.conf.beat_schedule[BEAT_KEY]
    schedule = entry["schedule"]
    assert isinstance(schedule, (int, float)), (
        f"{BEAT_KEY!r} schedule must be a plain interval, got {schedule!r}"
    )
    assert float(schedule) == 1800.0, (
        f"{BEAT_KEY!r} must tick every 30 minutes (1800s), got {schedule!r}. A "
        "recovery sentinel is only worth its cost if the outage window it bounds "
        "is short."
    )


def test_graph_emptiness_sentinel_lands_on_a_consumed_queue() -> None:
    queue = celery_app.conf.beat_schedule[BEAT_KEY]["options"]["queue"]
    assert queue in LIGHT_WORKER_QUEUES, (
        f"{BEAT_KEY!r} is routed to {queue!r}, which worker_light does not consume "
        f"({sorted(LIGHT_WORKER_QUEUES)}). Beat would report 'sent' every tick while "
        "the message rots in an unconsumed queue."
    )
    assert queue == "quick", (
        f"{BEAT_KEY!r} must sit on 'quick' (the light-worker fast lane), got {queue!r}"
    )
