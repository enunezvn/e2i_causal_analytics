"""Offline guard: the durable corpus-sync task is registered + scheduled (F3b).

Without this, the corpus would still rely on a one-off manual run. Importing
``src.tasks`` runs the load-bearing registration imports in src/tasks/__init__.py;
the assertions then confirm the Celery worker will discover the task and the beat
scheduler will fire it. CI-collectable (no live backends needed for registration).
"""

import src.tasks  # noqa: F401  (triggers task registration via __init__)
from src.workers.celery_app import celery_app

TASK_NAME = "src.tasks.sync_operational_corpus"
BEAT_KEY = "sync-operational-corpus"


def test_task_is_registered_with_celery():
    assert TASK_NAME in celery_app.tasks, (
        f"{TASK_NAME} not registered; the worker will not discover it"
    )
    assert src.tasks.sync_operational_corpus.name == TASK_NAME


def test_beat_schedule_fires_the_sync_daily():
    bs = celery_app.conf.beat_schedule
    assert BEAT_KEY in bs, "no beat entry -> corpus never re-syncs (regresses to one-off)"
    entry = bs[BEAT_KEY]
    assert entry["task"] == TASK_NAME
    assert entry["schedule"] == 86400.0  # daily
    assert entry["options"]["queue"] == "analytics"
