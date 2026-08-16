"""Offline guard: the durable corpus-sync task is registered + scheduled (F3b).

Without this, the corpus would still rely on a one-off manual run. Importing
``src.tasks`` runs the load-bearing registration imports in src/tasks/__init__.py;
the assertions then confirm the Celery worker will discover the task and the beat
scheduler will fire it. CI-collectable (no live backends needed for registration).

Honesty note (#1645/#1649): "registered + scheduled" is NOT "runs". Every
assertion in this file passed for months while the task had never once executed —
the entry was a bare ``86400.0`` interval and beat's ``last_run_at`` lived in the
scheduler container's ephemeral /tmp, so a deploy reset the clock before the
interval could ever elapse. The wall-clock assertion below is the part that
cannot silently mean "at deploy time + 24h"; the durability half is guarded by
``tests/unit/test_docker/test_compose_beat_state_volume_1645.py``.
"""

from celery.schedules import crontab

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
    assert entry["schedule"] == crontab(hour=4, minute=0)  # daily 04:00 UTC (#1645)
    assert entry["options"]["queue"] == "analytics"
