"""Offline guard: the chat-RAG chunk-corpus sync task is registered + scheduled (#1373).

Mirrors ``test_corpus_sync_task_registered`` for the episodic corpus. Without a
beat entry the chunk corpus would rely on a one-off manual run and could never
pick up new business_metrics facts. Importing ``src.tasks`` runs the
registration imports in src/tasks/__init__.py; the assertions confirm the worker
discovers the task and beat fires it. CI-collectable (no live backends).

Honesty note (#1645): "registered + scheduled" is NOT "runs" — see the sibling
episodic-corpus guard's docstring for the failure this file also passed through.
"""

from celery.schedules import crontab

import src.tasks  # noqa: F401  (triggers task registration via __init__)
from src.workers.celery_app import celery_app

TASK_NAME = "src.tasks.sync_chunk_corpus"
BEAT_KEY = "sync-chunk-corpus"


def test_task_is_registered_with_celery():
    assert TASK_NAME in celery_app.tasks, (
        f"{TASK_NAME} not registered; the worker will not discover it"
    )
    assert src.tasks.sync_chunk_corpus.name == TASK_NAME


def test_beat_schedule_fires_the_sync_daily():
    bs = celery_app.conf.beat_schedule
    assert BEAT_KEY in bs, "no beat entry -> chunk corpus never re-syncs (regresses to one-off)"
    entry = bs[BEAT_KEY]
    assert entry["task"] == TASK_NAME
    assert entry["schedule"] == crontab(hour=4, minute=15)  # daily 04:15 UTC (#1645)
    assert entry["options"]["queue"] == "analytics"
