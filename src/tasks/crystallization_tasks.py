"""Celery tasks for the crystallization subsystem (#376 Phase 4).

- ``crystallize_portfolio``  : every 6h on the ``analytics`` queue.
                               Iterates the configured brand list and
                               aggregates per-brand counts.

Schedule semantics (codex iter-1 M3 honest-doc):
The beat entry in ``src/workers/celery_app.py`` uses the
relative-interval form (``schedule: 21600.0``), which runs every 6
hours measured from the beat scheduler start, NOT from a fixed
wall-clock time. The implementation does NOT enforce any offset
relative to the daily ``insight-lifecycle-consolidate`` task; the
two run independently. Plan §Phase 4 line 141's "30 min after
consolidation" framing was a planning-level suggestion, not a
load-bearing operational contract.

Migration path to a strict wall-clock offset (codex iter-2 M3
symmetric-doc with celery_app.py:350-368): if CI / production
observability surfaces a real race between the consolidator and
this task, swap the relative-interval form for
``celery.schedules.crontab(hour='*/6', minute=30)`` in the beat
entry. The crontab form is supported out-of-the-box by Celery beat
and adjacent modules in this repo already import ``crontab`` —
no new dependency. Absent a demonstrated need, the relative form
is simpler, idempotent, and avoids cron-style timezone surprises.

The task wraps :meth:`src.memory.crystallization.crystallizer.Crystallizer.crystallize_portfolio`
and returns a JSON-serializable summary so beat-logs / dispatcher
audits can inspect the per-run result without re-querying the DB.

Idempotency: re-running within the schedule window collides on the
partial-unique-index ``uix_executive_insights_active_causal_path``
(see ``database/memory/021_insight_lifecycle.sql:219-226``); the
crystallizer treats the collision as a skip-signal and continues.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_crystallize_portfolio() -> Any:
    """Synchronous bridge wrapping
    :meth:`Crystallizer.crystallize_portfolio` so the Celery task body
    stays asyncio-import-free.

    Extracted as a module-level callable so unit tests can patch it
    without monkeypatching the whole crystallizer.
    """
    from src.memory.crystallization.crystallizer import Crystallizer

    return asyncio.run(Crystallizer().crystallize_portfolio())


@celery_app.task(
    bind=True,
    name="src.tasks.crystallization_tasks.crystallize_portfolio",
)
def crystallize_portfolio(self) -> Dict[str, Any]:
    """6-hourly portfolio-crystallization pass.

    Returns a JSON-serializable summary. Exceptions propagate so the
    Celery autoretry / DLQ pipeline kicks in.
    """
    try:
        result = _run_crystallize_portfolio()
        return {
            "examined_groups": int(getattr(result, "examined_groups", 0) or 0),
            "insights_created": int(getattr(result, "insights_created", 0) or 0),
            "edges_created": int(getattr(result, "edges_created", 0) or 0),
            "by_brand": dict(getattr(result, "by_brand", {}) or {}),
            "errors": list(getattr(result, "errors", []) or []),
        }
    except Exception:
        logger.exception("crystallize_portfolio task failed")
        raise
