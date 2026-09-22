"""Feast tracking-table recording for the scheduled Feast tasks (#2207).

The three tracking tables (``ml_feast_feature_views``, ``ml_feast_materialization_jobs``,
``ml_feast_feature_freshness``; migration ml/025, b6ca3f308) had repositories but no
producer: the beat tasks in :mod:`src.tasks.feast_tasks` never referenced them, so the
tables sat at 0 rows while the tasks ran every 6 h / 4 h. These helpers record the
REAL outcome of each scheduled run — including the failure the worker image produces
today (``Failed to initialize Feast client``: the app/worker image cannot
``import feast``, #307; the e2i_feast_materializer sidecar owns the actual
materialize, #556). A ``failed`` job row per run is the truthful record of that; a
freshness run that could not verify a view records it as ``unknown`` (#556:
unverifiable is not fresh).

Contract: recording is best-effort and NEVER raises into the parent task — a broken
tracking backend must not turn a materialize/freshness beat into a failed task.

The repositories drive the SYNC supabase client (``.execute()`` without ``await``),
so the helpers take the client from ``get_supabase_client``, not the async factory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from src.feature_store.feast_views import FEAST_FEATURE_VIEW_SOURCE_TABLES
from src.repositories.feast_tracking import (
    FeastFeatureViewRepository,
    FeastFreshnessRepository,
    FeastMaterializationRepository,
    MaterializationStatus,
)

logger = logging.getLogger(__name__)

FEAST_PROJECT = "e2i_causal_analytics"
_REGISTERED_BY = "src.tasks.feast_tasks"


def targeted_feature_views(feature_views: Optional[List[str]]) -> List[str]:
    """The views a run targeted: the explicit list, else every real Feast view."""
    return list(feature_views) if feature_views else list(FEAST_FEATURE_VIEW_SOURCE_TABLES)


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


async def ensure_feature_view_rows(client: Any, names: List[str]) -> Dict[str, Optional[UUID]]:
    """Return ``{name: registry row id}`` for each view, creating missing registry rows.

    The registry row (``ml_feast_feature_views``) is the FK target of the job and
    freshness rows. A view whose row cannot be created still gets its job/freshness
    rows (the FK columns are nullable) — the id is just ``None``.
    """
    repo = FeastFeatureViewRepository(client)
    ids: Dict[str, Optional[UUID]] = {}
    for name in names:
        try:
            existing = await repo.get_by_name(name, project=FEAST_PROJECT)
            if existing is None:
                existing = await repo.create_feature_view(
                    name=name,
                    project=FEAST_PROJECT,
                    source_type="batch",
                    source_name=FEAST_FEATURE_VIEW_SOURCE_TABLES.get(name),
                    tags={"registered_by": _REGISTERED_BY},
                )
            ids[name] = existing.id if existing is not None else None
        except Exception as e:  # noqa: BLE001 — best-effort registry
            logger.warning("Feast tracking: could not ensure registry row for %s: %s", name, e)
            ids[name] = None
    return ids


def _job_status(result: Dict[str, Any]) -> tuple[MaterializationStatus, Optional[str]]:
    status = result.get("status")
    if status == "completed":
        return MaterializationStatus.SUCCESS, None
    if status == "skipped":
        reason = result.get("reason") or "feast unavailable in this runtime"
        return MaterializationStatus.FAILED, f"skipped: {reason}"
    return MaterializationStatus.FAILED, str(result.get("error") or f"status={status!r}")


async def record_materialization_jobs(
    client: Any,
    *,
    job_type: str,
    requested_start: Optional[datetime],
    requested_end: datetime,
    feature_views: Optional[List[str]],
    result: Dict[str, Any],
    task_id: Optional[str] = None,
) -> int:
    """One ``ml_feast_materialization_jobs`` row per targeted view, carrying the run's
    real status. Returns the number of rows written."""
    views = targeted_feature_views(feature_views)
    ids = await ensure_feature_view_rows(client, views)
    status, error_message = _job_status(result)
    start_time = _parse_ts(result.get("start_date")) or requested_start or requested_end
    end_time = _parse_ts(result.get("end_date")) or requested_end
    duration = result.get("duration_seconds")
    # rows_materialized is a per-run total from FeastClient.materialize; it is only a
    # per-view number when the run targeted exactly one view.
    rows = result.get("rows_materialized") if len(views) == 1 else None

    repo = FeastMaterializationRepository(client)
    written = 0
    for name in views:
        # One atomic insert carrying the terminal status (codex r3): a create +
        # update pair could leave a `pending` row when the close-out failed, and
        # counting it as recorded would be a lie. `written` is rows that landed.
        job = await repo.create_job(
            feature_view_id=ids.get(name),  # type: ignore[arg-type]
            feature_view_name=name,
            start_time=start_time,
            end_time=end_time,
            job_type=job_type,
            job_id=task_id,
            status=status.value,
            error_message=error_message,
            rows_materialized=rows,
            duration_seconds=float(duration) if duration is not None else None,
        )
        if job is None or job.id is None:
            logger.warning("Feast tracking: job row for %s (%s) did not land", name, job_type)
            continue
        written += 1
    return written


async def record_freshness_checks(
    client: Any,
    *,
    result: Dict[str, Any],
    feature_views: Optional[List[str]],
    max_staleness_hours: float,
) -> int:
    """One ``ml_feast_feature_freshness`` row per view the run reported on.

    ``fresh_features`` / ``stale_features`` carry a real recency; ``errors`` (a view the
    probe could not verify) and a run that failed before probing anything record
    ``unknown`` for every view they cover. Returns the number of rows written.
    """
    threshold = int(float(max_staleness_hours) * 3600)
    entries: List[tuple[str, str, Optional[datetime], Optional[int]]] = []
    if result.get("status") == "completed":
        for item in result.get("fresh_features", []):
            age = item.get("age_hours")
            entries.append(
                (
                    item["feature_view"],
                    "fresh",
                    _parse_ts(item.get("last_updated")),
                    int(float(age) * 3600) if age is not None else None,
                )
            )
        for item in result.get("stale_features", []):
            age = item.get("age_hours")
            entries.append(
                (
                    item["feature_view"],
                    "stale",
                    _parse_ts(item.get("last_updated")),
                    int(float(age) * 3600) if age is not None else None,
                )
            )
        for item in result.get("errors", []):
            entries.append((item["feature_view"], "unknown", None, None))
    else:
        # The probe never ran (e.g. "Failed to initialize Feast client" on the worker
        # image): every targeted view is unverifiable this run.
        for name in targeted_feature_views(feature_views):
            entries.append((name, "unknown", None, None))

    ids = await ensure_feature_view_rows(client, [name for name, *_ in entries])
    repo = FeastFreshnessRepository(client)
    written = 0
    for name, status, last_materialization, staleness in entries:
        row = await repo.record_freshness(
            feature_view_id=ids.get(name),  # type: ignore[arg-type]
            feature_view_name=name,
            last_materialization_time=last_materialization,
            staleness_seconds=staleness,
            freshness_status=status,
            staleness_threshold_seconds=threshold,
        )
        if row is not None:
            written += 1
    return written


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
