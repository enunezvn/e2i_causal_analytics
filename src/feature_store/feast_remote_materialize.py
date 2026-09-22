"""Remote-mode Feast materialization through the e2i_feast sidecar's HTTP API (#2207).

The app/worker image cannot ``import feast`` (#307: feast pins tenacity<9, prod runs
tenacity 9.1.2), so ``FeastClient`` runs in REMOTE mode there (``FEAST_URL``, #532) and
had no materialize path: the 6 h / weekly beats returned ``Failed to initialize Feast
client`` on every run (measured 2026-09-22). The feast 0.43 feature server the sidecar
runs already exposes ``POST /materialize`` and ``POST /materialize-incremental`` — the
same calls the sidecar's own shell loop makes — and worker_medium reaches it by service
name. This module mirrors the sidecar's request schema exactly (read from its
``/openapi.json`` inside worker_medium, 2026-09-22):

    MaterializeRequest            {start_ts: str, end_ts: str, feature_views?: [str] | null}
    MaterializeIncrementalRequest {end_ts: str, feature_views?: [str] | null}

Timestamps are ISO-8601 (the server parses them with dateutil and makes them tz-aware);
``feature_views: null`` means every ONLINE view in the sidecar's registry — an explicit
list naming a non-online view (``market_dynamics_features``, #556) makes the server 422.
The response body is empty; a 2xx is the success signal. Failures (non-2xx, unreachable)
are returned as ``{"status": "failed", "error": ...}`` so the beat records the truthful
outcome and then fails loud (``src/tasks/feast_tasks.py``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# The materialize job's init-failure marker (``scripts/feast_materialize.py``) — shared so
# the beat can tell "could not even build a client" from a failed sidecar call.
INIT_FAILURE_ERROR = "Failed to initialize Feast client"


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


async def _post(base_url: str, path: str, payload: Dict[str, Any], timeout: float) -> Optional[str]:
    """POST ``payload``; ``None`` on 2xx, else an error string (never raises)."""
    url = f"{base_url.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 300:
                body = (response.text or "")[:300]
                return f"HTTP {response.status_code} from {url}: {body}"
            return None
    except httpx.HTTPError as exc:
        return f"request to {url} failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — transport-level surprises are failures too
        return f"request to {url} failed: {type(exc).__name__}: {exc}"


async def post_materialize(
    base_url: str,
    *,
    start_date: datetime,
    end_date: datetime,
    feature_views: Optional[List[str]],
    timeout: float,
    default_feature_views: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """``POST /materialize`` for ``[start_date, end_date]``; see the module docstring."""
    started = datetime.now(timezone.utc)
    payload = {
        "start_ts": _iso(start_date),
        "end_ts": _iso(end_date),
        "feature_views": list(feature_views) if feature_views else None,
    }
    error = await _post(base_url, "/materialize", payload, timeout)
    if error:
        logger.error("Remote Feast materialize failed: %s", error)
        return {"status": "failed", "error": error, "feature_views": feature_views or ["all"]}
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    views = list(feature_views) if feature_views else list(default_feature_views or [])
    logger.info("Remote Feast materialize completed in %.2fs: %s", duration, views or "all")
    return {
        "status": "completed",
        "feature_views": views,
        "start_date": payload["start_ts"],
        "end_date": payload["end_ts"],
        "duration_seconds": duration,
        "materializer": f"remote:{base_url.rstrip('/')}",
    }


async def post_materialize_incremental(
    base_url: str,
    *,
    end_date: datetime,
    feature_views: Optional[List[str]],
    timeout: float,
    default_feature_views: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """``POST /materialize-incremental`` up to ``end_date``; see the module docstring."""
    started = datetime.now(timezone.utc)
    payload = {
        "end_ts": _iso(end_date),
        "feature_views": list(feature_views) if feature_views else None,
    }
    error = await _post(base_url, "/materialize-incremental", payload, timeout)
    if error:
        logger.error("Remote Feast incremental materialize failed: %s", error)
        return {"status": "failed", "error": error}
    duration = (datetime.now(timezone.utc) - started).total_seconds()
    views = list(feature_views) if feature_views else list(default_feature_views or [])
    logger.info(
        "Remote Feast incremental materialize completed in %.2fs: %s", duration, views or "all"
    )
    return {
        "status": "completed",
        "feature_views": views,
        "end_date": payload["end_ts"],
        "duration_seconds": duration,
        "incremental": True,
        "materializer": f"remote:{base_url.rstrip('/')}",
    }


__all__ = ["INIT_FAILURE_ERROR", "post_materialize", "post_materialize_incremental"]
