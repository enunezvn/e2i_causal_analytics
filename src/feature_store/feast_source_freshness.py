"""Feast freshness of the views SOURCED FROM a training run's data source (#2207 follow-up).

Why this module exists (owner decision 2026-09-22, PR #2223 blocker 2): the data-prep
QC gate in ``feast_registrar`` used to probe a ``feature_analyzer_<experiment_id>``
view — a name in no Feast registry and in no source map — so its recency was always
``None``, every run was "unverifiable", and the gate hard-blocked training on the
worker image (which cannot ``import feast``, #307) *and* on any box with feast
installed. It was unpassable by construction.

What the gate is for (the 2026-06-03 investigation's framing, kept): Feast freshness is a
feature-SERVING concern. A training run reads a Supabase table, a file bundle or the
synthetic sample (``data_loader``), never the Feast online store; the only thing Feast
staleness can say about such a run is whether the ONLINE copies of the features derived
from the same source table are behind the table the model is trained on. That is worth
recording — and it is what this probe measures — but it is a property of serving, not of
the training data, so it is advisory unless the run really trains on Feast-served
features (``features_served_by_feast`` in the data_preparer state).

What "fresh" means here (codex r4 HIGH-1, stated so nobody reads more into it): the
#559 signal is the SOURCE TABLE's recency (``MAX(<raw timestamp column>)``), i.e. "is
the data the online views are built from current" — not whether the online store has
been materialized since. Materialization state is a separate signal, recorded per run in
``ml_feast_materialization_jobs`` (#2207); a Feast-read training path that wants both
must also consult that table. Today no training path reads Feast.

How: the run's ``data_source`` is a table name (or a file dict); the inverse of
``FEAST_FEATURE_VIEW_SOURCE_TABLES`` names the views sourced from it; the #559 recency
probe (``MAX(<raw timestamp column>)`` over PostgREST, no feast import) gives the table's
recency; age vs ``max_staleness_hours`` classifies every one of those views. No recency
signal is "unverifiable", which is not fresh (#556) — never a fabricated ``now()``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.feature_store.feast_views import feast_views_for_source_table

logger = logging.getLogger(__name__)

RecencyQuery = Callable[[str], Awaitable[Optional[datetime]]]


def source_table_of(data_source: Any) -> Optional[str]:
    """The Supabase table a run loads from, or ``None`` for file / absent sources."""
    if isinstance(data_source, str) and data_source.strip():
        return data_source.strip()
    return None


def describe_source(data_source: Any) -> str:
    """``table`` | ``file_dir`` | ``files`` | ``dict`` | ``none`` — for log lines / results."""
    if isinstance(data_source, dict):
        kind = data_source.get("type")
        return str(kind) if kind in ("file_dir", "files") else "dict"
    if source_table_of(data_source):
        return "table"
    return "none"


async def default_source_recency(table: str) -> Optional[datetime]:
    """The #559 recency of ``table``: ``MAX(<raw timestamp column>)`` over PostgREST.

    Resolves the shared sync Supabase client exactly as the freshness adapter's prod
    path does; ``None`` when Supabase is unavailable or the table has no mapped
    timestamp column (recency genuinely unknown → unverifiable downstream).
    """
    from src.api.dependencies.supabase_client import get_supabase
    from src.feature_store.feast_client import FeastClient

    client = get_supabase()
    if client is None:
        logger.warning("Feast source freshness: no Supabase client; recency of %s unknown", table)
        return None
    # ``_query_max_recency`` is the one implementation of the #559 query (same package);
    # a bare FeastClient needs no initialize() and imports no feast for it.
    return await FeastClient()._query_max_recency(client, table)


async def probe_source_freshness(
    data_source: Any,
    max_staleness_hours: float = 24.0,
    *,
    now: Optional[datetime] = None,
    recency_query: Optional[RecencyQuery] = None,
) -> Dict[str, Any]:
    """Freshness of the Feast views sourced from ``data_source``.

    Returns a dict with ``fresh`` (``True`` all mapped views within threshold, ``False``
    stale or unverifiable, ``None`` when no Feast view is sourced from this data source —
    freshness not applicable), ``feast_backed``, ``source_kind``, ``source_table``,
    ``feature_views``, ``last_updated`` (ISO or None), ``age_hours``, ``stale_features``,
    ``feature_ages`` (per view), ``recommendations``, ``max_staleness_hours``,
    ``checked_at``.
    """
    now = now or datetime.now(timezone.utc)
    kind = describe_source(data_source)
    table = source_table_of(data_source)
    views: List[str] = feast_views_for_source_table(table)
    result: Dict[str, Any] = {
        "fresh": None,
        "feast_backed": bool(views),
        "source_kind": kind,
        "source_table": table,
        "feature_views": views,
        "last_updated": None,
        "age_hours": None,
        "stale_features": [],
        "feature_ages": {},
        "recommendations": [],
        "max_staleness_hours": float(max_staleness_hours),
        "checked_at": now.isoformat(),
    }

    if not views:
        what = table if table else f"a {kind} source"
        result["recommendations"].append(
            f"No Feast feature view is sourced from {what}: the run's features are "
            "not Feast-backed, so Feast freshness does not apply to them (advisory)."
        )
        return result

    query = recency_query or default_source_recency
    recency = await query(table)  # type: ignore[arg-type]  # views non-empty => table is str
    if recency is None:
        result["fresh"] = False
        result["stale_features"] = list(views)
        result["feature_ages"] = dict.fromkeys(views)
        result["recommendations"].append(
            f"Recency of {table} is unverifiable (no MAX({table}) signal); the views "
            f"sourced from it ({', '.join(views)}) cannot be verified fresh (#556)."
        )
        return result

    if recency.tzinfo is None:
        recency = recency.replace(tzinfo=timezone.utc)
    age_hours = (now - recency).total_seconds() / 3600.0
    result["last_updated"] = recency.isoformat()
    result["age_hours"] = age_hours
    result["feature_ages"] = dict.fromkeys(views, age_hours)
    if age_hours > float(max_staleness_hours):
        result["fresh"] = False
        result["stale_features"] = list(views)
        result["recommendations"].append(
            f"Source table {table} is stale for Feast serving: newest row is "
            f"{age_hours:.1f} h old (max {float(max_staleness_hours):.1f} h); views sourced "
            f"from it: {', '.join(views)}. Run materialization once the table is refreshed."
        )
    else:
        result["fresh"] = True
    return result


__all__ = [
    "default_source_recency",
    "describe_source",
    "probe_source_freshness",
    "source_table_of",
]
