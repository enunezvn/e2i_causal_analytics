"""Shared helpers for ETL modules in this package.

Three small utilities live here, extracted from
``business_metrics_per_hcp_etl`` (block 6B-infra-2a) and
``patient_adherence_etl`` (block 6B-infra-2b). 6B-infra-2c will land a third
ETL imminently and import the same helpers from the start, so the natural
moment to extract was now (before a third copy could drift).

The three helpers
-----------------

* :func:`_resolve_db_connection_string` — read ``SUPABASE_DB_URL`` from env;
  raise ``RuntimeError`` with a friendly message if unset/empty. The Celery
  task wrappers in each ETL surface that error through their existing
  ``task_failure`` handler.
* :func:`_connect_to_db` — open a psycopg2 connection wrapped in tenacity-
  backed retry. Three attempts with 1-10s exponential backoff. The retry
  classes are ``(psycopg2.OperationalError, ConnectionError, TimeoutError,
  OSError)``; ``psycopg2.OperationalError`` is listed explicitly because its
  MRO does NOT inherit from the stdlib ``ConnectionError`` despite the name
  (``OperationalError -> DatabaseError -> Error -> Exception``), so without
  the explicit type the retry would silently miss the canonical Postgres
  connect-time failure modes.
* :func:`_resolve_window` — turn ISO-format date / datetime strings into a
  ``(start, end)`` tuple of UTC-aware datetimes, with half-open semantics
  (``[start, end)``). Naive datetimes from date-only ISO inputs are
  normalised to UTC before any comparison so an aware/naive ``TypeError``
  cannot mask the friendlier ``ValueError`` raised when ``start >= end``.

Logging convention
------------------
Each helper uses ``logger = logging.getLogger(__name__)`` so the module path
appears in log lines (``src.etl._common``). The behaviour is identical to
what the per-ETL copies did before extraction; tests pin no log content.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2
from tenacity import (
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

#: Default lookback in seconds for :func:`_resolve_window` when no
#: ``start_date`` is supplied. 86400 = 24h. Each ETL pins its own
#: ``DEFAULT_WINDOW_HOURS`` constant for readability and overrides this on
#: call.
DEFAULT_LOOKBACK_SECONDS: int = 86400


# -----------------------------------------------------------------------------
# DB connection
# -----------------------------------------------------------------------------


def _resolve_db_connection_string() -> str:
    """Read the Supabase Postgres URL from env.

    Raises:
        RuntimeError: if ``SUPABASE_DB_URL`` is missing or empty. Each ETL's
            Celery task wrapper surfaces this via the existing
            ``task_failure`` handler in ``celery_app``.
    """
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL environment variable is required for ETL jobs")
    return db_url


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    # psycopg2.OperationalError covers connection-refused / server-starting /
    # SSL-handshake failures; the rest covers raw socket failures. NB:
    # psycopg2.OperationalError does NOT inherit from ConnectionError despite
    # the name -- its MRO is OperationalError -> DatabaseError -> Error ->
    # Exception. So adding it explicitly is what makes this retry actually
    # cover the canonical Postgres connect-time failure modes.
    retry=retry_if_exception_type(
        (psycopg2.OperationalError, ConnectionError, TimeoutError, OSError)
    ),
    before=before_log(logger, logging.WARNING),
    reraise=True,
)
def _connect_to_db(connection_string: Optional[str] = None) -> Any:
    """Open a psycopg2 connection with tenacity-backed retry.

    Three attempts with 1-10 second exponential backoff. ``psycopg2`` is
    imported at module level since it is already a hard dependency via the
    Supabase client.

    Args:
        connection_string: Explicit DSN. If ``None`` (the default), reads
            ``SUPABASE_DB_URL`` via :func:`_resolve_db_connection_string`.

    Returns:
        A live ``psycopg2.connection``.
    """
    dsn = connection_string if connection_string is not None else _resolve_db_connection_string()
    return psycopg2.connect(dsn)


# -----------------------------------------------------------------------------
# Window resolution
# -----------------------------------------------------------------------------


def _resolve_window(
    start_date: Optional[str],
    end_date: Optional[str],
    default_lookback_seconds: int = DEFAULT_LOOKBACK_SECONDS,
) -> tuple[datetime, datetime]:
    """Resolve ISO date strings to a ``(start, end)`` UTC datetime tuple.

    ``end_date`` defaults to ``now(UTC)``; ``start_date`` defaults to
    ``end_date - default_lookback_seconds``. Strings can be ISO datetime
    (``2024-01-01T00:00:00Z``) or ISO date (``2024-01-01``) — both are
    accepted via :meth:`datetime.fromisoformat`.

    Naive results from ``datetime.fromisoformat`` (date-only inputs return a
    tz-naive datetime at midnight) are normalised to UTC-aware so downstream
    aware/naive comparisons cannot raise ``TypeError``. Without that step a
    mixed call such as ``_resolve_window("2024-01-01",
    "2024-01-02T00:00:00+00:00")`` would raise ``TypeError`` from the
    aware/naive comparison and mask the friendlier ``ValueError`` below.

    Args:
        start_date: ISO datetime / ISO date for the window start.
            ``None`` defaults to ``end - default_lookback_seconds``.
        end_date: ISO datetime / ISO date for the window end (exclusive).
            ``None`` defaults to ``now(UTC)``.
        default_lookback_seconds: Lookback applied when ``start_date`` is
            ``None``. Defaults to 24h.

    Returns:
        ``(start, end)`` tuple of UTC-aware datetimes, ``[start, end)``
        half-open.

    Raises:
        ValueError: if ``start >= end``.
    """
    now_utc = datetime.now(timezone.utc)

    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00")) if end_date else now_utc
    start_dt = (
        datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if start_date
        else end_dt - timedelta(seconds=default_lookback_seconds)
    )

    # Normalise naive datetimes (date-only ISO strings) to UTC so
    # downstream comparisons + the SQL bind params are consistently aware.
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    if start_dt >= end_dt:
        raise ValueError(
            f"start_date ({start_dt.isoformat()}) must be strictly before "
            f"end_date ({end_dt.isoformat()})"
        )

    return start_dt, end_dt
