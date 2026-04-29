"""Unit tests for ``src.etl._common``.

These pin the canonical behaviour of the three helpers extracted from
``business_metrics_per_hcp_etl`` (6B-infra-2a) and ``patient_adherence_etl``
(6B-infra-2b):

* :func:`_resolve_db_connection_string` — env-driven Postgres DSN, friendly
  RuntimeError on missing/empty input.
* :func:`_connect_to_db` — tenacity-wrapped psycopg2.connect; we cover the
  delegation to :func:`_resolve_db_connection_string` here without standing
  up a real Postgres.
* :func:`_resolve_window` — ISO-string -> UTC-aware (start, end) tuple, with
  half-open ``[start, end)`` semantics, naive-input normalisation to UTC,
  and friendly ``ValueError`` on inverted windows.

Each per-ETL test file used to copy these tests verbatim. Post-extraction,
the canonical assertions live here; the per-ETL files keep only the tests
that prove a re-export is wired correctly (a single import-and-call check)
and the SQL-shape / orchestration tests that are actually per-ETL specific.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.etl import _common

# =============================================================================
# _resolve_db_connection_string
# =============================================================================


def test_resolve_db_connection_string_returns_env_value() -> None:
    """Returns the SUPABASE_DB_URL value verbatim when present."""
    fake_url = "postgresql://u:p@h:5432/db"
    with patch.dict(os.environ, {"SUPABASE_DB_URL": fake_url}, clear=False):
        assert _common._resolve_db_connection_string() == fake_url


def test_resolve_db_connection_string_raises_when_missing() -> None:
    """Raises RuntimeError when SUPABASE_DB_URL is unset."""
    env = {k: v for k, v in os.environ.items() if k != "SUPABASE_DB_URL"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="SUPABASE_DB_URL"):
            _common._resolve_db_connection_string()


def test_resolve_db_connection_string_raises_when_empty() -> None:
    """Raises RuntimeError when SUPABASE_DB_URL is the empty string.

    ``os.getenv`` returns ``""`` for an unset-looking env var that's
    actually present and empty; ``not ""`` is True so the helper rejects
    both states with the same message.
    """
    with patch.dict(os.environ, {"SUPABASE_DB_URL": ""}, clear=False):
        with pytest.raises(RuntimeError, match="SUPABASE_DB_URL"):
            _common._resolve_db_connection_string()


# =============================================================================
# _connect_to_db
# =============================================================================


def test_connect_to_db_uses_env_dsn_by_default() -> None:
    """Without an explicit DSN, the helper reads SUPABASE_DB_URL via
    :func:`_resolve_db_connection_string` and passes the result to
    ``psycopg2.connect``."""
    fake_url = "postgresql://u:p@h:5432/db"
    sentinel_conn = object()
    with patch.dict(os.environ, {"SUPABASE_DB_URL": fake_url}, clear=False):
        with patch.object(_common.psycopg2, "connect", return_value=sentinel_conn) as connect:
            result = _common._connect_to_db()

    assert result is sentinel_conn
    connect.assert_called_once_with(fake_url)


def test_connect_to_db_uses_explicit_dsn_when_supplied() -> None:
    """An explicit ``connection_string`` skips the env lookup entirely."""
    sentinel_conn = object()
    explicit = "postgresql://explicit:dsn@host:5432/db"
    with patch.object(_common.psycopg2, "connect", return_value=sentinel_conn) as connect:
        result = _common._connect_to_db(connection_string=explicit)

    assert result is sentinel_conn
    connect.assert_called_once_with(explicit)


# =============================================================================
# _resolve_window
# =============================================================================


def test_resolve_window_defaults_to_24h_ending_now() -> None:
    """When both ends are None, end=now(UTC), start=end - 24h (default)."""
    before = datetime.now(timezone.utc)
    start, end = _common._resolve_window(None, None)
    after = datetime.now(timezone.utc)

    assert before <= end <= after
    delta_seconds = (end - start).total_seconds()
    assert delta_seconds == pytest.approx(_common.DEFAULT_LOOKBACK_SECONDS, abs=1e-3)


def test_resolve_window_honours_custom_lookback_seconds() -> None:
    """A non-default lookback (e.g. 7 days) shifts start_dt accordingly."""
    seven_days = 7 * 86400
    start, end = _common._resolve_window(None, None, default_lookback_seconds=seven_days)
    assert (end - start).total_seconds() == pytest.approx(seven_days, abs=1e-3)


def test_resolve_window_parses_iso_z_suffix() -> None:
    """Trailing 'Z' is normalised to +00:00 before parsing."""
    start, end = _common._resolve_window("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
    assert start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert end == datetime(2024, 1, 2, tzinfo=timezone.utc)


def test_resolve_window_parses_iso_date_only() -> None:
    """Plain ISO dates parse cleanly and are normalised to UTC-aware."""
    start, end = _common._resolve_window("2024-01-01", "2024-01-02")
    assert start.year == 2024 and start.day == 1
    assert end.year == 2024 and end.day == 2
    assert start.tzinfo is not None
    assert end.tzinfo is not None


def test_resolve_window_normalises_naive_inputs_to_utc() -> None:
    """Mixing ISO date (naive) + ISO datetime+tz (aware) must NOT TypeError.

    Before the I3 fix-up in 6B-infra-2a, ``datetime.fromisoformat("2024-01-01")``
    returned a naive datetime which then collided with the aware ``end_dt``
    in the ``start_dt >= end_dt`` comparison, raising ``TypeError`` instead
    of the friendlier ``ValueError``.
    """
    start, end = _common._resolve_window("2024-01-01", "2024-01-02T00:00:00+00:00")
    assert start.tzinfo is not None
    assert end.tzinfo is not None
    assert start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert end == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert (end - start).total_seconds() == 86400


def test_resolve_window_rejects_inverted_range() -> None:
    """start_date >= end_date triggers ValueError."""
    with pytest.raises(ValueError, match="must be strictly before"):
        _common._resolve_window("2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z")


def test_resolve_window_rejects_zero_length_window() -> None:
    """Equal start and end is also invalid (need at least one second)."""
    with pytest.raises(ValueError, match="must be strictly before"):
        _common._resolve_window("2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z")
