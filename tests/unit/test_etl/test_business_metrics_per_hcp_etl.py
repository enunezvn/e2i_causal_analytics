"""Unit tests for ``src.etl.business_metrics_per_hcp_etl``.

These tests do not touch a real database. They verify:

* The SQL string contains the load-bearing CTEs, parameter placeholders,
  the deterministic ``metric_id`` shape, the ``ON CONFLICT (metric_id)``
  clause, and the brand-via-LATERAL pattern.
* ``_run_per_hcp_rollup_impl`` orchestrates the connect/execute/commit/close
  flow correctly with mocks, and surfaces ``status`` / ``rows_affected``
  faithfully. The Celery wrapper ``run_per_hcp_rollup`` is a thin
  one-liner over this and is exercised in the integration test.
* The shared helpers ``_resolve_db_connection_string``, ``_connect_to_db``
  and ``_resolve_window`` are re-exported here for backward compatibility
  (the canonical tests live in ``test_common.py`` since extraction in
  6B-infra-2b fix-up).

Behaviour-level assertions about market_share summing to 1.0 within a
territory live in the integration test (which spins up real synthetic
data); see ``tests/integration/test_business_metrics_per_hcp_etl_integration.py``.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

# Importing the module triggers `from src.workers.celery_app import celery_app`,
# which only requires standard library + celery (already installed). No DB
# connection happens at import time.
from src.etl import _common
from src.etl import business_metrics_per_hcp_etl as etl

# =============================================================================
# _common helper re-exports
# =============================================================================


def test_helpers_are_re_exported_from_common() -> None:
    """The three shared helpers must be accessible as module attributes on
    ``business_metrics_per_hcp_etl`` so existing test imports keep working.

    The detailed behaviour tests live in ``test_common.py``; this guards the
    re-export wiring so a future cleanup that drops the shim breaks loudly.
    """
    assert etl._resolve_db_connection_string is _common._resolve_db_connection_string
    assert etl._connect_to_db is _common._connect_to_db
    assert etl._resolve_window is _common._resolve_window


# =============================================================================
# SQL string structure
# =============================================================================


class TestSQLShape:
    """Pin the load-bearing structure of INSERT_PER_HCP_ROLLUP_SQL.

    These tests intentionally lean on substring presence rather than full
    SQL parse — a parser would be overkill for the small set of guarantees
    this query needs to keep across refactors.
    """

    def test_has_three_named_ctes(self) -> None:
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "triggers_with_brand AS" in sql
        assert "hcp_brand_daily AS" in sql
        assert "territory_totals AS" in sql

    def test_brand_derived_via_lateral_subquery(self) -> None:
        """Brand comes from patient_journeys via LATERAL, not from
        triggers.brand_id (which is sentinel 'UNKNOWN')."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "JOIN LATERAL" in sql
        assert "patient_journeys" in sql
        # The most-recent-prior journey pattern.
        assert "ORDER BY pj_inner.journey_start_date DESC" in sql
        assert "LIMIT 1" in sql

    def test_uses_named_parameters(self) -> None:
        """psycopg2 named-param style %()s is required so the params dict
        in run_per_hcp_rollup matches."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "%(start_date)s" in sql
        assert "%(end_date)s" in sql
        assert "%(metric_id_prefix)s" in sql
        assert "%(metric_type)s" in sql

    def test_sql_metric_id_uses_md5(self) -> None:
        """metric_id is built from the prefix + md5(natural-key).

        The SQL uses ``md5(hcp_id ':' brand ':' metric_date)`` so the
        result fits ``business_metrics.metric_id VARCHAR(50)`` (constant 43
        chars). The component order + separator MUST match
        ``_build_metric_id`` in the source module — this test pins the
        SQL side so the two cannot drift silently.
        """
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "md5(" in sql, "metric_id must hash with md5 to fit VARCHAR(50)"
        # Pin the natural-key component order: hcp_id : brand : metric_date.
        # Whitespace in the SQL is normalised before matching so newlines
        # / indentation don't break the assertion.
        normalised = re.sub(r"\s+", " ", sql)
        assert (
            "hbd.hcp_id || ':' || hbd.brand::TEXT || ':' || hbd.metric_date::TEXT"
        ) in normalised, "natural-key component order must match _build_metric_id"

    def test_on_conflict_uses_pk(self) -> None:
        """ON CONFLICT targets metric_id (the PK), not a (hcp_id, brand,
        metric_date) tuple — see module docstring for the reasoning."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "ON CONFLICT (metric_id) DO UPDATE SET" in sql

    def test_on_conflict_updates_volatile_metrics(self) -> None:
        """Idempotent re-run with new counts must overwrite the metric
        columns; static columns like metric_date stay put."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        # Slice from the ON CONFLICT clause forward so we only assert on
        # the SET list (and don't trip on column names that also appear in
        # the SELECT).
        on_conflict_block = sql.split("ON CONFLICT (metric_id) DO UPDATE SET", 1)[1]
        for col in (
            "trx_count",
            "nrx_count",
            "total_rx_count",
            "market_share",
            "conversion_rate",
        ):
            # Allow any whitespace between the column name and "=".
            pattern = rf"{re.escape(col)}\s*=\s*EXCLUDED\.{re.escape(col)}"
            assert re.search(pattern, on_conflict_block), f"missing UPDATE SET clause for {col}"

    def test_market_share_is_share_of_territory_total(self) -> None:
        """market_share = total_rx_count / territory_total, with a >0 guard."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "tt.territory_total > 0" in sql
        assert "hbd.total_rx_count::NUMERIC / tt.territory_total" in sql

    def test_conversion_rate_uses_nullif_guard(self) -> None:
        """Division by zero is guarded via NULLIF; default to 0 via COALESCE."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "NULLIF(COUNT(*) FILTER (WHERE delivery_status = 'delivered'), 0)" in sql
        assert "COALESCE(" in sql

    def test_engagement_score_and_call_frequency_left_null(self) -> None:
        """Plan calls for these columns; canonical schema lacks the source
        table. We document the omission inline so future readers see why.

        We strip line comments before checking, since the explanatory
        comment in the column list mentions both column names.
        """
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        # Strip everything from `--` to end-of-line so comments don't
        # spuriously match column names.
        stripped = re.sub(r"--[^\n]*", "", sql)
        # Now slice the INSERT INTO ... SELECT block.
        insert_clause = stripped.split("INSERT INTO business_metrics", 1)[1].split("SELECT", 1)[0]
        assert "engagement_score" not in insert_clause
        assert "call_frequency" not in insert_clause
        # The original SQL still mentions `interactions` in the comment so
        # future readers see the reasoning.
        assert "interactions" in sql

    def test_filters_to_window(self) -> None:
        """The trigger window filter is the [start, end) half-open interval."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert re.search(r"t\.trigger_timestamp\s*>=\s*%\(start_date\)s", sql), (
            "missing start_date >= filter"
        )
        assert re.search(r"t\.trigger_timestamp\s*<\s*%\(end_date\)s", sql), (
            "missing end_date < filter"
        )


# =============================================================================
# _build_metric_id — pins VARCHAR(50) length property
# =============================================================================


def test_metric_id_fits_in_varchar_50() -> None:
    """metric_id must fit ``business_metrics.metric_id VARCHAR(50)``.

    Worst-case inputs: max-length ``hcp_id`` (VARCHAR(20)), longest
    ``brand_type`` enum value ``Remibrutinib`` (12 chars), and an ISO
    date string (10 chars). With md5 the result is a constant 43 chars.
    """
    long_hcp = "H" * 20  # max VARCHAR(20) per hcp_profiles.hcp_id
    long_brand = "Remibrutinib"  # longest brand_type enum value
    iso_date = date(2030, 12, 31)
    result = etl._build_metric_id(long_hcp, long_brand, iso_date)
    assert len(result) <= 50, f"metric_id length {len(result)} exceeds VARCHAR(50): {result}"
    # Same inputs MUST yield the same id (idempotency contract).
    assert etl._build_metric_id(long_hcp, long_brand, iso_date) == result
    # Constant length 43 = len('hcp_rollup_') + 32-hex md5 digest.
    assert len(result) == 43


def test_metric_id_format_matches_sql_md5() -> None:
    """The Python helper must produce the same byte string as the SQL md5.

    Postgres' ``md5()`` returns the lowercase hex digest of the UTF-8 input
    — ``hashlib.md5(...).hexdigest()`` matches that exactly. We re-implement
    the natural-key concat here to assert the helper does not silently
    diverge.
    """
    hcp_id = "hcp_test_42"
    brand = "Fabhalta"
    metric_date = date(2024, 6, 15)
    natural_key = f"{hcp_id}:{brand}:{metric_date.isoformat()}"
    expected_digest = hashlib.md5(natural_key.encode("utf-8")).hexdigest()
    expected = f"{etl.METRIC_ID_PREFIX}_{expected_digest}"
    assert etl._build_metric_id(hcp_id, brand, metric_date) == expected


def test_metric_id_changes_when_natural_key_changes() -> None:
    """Different natural keys must yield different ids (no collisions on
    distinct inputs).

    md5 is not collision-free in general but for these short, structured
    inputs distinct triples are overwhelmingly distinct digests.
    """
    a = etl._build_metric_id("hcp_1", "Remibrutinib", date(2024, 1, 1))
    b = etl._build_metric_id("hcp_2", "Remibrutinib", date(2024, 1, 1))
    c = etl._build_metric_id("hcp_1", "Fabhalta", date(2024, 1, 1))
    d = etl._build_metric_id("hcp_1", "Remibrutinib", date(2024, 1, 2))
    assert len({a, b, c, d}) == 4


# =============================================================================
# _run_per_hcp_rollup_impl — orchestration
# =============================================================================


def _make_mock_conn(rowcount: int = 7) -> MagicMock:
    """Build a mock psycopg2 connection that exits its `with` block cleanly."""
    cur = MagicMock()
    cur.rowcount = rowcount
    cur.execute = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cur)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.close = MagicMock()
    return conn


def test_impl_completed_path() -> None:
    """Happy path: connect, execute, commit, close; status=completed."""
    conn = _make_mock_conn(rowcount=42)

    with patch.object(etl, "_connect_to_db", return_value=conn) as connect:
        result = etl._run_per_hcp_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "completed"
    assert result["rows_affected"] == 42
    assert result["window_start"].startswith("2024-01-01T00:00:00")
    assert result["window_end"].startswith("2024-01-02T00:00:00")

    connect.assert_called_once()
    # Cursor was used, query parameters bound by name include our prefix.
    cur = conn.cursor.return_value
    args, _ = cur.execute.call_args
    assert args[0] is etl.INSERT_PER_HCP_ROLLUP_SQL
    params = args[1]
    assert params["metric_id_prefix"] == etl.METRIC_ID_PREFIX
    assert params["metric_type"] == etl.METRIC_TYPE
    assert isinstance(params["start_date"], datetime)
    assert isinstance(params["end_date"], datetime)
    conn.close.assert_called_once()


def test_impl_no_data_path() -> None:
    """rowcount==0 reports status=no_data with rows_affected=0."""
    conn = _make_mock_conn(rowcount=0)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "no_data"
    assert result["rows_affected"] == 0


def test_impl_db_failure_returns_failed() -> None:
    """A connection / execute exception is caught and surfaced as failed."""
    with patch.object(etl, "_connect_to_db", side_effect=RuntimeError("boom")):
        result = etl._run_per_hcp_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert result["rows_affected"] == 0


def test_impl_invalid_window_returns_failed() -> None:
    """Inverted window short-circuits before any DB connect."""
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_per_hcp_rollup_impl(
            start_date="2024-01-02T00:00:00Z",
            end_date="2024-01-01T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert "must be strictly before" in result["error"]
    connect.assert_not_called()


def test_impl_default_window_is_24h() -> None:
    """No dates supplied -> window defaults to 24 hours ending now(UTC)."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl()

    assert result["status"] == "completed"
    start = datetime.fromisoformat(result["window_start"])
    end = datetime.fromisoformat(result["window_end"])
    delta_hours = (end - start).total_seconds() / 3600.0
    assert delta_hours == pytest.approx(etl.DEFAULT_WINDOW_HOURS, abs=1e-6)


def test_impl_closes_conn_even_on_error() -> None:
    """If execute() raises, the connection is still closed."""
    conn = _make_mock_conn(rowcount=0)
    cur = conn.cursor.return_value
    cur.execute.side_effect = RuntimeError("query exploded")

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    conn.close.assert_called_once()


def test_impl_passes_request_id_through() -> None:
    """The request_id arg is forwarded but does not change behaviour."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
            request_id="celery-task-xyz",
        )

    # request_id is purely a logging hint; doesn't appear in result.
    assert result["status"] == "completed"


# =============================================================================
# Celery task wrapper
# =============================================================================


def test_celery_task_delegates_to_impl() -> None:
    """The Celery task ``run_per_hcp_rollup`` is a thin shim over
    ``_run_per_hcp_rollup_impl`` — verify it forwards args + the
    request id."""
    sentinel_result = {"status": "completed", "rows_affected": 3}
    with patch.object(etl, "_run_per_hcp_rollup_impl", return_value=sentinel_result) as impl:
        # ``apply`` runs the task synchronously in-process. Args/kwargs
        # are forwarded to the underlying function.
        async_result = etl.run_per_hcp_rollup.apply(
            kwargs={
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T00:00:00Z",
            },
        )

    assert async_result.successful()
    assert async_result.result == sentinel_result
    impl.assert_called_once()
    call_kwargs = impl.call_args.kwargs
    assert call_kwargs["start_date"] == "2024-01-01T00:00:00Z"
    assert call_kwargs["end_date"] == "2024-01-02T00:00:00Z"
    # request_id is a non-empty string forwarded from the task.
    assert isinstance(call_kwargs["request_id"], str)
    assert call_kwargs["request_id"]


# =============================================================================
# Celery registration sanity
# =============================================================================


def test_task_is_registered_with_expected_name() -> None:
    """The Celery task name string is what the beat schedule references."""
    assert etl.run_per_hcp_rollup.name == "src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup"


def test_beat_schedule_entry_present() -> None:
    """The 24h beat entry routes the task to the analytics queue."""
    from src.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule.get("business-metrics-per-hcp-rollup")
    assert entry is not None, "beat schedule entry missing"
    assert entry["task"] == "src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup"
    assert entry["schedule"] == 86400.0
    assert entry["options"]["queue"] == "analytics"


# =============================================================================
# Provenance inheritance (issue #895)
# =============================================================================


class TestProvenanceInheritance:
    """Issue #895: the rollup must not launder synthetic provenance.

    ``business_metrics.is_synthetic`` (migration 063) defaults to ``false``,
    so an INSERT that omits the column writes derived rows that look "real"
    even when every aggregated input row is synthetic. The fix makes derived
    rows inherit ``is_synthetic = bool(any synthetic input)`` computed in the
    rollup SQL itself.
    """

    def test_insert_column_list_includes_is_synthetic(self) -> None:
        """The INSERT column list must name is_synthetic explicitly so the
        column default false can never apply to a derived row."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        # Strip comments so an explanatory comment can't satisfy the check.
        stripped = re.sub(r"--[^\n]*", "", sql)
        insert_clause = stripped.split("INSERT INTO business_metrics", 1)[1].split("SELECT", 1)[0]
        assert "is_synthetic" in insert_clause, (
            "is_synthetic missing from INSERT column list -- derived rows "
            "would land with the migration-063 default false (laundering)"
        )

    def test_provenance_aggregated_via_bool_or(self) -> None:
        """Row-level provenance must be collapsed with BOOL_OR (any synthetic
        input taints the aggregate)."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert re.search(r"BOOL_OR\s*\(", sql, re.IGNORECASE), (
            "no BOOL_OR aggregation of is_synthetic -- provenance cannot "
            "be inherited per-cell without it"
        )

    def test_all_three_source_tables_contribute_provenance(self) -> None:
        """triggers, patient_journeys (via the LATERAL brand join) and
        hcp_profiles all carry is_synthetic (migration 063); every one of
        them must feed the inherited flag."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "t.is_synthetic" in sql, "triggers provenance not read"
        assert "pj_inner.is_synthetic" in sql, (
            "patient_journeys provenance not read in the LATERAL subquery"
        )
        assert "hp.is_synthetic" in sql, "hcp_profiles provenance not read"

    def test_territory_denominator_contamination_propagates(self) -> None:
        """market_share divides by territory_totals.territory_total; if any
        HCP cell in that territory/brand/date is synthetic the denominator
        is synthetic-contaminated, so the territory_totals CTE must carry
        an any_synthetic flag that feeds the final is_synthetic."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "tt.any_synthetic" in sql, (
            "territory_totals provenance not propagated -- market_share "
            "denominators would silently mix synthetic counts into rows "
            "tagged real"
        )

    def test_on_conflict_update_arm_preserves_provenance(self) -> None:
        """Idempotent re-runs go through the DO UPDATE arm; laundered
        semantics must not survive via that path either."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        on_conflict_block = sql.split("ON CONFLICT (metric_id) DO UPDATE SET", 1)[1]
        assert re.search(r"is_synthetic\s*=\s*EXCLUDED\.is_synthetic", on_conflict_block), (
            "is_synthetic missing from the ON CONFLICT SET list -- a re-run "
            "would keep a stale provenance tag"
        )
