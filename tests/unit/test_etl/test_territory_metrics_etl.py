"""Unit tests for ``src.etl.territory_metrics_etl``.

These tests do not touch a real database. They verify:

* The SQL string contains the load-bearing CTEs, parameter placeholders,
  the four expected aggregations (total_trx / total_nrx / active_hcp_count
  / covered_lives), the 30-day INTERVAL for active_hcp_count, the
  ``ON CONFLICT (territory_id, metric_date)`` clause, and that
  ``market_potential`` / ``resource_allocation_score`` are written as
  explicit NULL on INSERT (with migration 033 dropping the legacy
  NOT NULL DEFAULT 0) but stay out of the ON CONFLICT SET clause so
  pre-existing 031 random seeds survive re-runs untouched.
* ``_run_territory_rollup_impl`` orchestrates the connect/execute/commit/
  close flow correctly with mocks, and surfaces ``status`` /
  ``rows_affected`` faithfully. The Celery wrapper ``run_territory_rollup``
  is a thin one-liner over this and is exercised in the integration test.
* The shared helpers ``_resolve_db_connection_string``, ``_connect_to_db``
  and ``_resolve_window`` are re-exported here for backward compatibility
  (the canonical tests live in ``test_common.py``).

Behaviour-level assertions about exact territorial sums on real synthetic
data live in
``tests/integration/test_territory_metrics_etl_integration.py``.
"""

from __future__ import annotations

import re
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Importing the module triggers `from src.workers.celery_app import celery_app`,
# which only requires standard library + celery (already installed). No DB
# connection happens at import time.
from src.etl import _common
from src.etl import territory_metrics_etl as etl

# =============================================================================
# _common helper re-exports
# =============================================================================


def test_helpers_are_re_exported_from_common() -> None:
    """The three shared helpers must be accessible as module attributes on
    ``territory_metrics_etl`` so existing test imports keep working.

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
    """Pin the load-bearing structure of INSERT_TERRITORY_ROLLUP_SQL.

    These tests intentionally lean on substring presence rather than full
    SQL parse — a parser would be overkill for the small set of guarantees
    this query needs to keep across refactors.
    """

    def test_has_named_ctes(self) -> None:
        """All CTEs the impl relies on are present by name."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "metric_dates AS" in sql
        assert "territories AS" in sql
        assert "territory_dates AS" in sql
        assert "per_hcp_in_territory AS" in sql
        assert "active_hcp_per_territory_date AS" in sql
        assert "territory_hcp_volume AS" in sql

    def test_uses_named_parameters(self) -> None:
        """psycopg2 named-param style %()s is required so the params dict
        in run_territory_rollup matches."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "%(start_date)s" in sql
        assert "%(end_date)s" in sql
        assert "%(per_hcp_metric_type)s" in sql

    def test_aggregations_present(self) -> None:
        """The four aggregations the plan asks for are all in the SQL."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        # SUM(trx_count) / SUM(nrx_count) per territory+date.
        assert re.search(
            r"SUM\(\s*COALESCE\(\s*bm\.trx_count,\s*0\s*\)\s*\)::BIGINT\s+AS\s+total_trx",
            normalised,
        ), "missing SUM(trx_count) AS total_trx"
        assert re.search(
            r"SUM\(\s*COALESCE\(\s*bm\.nrx_count,\s*0\s*\)\s*\)::BIGINT\s+AS\s+total_nrx",
            normalised,
        ), "missing SUM(nrx_count) AS total_nrx"
        # COUNT(DISTINCT hcp_id) AS active_hcp_count.
        assert re.search(
            r"COUNT\(DISTINCT\s+t\.hcp_id\)::BIGINT\s+AS\s+active_hcp_count",
            normalised,
        ), "missing COUNT(DISTINCT hcp_id) AS active_hcp_count"
        # SUM(total_patient_volume) AS covered_lives.
        assert re.search(
            r"SUM\(\s*COALESCE\(\s*total_patient_volume,\s*0\s*\)\s*\)::BIGINT\s+AS\s+covered_lives",
            normalised,
        ), "missing SUM(total_patient_volume) AS covered_lives"

    def test_active_hcp_uses_30_day_interval(self) -> None:
        """active_hcp_count uses a 30-day backward-looking window from
        metric_date (inclusive of metric_date itself)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        # Lower bound: trigger_timestamp >= metric_date - INTERVAL '30 days'.
        assert re.search(
            r"t\.trigger_timestamp\s*>=\s*td\.metric_date\s*-\s*INTERVAL\s*'30 days'",
            normalised,
        ), "missing 30-day backward-looking lower bound"
        # Upper bound: trigger_timestamp < metric_date + INTERVAL '1 day'.
        # (Inclusive of metric_date.)
        assert re.search(
            r"t\.trigger_timestamp\s*<\s*td\.metric_date\s*\+\s*INTERVAL\s*'1 day'",
            normalised,
        ), "missing inclusive-of-metric_date upper bound"

    def test_per_hcp_aggregation_filters_to_per_hcp_rollup_rows(self) -> None:
        """``per_hcp_in_territory`` joins business_metrics restricted to
        per-HCP rollup rows only (excluding the legacy aggregate
        per-(brand, region) rows that keep hcp_id IS NULL)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        assert "bm.metric_type = %(per_hcp_metric_type)s" in normalised
        assert "bm.hcp_id IS NOT NULL" in normalised

    def test_covered_lives_sources_from_total_patient_volume(self) -> None:
        """covered_lives comes from hcp_profiles.total_patient_volume per
        the plan; the CTE name + the SUM expression both reflect that."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "FROM hcp_profiles" in sql
        # The SUM expression appears in territory_hcp_volume CTE.
        assert "total_patient_volume" in sql

    def test_on_conflict_uses_pk(self) -> None:
        """ON CONFLICT targets (territory_id, metric_date) -- matches the
        territory_metrics PK exactly. No md5-hashing trick needed."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "ON CONFLICT (territory_id, metric_date) DO UPDATE SET" in sql

    def test_on_conflict_updates_only_real_aggregates(self) -> None:
        """The SET clause covers only the four real aggregates; market_
        potential and resource_allocation_score are intentionally OMITTED
        so existing values (e.g. migration 031's random seed) survive."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        on_conflict_block = sql.split("ON CONFLICT (territory_id, metric_date) DO UPDATE SET", 1)[1]

        # Required: the four real aggregates.
        for col in ("total_trx", "total_nrx", "active_hcp_count", "covered_lives"):
            pattern = rf"{re.escape(col)}\s*=\s*EXCLUDED\.{re.escape(col)}"
            assert re.search(pattern, on_conflict_block), f"missing UPDATE SET clause for {col}"

        # Forbidden: the two columns that must remain untouched. Use the
        # narrower SET-clause-only block so we don't trip on the OMIT
        # comment in the INSERT column list.
        for col in ("market_potential", "resource_allocation_score"):
            pattern = rf"{re.escape(col)}\s*=\s*EXCLUDED\."
            assert not re.search(pattern, on_conflict_block), (
                f"{col} must NOT be in the ON CONFLICT SET clause "
                "(real Reltio/Veeva source not yet integrated)"
            )

    def test_market_potential_and_resource_allocation_score_written_as_null(
        self,
    ) -> None:
        """Plan: NULL otherwise (NOT random). Migration 033 drops the legacy
        NOT NULL DEFAULT 0 that 031 had set, so writing NULL explicitly is
        well-defined and matches the spec ("NULL otherwise, NOT random")
        even on fresh databases that don't already carry 031's random
        seed. The ON CONFLICT SET clause excluding them is asserted by
        ``test_on_conflict_updates_only_real_aggregates``; this test
        focuses on the INSERT side.

        Strip line comments before checking, since the explanatory comments
        in the column list and SELECT block naming both columns must NOT
        count as column-list matches.
        """
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        # Strip everything from `--` to end-of-line so comments don't
        # spuriously match column names.
        stripped = re.sub(r"--[^\n]*", "", sql)
        # Slice the INSERT INTO ... ( column-list ) block.
        insert_clause = stripped.split("INSERT INTO territory_metrics", 1)[1].split("SELECT", 1)[0]
        assert "market_potential" in insert_clause, (
            "market_potential must be in the INSERT column list "
            "(migration 033 dropped the NOT NULL so NULL is well-defined)"
        )
        assert "resource_allocation_score" in insert_clause, (
            "resource_allocation_score must be in the INSERT column list "
            "(migration 033 dropped the NOT NULL so NULL is well-defined)"
        )

        # The SELECT block following INSERT INTO must write
        # CAST(NULL AS DOUBLE PRECISION) for both columns -- explicit NULL,
        # not table default, not COALESCE'd zero. Anchor on the
        # INSERT-INTO-... slice so we don't trip on the SELECT inside
        # the upstream WITH ... metric_dates CTE.
        post_insert = stripped.split("INSERT INTO territory_metrics", 1)[1]
        normalised_post_insert = re.sub(r"\s+", " ", post_insert)
        assert re.search(
            r"CAST\(NULL AS DOUBLE PRECISION\)\s+AS\s+market_potential",
            normalised_post_insert,
        ), "market_potential must be SELECTed as CAST(NULL AS DOUBLE PRECISION)"
        assert re.search(
            r"CAST\(NULL AS DOUBLE PRECISION\)\s+AS\s+resource_allocation_score",
            normalised_post_insert,
        ), "resource_allocation_score must be SELECTed as CAST(NULL AS DOUBLE PRECISION)"

        # The original SQL still names the missing real source so future
        # readers see why these columns are NULL.
        assert "Reltio" in sql or "Veeva" in sql, "module SQL must name the missing real source"

    def test_run_window_filters_metric_dates(self) -> None:
        """The metric_dates CTE filters on the [start_date, end_date)
        half-open run window."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert re.search(r"bm\.metric_date\s*>=\s*%\(start_date\)s::DATE", sql), (
            "missing metric_date >= start_date filter"
        )
        assert re.search(r"bm\.metric_date\s*<\s*%\(end_date\)s::DATE", sql), (
            "missing metric_date < end_date filter"
        )

    def test_left_joins_anchor_at_territory_dates(self) -> None:
        """All three aggregate CTEs are LEFT JOINed onto territory_dates so
        a territory with no business_metrics for the day still gets a row
        (with COALESCE'd zeros)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        normalised = re.sub(r"\s+", " ", sql)
        assert "LEFT JOIN per_hcp_in_territory" in normalised
        assert "LEFT JOIN active_hcp_per_territory_date" in normalised
        assert "LEFT JOIN territory_hcp_volume" in normalised


# =============================================================================
# _run_territory_rollup_impl — orchestration
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
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "completed"
    assert result["rows_affected"] == 42
    assert result["window_start"].startswith("2024-01-01T00:00:00")
    assert result["window_end"].startswith("2024-01-02T00:00:00")

    connect.assert_called_once()
    cur = conn.cursor.return_value
    args, _ = cur.execute.call_args
    assert args[0] is etl.INSERT_TERRITORY_ROLLUP_SQL
    params = args[1]
    assert params["per_hcp_metric_type"] == etl.PER_HCP_METRIC_TYPE
    assert isinstance(params["start_date"], datetime)
    assert isinstance(params["end_date"], datetime)
    # The ETL only needs three params -- no metric_id_prefix because there's
    # no md5 trick (PK is the natural key already).
    assert set(params.keys()) == {"start_date", "end_date", "per_hcp_metric_type"}
    conn.close.assert_called_once()


def test_impl_no_data_path() -> None:
    """rowcount==0 reports status=no_data with rows_affected=0."""
    conn = _make_mock_conn(rowcount=0)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "no_data"
    assert result["rows_affected"] == 0


def test_impl_db_failure_returns_failed() -> None:
    """A connection / execute exception is caught and surfaced as failed."""
    with patch.object(etl, "_connect_to_db", side_effect=RuntimeError("boom")):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    assert result["error"] == "boom"
    assert result["rows_affected"] == 0


def test_impl_invalid_window_returns_failed() -> None:
    """Inverted window short-circuits before any DB connect."""
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_territory_rollup_impl(
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
        result = etl._run_territory_rollup_impl()

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
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    assert result["status"] == "failed"
    conn.close.assert_called_once()


def test_impl_passes_request_id_through() -> None:
    """The request_id arg is forwarded but does not change behaviour."""
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
            request_id="celery-task-xyz",
        )

    assert result["status"] == "completed"


def test_impl_delegates_window_resolution_to_resolve_window() -> None:
    """``_resolve_window`` is the single source of truth for window math.

    Pin that the impl calls the helper with the user's args (don't re-test
    the helper itself -- ``test_common.py`` covers the resolver directly).
    """
    conn = _make_mock_conn(rowcount=1)
    with (
        patch.object(etl, "_connect_to_db", return_value=conn),
        patch.object(
            etl,
            "_resolve_window",
            wraps=etl._resolve_window,
        ) as resolve,
    ):
        etl._run_territory_rollup_impl(
            start_date="2024-01-01T00:00:00Z",
            end_date="2024-01-02T00:00:00Z",
        )

    resolve.assert_called_once_with("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")


# =============================================================================
# Celery task wrapper
# =============================================================================


def test_celery_task_delegates_to_impl() -> None:
    """The Celery task ``run_territory_rollup`` is a thin shim over
    ``_run_territory_rollup_impl`` — verify it forwards args + the
    request id."""
    sentinel_result = {"status": "completed", "rows_affected": 3}
    with patch.object(etl, "_run_territory_rollup_impl", return_value=sentinel_result) as impl:
        async_result = etl.run_territory_rollup.apply(
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
    assert isinstance(call_kwargs["request_id"], str)
    assert call_kwargs["request_id"]


# =============================================================================
# Celery registration sanity
# =============================================================================


def test_task_is_registered_with_expected_name() -> None:
    """The Celery task name string is what the beat schedule references."""
    assert etl.run_territory_rollup.name == "src.etl.territory_metrics_etl.run_territory_rollup"


def test_beat_schedule_entry_present() -> None:
    """The 24h beat entry routes the task to the analytics queue."""
    from src.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule.get("territory-metrics-rollup")
    assert entry is not None, "beat schedule entry missing"
    assert entry["task"] == "src.etl.territory_metrics_etl.run_territory_rollup"
    assert entry["schedule"] == 86400.0
    assert entry["options"]["queue"] == "analytics"


# =============================================================================
# Provenance inheritance (issue #895)
# =============================================================================


class TestProvenanceInheritance:
    """Issue #895: second-order laundering into territory_metrics.

    ``territory_metrics`` gains ``is_synthetic`` in migration 074. The
    rollup aggregates per-HCP ``business_metrics`` rows (themselves now
    provenance-tagged by 6B-infra-2a post-#895), raw ``triggers``, and
    ``hcp_profiles`` volumes -- all three carry is_synthetic, and the
    derived (territory_id, metric_date) row must inherit
    ``bool(any synthetic input)``.
    """

    def test_insert_column_list_includes_is_synthetic(self) -> None:
        """The INSERT column list must name is_synthetic explicitly so the
        migration-074 default false can never apply to a derived row."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        stripped = re.sub(r"--[^\n]*", "", sql)
        insert_clause = stripped.split("INSERT INTO territory_metrics", 1)[1].split("SELECT", 1)[0]
        assert "is_synthetic" in insert_clause, (
            "is_synthetic missing from INSERT column list -- derived rows "
            "would land with the migration-074 default false (laundering)"
        )

    def test_provenance_aggregated_via_bool_or(self) -> None:
        """Row-level provenance must be collapsed with BOOL_OR (any synthetic
        input taints the aggregate)."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert re.search(r"BOOL_OR\s*\(", sql, re.IGNORECASE), (
            "no BOOL_OR aggregation of is_synthetic -- provenance cannot "
            "be inherited per-cell without it"
        )

    def test_all_three_aggregate_ctes_contribute_provenance(self) -> None:
        """per_hcp_in_territory (business_metrics), active_hcp_per_territory_
        date (triggers) and territory_hcp_volume (hcp_profiles) each feed an
        aggregate column of the derived row, so each must carry provenance
        into the final is_synthetic."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        assert "bm.is_synthetic" in sql, "business_metrics provenance not read"
        assert "t.is_synthetic" in sql, "triggers provenance not read"
        for alias in ("pht.any_synthetic", "ahd.any_synthetic", "thv.any_synthetic"):
            assert alias in sql, f"{alias} not propagated to the final SELECT"

    def test_on_conflict_update_arm_preserves_provenance(self) -> None:
        """Idempotent re-runs go through the DO UPDATE arm; laundered
        semantics must not survive via that path either."""
        sql = etl.INSERT_TERRITORY_ROLLUP_SQL
        on_conflict_block = sql.split("ON CONFLICT (territory_id, metric_date) DO UPDATE SET", 1)[1]
        assert re.search(r"is_synthetic\s*=\s*EXCLUDED\.is_synthetic", on_conflict_block), (
            "is_synthetic missing from the ON CONFLICT SET list -- a re-run "
            "would keep a stale provenance tag"
        )
