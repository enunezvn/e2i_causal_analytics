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
            "triggers_delivered_count",
            "triggers_accepted_count",
            "triggers_total_count",
            "market_share",
            "conversion_rate",
        ):
            # Allow any whitespace between the column name and "=".
            pattern = rf"{re.escape(col)}\s*=\s*EXCLUDED\.{re.escape(col)}"
            assert re.search(pattern, on_conflict_block), f"missing UPDATE SET clause for {col}"

    def test_market_share_is_share_of_territory_total(self) -> None:
        """market_share = triggers_total_count / territory_total, with a >0 guard."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "tt.territory_total > 0" in sql
        assert "hbd.triggers_total_count::NUMERIC / tt.territory_total" in sql

    def test_conversion_rate_uses_nullif_guard(self) -> None:
        """Division by zero is guarded via NULLIF; default to 0 via COALESCE."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert (
            "NULLIF(COUNT(*) FILTER (WHERE delivery_status IN ('delivered', 'viewed')), 0)" in sql
        )
        assert "COALESCE(" in sql

    def test_trx_and_conversion_use_ruled_delivered_union(self) -> None:
        """#1387: 'viewed' is a further progression of 'delivered' (migrations
        090/092 denominators). Once accepted implies viewed, a
        delivered-exclusive filter counts only the never-viewed remainder and
        conversion_rate (accepted / delivered) could exceed 1."""
        sql = etl.INSERT_PER_HCP_ROLLUP_SQL
        assert "delivery_status = 'delivered'" not in sql
        assert sql.count("delivery_status IN ('delivered', 'viewed')") == 2

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
    # ORDER-pinned, not last-call-pinned (canonical TRx lane, codex r13-08): the impl now
    # runs the reconcile DELETE in the SAME transaction, so `call_args` — which is the LAST
    # call — became the reconcile. Measured on a bare MagicMock: after a second execute,
    # `call_args[0][0] is INSERT` is False. The old assertion would therefore have failed
    # because of a statement it was never about, while this one keeps its subject (the
    # upsert and its bound params) and additionally pins that the reconcile follows it.
    args, _ = cur.execute.call_args_list[0]
    assert args[0] is etl.INSERT_PER_HCP_ROLLUP_SQL
    assert cur.execute.call_args_list[1].args[0] is etl.RECONCILE_PER_HCP_ROLLUP_SQL
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


def test_impl_default_window_is_the_overlapping_arrival_window() -> None:
    """No dates supplied (the daily beat) -> an ARRIVAL window of one weekly cycle + margin ending now(UTC).

    Changed deliberately (canonical TRx lane, owner decision #2): the former 24 h
    trigger_timestamp window rolled up only Monday's share of each weekly trigger batch.
    The old assertion pinned ``DEFAULT_WINDOW_HOURS`` — 24 — as the correct default, and
    that default IS the defect: the host reseed lands a whole Tue..Mon week of triggers in
    one Monday 03:00 batch, each stamped 00:00 of its own day, so a 24 h window at the
    03:15 beat selects Monday alone. Measured consequence: since 2026-05-01, 89 trigger
    dates / 24,002 triggers have no per-HCP row, and the last 56 days have rows on Mondays
    only. So this is not an assertion made inconvenient by the fix — it asserted the wrong
    thing about the product, and leaving it would have required the fix to fail the suite.

    Duration alone would be a proxy: a 174 h window on the WRONG column would satisfy it.
    ``selected_by`` below binds the column, and the SQL text is pinned separately by
    ``test_the_scheduled_variant_selects_dates_by_arrival``.
    """
    conn = _make_mock_conn(rowcount=1)

    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl()

    assert result["status"] == "completed"
    assert result["selected_by"] == "arrival"
    start = datetime.fromisoformat(result["window_start"])
    end = datetime.fromisoformat(result["window_end"])
    delta_hours = (end - start).total_seconds() / 3600.0
    assert delta_hours == pytest.approx(etl.ARRIVAL_WINDOW_HOURS, abs=1e-6)


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
    """The daily beat entry routes the task to the analytics queue.

    #1645: the cadence is a wall-clock crontab, not the old bare ``86400.0``
    interval — an interval is measured from ``last_run_at``, which every deploy
    reset, so a 24h entry never became due. 03:15 UTC also puts this head of the
    ETL -> corpus chain after the Monday 03:00 host reseed; the full slot map is
    in ``src/workers/celery_app.py``.
    """
    from celery.schedules import crontab

    from src.workers.celery_app import celery_app

    entry = celery_app.conf.beat_schedule.get("business-metrics-per-hcp-rollup")
    assert entry is not None, "beat schedule entry missing"
    assert entry["task"] == "src.etl.business_metrics_per_hcp_etl.run_per_hcp_rollup"
    assert entry["schedule"] == crontab(hour=3, minute=15)
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


def test_the_rollup_writes_the_honest_trigger_count_columns():
    """Canonical TRx lane / migration 144: these are trigger funnel counts.

    The NEGATIVE assertion is what gives this teeth. The three positive
    ``AS <column>`` bindings alone would pass on a module that had renamed the
    SELECT aliases and left ``triggers_delivered_count`` in the INSERT list, which is the exact
    half-done rename worth catching; "no legacy name survives anywhere in the
    module" cannot pass while any site still writes the old name, comments
    included.

    The INSERT and ON CONFLICT bindings are asserted separately from the
    aliases because the negative assertion has one blind spot: a future edit
    that REMOVED the three columns from the INSERT entirely would satisfy both
    "no legacy name" and "the aliases exist", while the rollup silently stopped
    persisting the counts.
    """
    import inspect
    import re as _re

    from src.etl import business_metrics_per_hcp_etl as _etl
    from tests._checkout_guard import assert_same_checkout

    # Capability, not a literal path: the module must come from THIS checkout.
    # A `.worktrees/<lane>` substring was the first form and it could only ever
    # pass inside one directory on one machine -- see tests/_checkout_guard.py.
    assert_same_checkout(_etl, __file__)
    source = inspect.getsource(_etl)
    columns = ("triggers_delivered_count", "triggers_accepted_count", "triggers_total_count")
    for column in columns:
        assert f"AS {column}" in source, f"{column}: no SELECT alias binding it"
    insert_sql = _etl.INSERT_PER_HCP_ROLLUP_SQL
    on_conflict = insert_sql.split("ON CONFLICT (metric_id) DO UPDATE SET", 1)[1]
    # The INSERT COLUMN LIST, parsed — not `column in insert_sql`. Measured: each
    # name occurs SEVEN times in this statement (SELECT alias, territory_total
    # SUM, the column list, the hbd projection, the market_share divisor, and
    # both halves of the ON CONFLICT line), so a substring test over the whole
    # SQL is satisfied by six sites that are not the column list and cannot fail
    # when the column list is what broke. Proven: dropping
    # `triggers_total_count,` from the list left the substring version green.
    listed = re.search(r"INSERT INTO business_metrics\s*\((.*?)\n\)\s*\nSELECT", insert_sql, re.S)
    assert listed, "could not find the INSERT column list — has the statement been restructured?"
    insert_columns = {
        line.strip().rstrip(",")
        for line in listed.group(1).splitlines()
        if line.strip() and not line.strip().startswith("--")
    }
    for column in columns:
        assert column in insert_columns, f"{column}: aliased but not in the INSERT column list"
        assert f"EXCLUDED.{column}" in on_conflict, f"{column}: absent from the re-run arm"
    assert not _re.search(r"\b(trx_count|nrx_count|total_rx_count)\b", source)


# =============================================================================
# Late-arrival selection (canonical TRx lane, owner decision #2 2026-09-15)
# =============================================================================


def _cte_body(sql: str, name: str, following: str) -> str:
    return sql.split(f"{name} AS (", 1)[1].split(f"{following} AS (", 1)[0]


def test_every_variant_recomputes_each_touched_date_whole() -> None:
    """The window picks DATES; the aggregate reads all triggers of those dates, because
    market_share divides by the per-(territory, brand, date) total."""
    for sql in (
        etl.INSERT_PER_HCP_ROLLUP_SQL,
        etl.INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL,
        etl.PREVIEW_PER_HCP_ROLLUP_SQL,
    ):
        body = _cte_body(sql, "triggers_with_brand", "hcp_brand_daily")
        assert re.search(
            r"DATE\(t\.trigger_timestamp\)\s+IN\s+\(SELECT metric_date FROM affected_dates\)", body
        )
        assert "%(start_date)s" not in body and "%(end_date)s" not in body


def test_the_scheduled_variant_selects_dates_by_arrival() -> None:
    body = _cte_body(
        etl.INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL, "affected_dates", "triggers_with_brand"
    )
    assert re.search(r"t\.created_at\s*>=\s*%\(start_date\)s", body)
    assert re.search(r"t\.created_at\s*<\s*%\(end_date\)s", body)
    assert "t.trigger_timestamp >=" not in body


def test_the_variants_differ_only_in_the_window_column() -> None:
    assert "__WINDOW_COLUMN__" not in etl.INSERT_PER_HCP_ROLLUP_SQL
    assert etl.INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL.count("t.created_at") == 2
    assert (
        etl.INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL.replace("t.created_at", "t.trigger_timestamp")
        == etl.INSERT_PER_HCP_ROLLUP_SQL
    )


def test_the_arrival_window_spans_a_weekly_batch_cycle_plus_margin() -> None:
    """The host reseed lands one batch per week and the beat runs daily: every run within a
    week of a batch re-touches it, so a missed or failed beat self-heals."""
    from src.workers.celery_app import celery_app

    beat = celery_app.conf.beat_schedule["business-metrics-per-hcp-rollup"]["schedule"]
    assert beat.hour == {3} and beat.minute == {15}  # one run per day
    assert etl.ARRIVAL_WINDOW_HOURS == 7 * 24 + etl.ARRIVAL_MARGIN_HOURS
    assert etl.ARRIVAL_MARGIN_HOURS >= 6


def test_impl_scheduled_run_uses_the_arrival_variant() -> None:
    conn = _make_mock_conn(rowcount=3)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl(arrived_before="2026-09-14T03:15:00+00:00")
    args, _ = conn.cursor.return_value.execute.call_args_list[0]  # [1] is the reconcile
    assert args[0] is etl.INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL
    assert result["selected_by"] == "arrival"
    assert result["window_end"].startswith("2026-09-14T03:15:00")
    assert result["window_start"].startswith("2026-09-06T21:15:00")


def test_impl_explicit_dates_keep_trigger_timestamp_selection() -> None:
    conn = _make_mock_conn(rowcount=3)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl(start_date="2026-05-01", end_date="2026-09-16")
    args, _ = conn.cursor.return_value.execute.call_args_list[0]  # [1] is the reconcile
    assert args[0] is etl.INSERT_PER_HCP_ROLLUP_SQL
    assert result["selected_by"] == "trigger_timestamp"


def test_impl_refuses_arrived_before_with_explicit_dates() -> None:
    with patch.object(etl, "_connect_to_db") as connect:
        result = etl._run_per_hcp_rollup_impl(
            start_date="2026-05-01",
            end_date="2026-09-16",
            arrived_before="2026-09-14T03:15:00+00:00",
        )
    assert result["status"] == "failed" and "arrived_before" in result["error"]
    connect.assert_not_called()


def test_preview_reads_the_insert_text_and_writes_nothing() -> None:
    preview = etl.PREVIEW_PER_HCP_ROLLUP_SQL
    cte_text = etl.INSERT_PER_HCP_ROLLUP_SQL.split("INSERT INTO business_metrics", 1)[0]
    assert preview.startswith(cte_text.rstrip())
    assert etl._PER_HCP_ROLLUP_ROWS_SELECT in preview
    assert etl._PER_HCP_ROLLUP_ROWS_SELECT in etl.INSERT_PER_HCP_ROLLUP_SQL
    code = "\n".join(line.split("--", 1)[0] for line in preview.splitlines())
    assert not re.search(r"\b(INSERT|UPDATE|DELETE|MERGE)\b|ON CONFLICT", code)


def test_preview_runs_in_a_read_only_transaction() -> None:
    conn = _make_mock_conn()
    cur = conn.cursor.return_value
    cur.fetchone.return_value = (7, 120, 3, 40, 5, 0, date(2026, 5, 1), date(2026, 9, 14))
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl.preview_per_hcp_rollup("2026-05-01", "2026-09-16")
    first, second = cur.execute.call_args_list
    assert first.args == ("SET TRANSACTION READ ONLY",)
    assert second.args[0] is etl.PREVIEW_PER_HCP_ROLLUP_SQL
    assert (
        result["metric_dates"],
        result["rows_new"],
        result["rows_changed"],
        result["rows_existing"],
    ) == (7, 120, 3, 40)
    assert result["rows_obsolete"] == 5  # codex r13-08: rows the upsert cannot remove
    conn.close.assert_called_once()


def test_the_reconcile_deletes_rows_that_are_no_longer_produced() -> None:
    """codex r13-08: an upsert cannot delete. A group that lost its last trigger on a
    touched date must go, or that date's market shares exceed 1."""
    for sql, scope in (
        (etl.RECONCILE_PER_HCP_ROLLUP_SQL, "b.metric_date >= %(start_date)s::DATE"),
        (
            etl.RECONCILE_PER_HCP_ROLLUP_BY_ARRIVAL_SQL,
            "b.metric_date IN (SELECT metric_date FROM affected_dates)",
        ),
    ):
        assert "DELETE FROM business_metrics b" in sql
        assert scope in sql
        assert "NOT EXISTS (SELECT 1 FROM rollup r WHERE r.metric_id = b.metric_id)" in sql
    # The explicit variant must NOT scope to the touched dates: a date that lost every
    # trigger is not in affected_dates at all, and its rows would survive for ever.
    # Deviation from the spec, which asserted `"affected_dates)" not in ...`: that literal
    # carries a closing paren, so `FROM affected_dates )` with a space would satisfy it
    # while the scope was exactly the one degree too narrow this test exists to forbid.
    # The DELETE region has no legitimate use for the CTE at all, so forbid the bare name.
    delete_region = etl.RECONCILE_PER_HCP_ROLLUP_SQL.split("DELETE FROM", 1)[1]
    assert "affected_dates" not in delete_region


def test_impl_reconciles_in_the_same_transaction_as_the_upsert() -> None:
    conn = _make_mock_conn(rowcount=2)
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl._run_per_hcp_rollup_impl(arrived_before="2026-09-14T03:15:00+00:00")
    executed = [call.args[0] for call in conn.cursor.return_value.execute.call_args_list]
    assert executed == [
        etl.INSERT_PER_HCP_ROLLUP_BY_ARRIVAL_SQL,
        etl.RECONCILE_PER_HCP_ROLLUP_BY_ARRIVAL_SQL,
    ]
    assert result["status"] == "completed" and result["rows_deleted"] == 2


# --- the preview names the cohort data an obsolete row still carries (2026-09-22) -----------


def test_preview_reports_obsolete_rows_that_still_carry_cohort_data() -> None:
    """2026-09-21: a full-window backfill's reconcile deleted rows whose planted twin channels
    and outcome the ETL does not own, and the twin went dark. The preview counts those rows
    separately, so the operator sees the cohort data the reconcile would take with it."""
    conn = _make_mock_conn()
    cur = conn.cursor.return_value
    cur.fetchone.return_value = (7, 120, 3, 40, 5, 2, date(2026, 5, 1), date(2026, 9, 14))
    with patch.object(etl, "_connect_to_db", return_value=conn):
        result = etl.preview_per_hcp_rollup("2026-05-01", "2026-09-16")
    assert result["rows_obsolete"] == 5
    assert result["rows_obsolete_with_cohort_data"] == 2
    assert (result["first_date"], result["last_date"]) == (date(2026, 5, 1), date(2026, 9, 14))


def test_the_cohort_data_count_names_every_planted_column_and_only_obsolete_rows() -> None:
    from src.data.per_hcp_cohort_columns import PLANTED_COLUMNS

    sql = etl._PREVIEW_COUNTS_SQL
    body = sql.split("AS rows_obsolete,", 1)[1].split("AS rows_obsolete_with_cohort_data", 1)[0]
    for col in PLANTED_COLUMNS:
        assert re.search(rf"\bo\.{col} IS NOT NULL", body), col
    # the same obsolete predicate as rows_obsolete and the reconcile: no rollup row produces it
    assert "NOT EXISTS (SELECT 1 FROM rollup r2 WHERE r2.metric_id = o.metric_id)" in body
    assert "o.metric_type = %(metric_type)s" in body
    assert tuple(etl.COHORT_DATA_COLUMNS) == PLANTED_COLUMNS


def test_the_etl_module_does_not_import_the_twin_package() -> None:
    """The premise of the light contract module: importing the ETL must stay cheap."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; import src.etl.business_metrics_per_hcp_etl; "
        "print(sorted(m for m in sys.modules if m in ('src.digital_twin', 'sklearn', 'dowhy', 'shap')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    ).stdout.strip()
    assert out == "[]", out
