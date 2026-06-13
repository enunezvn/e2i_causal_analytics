"""Integration tests for ETL provenance inheritance (issue #895).

Proves against a real PostgreSQL instance that the daily rollup ETLs no
longer launder synthetic provenance:

* ``business_metrics_per_hcp_etl`` — derived rows inherit
  ``is_synthetic = bool(any synthetic input)`` across triggers,
  patient_journeys (LATERAL brand join) and hcp_profiles, INCLUDING
  market_share denominator contamination (a real HCP sharing a territory
  cell with a synthetic HCP gets tagged synthetic, because its
  market_share divides by a synthetic-contaminated territory total).
* ``territory_metrics_etl`` — inheritance composes through the two-stage
  rollup once migration 074 adds ``territory_metrics.is_synthetic``. The
  territory-side test is additionally gated on column existence so the
  suite stays green on a pre-074 schema (CI before the migration batch is
  applied).

Substrate layout (all rows isolated by a per-run ``test_run_id`` prefix and
a far-past Jan-2003 window no live data occupies):

* Territory ``TS`` — synthetic HCP ``S1`` + real HCP ``R2`` (mixed cell).
* Territory ``TR`` — real HCP ``R1`` (pure-real cell).

Expected tags: S1 -> true (own inputs synthetic), R2 -> true (denominator
contamination), R1 -> false (pure real). Territory TS -> true, TR -> false.

Run gate
--------
Mirrors ``test_business_metrics_per_hcp_etl_integration.py``: requires
``SUPABASE_DB_URL`` plus the explicit ``E2I_DB_INTEGRATION=1`` opt-in;
skipped entirely otherwise.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

psycopg2 = pytest.importorskip("psycopg2")

pytestmark = pytest.mark.skipif(
    not (os.getenv("SUPABASE_DB_URL") and os.getenv("E2I_DB_INTEGRATION") == "1"),
    reason=(
        "SUPABASE_DB_URL and/or E2I_DB_INTEGRATION not set; integration "
        "test requires real Postgres + explicit opt-in."
    ),
)

# Far-past window: no production or synthetic-substrate rows live here, so
# the window-scoped ETLs only see this module's rows and teardown can
# delete by date without touching anything real.
WINDOW_START = datetime(2003, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2003, 2, 1, tzinfo=timezone.utc)
TRIGGER_TS = datetime(2003, 1, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def db_conn() -> Any:
    """Open a single psycopg2 connection for the module's tests."""
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def test_run_id() -> str:
    """Unique prefix per pytest run so parallel suites do not collide."""
    return uuid.uuid4().hex[:10]


def _territory_metrics_has_is_synthetic(conn: Any) -> bool:
    """True when migration 074 has been applied to the connected DB."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'territory_metrics'
               AND column_name = 'is_synthetic'
            """
        )
        row = cur.fetchone()
    return bool(row and row[0])


#: Session-level advisory-lock key reserving the far-past window across ALL
#: clients of the database (not just this host). Two concurrent runs of this
#: module would otherwise both observe an empty window and one teardown
#: could delete the other run's territory_metrics rows (codex round-2 HIGH).
_WINDOW_ADVISORY_LOCK_KEY = "e2i_etl_provenance_895_jan2003_window"


@pytest.fixture(scope="module")
def mixed_substrate(db_conn: Any, test_run_id: str) -> dict:
    """Insert the mixed real/synthetic substrate; tear down at module end.

    Serializes on a Postgres session-level advisory lock for the whole
    module so the window-emptiness proof below cannot race a concurrent
    run of this same suite, then refuses to run if the far-past window is
    not empty (another suite or real data occupying it would make
    provenance assertions ambiguous and the teardown delete unsafe).
    """
    rid = test_run_id
    hcps = {
        "S1": {"hcp_id": f"hcp895_{rid}_S1", "territory": f"TS_{rid}", "synthetic": True},
        "R2": {"hcp_id": f"hcp895_{rid}_R2", "territory": f"TS_{rid}", "synthetic": False},
        "R1": {"hcp_id": f"hcp895_{rid}_R1", "territory": f"TR_{rid}", "synthetic": False},
    }

    # Reserve the window BEFORE proving it empty: session-level advisory
    # lock, held until teardown releases it (and released by Postgres on
    # disconnect if the process dies). pg_advisory_lock blocks, so a
    # concurrent run waits its turn instead of racing the check.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_lock(hashtext(%s))",
                (_WINDOW_ADVISORY_LOCK_KEY,),
            )

    # The window must be empty in BOTH the source table the ETLs scan and
    # the territory sink: the territory ETL cross-products every
    # hcp_profiles.territory_id with the window's metric_dates, and
    # teardown deletes territory_metrics by date range -- that delete is
    # only safe ("rows this run created, nothing else") because we prove
    # here (under the advisory lock) that no foreign rows pre-exist in
    # the window and no concurrent run can add any while we hold it.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM triggers
                      WHERE trigger_timestamp >= %s AND trigger_timestamp < %s),
                    (SELECT COUNT(*) FROM territory_metrics
                      WHERE metric_date >= %s AND metric_date < %s)
                """,
                (WINDOW_START, WINDOW_END, WINDOW_START.date(), WINDOW_END.date()),
            )
            row = cur.fetchone()
            preexisting_triggers, preexisting_tm = row if row else (0, 0)

    if preexisting_triggers or preexisting_tm:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))",
                    (_WINDOW_ADVISORY_LOCK_KEY,),
                )
        pytest.skip(
            f"far-past test window not empty ({preexisting_triggers} triggers, "
            f"{preexisting_tm} territory_metrics rows); refusing to run "
            "provenance assertions over (or delete) foreign rows"
        )

    with db_conn:
        with db_conn.cursor() as cur:
            for key, hcp in hcps.items():
                region = "northeast" if hcp["territory"].startswith("TS") else "south"
                cur.execute(
                    """
                    INSERT INTO hcp_profiles (
                        hcp_id, territory_id, geographic_region, is_synthetic
                    ) VALUES (%s, %s, %s::region_type, %s)
                    ON CONFLICT (hcp_id) DO NOTHING
                    """,
                    (hcp["hcp_id"], hcp["territory"], region, hcp["synthetic"]),
                )
                cur.execute(
                    """
                    INSERT INTO patient_journeys (
                        patient_journey_id, patient_id, journey_start_date,
                        journey_stage, journey_status, brand,
                        geographic_region, hcp_id, is_synthetic
                    ) VALUES (
                        %s, %s, %s, 'diagnosis'::journey_stage_type,
                        'active'::journey_status_type,
                        'Remibrutinib'::brand_type, %s::region_type, %s, %s
                    )
                    ON CONFLICT (patient_journey_id) DO NOTHING
                    """,
                    (
                        f"pj895_{rid}_{key}",
                        f"pat895_{rid}_{key}",
                        WINDOW_START.date(),
                        region,
                        hcp["hcp_id"],
                        hcp["synthetic"],
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO triggers (
                        trigger_id, patient_id, hcp_id, trigger_timestamp,
                        brand_id, delivery_status, acceptance_status,
                        is_synthetic
                    ) VALUES (%s, %s, %s, %s, 'UNKNOWN', 'delivered',
                              'responded', %s)
                    ON CONFLICT (trigger_id) DO NOTHING
                    """,
                    (
                        f"tr895_{rid}_{key}",
                        f"pat895_{rid}_{key}",
                        hcp["hcp_id"],
                        TRIGGER_TS,
                        hcp["synthetic"],
                    ),
                )

    yield {"run_id": rid, "hcps": hcps}

    # Teardown in reverse FK order; territory_metrics rows are deletable by
    # the far-past metric_date window (the territory ETL cross-products
    # every territory in hcp_profiles against the window's metric_dates, so
    # prefix-matching territory_id alone would leak foreign-territory rows
    # created by our own ETL run). The date-scoped delete is safe ONLY
    # because the fixture proved the window held zero territory_metrics
    # rows AND has held the window's advisory lock ever since -- everything
    # in it now is ours. The lock is released after the delete.
    try:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM territory_metrics WHERE metric_date >= %s AND metric_date < %s",
                    (WINDOW_START.date(), WINDOW_END.date()),
                )
                cur.execute(
                    "DELETE FROM business_metrics WHERE hcp_id LIKE %s",
                    (f"hcp895_{rid}_%",),
                )
                cur.execute("DELETE FROM triggers WHERE trigger_id LIKE %s", (f"tr895_{rid}_%",))
                cur.execute(
                    "DELETE FROM patient_journeys WHERE patient_journey_id LIKE %s",
                    (f"pj895_{rid}_%",),
                )
                cur.execute("DELETE FROM hcp_profiles WHERE hcp_id LIKE %s", (f"hcp895_{rid}_%",))
    finally:
        with db_conn:
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))",
                    (_WINDOW_ADVISORY_LOCK_KEY,),
                )


def _run_per_hcp(window_suffix: str = "") -> dict:
    from src.etl.business_metrics_per_hcp_etl import _run_per_hcp_rollup_impl

    return _run_per_hcp_rollup_impl(
        start_date=WINDOW_START.isoformat(),
        end_date=WINDOW_END.isoformat(),
        request_id=f"integration-895{window_suffix}",
    )


def _fetch_tags(db_conn: Any, rid: str) -> dict[str, bool]:
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT hcp_id, is_synthetic FROM business_metrics WHERE hcp_id LIKE %s",
            (f"hcp895_{rid}_%",),
        )
        return {hcp_id.rsplit("_", 1)[-1]: tag for hcp_id, tag in cur.fetchall()}


def test_per_hcp_rollup_inherits_provenance(db_conn: Any, mixed_substrate: dict) -> None:
    """Synthetic-input cells land tagged; pure-real cells stay real; a real
    HCP sharing a territory denominator with a synthetic HCP is tagged."""
    result = _run_per_hcp()
    assert result["status"] == "completed", f"ETL failed: {result}"

    tags = _fetch_tags(db_conn, mixed_substrate["run_id"])
    assert set(tags) == {"S1", "R2", "R1"}, f"unexpected rollup rows: {tags}"
    assert tags["S1"] is True, "all-synthetic input cell must be tagged synthetic"
    assert tags["R2"] is True, (
        "real HCP in a mixed territory cell must be tagged synthetic -- its "
        "market_share denominator includes synthetic counts"
    )
    assert tags["R1"] is False, "pure-real cell must NOT be tagged synthetic"


def test_per_hcp_rerun_keeps_tags_via_update_arm(db_conn: Any, mixed_substrate: dict) -> None:
    """Idempotent re-run goes through ON CONFLICT DO UPDATE; the provenance
    tag must be recomputed there too, not frozen or defaulted."""
    _run_per_hcp("-a")
    _run_per_hcp("-b")
    tags = _fetch_tags(db_conn, mixed_substrate["run_id"])
    assert tags == {"S1": True, "R2": True, "R1": False}


def test_territory_rollup_composes_provenance(db_conn: Any, mixed_substrate: dict) -> None:
    """Second-order rollup inherits through the tagged per-HCP rows.

    Skipped on a pre-074 schema: territory_metrics gains is_synthetic in
    migration 074, which ships with this fix but is applied in the
    campaign's batch phase. The territory ETL itself fails closed (42703)
    on a pre-074 schema rather than writing provenance-less rows.
    """
    if not _territory_metrics_has_is_synthetic(db_conn):
        pytest.skip("territory_metrics.is_synthetic absent (migration 074 not applied)")

    from src.etl.territory_metrics_etl import _run_territory_rollup_impl

    _run_per_hcp("-territory")
    result = _run_territory_rollup_impl(
        start_date=WINDOW_START.isoformat(),
        end_date=WINDOW_END.isoformat(),
        request_id="integration-895-territory",
    )
    assert result["status"] == "completed", f"territory ETL failed: {result}"

    rid = mixed_substrate["run_id"]
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT territory_id, BOOL_AND(is_synthetic), BOOL_OR(is_synthetic)"
            "  FROM territory_metrics"
            " WHERE territory_id IN (%s, %s)"
            " GROUP BY territory_id",
            (f"TS_{rid}", f"TR_{rid}"),
        )
        rows = {tid: (all_syn, any_syn) for tid, all_syn, any_syn in cur.fetchall()}

    assert f"TS_{rid}" in rows and f"TR_{rid}" in rows, f"missing rollup rows: {rows}"
    assert rows[f"TS_{rid}"][0] is True, "mixed territory must be tagged synthetic"
    assert rows[f"TR_{rid}"][1] is False, "pure-real territory must stay real"
