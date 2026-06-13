"""Integration tests for ``src.etl.territory_metrics_etl``.

Exercises the full ETL against a real PostgreSQL instance reached via
``SUPABASE_DB_URL``. Tests are skipped when the env var is unset so CI
unit-only runs stay green.

Synthetic dataset
-----------------
Three territories x five HCPs each, with deterministic per-HCP
``business_metrics`` rows (per_hcp_rollup type, simulating 6B-infra-2a
output) and ``triggers`` rows. The dataset shape lets us pin the four
aggregations:

* total_trx / total_nrx == SUM of per-HCP rows in the territory.
* covered_lives == SUM of total_patient_volume across the territory's HCPs.
* active_hcp_count == DISTINCT hcp_id with >= 1 trigger in the 30-day
  window ending on metric_date.

All rows are isolated by a unique ``test_run_id`` prefix on every primary
key so cleanup is deterministic. Cleanup runs in ``try/finally`` around the
``yield`` so an interrupted pytest still leaves the DB clean (mirrors the
I4 fix from 6B-infra-2a / 2b).

Order dependency
----------------
This ETL aggregates the per-HCP business_metrics rows produced by
6B-infra-2a. The synthetic fixture inserts those rows directly (rather
than running 2a) so the integration test stays decoupled from 2a's exact
trigger->rollup math. The values mirror what 2a would have produced for
the synthetic patterns.

Run gate
--------
Two env vars are required:

* ``SUPABASE_DB_URL`` -- Postgres URL pointing at a DB that has migrations
  031 + 033 applied.
* ``E2I_DB_INTEGRATION=1`` -- explicit opt-in; mirrors 6B-infra-2a/2b.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

# psycopg2 is a transitive dep of the Supabase client and the ETL itself,
# but unit-only environments may install without it. Skip the whole module
# rather than ImportError when the binary is absent.
psycopg2 = pytest.importorskip("psycopg2")

# Module-level skip: developers must opt in AND have a reachable Postgres URL
# before the suite executes. Mirrors the 6B-infra-2a/2b integration-test gate.
pytestmark = pytest.mark.skipif(
    not (os.getenv("SUPABASE_DB_URL") and os.getenv("E2I_DB_INTEGRATION") == "1"),
    reason=(
        "SUPABASE_DB_URL and/or E2I_DB_INTEGRATION not set; integration "
        "test requires real Postgres + explicit opt-in. Run with "
        "E2I_DB_INTEGRATION=1 once your DB has migrations 031 + 033 applied."
    ),
)


# Test config -- 3 territories x 5 HCPs each.
TERRITORIES: tuple[tuple[str, str], ...] = (
    ("T1", "northeast"),
    ("T2", "south"),
    ("T3", "west"),
)
HCPS_PER_TERRITORY: int = 5
NUM_DAYS: int = 3  # number of metric_dates we generate per-HCP rows for
TRX_PER_HCP_PER_DAY: int = 4
NRX_PER_HCP_PER_DAY: int = 2
PATIENT_VOLUME_PER_HCP: int = 100


@pytest.fixture(scope="module")
def db_conn() -> Any:
    """Open a single psycopg2 connection for the module's tests."""
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    yield conn
    conn.close()


@pytest.fixture(scope="module", autouse=True)
def _require_migration_074(db_conn: Any) -> None:
    """Skip the module on a pre-074 schema (issue #895).

    The rollup SQL now inherits provenance and names
    ``territory_metrics.is_synthetic`` (added by migration 074) in its
    INSERT column list; on a pre-074 schema the ETL fails closed (42703
    undefined_column, no rows written) by design. Skipping keeps local
    runs green until the migration batch is applied; the provenance
    behaviour itself is covered by
    ``test_etl_provenance_inheritance_895.py``.
    """
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'territory_metrics'
               AND column_name = 'is_synthetic'
            """
        )
        row = cur.fetchone()
    if not (row and row[0]):
        pytest.skip("territory_metrics.is_synthetic absent (migration 074 not applied)")


@pytest.fixture(scope="module")
def test_run_id() -> str:
    """Unique prefix per pytest run so parallel suites do not collide."""
    return uuid.uuid4().hex[:10]


@pytest.fixture(scope="module")
def synthetic_dataset(db_conn: Any, test_run_id: str) -> dict:
    """Insert HCPs, per-HCP business_metrics rows, and triggers; tear down
    at module end.

    Returns a dict with the IDs and the ``(start_date, end_date)`` window
    spanning the synthetic data, plus the expected per-territory sums so
    the assertions can pin them deterministically.

    Wrapped in ``try/finally`` so interrupted pytest still cleans up
    (mirrors I4 carry-over from 6B-infra-2a/2b).
    """
    base_date = date(2024, 6, 1)
    start_dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
    end_dt = datetime(2024, 6, 1, tzinfo=timezone.utc) + timedelta(days=NUM_DAYS)

    # Build the HCP roster.
    hcps: list[dict] = []
    for terr_idx, (territory_short, region) in enumerate(TERRITORIES):
        # Make the territory_id unique per run so we don't aggregate the
        # entire database's HCPs into the rollup.
        territory_id = f"{territory_short}_{test_run_id}"
        for j in range(HCPS_PER_TERRITORY):
            hcps.append(
                {
                    "hcp_id": f"hcp_{test_run_id}_{terr_idx}_{j:02d}",
                    "territory_id": territory_id,
                    "territory_short": territory_short,
                    "region": region,
                    "patient_volume": PATIENT_VOLUME_PER_HCP,
                }
            )

    # Pre-compute expected per-territory aggregations.
    expected_per_territory: dict[str, dict] = {}
    for territory_short, _ in TERRITORIES:
        territory_id = f"{territory_short}_{test_run_id}"
        # Per-day sums: each HCP contributes TRX_PER_HCP_PER_DAY trx and
        # NRX_PER_HCP_PER_DAY nrx; each territory has HCPS_PER_TERRITORY HCPs.
        expected_per_territory[territory_id] = {
            "total_trx_per_day": HCPS_PER_TERRITORY * TRX_PER_HCP_PER_DAY,
            "total_nrx_per_day": HCPS_PER_TERRITORY * NRX_PER_HCP_PER_DAY,
            "covered_lives": HCPS_PER_TERRITORY * PATIENT_VOLUME_PER_HCP,
            "active_hcp_count": HCPS_PER_TERRITORY,
        }

    try:
        with db_conn:
            with db_conn.cursor() as cur:
                # 1. Insert HCPs.
                for hcp in hcps:
                    cur.execute(
                        """
                        INSERT INTO hcp_profiles (
                            hcp_id, territory_id, geographic_region,
                            total_patient_volume, sales_rep_id
                        ) VALUES (%s, %s, %s::region_type, %s, NULL)
                        ON CONFLICT (hcp_id) DO NOTHING
                        """,
                        (
                            hcp["hcp_id"],
                            hcp["territory_id"],
                            hcp["region"],
                            hcp["patient_volume"],
                        ),
                    )

                # 2. Insert per-HCP business_metrics rows simulating
                # 6B-infra-2a output. metric_id is just the prefix +
                # uniquely-formed string (we don't need md5 here -- the
                # column accepts anything that fits VARCHAR(50)).
                bm_counter = 0
                for day_offset in range(NUM_DAYS):
                    metric_date = base_date + timedelta(days=day_offset)
                    for hcp in hcps:
                        bm_counter += 1
                        # 50-char limit: 'bm_<run_id>_<6-digit>' = 6 + 10 + 1 + 6 = 23 chars
                        metric_id = f"bm_{test_run_id}_{bm_counter:06d}"
                        cur.execute(
                            """
                            INSERT INTO business_metrics (
                                metric_id, metric_date, metric_type, brand,
                                region, hcp_id, trx_count, nrx_count,
                                total_rx_count, market_share, conversion_rate
                            ) VALUES (
                                %s, %s, 'per_hcp_rollup',
                                'Remibrutinib'::brand_type,
                                %s::region_type, %s, %s, %s, %s, 1.0, 0.5
                            )
                            ON CONFLICT (metric_id) DO NOTHING
                            """,
                            (
                                metric_id,
                                metric_date,
                                hcp["region"],
                                hcp["hcp_id"],
                                TRX_PER_HCP_PER_DAY,
                                NRX_PER_HCP_PER_DAY,
                                TRX_PER_HCP_PER_DAY + NRX_PER_HCP_PER_DAY,
                            ),
                        )

                # 3. Insert triggers. We want active_hcp_count == HCPS_PER_TERRITORY
                # for every (territory, metric_date) cell. To get every HCP
                # counted in the 30-day window ending on every metric_date in
                # the run window, give each HCP at least one trigger that
                # falls within 30 days of the EARLIEST metric_date and
                # within metric_date's inclusive upper bound for the LATEST.
                #
                # Simplest: one trigger per HCP at the earliest metric_date.
                # That trigger is within the 30-day backward-window of every
                # metric_date in the run.
                trigger_counter = 0
                trigger_ts = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
                for hcp in hcps:
                    trigger_counter += 1
                    trigger_id = f"tr_{test_run_id}_{trigger_counter:06d}"
                    # Use a real patient_id sentinel; FK is nullable in v3
                    # schema (per CREATE TABLE triggers).
                    patient_id = f"pat_{test_run_id}_{hcp['hcp_id']}"
                    cur.execute(
                        """
                        INSERT INTO triggers (
                            trigger_id, patient_id, hcp_id,
                            trigger_timestamp, trigger_type, brand_id
                        ) VALUES (%s, %s, %s, %s, 'engagement_gap', 'UNKNOWN')
                        ON CONFLICT (trigger_id) DO NOTHING
                        """,
                        (trigger_id, patient_id, hcp["hcp_id"], trigger_ts),
                    )

        yield {
            "test_run_id": test_run_id,
            "hcps": hcps,
            "expected": expected_per_territory,
            "start_date": start_dt,
            "end_date": end_dt,
            "num_days": NUM_DAYS,
            "base_date": base_date,
        }
    finally:
        # Teardown: delete in reverse FK order. Wrapped in try/finally so an
        # interrupted pytest still leaves the DB clean (mirrors I4
        # carry-over from 6B-infra-2a/2b).
        with db_conn:
            with db_conn.cursor() as cur:
                # territory_metrics keyed by (territory_id, metric_date) --
                # filter on territory_id pattern.
                cur.execute(
                    "DELETE FROM territory_metrics WHERE territory_id LIKE %s",
                    (f"%_{test_run_id}",),
                )
                # business_metrics has hcp_id FK with ON DELETE SET NULL,
                # but we want the rows gone. Filter by metric_id prefix.
                cur.execute(
                    "DELETE FROM business_metrics WHERE metric_id LIKE %s",
                    (f"bm_{test_run_id}_%",),
                )
                cur.execute(
                    "DELETE FROM triggers WHERE trigger_id LIKE %s",
                    (f"tr_{test_run_id}_%",),
                )
                cur.execute(
                    "DELETE FROM hcp_profiles WHERE hcp_id LIKE %s",
                    (f"hcp_{test_run_id}_%",),
                )


def _fetch_territory_rollup(db_conn: Any, test_run_id: str) -> list[tuple[Any, ...]]:
    """Read back territory_metrics rows for the test run, ordered by date."""
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT territory_id, metric_date, total_trx, total_nrx,
                   active_hcp_count, covered_lives,
                   market_potential, resource_allocation_score
              FROM territory_metrics
             WHERE territory_id LIKE %s
             ORDER BY territory_id, metric_date
            """,
            (f"%_{test_run_id}",),
        )
        return cur.fetchall()


def test_territorial_sums_match_per_hcp_sums(db_conn: Any, synthetic_dataset: dict) -> None:
    """Per the plan: assert territorial sums match per-HCP sums.

    For each (territory, metric_date) cell, total_trx must equal SUM(per-HCP
    trx_count) and total_nrx must equal SUM(per-HCP nrx_count). Drives off
    the synthetic fixture's deterministic per-day rates.
    """
    from src.etl.territory_metrics_etl import _run_territory_rollup_impl

    result = _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-territory-sums",
    )
    assert result["status"] == "completed", f"ETL failed: {result}"

    rows = _fetch_territory_rollup(db_conn, synthetic_dataset["test_run_id"])
    assert rows, "no territory_metrics rows materialised for the test run"

    # 3 territories x NUM_DAYS days = NUM_DAYS rows per territory.
    expected_total_rows = len(synthetic_dataset["expected"]) * synthetic_dataset["num_days"]
    assert len(rows) == expected_total_rows

    expected_per_territory = synthetic_dataset["expected"]
    for (
        territory_id,
        _metric_date,
        total_trx,
        total_nrx,
        _active_hcp_count,
        _covered_lives,
        _mp,
        _ras,
    ) in rows:
        exp = expected_per_territory[territory_id]
        assert total_trx == exp["total_trx_per_day"], (
            f"total_trx mismatch for {territory_id}: "
            f"expected {exp['total_trx_per_day']}, got {total_trx}"
        )
        assert total_nrx == exp["total_nrx_per_day"], (
            f"total_nrx mismatch for {territory_id}: "
            f"expected {exp['total_nrx_per_day']}, got {total_nrx}"
        )


def test_covered_lives_matches_total_patient_volume_sum(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """covered_lives = SUM(total_patient_volume) for HCPs in the territory.

    Time-invariant -- same value on every (territory, date) row.
    """
    from src.etl.territory_metrics_etl import _run_territory_rollup_impl

    _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-covered-lives",
    )

    rows = _fetch_territory_rollup(db_conn, synthetic_dataset["test_run_id"])
    expected_per_territory = synthetic_dataset["expected"]
    for (
        territory_id,
        _metric_date,
        _total_trx,
        _total_nrx,
        _active_hcp_count,
        covered_lives,
        _mp,
        _ras,
    ) in rows:
        exp = expected_per_territory[territory_id]
        assert covered_lives == exp["covered_lives"], (
            f"covered_lives mismatch for {territory_id}: "
            f"expected {exp['covered_lives']}, got {covered_lives}"
        )


def test_active_hcp_count_uses_30_day_window(db_conn: Any, synthetic_dataset: dict) -> None:
    """active_hcp_count = DISTINCT hcp_id with >= 1 trigger in the 30-day
    window ending on metric_date.

    The fixture seeds one trigger per HCP at base_date (June 1). For every
    metric_date in the run window (June 1, 2, 3) the trigger falls within
    the 30-day backward window, so active_hcp_count == HCPS_PER_TERRITORY
    on every (territory, date) row.
    """
    from src.etl.territory_metrics_etl import _run_territory_rollup_impl

    _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-active-hcp",
    )

    rows = _fetch_territory_rollup(db_conn, synthetic_dataset["test_run_id"])
    expected_per_territory = synthetic_dataset["expected"]
    for (
        territory_id,
        _metric_date,
        _total_trx,
        _total_nrx,
        active_hcp_count,
        _covered_lives,
        _mp,
        _ras,
    ) in rows:
        exp = expected_per_territory[territory_id]
        assert active_hcp_count == exp["active_hcp_count"], (
            f"active_hcp_count mismatch for {territory_id}: "
            f"expected {exp['active_hcp_count']}, got {active_hcp_count}"
        )


def test_market_potential_and_resource_score_preserved_across_etl(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """market_potential / resource_allocation_score must NOT be touched by
    the ETL.

    Migration 031 declares both as NOT NULL DEFAULT 0; the ETL omits them
    from the INSERT column list and from the ON CONFLICT SET clause. To
    prove the SET clause leaves them alone (independent of test execution
    order under xdist), we:

    1. Run the ETL once to materialise rows.
    2. Stamp distinctive non-default values (0.123 / 0.456) onto the rows.
    3. Re-run the ETL with the SAME window so it hits ON CONFLICT.
    4. Assert the stamped values are unchanged after the conflict path.

    The idempotency test below uses different stamp values (0.42 / 0.84)
    so the two tests don't depend on each other when run in any order.
    """
    from src.etl.territory_metrics_etl import _run_territory_rollup_impl

    # Step 1: materialise rows.
    result = _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-mp-ras-1",
    )
    assert result["status"] == "completed", f"ETL failed: {result}"

    # Step 2: stamp distinctive values on the test rows.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE territory_metrics
                   SET market_potential = 0.123,
                       resource_allocation_score = 0.456
                 WHERE territory_id LIKE %s
                """,
                (f"%_{synthetic_dataset['test_run_id']}",),
            )

    # Step 3: re-run ETL -- this hits the ON CONFLICT path on every row.
    _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-mp-ras-2",
    )

    # Step 4: stamps survived the ON CONFLICT SET clause (because the SET
    # clause omits market_potential / resource_allocation_score).
    rows = _fetch_territory_rollup(db_conn, synthetic_dataset["test_run_id"])
    for (
        _territory_id,
        _metric_date,
        _total_trx,
        _total_nrx,
        _active_hcp_count,
        _covered_lives,
        market_potential,
        resource_allocation_score,
    ) in rows:
        assert float(market_potential) == pytest.approx(0.123, abs=1e-9), (
            f"ON CONFLICT SET clause must NOT touch market_potential -- got {market_potential}"
        )
        assert float(resource_allocation_score) == pytest.approx(0.456, abs=1e-9), (
            "ON CONFLICT SET clause must NOT touch resource_allocation_score "
            f"-- got {resource_allocation_score}"
        )


def test_idempotent_rerun_preserves_market_potential_seed(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """Re-run idempotency, plus: market_potential / resource_allocation_score
    on existing rows must survive the ON CONFLICT SET clause unchanged.

    Run the ETL once, then artificially overwrite both columns to non-zero
    values for our test rows, then re-run the ETL. The four real aggregates
    should be unchanged (idempotency), and the two preserved columns should
    keep our injected values.
    """
    from src.etl.territory_metrics_etl import _run_territory_rollup_impl

    _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="idempotency-1",
    )

    first_snapshot = _fetch_territory_rollup(db_conn, synthetic_dataset["test_run_id"])
    assert first_snapshot, "first run produced no rows"

    # Stamp non-default market_potential / resource_allocation_score on the
    # test rows so the assertion that they survive UPDATE is not vacuous.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE territory_metrics
                   SET market_potential = 0.42,
                       resource_allocation_score = 0.84
                 WHERE territory_id LIKE %s
                """,
                (f"%_{synthetic_dataset['test_run_id']}",),
            )

    _run_territory_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="idempotency-2",
    )

    second_snapshot = _fetch_territory_rollup(db_conn, synthetic_dataset["test_run_id"])

    # Same number of rows, same per-row real-aggregate values.
    assert len(first_snapshot) == len(second_snapshot)
    for first_row, second_row in zip(first_snapshot, second_snapshot, strict=True):
        # Compare the four real aggregates (indexes 0..5 are everything
        # except mp / ras).
        assert first_row[:6] == second_row[:6], (
            "second run changed real aggregates -- not idempotent"
        )

    # market_potential / resource_allocation_score on every row must be
    # the values we stamped, NOT zero.
    for (
        _territory_id,
        _metric_date,
        _total_trx,
        _total_nrx,
        _active_hcp_count,
        _covered_lives,
        market_potential,
        resource_allocation_score,
    ) in second_snapshot:
        assert float(market_potential) == pytest.approx(0.42, abs=1e-9), (
            f"ON CONFLICT SET clause must NOT touch market_potential -- got {market_potential}"
        )
        assert float(resource_allocation_score) == pytest.approx(0.84, abs=1e-9), (
            "ON CONFLICT SET clause must NOT touch resource_allocation_score "
            f"-- got {resource_allocation_score}"
        )
