"""Integration tests for ``src.etl.business_metrics_per_hcp_etl``.

Exercises the full ETL against a real PostgreSQL instance reached via
``SUPABASE_DB_URL``. Tests are skipped when the env var is unset so CI
unit-only runs stay green.

Synthetic dataset
-----------------
The tests stand up:

* 5 HCPs across two territories (T_NE and T_S), assigned to ``northeast``
  and ``south`` regions.
* 3 brands from the ``brand_type`` enum (``Remibrutinib``, ``Fabhalta``,
  ``Kisqali``).
* 30 days of patient_journeys (one per (patient, brand)) and triggers
  (multiple per (HCP, brand, day)) so each cell of the rollup has data.
* All rows are isolated by a unique ``test_run_id`` prefix on every
  primary key + the ``metric_id`` so we can clean up deterministically.

Assertions
----------

* Per-HCP rollup rows materialise (one row per (hcp_id, brand,
  metric_date) cell that has triggers).
* ``SUM(market_share) ≈ 1.0`` within each (territory_id, brand,
  metric_date) — within 1e-6 float tolerance.
* Re-running the ETL is a no-op on row count (idempotency via metric_id).

Run gate
--------
Two env vars are required:

* ``SUPABASE_DB_URL`` — Postgres URL pointing at a DB that has migration
  033 applied.
* ``E2I_DB_INTEGRATION=1`` — explicit opt-in. Mirrors the
  ``FEAST_INTEGRATION`` pattern in ``tests/integration/test_feast_*``: the
  ``.env`` file ships ``SUPABASE_DB_URL`` for ops scripts, so we require
  a second flag to actually exercise it from pytest. Without it the
  entire module is skipped.
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

# Module-level skip: developers must opt in AND have a reachable Postgres URL
# before the suite executes. This stops `pytest tests/integration/` from
# trying to reach prod Supabase when the .env happens to ship a connection
# string.
pytestmark = pytest.mark.skipif(
    not (os.getenv("SUPABASE_DB_URL") and os.getenv("E2I_DB_INTEGRATION") == "1"),
    reason=(
        "SUPABASE_DB_URL and/or E2I_DB_INTEGRATION not set; integration "
        "test requires real Postgres + explicit opt-in. Run with "
        "E2I_DB_INTEGRATION=1 once your DB has migration 033 applied."
    ),
)

# Test config
TERRITORIES: tuple[tuple[str, str], ...] = (
    ("T_NE", "northeast"),
    ("T_S", "south"),
)
BRANDS: tuple[str, ...] = ("Remibrutinib", "Fabhalta", "Kisqali")
NUM_HCPS: int = 5
NUM_DAYS: int = 30
TRIGGERS_PER_HCP_BRAND_DAY: int = 3


@pytest.fixture(scope="module")
def db_conn() -> Any:
    """Open a single psycopg2 connection for the module's tests."""
    import psycopg2  # local import: optional in unit-only environments

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def test_run_id() -> str:
    """Unique prefix per pytest run so parallel suites do not collide."""
    return uuid.uuid4().hex[:10]


@pytest.fixture(scope="module")
def synthetic_dataset(db_conn: Any, test_run_id: str) -> dict:
    """Insert HCPs, patient_journeys, and triggers; tear down at module end.

    Returns a dict with the IDs and the ``(start_date, end_date)`` window
    spanning the synthetic data. Cleanup runs in reverse FK order so
    SET NULL on hcp_id (per migration 033) is respected.
    """
    base_date = date(2024, 1, 1)
    end_dt = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=NUM_DAYS)
    start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)

    hcps = []
    for terr_idx, (territory_id, region) in enumerate(TERRITORIES):
        # Spread HCPs across territories: first three in T_NE, last two in T_S.
        bucket_size = NUM_HCPS // len(TERRITORIES)
        if terr_idx == 0:
            ids = list(range(NUM_HCPS - bucket_size))
        else:
            ids = list(range(NUM_HCPS - bucket_size, NUM_HCPS))
        for i in ids:
            hcps.append(
                {
                    "hcp_id": f"hcp_{test_run_id}_{i:02d}",
                    "territory_id": f"{territory_id}_{test_run_id}",
                    "region": region,
                }
            )

    with db_conn:
        with db_conn.cursor() as cur:
            # Insert HCPs.
            for hcp in hcps:
                cur.execute(
                    """
                    INSERT INTO hcp_profiles (
                        hcp_id, territory_id, geographic_region, sales_rep_id
                    ) VALUES (%s, %s, %s::region_type, NULL)
                    ON CONFLICT (hcp_id) DO NOTHING
                    """,
                    (hcp["hcp_id"], hcp["territory_id"], hcp["region"]),
                )

            # Insert patient_journeys: one journey per (patient, brand) on
            # base_date so it is "before" every trigger we generate later.
            for hcp in hcps:
                for brand in BRANDS:
                    patient_id = f"pat_{test_run_id}_{hcp['hcp_id']}_{brand}"
                    journey_id = f"pj_{test_run_id}_{hcp['hcp_id']}_{brand}"
                    cur.execute(
                        """
                        INSERT INTO patient_journeys (
                            patient_journey_id, patient_id, journey_start_date,
                            journey_stage, journey_status, brand,
                            geographic_region, hcp_id
                        ) VALUES (
                            %s, %s, %s, 'diagnosis'::journey_stage_type,
                            'active'::journey_status_type, %s::brand_type,
                            %s::region_type, %s
                        )
                        ON CONFLICT (patient_journey_id) DO NOTHING
                        """,
                        (
                            journey_id,
                            patient_id,
                            base_date,
                            brand,
                            hcp["region"],
                            hcp["hcp_id"],
                        ),
                    )

            # Insert triggers: 3 per (HCP, brand, day) for NUM_DAYS days.
            # Mix delivery_status / acceptance_status so trx, nrx, conversion
            # are nonzero.
            counter = 0
            for day_offset in range(NUM_DAYS):
                ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(
                    days=day_offset
                )
                for hcp in hcps:
                    for brand in BRANDS:
                        patient_id = f"pat_{test_run_id}_{hcp['hcp_id']}_{brand}"
                        for k in range(TRIGGERS_PER_HCP_BRAND_DAY):
                            counter += 1
                            trigger_id = f"tr_{test_run_id}_{counter:06d}"
                            # Make k=0 not delivered, k=1 delivered+responded,
                            # k=2 delivered+pending so trx_count and nrx_count
                            # diverge.
                            delivery = "delivered" if k > 0 else "pending"
                            acceptance = "responded" if k == 1 else "pending"
                            cur.execute(
                                """
                                INSERT INTO triggers (
                                    trigger_id, patient_id, hcp_id,
                                    trigger_timestamp, brand_id,
                                    delivery_status, acceptance_status,
                                    delivery_timestamp, acceptance_timestamp
                                ) VALUES (
                                    %s, %s, %s, %s, 'UNKNOWN', %s, %s, %s, %s
                                )
                                ON CONFLICT (trigger_id) DO NOTHING
                                """,
                                (
                                    trigger_id,
                                    patient_id,
                                    hcp["hcp_id"],
                                    ts,
                                    delivery,
                                    acceptance,
                                    ts if delivery == "delivered" else None,
                                    ts if acceptance == "responded" else None,
                                ),
                            )

    yield {
        "test_run_id": test_run_id,
        "hcps": hcps,
        "start_date": start_dt,
        "end_date": end_dt,
    }

    # Teardown: delete in reverse FK order. business_metrics first (its
    # hcp_id FK has ON DELETE SET NULL but we want the rows gone), then
    # triggers, patient_journeys, hcp_profiles.
    with db_conn:
        with db_conn.cursor() as cur:
            cur.execute(
                "DELETE FROM business_metrics WHERE metric_id LIKE %s",
                (f"hcp_rollup_hcp_{test_run_id}_%",),
            )
            cur.execute(
                "DELETE FROM triggers WHERE trigger_id LIKE %s",
                (f"tr_{test_run_id}_%",),
            )
            cur.execute(
                "DELETE FROM patient_journeys WHERE patient_journey_id LIKE %s",
                (f"pj_{test_run_id}_%",),
            )
            cur.execute(
                "DELETE FROM hcp_profiles WHERE hcp_id LIKE %s",
                (f"hcp_{test_run_id}_%",),
            )


def test_per_hcp_rollup_materialises_rows(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """ETL runs cleanly and produces at least one row per (HCP, brand, day)."""
    from src.etl.business_metrics_per_hcp_etl import _run_per_hcp_rollup_impl

    # Use the implementation helper directly so we bypass the Celery
    # wrapper / broker; the wrapper is exercised in the unit tests.
    result = _run_per_hcp_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test",
    )

    assert result["status"] == "completed", f"ETL failed: {result}"
    # Lower bound: NUM_HCPS x NUM_BRANDS x NUM_DAYS rows. Other tests in the
    # DB may also write rollups; we use >= for safety.
    assert result["rows_affected"] >= NUM_HCPS * len(BRANDS) * NUM_DAYS

    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM business_metrics
             WHERE metric_id LIKE %s
            """,
            (f"hcp_rollup_hcp_{synthetic_dataset['test_run_id']}_%",),
        )
        row = cur.fetchone()
        count = row[0] if row else 0

    assert count == NUM_HCPS * len(BRANDS) * NUM_DAYS


def test_market_share_sums_to_one_per_territory(
    db_conn: Any, synthetic_dataset: dict
) -> None:
    """Within each (territory_id, brand, metric_date) the rollup
    market_shares add up to 1.0 (modulo float epsilon)."""
    from src.etl.business_metrics_per_hcp_etl import _run_per_hcp_rollup_impl

    _run_per_hcp_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test",
    )

    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT hp.territory_id, bm.brand, bm.metric_date,
                   SUM(bm.market_share)
              FROM business_metrics bm
              JOIN hcp_profiles      hp ON bm.hcp_id = hp.hcp_id
             WHERE bm.metric_id LIKE %s
             GROUP BY hp.territory_id, bm.brand, bm.metric_date
            """,
            (f"hcp_rollup_hcp_{synthetic_dataset['test_run_id']}_%",),
        )
        groups = cur.fetchall()

    assert groups, "no groups found"
    for territory_id, brand, metric_date, total in groups:
        assert total == pytest.approx(
            1.0, abs=1e-6
        ), f"share sum != 1.0 for ({territory_id}, {brand}, {metric_date}): {total}"


def test_idempotent_rerun(db_conn: Any, synthetic_dataset: dict) -> None:
    """Running the ETL twice produces the same row count (no duplicates)."""
    from src.etl.business_metrics_per_hcp_etl import _run_per_hcp_rollup_impl

    _run_per_hcp_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-1",
    )

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM business_metrics WHERE metric_id LIKE %s",
            (f"hcp_rollup_hcp_{synthetic_dataset['test_run_id']}_%",),
        )
        row = cur.fetchone()
        first_count = row[0] if row else 0

    _run_per_hcp_rollup_impl(
        start_date=synthetic_dataset["start_date"].isoformat(),
        end_date=synthetic_dataset["end_date"].isoformat(),
        request_id="integration-test-2",
    )

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM business_metrics WHERE metric_id LIKE %s",
            (f"hcp_rollup_hcp_{synthetic_dataset['test_run_id']}_%",),
        )
        row = cur.fetchone()
        second_count = row[0] if row else 0

    assert first_count == second_count, "second run created duplicates"
    assert first_count > 0
