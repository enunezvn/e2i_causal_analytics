"""Integration tests for ``run_feedback_loop`` on ``prediction_type='risk'``
(issue #173 scope item 5).

Verifies the Postgres-side feedback loop (migration 006) for risk-score
predictions:

    1. Migration 006 registered a ``risk`` row in
       ``ml_feedback_loop_config`` with 180-day ``observation_window_days``
       and 90-day ``min_observation_days``.

    2. ``run_feedback_loop('risk')`` is callable end-to-end and returns
       a tabular result.

    3. Time-travel POSITIVE: a 200-day-old ``PENDING`` risk prediction
       PLUS 6 monthly prescription fills (PDC = 1.0 ≥ 0.80) backfills
       ``actual_outcome = 1.0`` AND ``outcome_label = 'POSITIVE'`` AND a
       non-null ``truth_source``.

    4. Time-travel NEGATIVE: a 200-day-old ``PENDING`` risk prediction
       with NO fills backfills ``actual_outcome = 0.0`` AND
       ``outcome_label = 'NEGATIVE'``.

Isolation (codex pass-2 HIGH-2): time-travel tests open an explicit
transaction with ``conn.autocommit = False`` and ``conn.rollback()`` in
``finally``. The seeded rows AND any side effects of
``run_feedback_loop`` (which labels every old ``PENDING`` risk row in
the DB up to ``LIMIT 1000``) are rolled back together — no shared-DB
leakage, no flake risk when the DB has many old ``PENDING`` rows.

Brand-matching (codex pass-2 HIGH-1 + issue #176): migration 038
``CREATE OR REPLACE``-d all five truth-assignment functions without the
``p.brand`` SELECT or the ``te.brand::text = pc.brand::text`` joins;
migration 006 §2.4 was patched in place to drop ``brand`` from the two
historically-named ``idx_predictions_*_brand`` index keys (which had
prevented fresh-DB ``--single-transaction`` replay — codex pass-1
HIGH-1). The test still seeds ``treatment_events.brand='Fabhalta'`` (and
``ml_predictions.brand`` if the column happens to exist on a legacy DB)
as defense in depth. See
``test_truth_assignment_function_has_no_brand_reference`` (parametrized
across all 5 functions) and ``test_brand_indexes_have_no_brand_column``
for explicit regression pins on the post-038 state.

DB-touching tests are gated on a reachable Postgres at
``RISK_SCORE_DB_URL`` / ``SUPABASE_DB_URL`` / ``DATABASE_URL``; in CI
these env vars are typically unset, so DB tests are auto-skipped.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest


def _resolved_db_url() -> str | None:
    return (
        os.environ.get("RISK_SCORE_DB_URL")
        or os.environ.get("SUPABASE_DB_URL")
        or os.environ.get("DATABASE_URL")
    )


_DB_URL = _resolved_db_url()


def _db_reachable() -> bool:
    if not _DB_URL:
        return False
    try:
        import psycopg  # type: ignore[import-untyped]

        with psycopg.connect(_DB_URL, connect_timeout=2) as conn:  # noqa: F841
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


_db_available = pytest.mark.skipif(
    not _db_reachable(),
    reason=(
        "Postgres not reachable via RISK_SCORE_DB_URL / SUPABASE_DB_URL / "
        "DATABASE_URL — skipping feedback-loop integration tests."
    ),
)


def _ml_predictions_has_brand_column(conn) -> bool:
    """True if the ``ml_predictions`` table has a ``brand`` column on this DB.

    Migration 006 references ``p.brand`` in the risk truth function but
    core schema doesn't add it. We use this to decide whether to seed
    the column in our test rows.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'ml_predictions' AND column_name = 'brand'"
        )
        return cur.fetchone() is not None


@_db_available
class TestFeedbackLoopRiskRoundTrip:
    @pytest.fixture
    def conn(self):
        import psycopg  # type: ignore[import-untyped]

        c = psycopg.connect(_DB_URL)
        # Explicit-transaction mode for isolation across run_feedback_loop calls.
        c.autocommit = False
        yield c
        c.close()

    def test_risk_prediction_type_is_configured(self, conn) -> None:
        """Migration 006 should have registered a 'risk' row in the
        feedback-loop config table with the 180/90 day windows.
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT observation_window_days, min_observation_days, is_active "
                "FROM ml_feedback_loop_config WHERE prediction_type = 'risk'"
            )
            row = cur.fetchone()
        assert row is not None, (
            "ml_feedback_loop_config has no 'risk' row — migration 006 not applied?"
        )
        obs_win, min_obs, is_active = row
        assert obs_win == 180
        assert min_obs == 90
        assert is_active is True
        conn.rollback()  # read-only; nothing to commit

    @pytest.mark.parametrize(
        "fn_name",
        [
            "assign_truth_hcp_churn",
            "assign_truth_script_conversion",
            "assign_truth_treatment_response",
            "assign_truth_next_best_action",
            "assign_truth_market_share",
        ],
    )
    def test_truth_assignment_function_has_no_brand_reference(self, conn, fn_name) -> None:
        """Issue #176 — migration 038 stripped ``p.brand`` and the
        ``te.brand::text = pc.brand::text`` joins from all five
        truth-assignment function bodies. This regression test pins the
        post-038 state per function so a future hand-edit of migration
        006 / 038 cannot silently re-introduce the dead-on-arrival
        column reference.

        Codex pass-1 MEDIUM-3 (2026-05-13): parametrized across all five
        functions, not just ``assign_truth_treatment_response``.
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_get_functiondef(oid) FROM pg_proc WHERE proname = %s",
                (fn_name,),
            )
            row = cur.fetchone()
        assert row is not None, f"{fn_name} function not found — migration 006 not applied?"
        fn_src = row[0]
        assert "p.brand" not in fn_src, (
            f"{fn_name} still references p.brand — "
            "migration 038 (issue #176) not applied or was reverted."
        )
        assert "te.brand::text = pc.brand::text" not in fn_src, (
            f"{fn_name} still joins on te.brand::text = pc.brand::text — "
            "migration 038 (issue #176) not applied or was reverted."
        )
        conn.rollback()

    def test_brand_indexes_have_no_brand_column(self, conn) -> None:
        """Issue #176 — migration 006 originally keyed
        ``idx_predictions_hcp_brand`` / ``idx_predictions_patient_brand``
        on a nonexistent ``ml_predictions.brand`` column, which aborted
        the runner's ``--single-transaction`` replay on a fresh DB
        (codex pass-1 HIGH-1). The in-place fix to 006 §2.4 strips
        ``brand`` from the index keys.

        This test asserts that if either historically-named index
        exists, its ``indexdef`` does NOT mention ``brand``.
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE indexname IN "
                "('idx_predictions_hcp_brand', 'idx_predictions_patient_brand')"
            )
            rows = cur.fetchall()
        for name, indexdef in rows:
            assert "brand" not in indexdef, (
                f"Index {name!r} still references a brand column: "
                f"{indexdef!r} — migration 006 §2.4 fix (issue #176) "
                "not applied or was reverted."
            )
        conn.rollback()

    def test_run_feedback_loop_callable(self, conn) -> None:
        """``run_feedback_loop('risk')`` is callable end-to-end and
        returns a tabular result. Side effects rolled back so other
        tests are not affected.
        """
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM run_feedback_loop('risk')")
            rows = cur.fetchall()
        assert isinstance(rows, list)
        conn.rollback()

    def _isolate_other_pending_risk_rows(self, conn) -> None:
        """Temporarily move other eligible PENDING risk predictions out
        of the ``assign_truth_treatment_response`` candidate set so
        our test row is guaranteed to be selected.

        Codex pass-3 MEDIUM: migration 006's risk truth path uses
        ``LIMIT 1000`` with no ``ORDER BY`` (line 373). On a shared DB
        with > 1000 old eligible PENDING risk rows, ``run_feedback_loop``
        could skip our test row, causing the POSITIVE/NEGATIVE
        assertions to fail. We tag every other eligible row with the
        sentinel ``outcome_label='EXCLUDED'`` inside our transaction,
        then ``rollback`` in ``finally`` restores them — net-zero
        side effect on the production DB.

        Idempotent: rolled back along with everything else if any test
        step fails.
        """
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ml_predictions SET outcome_label = 'EXCLUDED' "
                "WHERE prediction_type = 'risk' "
                "AND outcome_label = 'PENDING' "
                "AND prediction_timestamp < NOW() - INTERVAL '180 days'"
            )

    def _seed_prediction_and_journey(
        self,
        conn,
        prediction_id: str,
        patient_id: str,
        pjid: str,
        old_ts: datetime,
        brand: str = "Fabhalta",
    ) -> None:
        """Seed a 200-day-old PENDING risk prediction + its patient journey.

        Sets ``brand`` on ``ml_predictions`` if the column exists on the
        target DB; otherwise omits it (the live installed function
        doesn't reference ``p.brand``).
        """
        has_brand = _ml_predictions_has_brand_column(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                "VALUES (%s, %s, %s, 'initial_treatment')",
                (pjid, patient_id, old_ts.date()),
            )
            if has_brand:
                cur.execute(
                    "INSERT INTO ml_predictions "
                    "(prediction_id, model_version, model_type, prediction_timestamp, "
                    "patient_id, prediction_type, prediction_value, confidence_score, "
                    "outcome_label, brand) "
                    "VALUES (%s, 'risk_score_v1', 'xgboost', %s, %s, 'risk', 0.42, "
                    "0.42, 'PENDING', %s)",
                    (prediction_id, old_ts, patient_id, brand),
                )
            else:
                cur.execute(
                    "INSERT INTO ml_predictions "
                    "(prediction_id, model_version, model_type, prediction_timestamp, "
                    "patient_id, prediction_type, prediction_value, confidence_score, "
                    "outcome_label) "
                    "VALUES (%s, 'risk_score_v1', 'xgboost', %s, %s, 'risk', 0.42, "
                    "0.42, 'PENDING')",
                    (prediction_id, old_ts, patient_id),
                )

    def test_time_travel_backfills_positive_outcome_from_treatment_events(self, conn) -> None:
        """A 200-day-old PENDING risk prediction PLUS 6 monthly
        prescription fills (PDC=1.0 ≥ 0.80) → ``actual_outcome=1.0`` AND
        ``outcome_label='POSITIVE'`` AND non-null ``truth_source``.

        Isolation (codex pass-2 HIGH-2): all writes including
        ``run_feedback_loop``'s side effects on other PENDING rows are
        rolled back in ``finally``. Brand-matching defense (codex
        pass-2 HIGH-1): seeds ``treatment_events.brand='Fabhalta'`` (and
        ``ml_predictions.brand`` if the column exists) so the test is
        robust against migration-006 brand-join drift.
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PT173_{suffix}"  # 16 chars
        pjid = f"PJ173_{suffix}"  # 16 chars
        prediction_id = f"rsc173_{suffix}_{uuid.uuid4().hex[:6]}"
        old_ts = datetime.now(timezone.utc) - timedelta(days=200)

        try:
            self._isolate_other_pending_risk_rows(conn)
            self._seed_prediction_and_journey(conn, prediction_id, patient_id, pjid, old_ts)
            with conn.cursor() as cur:
                # 6 monthly fills, 30-day supply each => 180 days covered
                # over the 180-day observation window => PDC=1.0 ≥ 0.80
                # threshold => POSITIVE label.
                for month in range(6):
                    fill_date = (old_ts + timedelta(days=month * 30)).date()
                    event_id = f"TE_{uuid.uuid4().hex[:14]}"
                    cur.execute(
                        "INSERT INTO treatment_events "
                        "(treatment_event_id, patient_journey_id, patient_id, "
                        "event_date, event_type, duration_days, brand) "
                        "VALUES (%s, %s, %s, %s, 'prescription', 30, 'Fabhalta')",
                        (event_id, pjid, patient_id, fill_date),
                    )

                cur.execute("SELECT * FROM run_feedback_loop('risk')")
                loop_rows = cur.fetchall()
                assert len(loop_rows) >= 1
                risk_row = next((r for r in loop_rows if r[0] == "risk"), None)
                assert risk_row is not None
                assert risk_row[1] == "COMPLETED"

                cur.execute(
                    "SELECT actual_outcome, outcome_label, truth_source, "
                    "       outcome_recorded_at "
                    "FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()

            assert row is not None
            actual, outcome_label, truth_source, outcome_recorded_at = row
            assert actual is not None, (
                f"actual_outcome was NOT backfilled for {prediction_id} "
                f"despite 6 monthly fills spanning the observation window. "
                f"loop_rows={loop_rows}"
            )
            assert float(actual) == pytest.approx(1.0, abs=1e-3), (
                f"Expected actual_outcome=1.0 (POSITIVE — PDC=1.0 ≥ 0.80) but got {actual}"
            )
            assert outcome_label == "POSITIVE", (
                f"Expected outcome_label='POSITIVE' but got {outcome_label!r}"
            )
            assert truth_source is not None
            assert outcome_recorded_at is not None
        finally:
            # Roll back EVERYTHING: our seeded rows + any side effects
            # of run_feedback_loop on other PENDING rows in the DB.
            conn.rollback()

    def test_time_travel_backfills_negative_outcome_no_fills(self, conn) -> None:
        """A 200-day-old PENDING risk prediction with NO prescription fills
        → ``actual_outcome=0.0`` AND ``outcome_label='NEGATIVE'``.

        Side effects rolled back (codex pass-2 HIGH-2).
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PT173n{suffix[:5]}"  # 13 chars
        pjid = f"PJ173n{suffix[:5]}"
        prediction_id = f"rscn173_{suffix}_{uuid.uuid4().hex[:5]}"
        old_ts = datetime.now(timezone.utc) - timedelta(days=200)

        try:
            self._isolate_other_pending_risk_rows(conn)
            self._seed_prediction_and_journey(conn, prediction_id, patient_id, pjid, old_ts)
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM run_feedback_loop('risk')")
                cur.fetchall()
                cur.execute(
                    "SELECT actual_outcome, outcome_label FROM ml_predictions "
                    "WHERE prediction_id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()
            assert row is not None
            actual, outcome_label = row
            assert actual is not None
            assert float(actual) == pytest.approx(0.0, abs=1e-3)
            assert outcome_label == "NEGATIVE"
        finally:
            conn.rollback()
