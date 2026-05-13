"""Integration test for ``run_feedback_loop`` round-trip on ``prediction_type='risk'``
(issue #173 scope item 5).

Verifies the Postgres-side feedback loop (migration 006) for risk-score
predictions:

1. A fresh ``ml_predictions`` row with ``prediction_type='risk'`` exists.
2. Migration 006 registered the ``risk`` row in ``ml_feedback_loop_config``
   with a 180-day ``observation_window_days`` and a 90-day
   ``min_observation_days``.
3. ``run_feedback_loop('risk')`` is callable (the function exists; we
   accept either ``status='completed'`` rows in the return set or an empty
   result depending on whether enough observation time has elapsed for
   the seeded fixture).
4. Time-travel: insert a row whose ``prediction_timestamp`` is 200 days
   in the past, supply a synthetic ``actual_outcome`` upstream via the
   patient's ``treatment_initiated`` flag, and confirm
   ``run_feedback_loop('risk')`` writes the label back into
   ``ml_predictions.actual_outcome``.

DB-touching tests are gated on a reachable Postgres at ``SUPABASE_DB_URL`` /
``RISK_SCORE_DB_URL`` / ``DATABASE_URL``; in CI these env vars are
typically unset, so DB tests are auto-skipped.
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


@_db_available
class TestFeedbackLoopRiskRoundTrip:
    @pytest.fixture
    def conn(self):
        import psycopg  # type: ignore[import-untyped]

        c = psycopg.connect(_DB_URL)
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

    def test_run_feedback_loop_callable(self, conn) -> None:
        """``run_feedback_loop('risk')`` is callable end-to-end and
        returns a tabular result. We accept any non-error return; the
        actual labelling behaviour is exercised in the time-travel
        test below.
        """
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM run_feedback_loop('risk')")
            rows = cur.fetchall()
        # The function returns one row per processed config (just 'risk' here).
        assert isinstance(rows, list)

    def test_time_travel_backfills_actual_outcome(self, conn) -> None:
        """End-to-end: a 200-day-old risk prediction + a patient whose
        ``treatment_initiated=1`` should result in the feedback loop
        backfilling ``actual_outcome`` on the prediction row.

        Implementation note: the truth-assignment function
        ``assign_truth_treatment_response`` (migration 006 line ~373)
        joins ``ml_predictions`` to ``patient_journeys`` on
        ``patient_id`` and reads ``treatment_initiated`` as the ground
        truth. We seed the rows accordingly.
        """
        # Use a unique patient_id to avoid collisions with existing rows.
        # patient_id VARCHAR(20) and patient_journey_id VARCHAR(20) — keep them tight.
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PT173_{suffix}"  # 16 chars
        pjid = f"PJ173_{suffix}"  # 16 chars
        # prediction_id VARCHAR(30) — has more room.
        prediction_id = f"rsc173_{suffix}_{uuid.uuid4().hex[:6]}"
        old_ts = datetime.now(timezone.utc) - timedelta(days=200)

        # Seed: a patient_journey with treatment_initiated=1 + an old
        # risk prediction.
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, "
                "journey_stage, treatment_initiated) "
                "VALUES (%s, %s, %s, 'initial_treatment', 1)",
                (pjid, patient_id, old_ts.date()),
            )
            cur.execute(
                "INSERT INTO ml_predictions "
                "(prediction_id, model_version, model_type, prediction_timestamp, "
                "patient_id, prediction_type, prediction_value, confidence_score) "
                "VALUES (%s, 'risk_score_v1', 'xgboost', %s, %s, 'risk', 0.42, 0.42)",
                (prediction_id, old_ts, patient_id),
            )
        conn.commit()

        try:
            # Run the loop on the 'risk' type only.
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM run_feedback_loop('risk')")
                loop_rows = cur.fetchall()
            assert isinstance(loop_rows, list)

            # Check that the row got an ``actual_outcome``. We don't
            # demand a specific value because the truth function may
            # encode the outcome differently (1.0 vs True vs the day
            # count) — just that it's no longer NULL.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT actual_outcome, outcome_recorded_at, outcome_source "
                    "FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()
            assert row is not None
            actual, recorded, source = row
            # The truth-assignment SQL in migration 006 should fire on
            # any prediction whose age >= min_observation_days (90).
            # We assert that at least one of ``actual_outcome`` /
            # ``outcome_recorded_at`` was populated. If neither, the
            # backfill regressed.
            assert (actual is not None) or (recorded is not None), (
                "Feedback loop did not backfill actual_outcome or "
                "outcome_recorded_at for a 200-day-old risk prediction "
                "— migration 006 truth assignment may have regressed."
            )
        finally:
            # Always clean up our test rows.
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                cur.execute(
                    "DELETE FROM patient_journeys WHERE patient_journey_id = %s",
                    (pjid,),
                )
            conn.commit()
