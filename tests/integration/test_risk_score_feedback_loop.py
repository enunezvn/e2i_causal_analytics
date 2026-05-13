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

    def test_time_travel_backfills_positive_outcome_from_treatment_events(self, conn) -> None:
        """Codex pass-1 HIGH-3: the previous version of this test seeded
        ``patient_journeys.treatment_initiated=1`` and accepted
        ``actual_outcome IS NOT NULL OR outcome_recorded_at IS NOT NULL``.
        That was a false-positive: the installed
        ``assign_truth_treatment_response`` function joins on
        ``treatment_events`` (prescription fills), not on
        ``patient_journeys.treatment_initiated``, and falls into
        ``NEGATIVE`` (outcome=0) when no fills exist — which would
        populate ``outcome_recorded_at`` regardless of whether the
        ground truth was right.

        Tightened version: seed a 200-day-old risk prediction in
        ``PENDING`` state PLUS three monthly prescription fill events
        spanning the observation window. This is the ground-truth
        scenario that should produce a ``POSITIVE`` label (PDC ≥ 0.80).
        We assert ``actual_outcome == 1.0`` AND ``outcome_label='POSITIVE'``
        AND a non-null ``truth_source``.
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PT173_{suffix}"  # 16 chars
        pjid = f"PJ173_{suffix}"  # 16 chars
        prediction_id = f"rsc173_{suffix}_{uuid.uuid4().hex[:6]}"
        old_ts = datetime.now(timezone.utc) - timedelta(days=200)

        # Seed (a) the patient_journey, (b) the old risk prediction in
        # PENDING state, (c) 6x monthly prescription fills with 30-day
        # supply spanning days 0..150 from the prediction.
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                "VALUES (%s, %s, %s, 'initial_treatment')",
                (pjid, patient_id, old_ts.date()),
            )
            cur.execute(
                "INSERT INTO ml_predictions "
                "(prediction_id, model_version, model_type, prediction_timestamp, "
                "patient_id, prediction_type, prediction_value, confidence_score, "
                "outcome_label) "
                "VALUES (%s, 'risk_score_v1', 'xgboost', %s, %s, 'risk', 0.42, 0.42, "
                "'PENDING')",
                (prediction_id, old_ts, patient_id),
            )
            # 6 monthly fills, 30-day supply each => 180 days covered out
            # of 180-day window => PDC=1.0, well above the default 0.80
            # threshold => POSITIVE label.
            for month in range(6):
                fill_date = (old_ts + timedelta(days=month * 30)).date()
                event_id = f"TE_{uuid.uuid4().hex[:14]}"
                cur.execute(
                    "INSERT INTO treatment_events "
                    "(treatment_event_id, patient_journey_id, patient_id, "
                    "event_date, event_type, duration_days) "
                    "VALUES (%s, %s, %s, %s, 'prescription', 30)",
                    (event_id, pjid, patient_id, fill_date),
                )
        conn.commit()

        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM run_feedback_loop('risk')")
                loop_rows = cur.fetchall()
            # The loop should report at least one labelled prediction
            # (ours). Migration 006 returns:
            #   (prediction_type, run_status, predictions_evaluated,
            #    predictions_labeled, predictions_excluded, duration_s)
            assert len(loop_rows) >= 1
            risk_row = next((r for r in loop_rows if r[0] == "risk"), None)
            assert risk_row is not None
            # predictions_labeled is the 4th column. We don't pin to ==1
            # because the global DB has many other PENDING risk rows that
            # may also have been labelled this run; we only need our
            # prediction's row to be labelled.
            assert risk_row[1] == "COMPLETED"

            with conn.cursor() as cur:
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
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM treatment_events WHERE patient_id = %s",
                    (patient_id,),
                )
                cur.execute(
                    "DELETE FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                cur.execute(
                    "DELETE FROM patient_journeys WHERE patient_journey_id = %s",
                    (pjid,),
                )
            conn.commit()

    def test_time_travel_backfills_negative_outcome_no_fills(self, conn) -> None:
        """Codex pass-1 HIGH-3 follow-on: the NEGATIVE branch must also
        be reachable. Seed a 200-day-old PENDING risk prediction with
        NO prescription fills — should label NEGATIVE (outcome=0).
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PT173n{suffix[:5]}"  # 13 chars
        pjid = f"PJ173n{suffix[:5]}"
        prediction_id = f"rscn173_{suffix}_{uuid.uuid4().hex[:5]}"
        old_ts = datetime.now(timezone.utc) - timedelta(days=200)

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                "VALUES (%s, %s, %s, 'initial_treatment')",
                (pjid, patient_id, old_ts.date()),
            )
            cur.execute(
                "INSERT INTO ml_predictions "
                "(prediction_id, model_version, model_type, prediction_timestamp, "
                "patient_id, prediction_type, prediction_value, confidence_score, "
                "outcome_label) "
                "VALUES (%s, 'risk_score_v1', 'xgboost', %s, %s, 'risk', 0.42, 0.42, "
                "'PENDING')",
                (prediction_id, old_ts, patient_id),
            )
        conn.commit()

        try:
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
