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

        Codex pass-2 MEDIUM-2: assert the indexed COLUMN list specifically
        excludes ``brand``, not the whole ``indexdef`` (which embeds the
        index name and would false-positive on the historical
        ``idx_predictions_*_brand`` token).
        """
        # Inspect the indexed columns via pg_index / pg_attribute, not
        # the indexdef text. ``pg_index.indkey`` is an int2vector of
        # attribute numbers; ``pg_attribute`` resolves them to names.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT i.relname AS indexname, "
                "       array_agg(a.attname ORDER BY x.ord) AS columns "
                "FROM pg_index ix "
                "JOIN pg_class i ON i.oid = ix.indexrelid "
                "JOIN pg_class t ON t.oid = ix.indrelid "
                "JOIN unnest(ix.indkey) WITH ORDINALITY AS x(attnum, ord) "
                "  ON TRUE "
                "JOIN pg_attribute a "
                "  ON a.attrelid = t.oid AND a.attnum = x.attnum "
                "WHERE i.relname IN "
                "  ('idx_predictions_hcp_brand', "
                "   'idx_predictions_patient_brand') "
                "GROUP BY i.relname"
            )
            rows = cur.fetchall()
        for indexname, columns in rows:
            assert "brand" not in columns, (
                f"Index {indexname!r} key columns still include 'brand': "
                f"{columns!r} — migration 006 §2.4 fix (issue #176) "
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

    @pytest.mark.parametrize(
        "prediction_type,truth_fn",
        [
            ("trigger", "assign_truth_script_conversion"),
            ("next_best_action", "assign_truth_next_best_action"),
        ],
    )
    def test_run_feedback_loop_trigger_and_nba_callable_post_039(
        self, conn, prediction_type, truth_fn
    ) -> None:
        """Issue #182 — migration 039 stripped ``LEFT JOIN triggers t ON
        t.prediction_id = p.prediction_id`` plus the ``t.trigger_id`` /
        ``t.status`` / ``t.trigger_status`` references from
        ``assign_truth_script_conversion`` and
        ``assign_truth_next_best_action``. Pre-039, every execution of
        either function (reachable via ``run_feedback_loop('trigger')``
        and ``run_feedback_loop('next_best_action')`` — wired into the
        4-hourly Celery beat schedule at
        ``src/tasks/feedback_loop_tasks.py:40``) raises
        ``column t.prediction_id does not exist`` at plan time because
        the real ``triggers`` schema has neither column.

        This test pins the post-039 callable state for both functions.
        We call ``run_feedback_loop(...)`` rather than the underlying
        truth-assignment helpers directly so the orchestrator's CASE
        dispatch (006 §6 lines 704-714) is exercised too.

        Side effects rolled back so this test does not interact with
        the time-travel tests below.
        """
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_proc WHERE proname = %s", (truth_fn,))
            assert cur.fetchone() is not None, (
                f"{truth_fn} not installed — migration 006/038/039 may be missing"
            )
            cur.execute("SELECT * FROM run_feedback_loop(%s)", (prediction_type,))
            rows = cur.fetchall()
        assert isinstance(rows, list)
        # ``run_feedback_loop`` returns one row per active
        # ml_feedback_loop_config entry matching the filter. The
        # 'trigger' and 'next_best_action' rows are both is_active=true
        # per migration 006 §7 seeding.
        assert len(rows) >= 1, (
            f"run_feedback_loop({prediction_type!r}) returned no rows — "
            "ml_feedback_loop_config row missing or inactive"
        )
        type_row = next((r for r in rows if r[0] == prediction_type), None)
        assert type_row is not None, (
            f"No {prediction_type!r} row in run_feedback_loop output: {rows!r}"
        )
        # 006 §6: COMPLETED if no exception, FAILED otherwise. Pre-039
        # would be FAILED with the t.prediction_id schema-drift error.
        assert type_row[1] == "COMPLETED", (
            f"run_feedback_loop({prediction_type!r}) status = {type_row[1]!r} "
            "(expected COMPLETED). Pre-039 this raises on the "
            f"LEFT JOIN triggers schema drift. Full row: {type_row!r}"
        )
        conn.rollback()

    @pytest.mark.parametrize(
        "fn_name",
        ["assign_truth_script_conversion", "assign_truth_next_best_action"],
    )
    def test_truth_assignment_function_has_no_triggers_join(self, conn, fn_name) -> None:
        """Issue #182 — migration 039 must strip the ``LEFT JOIN triggers``
        clause and every ``t.<column>`` reference (``t.prediction_id``,
        ``t.status``, ``t.trigger_id``, ``t.trigger_status``) from
        ``assign_truth_script_conversion`` and
        ``assign_truth_next_best_action``. The real ``triggers`` table has
        no ``prediction_id`` / ``status`` columns and no FK back to
        ``ml_predictions`` (see ``database/core/e2i_ml_complete_v3_schema.sql``
        §3.6 lines 579-619 — real status columns are ``delivery_status``
        and ``acceptance_status``).
        """
        import re

        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_get_functiondef(oid) FROM pg_proc WHERE proname = %s",
                (fn_name,),
            )
            row = cur.fetchone()
        assert row is not None, f"{fn_name} function not found — migration 006/038/039 not applied?"
        fn_src_lower = row[0].lower()
        # Codex pass-1 LOW-2: normalize whitespace and match schema-qualified
        # / quoted variants so a future hand-edit using `JOIN public.triggers`,
        # `JOIN "triggers"`, or a newline between JOIN and the relation name
        # cannot bypass the regression pin. The post-039 bodies contain zero
        # references to the ``triggers`` relation under any form.
        fn_src_collapsed = re.sub(r"\s+", " ", fn_src_lower)
        triggers_join_re = re.compile(r"\bjoin\s+(?:public\s*\.\s*)?\"?triggers\"?\b")
        assert not triggers_join_re.search(fn_src_collapsed), (
            f"{fn_name} still joins on the `triggers` table — migration 039 "
            "(issue #182) not applied or was reverted."
        )
        # Direct token assertions for the specific column drifts that
        # caused the runtime failure pre-039.
        for tok in ("t.prediction_id", "t.status", "t.trigger_id", "t.trigger_status"):
            assert tok not in fn_src_lower, (
                f"{fn_name} still references `{tok}` — migration 039 "
                "(issue #182) not applied or was reverted."
            )
        conn.rollback()

    def test_assign_truth_script_conversion_labels_positive_on_prescription(self, conn) -> None:
        """Issue #182 acceptance criterion 2 — integration test exercises
        ``assign_truth_script_conversion`` via the
        ``run_feedback_loop('trigger')`` caller path. Seeds an old
        PENDING 'trigger' prediction + a single in-window prescription
        for the same HCP, then asserts the truth-backfill produces
        POSITIVE with ``truth_source='treatment_events'`` (the post-039
        value, ex-``triggers_treatment_events``).

        Isolation: every other PENDING 'trigger' row is moved to
        EXCLUDED inside the transaction so our seeded row wins the
        LIMIT 1000 race; everything rolls back in `finally`.
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PT182_{suffix}"
        pjid = f"PJ182_{suffix}"
        prediction_id = f"trg182_{suffix}_{uuid.uuid4().hex[:6]}"
        hcp_id = f"HCP182_{suffix[:8]}"
        # observation_window_days = 21 per 006 §7 seeding for 'trigger'
        old_ts = datetime.now(timezone.utc) - timedelta(days=60)

        try:
            with conn.cursor() as cur:
                # Isolate other eligible PENDING 'trigger' rows.
                cur.execute(
                    "UPDATE ml_predictions SET outcome_label = 'EXCLUDED' "
                    "WHERE prediction_type = 'trigger' "
                    "AND outcome_label = 'PENDING' "
                    "AND prediction_timestamp < NOW() - INTERVAL '21 days' "
                    "AND prediction_id <> %s",
                    (prediction_id,),
                )

                # Seed HCP, patient journey, prediction, prescription event.
                cur.execute(
                    "INSERT INTO hcp_profiles (hcp_id, first_name, last_name, "
                    "specialty, npi) VALUES (%s, 'Test', 'HCP', 'Dermatology', %s) "
                    "ON CONFLICT (hcp_id) DO NOTHING",
                    (hcp_id, f"99{suffix[:8]}"),
                )
                cur.execute(
                    "INSERT INTO patient_journeys "
                    "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                    "VALUES (%s, %s, %s, 'initial_treatment')",
                    (pjid, patient_id, old_ts.date()),
                )
                cur.execute(
                    "INSERT INTO ml_predictions "
                    "(prediction_id, model_version, model_type, "
                    "prediction_timestamp, patient_id, hcp_id, "
                    "prediction_type, prediction_value, confidence_score, "
                    "outcome_label) "
                    "VALUES (%s, 'trigger_v1', 'xgboost', %s, %s, %s, "
                    "'trigger', 0.50, 0.50, 'PENDING')",
                    (prediction_id, old_ts, patient_id, hcp_id),
                )
                # In-window prescription (post-prediction, within 21 days).
                fill_date = (old_ts + timedelta(days=5)).date()
                event_id = f"TE_{uuid.uuid4().hex[:14]}"
                cur.execute(
                    "INSERT INTO treatment_events "
                    "(treatment_event_id, patient_journey_id, patient_id, hcp_id, "
                    "event_date, event_type, duration_days, brand) "
                    "VALUES (%s, %s, %s, %s, %s, 'prescription', 30, 'Fabhalta')",
                    (event_id, pjid, patient_id, hcp_id, fill_date),
                )

                cur.execute("SELECT * FROM run_feedback_loop('trigger')")
                loop_rows = cur.fetchall()
                trigger_row = next((r for r in loop_rows if r[0] == "trigger"), None)
                assert trigger_row is not None
                assert trigger_row[1] == "COMPLETED", (
                    f"run_feedback_loop('trigger') failed: {trigger_row!r}"
                )

                cur.execute(
                    "SELECT actual_outcome, outcome_label, truth_source "
                    "FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()

            assert row is not None
            actual, outcome_label, truth_source = row
            assert actual is not None, (
                "actual_outcome was not backfilled — script_conversion "
                "truth-assignment didn't pick up the seeded prediction. "
                f"loop_rows={loop_rows!r}"
            )
            assert float(actual) == pytest.approx(1.0, abs=1e-3)
            assert outcome_label == "POSITIVE"
            # Post-039: truth_source is 'treatment_events' (was
            # 'treatment_events' on the live droplet too — never
            # 'triggers_treatment_events' for script_conversion; that
            # literal applied only to next_best_action pre-039).
            assert truth_source == "treatment_events", (
                f"Expected truth_source='treatment_events' but got {truth_source!r}"
            )
        finally:
            conn.rollback()

    def test_assign_truth_next_best_action_labels_negative_no_activity(self, conn) -> None:
        """Issue #182 acceptance criterion 2 — integration test exercises
        ``assign_truth_next_best_action`` via ``run_feedback_loop('next_best_action')``.
        Seeds an old PENDING 'next_best_action' prediction with NO
        downstream treatment_events; asserts NEGATIVE with
        ``truth_source='treatment_events'`` (post-039 value;
        ex-``triggers_treatment_events`` pre-039).
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PTN182_{suffix[:9]}"
        pjid = f"PJN182_{suffix[:9]}"
        prediction_id = f"nba182_{suffix}_{uuid.uuid4().hex[:6]}"
        hcp_id = f"HCN182_{suffix[:8]}"
        # observation_window_days = 30 per 006 §7 seeding for 'next_best_action'
        old_ts = datetime.now(timezone.utc) - timedelta(days=60)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ml_predictions SET outcome_label = 'EXCLUDED' "
                    "WHERE prediction_type = 'next_best_action' "
                    "AND outcome_label = 'PENDING' "
                    "AND prediction_timestamp < NOW() - INTERVAL '30 days' "
                    "AND prediction_id <> %s",
                    (prediction_id,),
                )

                cur.execute(
                    "INSERT INTO hcp_profiles (hcp_id, first_name, last_name, "
                    "specialty, npi) VALUES (%s, 'Test', 'HCP', 'Dermatology', %s) "
                    "ON CONFLICT (hcp_id) DO NOTHING",
                    (hcp_id, f"98{suffix[:8]}"),
                )
                cur.execute(
                    "INSERT INTO patient_journeys "
                    "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                    "VALUES (%s, %s, %s, 'initial_treatment')",
                    (pjid, patient_id, old_ts.date()),
                )
                cur.execute(
                    "INSERT INTO ml_predictions "
                    "(prediction_id, model_version, model_type, "
                    "prediction_timestamp, patient_id, hcp_id, "
                    "prediction_type, prediction_value, confidence_score, "
                    "outcome_label) "
                    "VALUES (%s, 'nba_v1', 'xgboost', %s, %s, %s, "
                    "'next_best_action', 0.30, 0.30, 'PENDING')",
                    (prediction_id, old_ts, patient_id, hcp_id),
                )
                # NO downstream treatment_events seeded → NEGATIVE.

                cur.execute("SELECT * FROM run_feedback_loop('next_best_action')")
                loop_rows = cur.fetchall()
                nba_row = next((r for r in loop_rows if r[0] == "next_best_action"), None)
                assert nba_row is not None
                assert nba_row[1] == "COMPLETED", (
                    f"run_feedback_loop('next_best_action') failed: {nba_row!r}"
                )

                cur.execute(
                    "SELECT actual_outcome, outcome_label, truth_source, truth_confidence "
                    "FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()

            assert row is not None
            actual, outcome_label, truth_source, truth_confidence = row
            assert actual is not None
            assert float(actual) == pytest.approx(0.0, abs=1e-3)
            assert outcome_label == "NEGATIVE"
            assert truth_source == "treatment_events", (
                f"Expected post-039 truth_source='treatment_events' "
                f"(was 'triggers_treatment_events' pre-039) but got {truth_source!r}"
            )
            # Codex pass-1 LOW-3: pin the post-039 truth_confidence value.
            # Pre-039 NBA had bifurcated 0.90 (if trigger_status='accepted')
            # vs 0.70 (otherwise); post-039 we collapse to a single 0.70
            # since trigger acceptance is no longer part of the evidence.
            # Future hand-edit that adds back a confidence branch on a
            # column we DO have (e.g. acceptance_status from a corrected
            # join) would change this value and trip the assertion.
            assert truth_confidence is not None
            assert float(truth_confidence) == pytest.approx(0.70, abs=1e-3), (
                f"Expected post-039 NBA truth_confidence=0.70 but got {truth_confidence!r}"
            )
        finally:
            conn.rollback()

    def test_assign_truth_next_best_action_labels_positive_with_activity(self, conn) -> None:
        """Issue #182 acceptance criterion 2 — POSITIVE branch of
        ``assign_truth_next_best_action`` via
        ``run_feedback_loop('next_best_action')``. Seeds an old PENDING
        'next_best_action' prediction PLUS a downstream treatment_event
        for the same HCP in the observation window; asserts POSITIVE
        with ``truth_source='treatment_events'`` and the post-039
        ``truth_confidence=0.70`` (codex pass-1 LOW-3: pin the value on
        the POSITIVE branch too, not just NEGATIVE).
        """
        suffix = uuid.uuid4().hex[:10]
        patient_id = f"PTP182_{suffix[:9]}"
        pjid = f"PJP182_{suffix[:9]}"
        prediction_id = f"nbap182_{suffix}_{uuid.uuid4().hex[:5]}"
        hcp_id = f"HCP182_{suffix[:8]}"
        old_ts = datetime.now(timezone.utc) - timedelta(days=60)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ml_predictions SET outcome_label = 'EXCLUDED' "
                    "WHERE prediction_type = 'next_best_action' "
                    "AND outcome_label = 'PENDING' "
                    "AND prediction_timestamp < NOW() - INTERVAL '30 days' "
                    "AND prediction_id <> %s",
                    (prediction_id,),
                )

                cur.execute(
                    "INSERT INTO hcp_profiles (hcp_id, first_name, last_name, "
                    "specialty, npi) VALUES (%s, 'Test', 'HCP', 'Dermatology', %s) "
                    "ON CONFLICT (hcp_id) DO NOTHING",
                    (hcp_id, f"97{suffix[:8]}"),
                )
                cur.execute(
                    "INSERT INTO patient_journeys "
                    "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                    "VALUES (%s, %s, %s, 'initial_treatment')",
                    (pjid, patient_id, old_ts.date()),
                )
                cur.execute(
                    "INSERT INTO ml_predictions "
                    "(prediction_id, model_version, model_type, "
                    "prediction_timestamp, patient_id, hcp_id, "
                    "prediction_type, prediction_value, confidence_score, "
                    "outcome_label) "
                    "VALUES (%s, 'nba_v1', 'xgboost', %s, %s, %s, "
                    "'next_best_action', 0.50, 0.50, 'PENDING')",
                    (prediction_id, old_ts, patient_id, hcp_id),
                )
                # Downstream treatment_event for same HCP, 10 days
                # post-prediction (within 30-day NBA window).
                event_date = (old_ts + timedelta(days=10)).date()
                event_id = f"TE_{uuid.uuid4().hex[:14]}"
                cur.execute(
                    "INSERT INTO treatment_events "
                    "(treatment_event_id, patient_journey_id, patient_id, hcp_id, "
                    "event_date, event_type, duration_days, brand) "
                    "VALUES (%s, %s, %s, %s, %s, 'consultation', 30, 'Fabhalta')",
                    (event_id, pjid, patient_id, hcp_id, event_date),
                )

                cur.execute("SELECT * FROM run_feedback_loop('next_best_action')")
                loop_rows = cur.fetchall()
                nba_row = next((r for r in loop_rows if r[0] == "next_best_action"), None)
                assert nba_row is not None
                assert nba_row[1] == "COMPLETED"

                cur.execute(
                    "SELECT actual_outcome, outcome_label, truth_source, "
                    "       truth_confidence, exclusion_reason "
                    "FROM ml_predictions WHERE prediction_id = %s",
                    (prediction_id,),
                )
                row = cur.fetchone()

            assert row is not None
            actual, outcome_label, truth_source, truth_confidence, exclusion_reason = row
            assert actual is not None
            assert float(actual) == pytest.approx(1.0, abs=1e-3)
            assert outcome_label == "POSITIVE"
            assert truth_source == "treatment_events"
            # Codex pass-1 LOW-3: pin POSITIVE-branch truth_confidence.
            assert truth_confidence is not None
            assert float(truth_confidence) == pytest.approx(0.70, abs=1e-3)
            # Codex pass-1 LOW-1: post-039 explicitly clears
            # exclusion_reason on labeled rows.
            assert exclusion_reason is None, (
                f"exclusion_reason should be NULL on POSITIVE label, got {exclusion_reason!r}"
            )
        finally:
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
