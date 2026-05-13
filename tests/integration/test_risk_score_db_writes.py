"""Integration tests for ``src/tasks/risk_score_prediction_tasks.py`` (issue #173).

Verifies:
    1. Deterministic ``prediction_id`` is stable across calls (idempotency primitive).
    2. ``write_risk_score_predictions`` upserts ``ml_predictions`` rows
       and is idempotent on re-run (no duplicate-key errors; row count unchanged).
    3. ``update_patient_journey_risk_scores`` only updates rows whose
       ``journey_stage`` is in ``RISK_ELIGIBLE_JOURNEY_STAGES``.
    4. The task gracefully skips when no DB URL is configured.
    5. Decimal(3,2) clamping defends against probability-passed-as-score bugs.

DB-touching tests are gated on a reachable Postgres at ``SUPABASE_DB_URL`` /
``RISK_SCORE_DB_URL`` / ``DATABASE_URL``; in CI these env vars are
typically unset, so DB tests are auto-skipped. The deterministic-ID +
clamp + no-DB tests run unconditionally.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from src.tasks.risk_score_prediction_tasks import (
    DEFAULT_MODEL_VERSION,
    RISK_ELIGIBLE_JOURNEY_STAGES,
    _coerce_decimal_3_2,
    _resolve_db_url,
    make_deterministic_prediction_id,
    update_patient_journey_risk_scores,
    upsert_ml_predictions,
    write_risk_score_predictions,
)

# ---------------------------------------------------------------------------
# Reachability gate
# ---------------------------------------------------------------------------


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
        "DATABASE_URL — skipping DB-touching integration tests."
    ),
)


# ---------------------------------------------------------------------------
# Pure / no-DB tests
# ---------------------------------------------------------------------------


class TestDeterministicPredictionId:
    """Idempotency primitive — same triple, same ID."""

    def test_stable_within_a_calendar_day(self) -> None:
        t1 = datetime(2026, 5, 13, 9, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 5, 13, 23, 59, tzinfo=timezone.utc)
        id1 = make_deterministic_prediction_id("v1", "PAT_001", t1)
        id2 = make_deterministic_prediction_id("v1", "PAT_001", t2)
        assert id1 == id2

    def test_changes_across_days(self) -> None:
        t1 = datetime(2026, 5, 13, 23, 59, tzinfo=timezone.utc)
        t2 = datetime(2026, 5, 14, 0, 0, tzinfo=timezone.utc)
        assert make_deterministic_prediction_id("v1", "PAT_001", t1) != (
            make_deterministic_prediction_id("v1", "PAT_001", t2)
        )

    def test_changes_with_model_version(self) -> None:
        t = datetime(2026, 5, 13, tzinfo=timezone.utc)
        assert make_deterministic_prediction_id("v1", "PAT_001", t) != (
            make_deterministic_prediction_id("v2", "PAT_001", t)
        )

    def test_changes_with_patient(self) -> None:
        t = datetime(2026, 5, 13, tzinfo=timezone.utc)
        assert make_deterministic_prediction_id("v1", "PAT_001", t) != (
            make_deterministic_prediction_id("v1", "PAT_002", t)
        )

    def test_thirty_chars(self) -> None:
        out = make_deterministic_prediction_id(
            "v1", "PAT_001", datetime(2026, 5, 13, tzinfo=timezone.utc)
        )
        assert len(out) == 30
        assert out.startswith("rsc_")

    def test_naive_datetime_rejected(self) -> None:
        """Codex pass-1 MEDIUM-1: a naive datetime would silently produce
        different UTC dates across workers in different time zones. We
        reject it explicitly so idempotency is never accidentally broken
        by an unzoned timestamp.
        """
        with pytest.raises(ValueError, match="timezone-aware"):
            make_deterministic_prediction_id("v1", "PAT_001", datetime(2026, 5, 13, 12, 0))

    def test_empty_inputs_raise(self) -> None:
        t = datetime(2026, 5, 13, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            make_deterministic_prediction_id("", "PAT_001", t)
        with pytest.raises(ValueError):
            make_deterministic_prediction_id("v1", "", t)


class TestDecimalClamp:
    def test_clamps_high(self) -> None:
        assert _coerce_decimal_3_2(99.5) == 9.99

    def test_clamps_low(self) -> None:
        assert _coerce_decimal_3_2(-0.5) == 0.0

    def test_rounds_two_dp(self) -> None:
        assert _coerce_decimal_3_2(3.456) == 3.46


class TestNoDbDeferral:
    def test_task_skips_when_no_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in ("RISK_SCORE_DB_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
            monkeypatch.delenv(v, raising=False)
        out = write_risk_score_predictions.apply(args=([{"prediction_id": "x"}],)).result
        assert out["status"] == "skipped"
        assert out["reason"] == "no_db_url"
        assert out["predictions"]["submitted"] == 1

    def test_resolve_url_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in ("RISK_SCORE_DB_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("DATABASE_URL", "postgres://fallback")
        monkeypatch.setenv("SUPABASE_DB_URL", "postgres://supabase")
        # SUPABASE_DB_URL beats DATABASE_URL.
        assert _resolve_db_url(None) == "postgres://supabase"
        monkeypatch.setenv("RISK_SCORE_DB_URL", "postgres://primary")
        assert _resolve_db_url(None) == "postgres://primary"
        # Explicit arg beats env.
        assert _resolve_db_url("postgres://explicit") == "postgres://explicit"


class TestEligibleStages:
    def test_includes_legacy_4_stage_on_treatment(self) -> None:
        # Legacy 4-stage on-treatment subset (issue body §4 literal).
        assert "initial_treatment" in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "maintenance" in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "treatment_optimization" in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "treatment_switch" in RISK_ELIGIBLE_JOURNEY_STAGES

    def test_includes_v3_7_stage_on_treatment(self) -> None:
        """Codex pass-1 HIGH-2: the 7-stage enum's on-treatment subset is
        ``prescribed, first_fill, adherent, maintained`` per
        ``journey_stage_type`` ENUM. The original PR set used the legacy
        names by mistake.
        """
        assert "prescribed" in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "first_fill" in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "adherent" in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "maintained" in RISK_ELIGIBLE_JOURNEY_STAGES

    def test_excludes_pre_treatment_and_terminal(self) -> None:
        # Pre-treatment stages — these patients haven't initiated yet.
        assert "diagnosis" not in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "aware" not in RISK_ELIGIBLE_JOURNEY_STAGES
        assert "considering" not in RISK_ELIGIBLE_JOURNEY_STAGES
        # Terminal stage — no future risk to score.
        assert "discontinued" not in RISK_ELIGIBLE_JOURNEY_STAGES


# ---------------------------------------------------------------------------
# Real-DB integration tests
# ---------------------------------------------------------------------------


@_db_available
class TestRealDbWrites:
    """Verifies real Postgres round-trip for the Celery task body."""

    @pytest.fixture
    def conn(self):
        import psycopg  # type: ignore[import-untyped]

        c = psycopg.connect(_DB_URL)
        yield c
        c.close()

    @pytest.fixture
    def test_patient_id(self) -> str:
        # patient_id VARCHAR(20) — keep it short.
        return f"PT_{uuid.uuid4().hex[:12]}"  # 15 chars

    @pytest.fixture
    def seeded_journey(self, conn: Any, test_patient_id: str) -> str:
        """Insert a single test journey row and clean it up after the test."""
        # patient_journey_id VARCHAR(20) — keep it short.
        pjid = f"PJ_{uuid.uuid4().hex[:12]}"  # 15 chars
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                "VALUES (%s, %s, %s, 'initial_treatment')",
                (pjid, test_patient_id, "2026-05-01"),
            )
        conn.commit()
        yield pjid
        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM patient_journeys WHERE patient_journey_id = %s", (pjid,))
        conn.commit()

    @pytest.fixture
    def seeded_journey_ineligible(self, conn: Any, test_patient_id: str) -> str:
        """A journey in a non-eligible stage ('diagnosis')."""
        pjid = f"PJX_{uuid.uuid4().hex[:12]}"  # 16 chars
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                "VALUES (%s, %s, %s, 'diagnosis')",
                (pjid, test_patient_id, "2026-05-01"),
            )
        conn.commit()
        yield pjid
        with conn.cursor() as cur:
            cur.execute("DELETE FROM patient_journeys WHERE patient_journey_id = %s", (pjid,))
        conn.commit()

    @pytest.fixture
    def seeded_journey_7stage(self, conn: Any, test_patient_id: str) -> str:
        """A journey in the 7-stage 'adherent' enum (codex pass-1 HIGH-2).

        Skips if the DB's journey_stage_type enum doesn't yet include
        ``'adherent'`` — local Supabase stacks on older schema revisions
        only have the 4-stage values. The 4-stage-only test
        ``test_journey_update_only_eligible_stage`` still exercises the
        4-stage on-treatment path on those DBs.
        """
        with conn.cursor() as cur:
            cur.execute("SELECT 'adherent' = ANY(enum_range(NULL::journey_stage_type)::text[])")
            row = cur.fetchone()
        if not row or not row[0]:
            pytest.skip(
                "journey_stage_type enum on this DB does not include 'adherent' "
                "(pre-v3 7-stage revision); 4-stage subset test still covers the gate."
            )
        pjid = f"PJ7_{uuid.uuid4().hex[:12]}"  # 16 chars
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO patient_journeys "
                "(patient_journey_id, patient_id, journey_start_date, journey_stage) "
                "VALUES (%s, %s, %s, 'adherent')",
                (pjid, test_patient_id, "2026-05-01"),
            )
        conn.commit()
        yield pjid
        with conn.cursor() as cur:
            cur.execute("DELETE FROM patient_journeys WHERE patient_journey_id = %s", (pjid,))
        conn.commit()

    def _make_payload(
        self,
        patient_id: str,
        pred_id: str,
        prob: float,
        ts: datetime,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> dict[str, Any]:
        return {
            "prediction_id": pred_id,
            "prediction_timestamp": ts.isoformat(),
            "model_version": model_version,
            "model_type": "xgboost",
            "patient_id": patient_id,
            "prediction_type": "risk",
            "prediction_value": prob,
            "prediction_class": "low" if prob < 0.5 else "high",
            "confidence_score": prob,
            "probability_scores": {"positive": prob, "negative": 1 - prob},
            "feature_importance": {"feat_a": 0.5, "feat_b": 0.3},
            "shap_values": {},
            "top_features": [{"feature": "feat_a", "gain": 0.5}],
            "model_auc": 0.78,
            "model_pr_auc": 0.10,
            "model_precision": 0.0,
            "model_recall": 0.0,
            "calibration_score": 1.0,
            "brier_score": 0.04,
            "features_available_at_prediction": {"set": "OPTUM_SAFE_FEATURES"},
        }

    def test_upsert_ml_predictions_inserts_then_updates(
        self, conn: Any, test_patient_id: str
    ) -> None:
        ts = datetime.now(timezone.utc)
        pid = make_deterministic_prediction_id(DEFAULT_MODEL_VERSION, test_patient_id, ts)
        p1 = self._make_payload(test_patient_id, pid, 0.10, ts)
        result1 = upsert_ml_predictions(conn, [p1])
        assert result1 == {"inserted": 1, "updated": 0, "submitted": 1}

        # Same prediction_id, different probability -> UPDATE
        p2 = self._make_payload(test_patient_id, pid, 0.85, ts)
        result2 = upsert_ml_predictions(conn, [p2])
        assert result2 == {"inserted": 0, "updated": 1, "submitted": 1}

        with conn.cursor() as cur:
            cur.execute(
                "SELECT prediction_value, prediction_class FROM ml_predictions "
                "WHERE prediction_id = %s",
                (pid,),
            )
            row = cur.fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(0.85, abs=1e-3)
        assert row[1] == "high"

        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ml_predictions WHERE prediction_id = %s", (pid,))
        conn.commit()

    def test_idempotent_on_full_re_run(self, conn: Any, test_patient_id: str) -> None:
        ts = datetime.now(timezone.utc)
        pid = make_deterministic_prediction_id(DEFAULT_MODEL_VERSION, test_patient_id, ts)
        payload = self._make_payload(test_patient_id, pid, 0.42, ts)
        upsert_ml_predictions(conn, [payload])
        upsert_ml_predictions(conn, [payload])
        upsert_ml_predictions(conn, [payload])
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ml_predictions WHERE prediction_id = %s", (pid,))
            assert cur.fetchone()[0] == 1
            cur.execute("DELETE FROM ml_predictions WHERE prediction_id = %s", (pid,))
        conn.commit()

    def test_journey_update_only_eligible_stage(
        self,
        conn: Any,
        seeded_journey: str,
        seeded_journey_ineligible: str,
    ) -> None:
        """Legacy 4-stage gate — runs on every DB regardless of enum
        revision. (Codex pass-2 MEDIUM-1: split from the 7-stage assertion
        so older DBs still test the gate.)
        """
        result = update_patient_journey_risk_scores(
            conn,
            [
                {"patient_journey_id": seeded_journey, "risk_score": 7.42},
                {"patient_journey_id": seeded_journey_ineligible, "risk_score": 7.42},
                {"patient_journey_id": "PJ_no_exist_zzzz", "risk_score": 5.0},
            ],
        )
        assert result["updated"] == 1  # only the eligible 4-stage row
        assert result["skipped_ineligible"] == 1
        assert result["not_in_db"] == 1

        # Verify the eligible row was actually written.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT risk_score FROM patient_journeys WHERE patient_journey_id = %s",
                (seeded_journey,),
            )
            row = cur.fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(7.42, abs=0.01)

        # And the ineligible row was NOT.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT risk_score FROM patient_journeys WHERE patient_journey_id = %s",
                (seeded_journey_ineligible,),
            )
            row = cur.fetchone()
        assert row is not None
        assert row[0] is None

    def test_celery_task_eager_round_trip(
        self, conn: Any, test_patient_id: str, seeded_journey: str
    ) -> None:
        ts = datetime.now(timezone.utc)
        pid = make_deterministic_prediction_id(DEFAULT_MODEL_VERSION, test_patient_id, ts)
        payload = self._make_payload(test_patient_id, pid, 0.55, ts)
        result = write_risk_score_predictions.apply(
            args=(
                [payload],
                [{"patient_journey_id": seeded_journey, "risk_score": 5.5}],
                _DB_URL,
            )
        ).result
        assert result["status"] == "completed"
        assert result["predictions"]["submitted"] == 1
        assert result["predictions"]["inserted"] + result["predictions"]["updated"] == 1
        assert result["journeys"]["updated"] == 1
        # Cleanup
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ml_predictions WHERE prediction_id = %s", (pid,))
        conn.commit()

    def test_journey_update_includes_7stage_when_enum_supports(
        self,
        conn: Any,
        seeded_journey_7stage: str,
    ) -> None:
        """Codex pass-1 HIGH-2 + pass-2 MEDIUM-1: when the DB enum
        includes 7-stage values, the 'adherent' journey must be updated.
        Skipped automatically on pre-v3 enum revisions.
        """
        result = update_patient_journey_risk_scores(
            conn,
            [{"patient_journey_id": seeded_journey_7stage, "risk_score": 3.14}],
        )
        assert result["updated"] == 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT risk_score FROM patient_journeys WHERE patient_journey_id = %s",
                (seeded_journey_7stage,),
            )
            row = cur.fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(3.14, abs=0.01)
