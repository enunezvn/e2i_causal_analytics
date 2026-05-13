"""Risk-score prediction Celery tasks (issue #173).

End-to-end write path for the calibrated risk_score model trained by
``scripts/train_risk_score_model.py``:

1. Load the calibrated estimator from an MLflow run (``mlflow.sklearn.load_model``).
2. Score the supplied feature frame.
3. UPDATE ``patient_journeys.risk_score`` for matching ``patient_journey_id``s
   (gated on ``journey_stage IN ('initial_treatment','maintenance')`` per
   the issue body — extended to also include the 7-stage equivalents
   ``'treatment_optimization','treatment_switch'`` when present).
4. UPSERT one ``ml_predictions`` row per scored patient. Idempotency is
   provided by a deterministic ``prediction_id`` derived from
   ``(model_version, patient_id, prediction_timestamp_yyyymmdd)`` so
   re-runs on the same day overwrite, while runs on different days append.

DB writes use ``psycopg`` (sync) — matching the pattern in
``src/tasks/nppes_tasks.py``. The task is *broker-agnostic*: when
executed eagerly (``celery.send_task`` with ``CELERY_TASK_ALWAYS_EAGER=1``)
it behaves like a plain function for integration tests.

Why a separate file (not the trainer module): trainers are import-heavy
(MLflow + Optuna + XGBoost + SHAP). Celery imports this module at worker
start; we want a thin DB-write path that lazy-loads model frameworks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Stages eligible for risk_score write per issue #173 scope item 4.
#
# The ``journey_stage_type`` enum (database/core/e2i_ml_complete_v3_schema.sql:155)
# contains both a legacy 4-stage set and a 7-stage extension. We score
# patients who are post-initiation but still on (or recently on) therapy:
#
#   * Legacy 4-stage (issue body §4 literal):
#       initial_treatment, maintenance
#   * Legacy 4-stage (transitional / mid-treatment — included for parity
#     with the cohort builder which routes mid-treatment patients here):
#       treatment_optimization, treatment_switch
#   * 7-stage extension (codex pass-1 HIGH-2 fix — these are the
#     canonical 7-stage equivalents):
#       prescribed, first_fill, adherent, maintained
#
# We intentionally exclude pre-treatment stages (``diagnosis``,
# ``aware``, ``considering``) and the terminal ``discontinued`` stage.
RISK_ELIGIBLE_JOURNEY_STAGES: frozenset[str] = frozenset(
    {
        # Legacy 4-stage (on-treatment subset)
        "initial_treatment",
        "maintenance",
        "treatment_optimization",
        "treatment_switch",
        # 7-stage extension (on-treatment subset)
        "prescribed",
        "first_fill",
        "adherent",
        "maintained",
    }
)

# Default model_version tag. Production callers SHOULD override; the
# placeholder lets the integration test exercise the write path before a
# semver is pinned.
DEFAULT_MODEL_VERSION = "risk_score_v1"

# Issue #188: downstream gate for the AUC-PR floor mechanism. When the
# trainer produced honest_failures (AUC-PR floor not met OR calibration
# unmet), this task MUST refuse to promote the model's per-patient
# outputs to actionable DB writes — even if a developer mistakenly
# invokes the task with raw payloads from a non-promotable run.
#
# The gate inspects the per-call ``honest_failures`` argument; if
# non-empty, predictions are NOT written to ml_predictions, and
# patient_journeys.risk_score is NOT updated. A structured audit event
# ``risk_score.skipped.honest_failure`` is logged.
#
# Callers that have NO honest_failures metadata available (e.g. an old
# pre-#188 model artifact whose payloads omit the field) are treated
# according to ``honest_failures_default_gated``: when ``True`` (the
# safe default), missing metadata is treated as a failure-gate; when
# ``False``, missing metadata is treated as a pass (back-compat for
# callers that pre-validated upstream).
GATED_SENTINEL_PREDICTION_CLASS: str = "gated_honest_failure"


# ---------------------------------------------------------------------------
# Helpers (pure; unit-testable without DB or Celery)
# ---------------------------------------------------------------------------


def make_deterministic_prediction_id(
    model_version: str,
    patient_id: str,
    prediction_timestamp: datetime,
) -> str:
    """Return a 30-char deterministic ``prediction_id`` for idempotency.

    Format: ``rsc_<26 hex>`` where the hex is the first 13 bytes of
    ``sha256(model_version || '|' || patient_id || '|' || YYYY-MM-DD)``.

    The same (model_version, patient_id, calendar-UTC-date) triple
    therefore always produces the same ID, so an ON CONFLICT UPDATE on
    the PRIMARY KEY ``prediction_id`` provides idempotent upsert
    semantics without requiring a unique-index migration.
    """
    if not model_version or not patient_id:
        raise ValueError("model_version and patient_id must be non-empty")
    # Codex pass-1 MEDIUM-1: reject naive datetimes. ``astimezone()`` on a
    # naive datetime treats it as local time, which means workers in
    # different time zones (or across DST boundaries) would derive
    # different UTC dates from the same input and silently break
    # idempotency. Require explicit tzinfo.
    if prediction_timestamp.tzinfo is None:
        raise ValueError(
            "prediction_timestamp must be timezone-aware (tzinfo is None). "
            "Pass datetime.now(timezone.utc) or attach tzinfo explicitly."
        )
    ts_utc = prediction_timestamp.astimezone(timezone.utc).date().isoformat()
    digest = hashlib.sha256(f"{model_version}|{patient_id}|{ts_utc}".encode("utf-8")).hexdigest()
    return f"rsc_{digest[:26]}"


def _coerce_decimal_3_2(score: float) -> float:
    """Clamp the risk score into the DECIMAL(3,2) range [0.00, 9.99].

    The trainer already clamps in :func:`probability_to_risk_score`, but
    we defend in depth here so a bug upstream does not poison a DB write.
    """
    if score < 0.0:
        return 0.0
    if score > 9.99:
        return 9.99
    return round(float(score), 2)


# ---------------------------------------------------------------------------
# DB-write primitives (sync psycopg; matches nppes_tasks pattern)
# ---------------------------------------------------------------------------


def _resolve_db_url(db_url: Optional[str]) -> Optional[str]:
    """Resolve a Postgres URL from the argument or env, returning ``None``
    if no usable URL is configured.

    Order of precedence:
        1. Explicit ``db_url`` argument.
        2. ``RISK_SCORE_DB_URL`` (cohort-scoped override).
        3. ``SUPABASE_DB_URL`` (local supabase stack).
        4. ``DATABASE_URL`` (generic).
    """
    return (
        db_url
        or os.environ.get("RISK_SCORE_DB_URL")
        or os.environ.get("SUPABASE_DB_URL")
        or os.environ.get("DATABASE_URL")
    )


def update_patient_journey_risk_scores(
    conn: Any,
    rows: Iterable[Mapping[str, Any]],
    eligible_stages: frozenset[str] = RISK_ELIGIBLE_JOURNEY_STAGES,
    *,
    commit: bool = True,
) -> dict[str, int]:
    """UPDATE ``patient_journeys.risk_score`` for the supplied rows.

    Each row must contain ``patient_journey_id`` and ``risk_score``.
    The DB-side filter ``WHERE journey_stage IN (...)`` enforces the
    issue-scoped stage gate even if the caller forgot to pre-filter.
    Returns ``{"updated": N, "skipped_ineligible": M}`` where
    ``skipped_ineligible`` is rows that matched a ``patient_journey_id``
    but whose stage was outside the gate. Missing IDs are silently
    counted as ``not_in_db``.

    Codex pass-1 MEDIUM-3: ``commit`` defaults to True for backward
    compat in unit tests, but the Celery task calls with ``commit=False``
    so both this UPDATE and the ``upsert_ml_predictions`` INSERT
    share a single transaction — atomicity across both tables.
    """
    rows = list(rows)
    if not rows:
        return {"updated": 0, "skipped_ineligible": 0, "not_in_db": 0, "submitted": 0}
    eligible_tuple = tuple(sorted(eligible_stages))
    updated = 0
    ineligible = 0
    submitted = len(rows)

    with conn.cursor() as cur:
        # Pre-query: which patient_journey_ids exist and which are stage-eligible?
        ids = [r["patient_journey_id"] for r in rows]
        cur.execute(
            "SELECT patient_journey_id, journey_stage FROM patient_journeys "
            "WHERE patient_journey_id = ANY(%s)",
            (ids,),
        )
        present = {row[0]: row[1] for row in cur.fetchall()}
        for r in rows:
            pjid = r["patient_journey_id"]
            if pjid not in present:
                continue  # not_in_db; counted below
            if present[pjid] not in eligible_stages:
                ineligible += 1
                continue
            # Cast journey_stage to text so the comparison works even on
            # DBs whose ``journey_stage_type`` enum hasn't yet been
            # extended with all of our eligible labels (e.g. a Postgres
            # instance still on the 4-stage enum revision will reject
            # casting ``'adherent'::journey_stage_type``). Text equality
            # passes through the existing app-level set check so the
            # gate is preserved either way.
            cur.execute(
                "UPDATE patient_journeys SET risk_score = %s, updated_at = NOW() "
                "WHERE patient_journey_id = %s "
                "AND journey_stage::text = ANY(%s)",
                (_coerce_decimal_3_2(float(r["risk_score"])), pjid, list(eligible_tuple)),
            )
            updated += cur.rowcount
    if commit:
        conn.commit()
    return {
        "updated": updated,
        "skipped_ineligible": ineligible,
        "not_in_db": submitted - len(present),
        "submitted": submitted,
    }


def upsert_ml_predictions(
    conn: Any,
    payloads: Iterable[Mapping[str, Any]],
    *,
    commit: bool = True,
) -> dict[str, int]:
    """UPSERT ``ml_predictions`` rows by ``prediction_id`` (PRIMARY KEY).

    Each payload should be the dict returned by
    :meth:`RiskScoreTrainer.build_ml_predictions_payload`. JSONB columns
    (``probability_scores``, ``feature_importance``, ``shap_values``,
    ``top_features``, ``features_available_at_prediction``) are
    serialised here so the caller does not need to know which columns
    are JSONB.

    Idempotency: ON CONFLICT (prediction_id) DO UPDATE — combined with
    :func:`make_deterministic_prediction_id` upstream, this gives
    "same model + same patient + same UTC day -> same row".
    """
    payloads = list(payloads)
    if not payloads:
        return {"inserted": 0, "updated": 0, "submitted": 0}

    cols = [
        "prediction_id",
        "model_version",
        "model_type",
        "prediction_timestamp",
        "patient_id",
        "prediction_type",
        "prediction_value",
        "prediction_class",
        "confidence_score",
        "probability_scores",
        "feature_importance",
        "shap_values",
        "top_features",
        "model_auc",
        "model_pr_auc",
        "model_precision",
        "model_recall",
        "calibration_score",
        "brier_score",
        "features_available_at_prediction",
    ]
    jsonb_cols = {
        "probability_scores",
        "feature_importance",
        "shap_values",
        "top_features",
        "features_available_at_prediction",
    }
    placeholders = ", ".join(["%s"] * len(cols))
    update_pairs = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "prediction_id")
    sql = (
        f"INSERT INTO ml_predictions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (prediction_id) DO UPDATE SET {update_pairs} "
        f"RETURNING (xmax = 0) AS inserted"
    )

    inserted = 0
    updated = 0
    with conn.cursor() as cur:
        for payload in payloads:
            values: list[Any] = []
            for c in cols:
                v = payload.get(c)
                if c in jsonb_cols:
                    values.append(json.dumps(v if v is not None else {}))
                else:
                    values.append(v)
            cur.execute(sql, values)
            res = cur.fetchone()
            # xmax=0 means a fresh insert; otherwise update.
            if res and res[0]:
                inserted += 1
            else:
                updated += 1
    if commit:
        conn.commit()
    return {"inserted": inserted, "updated": updated, "submitted": len(payloads)}


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------


@celery_app.task(
    bind=True,
    name="src.tasks.write_risk_score_predictions",
    queue="ml",
)
def write_risk_score_predictions(
    self,
    payloads: list[dict[str, Any]],
    journey_updates: Optional[list[dict[str, Any]]] = None,
    db_url: Optional[str] = None,
    honest_failures: Optional[list[str]] = None,
    honest_failures_default_gated: bool = True,
) -> dict[str, Any]:
    """Persist a batch of risk_score predictions to Postgres.

    Args:
        payloads: list of ``ml_predictions`` row dicts (typically built
            via ``RiskScoreTrainer.build_ml_predictions_payload``). For
            idempotency, each payload's ``prediction_id`` SHOULD have
            been produced by :func:`make_deterministic_prediction_id`.
        journey_updates: list of ``{patient_journey_id, risk_score}``
            dicts; ``risk_score`` is the DECIMAL(3,2) tier score
            (NOT the probability). If ``None``, no journey writes occur.
        db_url: Postgres URL; falls back to ``RISK_SCORE_DB_URL`` /
            ``SUPABASE_DB_URL`` / ``DATABASE_URL`` env vars in that
            order.
        honest_failures: list of failure messages from the trainer
            (``RiskScoreTrainingResult.honest_failures``). When this
            list is non-empty, the task switches to GATED mode: it
            does NOT update ``patient_journeys.risk_score`` and it does
            NOT propagate ``prediction_class`` to ``ml_predictions``
            (the audit row is still written with
            ``prediction_class='gated_honest_failure'`` so the gating
            event is observable in downstream dashboards). Issue #188.
        honest_failures_default_gated: behavior when ``honest_failures``
            is ``None`` (e.g. an old caller that pre-dates #188 and
            does not pass the kwarg). When ``True`` (the safe default),
            missing metadata is treated as a failure-gate; pass
            ``False`` to opt out (back-compat for callers that
            already validated upstream).

    Returns:
        ``{
            "status": "completed" | "skipped" | "failed" | "gated_honest_failure",
            "predictions": {"inserted", "updated", "submitted"},
            "journeys":    {"updated", "skipped_ineligible", "not_in_db", "submitted"},
            "task_id": ...,
            "honest_failures": [...],     # if gated
        }``

    The task is *idempotent on re-run*: the same payloads + journey
    updates can be submitted N times and the database row count is
    the same, with row contents matching the LAST submission.
    """
    task_id = getattr(self.request, "id", "eager") if self else "eager"

    # Issue #188: resolve gate decision.
    if honest_failures is None:
        # Caller did not pass honest_failures. Default behavior is to
        # treat missing metadata as a failure-gate (safe default).
        is_gated = bool(honest_failures_default_gated)
        gate_reason = (
            "missing_honest_failures_metadata"
            if is_gated
            else "missing_metadata_opt_out"
        )
        resolved_honest_failures: list[str] = (
            ["honest_failures metadata missing (treated as gated per issue #188)"]
            if is_gated
            else []
        )
    else:
        resolved_honest_failures = list(honest_failures)
        is_gated = len(resolved_honest_failures) > 0
        gate_reason = "honest_failures_non_empty" if is_gated else "honest_failures_empty"

    logger.info(
        "write_risk_score_predictions: task=%s payloads=%d journeys=%d "
        "gated=%s reason=%s",
        task_id,
        len(payloads),
        len(journey_updates or []),
        is_gated,
        gate_reason,
    )

    if is_gated:
        # Structured audit event for downstream dashboards (do NOT update
        # patient_journeys.risk_score; do NOT write prediction_class).
        logger.warning(
            "risk_score.skipped.honest_failure task=%s reason=%s "
            "honest_failures=%s",
            task_id,
            gate_reason,
            resolved_honest_failures,
        )

    resolved_url = _resolve_db_url(db_url)
    if not resolved_url:
        logger.warning(
            "write_risk_score_predictions: no DB URL configured "
            "(RISK_SCORE_DB_URL / SUPABASE_DB_URL / DATABASE_URL all unset). "
            "Skipping DB writes — surface as honest deferral."
        )
        return {
            "status": "skipped",
            "reason": "no_db_url",
            "task_id": task_id,
            "predictions": {"inserted": 0, "updated": 0, "submitted": len(payloads)},
            "journeys": {
                "updated": 0,
                "skipped_ineligible": 0,
                "not_in_db": 0,
                "submitted": len(journey_updates or []),
            },
            "gated": is_gated,
            "honest_failures": resolved_honest_failures,
        }

    try:
        import psycopg  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "write_risk_score_predictions: psycopg not installed. Install via "
            "`pip install psycopg[binary]` to enable DB writes."
        )
        return {
            "status": "failed",
            "reason": "psycopg_missing",
            "task_id": task_id,
        }

    # Issue #188: when gated, we still write an audit row to ml_predictions
    # so downstream dashboards can observe which (model, patient, day)
    # triples were NOT promoted — but we (a) overwrite the actionable
    # ``prediction_class`` with the sentinel ``'gated_honest_failure'``
    # and (b) skip the patient_journeys.risk_score UPDATE entirely.
    if is_gated:
        gated_payloads = [
            {**p, "prediction_class": GATED_SENTINEL_PREDICTION_CLASS} for p in payloads
        ]
    else:
        gated_payloads = list(payloads)

    # Codex pass-1 MEDIUM-3: open a single transaction so both writes
    # (ml_predictions upsert + patient_journeys.risk_score UPDATE) are
    # atomic. ``psycopg.connect()`` as a context manager auto-commits on
    # clean exit and rolls back on exception — so we pass ``commit=False``
    # to the helpers and let the connection context handle the
    # transaction boundary.
    try:
        with psycopg.connect(resolved_url) as conn:
            pred_result = upsert_ml_predictions(conn, gated_payloads, commit=False)
            # Issue #188: journey writes are SUPPRESSED in gated mode.
            if is_gated:
                journey_result = {
                    "updated": 0,
                    "skipped_ineligible": 0,
                    "not_in_db": 0,
                    "submitted": len(journey_updates or []),
                    "skipped_gated_honest_failure": len(journey_updates or []),
                }
            else:
                journey_result = (
                    update_patient_journey_risk_scores(conn, journey_updates, commit=False)
                    if journey_updates
                    else {
                        "updated": 0,
                        "skipped_ineligible": 0,
                        "not_in_db": 0,
                        "submitted": 0,
                    }
                )
            # Explicit commit at the transaction boundary so a journey
            # failure rolls BOTH writes back, not just the journey
            # write.
            conn.commit()
    except Exception as exc:  # pragma: no cover - real-DB error path
        logger.exception("write_risk_score_predictions: DB write failed: %s", exc)
        return {
            "status": "failed",
            "reason": f"db_error: {type(exc).__name__}: {exc}",
            "task_id": task_id,
            "gated": is_gated,
            "honest_failures": resolved_honest_failures,
        }

    logger.info(
        "write_risk_score_predictions: predictions=%s journeys=%s gated=%s",
        pred_result,
        journey_result,
        is_gated,
    )
    return {
        "status": "gated_honest_failure" if is_gated else "completed",
        "task_id": task_id,
        "predictions": pred_result,
        "journeys": journey_result,
        "gated": is_gated,
        "honest_failures": resolved_honest_failures,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
