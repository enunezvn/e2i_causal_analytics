"""Real-DB end-to-end integration test for the gold-standard INITIATION eval (T11).

This is the culmination of P1: it runs ``run_initiation_eval.run()`` against the
LIVE synthetic Remibrutinib cohort in the self-hosted Supabase docker DB and
asserts the REAL effects landed — a multi-month walk-forward AUC trend, a holdout
headline, idempotency on re-run, and that the widened read-path endpoint
(``get_performance_trend``, T10) actually surfaces the recorded backfill.

Gate: ``E2I_DB_INTEGRATION=1`` + a reachable async Supabase client
(``SUPABASE_URL`` + key). Mirrors ``test_feature_builder_live.py`` (T4) so unit-
only CI lanes never touch the DB.

NO mocks — every assertion runs against real rows. REAL prod writes are
authorized for this pipeline but MUST be idempotent (the test re-runs and
asserts no duplication).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason=(
        "E2I_DB_INTEGRATION!=1; set to 1 to run against the real Supabase "
        "DB. Requires SUPABASE_URL + key in environment."
    ),
)


# ---------------------------------------------------------------------------
# Fixture: fresh async client per test (mirrors test_feature_builder_live)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the global cached async client so each test gets a fresh one on its
    own event loop — the cached httpx.AsyncClient is bound to the creating loop
    and would raise 'Event loop is closed' on reuse across per-test loops.
    """
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _rows_for_current_handle(repo, handle: str, source: str) -> list[dict]:
    """auc_roc rows for the CURRENT registry id of ``handle`` and ``source``.

    The handle (model_name) is the stable identity. ``register_model_row`` now
    UPSERTS the registry row in place on re-run (preserving its ``id`` so the
    RESTRICT FKs from ml_performance_metrics / ml_drift_history survive), so the
    metric rows keep pointing at the same ``model_id``. Resolving the current id
    by handle each run is still correct (and robust to any future id change), and
    idempotency remains "the same COUNT of rows for the model".
    """
    from src.repositories.drift_monitoring import _resolve_model_id

    model_id = await _resolve_model_id(repo.client, handle)
    if model_id is None:
        return []
    rows = (
        await repo.client.table("ml_performance_metrics")
        .select("*")
        .eq("source", source)
        .eq("metric_name", "auc_roc")
        .eq("model_id", model_id)
        .execute()
    )
    return rows.data or []


# ---------------------------------------------------------------------------
# The end-to-end test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_initiation_eval_end_to_end_real_db() -> None:
    """Full pipeline: champion registered, multi-month trend, idempotent, endpoint surfaces it."""
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.gold_standard_eval.cohort_deployer import GOLDSTD_MODEL_NAME
    from src.mlops.gold_standard_eval.run_initiation_eval import run
    from src.repositories.drift_monitoring import (
        PerformanceMetricRepository,
        _resolve_model_id,
    )

    db = await get_async_supabase_client()
    repo = PerformanceMetricRepository(db)

    # --- Run the real pipeline (REAL prod writes; idempotent). -------------- #
    report = await run(db)

    # --- Champion registered. ---------------------------------------------- #
    assert report["champion_registered"], f"champion not registered: {report}"
    assert report["model_handle"] == GOLDSTD_MODEL_NAME

    # The handle must resolve to a real registry uuid (so the metric rows carry a
    # real model_id FK, not just the preserved-handle fallback).
    model_id = await _resolve_model_id(db, GOLDSTD_MODEL_NAME)
    assert model_id is not None, "registered handle did not resolve to a model_id uuid"

    # --- Real multi-month walk-forward trend landed. ----------------------- #
    rows = await _rows_for_current_handle(repo, GOLDSTD_MODEL_NAME, "backtest_wf")
    assert rows, "no backtest_wf auc_roc rows for the registered model"

    months = sorted({r["measured_at"][:7] for r in rows})
    assert len(months) >= 3, f"expected >=3 distinct backtest months, got {months}"

    # Real, sane AUCs (probabilistic classifier on real synthetic data).
    assert all(0.5 <= r["metric_value"] <= 1.0 for r in rows), (
        f"backtest AUCs out of [0.5, 1.0]: {sorted({round(r['metric_value'], 4) for r in rows})}"
    )

    # Report agrees with what landed.
    assert report["n_backtest_points"] == len(rows), (
        f"report n_backtest_points={report['n_backtest_points']} != landed rows={len(rows)}"
    )

    # --- Idempotency: a second run does not duplicate. --------------------- #
    # Compare COUNTS resolved against the CURRENT handle each time (the registry
    # id is replaced on every run; the handle is the stable identity). A broken
    # idempotency would show 2x the rows (the delete-then-insert recorder + the
    # pre-registration FK cleanup are what keep it 1x).
    n_before = len(rows)
    await run(db)
    rows2 = await _rows_for_current_handle(repo, GOLDSTD_MODEL_NAME, "backtest_wf")
    assert len(rows2) == n_before, (
        f"idempotency broken: backtest_wf rows {n_before} -> {len(rows2)} after re-run"
    )
    # And still a real multi-month trend after the re-run.
    months2 = sorted({r["measured_at"][:7] for r in rows2})
    assert len(months2) >= 3, f"re-run lost the multi-month trend: {months2}"

    # --- Holdout headline present. ----------------------------------------- #
    hold = await _rows_for_current_handle(repo, GOLDSTD_MODEL_NAME, "holdout")
    assert len(hold) >= 1, "no holdout auc_roc row recorded"
    # Holdout is a single headline point — idempotent re-run keeps exactly one.
    assert len(hold) == 1, f"expected exactly 1 holdout headline row, got {len(hold)}"
    assert 0.5 <= hold[0]["metric_value"] <= 1.0, (
        f"holdout AUC out of [0.5, 1.0]: {hold[0]['metric_value']}"
    )

    # --- Endpoint read-path (T10 widening) surfaces the backfill. ---------- #
    # Call the HTTP route function directly: it takes `days` and builds the
    # `history` list from repo.get_metric_trend(model_id, metric_name, days).
    # The walk-forward months span back ~3 years, so the 1825-day (~5yr) window
    # is what makes the multi-month history visible (a 30/365-day window would
    # truncate it) — this is exactly what T10 widened.
    from src.api.routes.monitoring import get_performance_trend

    response = await get_performance_trend(
        model_id=GOLDSTD_MODEL_NAME,
        metric_name="auc_roc",
        days=1825,
    )
    assert response.history, (
        "endpoint returned empty history (read-path did not surface the backfill)"
    )
    endpoint_months = sorted({item.recorded_at.strftime("%Y-%m") for item in response.history})
    assert len(endpoint_months) >= 3, f"endpoint history is not multi-month: {endpoint_months}"
