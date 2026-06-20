"""Real-DB end-to-end integration test for the gold-standard PERSISTENCE +
DISCONTINUATION eval (P2-T5).

Runs ``run_persistence_eval.run()`` against the LIVE synthetic cohort in the
self-hosted Supabase docker DB and asserts the REAL effects landed:
- Both cohort models are registered at staging.
- Each model has a multi-month walk-forward AUC trend (backtest_wf rows).
- Each model has exactly one holdout headline row.
- A re-run leaves both model handle row-counts unchanged (idempotency).
- Holdout AUCs are finite and sane (>= 0.5).

Gate: ``E2I_DB_INTEGRATION=1`` + a reachable async Supabase client
(``SUPABASE_URL`` + key). Mirrors ``test_gold_standard_initiation_eval.py`` so
unit-only CI lanes never touch the DB.

NO mocks — every assertion runs against real rows.  REAL prod writes are
authorized for this pipeline but MUST be idempotent (the test re-runs and
asserts no duplication).
"""

from __future__ import annotations

import math
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
# Fixture: fresh async client per test (mirrors test_gold_standard_initiation_eval)
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
async def test_persistence_eval_end_to_end_real_db() -> None:
    """Full pipeline: both cohort champions registered, multi-month trends, idempotent."""
    from src.memory.services.factories import get_async_supabase_client
    from src.mlops.gold_standard_eval.run_persistence_eval import (
        DISCONTINUATION_MODEL_NAME,
        PERSISTENCE_MODEL_NAME,
        run,
    )
    from src.repositories.drift_monitoring import (
        PerformanceMetricRepository,
        _resolve_model_id,
    )

    db = await get_async_supabase_client()
    repo = PerformanceMetricRepository(db)

    # --- Run the real pipeline (REAL prod writes; idempotent). -------------- #
    report = await run(db)

    assert "persistence" in report and "discontinuation" in report, (
        f"run() must return both sub-results: {report.keys()}"
    )

    for cohort_key, model_name in (
        ("persistence", PERSISTENCE_MODEL_NAME),
        ("discontinuation", DISCONTINUATION_MODEL_NAME),
    ):
        sub = report[cohort_key]

        # --- Sub-result shape ------------------------------------------------ #
        assert sub.get("backtest_points", 0) > 0, (
            f"[{cohort_key}] backtest_points must be > 0, got {sub}"
        )
        holdout_auc = sub.get("holdout_auc")
        assert holdout_auc is not None and math.isfinite(holdout_auc), (
            f"[{cohort_key}] holdout_auc must be finite, got {holdout_auc}"
        )
        assert holdout_auc >= 0.5, (
            f"[{cohort_key}] holdout_auc={holdout_auc:.4f} below 0.5 — "
            "degenerate model or data issue"
        )

        # --- Model handle resolves to a real registry uuid ------------------- #
        model_id = await _resolve_model_id(db, model_name)
        assert model_id is not None, (
            f"[{cohort_key}] registered handle {model_name!r} did not resolve to a model_id uuid"
        )

        # --- Real multi-month walk-forward trend landed ---------------------- #
        rows = await _rows_for_current_handle(repo, model_name, "backtest_wf")
        assert rows, f"[{cohort_key}] no backtest_wf auc_roc rows for {model_name!r}"

        months = sorted({r["measured_at"][:7] for r in rows})
        assert len(months) >= 3, (
            f"[{cohort_key}] expected >=3 distinct backtest months, got {months}"
        )

        # Real, sane AUCs.
        assert all(0.5 <= r["metric_value"] <= 1.0 for r in rows), (
            f"[{cohort_key}] backtest AUCs out of [0.5, 1.0]: "
            f"{sorted({round(r['metric_value'], 4) for r in rows})}"
        )

        # Report agrees with what landed.
        assert sub["backtest_points"] == len(rows), (
            f"[{cohort_key}] report backtest_points={sub['backtest_points']} "
            f"!= landed rows={len(rows)}"
        )

    # --- Idempotency: a second run does not duplicate for EITHER cohort ------ #
    n_before: dict[str, int] = {}
    for cohort_key, model_name in (
        ("persistence", PERSISTENCE_MODEL_NAME),
        ("discontinuation", DISCONTINUATION_MODEL_NAME),
    ):
        rows = await _rows_for_current_handle(repo, model_name, "backtest_wf")
        n_before[cohort_key] = len(rows)

    await run(db)

    for cohort_key, model_name in (
        ("persistence", PERSISTENCE_MODEL_NAME),
        ("discontinuation", DISCONTINUATION_MODEL_NAME),
    ):
        rows2 = await _rows_for_current_handle(repo, model_name, "backtest_wf")
        assert len(rows2) == n_before[cohort_key], (
            f"[{cohort_key}] idempotency broken: backtest_wf rows "
            f"{n_before[cohort_key]} -> {len(rows2)} after re-run"
        )

    # --- Holdout headline present (exactly 1 per cohort) --------------------- #
    for cohort_key, model_name in (
        ("persistence", PERSISTENCE_MODEL_NAME),
        ("discontinuation", DISCONTINUATION_MODEL_NAME),
    ):
        hold = await _rows_for_current_handle(repo, model_name, "holdout")
        assert len(hold) >= 1, f"[{cohort_key}] no holdout auc_roc row recorded for {model_name!r}"
        assert len(hold) == 1, (
            f"[{cohort_key}] expected exactly 1 holdout headline row, got {len(hold)}"
        )
        assert 0.5 <= hold[0]["metric_value"] <= 1.0, (
            f"[{cohort_key}] holdout AUC out of [0.5, 1.0]: {hold[0]['metric_value']}"
        )
