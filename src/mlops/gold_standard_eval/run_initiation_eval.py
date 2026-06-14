"""run_initiation_eval — the real-DB end-to-end gold-standard INITIATION pipeline (T11).

This is the culmination of P1: it wires the committed primitives
(:class:`FeatureBuilder`, :func:`train_cohort_model`, :func:`serialize_model`,
:func:`register_cohort_model`, :class:`WalkForwardRunner`,
:class:`MetricRecorder`) into ONE end-to-end run against the REAL synthetic
Remibrutinib cohort in the self-hosted Supabase docker DB, and records the
results so the monitoring Time-Series page reads a real, multi-month trend.

What it does (all on real data, all idempotent — re-run safe)
------------------------------------------------------------
1. Resolve the faithful async client (``get_async_supabase_client`` — reaches the
   docker DB that holds the real synthetic holdout rows). Fail-closed: a ``None``
   client raises rather than silently no-op'ing into fabricated success.
2. Load the full brand frame once (``FeatureBuilder.load_frame(splits=None)``).
3. Train + register the champion: FIT a :class:`FeatureBuilder` on the
   ``train`` + ``validation`` rows, ``train_cohort_model`` a calibrated LR on that
   encoded frame, ``serialize_model`` a real artifact, and ``register_cohort_model``
   it at ``stage='staging'`` (collision-safe vs the serving champion).
4. Walk-forward over the full frame → ~36 monthly out-of-sample AUC points.
5. Record the walk-forward trend (``source='backtest_wf'``, ``split_version=None``).
6. Holdout headline: score the registered champion on the holdout rows and record
   ONE point (``source='holdout'``, ``split_version=None``).

``split_version`` is ``None`` on BOTH records on purpose. The recorder's delete
step (idempotency) is keyed by ``(model_id, source)`` and DOES filter on
``metadata->>split_version`` when a split_version is passed — but
``record_metrics`` does not write split_version into the row metadata. So passing
a split_version would make the delete filter on a key that is never written → the
delete would match nothing → a re-run would DUPLICATE rows. Keeping it ``None``
keeps the delete scope equal to the insert scope, preserving idempotency.

Run as a CLI on the target box::

    E2I_DB_INTEGRATION=1 python -m src.mlops.gold_standard_eval.run_initiation_eval
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
from src.mlops.gold_standard_eval.recorder import MetricRecorder
from src.mlops.gold_standard_eval.scorer import score
from src.mlops.gold_standard_eval.walk_forward import WalkForwardRunner

logger = logging.getLogger(__name__)

# Train the champion on these splits (everything except the held-out test +
# holdout). The holdout headline scores STRICTLY out-of-sample on `holdout`.
_CHAMPION_TRAIN_SPLITS = ("train", "validation")
_HOLDOUT_SPLIT = "holdout"

# Writable artifact directory (the prod api container mounts a named volume here;
# /app/data itself is read-only — see #857). Resolved relative to the repo root so
# it is stable regardless of the process CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_DIR = _REPO_ROOT / "data" / "ml_artifacts" / "csu_treatment_initiation"

_BACKTEST_SOURCE = "backtest_wf"
_HOLDOUT_SOURCE = "holdout"


async def _resolve_client(db: Any) -> Any:
    """Return the faithful async Supabase client (the docker DB), fail-closed.

    When ``db`` is None we resolve ``get_async_supabase_client`` — which itself
    raises ``ServiceConnectionError`` if Supabase is unconfigured. We additionally
    refuse a ``None`` result so this pipeline never silently degrades into a no-op
    that would fabricate an empty success (the #845 / #840 fail-open trap).
    """
    if db is not None:
        return db
    from src.memory.services.factories import (
        ServiceConnectionError,
        get_async_supabase_client,
    )

    client = await get_async_supabase_client()
    if client is None:
        raise ServiceConnectionError(
            "Supabase",
            "async Supabase client resolved to None for the gold-standard "
            "initiation eval (refusing to run a no-op).",
        )
    return client


async def run(db: Any = None) -> dict[str, Any]:
    """Run the full real-DB initiation gold-standard pipeline; return a report.

    Parameters
    ----------
    db:
        Optional async Supabase client. When None the faithful docker client is
        resolved (fail-closed). Tests pass the same client they assert against.

    Returns
    -------
    dict with at least::

        {
          "champion_registered": bool,
          "model_handle": str,
          "n_backtest_points": int,
          "backtest_months": list[str],
          "holdout_auc": float,
          "n_holdout": int,
          "n_train_champion": int,
          "n_frame": int,
          "artifact_path": str,
          "skipped_months": int,
        }
    """
    from src.mlops.gold_standard_eval.cohort_deployer import (
        GOLDSTD_MODEL_NAME,
        register_cohort_model,
        serialize_model,
        train_cohort_model,
    )
    from src.repositories.drift_monitoring import (
        PerformanceMetricRepository,
        _resolve_model_id,
    )

    client = await _resolve_client(db)

    # --- 2. Load the full brand frame once (all splits, all months). --------- #
    fb_full = FeatureBuilder(INITIATION)
    frame = await fb_full.load_frame(client, splits=None)
    if frame.empty:
        raise RuntimeError(
            "run_initiation_eval: load_frame returned an empty frame for "
            f"brand={INITIATION.brand!r} is_synthetic=True — refusing to fabricate "
            "a result. (Is the synthetic cohort seeded in this DB?)"
        )
    n_frame = int(len(frame))
    logger.info(
        "Loaded full frame: %d rows, %d months.", n_frame, frame["journey_start_date"].nunique()
    )

    # --- 3. Train + register the champion on train+validation. --------------- #
    train_mask = frame["data_split"].isin(_CHAMPION_TRAIN_SPLITS)
    champion_frame = frame.loc[train_mask]
    if champion_frame.empty:
        raise RuntimeError(
            "run_initiation_eval: no train/validation rows found "
            f"(data_split in {_CHAMPION_TRAIN_SPLITS}); cannot train a champion."
        )
    n_train_champion = int(len(champion_frame))

    champion_fb = FeatureBuilder(INITIATION)
    x_train, y_train = champion_fb.build_from_frame(champion_frame)  # FIT
    champion = train_cohort_model(INITIATION, x_train, y_train)
    logger.info(
        "Trained champion on %d train+val rows (%d encoded features).",
        n_train_champion,
        len(champion_fb.feature_columns),
    )

    # --- 6 (compute first). Holdout headline AUC for the registered champion. -#
    # The holdout AUC is BOTH the registry `auc` (honest: the model's real
    # held-out performance) AND the recorded 'holdout' headline point.
    holdout_frame = frame.loc[frame["data_split"] == _HOLDOUT_SPLIT]
    if holdout_frame.empty:
        raise RuntimeError(
            f"run_initiation_eval: no '{_HOLDOUT_SPLIT}' rows found; cannot "
            "compute the holdout headline."
        )
    n_holdout = int(len(holdout_frame))
    x_holdout = champion_fb.transform(holdout_frame)  # APPLY (aligned to fit)
    y_holdout = holdout_frame[INITIATION.label_column].astype(int).to_numpy()
    proba = champion.predict_proba(x_holdout.to_numpy(dtype=float))
    if proba.shape[1] == 1:
        # Degenerate single-class fit: positive-class score is constant.
        only_class = int(champion.classes_[0])
        y_score = np.full(n_holdout, float(only_class))
    else:
        pos_idx = list(champion.classes_).index(1) if 1 in champion.classes_ else 0
        y_score = proba[:, pos_idx]
    holdout_metrics = score(y_holdout, y_score)
    holdout_auc = float(holdout_metrics["auc_roc"])
    logger.info("Holdout headline: n=%d auc_roc=%.4f", n_holdout, holdout_auc)

    repo = PerformanceMetricRepository(client)
    recorder = MetricRecorder(repo)

    # --- Re-run safety: clear dependent metric rows BEFORE replacing the
    #     registry row. ----------------------------------------------------- #
    # register_cohort_model -> register_model_row DELETEs the existing
    # (model_name, model_version) registry row and re-INSERTs it with a NEW
    # gen_random_uuid() id. ml_performance_metrics.model_id REFERENCES
    # ml_model_registry(id) with NO ON DELETE CASCADE (RESTRICT), so on a re-run
    # the prior run's metric rows still point at the OLD registry id and the
    # registry DELETE 23503's. We clear those dependent rows first (same scope
    # the recorder uses for idempotency: model_id + source), which unblocks the
    # registry replace AND keeps the whole pipeline re-run safe. On the very
    # first run this resolves to no prior id and is a harmless no-op.
    prior_model_id = await _resolve_model_id(client, GOLDSTD_MODEL_NAME)
    if prior_model_id is not None:
        cleared = 0
        for src in (_BACKTEST_SOURCE, _HOLDOUT_SOURCE):
            cleared += await repo.delete_metrics(prior_model_id, src, None)
        logger.info(
            "Re-run cleanup: cleared %d dependent metric row(s) for prior "
            "model_id=%r before re-registering.",
            cleared,
            prior_model_id,
        )

    # Serialize a real artifact (loadability/honesty) and register at staging.
    artifact_path = serialize_model(champion, _ARTIFACT_DIR, GOLDSTD_MODEL_NAME)
    model_handle = await register_cohort_model(
        client,
        INITIATION,
        artifact_path=artifact_path,
        auc=holdout_auc,
        feature_count=len(champion_fb.feature_columns),
        training_samples=n_train_champion,
    )
    champion_registered = model_handle == GOLDSTD_MODEL_NAME
    logger.info("Registered champion handle=%r (staging).", model_handle)

    # --- 4. Walk-forward over the full frame → monthly OOS points. ----------- #
    runner = WalkForwardRunner(INITIATION)  # expanding window (experiment default)
    points = runner.run(frame)
    n_backtest_points = len(points)
    backtest_months = sorted({p[0].strftime("%Y-%m") for p in points})
    logger.info(
        "Walk-forward emitted %d point(s) across %d month(s); skipped %d.",
        n_backtest_points,
        len(backtest_months),
        len(runner.skipped),
    )

    # --- 5. Record the walk-forward trend (idempotent; split_version=None). -- #
    await recorder.record_run(
        model_handle,
        points,
        source=_BACKTEST_SOURCE,
        split_version=None,
    )

    # --- 6 (record). Holdout headline as ONE point (source='holdout'). ------- #
    # The measured_at is the DATA BOUNDARY (latest holdout journey_start_date),
    # not now() — so the point plots at the end of the real cohort timeline rather
    # than at the current wall-clock date, which may be months/years later.
    import pandas as _pd

    latest = _pd.to_datetime(holdout_frame["journey_start_date"]).max()
    holdout_ts = latest.to_pydatetime().replace(tzinfo=timezone.utc)
    await recorder.record_run(
        model_handle,
        [(holdout_ts, holdout_metrics, n_holdout)],
        source=_HOLDOUT_SOURCE,
        split_version=None,
    )

    return {
        "champion_registered": champion_registered,
        "model_handle": model_handle,
        "n_backtest_points": n_backtest_points,
        "backtest_months": backtest_months,
        "holdout_auc": holdout_auc,
        "holdout_metrics": holdout_metrics,
        "n_holdout": n_holdout,
        "n_train_champion": n_train_champion,
        "n_frame": n_frame,
        "artifact_path": artifact_path,
        "skipped_months": len(runner.skipped),
    }


def _print_report(report: dict[str, Any]) -> None:
    logger.info("=== gold-standard initiation eval report ===")
    for key in (
        "champion_registered",
        "model_handle",
        "n_frame",
        "n_train_champion",
        "n_backtest_points",
        "holdout_auc",
        "n_holdout",
        "skipped_months",
        "artifact_path",
    ):
        logger.info("  %-22s %s", key, report.get(key))
    logger.info("  backtest_months         %s", report.get("backtest_months"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Run the real-DB gold-standard INITIATION eval (walk-forward + holdout)."
    )
    parser.parse_args()
    report = asyncio.run(run())
    _print_report(report)


if __name__ == "__main__":
    main()
