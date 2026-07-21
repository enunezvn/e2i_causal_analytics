"""run_persistence_eval — real-DB end-to-end gold-standard pipeline for the
PERSISTENCE and DISCONTINUATION cohorts (P2-T5).

This is the persistence-cohort counterpart to ``run_initiation_eval``.  It wires
the same committed primitives (:class:`FeatureBuilder`, :func:`train_cohort_model`,
:func:`serialize_model`, :func:`register_cohort_model`, :class:`WalkForwardRunner`,
:class:`MetricRecorder`) into ONE end-to-end run for BOTH the
``pnh_persistence`` and ``pnh_discontinuation`` targets against the REAL
synthetic cohort in the self-hosted Supabase docker DB, and records the results
so the monitoring Time-Series page reads a real, multi-month trend for each.

What it does (all on real data, all idempotent — re-run safe)
------------------------------------------------------------
For EACH cohort (persistence, then discontinuation):

1. Resolve the faithful async client (``get_async_supabase_client`` — reaches the
   docker DB that holds the real synthetic holdout rows). Fail-closed: a ``None``
   client raises rather than silently no-op'ing into fabricated success.
2. Load the full cohort frame once (``FeatureBuilder.load_frame(splits=None)``),
   selecting the cohort-specific label column (``persistent_180d`` for persistence,
   ``discontinued_180d`` for discontinuation). The two cohorts share the same base
   patient_journeys rows but each selects its OWN outcome column.
3. Train + register the champion: FIT a :class:`FeatureBuilder` on the
   ``train`` + ``validation`` rows, ``train_cohort_model`` a calibrated LR on that
   encoded frame, ``serialize_model`` a real artifact, and ``register_cohort_model``
   it at ``stage='staging'`` (collision-safe vs the serving champion).
4. Walk-forward over the full frame → ~36 monthly out-of-sample AUC points.
5. Record the walk-forward trend (``source='backtest_wf'``, ``split_version=None``).
6. Holdout headline: score the registered champion on the holdout rows and record
   ONE point (``source='holdout'``, ``split_version=None``).

After both cohorts are recorded, a complement-validation log check is emitted:
``persistent_180d == 1 - discontinued_180d`` in the synthetic DGP, so the two
holdout AUCs should be very close.  A warning is logged if they diverge by more
than 0.05 (this is a signal, not a failure).

``split_version`` is ``None`` on BOTH records on purpose. See
``run_initiation_eval`` docstring for the full idempotency rationale.

Run as a CLI on the target box::

    E2I_DB_INTEGRATION=1 python -m src.mlops.gold_standard_eval.run_persistence_eval
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.mlops.gold_standard_eval.cohort_spec import DISCONTINUATION, PERSISTENCE
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
from src.mlops.gold_standard_eval.recorder import HOLDOUT_CURVE_SOURCE, MetricRecorder
from src.mlops.gold_standard_eval.scorer import (
    calibration_slope_ci,
    holdout_curve_records,
    score,
)
from src.mlops.gold_standard_eval.walk_forward import WalkForwardRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-cohort model / experiment constants
# ---------------------------------------------------------------------------
PERSISTENCE_MODEL_NAME = "pnh_persistence_goldstd_lr_v1"
PERSISTENCE_EXPERIMENT_NAME = "persistence_goldstd_eval_v1"
DISCONTINUATION_MODEL_NAME = "pnh_discontinuation_goldstd_lr_v1"
DISCONTINUATION_EXPERIMENT_NAME = "discontinuation_goldstd_eval_v1"
GOLDSTD_MODEL_VERSION = "1.0"

# Train the champion on these splits (everything except the held-out test +
# holdout). The holdout headline scores STRICTLY out-of-sample on `holdout`.
_CHAMPION_TRAIN_SPLITS = ("train", "validation")
_HOLDOUT_SPLIT = "holdout"

# Writable artifact directory (the prod api container mounts a named volume here;
# /app/data itself is read-only — see #857). Resolved relative to the repo root so
# it is stable regardless of the process CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Artifacts live under a per-target subdir (data/ml_artifacts/<spec.target>/) so the
# persistence + discontinuation pickles are organised by cohort, not dumped into the
# initiation dir. serialize_model mkdirs the subdir on the writable volume.
_ARTIFACT_BASE = _REPO_ROOT / "data" / "ml_artifacts"

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
            "persistence eval (refusing to run a no-op).",
        )
    return client


async def _run_one_cohort(
    client: Any,
    spec: Any,
    *,
    model_name: str,
    experiment_name: str,
) -> dict[str, Any]:
    """Run the full gold-standard pipeline for one cohort.

    Performs the train→holdout-headline→clear-dependent-metrics→serialize→
    register→walk-forward→record sequence for the given ``spec`` / ``model_name``
    / ``experiment_name``.  Mirrors ``run_initiation_eval.run()`` exactly but is
    parametrized by the caller so it can be reused for both persistence and
    discontinuation without duplication.

    Parameters
    ----------
    client:
        Faithful async Supabase client (already resolved by the caller).
    spec:
        A :class:`~src.mlops.gold_standard_eval.cohort_spec.CohortSpec`
        (``PERSISTENCE`` or ``DISCONTINUATION``).
    model_name:
        The ``ml_model_registry.model_name`` for this cohort's eval model.
    experiment_name:
        The ``ml_experiments.experiment_name`` for this cohort's eval run.

    Returns
    -------
    dict with keys:
        ``model`` (str), ``holdout_auc`` (float), ``backtest_points`` (int),
        ``n_train`` (int), ``n_holdout`` (int).
    """
    from src.mlops.gold_standard_eval.cohort_deployer import (
        register_cohort_model,
        serialize_model,
        train_cohort_model,
    )
    from src.repositories.drift_monitoring import (
        PerformanceMetricRepository,
        _resolve_model_id,
    )

    # --- 1. Load the full cohort frame once (all splits, all months). -------- #
    fb_full = FeatureBuilder(spec)
    frame = await fb_full.load_frame(client, splits=None)
    if frame.empty:
        raise RuntimeError(
            f"run_persistence_eval: load_frame returned an empty frame for "
            f"cohort={spec.name!r} brand={spec.brand!r} is_synthetic=True — "
            "refusing to fabricate a result. "
            "(Is the synthetic cohort seeded in this DB?)"
        )
    n_frame = int(len(frame))
    logger.info(
        "[%s] Loaded full frame: %d rows, %d months.",
        spec.name,
        n_frame,
        frame["journey_start_date"].nunique(),
    )

    # --- 2. Train champion on train+validation. ------------------------------- #
    train_mask = frame["data_split"].isin(_CHAMPION_TRAIN_SPLITS)
    champion_frame = frame.loc[train_mask]
    if champion_frame.empty:
        raise RuntimeError(
            f"run_persistence_eval [{spec.name}]: no train/validation rows found "
            f"(data_split in {_CHAMPION_TRAIN_SPLITS}); cannot train a champion."
        )
    n_train = int(len(champion_frame))

    champion_fb = FeatureBuilder(spec)
    x_train, y_train = champion_fb.build_from_frame(champion_frame)  # FIT
    champion = train_cohort_model(spec, x_train, y_train)
    logger.info(
        "[%s] Trained champion on %d train+val rows (%d encoded features).",
        spec.name,
        n_train,
        len(champion_fb.feature_columns),
    )

    # --- 3. Holdout headline AUC (compute before registering). --------------- #
    # The holdout AUC is BOTH the registry `auc` (honest: the model's real
    # held-out performance) AND the recorded 'holdout' headline point.
    holdout_frame = frame.loc[frame["data_split"] == _HOLDOUT_SPLIT]
    if holdout_frame.empty:
        raise RuntimeError(
            f"run_persistence_eval [{spec.name}]: no '{_HOLDOUT_SPLIT}' rows "
            "found; cannot compute the holdout headline."
        )
    n_holdout = int(len(holdout_frame))
    x_holdout = champion_fb.transform(holdout_frame)  # APPLY (aligned to fit)
    y_holdout = holdout_frame[spec.label_column].astype(int).to_numpy()
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
    logger.info("[%s] Holdout headline: n=%d auc_roc=%.4f", spec.name, n_holdout, holdout_auc)

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
    prior_model_id = await _resolve_model_id(client, model_name)
    if prior_model_id is not None:
        cleared = 0
        for src in (_BACKTEST_SOURCE, _HOLDOUT_SOURCE, HOLDOUT_CURVE_SOURCE):
            cleared += await repo.delete_metrics(prior_model_id, src, None)
        logger.info(
            "[%s] Re-run cleanup: cleared %d dependent metric row(s) for prior "
            "model_id=%r before re-registering.",
            spec.name,
            cleared,
            prior_model_id,
        )

    # Serialize a real artifact (loadability/honesty) and register at staging.
    artifact_path = serialize_model(champion, _ARTIFACT_BASE / spec.target, model_name)
    model_handle = await register_cohort_model(
        client,
        spec,
        model_name=model_name,
        experiment_name=experiment_name,
        artifact_path=artifact_path,
        auc=holdout_auc,
        feature_count=len(champion_fb.feature_columns),
        training_samples=n_train,
    )
    logger.info("[%s] Registered champion handle=%r (staging).", spec.name, model_handle)

    # --- 4. Walk-forward over the full frame → monthly OOS points. ----------- #
    runner = WalkForwardRunner(spec)  # expanding window (experiment default)
    points = runner.run(frame)
    n_backtest_points = len(points)
    logger.info(
        "[%s] Walk-forward emitted %d point(s); skipped %d.",
        spec.name,
        n_backtest_points,
        len(runner.skipped),
    )

    # --- 5. Record the walk-forward trend (idempotent; split_version=None). -- #
    await recorder.record_run(
        model_handle,
        points,
        source=_BACKTEST_SOURCE,
        split_version=None,
    )

    # --- 6. Holdout headline as ONE point (source='holdout'). ---------------- #
    # The measured_at is the DATA BOUNDARY (latest holdout journey_start_date),
    # not now() — so the point plots at the end of the real cohort timeline rather
    # than at the current wall-clock date, which may be months/years later.
    import pandas as _pd

    latest = _pd.to_datetime(holdout_frame["journey_start_date"]).max()
    holdout_ts = latest.to_pydatetime().replace(tzinfo=timezone.utc)
    # B2: bootstrap percentile CI for the holdout calibration slope, written
    # into the calibration_slope row's existing ci_lower/ci_upper columns
    # (n rides the row's sample_size). None when unfittable — never fabricated.
    slope_ci = calibration_slope_ci(y_holdout, y_score)
    await recorder.record_run(
        model_handle,
        [(holdout_ts, holdout_metrics, n_holdout)],
        source=_HOLDOUT_SOURCE,
        split_version=None,
        cis=(
            {"calibration_slope": (slope_ci["ci_lower"], slope_ci["ci_upper"])}
            if slope_ci
            else None
        ),
    )

    # --- 6b. Holdout confusion matrix + ROC curve (source='holdout_curve'). --- #
    # Computed from the SAME y_holdout / y_score as the headline metrics and
    # persisted under a disjoint source so the scalar 'holdout' rows are untouched.
    await recorder.record_curves(
        model_handle,
        holdout_curve_records(y_holdout, y_score),
        measured_at=holdout_ts,
        sample_size=n_holdout,
    )

    return {
        "model": model_handle,
        "holdout_auc": holdout_auc,
        "backtest_points": n_backtest_points,
        "n_train": n_train,
        "n_holdout": n_holdout,
    }


async def run(db: Any = None) -> dict[str, Any]:
    """Run the full real-DB persistence + discontinuation gold-standard pipeline.

    Runs ``_run_one_cohort`` sequentially for PERSISTENCE then DISCONTINUATION,
    then emits a complement-validation log check (the two AUCs should be close
    since ``persistent_180d == 1 − discontinued_180d`` in the synthetic DGP).

    Parameters
    ----------
    db:
        Optional async Supabase client. When None the faithful docker client is
        resolved (fail-closed). Tests pass the same client they assert against.

    Returns
    -------
    dict::

        {
          "persistence": {
            "model": str,
            "holdout_auc": float,
            "backtest_points": int,
            "n_train": int,
            "n_holdout": int,
          },
          "discontinuation": { ... same shape ... },
        }
    """
    client = await _resolve_client(db)

    pers_result = await _run_one_cohort(
        client,
        PERSISTENCE,
        model_name=PERSISTENCE_MODEL_NAME,
        experiment_name=PERSISTENCE_EXPERIMENT_NAME,
    )

    disc_result = await _run_one_cohort(
        client,
        DISCONTINUATION,
        model_name=DISCONTINUATION_MODEL_NAME,
        experiment_name=DISCONTINUATION_EXPERIMENT_NAME,
    )

    # Complement validation: persistent_180d == 1 - discontinued_180d in the
    # synthetic DGP, so mirror models should yield very similar holdout AUCs.
    pers_auc = pers_result["holdout_auc"]
    disc_auc = disc_result["holdout_auc"]
    logger.info(
        "Complement validation: persistence holdout_auc=%.4f  "
        "discontinuation holdout_auc=%.4f  delta=%.4f",
        pers_auc,
        disc_auc,
        abs(pers_auc - disc_auc),
    )
    if abs(pers_auc - disc_auc) > 0.05:
        logger.warning(
            "Complement AUC divergence > 0.05 (persistence=%.4f, "
            "discontinuation=%.4f): mirror models should match; investigate "
            "data imbalance or feature encoding drift.",
            pers_auc,
            disc_auc,
        )

    return {
        "persistence": pers_result,
        "discontinuation": disc_result,
    }


def _print_report(report: dict[str, Any]) -> None:
    logger.info("=== gold-standard persistence + discontinuation eval report ===")
    for cohort in ("persistence", "discontinuation"):
        sub = report.get(cohort, {})
        logger.info("  [%s]", cohort)
        for key in ("model", "holdout_auc", "backtest_points", "n_train", "n_holdout"):
            logger.info("    %-20s %s", key, sub.get(key))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-DB gold-standard PERSISTENCE + DISCONTINUATION eval "
            "(walk-forward + holdout for both cohorts)."
        )
    )
    parser.parse_args()
    report = asyncio.run(run())
    _print_report(report)


if __name__ == "__main__":
    main()
