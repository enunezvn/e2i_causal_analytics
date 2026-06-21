"""Backfill pr_auc / brier_score / calibration_slope onto the 12 gold-standard
models' holdout rows WITHOUT a full eval re-run.

A full re-run retrains AND re-registers each model, which trips the
``ml_drift_history`` RESTRICT FK (23503). This script avoids that: it re-scores
each model's holdout and records the (now-extended) metrics against the EXISTING
model_id (resolved by name) via ``MetricRecorder.record_run`` — no
``register_cohort_model`` call, so no FK trip.

WHY THIS IS FAITHFUL (not a fresh, divergent model): the gold-standard cohort
frame is unchanged since registration and ``FeatureBuilder`` fit +
``train_cohort_model`` are deterministic, so a re-train reproduces the registered
champion EXACTLY (verified: holdout AUC/accuracy delta 0.00000). As a hard guard,
each model's recomputed accuracy + auc_roc MUST match the stored holdout values
within ``TOL`` before anything is written; a drifted model is SKIPPED, never
overwritten. ``record_run(source='holdout')`` delete-then-inserts the holdout
SCALAR rows (the 5 recomputed ~= originals + 3 new); the disjoint
``holdout_curve`` (confusion/ROC) source is untouched. Idempotent + re-runnable.

Run from a checkout with the EXTENDED scorer + a loaded ``.env``:
  .venv/bin/python scripts/backfill_goldstd_holdout_metrics.py --dry-run
  .venv/bin/python scripts/backfill_goldstd_holdout_metrics.py
  .venv/bin/python scripts/backfill_goldstd_holdout_metrics.py --model initiation_kisqali_goldstd_lr_v1
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import sys
from datetime import timezone

import numpy as np
import pandas as pd

SEED = 42
TOL = 1e-3  # recomputed accuracy/auc must match stored within this to be faithful
NEW_METRICS = ("pr_auc", "brier_score", "calibration_slope")


def _specs() -> list[tuple[str, object]]:
    """The 12 (model_name, CohortSpec) slots: 9 patient + 3 HCP."""
    from src.mlops.gold_standard_eval.cohort_spec import (
        BRANDS,
        HCP_ADOPTION_COHORT,
        PATIENT_COHORTS,
        goldstd_model_name,
        make_hcp_spec,
        make_patient_spec,
    )

    out: list[tuple[str, object]] = []
    for cohort in PATIENT_COHORTS:
        for brand in BRANDS:
            out.append((goldstd_model_name(cohort, brand), make_patient_spec(cohort, brand)))
    for brand in BRANDS:
        out.append((goldstd_model_name(HCP_ADOPTION_COHORT, brand), make_hcp_spec(brand)))
    return out


async def _rescore_one(client, spec):
    """Deterministic re-train + holdout re-score. Returns (metrics, ts, n) or (None, reason)."""
    from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
    from src.mlops.gold_standard_eval.scorer import score

    np.random.seed(SEED)
    fb_full = FeatureBuilder(spec)
    frame = await fb_full.load_frame(client, splits=None)
    if frame.empty:
        return None, "empty_frame"
    champ_frame = frame.loc[frame["data_split"].isin(("train", "validation"))]
    if champ_frame.empty:
        return None, "no_train_rows"
    cfb = FeatureBuilder(spec)
    x_tr, y_tr = cfb.build_from_frame(champ_frame)
    champ = train_cohort_model(spec, x_tr, y_tr)

    hold = frame.loc[frame["data_split"] == "holdout"]
    if hold.empty:
        return None, "empty_holdout"
    x_h = cfb.transform(hold)
    y_h = hold[spec.label_column].astype(int).to_numpy()
    proba = champ.predict_proba(x_h.to_numpy(dtype=float))
    if proba.shape[1] == 1:
        y_s = np.full(len(y_h), float(champ.classes_[0]))
    else:
        pos = list(champ.classes_).index(1) if 1 in champ.classes_ else 0
        y_s = proba[:, pos]
    metrics = score(y_h, y_s)
    ts = (
        pd.to_datetime(hold["journey_start_date"]).max().to_pydatetime().replace(tzinfo=timezone.utc)
    )
    return (metrics, ts, int(len(y_h))), None


async def _stored_holdout(client, model_name):
    from src.repositories.drift_monitoring import _resolve_model_id

    mid = await _resolve_model_id(client, model_name)
    if mid is None:
        return None
    res = await (
        client.table("ml_performance_metrics")
        .select("metric_name,metric_value")
        .eq("model_id", mid)
        .eq("source", "holdout")
        .execute()
    )
    return {r["metric_name"]: float(r["metric_value"]) for r in (res.data or [])}


async def main(dry_run: bool, only: str | None) -> int:
    from src.mlops.gold_standard_eval.recorder import MetricRecorder
    from src.repositories.drift_monitoring import (
        PerformanceMetricRepository,
        get_drift_monitoring_client,
    )

    client = await get_drift_monitoring_client()
    recorder = MetricRecorder(PerformanceMetricRepository(client))
    specs = [(n, s) for (n, s) in _specs() if only is None or n == only]
    print(f"backfill goldstd holdout extras: {len(specs)} model(s); dry_run={dry_run}")
    ok = skipped = failed = 0
    for model_name, spec in specs:
        try:
            res, err = await _rescore_one(client, spec)
            if err:
                print(f"  SKIP {model_name}: {err}")
                skipped += 1
                continue
            metrics, ts, n = res
            stored = await _stored_holdout(client, model_name)
            if not stored:
                print(f"  SKIP {model_name}: not registered / no stored holdout")
                skipped += 1
                continue
            d_auc = abs(metrics["auc_roc"] - stored.get("auc_roc", -9.0))
            d_acc = abs(metrics["accuracy"] - stored.get("accuracy", -9.0))
            if d_auc > TOL or d_acc > TOL:
                print(
                    f"  SKIP {model_name}: DRIFT auc_d={d_auc:.4f} acc_d={d_acc:.4f} "
                    "(refusing to overwrite a divergent re-train)"
                )
                skipped += 1
                continue
            extras = {k: round(float(metrics[k]), 4) for k in NEW_METRICS if k in metrics}
            if dry_run:
                print(f"  OK(dry) {model_name}: faithful (auc_d={d_auc:.5f}); would add {extras}")
                ok += 1
                continue
            await recorder.record_run(
                model_name, [(ts, metrics, n)], source="holdout", split_version=None
            )
            print(f"  DONE {model_name}: recorded {len(metrics)} holdout metrics incl {extras}")
            ok += 1
        except Exception as e:  # noqa: BLE001 — one model's failure must not abort the rest
            print(f"  FAIL {model_name}: {type(e).__name__}: {str(e)[:160]}")
            failed += 1
        finally:
            gc.collect()
    print(f"summary: ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="re-score + faithfulness-check only; no writes")
    ap.add_argument("--model", default=None, help="restrict to one model_name")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.dry_run, args.model)))
