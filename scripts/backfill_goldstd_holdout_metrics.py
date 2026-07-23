"""Re-record the 12 gold-standard models' holdout headline on the OOS-union
window (``test`` + ``holdout``) WITHOUT a full eval re-run.

A full re-run retrains AND re-registers each model, which trips the
``ml_drift_history`` RESTRICT FK (23503). This script avoids that: it re-scores
each model's OOS-union window and records the metrics against the EXISTING
model_id (resolved by name) via ``MetricRecorder.record_run`` — no
``register_cohort_model`` call, so no FK trip. The registry row's ``auc``
column is refreshed with a plain UPDATE (no delete/insert, so no FK exposure).

OOS-UNION POLICY (2026-07-23): the headline scores every row the champion never
saw — ``data_split IN ('test','holdout')`` — matching ``_OOS_EVAL_SPLITS`` in
``run_persistence_eval``. Rationale lives there: single-window draws at the old
sizes (patient n~850, hcp n=250) made the calibration-slope KPI a window
lottery; the union doubles n and tightens every CI. Alongside the scalars this
script now also writes the calibration-slope bootstrap CI (the B2 columns) and
refreshes the disjoint ``holdout_curve`` rows (confusion + ROC) from the SAME
scores, so every holdout artifact describes the same window.

WHY THIS IS FAITHFUL (not a fresh, divergent model): the gold-standard cohort
frame is unchanged since registration and ``FeatureBuilder`` fit +
``train_cohort_model`` are deterministic, so a re-train reproduces the registered
champion EXACTLY (verified: holdout AUC/accuracy delta 0.00000). As a hard
guard, each re-trained model must reproduce the STORED holdout auc_roc/accuracy
within ``TOL`` on either the legacy ``holdout``-only window (rows recorded
before this policy) or the OOS union (rows recorded by a prior run of this
script) before anything is written; a drifted model is SKIPPED, never
overwritten. All writes are idempotent delete-then-inserts. Re-runnable.

Run from a checkout with a loaded ``.env``:
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
OOS_EVAL_SPLITS = ("test", "holdout")  # lockstep with run_persistence_eval._OOS_EVAL_SPLITS
LEGACY_EVAL_SPLITS = ("holdout",)  # pre-policy stored rows were scored on this window


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


def _window_scores(champ, fb, frame, spec, splits):
    """Score the champion on one data_split window; returns (y_true, y_score)."""
    sub = frame.loc[frame["data_split"].isin(splits)]
    if sub.empty:
        return None, None, None
    x = fb.transform(sub)
    y = sub[spec.label_column].astype(int).to_numpy()
    proba = champ.predict_proba(x.to_numpy(dtype=float))
    if proba.shape[1] == 1:
        y_s = np.full(len(y), float(champ.classes_[0]))
    else:
        pos = list(champ.classes_).index(1) if 1 in champ.classes_ else 0
        y_s = proba[:, pos]
    return sub, y, y_s


async def _rescore_one(client, spec):
    """Deterministic re-train + OOS-union re-score.

    Returns ``(result, None)`` or ``(None, reason)`` where result is a dict with
    the union metrics/CI/curves plus the legacy-window auc/accuracy used by the
    faithfulness guard.
    """
    from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
    from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
    from src.mlops.gold_standard_eval.scorer import (
        calibration_slope_ci,
        holdout_curve_records,
        score,
    )

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

    union, y_u, s_u = _window_scores(champ, cfb, frame, spec, OOS_EVAL_SPLITS)
    if union is None:
        return None, "empty_oos_window"
    metrics = score(y_u, s_u)
    slope_ci = calibration_slope_ci(y_u, s_u)
    curves = holdout_curve_records(y_u, s_u)
    ts = (
        pd.to_datetime(union["journey_start_date"])
        .max()
        .to_pydatetime()
        .replace(tzinfo=timezone.utc)
    )

    _, y_l, s_l = _window_scores(champ, cfb, frame, spec, LEGACY_EVAL_SPLITS)
    legacy = score(y_l, s_l) if y_l is not None else None
    return (
        {
            "metrics": metrics,
            "slope_ci": slope_ci,
            "curves": curves,
            "ts": ts,
            "n": int(len(y_u)),
            "legacy": legacy,
        },
        None,
    )


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


def _faithful(recomputed: dict | None, stored: dict) -> tuple[bool, float, float]:
    """Stored auc/accuracy reproduced within TOL by this recomputed window?"""
    if not recomputed:
        return False, 9.0, 9.0
    d_auc = abs(recomputed["auc_roc"] - stored.get("auc_roc", -9.0))
    d_acc = abs(recomputed["accuracy"] - stored.get("accuracy", -9.0))
    return (d_auc <= TOL and d_acc <= TOL), d_auc, d_acc


async def _update_registry_auc(client, model_name: str, auc: float) -> None:
    """Plain UPDATE of the registry row's auc (keeps id — no FK exposure)."""
    await (
        client.table("ml_model_registry")
        .update({"auc": round(float(auc), 4)})
        .eq("model_name", model_name)
        .execute()
    )


async def main(dry_run: bool, only: str | None) -> int:
    from src.mlops.gold_standard_eval.recorder import MetricRecorder
    from src.repositories.drift_monitoring import (
        PerformanceMetricRepository,
        get_drift_monitoring_client,
    )

    client = await get_drift_monitoring_client()
    recorder = MetricRecorder(PerformanceMetricRepository(client))
    specs = [(n, s) for (n, s) in _specs() if only is None or n == only]
    print(f"goldstd OOS-union holdout re-record: {len(specs)} model(s); dry_run={dry_run}")
    ok = skipped = failed = 0
    for model_name, spec in specs:
        try:
            res, err = await _rescore_one(client, spec)
            if err:
                print(f"  SKIP {model_name}: {err}")
                skipped += 1
                continue
            stored = await _stored_holdout(client, model_name)
            if not stored:
                print(f"  SKIP {model_name}: not registered / no stored holdout")
                skipped += 1
                continue
            # Faithfulness: the deterministic re-train must reproduce the stored
            # headline on the window it was recorded from — legacy holdout-only
            # (pre-policy rows) OR the OOS union (a prior run of this script).
            legacy_ok, dl_auc, dl_acc = _faithful(res["legacy"], stored)
            union_ok, du_auc, du_acc = _faithful(res["metrics"], stored)
            if not (legacy_ok or union_ok):
                print(
                    f"  SKIP {model_name}: DRIFT legacy(auc_d={dl_auc:.4f} acc_d={dl_acc:.4f}) "
                    f"union(auc_d={du_auc:.4f} acc_d={du_acc:.4f}) "
                    "(refusing to overwrite a divergent re-train)"
                )
                skipped += 1
                continue
            window = "union" if union_ok else "legacy"
            m = res["metrics"]
            slope = m.get("calibration_slope")
            summary = (
                f"n={res['n']} auc={m['auc_roc']:.4f} f1={m['f1']:.4f} "
                f"brier={m['brier_score']:.4f} slope={slope:.4f}"
                if slope is not None
                else f"n={res['n']} auc={m['auc_roc']:.4f} f1={m['f1']:.4f}"
            )
            if dry_run:
                print(f"  OK(dry) {model_name}: faithful({window}); would record {summary}")
                ok += 1
                continue
            await recorder.record_run(
                model_name,
                [(res["ts"], m, res["n"])],
                source="holdout",
                split_version=None,
                cis=(
                    {
                        "calibration_slope": (
                            res["slope_ci"]["ci_lower"],
                            res["slope_ci"]["ci_upper"],
                        )
                    }
                    if res["slope_ci"]
                    else None
                ),
            )
            await recorder.record_curves(
                model_name,
                res["curves"],
                measured_at=res["ts"],
                sample_size=res["n"],
            )
            await _update_registry_auc(client, model_name, m["auc_roc"])
            print(f"  DONE {model_name}: recorded faithful({window}) {summary}")
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
    ap.add_argument(
        "--dry-run", action="store_true", help="re-score + faithfulness-check only; no writes"
    )
    ap.add_argument("--model", default=None, help="restrict to one model_name")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.dry_run, args.model)))
