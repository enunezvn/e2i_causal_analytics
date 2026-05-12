"""v5 Gate B2 — Harrell c-index contrast vs binary val_AUC.

Acceptance per docs/specs/v5_b2_survival_modeling_prespec_2026-05-12.md
section 6:
- IMPROVEMENT: c_best - val_auc >= 0.03 on at least one cohort.
- NULL: |c_best - val_auc| < 0.03 on every cohort.
- REGRESSION: c_best - val_auc <= -0.03 on every cohort.

Decision rule per pre-spec §9 locked BEFORE this script runs.

Methodology mirrors B3 contrast (scripts/measure_b3_val_auc_contrast.py):
- Manifest-filtered pre-anchor feature surface (production parity).
- 5-fold StratifiedKFold on the event indicator (same seed both arms).
- Baseline: LogisticRegression(class_weight='balanced', max_iter=2000).
- Survival arms: Cox(alpha=1e-3) + RSF(n_estimators=100, min_samples_leaf=15).
- Pre-processing: SimpleImputer(median, keep_empty_features=True) + StandardScaler.
- Metric: Harrell c-index (sksurv.metrics.concordance_index_censored) vs roc_auc_score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


def _filter_to_manifest_safe(
    X: pd.DataFrame, manifest_source: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter X to manifest-declared pre-anchor columns (production parity).

    Identical to B3 contrast script's filter so the baseline AUC is on
    the same surface in both gates.
    """
    from src.data.manifests import MANIFEST_SOURCES

    lookup = MANIFEST_SOURCES.get(manifest_source)
    if lookup is None:
        return X, list(X.columns)

    safe_cols: List[str] = []
    dropped: List[str] = []
    for col in X.columns:
        contract = lookup(col)
        if contract is None:
            dropped.append(col)
            continue
        if contract.knowable_at.is_pre_or_at_index():
            safe_cols.append(col)
        else:
            dropped.append(col)
    return X[safe_cols].copy(), dropped


def _build_pipeline(seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )


def _cross_val(
    X: pd.DataFrame,
    y_binary: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> Dict[str, Any]:
    """Run 5-fold CV computing val_AUC (binary LR) + c-index (Cox + RSF) per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from src.agents.ml_foundation.model_trainer.nodes.survival_model import (
        fit_cox,
        fit_rsf,
        survival_concordance,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    c_cox: List[float] = []
    c_rsf: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X.values, y_binary)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr_bin, y_va_bin = y_binary[train_idx], y_binary[val_idx]
        t_tr, t_va = time[train_idx], time[val_idx]
        e_tr, e_va = event[train_idx], event[val_idx]

        # Skip degenerate folds.
        if len(np.unique(y_va_bin)) < 2:
            continue

        pre = _build_pipeline(seed)
        X_tr_pre = pre.fit_transform(X_tr.values)
        X_va_pre = pre.transform(X_va.values)

        # Wrap as DataFrames for the survival fitters (they call .values internally).
        X_tr_df = pd.DataFrame(X_tr_pre, columns=X_tr.columns)
        X_va_df = pd.DataFrame(X_va_pre, columns=X_va.columns)

        # Binary baseline.
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        lr.fit(X_tr_pre, y_tr_bin)
        auc = float(roc_auc_score(y_va_bin, lr.predict_proba(X_va_pre)[:, 1]))
        aucs.append(auc)

        # Cox.
        try:
            cox = fit_cox(X_tr_df, t_tr, e_tr, alpha=1e-3, seed=seed)
            c_c = survival_concordance(cox, X_va_df, t_va, e_va)
            c_cox.append(c_c)
        except Exception as exc:  # noqa: BLE001
            print(f"  fold {fold_idx}: Cox fit failed: {exc}", file=sys.stderr)
            c_cox.append(float("nan"))

        # RSF.
        try:
            rsf = fit_rsf(
                X_tr_df, t_tr, e_tr, n_estimators=100, min_samples_leaf=15, seed=seed
            )
            c_r = survival_concordance(rsf, X_va_df, t_va, e_va)
            c_rsf.append(c_r)
        except Exception as exc:  # noqa: BLE001
            print(f"  fold {fold_idx}: RSF fit failed: {exc}", file=sys.stderr)
            c_rsf.append(float("nan"))

    def _summary(vals: List[float]) -> Dict[str, Any]:
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            return {"mean": float("nan"), "std": float("nan"), "n_folds": 0, "per_fold": vals}
        return {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "n_folds": len(clean),
            "per_fold": [float(v) for v in vals],
        }

    return {
        "binary_auc": _summary(aucs),
        "cox_cindex": _summary(c_cox),
        "rsf_cindex": _summary(c_rsf),
    }


def _measure_cohort(
    cohort_label: str,
    seed: int,
    repo_root: Path,
) -> Dict[str, Any]:
    """Run the contrast for one cohort."""
    from scripts.run_tier1b_b2_experiment import (
        _build_features_and_target,
        _load_patient_journeys,
    )
    from src.agents.ml_foundation.model_trainer.nodes.survival_model import (
        derive_survival_target,
    )

    # Cohort -> (data_dir, manifest_source, treatment_events_loader).
    if cohort_label == "csu":
        data_dir = repo_root / "data" / "rwd" / "csu"
        manifest_source = "csu"
        ev_path = data_dir / "e2i_ml_v3_treatment_events.json"
        if ev_path.exists():
            with open(ev_path) as f:
                ev = pd.DataFrame(json.load(f))
        else:
            ev = None
    elif cohort_label == "optum_initiation":
        data_dir = repo_root / "data" / "rwd" / "optum" / "initiation"
        manifest_source = "optum"
        ev_path = data_dir / "e2i_ml_v3_treatment_events.parquet"
        ev = pd.read_parquet(ev_path) if ev_path.exists() else None
    else:
        raise ValueError(f"unknown cohort_label={cohort_label!r}")

    if not data_dir.exists():
        return {"skipped": True, "reason": f"data dir absent: {data_dir}"}

    pj = _load_patient_journeys(data_dir)
    X_raw, y = _build_features_and_target(pj, target_col="treatment_initiated")
    X_baseline, dropped = _filter_to_manifest_safe(X_raw, manifest_source)

    print(
        f"[{cohort_label}] baseline surface: {X_baseline.shape[1]} pre-anchor features "
        f"(dropped {len(dropped)} non-manifest/post-anchor cols)"
    )

    time, event = derive_survival_target(pj, manifest_source, treatment_events=ev)
    y_binary = y.to_numpy()
    n_events = int(event.sum())
    print(
        f"[{cohort_label}] n={len(time)} n_events={n_events} "
        f"event_time_median={np.median(time[event]):.1f}d "
        f"censored_time_median={np.median(time[~event]):.1f}d"
    )

    metrics = _cross_val(X_baseline, y_binary, time, event, seed=seed)
    auc_mean = metrics["binary_auc"]["mean"]
    c_cox_mean = metrics["cox_cindex"]["mean"]
    c_rsf_mean = metrics["rsf_cindex"]["mean"]
    c_best = max(c_cox_mean, c_rsf_mean) if not (np.isnan(c_cox_mean) and np.isnan(c_rsf_mean)) else float("nan")
    delta_cox = c_cox_mean - auc_mean if not np.isnan(c_cox_mean) else float("nan")
    delta_rsf = c_rsf_mean - auc_mean if not np.isnan(c_rsf_mean) else float("nan")
    delta_best = c_best - auc_mean if not np.isnan(c_best) else float("nan")
    if np.isnan(delta_best):
        verdict = "error"
    elif delta_best >= 0.03:
        verdict = "improvement"
    elif delta_best <= -0.03:
        verdict = "regression"
    else:
        verdict = "null"

    print(
        f"[{cohort_label}] binary_auc={auc_mean:.4f} cox_c={c_cox_mean:.4f} rsf_c={c_rsf_mean:.4f}"
    )
    print(
        f"[{cohort_label}] delta_cox={delta_cox:+.4f} delta_rsf={delta_rsf:+.4f} "
        f"delta_best={delta_best:+.4f} -> {verdict} (threshold >= 0.03)"
    )

    return {
        "cohort": cohort_label,
        "manifest_source": manifest_source,
        "n_rows": int(len(time)),
        "n_events": n_events,
        "metrics": metrics,
        "delta_cox": delta_cox,
        "delta_rsf": delta_rsf,
        "delta_best": delta_best,
        "verdict": verdict,
        "acceptance_threshold": 0.03,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/calibration/b2_cindex_contrast_20260512.json",
    )
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cohort_order = ["csu", "optum_initiation"]
    report: Dict[str, Any] = {}

    for label in cohort_order:
        try:
            report[label] = _measure_cohort(label, seed=args.seed, repo_root=repo_root)
        except FileNotFoundError as exc:
            if args.skip_missing:
                print(f"[{label}] SKIP — {exc}")
                report[label] = {"skipped": True, "reason": str(exc)}
                continue
            print(f"[{label}] FAIL — {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] ERROR — {exc}", file=sys.stderr)
            report[label] = {"error": str(exc)}
            return 1

    output_path = repo_root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {output_path}")

    improvements = [k for k, r in report.items() if isinstance(r, dict) and r.get("verdict") == "improvement"]
    nulls = [k for k, r in report.items() if isinstance(r, dict) and r.get("verdict") == "null"]
    regressions = [k for k, r in report.items() if isinstance(r, dict) and r.get("verdict") == "regression"]
    print(f"\nv5 B2 contrast summary: improvements={improvements}, nulls={nulls}, regressions={regressions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
