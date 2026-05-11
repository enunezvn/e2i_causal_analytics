"""v5 Gate B3 — val_AUC contrast measurement (baseline vs. B3-engineered).

Acceptance per docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md
section 5.2:
- Improvement: val_AUC(B) - val_AUC(A) >= 0.02 on at least one cohort.
- Null: |delta| < 0.02 OR every candidate failed Layer 3.

Both outcomes close B3. Null is documented with the contrast measurement
included.

Methodology:
- Arm A (baseline): feature matrix filtered to manifest pre-anchor
  features (the same set the production pipeline keeps after Layer 1).
- Arm B (engineered): Arm A plus the surviving B3 candidates from
  feature_engineering.engineer_features helper.
- Same stratified 5-fold cross-validation seed (cohort-fixed) on both
  arms; mean fold AUC reported. CV mean is robust to lucky-split noise
  (cf. optum_revalidation_20260510 memo where a single 80/20 split gave
  AUC=0.79 vs. CV mean 0.68).
- Model: sklearn LogisticRegression(class_weight="balanced", max_iter=
  2000). Linear + imbalance-aware; matches CSU honest band [0.62, 0.68].

The model is deliberately simple to keep the comparison interpretable.
A more capable model could absorb the same signal differently; the
v5 B3 contrast is about whether engineered features ADD anything the
base surface lacks.

Pre-spec dating: this script is run AFTER the pre-spec memo at
docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md is
committed. The acceptance threshold (>=0.02) is locked there. No
threshold-shopping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd


def _filter_to_manifest_safe(
    X: pd.DataFrame, manifest_source: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Filter X to columns declared pre-anchor in the cohort manifest.

    Production parity: matches the ``_select_features`` behavior in
    ``adaptive_validity_check`` where unknown / post-anchor features
    are excluded BEFORE model training. Without this filter, the
    val_AUC contrast measures the model's behavior on a known-leaky
    surface (e.g., journey_duration_days on CSU) rather than the
    clean post-Layer-1 surface the pipeline actually trains on.
    """
    from src.data.manifests import MANIFEST_SOURCES

    lookup = MANIFEST_SOURCES.get(manifest_source)
    if lookup is None:
        # Unknown manifest: fall back to all-numeric columns (no filter).
        return X, list(X.columns)

    safe_cols: List[str] = []
    dropped: List[str] = []
    for col in X.columns:
        contract = lookup(col)
        if contract is None:
            # Column not in manifest at all — drop conservatively.
            dropped.append(col)
            continue
        if contract.knowable_at.is_pre_or_at_index():
            safe_cols.append(col)
        else:
            dropped.append(col)

    return X[safe_cols].copy(), dropped


def _cross_val_auc(X: pd.DataFrame, y: pd.Series, seed: int, n_splits: int = 5) -> Dict[str, float]:
    """Return CV AUC mean + std over n_splits folds."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: List[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X.values, y.values)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # M3 (codex): use sklearn SimpleImputer rather than pandas
        # .fillna(median()). pandas .median() drops all-NaN columns
        # from the result, which means X_val_fold.fillna() leaves
        # NaN in those columns — StandardScaler would then crash.
        # SimpleImputer(strategy="median") fits on train and applies
        # the same column-wise medians (or 0 for all-NaN columns)
        # to val, with a deterministic fail-loud path if values
        # cannot be imputed.
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        )
        pipe.fit(X_train_fold.values, y_train_fold.values)
        scores = pipe.predict_proba(X_val_fold.values)[:, 1]
        if y_val_fold.nunique() < 2:
            # Degenerate fold — skip.
            continue
        aucs.append(float(roc_auc_score(y_val_fold.values, scores)))

    if not aucs:
        return {"mean": float("nan"), "std": float("nan"), "n_folds": 0}
    return {
        "mean": float(np.mean(aucs)),
        "std": float(np.std(aucs)),
        "n_folds": len(aucs),
        "per_fold": [float(a) for a in aucs],
    }


def _measure_cohort(
    cohort_label: str,
    data_dir: Path,
    target_col: str,
    manifest_source: str,
    seed: int,
) -> Dict[str, dict]:
    """Run baseline + engineered val_AUC contrast for one cohort."""
    from scripts.run_tier1b_b2_experiment import (
        _build_features_and_target,
        _load_patient_journeys,
    )
    from src.agents.ml_foundation.data_preparer.nodes.feature_engineering import (
        engineer_features,
    )

    df = _load_patient_journeys(data_dir)
    X_raw, y = _build_features_and_target(df, target_col=target_col)

    # Filter to manifest pre-anchor surface (production parity).
    X_baseline, dropped_pre_filter = _filter_to_manifest_safe(X_raw, manifest_source)
    print(
        f"[{cohort_label}] baseline surface: {X_baseline.shape[1]} pre-anchor "
        f"features (dropped {len(dropped_pre_filter)} non-manifest/post-anchor cols)"
    )

    # Compute baseline AUC.
    baseline_metrics = _cross_val_auc(X_baseline, y, seed=seed)
    print(
        f"[{cohort_label}] baseline CV AUC: {baseline_metrics['mean']:.4f} "
        f"(+/- {baseline_metrics['std']:.4f}, n_folds={baseline_metrics['n_folds']})"
    )

    # H2 (codex): build engineered features from the RAW df (which
    # still has categorical columns like insurance_type) rather than
    # from X_raw (which has been numeric-filtered by
    # _build_features_and_target). This is required for
    # age_x_insurance_interaction whose insurance_type input is
    # categorical/object dtype. Previously the engineered feature was
    # silently dropped at audit time, masquerading as a
    # missing-input skip in the INFO log; the resulting val_AUC
    # contrast was on 3 of 4 CSU candidates rather than 4.
    df_for_engineering = df.copy()
    df_with_engineered, materialized = engineer_features(df_for_engineering, manifest_source)
    eng_cols = [c for c in materialized if c in df_with_engineered.columns]
    # Concat materialized engineered columns (only) onto the
    # production-parity baseline surface.
    X_engineered = pd.concat(
        [
            X_baseline.reset_index(drop=True),
            df_with_engineered[eng_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    print(
        f"[{cohort_label}] engineered surface: +{len(eng_cols)} features = "
        f"{X_engineered.shape[1]} total ({eng_cols})"
    )

    engineered_metrics = _cross_val_auc(X_engineered, y, seed=seed)
    print(
        f"[{cohort_label}] engineered CV AUC: {engineered_metrics['mean']:.4f} "
        f"(+/- {engineered_metrics['std']:.4f}, n_folds={engineered_metrics['n_folds']})"
    )

    delta = engineered_metrics["mean"] - baseline_metrics["mean"]
    verdict = "improvement" if delta >= 0.02 else ("null" if abs(delta) < 0.02 else "regression")
    print(f"[{cohort_label}] delta = {delta:+.4f} ({verdict}; acceptance threshold >= 0.02)")

    return {
        "cohort": cohort_label,
        "n_rows": int(len(X_raw)),
        "n_pos": int(y.sum()),
        "baseline": baseline_metrics,
        "engineered": engineered_metrics,
        "engineered_features_added": eng_cols,
        "delta_mean_auc": delta,
        "verdict": verdict,
        "acceptance_threshold": 0.02,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="docs/calibration/b3_val_auc_contrast_20260511.json",
    )
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cohorts: Mapping[str, Tuple[Path, str, str]] = {
        "csu": (repo_root / "data" / "rwd" / "csu", "treatment_initiated", "csu"),
        "optum_initiation": (
            repo_root / "data" / "rwd" / "optum" / "initiation",
            "treatment_initiated",
            "optum",
        ),
    }

    report: Dict[str, dict] = {}
    for label, (data_dir, target, manifest_source) in cohorts.items():
        if not data_dir.exists():
            if args.skip_missing:
                print(f"[{label}] SKIP — data dir absent")
                report[label] = {"skipped": True}
                continue
            print(f"[{label}] FAIL — data dir absent: {data_dir}", file=sys.stderr)
            return 2
        try:
            report[label] = _measure_cohort(
                cohort_label=label,
                data_dir=data_dir,
                target_col=target,
                manifest_source=manifest_source,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] ERROR — {exc}", file=sys.stderr)
            report[label] = {"error": str(exc)}
            return 1

    output_path = repo_root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to {output_path}")

    # Summarize the contrast.
    improvements = [
        label
        for label, r in report.items()
        if isinstance(r, dict) and r.get("verdict") == "improvement"
    ]
    nulls = [
        label for label, r in report.items() if isinstance(r, dict) and r.get("verdict") == "null"
    ]
    regressions = [
        label
        for label, r in report.items()
        if isinstance(r, dict) and r.get("verdict") == "regression"
    ]
    print(
        f"\nv5 B3 contrast summary: improvements={improvements}, "
        f"nulls={nulls}, regressions={regressions}"
    )
    if regressions:
        # Regression is a meaningful negative finding — still close B3
        # but call it out.
        print(
            "REGRESSION detected on at least one cohort. Per pre-spec, "
            "either outcome closes B3; documenting and proceeding."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
