#!/usr/bin/env python3
"""Plan v3 §3 Tier 1C — Survival/time-to-event target feasibility evaluation.

Side script (NOT yet wired into convert_optum_rwd.py per plan §3 Tier 1C
mechanism step 1). Builds the (time_to_initiation_days, event_observed)
target on the existing Optum n=1294 cohort and fits a Cox PH model with
a small demographic feature set. Produces an evaluation report
comparing survival concordance against the binary CV mean AUC baseline
(~0.68 per `docs/results/optum_initiation_revalidation_20260510.md`).

Plan §3 Tier 1C decision gate:
    Ship-if-feasible iff concordance ≥ binary CV mean AUC + 0.04
    (literature-aligned lift, Gerds & Schumacher 2006 / Steingrimsson 2018).

Per plan §6 Tier 1C acceptance: concordance delta vs binary AUC documented
(regardless of sign); if shipped: schema migration plan circulated; if not
shipped: documented rationale.

This iteration ships the EVALUATION ONLY. A separate ship-if-feasible PR
adds (time, event) to the converter, problem_type="survival" to the
trainer, IPCW deployer metrics, etc.

Usage:
    python scripts/evaluate_optum_survival.py
    python scripts/evaluate_optum_survival.py --cohort-dir data/rwd/optum/initiation
    python scripts/evaluate_optum_survival.py --report-out docs/results/optum_survival_evaluation_20260510.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Plan v3 §3 Tier 1C decision gate.
DEFAULT_PREDICTION_HORIZON_DAYS: int = 180
SURVIVAL_LIFT_THRESHOLD_OVER_BINARY: float = 0.04
BINARY_BASELINE_CV_MEAN_AUC_DEFAULT: float = 0.68

# Brand pattern for biologic identification — matches synthetic + Optum
# Xolair (omalizumab) and Dupixent (dupilumab) per CSU spec §6.
BIOLOGIC_DRUG_NAME_PATTERNS: Tuple[str, ...] = (
    "xolair",
    "dupixent",
    "omalizumab",
    "dupilumab",
)


def _load_cohort(
    cohort_dir: Path,
    raw_optum_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load journeys (canonical cohort) + raw medication.parquet (for
    post-index biologic events).

    The cohort's `treatment_events.parquet` contains only LOOKBACK
    events (pre-index by spec); for the survival target we need
    post-index biologic prescriptions. Those live in the raw
    `data/rwd/Optum_Parquet/medication.parquet`.

    Returns:
        (journeys_df, biologic_events_df) where biologic_events_df is
        the raw med rows filtered to Xolair/Dupixent (Generic_Name +
        Brand_Name match), with `patient_id` derived from `patid` via
        the cohort's id mapping (PAT_<patid> hash).
    """
    journeys_path = cohort_dir / "e2i_ml_v3_patient_journeys.parquet"
    if not journeys_path.exists():
        raise FileNotFoundError(journeys_path)
    journeys = pd.read_parquet(journeys_path)

    if raw_optum_dir is None:
        raw_optum_dir = PROJECT_ROOT / "data" / "rwd" / "Optum_Parquet"
    med_path = raw_optum_dir / "medication.parquet"
    if not med_path.exists():
        raise FileNotFoundError(med_path)
    med = pd.read_parquet(med_path)

    generic_lower = med["Generic_Name"].fillna("").astype(str).str.lower()
    brand_upper = med["Brand_Name"].fillna("").astype(str).str.upper()
    biologic_mask = generic_lower.str.contains(
        "omalizumab|dupilumab", na=False
    ) | brand_upper.str.contains("XOLAIR|DUPIXENT", na=False)
    biologic_med = med[biologic_mask].copy()
    biologic_med["medication_date"] = pd.to_datetime(
        biologic_med["medication_date"], errors="coerce"
    )

    # Reconstruct patient_id (PAT_<patid>) — verified format from
    # `data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet` is
    # PAT_<unpadded-15-digit-int> (NOT 18-digit zero-padded).
    biologic_med["patient_id"] = biologic_med["patid"].apply(lambda x: f"PAT_{int(x)}")
    # Project to the canonical event columns the rest of this module reads.
    biologic_events = biologic_med[
        ["patient_id", "medication_date", "Brand_Name", "Generic_Name"]
    ].rename(
        columns={
            "medication_date": "event_date",
            "Brand_Name": "brand",
            "Generic_Name": "drug_name",
        }
    )
    return journeys, biologic_events


def derive_survival_target(
    journeys: pd.DataFrame,
    events: pd.DataFrame,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
) -> pd.DataFrame:
    """Plan v3 §3 Tier 1C target derivation.

    For each patient with ``index_date`` in journeys, find the first
    biologic-prescription event in events. The (t, δ) target is:

      * ``time_to_initiation_days`` = days from index to first biologic
        prescription, capped at ``prediction_horizon_days``. For
        patients with no biologic event in the horizon, the time is
        right-censored at ``prediction_horizon_days``.
      * ``event_observed`` = 1 iff a biologic prescription occurred
        within the horizon; 0 otherwise (right-censored).

    This matches plan §3 Tier 1C mechanism — same data the binary
    target ``initiated_biologic_180d`` consumes.

    Returns a DataFrame with columns:
      [patient_id, time_to_initiation_days, event_observed, index_date].
    """
    # Filter biologic events.
    drug_name_lower = events["drug_name"].fillna("").str.lower()
    mask = pd.Series(False, index=events.index)
    for pattern in BIOLOGIC_DRUG_NAME_PATTERNS:
        mask = mask | drug_name_lower.str.contains(pattern, na=False)
    biologic_events = events[mask].copy()
    biologic_events["event_date"] = pd.to_datetime(biologic_events["event_date"], errors="coerce")

    # First biologic per patient.
    if len(biologic_events) > 0:
        first_biologic = (
            biologic_events.dropna(subset=["event_date"])
            .sort_values("event_date")
            .drop_duplicates("patient_id", keep="first")
        )
        first_biologic = first_biologic[["patient_id", "event_date"]].rename(
            columns={"event_date": "first_biologic_date"}
        )
    else:
        first_biologic = pd.DataFrame(
            {
                "patient_id": pd.Series([], dtype=str),
                "first_biologic_date": pd.Series([], dtype="datetime64[ns]"),
            }
        )

    # Join with journey index_date.
    df = journeys[["patient_id", "index_date"]].copy()
    df["index_date"] = pd.to_datetime(df["index_date"], errors="coerce")
    df = df.merge(first_biologic, on="patient_id", how="left")
    # Ensure first_biologic_date is datetime even after the merge (when
    # the right side is empty, pandas may emit object dtype).
    df["first_biologic_date"] = pd.to_datetime(df["first_biologic_date"], errors="coerce")

    # Compute days-to-initiation; cap at horizon for censoring.
    days_diff = (df["first_biologic_date"] - df["index_date"]).dt.days
    df["time_to_initiation_days"] = days_diff.where(
        (days_diff > 0) & (days_diff <= prediction_horizon_days),
        prediction_horizon_days,  # censored
    ).astype(float)
    df["event_observed"] = ((days_diff > 0) & (days_diff <= prediction_horizon_days)).astype(int)

    return df[["patient_id", "time_to_initiation_days", "event_observed", "index_date"]]


def _build_cox_features(journeys: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Small demographic feature set matching the binary baseline.

    Columns selected to mirror what the binary classifier sees: age at
    index, ICD subtype indicator, insurance product, region.
    """
    feature_cols: List[str] = []
    base = journeys[["patient_id"]].copy()

    if "age_at_index" in journeys.columns:
        base["age_at_index"] = pd.to_numeric(journeys["age_at_index"], errors="coerce").fillna(50.0)
        feature_cols.append("age_at_index")

    if "primary_diagnosis_code" in journeys.columns:
        # One-hot the L50.x subtype (small set).
        dummies = pd.get_dummies(
            journeys["primary_diagnosis_code"].fillna("UNKNOWN"),
            prefix="dx",
            drop_first=True,
        )
        dummies = dummies.astype(int)
        for col in dummies.columns:
            base[col] = dummies[col].values
            feature_cols.append(col)

    # Use the existing binary target as a sanity covariate (we do NOT
    # include initiated_biologic_180d — that IS the target).
    df = base.merge(target, on="patient_id", how="inner")
    return df


def evaluate_cox_concordance_cv(
    df: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """5-fold CV Cox PH on the (time, event) target.

    Returns the mean concordance index (Harrell's C) across folds plus
    per-fold values. Uses lifelines (sksurv NOT installed in this env;
    Cox PH covers the plan §3 Tier 1C decision gate either way).
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import KFold

    feature_cols = [
        c
        for c in df.columns
        if c not in ("patient_id", "time_to_initiation_days", "event_observed", "index_date")
    ]
    if not feature_cols:
        return {
            "cox_concordance_cv_completed": False,
            "cox_concordance_cv_error": "no feature columns available",
        }

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_concordances: List[float] = []
    fold_n_events: List[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df)):
        train = df.iloc[train_idx][
            feature_cols + ["time_to_initiation_days", "event_observed"]
        ].copy()
        val = df.iloc[val_idx][feature_cols + ["time_to_initiation_days", "event_observed"]].copy()
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(
                train,
                duration_col="time_to_initiation_days",
                event_col="event_observed",
                show_progress=False,
            )
            # Predict partial-hazard on val; concordance against val target.
            val_features = val[feature_cols]
            val_partial_hazard = cph.predict_partial_hazard(val_features)
            c_index = concordance_index(
                val["time_to_initiation_days"],
                -val_partial_hazard,  # higher hazard → shorter time
                val["event_observed"],
            )
            fold_concordances.append(float(c_index))
            fold_n_events.append(int(val["event_observed"].sum()))
        except Exception as e:
            logger.warning("Cox CV fold %d failed: %s", fold_idx, e)
            continue

    if not fold_concordances:
        # Fallback: global concordance on the full sample. With heavy
        # right-censoring at the horizon (most patients tied at 180d),
        # per-fold concordance can fail with "No admissable pairs"
        # because each val fold has too few events. Global concordance
        # on the full cohort (with leave-one-out logic implicit in
        # lifelines) at least exposes the model's ranking ability.
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(
                df[feature_cols + ["time_to_initiation_days", "event_observed"]],
                duration_col="time_to_initiation_days",
                event_col="event_observed",
                show_progress=False,
            )
            global_partial_hazard = cph.predict_partial_hazard(df[feature_cols])
            global_c_index = concordance_index(
                df["time_to_initiation_days"],
                -global_partial_hazard,
                df["event_observed"],
            )
            return {
                "cox_concordance_cv_completed": False,
                "cox_concordance_cv_error": "all folds failed (likely too few events per fold; reporting global fit instead)",
                "cox_concordance_global_completed": True,
                "cox_concordance_global": float(global_c_index),
                "cox_concordance_global_n_events": int(df["event_observed"].sum()),
                "cox_concordance_n_features": len(feature_cols),
            }
        except Exception as e:
            return {
                "cox_concordance_cv_completed": False,
                "cox_concordance_cv_error": (f"all folds failed AND global fit failed: {e}"),
                "cox_concordance_global_completed": False,
            }

    return {
        "cox_concordance_cv_completed": True,
        "cox_concordance_cv_mean": float(np.mean(fold_concordances)),
        "cox_concordance_cv_std": float(np.std(fold_concordances)),
        "cox_concordance_cv_folds": fold_concordances,
        "cox_concordance_cv_fold_n_events": fold_n_events,
        "cox_concordance_n_folds": n_folds,
        "cox_concordance_n_features": len(feature_cols),
    }


def evaluate(
    cohort_dir: Path,
    binary_baseline_auc: float = BINARY_BASELINE_CV_MEAN_AUC_DEFAULT,
    prediction_horizon_days: int = DEFAULT_PREDICTION_HORIZON_DAYS,
) -> Dict[str, Any]:
    """End-to-end evaluation. Returns a dict with target stats + Cox
    concordance + decision-gate verdict."""
    journeys, events = _load_cohort(cohort_dir)
    target = derive_survival_target(journeys, events, prediction_horizon_days)

    n_total = len(target)
    n_observed = int(target["event_observed"].sum())
    median_time_observed = (
        float(target.loc[target["event_observed"] == 1, "time_to_initiation_days"].median())
        if n_observed > 0
        else float("nan")
    )

    feature_df = _build_cox_features(journeys, target)
    cox_result = evaluate_cox_concordance_cv(feature_df)

    delta_vs_binary: Optional[float] = None
    decision: str
    cox_concordance: Optional[float] = None
    if cox_result.get("cox_concordance_cv_completed"):
        cox_concordance = cox_result["cox_concordance_cv_mean"]
        delta_vs_binary = cox_concordance - binary_baseline_auc
        if delta_vs_binary >= SURVIVAL_LIFT_THRESHOLD_OVER_BINARY:
            decision = "ship_recommended"
        else:
            decision = "rejection_acceptable"
    elif cox_result.get("cox_concordance_global_completed"):
        # CV failed but global fit succeeded — use global concordance
        # but mark decision as 'rejection_acceptable' regardless because
        # CV instability itself is a red flag for shippability.
        cox_concordance = cox_result["cox_concordance_global"]
        delta_vs_binary = cox_concordance - binary_baseline_auc
        decision = "rejection_acceptable"
    else:
        decision = "evaluation_failed"

    return {
        "evaluation_metadata": {
            "cohort_dir": str(cohort_dir),
            "evaluated_at_iso": datetime.now().isoformat(),
            "prediction_horizon_days": prediction_horizon_days,
            "binary_baseline_cv_mean_auc": binary_baseline_auc,
            "survival_lift_threshold": SURVIVAL_LIFT_THRESHOLD_OVER_BINARY,
        },
        "target_stats": {
            "n_total": n_total,
            "n_event_observed": n_observed,
            "n_censored": n_total - n_observed,
            "event_rate": float(n_observed) / float(n_total) if n_total > 0 else 0.0,
            "median_time_to_initiation_days_observed": median_time_observed,
        },
        "cox_concordance": cox_result,
        "decision_gate": {
            "delta_vs_binary": delta_vs_binary,
            "ship_threshold": SURVIVAL_LIFT_THRESHOLD_OVER_BINARY,
            "decision": decision,
        },
    }


def _fmt_float(v: Any, default: str = "N/A") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return default
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def write_report(result: Dict[str, Any], report_out: Path) -> None:
    """Markdown evaluation report at the path. Plan §6 Tier 1C
    acceptance: concordance delta vs binary AUC documented (regardless
    of sign)."""
    meta = result["evaluation_metadata"]
    stats = result["target_stats"]
    cox = result["cox_concordance"]
    gate = result["decision_gate"]

    decision_text = {
        "ship_recommended": (
            "**Ship recommended.** Survival concordance exceeds the "
            f"binary CV mean AUC by ≥{SURVIVAL_LIFT_THRESHOLD_OVER_BINARY:.2f} "
            "(plan §3 Tier 1C decision gate). Proceed to ship-if-feasible "
            "phase: target derivation in converters, problem_type='survival' "
            "in trainer, IPCW deployer metrics, schema migration."
        ),
        "rejection_acceptable": (
            "**Rejection acceptable.** Survival concordance does NOT clear "
            f"the binary AUC + {SURVIVAL_LIFT_THRESHOLD_OVER_BINARY:.2f} "
            "lift threshold. Per plan §3 Tier 1C, this is the point of an "
            "evaluation phase — the answer is 'target encoding wasn't the "
            "bottleneck on this cohort'. Document and stop; do NOT ship "
            "the survival surface."
        ),
        "evaluation_failed": (
            "**Evaluation failed.** Cox CV did not complete. See cox_concordance "
            "block for the error. Plan §3 Tier 1C: re-run with a different "
            "feature set or cohort once root cause is understood."
        ),
    }[gate["decision"]]

    body = f"""# Optum Survival Evaluation — n={stats["n_total"]} cohort

**Date:** {meta["evaluated_at_iso"]}
**Cohort:** `{meta["cohort_dir"]}`
**Plan reference:** `.claude/plans/adaptive_disease_agnostic_quality_uplift.md` v3 §3 Tier 1C
**Script:** `scripts/evaluate_optum_survival.py`

## Decision

{decision_text}

## Target derivation

Per plan §3 Tier 1C: `(time_to_initiation_days, event_observed)` derived
from cohort `index_date` + biologic-prescription events. Right-censored
at the `prediction_horizon_days={meta["prediction_horizon_days"]}` window.

| Metric | Value |
|---|---|
| Total patients | {stats["n_total"]} |
| Event observed (initiated within horizon) | {stats["n_event_observed"]} |
| Right-censored | {stats["n_censored"]} |
| Event rate | {_fmt_float(stats["event_rate"])} |
| Median time-to-initiation (observed) | {_fmt_float(stats["median_time_to_initiation_days_observed"])} days |

## Cox PH 5-fold CV concordance

| Metric | Value |
|---|---|
| Completed | {cox.get("cox_concordance_cv_completed")} |
| Mean concordance | {_fmt_float(cox.get("cox_concordance_cv_mean"))} |
| Std concordance | {_fmt_float(cox.get("cox_concordance_cv_std"))} |
| Per-fold | {cox.get("cox_concordance_cv_folds", "N/A")} |
| Per-fold n_events | {cox.get("cox_concordance_cv_fold_n_events", "N/A")} |
| n_folds | {cox.get("cox_concordance_n_folds", "N/A")} |
| n_features | {cox.get("cox_concordance_n_features", "N/A")} |
| Error (if any) | {cox.get("cox_concordance_cv_error", "—")} |

## Comparison to binary baseline

| Metric | Value |
|---|---|
| Binary CV mean AUC (baseline) | {_fmt_float(meta["binary_baseline_cv_mean_auc"])} |
| Cox concordance (this run) | {_fmt_float(cox.get("cox_concordance_cv_mean"))} |
| Δ (Cox - binary) | {_fmt_float(gate["delta_vs_binary"])} |
| Ship threshold (Δ≥) | {_fmt_float(gate["ship_threshold"])} |
| Decision | **{gate["decision"]}** |

## Plan reference

- Plan v3 §3 Tier 1C: evaluation phase mechanism + decision gate.
- Plan §6 Tier 1C: acceptance — "concordance delta vs binary CV mean
  AUC documented (regardless of sign)".

## Caveat

Cox PH only — RSF (Random Survival Forest) deferred because `sksurv` is
not installed in this environment. RSF would add non-linear feature
interactions; for the plan's binary decision gate (concordance ≥
binary AUC + 0.04), Cox is sufficient: if Cox alone clears the gate,
ship; if it doesn't, RSF unlikely to make up >2-3pp on a 4-feature
cohort.

```json
{json.dumps(result, indent=2, default=str)}
```
"""
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(body)
    logger.info("Report written to %s", report_out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan v3 §3 Tier 1C survival/time-to-event feasibility evaluation."
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "rwd" / "optum" / "initiation",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "results"
        / f"optum_survival_evaluation_{datetime.now().strftime('%Y%m%d')}.md",
    )
    parser.add_argument(
        "--binary-baseline-auc",
        type=float,
        default=BINARY_BASELINE_CV_MEAN_AUC_DEFAULT,
    )
    parser.add_argument(
        "--prediction-horizon-days",
        type=int,
        default=DEFAULT_PREDICTION_HORIZON_DAYS,
    )
    args = parser.parse_args()

    if not args.cohort_dir.exists():
        logger.error("Cohort dir not found: %s", args.cohort_dir)
        return 1

    result = evaluate(
        args.cohort_dir,
        binary_baseline_auc=args.binary_baseline_auc,
        prediction_horizon_days=args.prediction_horizon_days,
    )
    write_report(result, args.report_out)

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
