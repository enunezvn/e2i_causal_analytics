"""WS1-MP-002..008 + CM-004 model-quality metric stamper (Shard 09).

The synthetic prediction generator emits the causal columns (treatment_effect_estimate,
heterogeneous_effect, segment_assignment) but NOT the model-quality metrics the
model-performance KPIs read: model_auc (MP-001), model_pr_auc (MP-002),
model_precision/recall (MP-003), rank_metrics (MP-004), brier_score (MP-005),
calibration_score (MP-006), fairness_metrics (MP-008), shap_values (MP-007), plus
counterfactual_outcome (CM-004). Those columns exist on ml_predictions (faithful-DB
verified, all nullable) -- this stamps non-degenerate values onto the synthetic frame
so the KPIs return non-NULL from the synthetic substrate, not only the pre-existing
real rows. The loader carries the columns via TABLE_COLUMNS["ml_predictions"] (Task 1).
"""

import numpy as np
import pandas as pd


def stamp_model_metrics(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df with model-quality metric columns populated.

    AUC/PR-AUC land in a deployable band; precision/recall/brier/calibration are
    plausibly correlated with AUC. rank_metrics / fairness_metrics / shap_values are
    JSON objects (the jsonb columns the KPIs read). counterfactual_outcome derives
    from prediction_value when present, else a bounded draw.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    auc = np.round(rng.uniform(0.62, 0.92, n), 4)
    out["model_auc"] = auc
    out["model_pr_auc"] = np.round(np.clip(auc - rng.uniform(0.05, 0.15, n), 0.4, 0.95), 4)
    out["model_precision"] = np.round(np.clip(auc - rng.uniform(0.05, 0.20, n), 0.3, 0.95), 4)
    out["model_recall"] = np.round(np.clip(auc - rng.uniform(0.08, 0.25, n), 0.3, 0.95), 4)
    out["brier_score"] = np.round(rng.uniform(0.08, 0.22, n), 4)
    out["calibration_score"] = np.round(rng.uniform(0.85, 1.10, n), 4)

    out["rank_metrics"] = [
        {
            "recall_at_10": round(float(rng.uniform(0.20, 0.60)), 4),
            "precision_at_10": round(float(rng.uniform(0.15, 0.50)), 4),
        }
        for _ in range(n)
    ]
    out["fairness_metrics"] = [
        {
            "recall_gap": round(float(rng.uniform(0.0, 0.08)), 4),
            "demographic_parity": round(float(rng.uniform(0.0, 0.06)), 4),
        }
        for _ in range(n)
    ]
    out["shap_values"] = [
        {
            "disease_severity": round(float(rng.normal(0, 0.3)), 4),
            "academic_hcp": round(float(rng.normal(0, 0.2)), 4),
            "engagement_score": round(float(rng.normal(0, 0.25)), 4),
        }
        for _ in range(n)
    ]

    if "prediction_value" in out.columns:
        base = pd.to_numeric(out["prediction_value"], errors="coerce").fillna(0.4).to_numpy()
    else:
        base = rng.uniform(0.1, 0.7, n)
    out["counterfactual_outcome"] = np.round(
        np.clip(base - rng.uniform(0.02, 0.15, n), 0.0, 1.0), 4
    )
    return out
