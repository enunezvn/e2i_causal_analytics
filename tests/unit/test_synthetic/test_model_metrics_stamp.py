"""Shard 09 WS1-MP-002..008 + CM-004: the synthetic ml_predictions frame must carry
model-quality metric columns (model_auc/pr_auc/precision/recall/brier/calibration/
fairness/rank/shap) + counterfactual_outcome so the model-performance KPIs return
non-NULL from the synthetic substrate (not only the pre-existing real rows)."""

import pandas as pd

from src.ml.synthetic.generators.model_metrics import stamp_model_metrics

_METRIC_COLS = [
    "model_auc",
    "model_pr_auc",
    "model_precision",
    "model_recall",
    "brier_score",
    "calibration_score",
    "rank_metrics",
    "fairness_metrics",
    "shap_values",
    "counterfactual_outcome",
]


def test_model_metrics_populated_and_bounded():
    df = pd.DataFrame(
        {
            "prediction_id": [f"pr{i}" for i in range(40)],
            "prediction_value": [0.4] * 40,
            "is_synthetic": [True] * 40,
        }
    )
    out = stamp_model_metrics(df, seed=3)
    for c in _METRIC_COLS:
        assert c in out.columns, f"{c} not stamped"
        assert out[c].notna().all(), f"{c} has NULLs -> KPI reads N/A"
    # AUC / PR-AUC in the non-degenerate band
    assert out["model_auc"].between(0.6, 0.95).all()
    assert out["model_pr_auc"].between(0.4, 0.95).all()
    assert out["brier_score"].between(0.0, 0.5).all()
    # jsonb metric columns are dict-shaped
    assert out["rank_metrics"].apply(lambda v: isinstance(v, dict)).all()
    assert out["fairness_metrics"].apply(lambda v: isinstance(v, dict)).all()
    assert out["shap_values"].apply(lambda v: isinstance(v, dict)).all()


def test_model_metrics_does_not_mutate_input():
    df = pd.DataFrame({"prediction_id": ["p0"], "prediction_value": [0.3]})
    _ = stamp_model_metrics(df, seed=1)
    assert "model_auc" not in df.columns
