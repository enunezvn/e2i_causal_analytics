import numpy as np

from src.mlops.gold_standard_eval.scorer import METRIC_NAMES, score


def test_score_emits_page_aligned_names_and_real_values():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    out = score(y, s)
    assert set(out) == set(METRIC_NAMES) == {"accuracy", "precision", "recall", "f1", "auc_roc"}
    assert abs(out["auc_roc"] - 1.0) < 1e-9  # perfectly separable
    assert all(0.0 <= v <= 1.0 for v in out.values())
