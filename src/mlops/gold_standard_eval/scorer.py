"""Gold Standard Evaluation — Scorer.

Computes page-aligned classification metrics. Metric names MUST match the
strings the frontend Time-Series page queries:
  accuracy / precision / recall / f1 / auc_roc

Pure function: no I/O, no mocking, no side effects.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "auc_roc")


def score(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return a dict of page-aligned metrics for a binary classifier.

    Args:
        y_true:    Ground-truth binary labels (0/1).
        y_score:   Predicted probability scores for the positive class.
        threshold: Decision boundary applied to y_score (default 0.5).

    Returns:
        Dict with keys matching METRIC_NAMES; all values in [0.0, 1.0].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
    }
