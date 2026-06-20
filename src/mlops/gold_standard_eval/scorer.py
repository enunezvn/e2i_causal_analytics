"""Gold Standard Evaluation — Scorer.

Computes page-aligned classification metrics. Metric names MUST match the
strings the frontend Time-Series page queries:
  accuracy / precision / recall / f1 / auc_roc

Pure functions: no I/O, no mocking, no side effects.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


def confusion(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Exact 2x2 confusion-matrix counts at a decision threshold.

    Returns the EXACT counts (not derived from rounded scalar metrics) so the
    monitoring page can render a faithful confusion matrix. ``labels=[0, 1]``
    pins the orientation even when the holdout predicts a single class.

    Args:
        y_true:    Ground-truth binary labels (0/1).
        y_score:   Predicted probability scores for the positive class.
        threshold: Decision boundary applied to y_score (default 0.5).

    Returns:
        ``{"tn", "fp", "fn", "tp", "threshold"}`` — integer counts + the
        threshold used.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def roc_points(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    max_points: int = 100,
) -> dict:
    """ROC curve as a bounded, JSON-safe list of (fpr, tpr, threshold) points.

    The full ROC curve can have up to ``n_samples + 1`` points (thousands on a
    holdout) — far too many for a chart payload. We downsample to at most
    ``max_points`` evenly-spaced points, ALWAYS keeping the (0,0) and (1,1)
    corners. sklearn sets ``thresholds[0] = inf`` (not JSON-serialisable), so
    thresholds are clamped to ``[0, 1]``.

    A single-class ``y_true`` has no defined ROC, so we return empty points +
    ``auc=0.0`` rather than raising (keeps the eval resilient).

    Args:
        y_true:     Ground-truth binary labels (0/1).
        y_score:    Predicted probability scores for the positive class.
        max_points: Maximum number of curve points to emit (default 100).

    Returns:
        ``{"points": [{"fpr", "tpr", "threshold"}, ...], "auc": float}``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return {"points": [], "auc": 0.0}

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    n = len(fpr)
    if n > max_points:
        # Evenly-spaced indices over the curve; np.unique keeps first + last.
        idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
        fpr, tpr, thresholds = fpr[idx], tpr[idx], thresholds[idx]

    points = [
        {
            "fpr": float(f),
            "tpr": float(t),
            # Clamp to [0, 1]: handles sklearn's inf first-threshold + any noise.
            "threshold": float(max(0.0, min(1.0, th))),
        }
        for f, t, th in zip(fpr, tpr, thresholds, strict=False)
    ]
    return {"points": points, "auc": auc}


def holdout_curve_records(
    y_true: "np.ndarray",
    y_score: "np.ndarray",
    threshold: float = 0.5,
) -> list:
    """Package confusion + ROC artifacts as ``(kind, scalar, payload)`` tuples.

    Shape matches ``MetricRecorder.record_curves`` / ``record_curve``: each tuple
    is ``(metric_name, representative_scalar, payload_dict)``. The confusion
    matrix is always emitted (it is defined even for a single predicted class);
    the ROC curve is emitted only when it is defined (skipped for single-class
    ``y_true``, where ``roc_points`` returns no points) so the eval records what
    it can rather than crashing.

    Args:
        y_true:    Ground-truth binary labels (0/1).
        y_score:   Predicted probability scores for the positive class.
        threshold: Decision boundary for the confusion matrix (default 0.5).

    Returns:
        List of ``(kind, scalar, payload)`` tuples ready for ``record_curves``.
    """
    cm = confusion(y_true, y_score, threshold)
    n = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    accuracy = (cm["tp"] + cm["tn"]) / n if n else 0.0
    records: list = [("confusion_matrix", float(accuracy), cm)]

    roc = roc_points(y_true, y_score)
    if roc["points"]:
        records.append(("roc_curve", float(roc["auc"]), roc))
    return records
