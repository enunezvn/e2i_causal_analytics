"""Model-performance strategic insight: diagnose a model's health + next action."""

from __future__ import annotations

import logging
from typing import Any

from src.insights.common import normalize_list, run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ModelPerformanceInsightSignature(dspy.Signature):
        """Diagnose a deployed classifier's health for an ML/commercial analyst,
        STRICTLY grounded in the provided metrics. Use ONLY the numbers given; never
        invent metrics or thresholds. State whether the model is healthy vs degrading,
        what the confusion/ROC imply (e.g. precision vs recall trade-off), and the
        single most appropriate next action (monitor / retrain / investigate drift).

        Write every output as PLAIN PROSE — no markdown syntax: no asterisks,
        no underscore emphasis, no backticks, no # heading markers, no
        bullet-list markers, no numbered-list markers — write flowing prose."""

        model_version: str = dspy.InputField(desc="Model version/identifier")
        trend_summary: str = dspy.InputField(
            desc=(
                "Current vs baseline value of the trended metric, its trend "
                "classification and the statistical basis for that label"
            )
        )
        confusion_summary: str = dspy.InputField(desc="Precision, recall, specificity, F1 + counts")
        auc_summary: str = dspy.InputField(desc="ROC AUC")
        alerts_summary: str = dspy.InputField(desc="Active performance alerts (or none)")

        interpretation: str = dspy.OutputField(desc="Health diagnosis grounded in the metrics")
        key_takeaways: list = dspy.OutputField(
            desc="3-5 grounded takeaways incl. recommended action"
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ModelPerformanceInsightSignature = None  # type: ignore[assignment,misc]


def _prf(cm: dict[str, Any]) -> dict[str, float]:
    tp, fp, fn, tn = (float(cm.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "specificity": spec, "f1": f1}


_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "auc_roc": "AUC-ROC",
}


def build_grounding(
    model_version: str,
    current_value: float,
    baseline_value: float,
    trend: str,
    confusion: dict[str, Any] | None,
    auc: float | None,
    alerts: list[dict[str, Any]] | None,
    *,
    metric_name: str = "auc_roc",
    trend_reason: str = "",
) -> dict[str, Any]:
    """Ground the insight in the SAME trended metric the page card shows.

    ``metric_name`` is whichever metric the caller trended (the page passes
    its selector; default auc_roc = the page default). ``trend_reason`` is the
    tracker's one-sentence basis for the label (e.g. "within sampling noise at
    n=134") so the LLM does not narrate a −13% single-fold wobble as decay.
    """
    label = _METRIC_LABELS.get(metric_name, metric_name)
    delta = float(current_value) - float(baseline_value)
    trend_summary = (
        f"{label} {current_value:.3f} vs baseline {baseline_value:.3f} "
        f"(Δ{delta:+.3f}), trend {trend}"
    )
    if trend_reason:
        trend_summary += f" — {trend_reason}"
    chips: list[dict[str, str]] = [
        {"label": f"Current {label}", "value": f"{current_value:.3f}"},
        {"label": f"Baseline {label}", "value": f"{baseline_value:.3f}"},
        {"label": "Trend", "value": str(trend)},
    ]
    if confusion:
        m = _prf(confusion)
        confusion_summary = (
            f"precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
            f"specificity {m['specificity']:.2f}, F1 {m['f1']:.2f} "
            f"(TP={confusion.get('tp')}, FP={confusion.get('fp')}, "
            f"FN={confusion.get('fn')}, TN={confusion.get('tn')})"
        )
        chips.append({"label": "F1", "value": f"{m['f1']:.2f}"})
    else:
        confusion_summary = "no confusion matrix available"
    auc_summary = f"holdout ROC AUC {auc:.3f}" if auc is not None else "no ROC curve available"
    if auc is not None:
        chips.append({"label": "Holdout AUC", "value": f"{auc:.3f}"})
    alerts = alerts or []
    alerts_summary = (
        "; ".join(f"{a.get('metric_name')} ({a.get('severity')})" for a in alerts)
        if alerts
        else "no active alerts"
    )
    return {
        "model_version": model_version,
        "metric_name": metric_name,
        "trend_summary": trend_summary,
        "confusion_summary": confusion_summary,
        "auc_summary": auc_summary,
        "alerts_summary": alerts_summary,
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Model {g['model_version']}: {g['trend_summary']}. "
        f"{g['confusion_summary']}. {g['auc_summary']}. Alerts: {g['alerts_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["trend_summary"], g["confusion_summary"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        ModelPerformanceInsightSignature,
        model_version=g["model_version"],
        trend_summary=g["trend_summary"],
        confusion_summary=g["confusion_summary"],
        auc_summary=g["auc_summary"],
        alerts_summary=g["alerts_summary"],
    )
    if pred is None:
        return _fallback(g)
    interpretation = str(getattr(pred, "interpretation", "")).strip()
    if not interpretation:
        return _fallback(g)
    return {
        "insight": interpretation,
        "key_takeaways": normalize_list(getattr(pred, "key_takeaways", [])),
        "grounding": g["grounding"],
        "is_fallback": False,
    }
