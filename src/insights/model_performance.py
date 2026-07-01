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
        single most appropriate next action (monitor / retrain / investigate drift)."""

        model_version: str = dspy.InputField(desc="Model version/identifier")
        accuracy_summary: str = dspy.InputField(desc="Current vs baseline accuracy + trend")
        confusion_summary: str = dspy.InputField(
            desc="Precision, recall, specificity, F1 + counts"
        )
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


def build_grounding(
    model_version: str,
    current_accuracy: float,
    baseline_accuracy: float,
    trend: str,
    confusion: dict[str, Any] | None,
    auc: float | None,
    alerts: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    delta = float(current_accuracy) - float(baseline_accuracy)
    accuracy_summary = (
        f"accuracy {current_accuracy:.3f} vs baseline {baseline_accuracy:.3f} "
        f"(Δ{delta:+.3f}), trend {trend}"
    )
    chips: list[dict[str, str]] = [
        {"label": "Accuracy", "value": f"{current_accuracy:.3f}"},
        {"label": "Baseline", "value": f"{baseline_accuracy:.3f}"},
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
    auc_summary = f"ROC AUC {auc:.3f}" if auc is not None else "no ROC curve available"
    if auc is not None:
        chips.append({"label": "AUC", "value": f"{auc:.3f}"})
    alerts = alerts or []
    alerts_summary = (
        "; ".join(f"{a.get('metric_name')} ({a.get('severity')})" for a in alerts)
        if alerts else "no active alerts"
    )
    return {
        "model_version": model_version,
        "accuracy_summary": accuracy_summary,
        "confusion_summary": confusion_summary,
        "auc_summary": auc_summary,
        "alerts_summary": alerts_summary,
        "grounding": chips,
    }


def _fallback(g: dict[str, Any]) -> dict[str, Any]:
    insight = (
        f"Model {g['model_version']}: {g['accuracy_summary']}. "
        f"{g['confusion_summary']}. {g['auc_summary']}. Alerts: {g['alerts_summary']}. "
        "(Factual summary — LLM interpretation unavailable.)"
    )
    return {
        "insight": insight,
        "key_takeaways": [g["accuracy_summary"], g["confusion_summary"]],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    pred = run_signature(
        ModelPerformanceInsightSignature,
        model_version=g["model_version"],
        accuracy_summary=g["accuracy_summary"],
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
