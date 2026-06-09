"""Per-recipient heuristic GEPA metrics (Gap B §5.1).

Each recipient gets a DETERMINISTIC heuristic metric that scores a freshly
generated output for completeness/grounding over the real signature inputs — no
LM cost, no CI flakiness (#504). Consistent with the learner side's
``compute_reward``.

Contract: a metric is ``(gold, pred, trace=None, pred_name=None,
pred_trace=None) -> dspy.Prediction(score: float in [0,1], feedback: str)`` — the
dspy 3.1 ScoreWithFeedback shape GEPA's valset ``Evaluate`` sums over.

Default = a GENERIC grounding/completeness heuristic that rewards a non-empty,
reasonably-bounded output that references the signature's input VALUES (or, as a
weaker signal, its input field names). This is enough to make a real-data
optimization run meaningful for any recipient out of the box.

Per-recipient OVERRIDE convention (so B1-B4 add a sharper metric WITHOUT editing
this shared file): if ``src/agents/<agent>/dspy_signal_heuristic.py`` defines a
``score(gold, pred, ...)`` callable, :func:`get_recipient_metric` uses it; else
it falls back to the generic heuristic. The override may return a float, a
``{"score", "feedback"}`` dict, or a ``dspy.Prediction`` — all are normalized
here to the Prediction contract.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

# Bounds for the generic length component. Below MIN -> likely truncated/empty;
# far above MAX -> likely rambling. Both ends are penalized softly.
_MIN_REASONABLE_LEN = 40
_MAX_REASONABLE_LEN = 4000


def _coerce_to_prediction(result: Any) -> Any:
    """Normalize a heuristic return into dspy.Prediction(score, feedback)."""
    import dspy

    if isinstance(result, dict):
        return dspy.Prediction(
            score=float(result.get("score", 0.0)),
            feedback=str(result.get("feedback", "")),
        )
    if isinstance(result, (int, float, bool)):
        return dspy.Prediction(score=float(result), feedback="")
    return result  # already a Prediction / ScoreWithFeedback


def _gold_input_values(gold: Any) -> List[str]:
    """Best-effort extraction of the input field VALUES from a gold Example."""
    values: List[str] = []
    try:
        inputs = gold.inputs() if hasattr(gold, "inputs") else gold
        items = (
            inputs.items() if hasattr(inputs, "items") else getattr(inputs, "_store", {}).items()
        )
        for _k, v in items:
            if v is None:
                continue
            values.append(str(v))
    except Exception:  # noqa: BLE001 - introspection is best-effort
        pass
    return values


def _gold_input_keys(gold: Any) -> List[str]:
    """Best-effort extraction of the input field NAMES from a gold Example."""
    try:
        inputs = gold.inputs() if hasattr(gold, "inputs") else gold
        if hasattr(inputs, "keys"):
            return [str(k) for k in inputs.keys()]
        store = getattr(inputs, "_store", {})
        return [str(k) for k in store.keys()]
    except Exception:  # noqa: BLE001
        return []


def _pred_text(pred: Any) -> str:
    """Concatenate all string-ish output fields of a prediction into one blob."""
    parts: List[str] = []
    try:
        store = getattr(pred, "_store", None)
        items = store.items() if store is not None else vars(pred).items()
        for _k, v in items:
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, (list, tuple)):
                parts.extend(str(x) for x in v)
    except Exception:  # noqa: BLE001
        pass
    if not parts:
        # last resort: stringify the whole prediction
        return str(pred)
    return "\n".join(parts)


def _grounding_score(text: str, gold_values: List[str], gold_keys: List[str]) -> float:
    """Fraction of gold input signals the output references (values, then keys)."""
    if not text:
        return 0.0
    lowered = text.lower()

    def _ref_fraction(tokens: List[str]) -> float:
        candidates = [t for t in tokens if t and len(t) >= 2]
        if not candidates:
            return 0.0
        hits = 0
        for tok in candidates:
            # For long values, match a leading slice; for short tokens, exact-ish.
            needle = tok.lower()[:24]
            if needle and needle in lowered:
                hits += 1
        return hits / len(candidates)

    value_frac = _ref_fraction(gold_values)
    if value_frac > 0:
        return value_frac
    # Values may be numeric/format-mangled; fall back to field-name references.
    return _ref_fraction(gold_keys) * 0.5


def _length_score(text: str) -> float:
    n = len(text.strip())
    if n == 0:
        return 0.0
    if n < _MIN_REASONABLE_LEN:
        return max(0.0, n / _MIN_REASONABLE_LEN) * 0.5
    if n > _MAX_REASONABLE_LEN:
        return 0.6
    return 1.0


def generic_grounding_metric(
    gold: Any,
    pred: Any,
    trace: Any = None,
    pred_name: Any = None,
    pred_trace: Any = None,
) -> Any:
    """Deterministic completeness/grounding heuristic (the default metric).

    Score = 0.5 * length_component + 0.5 * grounding_component, both in [0,1]:
      - length_component: non-empty + within a reasonable byte bound.
      - grounding_component: references the gold input values (or field names).
    Returns dspy.Prediction(score in [0,1], feedback=str).
    """
    import dspy

    text = _pred_text(pred)
    length = _length_score(text)
    grounding = _grounding_score(text, _gold_input_values(gold), _gold_input_keys(gold))
    score = round(0.5 * length + 0.5 * grounding, 4)

    if not text.strip():
        feedback = "Output is empty; must produce a grounded, non-empty response."
    elif grounding == 0.0:
        feedback = "Output does not reference the provided inputs; ground it in the given data."
    elif length < 1.0:
        feedback = (
            "Output length is outside the reasonable bound; aim for a fuller, concise answer."
        )
    else:
        feedback = "Grounded and complete; reinforce explicit reference to the input data."

    return dspy.Prediction(score=float(score), feedback=feedback)


def _load_override(agent_name: str) -> Callable | None:
    """Return ``score`` from src/agents/<agent>/dspy_signal_heuristic.py, if present."""
    try:
        mod = importlib.import_module(f"src.agents.{agent_name}.dspy_signal_heuristic")
    except ModuleNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001 - a broken override must not break optimize
        logger.warning("Recipient heuristic override for %s failed to import: %s", agent_name, e)
        return None
    score_fn = getattr(mod, "score", None)
    return score_fn if callable(score_fn) else None


def get_recipient_metric(agent_name: str) -> Callable:
    """Return a deterministic GEPA-compatible metric for the recipient.

    Uses a per-recipient override (``dspy_signal_heuristic.score``) if one exists,
    else the generic grounding/completeness heuristic. The returned callable
    always yields a ``dspy.Prediction(score in [0,1], feedback=str)``.
    """
    override = _load_override(agent_name)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        if override is not None:
            try:
                try:
                    raw = override(gold, pred, trace, pred_name, pred_trace)
                except TypeError:
                    raw = override(gold, pred, trace)
                return _coerce_to_prediction(raw)
            except Exception as e:  # noqa: BLE001 - never let an override crash GEPA
                logger.error("Recipient override metric for %s raised: %s", agent_name, e)
        return generic_grounding_metric(gold, pred, trace, pred_name, pred_trace)

    return metric
