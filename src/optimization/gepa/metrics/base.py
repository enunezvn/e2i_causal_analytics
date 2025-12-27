"""Base GEPA Metric Protocol for E2I Agents.

This module defines the E2IGEPAMetric protocol that all GEPA metrics must implement.
GEPA uses ScoreWithFeedback to enable reflective evolution through rich textual feedback.

All metrics return a dict with:
- score (float): 0.0-1.0 composite score
- feedback (str): Rich textual feedback for GEPA reflection

Example:
    class MyMetric:
        def __call__(self, gold, pred, trace, pred_name, pred_trace) -> ScoreWithFeedback:
            score = 0.85
            feedback = "Component A passed. Component B needs improvement."
            return {"score": score, "feedback": feedback}
"""

from typing import Optional, Protocol, Union

from dspy import Example, Prediction

# Type aliases
DSPyTrace = list[tuple]
ScoreWithFeedback = dict[str, Union[float, str]]


class E2IGEPAMetric(Protocol):
    """Protocol for E2I GEPA metrics.

    All GEPA metrics must implement this protocol to be compatible with
    the GEPA optimizer. The __call__ method receives:

    - gold: Ground truth Example
    - pred: Model Prediction
    - trace: Full execution trace (optional)
    - pred_name: Name of predictor being optimized (optional)
    - pred_trace: Trace specific to this predictor (optional)

    Returns:
        Either a float score (0.0-1.0) or a ScoreWithFeedback dict.
        ScoreWithFeedback enables GEPA's reflective evolution.
    """

    def __call__(
        self,
        gold: Example,
        pred: Prediction,
        trace: Optional[DSPyTrace],
        pred_name: Optional[str],
        pred_trace: Optional[DSPyTrace],
    ) -> Union[float, ScoreWithFeedback]:
        """Compute score and feedback for GEPA optimization.

        Args:
            gold: Ground truth Example with expected outputs
            pred: Model Prediction to evaluate
            trace: Full DSPy execution trace (optional)
            pred_name: Name of the predictor being optimized (optional)
            pred_trace: Execution trace for this specific predictor (optional)

        Returns:
            Either a float score or a dict with 'score' and 'feedback' keys
        """
        ...


__all__ = [
    "E2IGEPAMetric",
    "ScoreWithFeedback",
    "DSPyTrace",
]
