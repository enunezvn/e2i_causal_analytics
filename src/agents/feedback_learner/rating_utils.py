"""Shared rating normalization for the feedback-learner pipeline.

Every consumer that averages or thresholds user ratings MUST normalize through
this function. The real feedback sources emit ratings in different shapes on
the same logical 1-5 scale:

- ``chatbot_message_feedback``: enum STRINGS (``thumbs_up``/``thumbs_down``)
- ``learning_signals`` (via ``LearningSignalsFeedbackStore``): floats already
  remapped from the raw 0..1 workflow reward onto 1..5

F15 (audit): the low-rating pattern detector previously accepted only
``int``/``float`` — silently dropping every real string rating, so collected
feedback produced zero patterns and ``update_effectiveness`` stayed pinned at
0.0. A codex round then found the same isinstance-gate bug duplicated in the
collector's summary and the per-agent negative detector — hence one shared
normalizer instead of per-site copies.
"""

from typing import Any, Optional


def rating_to_numeric(value: Any) -> Optional[float]:
    """Normalize a user rating to a 1-5 numeric scale.

    Handles numeric ratings, booleans, and the thumbs string forms; returns
    None for genuinely unknown values (excluded, not coerced).
    """
    if isinstance(value, bool):
        return 5.0 if value else 1.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        positive = {"thumbs_up", "thumbsup", "up", "positive", "good", "helpful", "yes", "👍"}
        negative = {"thumbs_down", "thumbsdown", "down", "negative", "bad", "unhelpful", "no", "👎"}
        if v in positive:
            return 5.0
        if v in negative:
            return 1.0
        try:
            return float(v)  # numeric-as-string (e.g. "4")
        except ValueError:
            return None
    return None
