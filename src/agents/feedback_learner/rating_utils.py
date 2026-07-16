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

from typing import Any, Dict, Mapping, Optional

# ---------------------------------------------------------------------------
# Reward surfaces (#1251)
#
# Ratings on the shared 1-5 scale come from surfaces with structurally
# different reward ceilings: the cognitive workflow's composite reaches 1.0,
# while the copilot turn grader (#1240, ``_grade_copilot_turn``) tops out at
# 0.8 because that surface has no evidence-board/visualization axes. The raw
# scales agree at the bottom (reward 0.5 = rating 3.0 on both — every hard
# gate is bottom-anchored), so per-signal rescaling was rejected (codex-1240
# M2: it inflates exactly the region the gates consume). The residual issue
# is POOLED MEANS: a mixed pool's distance-to-gate depends on source mix, not
# just quality. Remedy: group aggregation by surface. These helpers are the
# aggregation layer's shared surface knowledge.
# ---------------------------------------------------------------------------

COPILOT_SURFACE = "copilotkit"
COGNITIVE_SURFACE = "cognitive"
EXPLICIT_SURFACE = "explicit"

# Rating-space ceilings (1 + 4*reward_ceiling). Any FUTURE top-anchored
# consumer (e.g. a "performing well" pattern requiring avg > 4.5) MUST
# compare each surface pool against ITS OWN ceiling from this map — a raw
# top-anchored threshold structurally excludes copilot-only pools.
SURFACE_RATING_CEILINGS: Dict[str, float] = {
    COPILOT_SURFACE: 1.0 + 4.0 * 0.8,  # 4.2 — honest copilot ceiling (#1240)
    COGNITIVE_SURFACE: 5.0,
    EXPLICIT_SURFACE: 5.0,
}


def rating_surface(metadata: Optional[Mapping[str, Any]]) -> str:
    """Derive the reward surface of one rating item from its metadata.

    ``LearningSignalsFeedbackStore`` stamps ``source="learning_signals"`` on
    every mapped reward signal and passes through ``source_path`` — set to
    ``"copilotkit"`` only by the copilot collector (#1240); the cognitive
    Reflector never sets it. Anything else (chatbot thumbs, unknown) is
    explicit human feedback on the full 1-5 scale.
    """
    meta = metadata or {}
    if meta.get("source") == "learning_signals":
        if meta.get("source_path") == COPILOT_SURFACE:
            return COPILOT_SURFACE
        return COGNITIVE_SURFACE
    return EXPLICIT_SURFACE


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
