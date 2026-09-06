"""Investigator reward honours the investigator's own sufficiency verdict (#1904).

The reward was ``min(1, sum(relevance) / 3)``. The investigator declares
sufficiency at two items (``len(evidence_board) >= 2``), so a two-item run
needed mean relevance >= 0.75 to clear the bottom-anchored negative line
(reward 0.5 = rating 3.0 on the feedback learner's 1-5 scale). Live prod
signals: 8 of 8 two-item sufficient runs graded negative, and the pattern
analyzer needs only three such rows to raise a "high negative feedback"
pattern for cognitive_investigator.
"""

from unittest.mock import MagicMock

import pytest

# DSPy import has parallel-worker race conditions; pin to one worker.
pytestmark = pytest.mark.xdist_group(name="dspy_integration")


def _module():
    from src.rag.cognitive_rag_dspy import ReflectorModule

    return ReflectorModule(
        {"episodic": MagicMock(), "semantic": MagicMock(), "procedural": MagicMock()},
        MagicMock(),
    )


def _state(relevances, *, sufficient):
    from src.rag.cognitive_rag_dspy import CognitiveState, Evidence, MemoryType

    state = CognitiveState(user_query="q", conversation_id="c")
    state.evidence_board = [
        Evidence(source=list(MemoryType)[0], hop_number=1, content="e", relevance_score=r)
        for r in relevances
    ]
    state.sufficient_evidence = sufficient
    return state


def _investigator_reward(state, user_feedback=None):
    signals = _module()._collect_training_signals(state, user_feedback=user_feedback)
    return next(s for s in signals if s["type"] == "investigator")["reward"]


class TestInvestigatorRewardSufficiencyFloor:
    def test_two_item_sufficient_run_is_never_negative(self):
        """Live shape: two items at ~0.6 relevance, investigator said sufficient.
        Old formula: 1.2/3 = 0.4 -> rating 2.6 -> negative. Floored at the
        bottom anchor (0.5 = neutral rating 3.0)."""
        reward = _investigator_reward(_state([0.6, 0.6], sufficient=True))
        assert reward == pytest.approx(0.5)

    def test_explicit_negative_feedback_still_pushes_a_sufficient_run_under_the_line(self):
        """The floor is applied before the user-feedback adjustment, so an
        explicit negative rating can still grade a sufficient run negative."""
        state = _state([0.6, 0.6], sufficient=True)
        reward = _investigator_reward(state, user_feedback="negative")
        assert reward == pytest.approx(0.4)

    def test_insufficient_run_is_not_lifted(self):
        """Pin: the floor follows the investigator's verdict, not an item count."""
        reward = _investigator_reward(_state([0.6, 0.6], sufficient=False))
        assert reward == pytest.approx(0.4)

    def test_empty_board_stays_zero_even_if_marked_sufficient(self):
        """Pin: no evidence is never neutral, whatever the flag says."""
        reward = _investigator_reward(_state([], sufficient=True))
        assert reward == 0.0

    def test_strong_run_is_not_capped_by_the_floor(self):
        """Pin: the floor only lifts; 3 x 0.9 -> 0.9 as before."""
        reward = _investigator_reward(_state([0.9, 0.9, 0.9], sufficient=True))
        assert reward == pytest.approx(0.9)
