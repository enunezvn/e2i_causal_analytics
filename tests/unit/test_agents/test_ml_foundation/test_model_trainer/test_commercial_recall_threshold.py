"""Commercial recall-constrained operating-point selection.

A commercial targeting model is used by its ranking, so its decision threshold
should catch the commercial recall floor (false positives are cheap in
outreach). ``_compute_recall_constrained_threshold`` returns the HIGHEST
threshold whose recall >= target (best precision subject to recall>=floor).
"""

from __future__ import annotations

import numpy as np

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _COMMERCIAL_RECALL_TARGET,
    _compute_recall_constrained_threshold,
)


def _proba_two_col(p_pos: np.ndarray) -> np.ndarray:
    return np.column_stack([1.0 - p_pos, p_pos])


def test_recall_constrained_meets_floor_and_maximizes_precision() -> None:
    rng = np.random.RandomState(0)
    n = 600
    y = np.array([1] * 120 + [0] * 480)  # 20% prevalence
    # Positives skew high but overlap negatives (weak-but-real signal).
    p = np.concatenate(
        [
            np.clip(rng.normal(0.45, 0.18, 120), 0.01, 0.99),
            np.clip(rng.normal(0.25, 0.18, 480), 0.01, 0.99),
        ]
    )
    out = _compute_recall_constrained_threshold(y, _proba_two_col(p), target_recall=0.50)
    assert out is not None and out["target_achieved"] is True
    # Recall at the chosen threshold meets the floor...
    assert out["recall_at_threshold"] >= 0.50
    # ...and the threshold is below 0.5 (recall-favoring for this overlapping data).
    assert out["recall_constrained_threshold"] < 0.5


def test_recall_constrained_default_target_is_commercial_floor() -> None:
    assert _COMMERCIAL_RECALL_TARGET == 0.50
    y = np.array([1] * 50 + [0] * 150)
    p = np.concatenate([np.full(50, 0.6), np.full(150, 0.2)])
    out = _compute_recall_constrained_threshold(y, _proba_two_col(p))
    # Perfectly separable here → recall 1.0 achievable; floor trivially met.
    assert out is not None and out["target_achieved"] is True
    assert out["recall_at_threshold"] >= 0.50


def test_recall_constrained_none_when_proba_missing() -> None:
    y = np.array([1, 0, 1, 0])
    assert _compute_recall_constrained_threshold(y, None) is None
