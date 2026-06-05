"""P6 / H5 — uplift metric scaling: Qini + AUUC normalize by AREA, not height.

Findings (causal-validation-pipeline-review-20260605):
- H5: qini_coefficient normalized a true np.trapz AREA by a curve HEIGHT
  (perfect_area = qini_values[-1]) → dimensionally inconsistent, not a real Qini.
- MED: AUUC had the same height-as-area defect (random_area = 0.5·overall_uplift).

Oracle: an in-hindsight optimally-ranked model achieves the perfect curve, so its
normalized coefficient must be ≈ 1.0. The buggy normalization gives ~0.75 for the
Qini oracle (height denominator), so this is a discriminating red.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.causal_engine.uplift.metrics import auuc, qini_coefficient


def _dgp(seed: int = 1, n: int = 4000):
    rng = np.random.RandomState(seed)
    w = rng.binomial(1, 0.5, n)
    tau = rng.uniform(0, 1, n)
    base = rng.uniform(0, 1, n)
    y = (base + w * tau > 1.0).astype(float)
    return w, y


def _oracle_scores(w, y):
    """Optimal in-hindsight ranking: greedy by per-unit marginal Qini gain."""
    w = np.asarray(w)
    n_t = w.sum()
    n_c = len(w) - n_t
    ratio = (n_t / n_c) if n_c > 0 else 1.0
    return w * y - (1 - w) * y * ratio


class TestQiniNormalization:
    def test_oracle_model_coefficient_is_one(self):
        w, y = _dgp()
        coef = qini_coefficient(_oracle_scores(w, y), w, y)
        # The optimal-ranking model achieves the perfect curve → coefficient ≈ 1.
        # (Buggy height-denominator gives ~0.75.)
        assert coef == pytest.approx(1.0, abs=0.05), f"oracle Qini must be ≈1.0, got {coef}"

    def test_random_model_near_zero(self):
        w, y = _dgp()
        rng = np.random.RandomState(99)
        coef = qini_coefficient(rng.uniform(0, 1, len(w)), w, y)
        assert abs(coef) < 0.15, f"random Qini should be ≈0, got {coef}"

    def test_scale_invariant(self):
        w, y = _dgp()
        s = _oracle_scores(w, y)
        c1 = qini_coefficient(s, w, y)
        c2 = qini_coefficient(s, w, y * 1000.0)
        assert c1 == pytest.approx(c2, abs=1e-6)


class TestAUUCNormalization:
    def test_oracle_model_normalized_is_one(self):
        w, y = _dgp()
        a = auuc(_oracle_scores(w, y), w, y, normalize=True)
        # Normalized against the perfect-targeting AREA → oracle ≈ 1.0.
        assert a == pytest.approx(1.0, abs=0.05), f"oracle AUUC must be ≈1.0, got {a}"

    def test_perfect_beats_random(self):
        w, y = _dgp()
        rng = np.random.RandomState(7)
        a_oracle = auuc(_oracle_scores(w, y), w, y, normalize=True)
        a_random = auuc(rng.uniform(0, 1, len(w)), w, y, normalize=True)
        assert a_oracle > a_random

    def test_scale_invariant(self):
        w, y = _dgp()
        s = _oracle_scores(w, y)
        a1 = auuc(s, w, y, normalize=True)
        a2 = auuc(s, w, y * 1000.0, normalize=True)
        assert a1 == pytest.approx(a2, abs=1e-6)
