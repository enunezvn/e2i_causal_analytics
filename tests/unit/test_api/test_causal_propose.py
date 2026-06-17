"""Unit coverage for the propose-questions screening signal.

``_adjusted_partial_corr`` is the data-driven ranking signal behind
``GET /causal/propose-questions`` (the agent proposes ranked treatment->outcome
questions, the user confirms). It must (a) rank a strong treatment->outcome
association above a weak one, and (b) remove confounding — a pair that shares a
confounder but has NO direct effect must score ~0 after adjustment. Pure
function, synthetic data, no DB.
"""

import numpy as np
import pandas as pd
import pytest

from src.api.routes.causal import _adjusted_partial_corr


@pytest.mark.unit
def test_ranks_strong_association_above_weak():
    rng = np.random.default_rng(3)
    n = 1500
    c = rng.normal(size=n)
    t = (0.9 * c + rng.normal(size=n) > 0).astype(float)
    strong = 0.8 * t + 0.5 * c + rng.normal(size=n) * 0.3
    weak = 0.05 * t + 0.5 * c + rng.normal(size=n)
    df = pd.DataFrame({"t": t, "strong": strong, "weak": weak, "c": c})

    s = abs(_adjusted_partial_corr(df, "t", "strong", ["c"]))
    w = abs(_adjusted_partial_corr(df, "t", "weak", ["c"]))
    assert s > w


@pytest.mark.unit
def test_adjustment_removes_confounding():
    """t and o share a confounder c but t has NO direct effect on o: the raw
    correlation is large (confounded) while the adjusted partial correlation
    collapses toward 0 — so the screening signal won't propose a spurious pair."""
    rng = np.random.default_rng(5)
    n = 2000
    c = rng.normal(size=n)
    t = 0.9 * c + rng.normal(size=n) * 0.4
    o = 0.9 * c + rng.normal(size=n) * 0.4  # depends on c, NOT on t
    df = pd.DataFrame({"t": t, "o": o, "c": c})

    raw = abs(float(np.corrcoef(t, o)[0, 1]))
    adjusted = abs(_adjusted_partial_corr(df, "t", "o", ["c"]))
    assert raw > 0.3
    assert adjusted < 0.1


@pytest.mark.unit
def test_zero_variance_returns_none():
    df = pd.DataFrame({"t": [1.0, 1.0, 1.0], "o": [0.0, 1.0, 0.0], "c": [0.1, 0.2, 0.3]})
    assert _adjusted_partial_corr(df, "t", "o", ["c"]) is None
