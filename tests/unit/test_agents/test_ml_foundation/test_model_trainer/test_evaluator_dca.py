"""Unit tests for ``_compute_dca_curves`` (Phase 1 W1 day 1).

Covers shard 20 §F DCA rows of the test plan:
- ``test_dca_curves_shape_and_serializability``
- ``test_dca_treat_all_matches_vickers_2006_eq3``
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _compute_dca_curves,
)


def _logistic_dgp(
    n: int = 1000, prevalence: float = 0.20, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < prevalence).astype(int)
    proba_pos = np.where(
        y == 1,
        rng.beta(5.0, 2.0, size=n),
        rng.beta(2.0, 5.0, size=n),
    )
    return y, proba_pos


def test_dca_curves_shape_and_serializability() -> None:
    """All 4 arrays match ``n_grid_points``; result is JSON-dumpable."""
    y, p = _logistic_dgp(n=1500, prevalence=0.20)
    tau_grid = np.linspace(0.05, 0.30, 21)
    res = _compute_dca_curves(y, p, tau_grid)

    assert res["n_grid_points"] == 21
    assert len(res["tau_grid"]) == 21
    assert len(res["nb_model"]) == 21
    assert len(res["nb_treat_all"]) == 21
    assert len(res["nb_treat_none"]) == 21
    assert all(v == 0.0 for v in res["nb_treat_none"])
    assert 0.0 <= res["prevalence"] <= 1.0
    assert res["tau_low"] == pytest.approx(0.05)
    assert res["tau_high"] == pytest.approx(0.30)

    # Must be JSON-serializable for ``mlflow.log_dict``.
    serialized = json.dumps(res)
    assert "tau_grid" in serialized
    # Round-trip keeps the same shapes.
    restored = json.loads(serialized)
    assert len(restored["nb_model"]) == 21


def test_dca_treat_all_matches_vickers_2006_eq3() -> None:
    """``nb_treat_all[k] == prev − (1−prev)·τ_k/(1−τ_k)`` to 1e-9."""
    y, p = _logistic_dgp(n=2000, prevalence=0.20)
    tau_grid = np.linspace(0.05, 0.30, 21)
    res = _compute_dca_curves(y, p, tau_grid)
    prev = res["prevalence"]

    expected = [prev - (1.0 - prev) * float(t) / (1.0 - float(t)) for t in tau_grid]
    for actual, want in zip(res["nb_treat_all"], expected, strict=True):
        assert actual == pytest.approx(want, abs=1e-9)


def test_dca_curves_empty_grid_returns_empty_lists() -> None:
    """An empty grid yields zero-length arrays + NaN bounds."""
    y, p = _logistic_dgp(n=200, prevalence=0.20)
    res = _compute_dca_curves(y, p, [])
    assert res["n_grid_points"] == 0
    assert res["tau_grid"] == []
    assert res["nb_model"] == []
    assert res["nb_treat_all"] == []
    assert res["nb_treat_none"] == []
    assert math.isnan(res["tau_low"])
    assert math.isnan(res["tau_high"])
