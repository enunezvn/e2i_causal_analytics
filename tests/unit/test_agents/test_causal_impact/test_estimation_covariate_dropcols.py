"""Shard 07 C1: is_synthetic must never enter the causal_impact design matrix even
on an include_synthetic=True validation run (the all-other-columns covariate branch
would otherwise capture the tag as a constant covariate)."""

import pandas as pd
import pytest

from src.agents.causal_impact.nodes.estimation import EstimationNode


@pytest.mark.unit
def test_is_synthetic_excluded_from_covariates(monkeypatch):
    captured = {}
    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1],
            "outcome": [1.0, 2.0, 1.5, 2.5],
            "x1": [10, 11, 12, 13],
            "is_synthetic": [True, True, True, True],
        }
    )
    node = EstimationNode.__new__(EstimationNode)  # bypass __init__ deps

    class _Sel:
        def select(self, treatment, outcome, covariates, **kw):
            captured["cols"] = list(covariates.columns)
            raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(node, "_get_estimator_selector", lambda *a, **k: _Sel())
    with pytest.raises(RuntimeError):
        node._select_estimator_with_energy_score(df, "treatment", "outcome", None, "ensemble", None)
    assert "is_synthetic" not in captured["cols"]
    assert "x1" in captured["cols"]


@pytest.mark.unit
def test_is_synthetic_dropped_from_explicit_adjustment_set(monkeypatch):
    """Shard 07 C1: is_synthetic must be stripped even when a caller passes it
    EXPLICITLY in the adjustment_set (not just in the all-other-columns branch)."""
    captured = {}
    df = pd.DataFrame(
        {
            "treatment": [0, 1, 0, 1],
            "outcome": [1.0, 2.0, 1.5, 2.5],
            "x1": [10, 11, 12, 13],
            "is_synthetic": [True, True, True, True],
        }
    )
    node = EstimationNode.__new__(EstimationNode)

    class _Sel:
        def select(self, treatment, outcome, covariates, **kw):
            captured["cols"] = list(covariates.columns)
            raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(node, "_get_estimator_selector", lambda *a, **k: _Sel())
    with pytest.raises(RuntimeError):
        node._select_estimator_with_energy_score(
            df,
            "treatment",
            "outcome",
            ["x1", "is_synthetic"],  # explicit adjustment set INCLUDES the tag
            "ensemble",
            None,
        )
    assert "is_synthetic" not in captured["cols"]
    assert "x1" in captured["cols"]
