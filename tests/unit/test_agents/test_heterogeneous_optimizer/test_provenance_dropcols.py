"""Shard 07 C2: is_synthetic must never reach a heterogeneous-optimizer design
matrix — neither as a CATE effect modifier, nor as a routed confounder (W), nor
via the hierarchical-analyzer uplift feature matrix — even when a caller passes
it explicitly.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.heterogeneous_optimizer.nodes.hierarchical_analyzer import (
    HierarchicalAnalyzerNode,
)


@pytest.mark.unit
def test_resolve_confounders_strips_is_synthetic():
    node = CATEEstimatorNode.__new__(CATEEstimatorNode)
    state = {"confounders": ["age", "is_synthetic", "region"]}
    available = ["age", "is_synthetic", "region"]
    resolved = node._resolve_confounders(state, available)  # type: ignore[arg-type]
    assert "is_synthetic" not in resolved, (
        f"is_synthetic must not be routed into the nuisance W matrix: {resolved}"
    )
    assert "age" in resolved and "region" in resolved


@pytest.mark.unit
def test_hierarchical_prepare_data_strips_is_synthetic_explicit():
    node = HierarchicalAnalyzerNode.__new__(HierarchicalAnalyzerNode)
    df = pd.DataFrame(
        {
            "T": [0, 1, 0, 1],
            "Y": [1.0, 2.0, 1.5, 2.5],
            "x1": [10, 11, 12, 13],
            "is_synthetic": [False, False, False, False],
        }
    )
    state = {
        "treatment_var": "T",
        "outcome_var": "Y",
        "effect_modifiers": ["x1", "is_synthetic"],  # explicit list includes tag
        "segment_vars": [],
    }
    X, _t, _y = node._prepare_data(df, state)  # type: ignore[arg-type]
    assert "is_synthetic" not in X.columns, (
        f"is_synthetic must not enter the uplift design matrix: {list(X.columns)}"
    )
    assert "x1" in X.columns


@pytest.mark.unit
def test_hierarchical_prepare_data_strips_is_synthetic_all_numeric():
    """When effect_modifiers is empty, the all-numeric fallback must also drop
    the provenance column."""
    node = HierarchicalAnalyzerNode.__new__(HierarchicalAnalyzerNode)
    df = pd.DataFrame(
        {
            "T": [0, 1, 0, 1],
            "Y": [1.0, 2.0, 1.5, 2.5],
            "x1": [10, 11, 12, 13],
            "is_synthetic": np.array([0, 0, 0, 0]),  # numeric -> would survive
        }
    )
    state = {
        "treatment_var": "T",
        "outcome_var": "Y",
        "effect_modifiers": [],
        "segment_vars": [],
    }
    X, _t, _y = node._prepare_data(df, state)  # type: ignore[arg-type]
    assert "is_synthetic" not in X.columns, (
        f"all-numeric fallback leaked is_synthetic: {list(X.columns)}"
    )
    assert "x1" in X.columns
