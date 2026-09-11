"""``rank_drivers`` settings apply to the call that passes them, never to later calls (#2003).

#2003 declared ``importance_percentile`` to the planner (``concordance_threshold`` already
was). ``DriverRankerTool`` is a process-wide singleton (``get_ranker_tool``) and its
``invoke`` wrote a non-default value onto the shared ``DriverRanker`` without ever
resetting it, so one request's ``importance_percentile=0.5`` silently re-scored every
later request that omitted it (or passed the default). Real ranker, real inputs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import numpy as np

from src.tool_registry.tools.causal_discovery import DriverRankerTool

FEATURES = [f"f{i}" for i in range(8)]


def _inputs(**overrides: Any) -> Dict[str, Any]:
    rng = np.random.default_rng(2003)
    # f0..f3 cause y; f4..f7 are graph nodes with no path to y but carry the most SHAP.
    edges = [{"source": f"f{i}", "target": "y"} for i in range(4)]
    edges += [{"source": "f4", "target": "f5"}, {"source": "f6", "target": "f7"}]
    scale = np.array([0.1, 0.2, 0.3, 0.4, 4.0, 3.0, 2.0, 1.0])
    shap = rng.normal(size=(80, len(FEATURES))) * scale
    return {
        "dag_edge_list": edges,
        "target": "y",
        "shap_values": shap.tolist(),
        "feature_names": FEATURES,
        **overrides,
    }


def _categories(tool: DriverRankerTool, **overrides: Any) -> Dict[str, Any]:
    out = asyncio.run(tool.invoke(_inputs(**overrides)))
    assert out.success, out.errors
    return {
        "causal_only": sorted(out.causal_only_features),
        "predictive_only": sorted(out.predictive_only_features),
        "concordant": sorted(out.concordant_features),
    }


def test_the_non_default_settings_change_the_result():
    """Positive control: the settings are observable, so the leak checks below are not vacuous."""
    assert _categories(DriverRankerTool()) != _categories(
        DriverRankerTool(), importance_percentile=0.5, concordance_threshold=0
    )


def test_an_omitted_setting_does_not_inherit_the_previous_call():
    baseline = _categories(DriverRankerTool())
    shared = DriverRankerTool()
    _categories(shared, importance_percentile=0.5, concordance_threshold=0)
    assert _categories(shared) == baseline


def test_an_explicit_default_does_not_inherit_the_previous_call():
    baseline = _categories(DriverRankerTool())
    shared = DriverRankerTool()
    _categories(shared, importance_percentile=0.5, concordance_threshold=0)
    assert _categories(shared, importance_percentile=0.25, concordance_threshold=2) == baseline
