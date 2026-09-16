"""#2027 part B: the agent's output builder passes the hierarchical node's nested CI
and its exclusion list through to ``HeterogeneousOptimizerOutput``.

``HeterogeneousOptimizerOutput`` declares ``nested_ci`` and (since #2027)
``nested_ci_excluded_segments``, but ``_build_output`` never copied either from the
final state, so on the agent surface the exclusions reached consumers only through
``warnings``. These tests drive the builder directly with a final state carrying
both keys, and with neither.
"""

from __future__ import annotations

from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent
from src.agents.heterogeneous_optimizer.connectors import MockDataConnector

_EXCLUDED = [
    {
        "segment_id": 0,
        "segment_name": "seg0",
        "n": 100,
        "reason": "no_measured_uncertainty",
        "detail": "segment has no measured standard error and/or confidence-interval bound",
    }
]
_NESTED_CI = {"aggregate_ate": 0.2, "n_segments_included": 1}


def test_build_output_passes_nested_ci_and_exclusions_through() -> None:
    agent = HeterogeneousOptimizerAgent(data_connector=MockDataConnector())
    out = agent._build_output(  # type: ignore[arg-type]
        {
            "overall_ate": 0.3,
            "nested_ci": _NESTED_CI,
            "nested_ci_excluded_segments": _EXCLUDED,
            "warnings": [],
            "errors": [],
        }
    )

    assert out["nested_ci"] == _NESTED_CI
    assert out["nested_ci_excluded_segments"] == _EXCLUDED


def test_build_output_without_hierarchical_keys_is_none_and_empty() -> None:
    agent = HeterogeneousOptimizerAgent(data_connector=MockDataConnector())
    out = agent._build_output({"overall_ate": 0.3, "warnings": [], "errors": []})  # type: ignore[arg-type]

    assert out["nested_ci"] is None
    assert out["nested_ci_excluded_segments"] == []
