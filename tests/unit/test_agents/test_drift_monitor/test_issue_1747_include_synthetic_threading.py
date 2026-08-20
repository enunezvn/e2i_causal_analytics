"""Issue #1747 (part 2): the include_synthetic provenance opt-in must reach the
drift connectors.

Measured (2026-08-20, live DB): the drift feature store is 100% synthetic-tagged
(444,068 feature_values rows, zero non-synthetic in any month; all 37,865
ml_predictions rows synthetic). The supabase connector's query methods already
take ``include_synthetic: bool = False`` and apply the provenance
default-exclude predicate — but NO drift node passes the flag, and neither
``DriftMonitorInput`` nor ``DriftMonitorState`` carries it. Net effect: even a
dispatch that opted into synthetic provenance (the #872/#880 channels, ambient
on the showcase instance via E2I_INCLUDE_SYNTHETIC) reads ZERO rows — the flag
dies at the agent boundary.

This suite pins the threading seam by seam:
input model -> initial state -> each detector node -> connector kwargs, plus
signature compatibility on the base/mock connectors (the nodes must be able to
pass the kwarg to ANY connector implementation).
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Dict, List

import pytest

from src.agents.drift_monitor.agent import DriftMonitorAgent, DriftMonitorInput
from src.agents.drift_monitor.connectors.base import BaseDataConnector, TimeWindow
from src.agents.drift_monitor.connectors.mock_connector import MockDataConnector
from src.agents.drift_monitor.nodes.concept_drift import ConceptDriftNode
from src.agents.drift_monitor.nodes.data_drift import DataDriftNode
from src.agents.drift_monitor.nodes.model_drift import ModelDriftNode
from src.agents.drift_monitor.state import DriftMonitorState


class _CaptureConnector:
    """Signature-tolerant connector double that records every query call's
    kwargs. Returns are the emptiest shapes that keep each node's happy path
    walking far enough to AWAIT every query coroutine (concept_drift creates
    its feature coroutines eagerly but only awaits them when the labeled
    predictions clear ``_min_samples`` — an un-awaited coroutine never runs
    the capture). The assertions here are about WHAT the nodes pass, not what
    they compute."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def query_features(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({"method": "query_features", **kwargs})
        return {}

    async def query_predictions(self, **kwargs: Any) -> Any:
        import numpy as np

        from src.agents.drift_monitor.connectors.base import PredictionData

        self.calls.append({"method": "query_predictions", **kwargs})
        return PredictionData(model_id="model-1747", scores=np.zeros(40))

    async def query_labeled_predictions(self, **kwargs: Any) -> Any:
        import numpy as np

        from src.agents.drift_monitor.connectors.base import PredictionData

        self.calls.append({"method": "query_labeled_predictions", **kwargs})
        return PredictionData(model_id="model-1747", scores=np.zeros(40))


def _state(include_synthetic: bool = True) -> DriftMonitorState:
    return {
        "query": "drift check",
        "model_id": "model-1747",
        "features_to_monitor": ["trx_total", "hcp_engagement_frequency"],
        "time_window": "7d",
        "significance_level": 0.05,
        "psi_threshold": 0.1,
        "include_synthetic": include_synthetic,
    }


# ---------------------------------------------------------------------------
# Input model + state + initial-state threading
# ---------------------------------------------------------------------------


def test_input_model_declares_include_synthetic_default_false() -> None:
    inp = DriftMonitorInput(query="q", features_to_monitor=["f"])
    assert inp.include_synthetic is False


def test_input_model_accepts_include_synthetic_true() -> None:
    inp = DriftMonitorInput(query="q", features_to_monitor=["f"], include_synthetic=True)
    assert inp.include_synthetic is True


def test_state_declares_include_synthetic() -> None:
    # LangGraph silently DROPS undeclared state keys — the flag must be a
    # declared channel or it dies between the resolver and the nodes.
    assert "include_synthetic" in DriftMonitorState.__annotations__


def test_create_initial_state_threads_include_synthetic() -> None:
    agent = DriftMonitorAgent(enable_mlflow=False)
    inp = DriftMonitorInput(query="q", features_to_monitor=["f"], include_synthetic=True)
    state = agent._create_initial_state(inp)
    assert state.get("include_synthetic") is True

    inp_default = DriftMonitorInput(query="q", features_to_monitor=["f"])
    state_default = agent._create_initial_state(inp_default)
    assert state_default.get("include_synthetic") is False


# ---------------------------------------------------------------------------
# Node -> connector kwarg threading (RED on main: kwarg never passed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_drift_passes_include_synthetic_to_connector() -> None:
    connector = _CaptureConnector()
    node = DataDriftNode(connector=connector)  # type: ignore[arg-type]
    await node._fetch_data(_state(include_synthetic=True))
    feature_calls = [c for c in connector.calls if c["method"] == "query_features"]
    assert len(feature_calls) == 2, "baseline + current window fetches expected"
    for call in feature_calls:
        assert call.get("include_synthetic") is True


@pytest.mark.asyncio
async def test_data_drift_defaults_to_real_mode_when_flag_absent() -> None:
    connector = _CaptureConnector()
    node = DataDriftNode(connector=connector)  # type: ignore[arg-type]
    state = _state()
    state.pop("include_synthetic")
    await node._fetch_data(state)
    for call in connector.calls:
        assert call.get("include_synthetic") is False


@pytest.mark.asyncio
async def test_model_drift_passes_include_synthetic_to_connector() -> None:
    connector = _CaptureConnector()
    node = ModelDriftNode(connector=connector)  # type: ignore[arg-type]
    await node.execute(_state(include_synthetic=True))
    pred_calls = [c for c in connector.calls if c["method"] == "query_predictions"]
    assert len(pred_calls) == 2, "baseline + current prediction fetches expected"
    for call in pred_calls:
        assert call.get("include_synthetic") is True


@pytest.mark.asyncio
async def test_concept_drift_passes_include_synthetic_to_connector() -> None:
    connector = _CaptureConnector()
    node = ConceptDriftNode(connector=connector)  # type: ignore[arg-type]
    await node.execute(_state(include_synthetic=True))
    labeled = [c for c in connector.calls if c["method"] == "query_labeled_predictions"]
    features = [c for c in connector.calls if c["method"] == "query_features"]
    assert len(labeled) == 2, "baseline + current labeled-prediction fetches expected"
    assert len(features) == 2, "baseline + current correlation-feature fetches expected"
    for call in labeled + features:
        assert call.get("include_synthetic") is True


# ---------------------------------------------------------------------------
# Connector interface compatibility: the nodes must be able to pass the kwarg
# to ANY connector implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_name",
    ["query_features", "query_predictions", "query_labeled_predictions"],
)
def test_base_connector_declares_include_synthetic(method_name: str) -> None:
    sig = inspect.signature(getattr(BaseDataConnector, method_name))
    assert "include_synthetic" in sig.parameters, (
        f"BaseDataConnector.{method_name} must declare include_synthetic — the "
        "nodes pass it unconditionally"
    )
    assert sig.parameters["include_synthetic"].default is False


@pytest.mark.asyncio
async def test_mock_connector_accepts_include_synthetic() -> None:
    connector = MockDataConnector()
    window = TimeWindow(
        start=datetime(2026, 8, 1),
        end=datetime(2026, 8, 8),
        label="current",
    )
    # Must not raise TypeError: the intentional dev/harness connector has to
    # stay call-compatible with the nodes.
    await connector.query_features(feature_names=["f"], time_window=window, include_synthetic=True)
    await connector.query_predictions(model_id="m", time_window=window, include_synthetic=True)
    await connector.query_labeled_predictions(
        model_id="m", time_window=window, include_synthetic=True
    )
