"""Guided discovery must request bootstrap stability by default (fix 2)."""

from typing import Any, Dict, cast

import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery.base import DiscoveryConfig, DiscoveryResult


class _CapturingRunner:
    def __init__(self) -> None:
        self.config: DiscoveryConfig | None = None

    async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
        self.config = config
        return DiscoveryResult(success=False, config=config)


def _state(**overrides: Any) -> CausalImpactState:
    state: Dict[str, Any] = {
        "query": "What is the causal effect of t on y?",
        "treatment_var": "t",
        "outcome_var": "y",
        "confounders": ["c"],
        "modeled_confounders": ["c"],
        "data_cache": {
            "estimation_data": pd.DataFrame(
                {"t": [0.0, 1.0] * 20, "y": [0.0, 1.0] * 20, "c": [0.5] * 40}
            )
        },
        "auto_discover": True,
        "discovery_guided": True,
    }
    state.update(overrides)
    return cast(CausalImpactState, state)


class TestGuidedBootstrapWiring:
    @pytest.mark.asyncio
    async def test_guided_mode_defaults_to_bootstrap(self) -> None:
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node.execute(_state())
        assert runner.config is not None
        assert runner.config.bootstrap_resamples == 20

    @pytest.mark.asyncio
    async def test_state_key_overrides_default(self) -> None:
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node.execute(_state(discovery_bootstrap_resamples=0))
        assert runner.config is not None
        assert runner.config.bootstrap_resamples == 0

    @pytest.mark.asyncio
    async def test_unguided_mode_stays_bootstrap_off(self) -> None:
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node.execute(_state(discovery_guided=False))
        assert runner.config is not None
        assert runner.config.bootstrap_resamples == 0
        assert len(runner.config.algorithms) >= 2
