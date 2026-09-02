"""Guided discovery must request bootstrap stability by default (fix 2)."""

from typing import Any, Dict, List, Tuple, cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
)


class _CapturingRunner:
    def __init__(self) -> None:
        self.config: DiscoveryConfig | None = None

    async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
        self.config = config
        return DiscoveryResult(success=False, config=config)


class _AugmentingRunner:
    """Stub runner returning a canned, successful ``DiscoveryResult`` whose
    edges are engineered so the REAL gate evaluates to AUGMENT with exactly
    one high-confidence beyond-prior edge — the same trio (required edge
    stability 1.0, beyond-prior corroborated edge at the augment threshold,
    beyond-prior uncorroborated edge) pinned in
    ``test_gate.py::TestCorroboration::test_bootstrap_corroborated_edge_augments_at_threshold_boundary``.
    """

    def __init__(self, edges_with_stability: List[Tuple[Tuple[str, str], float]]) -> None:
        self._edges_with_stability = edges_with_stability
        self.config: DiscoveryConfig | None = None

    async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
        self.config = config
        dag = nx.DiGraph()
        discovered = []
        for (source, target), stability in self._edges_with_stability:
            dag.add_edge(source, target)
            discovered.append(
                DiscoveredEdge(
                    source=source,
                    target=target,
                    confidence=stability,
                    algorithm_votes=1,
                    algorithms=["pc"],
                    bootstrap_stability=stability,
                )
            )
        return DiscoveryResult(
            success=True,
            config=config,
            ensemble_dag=dag,
            edges=discovered,
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((2, 2), dtype=int),
                    edge_list=[edge for edge, _ in self._edges_with_stability],
                    runtime_seconds=0.01,
                    converged=True,
                )
            ],
        )


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


class TestAugmentActuallyAugments:
    """Fix: ``GateEvaluation.to_dict()`` used to serialize only
    ``n_high_confidence_edges`` (a count), so graph_builder's AUGMENT branch
    read an always-empty ``high_confidence_edges`` list and shipped the bare
    manual DAG — indistinguishable from REVIEW. With ``_state()``'s guided
    setup (treatment "t", outcome "y", modeled confounder "c"), the required
    edges are (t, y), (c, t), (c, y) — exactly the manual DAG's edges — so an
    AUGMENT that actually augments must ship one edge beyond that set."""

    @pytest.mark.asyncio
    async def test_augment_decision_ships_manual_dag_plus_high_confidence_edge(
        self,
    ) -> None:
        node = GraphBuilderNode()
        runner = _AugmentingRunner(
            edges_with_stability=[
                (("t", "y"), 1.0),  # prior-required -> excluded from augment
                (("z", "y"), 0.9),  # beyond-prior, corroborated -> augment-eligible
                (("q", "r"), 0.3),  # beyond-prior, uncorroborated -> not eligible
            ]
        )
        node._discovery_runner = runner  # type: ignore[assignment]

        result = await node.execute(_state())

        assert result.get("status") != "failed", result.get("error_message")
        gate_evaluation = result["discovery_gate_evaluation"]
        assert gate_evaluation["decision"] == "augment"
        # Consumer projection only — exact-dict-shape ownership lives in
        # test_gate.py::TestCorroboration::test_to_dict_serializes_high_confidence_edges.
        # This test cares that graph_builder can actually read the edge it consumes
        # (source/target), not the full serialized shape.
        high_conf_edges = gate_evaluation["high_confidence_edges"]
        assert [(e["source"], e["target"]) for e in high_conf_edges] == [("z", "y")]

        manual_edges = {("t", "y"), ("c", "t"), ("c", "y")}
        shipped_edges = set(result["causal_graph"]["edges"])
        assert shipped_edges == manual_edges | {("z", "y")}
        assert result["causal_graph"]["augmented_edges"] == [("z", "y")]
