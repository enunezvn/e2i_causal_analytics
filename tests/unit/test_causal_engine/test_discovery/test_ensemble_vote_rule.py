"""The ensemble vote must mean AGREEMENT (lane: discovery-vote-rule, 2026-09-22).

``DiscoveryRunner._build_ensemble`` computed ``min_votes = max(1, int(n * threshold))``.
With the two-voter default ([GES, PC]) and the default threshold 0.5 that is ONE vote:
an edge found by EITHER algorithm was included, so the "vote" was a union and a
single-voter edge shipped at confidence 0.5. The documented meaning of
``ensemble_threshold`` is "minimum fraction of algorithms that must AGREE on an
edge"; agreement needs at least two parties, and "at least a fraction t" is
``votes >= ceil(n * t)``, not ``int``.

The gate then has to be told how much the voters DISAGREED, because every edge
that survives an agreement filter is (by construction) agreed on: the ensemble
records a vote census on the DAG and the gate's corroboration reads it.

Every consumer that used to hard-code ``["ges", "pc"]`` reads
``DEFAULT_DISCOVERY_ALGORITHMS`` instead.
"""

from __future__ import annotations

from typing import Any, Dict, cast
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.base import (
    DEFAULT_DISCOVERY_ALGORITHM_NAMES,
    DEFAULT_DISCOVERY_ALGORITHMS,
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryGateDecision,
    DiscoveryResult,
)
from src.causal_engine.discovery.gate import DiscoveryGate
from src.causal_engine.discovery.runner import DiscoveryRunner

GES = DiscoveryAlgorithmType.GES
PC = DiscoveryAlgorithmType.PC
FCI = DiscoveryAlgorithmType.FCI
DL = DiscoveryAlgorithmType.DIRECT_LINGAM


def _run(algorithm, edges, converged=True, n=4):
    return AlgorithmResult(
        algorithm=algorithm,
        adjacency_matrix=np.zeros((n, n), dtype=int),
        edge_list=list(edges),
        runtime_seconds=0.01,
        converged=converged,
        metadata={} if converged else {"error": "boom"},
    )


NODES = ["A", "B", "C", "D"]


class TestAgreementRule:
    def test_two_converged_algorithms_require_agreement_at_the_default_threshold(self):
        """GES and PC each found two edges and share one: only the shared edge is a vote."""
        runner = DiscoveryRunner(enable_tracing=False)
        results = [_run(GES, [("A", "B"), ("B", "C")]), _run(PC, [("A", "B"), ("A", "C")])]
        edges, dag = runner._build_ensemble(results, NODES, threshold=0.5)
        assert {(e.source, e.target) for e in edges} == {("A", "B")}
        assert edges[0].algorithm_votes == 2
        assert edges[0].confidence == 1.0
        assert set(dag.edges()) == {("A", "B")}

    def test_min_votes_is_the_ceiling_of_the_fraction(self):
        """3 converged at 0.5: ``int(1.5) = 1`` was a union; ``ceil(1.5) = 2`` is agreement."""
        runner = DiscoveryRunner(enable_tracing=False)
        results = [
            _run(GES, [("A", "B"), ("B", "C")]),
            _run(PC, [("A", "B"), ("C", "D")]),
            _run(FCI, [("A", "B"), ("B", "C")]),
        ]
        edges, _ = runner._build_ensemble(results, NODES, threshold=0.5)
        by_edge = {(e.source, e.target): e for e in edges}
        assert set(by_edge) == {("A", "B"), ("B", "C")}
        assert by_edge[("A", "B")].confidence == pytest.approx(1.0)
        assert by_edge[("B", "C")].confidence == pytest.approx(2 / 3)
        assert dag_census(_) == {
            "n_converged": 3,
            "min_votes": 2,
            "n_candidate_edges": 3,
            "n_agreed_edges": 2,
            "agreement_rate": pytest.approx(2 / 3),
        }

    def test_threshold_one_requires_unanimity(self):
        runner = DiscoveryRunner(enable_tracing=False)
        results = [
            _run(GES, [("A", "B"), ("B", "C")]),
            _run(PC, [("A", "B"), ("B", "C")]),
            _run(FCI, [("A", "B")]),
        ]
        edges, _ = runner._build_ensemble(results, NODES, threshold=1.0)
        assert {(e.source, e.target) for e in edges} == {("A", "B")}

    def test_a_low_threshold_still_needs_two_voters(self):
        """A fraction that resolves below two votes is not a vote; the floor is 2."""
        runner = DiscoveryRunner(enable_tracing=False)
        results = [_run(GES, [("A", "B")]), _run(PC, [("B", "C")]), _run(FCI, [("C", "D")])]
        edges, _ = runner._build_ensemble(results, NODES, threshold=0.1)
        assert edges == []

    def test_single_converged_algorithm_keeps_every_edge_for_the_bootstrap_path(self):
        """One converged voter cannot agree with anyone: every edge ships at one vote
        (confidence 1.0) and corroboration is the bootstrap's job (unchanged)."""
        runner = DiscoveryRunner(enable_tracing=False)
        results = [_run(PC, [("A", "B"), ("B", "C")]), _run(GES, [("C", "D")], converged=False)]
        edges, dag = runner._build_ensemble(results, NODES, threshold=0.5)
        assert {(e.source, e.target) for e in edges} == {("A", "B"), ("B", "C")}
        assert all(e.algorithm_votes == 1 and e.confidence == 1.0 for e in edges)
        assert dag_census(dag)["min_votes"] == 1
        assert dag_census(dag)["n_converged"] == 1

    def test_failed_algorithms_do_not_count_as_voters(self):
        """Two converged of four: the quorum is over the two that ran (P12), and a
        single-voter edge is still dropped."""
        runner = DiscoveryRunner(enable_tracing=False)
        results = [
            _run(GES, [("A", "B"), ("B", "C")]),
            _run(PC, [("A", "B")]),
            _run(FCI, [], converged=False),
            _run(DL, [], converged=False),
        ]
        edges, dag = runner._build_ensemble(results, NODES, threshold=0.5)
        assert {(e.source, e.target) for e in edges} == {("A", "B")}
        assert edges[0].confidence == 1.0
        assert dag_census(dag)["n_converged"] == 2

    def test_float_rounding_does_not_inflate_the_quorum(self):
        """10 voters at 0.3 need 3 votes; ``ceil(10 * 0.3)`` in floating point is 4."""
        runner = DiscoveryRunner(enable_tracing=False)
        voters = [GES, PC, FCI, DL, DiscoveryAlgorithmType.ICA_LINGAM] * 2
        results = [_run(a, [("A", "B")] if i < 3 else [("C", "D")]) for i, a in enumerate(voters)]
        edges, dag = runner._build_ensemble(results, NODES, threshold=0.3)
        assert dag_census(dag)["min_votes"] == 3
        assert ("A", "B") in {(e.source, e.target) for e in edges}


def dag_census(dag: nx.DiGraph) -> Dict[str, Any]:
    return dict(dag.graph["vote_census"])


class TestGateReadsTheCensus:
    """Every surviving edge is agreed on by construction, so the gate cannot
    score corroboration from the survivors' confidences (they are all 1.0 with
    two voters). It scores the agreement RATE over the candidate edges."""

    def _result(self, results, threshold=0.5):
        runner = DiscoveryRunner(enable_tracing=False)
        edges, dag = runner._build_ensemble(results, NODES, threshold=threshold)
        return DiscoveryResult(
            success=True,
            config=DiscoveryConfig(algorithms=[r.algorithm for r in results]),
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=results,
        )

    def test_disagreement_lowers_corroboration_and_blocks_accept(self):
        # Candidates: A->B (both), B->C (GES only), A->C (PC only): 1 of 3 agreed.
        result = self._result(
            [_run(GES, [("A", "B"), ("B", "C")]), _run(PC, [("A", "B"), ("A", "C")])]
        )
        evaluation = DiscoveryGate().evaluate(result)
        assert evaluation.metadata["corroboration_basis"] == "algorithm_agreement"
        assert evaluation.metadata["corroboration_score"] == pytest.approx(1 / 3)
        assert evaluation.decision != DiscoveryGateDecision.ACCEPT

    def test_full_agreement_scores_one(self):
        result = self._result(
            [_run(GES, [("A", "B"), ("B", "C")]), _run(PC, [("A", "B"), ("B", "C")])]
        )
        evaluation = DiscoveryGate().evaluate(result)
        assert evaluation.metadata["corroboration_score"] == pytest.approx(1.0)
        assert evaluation.decision == DiscoveryGateDecision.ACCEPT

    def test_gate_confidence_is_calibrated_like_the_union_rule_was(self):
        """For two voters the union rule's mean edge confidence over the union was
        ``0.5 + 0.5 * rate``; its gate confidence ``0.8 * that + 0.2 * structure`` equals
        ``0.4 * rate + 0.4 * 1.0 + 0.2 * structure`` — the new formula. The ACCEPT /
        REVIEW / REJECT bands therefore keep their meaning; only the DAG loses the
        single-voter edges."""
        result = self._result(
            [_run(GES, [("A", "B"), ("B", "C")]), _run(PC, [("A", "B"), ("A", "C")])]
        )
        evaluation = DiscoveryGate().evaluate(result)
        structure = evaluation.metadata["structure_score"]
        assert evaluation.confidence == pytest.approx(0.4 * (1 / 3) + 0.4 * 1.0 + 0.2 * structure)

    def test_a_result_without_a_census_keeps_the_votes_per_edge_math(self):
        """Hand-built or externally built results (no census on the DAG) still score."""
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        edges = [
            DiscoveredEdge(source="A", target="B", confidence=1.0, algorithm_votes=3),
            DiscoveredEdge(source="B", target="C", confidence=1.0, algorithm_votes=3),
            DiscoveredEdge(source="A", target="C", confidence=2 / 3, algorithm_votes=2),
        ]
        results = [_run(GES, [("A", "B")]), _run(PC, [("A", "B")]), _run(FCI, [("A", "B")])]
        result = DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=results,
        )
        evaluation = DiscoveryGate().evaluate(result)
        assert evaluation.metadata["corroboration_score"] == pytest.approx((1 + 1 + 2 / 3) / 3)


class TestAlgorithmAgreementProperty:
    def test_property_reports_the_census_rate_not_the_survivors(self):
        runner = DiscoveryRunner(enable_tracing=False)
        results = [_run(GES, [("A", "B"), ("B", "C")]), _run(PC, [("A", "B"), ("A", "C")])]
        edges, dag = runner._build_ensemble(results, NODES, threshold=0.5)
        result = DiscoveryResult(
            success=True,
            config=DiscoveryConfig(),
            ensemble_dag=dag,
            edges=edges,
            algorithm_results=results,
        )
        # Survivors alone would say 2/2 = 1.0; the voters agreed on 1 of 3 candidates.
        assert result.algorithm_agreement == pytest.approx(1 / 3)


class TestDefaultAlgorithmsHaveOneSource:
    def test_config_default_is_the_shared_constant(self):
        assert DiscoveryConfig().algorithms == list(DEFAULT_DISCOVERY_ALGORITHMS)
        assert DEFAULT_DISCOVERY_ALGORITHM_NAMES == [a.value for a in DEFAULT_DISCOVERY_ALGORITHMS]
        assert (
            DiscoveryConfig().algorithms is not DiscoveryConfig().algorithms
        )  # a fresh list each time

    def test_tool_registry_schema_and_input_model_read_the_constant(self):
        from src.tool_registry.tools.causal_discovery import (
            DiscoverDagInput,
            register_discover_dag_tool,
        )

        assert DiscoverDagInput().algorithms == DEFAULT_DISCOVERY_ALGORITHM_NAMES
        with patch("src.tool_registry.tools.causal_discovery.get_registry") as get_registry:
            registry = MagicMock()
            get_registry.return_value = registry
            register_discover_dag_tool()
        schema = registry.register.call_args[1]["schema"]
        (param,) = [p for p in schema.input_parameters if p.name == "algorithms"]
        assert param.default == DEFAULT_DISCOVERY_ALGORITHM_NAMES

    @pytest.mark.asyncio
    async def test_graph_builder_unguided_default_reads_the_constant(self):
        from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
        from src.agents.causal_impact.state import CausalImpactState

        class _CapturingRunner:
            config: DiscoveryConfig | None = None

            async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
                self.config = config
                return DiscoveryResult(success=False, config=config)

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
            "discovery_guided": False,
        }
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node.execute(cast(CausalImpactState, state))
        assert runner.config is not None
        assert runner.config.algorithms == list(DEFAULT_DISCOVERY_ALGORITHMS)
