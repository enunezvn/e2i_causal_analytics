"""Fix 4 — two-channel confounder wiring in GraphBuilderNode.

``modeled_confounders`` is the ADJUSTMENT-GUARANTEE channel: every declared
covariate present in the shipped DAG (and not a descendant of treatment) is
unioned into the final adjustment sets, so a discovery miss can never silently
unadjust a declared covariate. ``anchored_confounders`` is the STRUCTURAL-PRIOR
channel: only these are seeded as required conf->treatment / conf->outcome
edges (see test_graph_builder_priors.py). The shipped ``causal_graph`` reports
per-edge provenance: ``required_prior`` / ``discovered`` / ``curated``.
"""

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
    DiscoveryResult,
    GateDecision,
)
from src.causal_engine.discovery.gate import GateEvaluation


class _AcceptingRunner:
    """Stub runner returning a canned successful result whose ensemble DAG
    mirrors the real runner's contract: every frame column is a node (the real
    ``_build_ensemble`` calls ``add_nodes_from(node_names)``), edges as given."""

    def __init__(self, edges: List[Tuple[str, str]]) -> None:
        self._edges = edges

    async def discover_dag(self, *, data, config, session_id=None) -> DiscoveryResult:
        dag = nx.DiGraph()
        dag.add_nodes_from(data.columns)
        dag.add_edges_from(self._edges)
        return DiscoveryResult(
            success=True,
            config=config,  # carries the prior graph_builder built (provenance source)
            ensemble_dag=dag,
            edges=[
                DiscoveredEdge(
                    source=u,
                    target=v,
                    confidence=0.95,
                    algorithm_votes=1,
                    algorithms=["pc"],
                    bootstrap_stability=0.95,
                )
                for u, v in self._edges
            ],
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((2, 2), dtype=int),
                    edge_list=list(self._edges),
                    runtime_seconds=0.01,
                    converged=True,
                )
            ],
        )


def _accept_gate(node: GraphBuilderNode) -> None:
    node.discovery_gate.evaluate = (  # type: ignore[method-assign]
        lambda result, expected=None: GateEvaluation(
            decision=GateDecision.ACCEPT, confidence=0.9, reasons=[]
        )
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0.0, 1.0] * 20,
            "y": [0.0, 1.0] * 20,
            "c1": [float(i % 3) for i in range(40)],
            "c2": [float(i % 5) for i in range(40)],
            "z": [float(i % 7) for i in range(40)],
        }
    )


def _state(**overrides: Any) -> CausalImpactState:
    state: Dict[str, Any] = {
        "query": "What is the causal effect of t on y?",
        "treatment_var": "t",
        "outcome_var": "y",
        "confounders": ["c1", "c2"],
        "modeled_confounders": ["c1", "c2"],
        "anchored_confounders": [],
        "data_cache": {"estimation_data": _frame()},
        "auto_discover": True,
        "discovery_guided": True,
    }
    state.update(overrides)
    return cast(CausalImpactState, state)


class TestAdjustmentGuaranteeUnion:
    @pytest.mark.asyncio
    async def test_declared_confounders_survive_a_discovery_miss(self) -> None:
        """The core guarantee: discovery ACCEPTs a DAG that carries NO edge for
        the declared covariates — they must still be in every final adjustment
        set (a structural miss must not silently unadjust the estimate)."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("t", "y")])  # type: ignore[assignment]
        _accept_gate(node)

        result = await node.execute(_state())

        assert result.get("status") != "failed", result.get("error_message")
        adjustment_sets = result["causal_graph"]["adjustment_sets"]
        assert adjustment_sets, "no adjustment sets shipped"
        for adj in adjustment_sets:
            assert {"c1", "c2"} <= set(adj), adjustment_sets

    @pytest.mark.asyncio
    async def test_union_extends_a_partial_backdoor_set(self) -> None:
        """Discovery finds ONE declared confounder's edges; the backdoor set from
        the DAG is {c1}. The guarantee unions the missed c2 back in."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner(  # type: ignore[assignment]
            [("t", "y"), ("c1", "t"), ("c1", "y")]
        )
        _accept_gate(node)

        result = await node.execute(_state())

        adjustment_sets = result["causal_graph"]["adjustment_sets"]
        assert adjustment_sets and set(adjustment_sets[0]) == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_union_skips_declared_names_absent_from_the_dag(self) -> None:
        """A declared covariate that is not a node of the shipped DAG (not a
        frame column) cannot be guaranteed into a set the estimator would then
        fail to resolve."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("t", "y")])  # type: ignore[assignment]
        _accept_gate(node)

        result = await node.execute(_state(modeled_confounders=["c1", "ghost_col"]))

        adjustment_sets = result["causal_graph"]["adjustment_sets"]
        assert adjustment_sets and set(adjustment_sets[0]) == {"c1"}

    @pytest.mark.asyncio
    async def test_union_skips_treatment_descendants(self) -> None:
        """A declared covariate the shipped DAG shows as POST-treatment is not a
        backdoor variable; forcing it in would violate the backdoor criterion
        the sets are published under. (Unreachable under guided tiers, which
        forbid treatment->covariate edges; guards non-tiered DAG shapes.)"""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner(  # type: ignore[assignment]
            [("t", "y"), ("t", "z"), ("c1", "t"), ("c1", "y")]
        )
        _accept_gate(node)

        result = await node.execute(_state(modeled_confounders=["c1", "z"]))

        adjustment_sets = result["causal_graph"]["adjustment_sets"]
        assert adjustment_sets
        for adj in adjustment_sets:
            assert "z" not in adj, adjustment_sets

    @pytest.mark.asyncio
    async def test_no_declared_confounders_keeps_validated_empty_backdoor(self) -> None:
        """RCT / exogenous-treatment path: with nothing declared, the validated
        empty backdoor ``[[]]`` must survive untouched (the estimator treats it
        as 'correctly unadjusted', distinct from a missing set)."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("t", "y")])  # type: ignore[assignment]
        _accept_gate(node)

        result = await node.execute(_state(confounders=[], modeled_confounders=[]))

        assert result["causal_graph"]["adjustment_sets"] == [[]]


class TestEdgeProvenance:
    @pytest.mark.asyncio
    async def test_accept_labels_prior_edges_required_and_the_rest_discovered(self) -> None:
        """Anchored c1's edges and the estimand edge come from the REQUIRED
        prior; the z->y edge is the data's contribution."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner(  # type: ignore[assignment]
            [("t", "y"), ("c1", "t"), ("c1", "y"), ("z", "y")]
        )
        _accept_gate(node)

        result = await node.execute(_state(anchored_confounders=["c1"]))

        provenance = {
            (e["source"], e["target"]): e["provenance"]
            for e in result["causal_graph"]["edge_provenance"]
        }
        assert provenance[("t", "y")] == "required_prior"
        assert provenance[("c1", "t")] == "required_prior"
        assert provenance[("c1", "y")] == "required_prior"
        assert provenance[("z", "y")] == "discovered"

    @pytest.mark.asyncio
    async def test_estimand_only_prior_labels_covariate_edges_discovered(self) -> None:
        """The production shape: empty structural channel means ONLY the
        estimand edge is prior-required; every covariate edge the data draws is
        honestly 'discovered'."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner(  # type: ignore[assignment]
            [("t", "y"), ("c1", "t"), ("c1", "y")]
        )
        _accept_gate(node)

        result = await node.execute(_state())

        provenance = {
            (e["source"], e["target"]): e["provenance"]
            for e in result["causal_graph"]["edge_provenance"]
        }
        assert provenance[("t", "y")] == "required_prior"
        assert provenance[("c1", "t")] == "discovered"
        assert provenance[("c1", "y")] == "discovered"

    @pytest.mark.asyncio
    async def test_manual_dag_edges_are_curated_without_discovery(self) -> None:
        result = await GraphBuilderNode().execute(_state(auto_discover=False))

        provenance = {
            (e["source"], e["target"]): e["provenance"]
            for e in result["causal_graph"]["edge_provenance"]
        }
        assert set(provenance) == {(u, v) for u, v in result["causal_graph"]["edges"]}
        assert set(provenance.values()) == {"curated"}

    @pytest.mark.asyncio
    async def test_appended_estimand_edge_is_not_credited_to_the_data(self) -> None:
        """codex iter-1 HIGH: on ACCEPT, graph_builder appends treatment->outcome
        for estimand consistency when discovery omitted it. Without a guided
        prior (unguided caller) that edge is neither prior-required nor
        ensemble-drawn — labeling it 'discovered' would credit the data with an
        edge it never produced. It must read 'curated'."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner(  # type: ignore[assignment]
            [("c1", "t"), ("c1", "y")]  # ensemble NEVER drew t->y
        )
        _accept_gate(node)

        result = await node.execute(_state(discovery_guided=False))

        provenance = {
            (e["source"], e["target"]): e["provenance"]
            for e in result["causal_graph"]["edge_provenance"]
        }
        # Unguided: no prior exists, so nothing may claim required_prior.
        assert ("t", "y") in provenance
        assert provenance[("t", "y")] == "curated"
        assert provenance[("c1", "t")] == "discovered"
        assert provenance[("c1", "y")] == "discovered"

    @pytest.mark.asyncio
    async def test_legacy_curated_readd_on_accept_is_not_credited_to_the_data(self) -> None:
        """codex iter-1 HIGH, second half: a legacy caller (no anchored key)
        keeps the full curated re-add on ACCEPT — confounder edges drawn onto
        the discovered DAG by that re-add are curated assertions, not data."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("t", "y")])  # type: ignore[assignment]
        _accept_gate(node)

        state = _state(discovery_guided=False)
        del state["anchored_confounders"]  # legacy shape
        result = await node.execute(state)

        provenance = {
            (e["source"], e["target"]): e["provenance"]
            for e in result["causal_graph"]["edge_provenance"]
        }
        assert provenance[("t", "y")] == "discovered"  # ensemble drew it
        assert provenance[("c1", "t")] == "curated"  # legacy re-add drew these
        assert provenance[("c1", "y")] == "curated"
        assert provenance[("c2", "t")] == "curated"
        assert provenance[("c2", "y")] == "curated"

    @pytest.mark.asyncio
    async def test_rejected_discovery_ships_all_curated_manual_dag(self) -> None:
        """When the gate withholds discovery, the manual DAG's edges are domain
        assertions — none may claim discovery or prior provenance."""
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("t", "y")])  # type: ignore[assignment]
        node.discovery_gate.evaluate = (  # type: ignore[method-assign]
            lambda result, expected=None: GateEvaluation(
                decision=GateDecision.REJECT, confidence=0.1, reasons=[]
            )
        )

        result = await node.execute(_state(anchored_confounders=["c1"]))

        labels = {e["provenance"] for e in result["causal_graph"]["edge_provenance"]}
        assert labels == {"curated"}
