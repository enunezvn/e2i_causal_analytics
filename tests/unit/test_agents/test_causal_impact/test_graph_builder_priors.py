"""Guided discovery seeds CausalPriorKnowledge.required_edges from the question's
MODELED confounders (confounder->treatment, confounder->outcome), replacing the
generic KNOWN_CAUSAL_RELATIONSHIPS constants that never match real covariates."""

import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.causal_engine.discovery.base import DiscoveryResult, GateDecision
from src.causal_engine.discovery.gate import GateEvaluation


@pytest.mark.asyncio
async def test_run_discovery_seeds_confounder_edges_from_modeled_confounders(monkeypatch):
    node = GraphBuilderNode()

    df = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0, 1.0, 0.0],
            "persistent_180d": [1.0, 0.0, 1.0, 1.0],
            "disease_severity": [2.0, 1.0, 3.0, 2.0],
            "academic_hcp": [1.0, 0.0, 1.0, 0.0],
            "geographic_region=south": [1.0, 0.0, 0.0, 1.0],
        }
    )

    captured: dict = {}

    async def _fake_discover_dag(*, data, config, session_id):
        captured["config"] = config
        return DiscoveryResult(success=True, config=config, ensemble_dag=None, edges=[])

    monkeypatch.setattr(node.discovery_runner, "discover_dag", _fake_discover_dag)
    monkeypatch.setattr(
        node.discovery_gate,
        "evaluate",
        lambda result, expected: GateEvaluation(
            decision=GateDecision.REVIEW, confidence=0.5, reasons=[]
        ),
    )

    state = {
        "data_cache": {"estimation_data": df},
        "discovery_guided": True,
        "modeled_confounders": ["disease_severity", "academic_hcp", "geographic_region=south"],
    }
    await node._run_discovery(state, "treatment_arm", "persistent_180d")

    prior = captured["config"].prior_knowledge
    assert prior is not None
    edges = set(prior.required_edges or [])
    # The estimand edge is still required.
    assert ("treatment_arm", "persistent_180d") in edges
    # Each modeled confounder -> treatment AND -> outcome is required.
    for conf in ("disease_severity", "academic_hcp", "geographic_region=south"):
        assert (conf, "treatment_arm") in edges
        assert (conf, "persistent_180d") in edges


@pytest.mark.asyncio
async def test_run_discovery_skips_modeled_confounders_absent_from_frame(monkeypatch):
    """A modeled confounder not present as a column is not forced as an edge
    (build_background_knowledge would ignore it; keep required_edges clean)."""
    node = GraphBuilderNode()
    df = pd.DataFrame(
        {
            "treatment_arm": [1.0, 0.0, 1.0],
            "persistent_180d": [1.0, 0.0, 1.0],
            "disease_severity": [2.0, 1.0, 3.0],
        }
    )
    captured: dict = {}

    async def _fake_discover_dag(*, data, config, session_id):
        captured["config"] = config
        return DiscoveryResult(success=True, config=config, ensemble_dag=None, edges=[])

    monkeypatch.setattr(node.discovery_runner, "discover_dag", _fake_discover_dag)
    monkeypatch.setattr(
        node.discovery_gate,
        "evaluate",
        lambda result, expected: GateEvaluation(
            decision=GateDecision.REVIEW, confidence=0.5, reasons=[]
        ),
    )
    state = {
        "data_cache": {"estimation_data": df},
        "discovery_guided": True,
        "modeled_confounders": ["disease_severity", "geographic_region=west"],
    }
    await node._run_discovery(state, "treatment_arm", "persistent_180d")
    edges = set(captured["config"].prior_knowledge.required_edges or [])
    assert ("disease_severity", "treatment_arm") in edges
    assert ("disease_severity", "persistent_180d") in edges
    # The absent confounder is NOT seeded as a required edge.
    assert ("geographic_region=west", "treatment_arm") not in edges
    assert ("geographic_region=west", "persistent_180d") not in edges
