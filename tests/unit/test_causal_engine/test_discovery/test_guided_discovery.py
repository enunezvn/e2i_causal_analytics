"""Guided causal discovery: domain priors orient otherwise-ambiguous edges.

Observational PC recovers only a Markov equivalence class, so edge orientation
is underdetermined and the algorithm can reverse causal edges (e.g. emit
``Y -> T`` instead of ``T -> Y``, or ``T -> confounder``). A
:class:`CausalPriorKnowledge` of tiers (confounders < treatment < outcome)
anchors what is known so the data still selects the rest. These tests use a
small synthetic confounded frame (no DB) so they run in CI.
"""

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.background_knowledge import build_background_knowledge
from src.causal_engine.discovery.base import (
    CausalPriorKnowledge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)
from src.causal_engine.discovery.runner import DiscoveryRunner


def _confounded_frame(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """C confounds T and Y; T causes Y (binary T/Y, continuous C)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    t = (0.9 * c + rng.normal(size=n) > 0).astype(float)
    y = (0.7 * c + 0.6 * t + rng.normal(size=n) > 0).astype(float)
    return pd.DataFrame({"T": t, "Y": y, "C": c})


def _pc_config(prior: CausalPriorKnowledge | None) -> DiscoveryConfig:
    # PC only: it is the algorithm that consumes the priors (BackgroundKnowledge).
    return DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC], prior_knowledge=prior)


class TestGuidedDiscovery:
    @pytest.mark.asyncio
    async def test_tiers_orient_confounder_and_treatment_correctly(self):
        """Tiers [C < T < Y] guarantee C is a source and Y is a sink, so the
        confounder and treatment->outcome edges orient correctly."""
        df = _confounded_frame()
        prior = CausalPriorKnowledge(tiers=[["C"], ["T"], ["Y"]])
        res = await DiscoveryRunner().discover_dag(df, _pc_config(prior))
        edges = set(res.ensemble_dag.edges())

        # Confounder is causally prior: nothing points INTO C; it points to both.
        assert ("C", "T") in edges, edges
        assert ("C", "Y") in edges, edges
        assert ("T", "C") not in edges
        assert ("Y", "C") not in edges
        # Outcome is a sink: it is the source of no edge.
        assert not any(src == "Y" for src, _ in edges), edges
        # Treatment -> outcome (the unguided run reverses this to Y -> T).
        assert ("T", "Y") in edges, edges

    @pytest.mark.asyncio
    async def test_priors_change_the_result_vs_unguided(self):
        """The priors must actually affect orientation: unguided PC reverses the
        treatment-outcome edge here; the guided run does not."""
        df = _confounded_frame()
        unguided = await DiscoveryRunner().discover_dag(df, _pc_config(None))
        guided = await DiscoveryRunner().discover_dag(
            df, _pc_config(CausalPriorKnowledge(tiers=[["C"], ["T"], ["Y"]]))
        )
        assert set(unguided.ensemble_dag.edges()) != set(guided.ensemble_dag.edges())
        # Guided never points into the confounder tier-0 node.
        assert not any(tgt == "C" for _, tgt in guided.ensemble_dag.edges())

    @pytest.mark.asyncio
    async def test_forbidden_edge_is_removed(self):
        """A forbidden edge must be absent even when the data supports it."""
        df = _confounded_frame()
        prior = CausalPriorKnowledge(
            tiers=[["C"], ["T"], ["Y"]],
            forbidden_edges=[("C", "T")],
        )
        res = await DiscoveryRunner().discover_dag(df, _pc_config(prior))
        edges = set(res.ensemble_dag.edges())
        assert ("C", "T") not in edges, edges
        assert ("T", "C") not in edges, edges


class TestBuildBackgroundKnowledge:
    def test_builds_and_ignores_unknown_names(self):
        prior = CausalPriorKnowledge(
            tiers=[["C"], ["T"], ["Y"]],
            required_edges=[("T", "Y"), ("ghost", "Y")],  # ghost ignored
            forbidden_edges=[("Y", "T")],
        )
        bk = build_background_knowledge(prior, node_names=["T", "Y", "C"])
        assert bk is not None

    def test_empty_prior_is_detected(self):
        assert CausalPriorKnowledge().is_empty()
        assert not CausalPriorKnowledge(tiers=[["a"]]).is_empty()
        assert not CausalPriorKnowledge(required_edges=[("a", "b")]).is_empty()
