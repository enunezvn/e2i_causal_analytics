"""Guided discovery runs the frame pre-flight, under a budget, and asserts the
estimand edge honestly (Lane D items 1-3 wired into ``GraphBuilderNode``).

Spec: docs/superpowers/specs/2026-09-22-real-data-causal-estimation-design.md,
"Lane D — guided discovery on claims frames". Evidence for every default:
docs/demos/results/2026-09-22_lane_d_guided_discovery_claims/README.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.graph_builder import (
    DISCOVERY_MAX_COVARIATES,
    DISCOVERY_MIN_RESAMPLES,
    DISCOVERY_TIME_BUDGET_S,
    GraphBuilderNode,
)
from src.agents.causal_impact.state import CausalImpactState
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveredEdge,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryResult,
)

T = "t"
Y = "y"


def _frame(n: int = 600, seed: int = 0) -> Tuple[pd.DataFrame, List[str]]:
    """Binary T/Y, continuous confounder ``sev``, a prognostic ``prog``, a
    duplicate of ``sev`` (exactly collinear), a constant, and eight noise
    columns — so with cap 4 something is capped and something is pruned."""
    rng = np.random.default_rng(seed)
    sev = rng.normal(size=n)
    prog = rng.normal(size=n)
    t = (0.9 * sev + rng.normal(size=n) > 0).astype(float)
    y = (0.8 * t + 0.8 * sev + 0.6 * prog + rng.normal(size=n) > 0).astype(float)
    frame = pd.DataFrame({T: t, Y: y, "sev": sev, "prog": prog})
    frame["sev_dup"] = sev
    frame["const"] = 1.0
    for i in range(8):
        frame[f"noise_{i}"] = rng.normal(size=n)
    covs = [c for c in frame.columns if c not in (T, Y)]
    return frame, covs


def _state(frame: pd.DataFrame, covs: List[str], **overrides: Any) -> CausalImpactState:
    state: Dict[str, Any] = {
        "query": f"effect of {T} on {Y}",
        "treatment_var": T,
        "outcome_var": Y,
        "confounders": covs,
        "modeled_confounders": covs,
        "anchored_confounders": [],
        "data_cache": {"estimation_data": frame},
        "auto_discover": True,
        "discovery_guided": True,
        "discovery_latent_diagnostic": False,
    }
    state.update(overrides)
    return cast(CausalImpactState, state)


class _CapturingRunner:
    def __init__(self) -> None:
        self.config: DiscoveryConfig | None = None
        self.data: pd.DataFrame | None = None

    async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
        self.config = config
        self.data = data
        return DiscoveryResult(success=False, config=config)


class _AcceptingRunner:
    """Returns an ensemble over the frame it was handed, engineered so the
    real gate ACCEPTs: every edge beyond the prior at stability 1.0. Draws
    ``edges`` (only those whose nodes are in the frame) and never the
    estimand edge unless asked."""

    def __init__(self, edges: List[Tuple[str, str]], draw_estimand: bool) -> None:
        self._edges = edges
        self._draw_estimand = draw_estimand
        self.config: DiscoveryConfig | None = None
        self.data: pd.DataFrame | None = None

    async def discover_dag(self, data, config, session_id=None) -> DiscoveryResult:
        self.config = config
        self.data = data
        nodes = list(data.columns)
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        drawn = [(s, t) for s, t in self._edges if s in nodes and t in nodes]
        if self._draw_estimand:
            drawn.append((T, Y))
        discovered = []
        for s, t in drawn:
            dag.add_edge(s, t)
            discovered.append(
                DiscoveredEdge(
                    source=s,
                    target=t,
                    confidence=1.0,
                    algorithm_votes=1,
                    algorithms=["pc"],
                    bootstrap_stability=1.0,
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
                    adjacency_matrix=np.zeros((len(nodes), len(nodes)), dtype=int),
                    edge_list=drawn,
                    runtime_seconds=0.01,
                    converged=True,
                )
            ],
            metadata={"bootstrap": {"n_resamples": 20, "n_succeeded": 20, "corroborated": True}},
        )


class TestDefaultsAreDerivedNotInvented:
    def test_time_budget_is_the_agent_timeout_minus_the_refutation_budget(self) -> None:
        from src.api.routes.causal._common import (
            _AGENT_HARD_TIMEOUT_S,
            _REFUTATION_COMPUTE_BUDGET_S,
        )

        assert DISCOVERY_TIME_BUDGET_S == _AGENT_HARD_TIMEOUT_S - _REFUTATION_COMPUTE_BUDGET_S

    def test_cap_and_min_resamples_defaults(self) -> None:
        assert DISCOVERY_MAX_COVARIATES == 20
        assert DISCOVERY_MIN_RESAMPLES == 10


class TestPreflightShapesTheLearningFrame:
    @pytest.mark.asyncio
    async def test_learner_sees_only_kept_covariates_plus_estimand(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node._run_discovery(_state(frame, covs, discovery_max_covariates=4), T, Y)
        assert runner.data is not None
        learned = list(runner.data.columns)
        assert T in learned and Y in learned
        assert "const" not in learned
        assert "sev_dup" not in learned
        # The cap is the largest symmetric top-k union that fits (codex r1
        # finding 3), so the learner sees AT MOST the cap, plus T and Y.
        assert 2 < len(learned) <= 4 + 2
        assert "sev" in learned  # strongest T- and Y-association survives the cap
        # column order follows the frame, never re-sorted
        assert learned == [c for c in frame.columns if c in set(learned)]

    @pytest.mark.asyncio
    async def test_tiers_and_budget_follow_the_preflight(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node._run_discovery(_state(frame, covs, discovery_max_covariates=4), T, Y)
        config = runner.config
        assert config is not None and config.prior_knowledge is not None
        tiers = config.prior_knowledge.tiers
        assert tiers is not None
        assert set(tiers[0]) == set(runner.data.columns) - {T, Y}  # type: ignore[union-attr]
        assert tiers[1] == [T] and tiers[2] == [Y]
        assert config.time_budget_s == DISCOVERY_TIME_BUDGET_S
        assert config.min_resamples == DISCOVERY_MIN_RESAMPLES

    @pytest.mark.asyncio
    async def test_state_overrides_budget_and_cap(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node._run_discovery(
            _state(
                frame,
                covs,
                discovery_max_covariates=2,
                discovery_time_budget_s=30.0,
                discovery_min_resamples=5,
            ),
            T,
            Y,
        )
        assert runner.config is not None
        assert runner.config.time_budget_s == 30.0
        assert runner.config.min_resamples == 5
        assert 2 < len(runner.data.columns) <= 2 + 2  # type: ignore[union-attr]
        assert "sev" in runner.data.columns  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_anchored_confounder_survives_the_cap_and_stays_required(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node._run_discovery(
            _state(frame, covs, discovery_max_covariates=2, anchored_confounders=["noise_7"]),
            T,
            Y,
        )
        assert "noise_7" in runner.data.columns  # type: ignore[union-attr]
        required = runner.config.prior_knowledge.required_edges  # type: ignore[union-attr]
        assert ("noise_7", T) in required and ("noise_7", Y) in required

    @pytest.mark.asyncio
    async def test_preflight_decisions_ride_in_the_result_metadata(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=True)  # type: ignore[assignment]
        result, _ = await node._run_discovery(_state(frame, covs, discovery_max_covariates=4), T, Y)
        preflight = result.metadata["preflight"]
        assert preflight["constant"] == ["const"]
        assert preflight["collinear"] == ["sev_dup"]
        assert 0 < len(preflight["kept"]) <= 4
        assert "sev" in preflight["kept"]
        assert set(preflight["capped"]) | set(preflight["kept"]) == set(covs) - {"const", "sev_dup"}
        assert preflight["protected_capped"] == []
        assert preflight["max_covariates"] == 4

    @pytest.mark.asyncio
    async def test_unguided_legacy_path_is_untouched(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        runner = _CapturingRunner()
        node._discovery_runner = runner  # type: ignore[assignment]
        await node._run_discovery(_state(frame, covs, discovery_guided=False), T, Y)
        assert list(runner.data.columns) == list(frame.columns)  # type: ignore[union-attr]
        assert runner.config is not None
        assert runner.config.time_budget_s is None
        assert runner.config.min_resamples is None


class TestRemovedCovariatesStayAdjusted:
    @pytest.mark.asyncio
    async def test_accepted_dag_carries_removed_covariates_and_the_guarantee_unions_them(
        self,
    ) -> None:
        """ACCEPT ships the ensemble DAG, whose nodes are the CAPPED frame's
        columns. The pre-flight-removed covariates must still be nodes of the
        shipped DAG (isolated — no structure was learned for them) so the
        adjustment guarantee unions every declared covariate into the set."""
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=True)  # type: ignore[assignment]
        out = await node.execute(_state(frame, covs, discovery_max_covariates=4))
        graph = out["causal_graph"]
        assert graph["discovery_gate_decision"] == "accept"
        assert out["discovery_result"]["metadata"]["preflight"]["capped"]
        assert set(covs) <= set(graph["nodes"])
        assert graph["adjustment_sets"], graph
        assert set(covs) <= set(graph["adjustment_sets"][0])
        # provenance: only the drawn edges exist; removed covariates are isolates
        provenance = {(e["source"], e["target"]): e["provenance"] for e in graph["edge_provenance"]}
        assert provenance[(T, Y)] == "required_prior"
        assert provenance[("sev", T)] == "discovered"
        assert not any(s == "noise_0" or t == "noise_0" for s, t in provenance)

    @pytest.mark.asyncio
    async def test_warning_names_what_was_pruned_and_capped(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=True)  # type: ignore[assignment]
        out = await node.execute(_state(frame, covs, discovery_max_covariates=4))
        lines = [w for w in out.get("warnings", []) if w.startswith("Discovery pre-flight")]
        assert len(lines) == 1
        line = lines[0]
        assert "const" in line and "sev_dup" in line
        capped = out["discovery_result"]["metadata"]["preflight"]["capped"]
        assert all(name in line for name in capped)
        assert "stays in the adjustment set" in line


class TestRequiredEdgeHonesty:
    @pytest.mark.asyncio
    async def test_missing_estimand_edge_is_reported_and_asserted_on_the_shipped_dag(
        self,
    ) -> None:
        """PC's skeleton phase can remove the required (T, Y) pair (causal-learn
        applies required edges at orientation only; measured on the real frame,
        marginal fisherz p = 0.21). The run must say so, and the shipped DAG
        must still carry T->Y with provenance required_prior."""
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=False)  # type: ignore[assignment]
        out = await node.execute(_state(frame, covs, discovery_max_covariates=4))
        missing = out["discovery_result"]["metadata"]["required_edges_missing"]
        assert [tuple(e) for e in missing] == [(T, Y)]
        assert out["discovery_result"]["metadata"]["required_edges_missing_cause"]
        graph = out["causal_graph"]
        assert graph["discovery_gate_decision"] == "accept"
        assert [T, Y] in [list(e) for e in graph["edges"]]
        provenance = {(e["source"], e["target"]): e["provenance"] for e in graph["edge_provenance"]}
        assert provenance[(T, Y)] == "required_prior"
        estimand_lines = [w for w in out.get("warnings", []) if f"{T} -> {Y}" in w]
        assert len(estimand_lines) == 1
        assert "required_prior" in estimand_lines[0]

    @pytest.mark.asyncio
    async def test_present_estimand_edge_raises_no_warning(self) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=True)  # type: ignore[assignment]
        out = await node.execute(_state(frame, covs, discovery_max_covariates=4))
        assert out["discovery_result"]["metadata"]["required_edges_missing"] == []
        assert not [w for w in out.get("warnings", []) if f"{T} -> {Y}" in w]


class _RejectingRunner(_AcceptingRunner):
    """A converged run that drew nothing: the gate REJECTs (too few edges)."""

    def __init__(self) -> None:
        super().__init__([], draw_estimand=False)


class TestEstimandEdgeOnEveryGatePath:
    """Spec item 3: 'in every case assert the estimand edge on the shipped
    DAG'. The CAPABILITY is the edge's presence on every path; the label
    follows the documented provenance design (``_compute_edge_provenance``:
    ``required_prior`` when the DAG shipped through discovery, ``curated`` when
    the shipped DAG is the manual construction — a manual DAG's edges did not
    come from the prior even where the prior agrees; codex iter-1 HIGH of the
    provenance PR, pinned by test_graph_builder_guarantee.py)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("runner", "expected_decision", "expected_label"),
        [
            (
                _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=False),
                "accept",
                "required_prior",
            ),
            (_RejectingRunner(), "reject", "curated"),
            (_CapturingRunner(), "reject", "curated"),
            # ACCEPT whose ensemble orients outcome->treatment: execute() discards
            # the discovered DAG for the manual one (dag_overridden), so the
            # shipped DAG is manual and its edges are curated.
            (_AcceptingRunner([(Y, T), ("sev", T)], draw_estimand=False), "accept", "curated"),
        ],
        ids=["accept", "reject-no-edges", "discovery-failed", "accept-overridden"],
    )
    async def test_estimand_edge_is_shipped_on_every_path(
        self, runner: Any, expected_decision: str, expected_label: str
    ) -> None:
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = runner
        out = await node.execute(_state(frame, covs, discovery_max_covariates=4))
        graph = out["causal_graph"]
        assert graph["discovery_gate_decision"] == expected_decision
        assert [T, Y] in [list(e) for e in graph["edges"]]
        provenance = {(e["source"], e["target"]): e["provenance"] for e in graph["edge_provenance"]}
        assert provenance[(T, Y)] == expected_label
        assert graph["adjustment_sets"] and set(covs) <= set(graph["adjustment_sets"][0])


class TestRequiredEdgeCauseIsEstablishedNotAssumed:
    """codex r1 finding 6: the cause of a missing required edge must name what
    actually happened — a run that did not converge, an edge PC drew that the
    ensemble's post-processing removed, or the skeleton-phase removal — never
    the skeleton explanation by default."""

    def _config(self) -> DiscoveryConfig:
        from src.causal_engine.discovery.base import CausalPriorKnowledge

        return DiscoveryConfig(
            prior_knowledge=CausalPriorKnowledge(
                tiers=[["sev"], [T], [Y]], required_edges=[(T, Y)], forbidden_edges=[]
            )
        )

    def _result(
        self,
        drawn: List[Tuple[str, str]],
        ensemble: List[Tuple[str, str]],
        converged: bool,
        error: str | None = None,
    ) -> DiscoveryResult:
        dag = nx.DiGraph()
        dag.add_nodes_from(["sev", T, Y])
        dag.add_edges_from(ensemble)
        meta: Dict[str, Any] = {}
        if error:
            meta["error"] = error
        return DiscoveryResult(
            success=converged,
            config=self._config(),
            ensemble_dag=dag if converged else None,
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((3, 3), dtype=int),
                    edge_list=drawn,
                    runtime_seconds=0.0,
                    converged=converged,
                    metadata={"error": error} if error else {},
                )
            ],
            metadata=meta,
        )

    def test_skeleton_cause_only_when_pc_converged_and_did_not_draw_it(self) -> None:
        result = self._result(drawn=[("sev", T)], ensemble=[("sev", T)], converged=True)
        GraphBuilderNode._annotate_required_edges(result, result.config)
        assert result.metadata["required_edges_missing"] == [[T, Y]]
        assert "skeleton" in result.metadata["required_edges_missing_cause"]

    def test_failed_run_names_the_failure_not_the_skeleton(self) -> None:
        result = self._result(drawn=[], ensemble=[], converged=False, error="timeout after 300s")
        GraphBuilderNode._annotate_required_edges(result, result.config)
        cause = result.metadata["required_edges_missing_cause"]
        assert result.metadata["required_edges_missing"] == [[T, Y]]
        assert "skeleton" not in cause
        assert "did not converge" in cause and "timeout after 300s" in cause

    def test_edge_drawn_but_removed_by_post_processing_says_so(self) -> None:
        result = self._result(drawn=[(T, Y), ("sev", T)], ensemble=[("sev", T)], converged=True)
        GraphBuilderNode._annotate_required_edges(result, result.config)
        cause = result.metadata["required_edges_missing_cause"]
        assert "skeleton" not in cause
        assert "post-processing" in cause


class TestIsolatedNodesAreNotBackdoorCandidates:
    """Measured on the real frame (acceptance_runs_final.txt, phase_seconds):
    the backdoor search took 316 s of a 502 s node wall because the 57
    covariates the pre-flight kept away from the learner come back as
    ISOLATED nodes of the shipped DAG (execute() adds them so the adjustment
    guarantee can union them) and the search enumerated every combination of
    up to 3 of all 77 candidates (76,153 criterion checks). An isolated node
    lies on no path, so it can neither block nor open one: no minimal backdoor
    set contains it and Z ∪ {isolated} is admissible iff Z is. The search must
    therefore never enumerate isolated nodes; the guarantee unions them
    afterwards because they are declared."""

    def _dag_with_isolates(self, n_isolated: int) -> Tuple[nx.DiGraph, List[str]]:
        dag = nx.DiGraph()
        dag.add_edges_from(
            [("c1", T), ("c1", Y), ("c2", T), ("c2", Y), ("c3", T), ("c4", Y), (T, Y)]
        )
        isolates = [f"iso_{i}" for i in range(n_isolated)]
        dag.add_nodes_from(isolates)
        return dag, isolates

    def test_isolated_nodes_never_enter_the_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = GraphBuilderNode()
        seen: List[frozenset] = []
        original = GraphBuilderNode._satisfies_backdoor_criterion

        def spy(self_, dag, adjustment_set, treatment, outcome):  # type: ignore[no-untyped-def]
            seen.append(frozenset(adjustment_set))
            return original(self_, dag, adjustment_set, treatment, outcome)

        monkeypatch.setattr(GraphBuilderNode, "_satisfies_backdoor_criterion", spy)
        connected, _ = self._dag_with_isolates(0)
        sets_without = node._find_adjustment_sets(connected, T, Y)
        checks_without = len(seen)
        seen.clear()
        with_isolates, isolates = self._dag_with_isolates(40)
        sets_with = node._find_adjustment_sets(with_isolates, T, Y)
        assert sets_with == sets_without
        assert len(seen) == checks_without  # the isolates added zero checks
        assert not any(set(isolates) & z for z in seen)

    @pytest.mark.asyncio
    async def test_removed_covariates_are_still_unioned_into_the_shipped_adjustment_set(
        self,
    ) -> None:
        """The guarantee, not the search, is what keeps a capped covariate in
        the adjustment set — the search change above must not weaken it."""
        frame, covs = _frame()
        node = GraphBuilderNode()
        node._discovery_runner = _AcceptingRunner([("sev", T), ("sev", Y)], draw_estimand=False)  # type: ignore[assignment]
        out = await node.execute(_state(frame, covs, discovery_max_covariates=3))
        graph = out["causal_graph"]
        assert graph["discovery_gate_decision"] == "accept"
        capped = out["discovery_result"]["metadata"]["preflight"]["capped"]
        assert capped
        assert set(covs) <= set(graph["adjustment_sets"][0])
        assert set(capped) <= set(graph["nodes"])


class TestRealPCRunsOnACollinearFrame:
    @pytest.mark.asyncio
    async def test_singular_frame_now_yields_a_converged_run(self) -> None:
        """Teeth: without the pre-flight this frame (exact duplicate + constant)
        makes fisherz refuse the singular correlation matrix and the run is a
        REJECT 'could not run'. With it, PC converges."""
        frame, covs = _frame(n=500)
        node = GraphBuilderNode()
        from src.causal_engine.discovery import DiscoveryRunner

        node._discovery_runner = DiscoveryRunner(enable_tracing=False)
        result, evaluation = await node._run_discovery(
            _state(frame, covs, discovery_max_covariates=6, discovery_bootstrap_resamples=0),
            T,
            Y,
        )
        assert result.success, result.metadata
        assert result.algorithm_results[0].converged
        assert result.metadata["preflight"]["collinear"] == ["sev_dup"]
        assert result.metadata["preflight"]["constant"] == ["const"]
        assert "could not run" not in " ".join(evaluation["reasons"])
