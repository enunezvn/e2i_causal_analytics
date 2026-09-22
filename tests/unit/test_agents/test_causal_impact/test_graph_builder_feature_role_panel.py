"""Lane E item 3(d) — the causal agent reads the feature-role panel.

``anchored_confounders`` = features whose APPROVED structure derives ``confounder``
and that carry no leak verdict; leak-verdict covariates are removed from
``modeled_confounders`` (and ``confounders``) with a NAMED warning in the
response, instead of being adjusted for blind. Approved ``instrument`` features
are routed to the state's ``instruments`` channel, not anchored: graph_builder
forces ``conf -> outcome`` for every anchored confounder, and an instrument must
NOT have that edge (see ``derive_confounder_channels`` for the reasoning).

Items 3(a-c) need Lane B's structural author; the seam they will feed is the
``approved_structure_roles`` state key (feature -> derived role).
"""

from __future__ import annotations

from typing import Any, Dict, cast

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
)
from src.causal_engine.discovery.gate import DiscoveryGateDecision, GateEvaluation
from src.causal_engine.feature_role_panel import (
    ConfounderChannels,
    FeatureRolePanel,
    FeatureRoleRecord,
    derive_confounder_channels,
)


def _record(name: str, *, leak: str | None = None) -> FeatureRoleRecord:
    return FeatureRoleRecord(
        feature=name,
        layer_1={"verdict": "post_index" if leak == "layer_1_post_index" else "pre_index"},
        layer_2={"signal": "no_signal", "edges": [], "mode": "shadow"},
        layer_3={"ran": leak != "layer_1_post_index"},
        layer_4={"fired": False},
        ensemble={"decided_by": "layer_1" if leak else "adversarial"},
        leak_verdict=leak is not None,
        leak_source=leak,
    )


def _panel(**leaks: str | None) -> FeatureRolePanel:
    names = ["c1", "c2", "post_dx", "z"]
    return FeatureRolePanel(
        manifest_source="optum_mart",
        treatment="t",
        outcome="y",
        n_rows=40,
        features=tuple(names),
        records={n: _record(n, leak=leaks.get(n)) for n in names},
        layer_activity={},
        activation_profile={},
        leakage_fdr={},
        promotion_eligibility={},
        built_at="2026-09-22T00:00:00+00:00",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": [0.0, 1.0] * 20,
            "y": [0.0, 1.0] * 20,
            "c1": [float(i % 3) for i in range(40)],
            "c2": [float(i % 5) for i in range(40)],
            "post_dx": [float(i % 2) for i in range(40)],
            "z": [float(i % 7) for i in range(40)],
        }
    )


def _state(**overrides: Any) -> CausalImpactState:
    state: Dict[str, Any] = {
        "query": "What is the causal effect of t on y?",
        "treatment_var": "t",
        "outcome_var": "y",
        "confounders": ["c1", "c2", "post_dx"],
        "modeled_confounders": ["c1", "c2", "post_dx"],
        "anchored_confounders": [],
        "data_cache": {"estimation_data": _frame()},
        "auto_discover": False,
    }
    state.update(overrides)
    return cast(CausalImpactState, state)


class TestDeriveConfounderChannels:
    def test_leak_covariates_leave_the_modeled_set_with_a_named_warning(self) -> None:
        panel = _panel(post_dx="layer_1_post_index")
        ch = derive_confounder_channels(panel, declared_covariates=["c1", "c2", "post_dx"])
        assert isinstance(ch, ConfounderChannels)
        assert ch.modeled_confounders == ["c1", "c2"]
        assert ch.removed == [("post_dx", "layer_1_post_index")]
        assert ch.anchored_confounders is None  # no approved structure → channel untouched
        assert ch.instruments == []
        assert len(ch.warnings) == 1
        assert "post_dx" in ch.warnings[0] and "layer_1_post_index" in ch.warnings[0]
        assert "modeled_confounders" in ch.warnings[0]

    def test_no_leaks_means_no_change_and_no_warning(self) -> None:
        ch = derive_confounder_channels(_panel(), declared_covariates=["c1", "c2"])
        assert ch.modeled_confounders == ["c1", "c2"]
        assert ch.removed == [] and ch.warnings == []

    def test_approved_confounders_anchor_unless_they_carry_a_leak(self) -> None:
        panel = _panel(c2="layer_3_high")
        ch = derive_confounder_channels(
            panel,
            declared_covariates=["c1", "c2", "z"],
            approved_structure_roles={"c1": "confounder", "c2": "confounder", "z": "instrument"},
        )
        assert ch.anchored_confounders == ["c1"]
        assert ch.instruments == ["z"]
        # The approved instrument leaves the modeled set (codex r1 HIGH): it is
        # neither adjusted for nor anchored.
        assert ch.modeled_confounders == ["c1"]
        assert ("c2", "layer_3_high") in ch.removed
        assert ("z", "approved_instrument") in ch.removed
        assert any("c2" in w and "not anchored" in w for w in ch.warnings), ch.warnings

    def test_approved_non_confounder_roles_leave_the_modeled_set(self) -> None:
        """codex r1 HIGH: an approved mediator / collider / descendant is not a
        backdoor variable and must NOT be adjusted for; an approved instrument
        leaves the modeled set too and goes to the ``instruments`` channel."""
        ch = derive_confounder_channels(
            _panel(),
            declared_covariates=["c1", "c2", "post_dx", "z"],
            approved_structure_roles={
                "c1": "confounder",
                "c2": "mediator",
                "post_dx": "collider",
                "z": "instrument",
            },
        )
        assert ch.modeled_confounders == ["c1"]
        assert ch.anchored_confounders == ["c1"]
        assert ch.instruments == ["z"]
        assert list(ch.removed) == [
            ("c2", "approved_mediator"),
            ("post_dx", "approved_collider"),
            ("z", "approved_instrument"),
        ]
        assert any("c2" in w and "mediator" in w for w in ch.warnings), ch.warnings
        assert any("z" in w and "instrument" in w and "not adjusted" in w for w in ch.warnings)

    def test_uncontracted_layer_3_exclusion_is_marked_for_temporal_review(self) -> None:
        """codex r1 HIGH: Layer 3 measures predictiveness of Y, not timing. An
        uncontracted Layer-3-high covariate is excluded per spec 3(b) but the
        warning must say its temporal status is UNKNOWN and needs review — never
        present it as proven leakage."""
        ch = derive_confounder_channels(_panel(c2="layer_3_high"), declared_covariates=["c1", "c2"])
        assert ch.modeled_confounders == ["c1"]
        assert ch.removed == [("c2", "layer_3_high")]
        w = ch.warnings[0]
        assert "c2" in w and "temporal status unknown" in w and "review" in w, w
        assert ch.review_required == ["c2"]
        # A post-index contract IS proven leakage: no review flag.
        ch1 = derive_confounder_channels(
            _panel(post_dx="layer_1_post_index"), declared_covariates=["post_dx"]
        )
        assert ch1.review_required == []

    def test_one_hot_dummies_inherit_their_source_columns_verdict(self) -> None:
        """codex r2: the agent route one-hot expands categoricals into the
        loader's stable ``<col>=<level>`` dummies BEFORE graph_builder runs, so
        a leak verdict on ``post_dx`` must reach ``post_dx=1`` / ``post_dx=2``
        or the expansion silently launders the leak back into the adjustment set."""
        ch = derive_confounder_channels(
            _panel(post_dx="layer_1_post_index"),
            declared_covariates=["c1", "post_dx=1", "post_dx=2", "c2=b"],
        )
        assert ch.modeled_confounders == ["c1", "c2=b"]
        assert ch.removed == [
            ("post_dx=1", "layer_1_post_index"),
            ("post_dx=2", "layer_1_post_index"),
        ]
        assert any("post_dx=1" in w and "post_dx" in w and "dummy" in w for w in ch.warnings), (
            ch.warnings
        )
        # A dummy whose source the panel never saw is still "not in the panel".
        assert not any("c2=b" in w for w in ch.warnings)

    def test_approved_roles_fan_out_to_one_hot_dummies(self) -> None:
        """codex r2: anchoring and role removal must reach the dummies the
        loader derived from an approved source column."""
        ch = derive_confounder_channels(
            _panel(),
            declared_covariates=["c1=a", "c1=b", "c2=x", "z"],
            approved_structure_roles={"c1": "confounder", "c2": "mediator", "z": "instrument"},
        )
        assert ch.anchored_confounders == ["c1=a", "c1=b"]
        assert ch.modeled_confounders == ["c1=a", "c1=b"]
        assert ("c2=x", "approved_mediator") in ch.removed
        assert ch.instruments == ["z"]

    def test_frame_columns_with_a_leak_verdict_are_excluded_even_when_not_declared(self) -> None:
        """codex r4: the denylist is derived against the FRAME, not only the declared
        covariates — a panel-marked leak that the route added to the frame but the
        caller did not declare must still leave discovery and estimation."""
        ch = derive_confounder_channels(
            _panel(post_dx="layer_1_post_index"),
            declared_covariates=["c1"],
            frame_columns=["t", "y", "c1", "c2", "post_dx", "post_dx=1", "extra"],
        )
        assert ch.modeled_confounders == ["c1"]
        assert ch.excluded_frame_columns == ["post_dx", "post_dx=1"]
        assert any("post_dx" in w and "frame" in w for w in ch.warnings), ch.warnings

    def test_approved_non_adjustment_roles_apply_before_the_not_in_panel_branch(self) -> None:
        """codex r4: an approved mediator the panel never saw must still leave the
        modeled set (and the frame) — the approved role is checked first."""
        ch = derive_confounder_channels(
            _panel(),
            declared_covariates=["c1", "extra"],
            approved_structure_roles={"extra": "mediator", "c1": "confounder"},
            frame_columns=["t", "y", "c1", "extra", "extra=b"],
        )
        assert ch.modeled_confounders == ["c1"]
        assert ("extra", "approved_mediator") in ch.removed
        assert ch.excluded_frame_columns == ["extra", "extra=b"]
        assert not any("not in the panel" in w and "extra" in w for w in ch.warnings)

    def test_accepts_the_serialised_panel(self) -> None:
        payload = _panel(post_dx="layer_1_post_index").to_dict()
        ch = derive_confounder_channels(payload, declared_covariates=["c1", "post_dx"])
        assert ch.modeled_confounders == ["c1"]

    def test_a_declared_covariate_absent_from_the_panel_is_kept_and_named(self) -> None:
        """The panel cannot vouch for a column it never saw; keep it (the
        guarantee channel's promise) but say so."""
        ch = derive_confounder_channels(_panel(), declared_covariates=["c1", "unseen"])
        assert ch.modeled_confounders == ["c1", "unseen"]
        assert any("unseen" in w and "not in the panel" in w for w in ch.warnings)


class TestGraphBuilderReadsThePanel:
    @pytest.mark.asyncio
    async def test_leak_covariate_is_not_adjusted_for_and_the_warning_is_named(self) -> None:
        node = GraphBuilderNode()
        result = await node.execute(
            _state(feature_role_panel=_panel(post_dx="layer_1_post_index").to_dict())
        )
        graph = result["causal_graph"]
        for adj in graph["adjustment_sets"]:
            assert "post_dx" not in adj, graph["adjustment_sets"]
        assert "post_dx" not in graph["nodes"]
        assert result["modeled_confounders"] == ["c1", "c2"]
        assert result["confounders"] == ["c1", "c2"]
        assert any("post_dx" in w and "layer_1_post_index" in w for w in result["warnings"])
        # One named provenance line says WHICH panel narrowed the set.
        assert any(
            "feature_role_panel applied" in w and "optum_mart" in w and "t -> y" in w
            for w in result["warnings"]
        ), result["warnings"]
        # The guarantee channel still holds for the survivors.
        assert all({"c1", "c2"} <= set(adj) for adj in graph["adjustment_sets"])

    @pytest.mark.asyncio
    async def test_approved_structure_anchors_confounders(self) -> None:
        node = GraphBuilderNode()
        result = await node.execute(
            _state(
                feature_role_panel=_panel().to_dict(),
                approved_structure_roles={"c1": "confounder", "z": "instrument"},
            )
        )
        assert result["anchored_confounders"] == ["c1"]
        assert result["instruments"] == ["z"]
        assert result["modeled_confounders"] == ["c1", "c2", "post_dx"]  # z was never declared
        assert not [
            w for w in result.get("warnings", []) if "removed from modeled_confounders" in w
        ]

    @pytest.mark.asyncio
    async def test_without_a_panel_nothing_changes(self) -> None:
        node = GraphBuilderNode()
        result = await node.execute(_state())
        graph = result["causal_graph"]
        assert any("post_dx" in adj for adj in graph["adjustment_sets"])
        assert result.get("modeled_confounders") == ["c1", "c2", "post_dx"]
        assert "anchored_confounders" in result and result["anchored_confounders"] == []


@pytest.mark.unit
def test_panel_and_approved_roles_survive_the_langgraph_input_filter() -> None:
    """``StateGraph(CausalImpactState)`` drops undeclared input keys before any
    node runs (wave-51 lesson). Both Lane E channels must be declared."""
    from langgraph.graph import END, StateGraph

    seen: dict = {}

    def probe(state):
        seen["feature_role_panel"] = state.get("feature_role_panel")
        seen["approved_structure_roles"] = state.get("approved_structure_roles")
        seen["sentinel"] = state.get("lane_e_undeclared_sentinel", "FILTERED")
        return {}

    g = StateGraph(CausalImpactState)
    g.add_node("probe", probe)
    g.set_entry_point("probe")
    g.add_edge("probe", END)
    g.compile().invoke(
        {
            "query": "q",
            "query_id": "t1",
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": [],
            "data_source": "synthetic",
            "feature_role_panel": {"features": ["c1"]},
            "approved_structure_roles": {"c1": "confounder"},
            "lane_e_undeclared_sentinel": True,
        }
    )
    assert seen["sentinel"] == "FILTERED"
    assert seen["feature_role_panel"] == {"features": ["c1"]}
    assert seen["approved_structure_roles"] == {"c1": "confounder"}


class _RecordingAcceptingRunner:
    """A runner that RECORDS the frame it was given and returns an ACCEPT-able
    DAG asserting the excluded column as a common cause of t and y — the exact
    shape that would re-adjust for it (codex r3 HIGH)."""

    def __init__(self, extra_edges) -> None:
        self.seen_columns: list = []
        self._edges = [("t", "y"), *extra_edges]

    async def discover_dag(self, *, data, config, session_id=None) -> DiscoveryResult:
        self.seen_columns = list(data.columns)
        dag = nx.DiGraph()
        dag.add_nodes_from(data.columns)
        dag.add_edges_from(
            [(u, v) for u, v in self._edges if u in data.columns and v in data.columns]
        )
        return DiscoveryResult(
            success=True,
            config=config,
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
                for u, v in dag.edges()
            ],
            algorithm_results=[
                AlgorithmResult(
                    algorithm=DiscoveryAlgorithmType.PC,
                    adjacency_matrix=np.zeros((2, 2), dtype=int),
                    edge_list=list(dag.edges()),
                    runtime_seconds=0.01,
                    converged=True,
                )
            ],
        )


def _accept(node: GraphBuilderNode) -> None:
    node.discovery_gate.evaluate = (  # type: ignore[method-assign]
        lambda result, expected=None: GateEvaluation(
            decision=DiscoveryGateDecision.ACCEPT, confidence=0.9, reasons=[]
        )
    )


class TestExcludedColumnsNeverReenterViaDiscovery:
    @pytest.mark.asyncio
    async def test_accept_path_cannot_readjust_for_an_excluded_column(self) -> None:
        node = GraphBuilderNode()
        runner = _RecordingAcceptingRunner([("post_dx", "t"), ("post_dx", "y")])
        node._discovery_runner = runner  # type: ignore[assignment]
        _accept(node)
        result = await node.execute(
            _state(
                auto_discover=True,
                discovery_guided=True,
                feature_role_panel=_panel(post_dx="layer_1_post_index").to_dict(),
            )
        )
        assert "post_dx" not in runner.seen_columns, "discovery must never see an excluded column"
        graph = result["causal_graph"]
        assert "post_dx" not in graph["nodes"]
        assert all("post_dx" not in adj for adj in graph["adjustment_sets"]), graph[
            "adjustment_sets"
        ]
        # The frame every downstream node (estimation's fallback included) reads
        # no longer carries the column.
        assert "post_dx" not in result["data_cache"]["estimation_data"].columns
        assert result["modeled_confounders"] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_dummies_of_an_excluded_source_leave_the_frame_too(self) -> None:
        node = GraphBuilderNode()
        runner = _RecordingAcceptingRunner([("post_dx=1", "t"), ("post_dx=1", "y")])
        node._discovery_runner = runner  # type: ignore[assignment]
        _accept(node)
        frame = _frame().rename(columns={"post_dx": "post_dx=1"})
        result = await node.execute(
            _state(
                confounders=["c1", "c2", "post_dx=1"],
                modeled_confounders=["c1", "c2", "post_dx=1"],
                data_cache={"estimation_data": frame},
                auto_discover=True,
                discovery_guided=True,
                feature_role_panel=_panel(post_dx="layer_1_post_index").to_dict(),
            )
        )
        assert "post_dx=1" not in runner.seen_columns
        assert all("post_dx=1" not in adj for adj in result["causal_graph"]["adjustment_sets"])
        assert "post_dx=1" not in result["data_cache"]["estimation_data"].columns

    @pytest.mark.asyncio
    async def test_the_provenance_line_says_precisely_what_was_checked(self) -> None:
        node = GraphBuilderNode()
        result = await node.execute(_state(feature_role_panel=_panel().to_dict()))
        line = next(w for w in result["warnings"] if w.startswith("feature_role_panel applied"))
        assert "caller-supplied" in line and "provenance not verified" in line
        assert "vetted" not in line and "identity checked" not in line
        # codex r4: say exactly what submit established, no more.
        for phrase in (
            "registered manifest",
            "exact treatment/outcome",
            "at least one covariate overlap",
            "structural invariants",
            "dataset-manifest binding only when the dataset declares one",
        ):
            assert phrase in line, (phrase, line)

    @pytest.mark.asyncio
    async def test_an_undeclared_frame_column_with_a_leak_verdict_leaves_the_frame(self) -> None:
        node = GraphBuilderNode()
        runner = _RecordingAcceptingRunner([("post_dx", "t"), ("post_dx", "y")])
        node._discovery_runner = runner  # type: ignore[assignment]
        _accept(node)
        result = await node.execute(
            _state(
                confounders=["c1", "c2"],
                modeled_confounders=["c1", "c2"],
                auto_discover=True,
                discovery_guided=True,
                feature_role_panel=_panel(post_dx="layer_1_post_index").to_dict(),
            )
        )
        assert "post_dx" not in runner.seen_columns
        assert "post_dx" not in result["data_cache"]["estimation_data"].columns
        assert all("post_dx" not in adj for adj in result["causal_graph"]["adjustment_sets"])
