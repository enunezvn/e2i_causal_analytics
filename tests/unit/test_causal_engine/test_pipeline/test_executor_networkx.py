"""Tests for the real-wired NetworkXExecutor (phase C-5 of GH #354).

Asserts the Role B specification documented in
`.claude/plans/354_c5_networkx_design_spike.md`:

- Uses real `networkx.DiGraph` (not hand-rolled dict)
- Builds backdoor confounder edges (confounder->treatment, confounder->outcome)
- Computes real centrality (degree, betweenness, in/out-degree)
- Computes real paths (all_simple_paths up to cutoff=4, shortest_path_length)
- Validates acyclicity (is_directed_acyclic_graph, simple_cycles when cyclic)
- Derives `confidence` from graph structure (NOT hardcoded 0.8)
- Inherits upstream `state["causal_graph"]` when present
- Fail-closed when no signal possible (no treatment AND no outcome AND no confounders)

These tests are RED against the placeholder body in
`src/causal_engine/pipeline/executors/networkx.py` and will turn GREEN
after the green-phase wrap lands.
"""

from unittest.mock import patch

import networkx as nx
import pytest

from src.causal_engine.pipeline.executors.networkx import NetworkXExecutor
from src.causal_engine.pipeline.router import CausalLibrary
from src.causal_engine.pipeline.state import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
)

# =============================================================================
# Fixtures
# =============================================================================


def _minimal_config() -> PipelineConfig:
    return {
        "mode": "sequential",
        "libraries_enabled": ["networkx"],
        "primary_library": "networkx",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": False,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }


def _state(
    *,
    treatment_var: str | None = "marketing_spend",
    outcome_var: str | None = "sales",
    confounders: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
    causal_graph: dict | None = None,
) -> PipelineState:
    return PipelineState(
        query="Does marketing spend cause sales?",
        question_type="causal_relationship",
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        confounders=confounders,
        effect_modifiers=effect_modifiers,
        data_source="test_data",
        filters=None,
        config=_minimal_config(),
        routed_libraries=["networkx"],
        routing_confidence=0.9,
        routing_rationale="Test",
        networkx_result=None,
        causal_graph=causal_graph,
        graph_metrics=None,
        dowhy_result=None,
        causal_effect=None,
        refutation_results=None,
        identification_method=None,
        econml_result=None,
        cate_by_segment=None,
        overall_ate=None,
        heterogeneity_score=None,
        causalml_result=None,
        uplift_scores=None,
        auuc=None,
        qini=None,
        targeting_recommendations=None,
        consensus_effect=None,
        consensus_confidence=None,
        library_agreement=None,
        nested_cate=None,
        segment_confidence_intervals=None,
        executive_summary=None,
        key_insights=None,
        recommended_actions=None,
        current_stage=PipelineStage.PENDING,
        stage_latencies={},
        total_latency_ms=0,
        libraries_executed=[],
        libraries_skipped=[],
        errors=[],
        warnings=[],
        status="pending",
    )


# =============================================================================
# Result-shape contract (new keys per design spike §2.1)
# =============================================================================


class TestNetworkXExecutorResultShape:
    """Asserts the LibraryExecutionResult["result"] payload conforms to §2.1."""

    @pytest.mark.asyncio
    async def test_result_contains_n_nodes_and_n_edges(self):
        """`result.result` exposes `n_nodes` and `n_edges` integers."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region", "season"])

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is True
        assert isinstance(result["result"]["n_nodes"], int)
        assert isinstance(result["result"]["n_edges"], int)
        # With treatment + outcome + 2 confounders we expect at least 4 nodes
        assert result["result"]["n_nodes"] >= 4

    @pytest.mark.asyncio
    async def test_result_centrality_has_real_metrics(self):
        """`centrality` has degree, betweenness, in_degree, out_degree submaps."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region", "season", "market_size"])

        result = await executor.execute(state, _minimal_config())

        centrality = result["result"]["centrality"]
        assert "degree" in centrality
        assert "betweenness" in centrality
        assert "in_degree" in centrality
        assert "out_degree" in centrality
        # Degree centrality must be populated (non-empty, real floats from networkx)
        assert len(centrality["degree"]) > 0, "degree centrality must not be empty stub"
        # All centrality values must be real numerics, NOT the hardcoded empty {} of the placeholder
        for node, val in centrality["degree"].items():
            assert isinstance(val, (int, float))

    @pytest.mark.asyncio
    async def test_result_paths_has_treatment_to_outcome(self):
        """`paths.treatment_to_outcome` is a list of simple paths and shortest_path_length is set."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region"])

        result = await executor.execute(state, _minimal_config())

        paths = result["result"]["paths"]
        assert "treatment_to_outcome" in paths
        assert "n_paths_treatment_to_outcome" in paths
        assert "shortest_path_length" in paths
        # With treatment->outcome direct edge, there is at least 1 path of length 1
        assert isinstance(paths["treatment_to_outcome"], list)
        assert paths["n_paths_treatment_to_outcome"] >= 1
        assert paths["shortest_path_length"] == 1

    @pytest.mark.asyncio
    async def test_result_includes_dag_validation_flags(self):
        """`is_dag`, `has_treatment_outcome_path`, and `cycles` are reported."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region"])

        result = await executor.execute(state, _minimal_config())

        assert "is_dag" in result["result"]
        assert "has_treatment_outcome_path" in result["result"]
        assert "cycles" in result["result"]
        # A backdoor-pattern graph from confounders is acyclic
        assert result["result"]["is_dag"] is True
        assert result["result"]["has_treatment_outcome_path"] is True
        # cycles must be an empty list (placeholder didn't expose this)
        assert result["result"]["cycles"] == []

    @pytest.mark.asyncio
    async def test_result_records_graph_source(self):
        """`graph_source` is either 'symbolic' (built from state vars) or 'upstream_state'."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region"])

        result = await executor.execute(state, _minimal_config())

        assert result["result"]["graph_source"] in {"symbolic", "upstream_state"}

    @pytest.mark.asyncio
    async def test_result_preserves_backward_compatible_keys(self):
        """Backward-compatible placeholder keys `nodes`, `edges`, `centrality`, `paths` still present."""
        executor = NetworkXExecutor()
        state = _state()

        result = await executor.execute(state, _minimal_config())

        # All keys that downstream consumers (DoWhyValidator, sequential summary) read
        for key in ("nodes", "edges", "centrality", "paths"):
            assert key in result["result"], f"backward-compat key {key!r} missing"


# =============================================================================
# Backdoor confounder edge construction
# =============================================================================


class TestNetworkXBackdoorEdges:
    """Confounders must produce confounder->treatment AND confounder->outcome edges."""

    @pytest.mark.asyncio
    async def test_confounders_create_backdoor_edges_to_treatment(self):
        """Each confounder has an outgoing edge to treatment_var."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", confounders=["C1", "C2"])

        result = await executor.execute(state, _minimal_config())

        edges = result["result"]["edges"]
        edge_set = {(e["from"], e["to"]) for e in edges}
        assert ("C1", "T") in edge_set
        assert ("C2", "T") in edge_set

    @pytest.mark.asyncio
    async def test_confounders_create_backdoor_edges_to_outcome(self):
        """Each confounder has an outgoing edge to outcome_var (backdoor pattern)."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", confounders=["C1", "C2"])

        result = await executor.execute(state, _minimal_config())

        edges = result["result"]["edges"]
        edge_set = {(e["from"], e["to"]) for e in edges}
        assert ("C1", "Y") in edge_set
        assert ("C2", "Y") in edge_set

    @pytest.mark.asyncio
    async def test_treatment_to_outcome_edge_still_present(self):
        """The treatment->outcome edge is still present (backward-compat with placeholder)."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", confounders=["C1"])

        result = await executor.execute(state, _minimal_config())

        edge_set = {(e["from"], e["to"]) for e in result["result"]["edges"]}
        assert ("T", "Y") in edge_set

    @pytest.mark.asyncio
    async def test_effect_modifiers_create_edges_to_outcome(self):
        """Effect modifiers M_i create M_i->outcome edges (modifier-to-outcome pattern)."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", effect_modifiers=["M1"])

        result = await executor.execute(state, _minimal_config())

        edge_set = {(e["from"], e["to"]) for e in result["result"]["edges"]}
        assert ("M1", "Y") in edge_set


# =============================================================================
# Confidence derived from structure (NOT hardcoded 0.8)
# =============================================================================


class TestNetworkXDerivedConfidence:
    """Confidence is computed from graph properties, not a constant."""

    @pytest.mark.asyncio
    async def test_confidence_high_when_dag_with_treatment_outcome_path_and_n_nodes_ge_3(self):
        """A well-formed DAG with treatment->outcome path and >=3 nodes gets confidence=1.0."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", confounders=["C1", "C2"])

        result = await executor.execute(state, _minimal_config())

        # 4 nodes, DAG with treatment->outcome path => confidence = 1.0
        assert result["result"]["n_nodes"] >= 3
        assert result["result"]["is_dag"] is True
        assert result["result"]["has_treatment_outcome_path"] is True
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_confidence_medium_when_dag_but_no_path_or_too_few_nodes(self):
        """A valid DAG with no treatment->outcome path OR fewer than 3 nodes gets confidence=0.5."""
        executor = NetworkXExecutor()
        # Only confounders, no treatment/outcome => no path possible
        state = _state(treatment_var=None, outcome_var=None, confounders=["C1", "C2"])

        result = await executor.execute(state, _minimal_config())

        assert result["result"]["is_dag"] is True
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_confidence_is_not_hardcoded_constant(self):
        """The placeholder returned 0.8 unconditionally; new wrap returns structure-derived values."""
        executor = NetworkXExecutor()
        # Build two different graph shapes; confidence values must NOT both equal 0.8
        s1 = _state(treatment_var="T", outcome_var="Y", confounders=["C1", "C2"])
        s2 = _state(treatment_var=None, outcome_var=None, confounders=["C1"])

        r1 = await executor.execute(s1, _minimal_config())
        r2 = await executor.execute(s2, _minimal_config())

        # At least one must differ from the placeholder's 0.8 constant
        assert not (r1["confidence"] == 0.8 and r2["confidence"] == 0.8)


# =============================================================================
# Upstream state.causal_graph inheritance
# =============================================================================


class TestNetworkXUpstreamGraphInheritance:
    """If state['causal_graph'] is populated by upstream, build from that DAG."""

    @pytest.mark.asyncio
    async def test_upstream_causal_graph_seeds_construction(self):
        """When state['causal_graph'] is set, executor uses its edges as the base graph."""
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["A", "B", "C", "T", "Y"],
            "edges": [
                {"from": "A", "to": "T"},
                {"from": "B", "to": "T"},
                {"from": "C", "to": "Y"},
                {"from": "T", "to": "Y"},
            ],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)

        result = await executor.execute(state, _minimal_config())

        edge_set = {(e["from"], e["to"]) for e in result["result"]["edges"]}
        # All upstream edges must be present
        assert ("A", "T") in edge_set
        assert ("B", "T") in edge_set
        assert ("C", "Y") in edge_set
        assert ("T", "Y") in edge_set
        assert result["result"]["graph_source"] == "upstream_state"

    @pytest.mark.asyncio
    async def test_cyclic_upstream_graph_yields_confidence_zero_with_cycles_and_warning(self):
        """Cyclic upstream causal_graph: is_dag=False, cycles populated, confidence=0.0, warning emitted.

        Addresses codex iter-0 LOW finding: the executor calls nx.simple_cycles,
        but no regression test exercised the `not is_dag => cycles populated`
        contract from the design spike.
        """
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", "Y", "Z"],
            "edges": [
                {"from": "T", "to": "Y"},
                {"from": "Y", "to": "Z"},
                {"from": "Z", "to": "T"},  # closes the cycle T -> Y -> Z -> T
            ],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is True
        assert result["result"]["is_dag"] is False
        # Per spike §2.1: cycles list is populated when not a DAG
        assert len(result["result"]["cycles"]) >= 1
        # Confidence MUST be 0.0 when not a DAG (no half-measures)
        assert result["confidence"] == 0.0
        # Warning must surface the cyclicity so downstream sees it
        assert any("cycle" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_malformed_upstream_node_fails_closed(self):
        """Non-string node in upstream causal_graph raises (defense-in-depth).

        Addresses codex iter-0 MEDIUM finding: the previous code silently
        filtered non-string nodes and could return success=True with a
        partial graph downstream consumers might trust.
        """
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", 42, "Y"],  # non-string node
            "edges": [{"from": "T", "to": "Y"}],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_malformed_upstream_edge_fails_closed(self):
        """Edge missing 'from'/'to' or with non-string endpoints raises.

        Companion to the node test: covers the second silent-filter branch
        flagged by codex iter-0 MEDIUM.
        """
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", "Y"],
            "edges": [{"src": "T", "dst": "Y"}],  # wrong keys
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_malformed_upstream_nodes_not_a_list_fails_closed(self):
        """Upstream `nodes` of wrong type (e.g. dict) raises rather than silently empty."""
        executor = NetworkXExecutor()
        upstream = {
            "nodes": {"a": "T", "b": "Y"},  # dict instead of list
            "edges": [{"from": "T", "to": "Y"}],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_malformed_causal_graph_falsey_non_dict_fails_closed(self):
        """`causal_graph=[]` (falsey but wrong type) fails closed, not falls back to symbolic.

        Addresses codex iter-2 MEDIUM: previous truthiness check
        ``if upstream_graph:`` let falsey-but-malformed values like ``[]``
        bypass upstream validation and silently fall back to symbolic mode
        with success=True. ``PipelineState`` types causal_graph as
        ``Optional[Dict[str, Any]]``, so non-dict (and non-None) is a
        malformed input and must fail closed.
        """
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=[])  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_malformed_causal_graph_truthy_non_dict_fails_closed(self):
        """`causal_graph=42` (truthy non-dict) also fails closed (companion to iter-2 fix)."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=42)  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0


# =============================================================================
# Real networkx.DiGraph used (not hand-rolled dict)
# =============================================================================


class TestNetworkXUsesRealLibrary:
    """The executor must call into real `networkx` APIs."""

    @pytest.mark.asyncio
    async def test_executor_constructs_real_nx_digraph(self):
        """At least one `nx.DiGraph` is constructed during execution."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region"])

        with patch(
            "src.causal_engine.pipeline.executors.networkx.nx.DiGraph",
            wraps=nx.DiGraph,
        ) as digraph_mock:
            await executor.execute(state, _minimal_config())

        assert digraph_mock.call_count >= 1, "Real nx.DiGraph must be constructed"

    @pytest.mark.asyncio
    async def test_executor_uses_betweenness_centrality(self):
        """`nx.betweenness_centrality` is called (real library, not hand-coded values)."""
        executor = NetworkXExecutor()
        state = _state(confounders=["region", "season"])

        with patch(
            "src.causal_engine.pipeline.executors.networkx.nx.betweenness_centrality",
            wraps=nx.betweenness_centrality,
        ) as bc_mock:
            await executor.execute(state, _minimal_config())

        # With >=3 nodes betweenness should be computed at least once
        assert bc_mock.call_count >= 1


# =============================================================================
# Fail-closed semantics (no signal possible)
# =============================================================================


class TestNetworkXFailClosed:
    """When no graph signal is possible, executor must fail-closed, NOT default to zero stub."""

    @pytest.mark.asyncio
    async def test_no_treatment_outcome_or_confounders_returns_failure(self):
        """All three missing => success=False with error message; do not return zero stub."""
        executor = NetworkXExecutor()
        # Note: validate_input would reject this, but execute() must ALSO fail-closed
        # if reached directly (defense in depth)
        state = _state(treatment_var=None, outcome_var=None, confounders=None)

        result = await executor.execute(state, _minimal_config())

        # Must NOT return success=True with empty placeholder values
        assert result["success"] is False
        assert result["error"] is not None
        # Confidence must be 0.0 (no signal); not the hardcoded 0.8
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_confounders_type_returns_failure(self):
        """If confounders is malformed (not iterable), executor fails closed."""
        executor = NetworkXExecutor()
        state = _state(confounders=123)  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_dict_shaped_confounders_fails_closed(self):
        """Dict for `confounders` raises rather than silently extracting keys.

        Addresses codex iter-1 MEDIUM: a dict like ``{"region": "abc"}`` is
        iterable and yields string keys. The previous loose form of
        `_coerce_str_list` would have happily turned that into
        ``["region"]`` and produced a partial-but-success graph.
        ``PipelineState`` types `confounders` as ``Optional[List[str]]`` —
        we must reject non-list containers strictly.
        """
        executor = NetworkXExecutor()
        # Dict whose keys happen to be strings — would have been
        # silently accepted as confounders=["region", "season"].
        state = _state(confounders={"region": "abc", "season": "def"})  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_tuple_shaped_effect_modifiers_fails_closed(self):
        """Tuple for `effect_modifiers` raises rather than being silently coerced."""
        executor = NetworkXExecutor()
        # Tuple is iterable but PipelineState promises a list.
        state = _state(effect_modifiers=("M1", "M2"))  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_non_string_treatment_var_fails_closed(self):
        """`treatment_var=42` raises rather than being silently passed to nx.add_node().

        Addresses codex iter-3 MEDIUM: networkx accepts any hashable node,
        so treatment_var=42 would silently build an integer-node graph and
        return success=True. PipelineState types it as Optional[str], so
        non-string non-None must fail closed.
        """
        executor = NetworkXExecutor()
        state = _state(treatment_var=42, outcome_var="Y", confounders=["C1"])  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_non_string_outcome_var_fails_closed(self):
        """`outcome_var=[1,2]` raises rather than being silently passed to nx.add_node()."""
        executor = NetworkXExecutor()
        state = _state(treatment_var="T", outcome_var=[1, 2], confounders=["C1"])  # type: ignore[arg-type]

        result = await executor.execute(state, _minimal_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert result["confidence"] == 0.0


# =============================================================================
# Existing contract preservation (R1 — ABC, library property, validate_input)
# =============================================================================


class TestNetworkXContractPreserved:
    """The C-1 LibraryExecutor contract is preserved (R1)."""

    def test_library_property_returns_networkx(self):
        executor = NetworkXExecutor()
        assert executor.library == CausalLibrary.NETWORKX

    def test_validate_input_passes_with_treatment_var(self):
        executor = NetworkXExecutor()
        state = _state()
        is_valid, error = executor.validate_input(state)
        assert is_valid is True
        assert error == ""

    def test_validate_input_passes_with_confounders_only(self):
        executor = NetworkXExecutor()
        state = _state(treatment_var=None, outcome_var=None, confounders=["region"])
        is_valid, error = executor.validate_input(state)
        assert is_valid is True
        assert error == ""

    def test_validate_input_fails_without_signal(self):
        executor = NetworkXExecutor()
        state = _state(treatment_var=None, outcome_var=None, confounders=None)
        is_valid, error = executor.validate_input(state)
        assert is_valid is False
        assert "treatment_var or confounders" in error


class TestPreciseCyclicIdentifiabilityGate:
    """M-fo2 (precise): a directed cycle only breaks backdoor identification of the
    (treatment, outcome) estimand when it sits on the ancestral subgraph
    ``An({T,Y}) ∪ {T,Y}`` (the subgraph that determines d-separation between T and
    Y). A cycle elsewhere leaves the estimand identifiable, so the structural
    confidence must NOT be penalized. Reciprocal 2-cycles (unoriented CPDAG/PAG
    edges) are flagged distinctly from genuine feedback loops.

    Replaces the blunt ``not is_dag => confidence 0.0`` contract: the existing
    ``test_cyclic_upstream_graph_yields_confidence_zero...`` (cycle T->Y->Z->T) still
    holds because that cycle is ON the relevant subgraph.
    """

    @pytest.mark.asyncio
    async def test_cycle_off_relevant_subgraph_keeps_full_confidence(self):
        """T->Y is clean; a disjoint A->B->C->A cycle does not touch An({T,Y}),
        so the estimand stays identifiable: cycle_affects_identification False,
        confidence 1.0 (path + >=3 nodes), no relevant cycles."""
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", "Y", "A", "B", "C"],
            "edges": [
                {"from": "T", "to": "Y"},
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "C", "to": "A"},  # disjoint 3-cycle, unrelated to T/Y
            ],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)
        result = await executor.execute(state, _minimal_config())
        res = result["result"]
        assert res["is_dag"] is False
        assert res["cycle_affects_identification"] is False
        assert res["cycles_on_relevant_subgraph"] == []
        assert res["orientation_ambiguity_only"] is False
        # Estimand is identifiable -> NO structural penalty.
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_cycle_on_ancestral_subgraph_blocks_identification(self):
        """A feedback loop among confounders that are ancestors of T (on a backdoor
        path) DOES break identification: cycle_affects_identification True,
        confidence 0.0."""
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", "Y", "A", "B", "C"],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "C", "to": "A"},  # 3-cycle...
                {"from": "C", "to": "T"},  # ...feeding T => A,B,C are ancestors of T
                {"from": "A", "to": "Y"},  # and Y => a backdoor path through the cycle
                {"from": "T", "to": "Y"},
            ],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)
        result = await executor.execute(state, _minimal_config())
        res = result["result"]
        assert res["is_dag"] is False
        assert res["cycle_affects_identification"] is True
        assert len(res["cycles_on_relevant_subgraph"]) >= 1
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_reciprocal_edge_off_path_flagged_orientation_ambiguity(self):
        """A reciprocal A<->B (an unoriented edge) disjoint from {T,Y} is an
        orientation ambiguity, not a feedback loop, and does not block this estimand."""
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", "Y", "A", "B"],
            "edges": [
                {"from": "T", "to": "Y"},
                {"from": "A", "to": "B"},
                {"from": "B", "to": "A"},  # reciprocal pair
            ],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)
        result = await executor.execute(state, _minimal_config())
        res = result["result"]
        assert res["is_dag"] is False
        assert res["orientation_ambiguity_only"] is True
        assert res["cycle_affects_identification"] is False
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_reciprocal_treatment_outcome_blocks_identification(self):
        """T<->Y (both directions) cannot be oriented, so identification is undefined:
        cycle_affects_identification True even though it is a reciprocal pair."""
        executor = NetworkXExecutor()
        upstream = {
            "nodes": ["T", "Y"],
            "edges": [
                {"from": "T", "to": "Y"},
                {"from": "Y", "to": "T"},  # reciprocal T<->Y
            ],
        }
        state = _state(treatment_var="T", outcome_var="Y", causal_graph=upstream)
        result = await executor.execute(state, _minimal_config())
        res = result["result"]
        assert res["is_dag"] is False
        assert res["cycle_affects_identification"] is True
        assert res["orientation_ambiguity_only"] is True
        assert result["confidence"] == 0.0
