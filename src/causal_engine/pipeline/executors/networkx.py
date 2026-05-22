"""NetworkX executor — graph analysis stage of the multi-library pipeline.

Wired in phase C-5 of GH #354 per `.claude/plans/354_c5_networkx_design_spike.md`
(Role B — symbolic-input graph analysis).

The executor constructs a real ``networkx.DiGraph`` from ``PipelineState``'s
variable names (treatment, outcome, confounders, effect_modifiers) and any
upstream ``state["causal_graph"]``, then computes:

- Centrality metrics (degree, betweenness, in/out-degree)
- Path analysis (all simple paths treatment->outcome up to cutoff=4,
  shortest path length, has-path flag)
- Structural validation (is_dag, simple_cycles when not a DAG)

Confidence is derived from graph structure rather than the placeholder's
hardcoded 0.8:

- 1.0 when ``is_dag and has_treatment_outcome_path and n_nodes >= 3``
- 0.5 when ``is_dag and (no path OR n_nodes < 3)``
- 0.0 when ``not is_dag`` (cycles present)

The executor inherits patterns from ``src/causal_engine/discovery/``
(e.g. ``nx.shortest_path_length`` and ``nx.betweenness_centrality`` usage in
``driver_ranker.py``; ``nx.is_directed_acyclic_graph`` and ``nx.simple_cycles``
in ``runner.py``'s cycle-removal). It does NOT re-implement causal-learn-driven
structure discovery — that lives in ``discovery/runner.py`` and requires a
``pd.DataFrame``, which ``PipelineState`` does not carry.

R1 contract preservation: ``LibraryExecutor`` ABC (``library``, ``execute``,
``validate_input``) and ``LibraryExecutionResult`` TypedDict shape are unchanged.
"""

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from ..router import CausalLibrary
from ..state import LibraryExecutionResult, PipelineConfig, PipelineState
from .base import LibraryExecutor

logger = logging.getLogger(__name__)

# Maximum simple-path length to enumerate between treatment and outcome.
# Bounded to avoid combinatorial blow-up on dense graphs; mirrors the cutoff=4
# used in `discovery/driver_ranker.py:329` (nx.all_simple_paths).
_MAX_PATH_CUTOFF = 4


class NetworkXExecutor(LibraryExecutor):
    """Executor for NetworkX graph analysis."""

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.NETWORKX

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute NetworkX graph construction and analysis.

        Builds a ``networkx.DiGraph`` from either the upstream
        ``state["causal_graph"]`` (preferred when populated) or from the
        symbolic inputs in state (treatment_var, outcome_var, confounders,
        effect_modifiers), then computes centrality, paths, and
        structural-validation outputs.

        Fail-closed behavior:
        - If ``validate_input`` would have rejected (no treatment_var AND no
          confounders) we still defend-in-depth in execute().
        - If graph construction raises (e.g. malformed inputs), return
          ``success=False`` with the error reason.
        """
        start_time = time.time()
        try:
            treatment = state.get("treatment_var")
            outcome = state.get("outcome_var")
            confounders = self._coerce_str_list(state.get("confounders"))
            effect_modifiers = self._coerce_str_list(state.get("effect_modifiers"))
            upstream_graph = state.get("causal_graph")

            if upstream_graph:
                graph, graph_source = self._build_from_upstream(
                    upstream_graph,
                    treatment=treatment,
                    outcome=outcome,
                    confounders=confounders,
                    effect_modifiers=effect_modifiers,
                )
            else:
                graph, graph_source = self._build_symbolic_graph(
                    treatment=treatment,
                    outcome=outcome,
                    confounders=confounders,
                    effect_modifiers=effect_modifiers,
                )

            # Fail-closed defense-in-depth: an empty graph means no signal
            # could be constructed from the supplied state. The placeholder
            # silently returned success=True with empty stubs; the real wrap
            # fails closed.
            if graph.number_of_nodes() == 0:
                raise ValueError(
                    "NetworkX cannot construct a graph: no treatment_var, "
                    "outcome_var, confounders, or upstream causal_graph supplied"
                )

            analysis = self._analyze_graph(
                graph,
                treatment=treatment,
                outcome=outcome,
                graph_source=graph_source,
            )

            confidence = self._compute_confidence(
                n_nodes=analysis["n_nodes"],
                is_dag=analysis["is_dag"],
                has_path=analysis["has_treatment_outcome_path"],
            )

            warnings: List[str] = []
            if not analysis["is_dag"]:
                warnings.append(
                    "NetworkX graph contains cycles; downstream causal-effect "
                    "estimators may misidentify backdoor paths"
                )

            latency_ms = int((time.time() - start_time) * 1000)
            return LibraryExecutionResult(
                library="networkx",
                success=True,
                latency_ms=latency_ms,
                result=analysis,
                error=None,
                confidence=confidence,
                warnings=warnings,
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"NetworkX execution failed: {e}")
            return LibraryExecutionResult(
                library="networkx",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(e),
                confidence=0.0,
                warnings=[],
            )

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for NetworkX analysis."""
        if not state.get("treatment_var") and not state.get("confounders"):
            return False, "NetworkX requires treatment_var or confounders"
        return True, ""

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        """Coerce a state field expected to be a list-of-strings.

        Empty list / None -> []. Non-iterable or non-string-element raises so
        we fail closed on malformed input (matches placeholder's prior
        exception path for ``confounders=123``).
        """
        if value is None:
            return []
        if isinstance(value, str):
            raise TypeError(
                f"expected list of strings, got bare string: {value!r}"
            )
        if not isinstance(value, Iterable):
            raise TypeError(
                f"expected iterable, got {type(value).__name__}"
            )
        result: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(
                    f"expected list of strings, found {type(item).__name__}: {item!r}"
                )
            result.append(item)
        return result

    def _build_symbolic_graph(
        self,
        *,
        treatment: Optional[str],
        outcome: Optional[str],
        confounders: List[str],
        effect_modifiers: List[str],
    ) -> Tuple[nx.DiGraph, str]:
        """Build a DiGraph from variable names alone (no DataFrame required).

        Edge construction follows the standard backdoor-confounder pattern:
        - Each confounder -> treatment AND confounder -> outcome
        - Each effect_modifier -> outcome (modifier influences outcome but
          not treatment)
        - treatment -> outcome (the hypothesized causal edge)

        Mirrors ``agents/causal_impact/nodes/graph_builder.py``'s
        KNOWN_CAUSAL_RELATIONSHIPS / COMMON_CONFOUNDERS patterns at a
        symbolic level (no domain-knowledge edge enrichment — that's the
        agent's job).
        """
        graph: nx.DiGraph = nx.DiGraph()

        if treatment:
            graph.add_node(treatment)
        if outcome:
            graph.add_node(outcome)
        for c in confounders:
            graph.add_node(c)
        for m in effect_modifiers:
            graph.add_node(m)

        if treatment and outcome:
            graph.add_edge(treatment, outcome)
        for c in confounders:
            if treatment:
                graph.add_edge(c, treatment)
            if outcome:
                graph.add_edge(c, outcome)
        for m in effect_modifiers:
            if outcome:
                graph.add_edge(m, outcome)

        return graph, "symbolic"

    def _build_from_upstream(
        self,
        upstream_graph: Dict[str, Any],
        *,
        treatment: Optional[str],
        outcome: Optional[str],
        confounders: List[str],
        effect_modifiers: List[str],
    ) -> Tuple[nx.DiGraph, str]:
        """Build a DiGraph from upstream ``state["causal_graph"]``.

        Upstream contributors may include a discovery-layer node (running
        causal-learn against real data outside the pipeline) or an
        agent-built DAG. We accept their node/edge lists strictly and add
        any treatment/outcome/confounder/effect-modifier nodes that aren't
        already present.

        Treats the upstream graph as the source of truth for structure;
        does NOT re-derive symbolic edges (avoids edge duplication that
        would create spurious cycles).

        Fail-closed: malformed upstream data raises ``TypeError`` /
        ``ValueError`` so the caller's ``except Exception`` branch returns
        ``success=False`` rather than silently producing a partial graph
        that downstream consumers might trust as complete (codex iter-0
        MEDIUM finding).
        """
        graph: nx.DiGraph = nx.DiGraph()
        upstream_nodes = upstream_graph.get("nodes")
        upstream_edges = upstream_graph.get("edges")

        # Strict validation: an upstream graph that supplies non-list
        # `nodes` or `edges` is malformed; refuse silently-partial outputs.
        if upstream_nodes is not None and not isinstance(upstream_nodes, list):
            raise TypeError(
                "upstream causal_graph['nodes'] must be a list of strings, "
                f"got {type(upstream_nodes).__name__}"
            )
        if upstream_edges is not None and not isinstance(upstream_edges, list):
            raise TypeError(
                "upstream causal_graph['edges'] must be a list of "
                f"{{from, to}} dicts, got {type(upstream_edges).__name__}"
            )

        for i, node in enumerate(upstream_nodes or []):
            if not isinstance(node, str):
                raise TypeError(
                    "upstream causal_graph['nodes'] must contain strings; "
                    f"found {type(node).__name__} at index {i}: {node!r}"
                )
            graph.add_node(node)

        for i, edge in enumerate(upstream_edges or []):
            if not isinstance(edge, dict):
                raise TypeError(
                    "upstream causal_graph['edges'] must contain dicts; "
                    f"found {type(edge).__name__} at index {i}: {edge!r}"
                )
            if "from" not in edge or "to" not in edge:
                raise ValueError(
                    "upstream causal_graph edge must have 'from' and 'to' keys; "
                    f"found {edge!r} at index {i}"
                )
            src = edge["from"]
            tgt = edge["to"]
            if not isinstance(src, str) or not isinstance(tgt, str):
                raise TypeError(
                    "upstream causal_graph edge 'from' and 'to' must be strings; "
                    f"found from={src!r} to={tgt!r} at index {i}"
                )
            graph.add_edge(src, tgt)

        for var in (treatment, outcome):
            if var and var not in graph:
                graph.add_node(var)
        for c in confounders:
            if c not in graph:
                graph.add_node(c)
        for m in effect_modifiers:
            if m not in graph:
                graph.add_node(m)

        return graph, "upstream_state"

    # ------------------------------------------------------------------ #
    # Graph analysis
    # ------------------------------------------------------------------ #

    def _analyze_graph(
        self,
        graph: nx.DiGraph,
        *,
        treatment: Optional[str],
        outcome: Optional[str],
        graph_source: str,
    ) -> Dict[str, Any]:
        """Compute centrality, paths, and structural validation on the graph."""
        nodes = sorted(graph.nodes())
        edges = [{"from": u, "to": v} for u, v in graph.edges()]
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        centrality = self._compute_centrality(graph)
        paths = self._compute_paths(graph, treatment=treatment, outcome=outcome)
        is_dag = nx.is_directed_acyclic_graph(graph)
        has_path = paths["shortest_path_length"] is not None
        cycles: List[List[str]] = []
        if not is_dag:
            try:
                cycles = [list(c) for c in nx.simple_cycles(graph)]
            except Exception as e:
                logger.warning(f"nx.simple_cycles failed: {e}")
                cycles = []

        return {
            # Backward-compatible placeholder keys
            "nodes": nodes,
            "edges": edges,
            "centrality": centrality,
            "paths": paths,
            # New structural fields
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "is_dag": is_dag,
            "has_treatment_outcome_path": has_path,
            "cycles": cycles,
            # Provenance
            "graph_source": graph_source,
            "treatment_var": treatment,
            "outcome_var": outcome,
        }

    def _compute_centrality(
        self, graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """Compute centrality metrics.

        Betweenness is only computed for graphs with >=3 nodes (it is
        identically zero for smaller graphs and ``nx.betweenness_centrality``
        on tiny graphs is wasted work).

        Returns a dict containing four sub-maps. ``degree`` and ``betweenness``
        are ``Dict[str, float]`` (normalized centralities); ``in_degree`` and
        ``out_degree`` are ``Dict[str, int]`` (raw counts). The return is
        typed ``Dict[str, Any]`` to accommodate the mixed value shapes; the
        downstream consumer reads individual sub-maps by key.
        """
        n = graph.number_of_nodes()
        degree: Dict[str, float] = {}
        betweenness: Dict[str, float] = {}
        in_degree: Dict[str, int] = {}
        out_degree: Dict[str, int] = {}

        if n > 0:
            degree = dict(nx.degree_centrality(graph))
            in_degree = {
                node: int(graph.in_degree(node)) for node in graph.nodes()
            }
            out_degree = {
                node: int(graph.out_degree(node)) for node in graph.nodes()
            }

        if n >= 3:
            try:
                betweenness = dict(nx.betweenness_centrality(graph))
            except Exception as e:
                logger.warning(f"nx.betweenness_centrality failed: {e}")
                betweenness = {}

        return {
            "degree": degree,
            "betweenness": betweenness,
            "in_degree": in_degree,
            "out_degree": out_degree,
        }

    def _compute_paths(
        self,
        graph: nx.DiGraph,
        *,
        treatment: Optional[str],
        outcome: Optional[str],
    ) -> Dict[str, Any]:
        """Compute treatment->outcome paths up to bounded cutoff."""
        treatment_to_outcome: List[List[str]] = []
        n_paths = 0
        shortest_path_length: Optional[int] = None

        if treatment and outcome and treatment in graph and outcome in graph:
            try:
                if nx.has_path(graph, treatment, outcome):
                    treatment_to_outcome = [
                        list(p)
                        for p in nx.all_simple_paths(
                            graph,
                            treatment,
                            outcome,
                            cutoff=_MAX_PATH_CUTOFF,
                        )
                    ]
                    n_paths = len(treatment_to_outcome)
                    shortest_path_length = int(
                        nx.shortest_path_length(graph, treatment, outcome)
                    )
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                logger.debug(f"No path treatment->outcome: {e}")

        return {
            "treatment_to_outcome": treatment_to_outcome,
            "n_paths_treatment_to_outcome": n_paths,
            "shortest_path_length": shortest_path_length,
        }

    @staticmethod
    def _compute_confidence(
        *, n_nodes: int, is_dag: bool, has_path: bool
    ) -> float:
        """Derive structural confidence per design spike §2.1.

        - 1.0 when DAG, treatment-outcome path present, >=3 nodes
        - 0.5 when DAG but path missing OR < 3 nodes
        - 0.0 when not a DAG (cycles present)
        """
        if not is_dag:
            return 0.0
        if has_path and n_nodes >= 3:
            return 1.0
        return 0.5
