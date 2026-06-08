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
from typing import Any, Dict, List, Optional, Tuple

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
            treatment = self._coerce_optional_str(state.get("treatment_var"), "treatment_var")
            outcome = self._coerce_optional_str(state.get("outcome_var"), "outcome_var")
            confounders = self._coerce_str_list(state.get("confounders"))
            effect_modifiers = self._coerce_str_list(state.get("effect_modifiers"))
            upstream_graph = state.get("causal_graph")

            # Strict type check on upstream_graph BEFORE the truthiness
            # branch (codex iter-2 MEDIUM): a malformed-but-falsey value
            # like `[]` or `0` previously bypassed `_build_from_upstream`
            # entirely and fell back to symbolic mode silently. The
            # TypedDict types `causal_graph` as
            # ``Optional[Dict[str, Any]]``, so anything other than dict or
            # None is a malformed input and must fail closed.
            if upstream_graph is not None and not isinstance(upstream_graph, dict):
                raise TypeError(
                    "state['causal_graph'] must be a dict or None, "
                    f"got {type(upstream_graph).__name__}: {upstream_graph!r}"
                )

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

            # M-fo2 (precise): a cycle only blocks identification when it lands on
            # the (T,Y) ancestral subgraph. Off-subgraph cycles leave the estimand
            # identifiable, so they incur no structural penalty.
            identification_blocked = (not analysis["is_dag"]) and analysis[
                "cycle_affects_identification"
            ]
            confidence = self._compute_confidence(
                n_nodes=analysis["n_nodes"],
                has_path=analysis["has_treatment_outcome_path"],
                identification_blocked=identification_blocked,
            )

            warnings: List[str] = []
            if not analysis["is_dag"]:
                if analysis["cycle_affects_identification"]:
                    # Keep the historical wording (downstream "cycle" matchers rely
                    # on it) — this is the identification-breaking case.
                    warnings.append(
                        "NetworkX graph contains cycles; downstream causal-effect "
                        "estimators may misidentify backdoor paths"
                    )
                    if analysis["orientation_ambiguity_only"]:
                        warnings.append(
                            "The cycle(s) are reciprocal 2-cycles (unoriented edges) "
                            "on the treatment-outcome ancestral subgraph; orient them "
                            "before estimating."
                        )
                else:
                    warnings.append(
                        "NetworkX graph contains a cycle OUTSIDE the "
                        "treatment-outcome ancestral subgraph; this estimand remains "
                        "identifiable (no structural penalty applied)."
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
    def _coerce_optional_str(value: Any, field_name: str) -> Optional[str]:
        """Coerce a state field expected to be ``Optional[str]``.

        ``PipelineState`` types ``treatment_var`` and ``outcome_var`` as
        ``Optional[str]``. None passes through; a real string passes
        through; anything else (int, bool, list, dict, etc.) raises
        ``TypeError`` so we fail closed on malformed input.

        Addresses codex iter-3 MEDIUM: previously, ``treatment_var=42``
        could be silently passed to ``graph.add_node(42)`` (networkx
        accepts hashable nodes) and the executor would return success=True
        with integer nodes — violating the TypedDict contract and the
        downstream consumer expectation that nodes are string variable
        names.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"state[{field_name!r}] must be a string or None, "
                f"got {type(value).__name__}: {value!r}"
            )
        return value

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        """Coerce a state field expected to be a list-of-strings.

        ``PipelineState`` types ``confounders`` and ``effect_modifiers`` as
        ``Optional[List[str]]``. Empty list / None -> ``[]``. Anything else
        (bare string, dict, int, tuple, set, etc.) raises ``TypeError`` so
        we fail closed on malformed input. This avoids the silent-acceptance
        trap codex iter-1 MEDIUM flagged: a dict like ``{"bad": "C"}`` is
        iterable and yields string keys, which the previous loose form
        would have happily turned into a one-element ``["bad"]`` confounder
        list — silently producing a partial-but-success graph.
        """
        if value is None:
            return []
        # Strict: must be a list (the type the TypedDict promises). Reject
        # bare strings, dicts, tuples, sets, generators, etc.
        if not isinstance(value, list):
            raise TypeError(f"expected list of strings, got {type(value).__name__}: {value!r}")
        result: List[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"expected list of strings, found {type(item).__name__}: {item!r}")
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
        cycle_affects_identification = False
        cycles_on_relevant_subgraph: List[List[str]] = []
        orientation_ambiguity_only = False
        if not is_dag:
            try:
                cycles = [list(c) for c in nx.simple_cycles(graph)]
            except Exception as e:
                logger.warning(f"nx.simple_cycles failed: {e}")
                cycles = []
            assessment = self._assess_cycle_identifiability(
                graph,
                cycles=cycles,
                treatment=treatment,
                outcome=outcome,
            )
            cycle_affects_identification = assessment["cycle_affects_identification"]
            cycles_on_relevant_subgraph = assessment["cycles_on_relevant_subgraph"]
            orientation_ambiguity_only = assessment["orientation_ambiguity_only"]

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
            # M-fo2 (precise): whether a detected cycle actually breaks backdoor
            # identification of the (treatment, outcome) estimand.
            "cycle_affects_identification": cycle_affects_identification,
            "cycles_on_relevant_subgraph": cycles_on_relevant_subgraph,
            "orientation_ambiguity_only": orientation_ambiguity_only,
            # Provenance
            "graph_source": graph_source,
            "treatment_var": treatment,
            "outcome_var": outcome,
        }

    def _assess_cycle_identifiability(
        self,
        graph: nx.DiGraph,
        *,
        cycles: List[List[str]],
        treatment: Optional[str],
        outcome: Optional[str],
    ) -> Dict[str, Any]:
        """M-fo2: decide whether detected cycles break backdoor identification.

        Backdoor identification of the (treatment, outcome) effect depends only on
        the subgraph induced by the ancestral set ``An({T,Y}) ∪ {T,Y}`` — the set
        that determines d-separation between T and Y. A directed cycle entirely
        OUTSIDE that set can neither open nor block a backdoor path between T and Y,
        so the estimand stays identifiable; a cycle that intersects it makes
        backdoor adjustment undefined.

        Conservative fail-closed: when treatment/outcome are unknown/absent, or the
        graph is non-DAG but ``simple_cycles`` yielded nothing (it raised), we
        cannot prove irrelevance → treat the cycle as identification-affecting.

        A reciprocal 2-cycle (``A<->B``) is an unoriented edge (CPDAG/PAG), flagged
        via ``orientation_ambiguity_only`` (informational); it still blocks
        identification when it lands on the relevant subgraph.

        Scope/assumption (codex MEDIUM): ``An({T,Y})`` is the relevant set for
        UNCONDITIONAL backdoor identification. If a downstream estimator conditioned
        on a DESCENDANT of T or Y, that collider could open a path through an
        otherwise off-subgraph cycle — which this check would not catch. That case
        is guarded elsewhere: the agent adjustment-set finder excludes
        colliders/descendants (M-gb2) and DoWhy's own backdoor criterion rejects a
        non-DAG input. ``nx.ancestors`` on a cyclic graph also OVER-includes
        (reachability via the cycle), which only tightens the gate (marks more
        cycles relevant) — a safe direction.
        """
        orientation_ambiguity_only = bool(cycles) and all(len(c) == 2 for c in cycles)

        if not treatment or not outcome or treatment not in graph or outcome not in graph:
            return {
                "cycle_affects_identification": True,
                "cycles_on_relevant_subgraph": [list(c) for c in cycles],
                "orientation_ambiguity_only": orientation_ambiguity_only,
            }
        if not cycles:
            # Non-DAG but cycle set unavailable -> cannot prove irrelevance.
            return {
                "cycle_affects_identification": True,
                "cycles_on_relevant_subgraph": [],
                "orientation_ambiguity_only": orientation_ambiguity_only,
            }

        relevant: set[str] = {treatment, outcome}
        relevant |= nx.ancestors(graph, treatment)
        relevant |= nx.ancestors(graph, outcome)

        on_relevant = [list(c) for c in cycles if relevant.intersection(c)]
        return {
            "cycle_affects_identification": bool(on_relevant),
            "cycles_on_relevant_subgraph": on_relevant,
            "orientation_ambiguity_only": orientation_ambiguity_only,
        }

    def _compute_centrality(self, graph: nx.DiGraph) -> Dict[str, Any]:
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
            in_degree = {node: int(graph.in_degree(node)) for node in graph.nodes()}
            out_degree = {node: int(graph.out_degree(node)) for node in graph.nodes()}

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
                    shortest_path_length = int(nx.shortest_path_length(graph, treatment, outcome))
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                logger.debug(f"No path treatment->outcome: {e}")

        return {
            "treatment_to_outcome": treatment_to_outcome,
            "n_paths_treatment_to_outcome": n_paths,
            "shortest_path_length": shortest_path_length,
        }

    @staticmethod
    def _compute_confidence(*, n_nodes: int, has_path: bool, identification_blocked: bool) -> float:
        """Derive structural confidence (M-fo2 precise refinement of spike §2.1).

        - 0.0 when identification is blocked (a directed cycle on the
          treatment-outcome ancestral subgraph → backdoor adjustment undefined)
        - 1.0 when identification holds + treatment-outcome path + >=3 nodes
          (a DAG, OR a non-DAG whose cycle is OFF the relevant subgraph)
        - 0.5 otherwise (identifiable but path missing OR < 3 nodes)
        """
        if identification_blocked:
            return 0.0
        if has_path and n_nodes >= 3:
            return 1.0
        return 0.5
