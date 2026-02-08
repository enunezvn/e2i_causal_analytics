"""Graph Builder Node - Causal DAG construction.

Constructs causal DAGs using domain knowledge and LLM assistance.
Identifies treatment/outcome variables and valid adjustment sets.
Computes DAG version hash for expert review workflow.

V4.4 Enhancement: Automatic causal structure learning with multi-algorithm
ensemble (GES, PC) and gated acceptance for discovered DAGs.

Version: 4.4
"""

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, cast

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

from src.agents.causal_impact.state import CausalGraph, CausalImpactState
from src.causal_engine import compute_dag_hash
from src.causal_engine.discovery import (
    DiscoveryAlgorithmType,
    DiscoveryConfig,
    DiscoveryGate,
    DiscoveryResult,
    DiscoveryRunner,
    GateDecision,
)


class GraphBuilderNode:
    """Builds causal DAG from query and domain knowledge.

    V4.4 Enhancement: Supports automatic DAG discovery using structure learning
    algorithms (GES, PC) with gated acceptance criteria.

    Modes:
    - Manual (default): Constructs DAG from domain knowledge
    - Auto-Discovery: Uses ensemble of algorithms to discover structure
    - Hybrid: Augments manual DAG with high-confidence discovered edges

    Performance target: <10s (manual), <30s (discovery)
    Type: Standard (computation-heavy)
    """

    # Domain knowledge: Common causal structures in pharma commercial data
    KNOWN_CAUSAL_RELATIONSHIPS = {
        # HCP engagement → patient outcomes
        ("hcp_engagement_level", "patient_conversion_rate"),
        ("hcp_meeting_frequency", "prescription_volume"),
        ("hcp_sample_provision", "new_patient_starts"),
        # Marketing → HCP behavior
        ("marketing_spend", "hcp_engagement_level"),
        ("digital_campaign_reach", "hcp_meeting_acceptance"),
        ("conference_attendance", "hcp_awareness"),
        # Patient journey
        ("patient_awareness", "patient_conversion_rate"),
        ("prior_authorization_time", "treatment_adherence"),
        ("copay_support", "prescription_abandonment"),
        # Market dynamics
        ("competitor_activity", "market_share"),
        ("formulary_status", "prescription_volume"),
        ("geographic_region", "hcp_engagement_level"),  # Confounder
        ("therapeutic_area_expertise", "prescription_volume"),  # Confounder
    }

    # Common confounders in pharma data
    COMMON_CONFOUNDERS = {
        "geographic_region",
        "therapeutic_area_expertise",
        "hcp_specialty",
        "practice_size",
        "patient_demographics",
        "market_size",
        "seasonality",
        "time_period",
    }

    def __init__(self):
        """Initialize graph builder with discovery components."""
        self._discovery_runner: Optional[DiscoveryRunner] = None
        self._discovery_gate: Optional[DiscoveryGate] = None

    @property
    def discovery_runner(self) -> DiscoveryRunner:
        """Lazy-initialize discovery runner."""
        if self._discovery_runner is None:
            self._discovery_runner = DiscoveryRunner()
        return self._discovery_runner

    @property
    def discovery_gate(self) -> DiscoveryGate:
        """Lazy-initialize discovery gate."""
        if self._discovery_gate is None:
            self._discovery_gate = DiscoveryGate()
        return self._discovery_gate

    async def execute(self, state: CausalImpactState) -> Dict:
        """Build causal DAG with optional auto-discovery.

        Args:
            state: Current workflow state with query and variables

        Returns:
            Updated state with causal_graph populated

        V4.4: If auto_discover=True, attempts structure learning first,
        then falls back to manual DAG based on gate decision.
        """
        start_time = time.time()

        try:
            # Extract or infer treatment/outcome (contract-aligned field names)
            treatment = state.get("treatment_var")
            outcome = state.get("outcome_var")
            confounders = state.get("confounders", [])

            if not treatment or not outcome:
                treatment, outcome = self._infer_variables_from_query(state.get("query", ""))

            # Check if auto-discovery is enabled
            auto_discover = state.get("auto_discover", False)
            discovery_result: Optional[DiscoveryResult] = None
            gate_evaluation: Optional[Dict[str, Any]] = None
            discovery_latency_ms: float = 0.0

            if auto_discover:
                logger.info("Auto-discovery enabled, attempting structure learning")
                discovery_start = time.time()

                try:
                    discovery_result, gate_evaluation = await self._run_discovery(
                        state, treatment, outcome
                    )
                    discovery_latency_ms = (time.time() - discovery_start) * 1000
                except Exception as e:
                    logger.warning(f"Discovery failed: {e}, falling back to manual DAG")
                    discovery_latency_ms = (time.time() - discovery_start) * 1000

            # Build DAG based on discovery results
            if discovery_result and gate_evaluation:
                dag, augmented_edges = self._build_dag_with_discovery(
                    treatment, outcome, confounders, discovery_result, gate_evaluation
                )
            else:
                # Manual DAG construction (original behavior)
                dag = self._construct_dag(treatment, outcome, confounders)
                augmented_edges = []

            # Find valid adjustment sets (backdoor criterion)
            adjustment_sets = self._find_adjustment_sets(dag, treatment, outcome)

            # Compute confidence based on discovery results
            if gate_evaluation and gate_evaluation.get("decision") == GateDecision.ACCEPT.value:
                confidence = gate_evaluation.get("confidence", 0.85)
            elif gate_evaluation and gate_evaluation.get("decision") == GateDecision.AUGMENT.value:
                # Hybrid confidence
                confidence = min(0.9, 0.85 + 0.05 * len(augmented_edges))
            else:
                confidence = 0.85 if treatment and outcome else 0.5

            # Convert to CausalGraph
            causal_graph: CausalGraph = {
                "nodes": list(dag.nodes()),
                "edges": list(dag.edges()),
                "treatment_nodes": [treatment],
                "outcome_nodes": [outcome],
                "adjustment_sets": adjustment_sets,
                "dag_dot": self._to_dot_format(dag),
                "confidence": confidence,
                # V4.4: Discovery metadata
                "discovery_enabled": auto_discover,
                "discovery_gate_decision": cast(
                    Literal["accept", "review", "reject", "augment"],
                    gate_evaluation.get("decision") if gate_evaluation else "accept",
                ),
                "discovery_algorithms_used": (
                    [a.value for a in discovery_result.config.algorithms]
                    if discovery_result and discovery_result.config
                    else []
                ),
                "discovery_confidence": gate_evaluation.get("confidence", 0.0)
                if gate_evaluation
                else 0.0,
                "discovery_n_edges": discovery_result.n_edges if discovery_result else 0,
                "augmented_edges": augmented_edges,
            }

            # Compute DAG version hash for expert review tracking
            dag_version_hash = compute_dag_hash(causal_graph=causal_graph.copy())  # type: ignore[arg-type]
            causal_graph["dag_version_hash"] = dag_version_hash

            latency_ms = (time.time() - start_time) * 1000

            result = {
                **state,
                "causal_graph": causal_graph,
                "dag_version_hash": dag_version_hash,
                "graph_builder_latency_ms": latency_ms,
                "current_phase": "estimating",
            }

            # Add discovery metadata if used
            if auto_discover:
                result["discovery_latency_ms"] = discovery_latency_ms
                if discovery_result:
                    result["discovery_result"] = discovery_result.to_dict()
                if gate_evaluation:
                    result["discovery_gate_evaluation"] = gate_evaluation

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Graph building failed: {e}")
            return {
                **state,
                "graph_builder_error": str(e),
                "graph_builder_latency_ms": latency_ms,
                "status": "failed",
                "error_message": f"Graph building failed: {e}",
            }

    def _infer_variables_from_query(self, query: str) -> Tuple[str, str]:
        """Infer treatment and outcome from query text.

        Args:
            query: Natural language query

        Returns:
            (treatment, outcome) tuple
        """
        query_lower = query.lower()

        # Treatment keywords
        treatment_keywords = {
            "hcp engagement": "hcp_engagement_level",
            "marketing": "marketing_spend",
            "sample": "hcp_sample_provision",
            "meeting": "hcp_meeting_frequency",
            "campaign": "digital_campaign_reach",
            "copay": "copay_support",
        }

        # Outcome keywords
        outcome_keywords = {
            "conversion": "patient_conversion_rate",
            "prescription": "prescription_volume",
            "nrx": "new_patient_starts",
            "trx": "total_prescriptions",
            "adherence": "treatment_adherence",
            "market share": "market_share",
        }

        treatment = None
        outcome = None

        for keyword, var in treatment_keywords.items():
            if keyword in query_lower:
                treatment = var
                break

        for keyword, var in outcome_keywords.items():
            if keyword in query_lower:
                outcome = var
                break

        # Defaults
        if not treatment:
            treatment = "hcp_engagement_level"
        if not outcome:
            outcome = "patient_conversion_rate"

        return treatment, outcome

    def _construct_dag(self, treatment: str, outcome: str, confounders: List[str]) -> nx.DiGraph:
        """Construct causal DAG from variables.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounding variables to adjust for

        Returns:
            NetworkX directed graph
        """
        dag = nx.DiGraph()

        # Add core nodes
        dag.add_node(treatment)
        dag.add_node(outcome)

        # Add confounders
        for conf in confounders:
            dag.add_node(conf)

        # Add known causal edges
        for source, target in self.KNOWN_CAUSAL_RELATIONSHIPS:
            if source in dag.nodes() and target in dag.nodes():
                dag.add_edge(source, target)

        # Add direct treatment → outcome edge if no path exists
        if not nx.has_path(dag, treatment, outcome):
            dag.add_edge(treatment, outcome)

        # Add common confounders
        confounders_to_add = [
            c for c in self.COMMON_CONFOUNDERS if c in confounders or c in dag.nodes()
        ]

        for confounder in confounders_to_add[:3]:  # Limit to top 3 confounders
            if confounder not in dag.nodes():
                dag.add_node(confounder)

            # Confounders affect both treatment and outcome
            if not dag.has_edge(confounder, treatment):
                dag.add_edge(confounder, treatment)
            if not dag.has_edge(confounder, outcome):
                dag.add_edge(confounder, outcome)

        return dag

    def _find_adjustment_sets(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """Find valid backdoor adjustment sets.

        Args:
            dag: Causal DAG
            treatment: Treatment node
            outcome: Outcome node

        Returns:
            List of adjustment sets (each is a list of variable names)
        """
        # Find all backdoor paths (paths that go into treatment)
        backdoor_paths = self._find_backdoor_paths(dag, treatment, outcome)

        if not backdoor_paths:
            return [[]]  # No confounding, empty adjustment set sufficient

        # Find minimal adjustment sets that block all backdoor paths
        all_nodes = set(dag.nodes()) - {treatment, outcome}
        adjustment_sets = []

        # Try individual nodes first
        for node in all_nodes:
            if self._blocks_all_backdoor_paths(dag, {node}, treatment, outcome):
                adjustment_sets.append([node])

        # Try pairs if no individual nodes work
        if not adjustment_sets:
            from itertools import combinations

            for node_pair in combinations(all_nodes, 2):
                if self._blocks_all_backdoor_paths(dag, set(node_pair), treatment, outcome):
                    adjustment_sets.append(list(node_pair))
                    if len(adjustment_sets) >= 3:  # Limit to 3 sets
                        break

        # Fallback: all non-descendants of treatment
        if not adjustment_sets:
            descendants = nx.descendants(dag, treatment)
            fallback_set = list(all_nodes - descendants)
            if fallback_set:
                adjustment_sets.append(fallback_set)

        return adjustment_sets[:3]  # Return top 3 adjustment sets

    def _find_backdoor_paths(
        self, dag: nx.DiGraph, treatment: str, outcome: str
    ) -> List[List[str]]:
        """Find all backdoor paths from treatment to outcome.

        A backdoor path is a path that enters treatment via an arrow
        pointing into treatment.
        """
        backdoor_paths = []

        # Get all simple paths
        try:
            all_paths = nx.all_simple_paths(dag.to_undirected(), treatment, outcome, cutoff=5)

            for path in all_paths:
                # Check if path enters treatment (backdoor)
                if len(path) >= 2:
                    # First edge in undirected path
                    if dag.has_edge(path[1], path[0]):  # Arrow into treatment
                        backdoor_paths.append(path)
        except nx.NetworkXNoPath:
            pass

        return backdoor_paths

    def _blocks_all_backdoor_paths(
        self, dag: nx.DiGraph, adjustment_set: Set[str], treatment: str, outcome: str
    ) -> bool:
        """Check if adjustment set blocks all backdoor paths.

        Args:
            dag: Causal DAG
            adjustment_set: Set of nodes to adjust for
            treatment: Treatment node
            outcome: Outcome node

        Returns:
            True if all backdoor paths are blocked
        """
        backdoor_paths = self._find_backdoor_paths(dag, treatment, outcome)

        for path in backdoor_paths:
            # Check if any node in adjustment set is on this path
            if not any(node in adjustment_set for node in path[1:-1]):
                return False  # Path not blocked

        return True

    def _to_dot_format(self, dag: nx.DiGraph) -> str:
        """Convert DAG to DOT format for visualization.

        Args:
            dag: NetworkX DAG

        Returns:
            DOT format string
        """
        lines = ["digraph CausalDAG {", "  rankdir=LR;", "  node [shape=box];", ""]

        for node in dag.nodes():
            label = node.replace("_", " ").title()
            lines.append(f'  "{node}" [label="{label}"];')

        lines.append("")

        for source, target in dag.edges():
            lines.append(f'  "{source}" -> "{target}";')

        lines.append("}")

        return "\n".join(lines)

    async def _run_discovery(
        self,
        state: CausalImpactState,
        treatment: str,
        outcome: str,
    ) -> Tuple[DiscoveryResult, Dict[str, Any]]:
        """Run structure learning and gate evaluation.

        Args:
            state: Current workflow state
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            Tuple of (DiscoveryResult, gate evaluation dict)
        """
        # Get data from state
        data_cache = state.get("data_cache", {})
        data = data_cache.get("data")

        if data is None:
            # Try to create synthetic data from variable info
            # In production, this would come from the data source
            logger.warning("No data in cache, using minimal discovery")
            raise ValueError("No data available for discovery")

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Build discovery config from state
        algorithms_str = state.get("discovery_algorithms", ["ges", "pc"])
        algorithms = []
        for algo in algorithms_str:
            try:
                algorithms.append(DiscoveryAlgorithmType(algo.lower()))
            except ValueError:
                logger.warning(f"Unknown algorithm: {algo}, skipping")

        if not algorithms:
            algorithms = [DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC]

        config = DiscoveryConfig(
            algorithms=algorithms,
            ensemble_threshold=state.get("discovery_ensemble_threshold", 0.5),
            alpha=state.get("discovery_alpha", 0.05),
        )

        # Run discovery
        session_id = state.get("session_id")
        from uuid import UUID

        session_uuid = UUID(session_id) if session_id else None

        result = await self.discovery_runner.discover_dag(
            data=data,
            config=config,
            session_id=session_uuid,
        )

        # Evaluate with gate
        expected_edges = [(treatment, outcome)]  # Minimal expectation
        evaluation = self.discovery_gate.evaluate(result, expected_edges)

        logger.info(
            f"Discovery complete: {result.n_edges} edges, "
            f"gate decision: {evaluation.decision.value}"
        )

        return result, evaluation.to_dict()

    def _build_dag_with_discovery(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        discovery_result: DiscoveryResult,
        gate_evaluation: Dict[str, Any],
    ) -> Tuple[nx.DiGraph, List[Tuple[str, str]]]:
        """Build DAG based on discovery results and gate decision.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            confounders: Confounder variables
            discovery_result: Result from discovery runner
            gate_evaluation: Result from discovery gate

        Returns:
            Tuple of (DAG, list of augmented edges)
        """
        decision = gate_evaluation.get("decision")
        augmented_edges: List[Tuple[str, str]] = []

        if decision == GateDecision.ACCEPT.value:
            # Use discovered DAG directly
            logger.info("Using discovered DAG (ACCEPT)")
            if discovery_result.ensemble_dag is not None:
                dag = discovery_result.ensemble_dag.copy()
                # Ensure treatment and outcome are present
                if treatment not in dag.nodes():
                    dag.add_node(treatment)
                if outcome not in dag.nodes():
                    dag.add_node(outcome)
                return dag, augmented_edges

        elif decision == GateDecision.AUGMENT.value:
            # Build manual DAG and augment with high-confidence discovered edges
            logger.info("Augmenting manual DAG with discovered edges (AUGMENT)")
            dag = self._construct_dag(treatment, outcome, confounders)

            # Get high-confidence edges from gate evaluation
            high_conf_edges = gate_evaluation.get("high_confidence_edges", [])
            for edge in high_conf_edges:
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    # Only add if doesn't create cycle
                    if not dag.has_edge(source, target):
                        dag.add_edge(source, target)
                        if nx.is_directed_acyclic_graph(dag):
                            augmented_edges.append((source, target))
                            logger.debug(f"Augmented edge: {source} -> {target}")
                        else:
                            dag.remove_edge(source, target)
                            logger.debug(f"Skipped edge (cycle): {source} -> {target}")

            return dag, augmented_edges

        elif decision == GateDecision.REVIEW.value:
            # Use manual DAG but flag for review
            logger.info("Using manual DAG, flagged for review (REVIEW)")
            dag = self._construct_dag(treatment, outcome, confounders)
            return dag, augmented_edges

        # REJECT or unknown: use manual DAG
        logger.info("Using manual DAG (REJECT or fallback)")
        dag = self._construct_dag(treatment, outcome, confounders)
        return dag, augmented_edges


# Standalone function for LangGraph integration
async def build_causal_graph(state: CausalImpactState) -> Dict:
    """Build causal DAG (standalone function).

    Args:
        state: Current workflow state

    Returns:
        Updated state with causal_graph
    """
    node = GraphBuilderNode()
    return await node.execute(state)
