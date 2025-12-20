"""Graph Builder Node - Causal DAG construction.

Constructs causal DAGs using domain knowledge and LLM assistance.
Identifies treatment/outcome variables and valid adjustment sets.
Computes DAG version hash for expert review workflow.

Version: 4.3
"""

import time
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from src.agents.causal_impact.state import CausalImpactState, CausalGraph
from src.causal_engine import compute_dag_hash


class GraphBuilderNode:
    """Builds causal DAG from query and domain knowledge.

    Performance target: <10s
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
        """Initialize graph builder."""
        pass

    async def execute(self, state: CausalImpactState) -> Dict:
        """Build causal DAG.

        Args:
            state: Current workflow state with query and variables

        Returns:
            Updated state with causal_graph populated
        """
        start_time = time.time()

        try:
            # Extract or infer treatment/outcome (contract-aligned field names)
            treatment = state.get("treatment_var")
            outcome = state.get("outcome_var")
            confounders = state.get("confounders", [])

            if not treatment or not outcome:
                treatment, outcome = self._infer_variables_from_query(
                    state.get("query", "")
                )

            # Build DAG
            dag = self._construct_dag(treatment, outcome, confounders)

            # Find valid adjustment sets (backdoor criterion)
            adjustment_sets = self._find_adjustment_sets(dag, treatment, outcome)

            # Convert to CausalGraph
            causal_graph: CausalGraph = {
                "nodes": list(dag.nodes()),
                "edges": list(dag.edges()),
                "treatment_nodes": [treatment],
                "outcome_nodes": [outcome],
                "adjustment_sets": adjustment_sets,
                "dag_dot": self._to_dot_format(dag),
                "confidence": 0.85 if treatment and outcome else 0.5,
            }

            # Compute DAG version hash for expert review tracking
            dag_version_hash = compute_dag_hash(causal_graph=causal_graph)
            causal_graph["dag_version_hash"] = dag_version_hash

            latency_ms = (time.time() - start_time) * 1000

            return {
                **state,
                "causal_graph": causal_graph,
                "dag_version_hash": dag_version_hash,
                "graph_builder_latency_ms": latency_ms,
                "current_phase": "estimating",
            }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
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

    def _construct_dag(
        self, treatment: str, outcome: str, confounders: List[str]
    ) -> nx.DiGraph:
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
                if self._blocks_all_backdoor_paths(
                    dag, set(node_pair), treatment, outcome
                ):
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
            all_paths = nx.all_simple_paths(
                dag.to_undirected(), treatment, outcome, cutoff=5
            )

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
