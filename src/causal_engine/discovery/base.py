"""
E2I Causal Analytics - Causal Discovery Base Classes
=====================================================

Base classes and interfaces for causal structure learning.

Provides:
- DiscoveryAlgorithm: Protocol for discovery algorithm wrappers
- DiscoveryResult: Standardized result container
- DiscoveryConfig: Configuration for structure learning
- GateDecision: Decision outcomes from DiscoveryGate

Author: E2I Causal Analytics Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple
from uuid import UUID

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class DiscoveryAlgorithmType(str, Enum):
    """Supported causal discovery algorithms."""

    GES = "ges"  # Greedy Equivalence Search
    PC = "pc"  # Peter-Clark
    FCI = "fci"  # Fast Causal Inference
    LINGAM = "lingam"  # Linear Non-Gaussian Acyclic Model
    DIRECT_LINGAM = "direct_lingam"  # DirectLiNGAM
    ICA_LINGAM = "ica_lingam"  # ICA-based LiNGAM


class GateDecision(str, Enum):
    """Decision outcomes from DiscoveryGate evaluation."""

    ACCEPT = "accept"  # High confidence, use discovered DAG
    REVIEW = "review"  # Medium confidence, requires expert validation
    REJECT = "reject"  # Low confidence, use manual DAG instead
    AUGMENT = "augment"  # Supplement manual DAG with high-confidence edges


class EdgeType(str, Enum):
    """Types of edges in discovered graphs."""

    DIRECTED = "directed"  # Definite causal direction: X -> Y
    UNDIRECTED = "undirected"  # Unknown direction: X - Y
    BIDIRECTED = "bidirected"  # Possible confounder: X <-> Y


@dataclass
class DiscoveredEdge:
    """A single discovered edge with confidence metadata.

    Attributes:
        source: Source node name
        target: Target node name
        edge_type: Type of edge (directed, undirected, bidirected)
        confidence: Confidence score [0, 1]
        algorithm_votes: Number of algorithms that found this edge
        algorithms: List of algorithm names that found this edge
        bootstrap_stability: Fraction of bootstrap resamples that recovered
            this edge (None when bootstrap resampling was not run)
    """

    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECTED
    confidence: float = 1.0
    algorithm_votes: int = 1
    algorithms: List[str] = field(default_factory=list)
    bootstrap_stability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "algorithm_votes": self.algorithm_votes,
            "algorithms": self.algorithms,
            "bootstrap_stability": self.bootstrap_stability,
        }


@dataclass
class CausalPriorKnowledge:
    """Domain priors that constrain causal structure learning.

    Observational discovery only recovers a Markov equivalence class — edge
    ORIENTATION is underdetermined, so unconstrained PC/GES can emit
    implausibly-directed edges (e.g. ``treatment -> confounder``). These priors
    anchor what is KNOWN so the DATA still decides the rest. Maps onto
    causal-learn's ``BackgroundKnowledge`` (see ``build_background_knowledge``):

    - ``tiers``: ordered groups; a variable in an earlier tier is causally PRIOR
      to (cannot be caused by) one in a later tier. E.g.
      ``[[confounders...], [treatment], [outcome]]`` forbids treatment->confounder
      and outcome->anything, so confounder/treatment/outcome edges orient
      correctly while the data still selects WHICH covariates are confounders
      (validated on patient_journeys).
    - ``required_edges``: ``(from, to)`` directed edges that must be present
      (e.g. the treatment->outcome hypothesis under test).
    - ``forbidden_edges``: ``(from, to)`` directed edges that must be absent.
    """

    tiers: Optional[List[List[str]]] = None
    required_edges: Optional[List[Tuple[str, str]]] = None
    forbidden_edges: Optional[List[Tuple[str, str]]] = None

    def is_empty(self) -> bool:
        """True when no prior is set (discovery runs fully unconstrained)."""
        return not (self.tiers or self.required_edges or self.forbidden_edges)


@dataclass
class DiscoveryConfig:
    """Configuration for causal structure learning.

    Attributes:
        algorithms: List of algorithms to run
        alpha: Significance level for conditional independence tests
        max_cond_vars: Maximum conditioning set size
        ensemble_threshold: Minimum fraction of algorithms that must agree on an edge
        max_iter: Maximum iterations for iterative algorithms
        random_state: Random seed for reproducibility
        score_func: Scoring function for score-based methods (BIC, BDeu)
        assume_linear: Assume linear relationships (enables LiNGAM)
        assume_gaussian: Assume Gaussian errors
        use_process_pool: Use ProcessPoolExecutor for true parallelism (default: False)
        max_workers: Maximum worker processes/threads (default: None = CPU count)
        bootstrap_resamples: Resamples for single-algorithm edge-stability
            measurement (0 = off)
    """

    algorithms: List[DiscoveryAlgorithmType] = field(
        default_factory=lambda: [DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC]
    )
    alpha: float = 0.05
    max_cond_vars: Optional[int] = None
    ensemble_threshold: float = 0.5
    max_iter: int = 10000
    random_state: int = 42
    score_func: str = "local_score_BIC"
    assume_linear: bool = True
    assume_gaussian: bool = False
    use_process_pool: bool = False
    max_workers: Optional[int] = None
    # Resamples for single-algorithm edge-stability measurement (0 = off).
    # The gate scores corroboration from these frequencies when only one
    # algorithm ran (agreement is vacuous there); see DiscoveryGate.
    bootstrap_resamples: int = 0
    # Domain priors (tiers / required / forbidden edges) for GUIDED discovery.
    # Only the PC algorithm consumes these (causal-learn BackgroundKnowledge);
    # when set, prefer ``algorithms=[PC]`` so the ensemble is not polluted by
    # unconstrained orientations from algorithms that ignore the priors.
    prior_knowledge: Optional[CausalPriorKnowledge] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "algorithms": [alg.value for alg in self.algorithms],
            "alpha": self.alpha,
            "max_cond_vars": self.max_cond_vars,
            "ensemble_threshold": self.ensemble_threshold,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "score_func": self.score_func,
            "assume_linear": self.assume_linear,
            "assume_gaussian": self.assume_gaussian,
            "use_process_pool": self.use_process_pool,
            "max_workers": self.max_workers,
            "bootstrap_resamples": self.bootstrap_resamples,
            "prior_knowledge": (
                {
                    "tiers": self.prior_knowledge.tiers,
                    "required_edges": [
                        list(e) for e in (self.prior_knowledge.required_edges or [])
                    ],
                    "forbidden_edges": [
                        list(e) for e in (self.prior_knowledge.forbidden_edges or [])
                    ],
                }
                if self.prior_knowledge is not None
                else None
            ),
        }


@dataclass
class AlgorithmResult:
    """Result from a single discovery algorithm.

    Attributes:
        algorithm: Algorithm type used
        adjacency_matrix: Adjacency matrix of discovered graph
        edge_list: List of discovered edges
        runtime_seconds: Time taken to run algorithm
        converged: Whether algorithm converged
        score: Model score (for score-based methods)
        metadata: Additional algorithm-specific metadata
    """

    algorithm: DiscoveryAlgorithmType
    adjacency_matrix: NDArray[np.int_]
    edge_list: List[Tuple[str, str]]
    runtime_seconds: float
    converged: bool = True
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Standardized result container for causal discovery.

    Attributes:
        success: Whether discovery succeeded
        config: Configuration used
        ensemble_dag: Final DAG from ensemble
        edges: List of discovered edges with confidence
        algorithm_results: Results from individual algorithms
        gate_decision: Gate evaluation result
        gate_confidence: Overall confidence score
        created_at: Timestamp of discovery
        session_id: Session ID for tracking
        metadata: Additional metadata
    """

    success: bool
    config: DiscoveryConfig
    ensemble_dag: Optional[nx.DiGraph] = None
    edges: List[DiscoveredEdge] = field(default_factory=list)
    algorithm_results: List[AlgorithmResult] = field(default_factory=list)
    gate_decision: Optional[GateDecision] = None
    gate_confidence: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        """Number of edges in ensemble DAG."""
        return len(self.edges)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in ensemble DAG."""
        if self.ensemble_dag is None:
            return 0
        return int(self.ensemble_dag.number_of_nodes())

    @property
    def algorithm_agreement(self) -> float:
        """Average agreement across algorithms for edges."""
        if not self.edges:
            return 0.0
        total_votes = sum(e.algorithm_votes for e in self.edges)
        max_votes = len(self.algorithm_results) * len(self.edges)
        return total_votes / max_votes if max_votes > 0 else 0.0

    def get_high_confidence_edges(self, threshold: float = 0.8) -> List[DiscoveredEdge]:
        """Get edges with confidence above threshold."""
        return [e for e in self.edges if e.confidence >= threshold]

    def to_adjacency_matrix(self, node_order: Optional[List[str]] = None) -> NDArray[np.int_]:
        """Convert ensemble DAG to adjacency matrix.

        Args:
            node_order: Order of nodes in matrix. If None, uses sorted node names.

        Returns:
            Adjacency matrix where A[i,j] = 1 means edge i -> j
        """
        if self.ensemble_dag is None:
            return np.array([[]])

        if node_order is None:
            node_order = sorted(self.ensemble_dag.nodes())

        n = len(node_order)
        adj = np.zeros((n, n), dtype=int)
        node_idx = {node: i for i, node in enumerate(node_order)}

        for edge in self.edges:
            if edge.source in node_idx and edge.target in node_idx:
                i, j = node_idx[edge.source], node_idx[edge.target]
                adj[i, j] = 1

        return adj

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "config": self.config.to_dict(),
            "n_edges": self.n_edges,
            "n_nodes": self.n_nodes,
            "edges": [e.to_dict() for e in self.edges],
            "gate_decision": self.gate_decision.value if self.gate_decision else None,
            "gate_confidence": self.gate_confidence,
            "algorithm_agreement": self.algorithm_agreement,
            "created_at": self.created_at.isoformat(),
            "session_id": str(self.session_id) if self.session_id else None,
            "metadata": self.metadata,
        }


class DiscoveryAlgorithm(Protocol):
    """Protocol for causal discovery algorithm wrappers.

    All algorithm wrappers must implement this interface.
    """

    @property
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return the algorithm type."""
        ...

    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run causal discovery on data.

        Args:
            data: Input data with variables as columns
            config: Discovery configuration

        Returns:
            AlgorithmResult with discovered structure
        """
        ...

    def supports_latent_confounders(self) -> bool:
        """Whether algorithm can detect latent confounders."""
        ...


class BaseDiscoveryAlgorithm(ABC):
    """Abstract base class for discovery algorithm wrappers.

    Provides common functionality for all algorithm implementations.
    """

    @property
    @abstractmethod
    def algorithm_type(self) -> DiscoveryAlgorithmType:
        """Return the algorithm type."""
        pass

    @abstractmethod
    def discover(
        self,
        data: pd.DataFrame,
        config: DiscoveryConfig,
    ) -> AlgorithmResult:
        """Run causal discovery on data."""
        pass

    @abstractmethod
    def supports_latent_confounders(self) -> bool:
        """Whether algorithm can detect latent confounders."""
        pass

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")

        if data.isnull().any().any():
            raise ValueError("Input data contains missing values. Please impute first.")

        if len(data.columns) < 2:
            raise ValueError("Need at least 2 variables for causal discovery")

    def _adjacency_to_edge_list(
        self,
        adj_matrix: NDArray[np.int_],
        node_names: List[str],
    ) -> List[Tuple[str, str]]:
        """Convert adjacency matrix to edge list.

        Args:
            adj_matrix: Adjacency matrix where A[i,j] = 1 means edge i -> j
            node_names: Names of nodes in order

        Returns:
            List of (source, target) tuples
        """
        edges = []
        rows, cols = np.where(adj_matrix != 0)
        for i, j in zip(rows, cols, strict=False):
            edges.append((node_names[i], node_names[j]))
        return edges
