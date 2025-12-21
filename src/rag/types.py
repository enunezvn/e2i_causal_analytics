"""
E2I Hybrid RAG - Core Types and Data Structures

This module defines the core types used throughout the hybrid RAG system:
- RetrievalSource: Enum for tracking which backend returned results
- RetrievalResult: Unified result format from any backend
- ExtractedEntities: Entities extracted from queries for graph search
- SearchStats: Statistics for query auditing

Part of Phase 1, Checkpoint 1.1.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class RetrievalSource(Enum):
    """Track which backend returned each result."""

    VECTOR = "supabase_vector"
    FULLTEXT = "supabase_fulltext"
    GRAPH = "falkordb_graph"

    def __str__(self) -> str:
        return self.value


class BackendStatus(Enum):
    """Health status for each search backend."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RetrievalResult:
    """
    Unified result format from any backend.

    Attributes:
        id: Unique identifier for the result
        content: The text content retrieved
        source: Which backend returned this result
        score: Normalized relevance score (0-1 after RRF)
        metadata: Additional context (source_type, agent_name, timestamps, etc.)
        graph_context: Connected nodes/edges if from graph search
        query_latency_ms: Time taken for this specific query
        raw_score: Original score before normalization
    """

    id: str
    content: str
    source: RetrievalSource
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_context: Optional[Dict[str, Any]] = None
    query_latency_ms: float = 0.0
    raw_score: float = 0.0

    def __post_init__(self):
        """Validate result fields."""
        if not self.id:
            raise ValueError("Result id cannot be empty")
        if self.score < 0:
            raise ValueError(f"Score cannot be negative: {self.score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source.value,
            "score": self.score,
            "metadata": self.metadata,
            "graph_context": self.graph_context,
            "query_latency_ms": self.query_latency_ms,
            "raw_score": self.raw_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            source=RetrievalSource(data["source"]),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
            graph_context=data.get("graph_context"),
            query_latency_ms=data.get("query_latency_ms", 0.0),
            raw_score=data.get("raw_score", 0.0),
        )


@dataclass
class ExtractedEntities:
    """
    Entities extracted from a query for graph search.

    These are E2I domain-specific entities, not clinical/medical entities.
    Used to build Cypher queries for FalkorDB.
    """

    brands: List[str] = field(default_factory=list)
    regions: List[str] = field(default_factory=list)
    kpis: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    journey_stages: List[str] = field(default_factory=list)
    time_references: List[str] = field(default_factory=list)
    hcp_segments: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if no entities were extracted."""
        return not any(
            [
                self.brands,
                self.regions,
                self.kpis,
                self.agents,
                self.journey_stages,
                self.time_references,
                self.hcp_segments,
            ]
        )

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary for Cypher query building."""
        return {
            "brands": self.brands,
            "regions": self.regions,
            "kpis": self.kpis,
            "agents": self.agents,
            "journey_stages": self.journey_stages,
            "time_references": self.time_references,
            "hcp_segments": self.hcp_segments,
        }

    def entity_count(self) -> int:
        """Total number of entities extracted."""
        return sum(
            [
                len(self.brands),
                len(self.regions),
                len(self.kpis),
                len(self.agents),
                len(self.journey_stages),
                len(self.time_references),
                len(self.hcp_segments),
            ]
        )


@dataclass
class BackendHealth:
    """Health status for a single backend."""

    status: BackendStatus
    latency_ms: float
    last_check: datetime
    consecutive_failures: int = 0
    error_message: Optional[str] = None

    def is_available(self) -> bool:
        """Check if backend is available for queries."""
        return self.status in (BackendStatus.HEALTHY, BackendStatus.DEGRADED)


@dataclass
class SearchStats:
    """
    Statistics from a hybrid search query for auditing.

    Stored for debugging and performance monitoring.
    """

    query: str
    total_latency_ms: float
    vector_count: int
    fulltext_count: int
    graph_count: int
    fused_count: int
    sources_used: Dict[str, bool]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional detailed timing
    vector_latency_ms: Optional[float] = None
    fulltext_latency_ms: Optional[float] = None
    graph_latency_ms: Optional[float] = None
    fusion_latency_ms: Optional[float] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "query": self.query,
            "total_latency_ms": self.total_latency_ms,
            "vector_count": self.vector_count,
            "fulltext_count": self.fulltext_count,
            "graph_count": self.graph_count,
            "fused_count": self.fused_count,
            "sources_used": self.sources_used,
            "timestamp": self.timestamp.isoformat(),
            "vector_latency_ms": self.vector_latency_ms,
            "fulltext_latency_ms": self.fulltext_latency_ms,
            "graph_latency_ms": self.graph_latency_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "errors": self.errors,
        }


@dataclass
class GraphPath:
    """
    Represents a causal path from FalkorDB graph.

    Used for graph context in RetrievalResult.
    """

    source_node: str
    target_node: str
    relationship_types: List[str]
    path_length: int
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relationship_types": self.relationship_types,
            "path_length": self.path_length,
            "nodes": self.nodes,
            "properties": self.properties,
        }
