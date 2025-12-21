"""
Pydantic models for RAG retrieval operations.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A single retrieval result from any retrieval method."""

    content: str = Field(..., description="Retrieved text content")
    source: str = Field(..., description="Source table name")
    source_id: str = Field(..., description="Record ID in source table")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    retrieval_method: Literal["dense", "sparse", "graph", "structured"] = Field(
        ..., description="Method used to retrieve this result"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalContext(BaseModel):
    """Full context from a retrieval operation."""

    query: Any = Field(..., description="Original ParsedQuery")
    results: List[RetrievalResult] = Field(default_factory=list, description="Retrieved results")
    total_retrieved: int = Field(..., ge=0, description="Total results retrieved")
    retrieval_time_ms: float = Field(..., ge=0, description="Retrieval latency in ms")


class RAGQuery(BaseModel):
    """Input contract for RAG operations (from Orchestrator)."""

    parsed_query: Any = Field(..., description="ParsedQuery from NLP layer")
    retrieval_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Override default retrieval settings"
    )
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results")


class RAGResponse(BaseModel):
    """Output contract for RAG operations (to Agents)."""

    context: RetrievalContext = Field(..., description="Retrieval context and results")
    enriched_insight: Optional[Any] = Field(default=None, description="LLM-enriched insight")
    suggested_followups: List[str] = Field(
        default_factory=list, description="Suggested follow-up queries"
    )
