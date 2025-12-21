"""
Pydantic models for RAG insights and chunks.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class EnrichedInsight(BaseModel):
    """LLM-enriched insight from retrieved content."""

    summary: str = Field(..., description="Concise summary of insights")
    key_findings: List[str] = Field(
        default_factory=list, description="Key findings as bullet points"
    )
    supporting_evidence: List[Any] = Field(
        default_factory=list, description="Supporting RetrievalResults"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score based on data quality"
    )
    data_freshness: Optional[datetime] = Field(
        default=None, description="Timestamp of most recent data used"
    )


class Chunk(BaseModel):
    """A chunk of content prepared for vector indexing."""

    content: str = Field(..., description="Chunk text content")
    source_type: Literal[
        "agent_analysis", "causal_path", "kpi_snapshot", "conversation"
    ] = Field(..., description="Type of source content")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding (384 dimensions for MiniLM)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata for filtering"
    )

    # Allow numpy arrays for embeddings
    model_config = ConfigDict(arbitrary_types_allowed=True)
