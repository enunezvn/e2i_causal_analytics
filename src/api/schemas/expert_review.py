"""
Expert Review API Schemas

Pydantic schemas for the human-in-the-loop expert-review queue (R6-F2).

These mirror the ``expert_reviews`` table / ``v_pending_expert_reviews`` view
(database/ml/010_causal_validation_tables.sql) and the
``ExpertReviewRepository`` method signatures
(src/repositories/expert_review.py).

A REVIEW-band causal estimate creates a ``pending`` ``expert_reviews`` row via
the repo-backed ``ExpertReviewGate``; an admin reads it via ``GET /pending`` and
resolves it via ``POST /{review_id}/resolve``.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PendingReviewItem(BaseModel):
    """A single pending expert review.

    Mirrors the ``v_pending_expert_reviews`` view columns (010 :312-322) /
    ``get_pending_reviews`` row shape. ``days_pending`` is computed by the view;
    when reading the base table directly it may be absent (left optional).

    ``dag_structure_json`` / ``agent_assessment_json`` (mig 097) are surfaced as
    OBJECTS: the repo write path serializes with ``json.dumps`` (a JSONB string
    scalar), so a string value is parsed here — the frontend never has to
    double-decode.
    """

    review_id: str
    review_type: Optional[str] = None
    dag_version_hash: Optional[str] = None
    brand: Optional[str] = None
    treatment_variable: Optional[str] = None
    outcome_variable: Optional[str] = None
    analysis_context: Optional[str] = None
    created_at: Optional[datetime] = None
    days_pending: Optional[float] = None
    dag_structure_json: Optional[Dict[str, Any]] = None
    agent_assessment_json: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("dag_structure_json", "agent_assessment_json", mode="before")
    @classmethod
    def _parse_json_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None
        return value


class PendingReviewsResponse(BaseModel):
    """Response for ``GET /expert-reviews/pending``."""

    reviews: List[PendingReviewItem]
    total: int


class ResolveReviewRequest(BaseModel):
    """Request body for ``POST /expert-reviews/{review_id}/resolve``.

    ``approval_status`` is constrained to the SAME vocabulary
    ``submit_review`` validates against (repo :157) so a mismatched value is a
    422 (FastAPI validation) rather than a silent repo ``False``.
    """

    approval_status: Literal["approved", "rejected"]
    checklist: Dict[str, Any] = Field(
        default_factory=dict,
        description="Completed reviewer checklist (the 010 checklist template items).",
    )
    comments: Optional[Dict[str, Any]] = Field(
        default=None, description="Reviewer notes / structured feedback."
    )
    concerns_raised: Optional[List[str]] = Field(
        default=None, description="Specific concerns raised during review."
    )
    conditions: Optional[str] = Field(
        default=None, description="Any conditions placed on an approval."
    )
    validity_days: int = Field(
        default=90, ge=1, le=365, description="Days until an approval expires."
    )


class ResolveReviewResponse(BaseModel):
    """Response for ``POST /expert-reviews/{review_id}/resolve``."""

    review_id: str
    approval_status: str
    success: bool


class ReviewSummaryResponse(BaseModel):
    """Response for ``GET /expert-reviews/summary``.

    Mirrors ``get_review_summary`` (repo :507-513).
    """

    pending: int
    approved: int
    rejected: int
    expired: int
    expiring_soon: int


class AgentAssessmentResponse(BaseModel):
    """Response for ``POST /expert-reviews/{review_id}/assessment``.

    ``assessment`` is the advisory verdict set
    (``src.insights.expert_review_assessment.generate_assessment`` shape:
    items[{id, question, verdict, rationale}], is_fallback, evidence).
    ``cached`` marks a replay of the stored ``agent_assessment_json``;
    ``persisted`` is honest about whether a fresh assessment reached the DB —
    the assessment itself is still returned when the cache write fails.
    """

    review_id: str
    assessment: Dict[str, Any]
    cached: bool
    persisted: bool
