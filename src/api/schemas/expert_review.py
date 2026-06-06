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

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PendingReviewItem(BaseModel):
    """A single pending expert review.

    Mirrors the ``v_pending_expert_reviews`` view columns (010 :312-322) /
    ``get_pending_reviews`` row shape. ``days_pending`` is computed by the view;
    when reading the base table directly it may be absent (left optional).
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

    model_config = ConfigDict(extra="ignore")


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
