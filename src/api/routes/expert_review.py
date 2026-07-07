"""
Expert Review API (R6-F2 Phase A — the human-in-the-loop consumer).

FastAPI endpoints backing the admin review-queue UI for causal-DAG expert
reviews. A REVIEW-band causal estimate creates a ``pending`` ``expert_reviews``
row (via the repo-backed ``ExpertReviewGate``, wired in Phase C); these endpoints
let an operator SEE that queue and RESOLVE (approve/reject) a review. The stored
approval is what lets a future identical-DAG run read PROCEED.

Endpoints (all ``require_operator`` — OD-1):
- GET  /expert-reviews/pending            -> oldest-first pending queue
- POST /expert-reviews/{review_id}/resolve -> approve/reject + checklist/comments
- GET  /expert-reviews/summary            -> status counts

Persistence: ``ExpertReviewRepository`` over an ASYNC Supabase (service-role)
client. The repo methods are ``await self.client.table(...).execute()`` so the
client MUST be async — mirrors ``digital_twin.py:_get_twin_repo`` (#705 H6).
``expert_reviews`` is service_role-only post-058, so a service-role backend
read/write is permitted; anon/authenticated are REVOKEd.

Author: E2I Causal Analytics Team
Version: 4.3.0
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies.auth import require_operator
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.schemas.expert_review import (
    AgentAssessmentResponse,
    PendingReviewItem,
    PendingReviewsResponse,
    ResolveReviewRequest,
    ResolveReviewResponse,
    ReviewSummaryResponse,
)

if TYPE_CHECKING:
    from src.repositories.expert_review import ExpertReviewRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/expert-reviews",
    tags=["Expert Review"],
    responses={
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


async def _get_expert_review_repo() -> "ExpertReviewRepository":
    """Build an ExpertReviewRepository backed by a real async Supabase client (fail-closed).

    Mirrors ``digital_twin.py:_get_twin_repo`` (#705 H6). The repo's queries are
    ``await self.client.table(...).execute()`` — they require an *async* client,
    so use ``get_async_supabase_client`` (NOT the sync ``get_supabase_client``).
    ``get_async_supabase_client`` raises ``ServiceConnectionError`` when the
    Supabase env is missing — we let it surface (fail-closed) rather than
    silently degrading to a None client that would no-op every read/write.
    """
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.expert_review import ExpertReviewRepository

    client = await get_async_supabase_client()
    return ExpertReviewRepository(supabase_client=client)


@router.get(
    "/pending",
    response_model=PendingReviewsResponse,
    summary="List pending expert reviews",
    operation_id="list_pending_expert_reviews",
)
async def list_pending_reviews(
    brand: Optional[str] = Query(None, description="Filter by brand"),
    reviewer_id: Optional[str] = Query(None, description="Filter by assigned reviewer"),
    limit: int = Query(50, ge=1, le=200, description="Maximum records to return"),
    user: Dict[str, Any] = Depends(require_operator),
) -> PendingReviewsResponse:
    """Return the pending review queue (oldest-first), RBAC-gated to operators.

    Each row carries the metadata an operator needs to decide
    (treatment/outcome/brand/analysis_context/dag_version_hash) — the v1 UI is
    metadata + approve/reject, no DAG graph render (OD-2).
    """
    repo = await _get_expert_review_repo()
    rows = await repo.get_pending_reviews(brand=brand, reviewer_id=reviewer_id, limit=limit)
    reviews = [PendingReviewItem.model_validate(row) for row in rows]
    return PendingReviewsResponse(reviews=reviews, total=len(reviews))


@router.post(
    "/{review_id}/resolve",
    response_model=ResolveReviewResponse,
    summary="Resolve (approve/reject) an expert review",
    operation_id="resolve_expert_review",
)
async def resolve_review(
    review_id: str,
    request: ResolveReviewRequest,
    user: Dict[str, Any] = Depends(require_operator),
) -> ResolveReviewResponse:
    """Approve or reject a pending review; the resolution persists.

    An ``approved`` resolution sets ``valid_from``/``valid_until``/``approved_at``
    inside ``submit_review`` (repo :169-173). A repo ``False`` is fail-closed —
    never a fabricated success. FIX B (codex HIGH): ``submit_review`` now returns
    False on a ZERO-ROW update (nonexistent / already-resolved review_id), so we
    surface that as 404 (the honest 'not found / not resolvable' code), not a
    fabricated 200. A genuine persistence error also returns False -> 404, which
    is still a correct non-200 (never a fake success); the repo logs the
    distinction (zero-row WARNING vs exception ERROR).
    """
    repo = await _get_expert_review_repo()
    success = await repo.submit_review(
        review_id=review_id,
        approval_status=request.approval_status,
        checklist=request.checklist,
        comments=request.comments,
        concerns_raised=request.concerns_raised,
        conditions=request.conditions,
        validity_days=request.validity_days,
    )
    if not success:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Review {review_id} was not found or is not resolvable "
                "(it may not exist or has already been resolved)."
            ),
        )
    return ResolveReviewResponse(
        review_id=review_id,
        approval_status=request.approval_status,
        success=True,
    )


async def _get_validation_rows(validation_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch the causal_validations rows a review links as evidence (097)."""
    if not validation_ids:
        return []
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.causal_validation import CausalValidationRepository

    client = await get_async_supabase_client()
    repo = CausalValidationRepository(supabase_client=client)
    return await repo.get_by_ids(validation_ids)


def _build_assessment(review: Dict[str, Any], validations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Grade the checklist questions from the stored evidence (BLOCKING LM call —
    run via asyncio.to_thread). Deferred import keeps the route module light."""
    from src.insights.expert_review_assessment import build_grounding, generate_assessment

    return generate_assessment(build_grounding(review, validations))


def _as_json_object(value: Any) -> Optional[Dict[str, Any]]:
    """agent_assessment_json arrives as dict (JSONB object) or a json.dumps string."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


@router.post(
    "/{review_id}/assessment",
    response_model=AgentAssessmentResponse,
    summary="Generate (or return cached) advisory agent assessment for a review",
    operation_id="generate_expert_review_assessment",
)
async def generate_review_assessment(
    review_id: str,
    force: bool = Query(False, description="Regenerate even when a cached assessment exists"),
    user: Dict[str, Any] = Depends(require_operator),
) -> AgentAssessmentResponse:
    """Advisory agent grading of the six reviewer-checklist questions.

    Grounded ONLY in what the review row stores: the DAG snapshot
    (``dag_structure_json``) and its linked refutation rows
    (``related_validation_ids`` -> ``causal_validations``). The result is cached
    in ``agent_assessment_json`` — kept separate from ``checklist_json``, which
    remains the human reviewer's own record. ``persisted`` is honest about the
    cache write; a failed write still returns the (valid) assessment.
    """
    repo = await _get_expert_review_repo()
    review = await repo.get_by_id(review_id)
    if not review:
        raise HTTPException(
            status_code=404,
            detail=f"Review {review_id} was not found.",
        )

    cached = _as_json_object(review.get("agent_assessment_json"))
    if cached is not None and not force:
        return AgentAssessmentResponse(
            review_id=review_id, assessment=cached, cached=True, persisted=True
        )

    validation_ids = review.get("related_validation_ids") or []
    validations = await _get_validation_rows(validation_ids)
    # run_signature is a BLOCKING LM call; keep the event loop free.
    assessment = await asyncio.to_thread(_build_assessment, review, validations)
    persisted = await repo.update_agent_assessment(review_id, assessment)
    return AgentAssessmentResponse(
        review_id=review_id, assessment=assessment, cached=False, persisted=persisted
    )


@router.get(
    "/summary",
    response_model=ReviewSummaryResponse,
    summary="Expert-review status counts",
    operation_id="get_expert_review_summary",
)
async def get_summary(
    brand: Optional[str] = Query(None, description="Filter by brand"),
    user: Dict[str, Any] = Depends(require_operator),
) -> ReviewSummaryResponse:
    """Return status counts (pending/approved/rejected/expired/expiring_soon)."""
    repo = await _get_expert_review_repo()
    summary = await repo.get_review_summary(brand=brand)
    return ReviewSummaryResponse(
        pending=summary.get("pending", 0),
        approved=summary.get("approved", 0),
        rejected=summary.get("rejected", 0),
        expired=summary.get("expired", 0),
        expiring_soon=summary.get("expiring_soon", 0),
    )
