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
- GET  /expert-reviews/{review_id}        -> one review (any status) + same-structure history

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
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies.auth import require_operator
from src.api.dependencies.inflight_lock import InflightLock
from src.api.errors import user_safe_503_detail
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.schemas.expert_review import (
    AgentAssessmentResponse,
    ExpertReviewDetailResponse,
    PendingReviewItem,
    PendingReviewsResponse,
    ResolveReviewRequest,
    ResolveReviewResponse,
    ReviewRecord,
    ReviewSummaryResponse,
)

if TYPE_CHECKING:
    from src.repositories.expert_review import ExpertReviewRepository

logger = logging.getLogger(__name__)

# R3: the repository re-raises store errors instead of returning [] / zeros
# (a "0 pending" page with HTTP 200 during an outage is a plausible-wrong
# value). The app's catch-all classifies exceptions by message keyword
# ("connection"/"unavailable" -> 503, otherwise a generic 500), which is not a
# contract, so these routes map a store failure themselves with the existing
# HTTPException(503) pattern (routes/causal.py, routes/digital_twin.py). The
# app's StarletteHTTPException handler MASKS a 503 detail unless it is marked
# with errors.user_safe_503_detail(); this one names no internals, so it is
# marked and reaches the client as the response ``message``. (The handler
# builds its own JSONResponse, so an HTTPException ``Retry-After`` header
# would be dropped -- its body already says "try again in 30 seconds".)
_STORE_UNAVAILABLE_DETAIL = "Expert-review store unavailable. Retry shortly."

# #1993: one LLM build per review id across BOTH gunicorn workers. Redis
# ``SET NX PX`` with a process-local fallback. The TTL (120 s) tracks nginx's
# ``proxy_read_timeout 120s`` for /api/ (docker/nginx/nginx.secure.conf:227),
# the longest a caller waits on a build; see dependencies/inflight_lock.py for
# the degradation and wait bounds and what happens when a build outlives it.
_ASSESSMENT_LOCK = InflightLock("expert_review:assessment:inflight")


def _store_unavailable(operation: str, exc: Exception) -> HTTPException:
    logger.error(f"Expert-review {operation} failed: {exc}", exc_info=True)
    return HTTPException(
        status_code=503,
        detail=user_safe_503_detail(_STORE_UNAVAILABLE_DETAIL),
    )


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
    try:
        rows = await repo.get_pending_reviews(brand=brand, reviewer_id=reviewer_id, limit=limit)
    except Exception as e:  # store failure (R3): honest 503, never an empty queue
        raise _store_unavailable("pending-queue read", e) from e
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

    The authenticated operator is recorded as the resolver: ``reviewer_name``
    (their profile name, else their email, else their id) and
    ``reviewer_email`` are written with the resolution, and ``resolved_at`` is
    stamped for BOTH statuses (migration 136). ``reviewer_id`` is left as the
    requester breadcrumb the gate wrote. An identity the token does not carry
    stays unrecorded.

    An ``approved`` resolution sets ``valid_from``/``valid_until``/``approved_at``
    inside ``submit_review`` (repo :169-173). A repo ``False`` is fail-closed —
    never a fabricated success. FIX B (codex HIGH): ``submit_review`` now returns
    False on a ZERO-ROW update (nonexistent / already-resolved review_id), so we
    surface that as 404 (the honest 'not found / not resolvable' code), not a
    fabricated 200. A genuine persistence error also returns False -> 404, which
    is still a correct non-200 (never a fake success); the repo logs the
    distinction (zero-row WARNING vs exception ERROR).
    """
    # The resolver's identity, from the verified token (dependencies/auth.py
    # builds ``id`` / ``email`` / ``user_metadata`` from the Supabase user).
    # Unknown stays unknown: when the token carries none of them, pass None.
    reviewer_name = (
        (user.get("user_metadata") or {}).get("name") or user.get("email") or user.get("id")
    )
    reviewer_email = user.get("email")
    repo = await _get_expert_review_repo()
    success = await repo.submit_review(
        review_id=review_id,
        approval_status=request.approval_status,
        checklist=request.checklist,
        comments=request.comments,
        concerns_raised=request.concerns_raised,
        conditions=request.conditions,
        validity_days=request.validity_days,
        reviewer_name=reviewer_name or None,
        reviewer_email=reviewer_email or None,
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

    # #1993: serialise the build per review id across workers. A request that
    # WAITED re-reads the row under the lock: a stored assessment that is new
    # since it first looked (absent then, or changed under ``force``) is the
    # winner's fresh result and is replayed as cached=True even under ``force``
    # (the winner just regenerated it). An unchanged row (the winner failed to
    # persist, or died) means we build ourselves: at most one extra build.
    async with _ASSESSMENT_LOCK.hold(review_id) as lease:
        if lease.waited:
            regenerated = await _reread_assessment(repo, review_id)
            if regenerated is not None and regenerated != cached:
                logger.info(
                    f"Expert-review {review_id}: assessment waited on an in-flight build "
                    f"(lock mode={lease.mode}); replaying the winner's stored result"
                )
                return AgentAssessmentResponse(
                    review_id=review_id, assessment=regenerated, cached=True, persisted=True
                )
            logger.info(
                f"Expert-review {review_id}: assessment waited on an in-flight build "
                f"(lock mode={lease.mode}) but no new stored result appeared (the winner "
                "failed to persist or died); building"
            )
        validation_ids = review.get("related_validation_ids") or []
        validations = await _get_validation_rows(validation_ids)
        # run_signature is a BLOCKING LM call; keep the event loop free.
        started = time.monotonic()
        assessment = await asyncio.to_thread(_build_assessment, review, validations)
        elapsed = time.monotonic() - started
        persisted = await repo.update_agent_assessment(review_id, assessment)
        # No p99 exists for this build anywhere; record the elapsed seconds so the
        # lock TTL (120 s, nginx proxy_read_timeout) can be revisited on data.
        logger.info(
            f"Expert-review {review_id}: assessment built in {elapsed:.1f}s "
            f"(lock mode={lease.mode}, force={force}, persisted={persisted})"
        )
    return AgentAssessmentResponse(
        review_id=review_id, assessment=assessment, cached=False, persisted=persisted
    )


async def _reread_assessment(
    repo: "ExpertReviewRepository", review_id: str
) -> Optional[Dict[str, Any]]:
    """The stored assessment as it is NOW (after waiting on the in-flight lock).
    A failed re-read is not a reason to fail the request: log it and let the
    caller build, exactly as if nothing had been stored."""
    try:
        row = await repo.get_by_id(review_id)
    except Exception as e:
        logger.warning(f"Expert-review {review_id}: post-wait re-read failed, building: {e}")
        return None
    return _as_json_object((row or {}).get("agent_assessment_json"))


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
    """Return status counts (pending/approved/rejected/expired/expiring_soon).

    A store failure is 503 (R3) -- never all-zero counts with a 200.
    """
    repo = await _get_expert_review_repo()
    try:
        summary = await repo.get_review_summary(brand=brand)
    except Exception as e:  # store failure (R3): honest 503, never zero counts
        raise _store_unavailable("summary read", e) from e
    return ReviewSummaryResponse(
        pending=summary.get("pending", 0),
        approved=summary.get("approved", 0),
        rejected=summary.get("rejected", 0),
        expired=summary.get("expired", 0),
        expiring_soon=summary.get("expiring_soon", 0),
    )


@router.get(
    "/{review_id}",
    response_model=ExpertReviewDetailResponse,
    summary="One expert review (any status) with its same-structure history",
    operation_id="get_expert_review",
    responses={
        404: {"model": ErrorResponse, "description": "Review not found"},
        503: {"model": ErrorResponse, "description": "Expert-review store unavailable"},
    },
)
async def get_expert_review(
    review_id: str,
    user: Dict[str, Any] = Depends(require_operator),
) -> ExpertReviewDetailResponse:
    """Return one review row in any status plus every review of the same DAG structure.

    Powers the linked-review card the causal drill-down deep-links to
    (``/expert-reviews?review=<id>``), so a run whose structure is pending,
    approved or rejected always resolves to its record. ``history`` is the full
    same-hash (and same-brand) list, newest first, expired included -- the read
    ``ExpertReviewGate.check_rejection`` performs. Declared LAST in this module
    so it cannot shadow ``/pending`` and ``/summary``.
    """
    # A malformed id is 404, not 503 (review 2026-09-09, measured live):
    # ``expert_reviews.review_id`` is a uuid column, so a non-UUID string makes
    # PostgREST raise APIError 22P02 ("invalid input syntax for type uuid"),
    # which the store-failure guard below would report as an outage with an
    # ERROR traceback -- while the sibling ``POST /{review_id}/resolve``
    # answers 404 for the same input. Pre-check for parity, and hand the store
    # the CANONICAL form so any form Python accepts (uppercase, braces) can
    # never trip the cast. The raw path value stays in the 404 messages.
    try:
        canonical_id = str(uuid.UUID(review_id))
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Review {review_id} was not found (not a valid review id)."
        ) from None
    try:
        # The client factory raises ServiceConnectionError when Supabase is
        # unset/unreachable; inside the try so that is a 503 as well, not a
        # 500 (pre-execution review 2026-09-08, codex MED).
        repo = await _get_expert_review_repo()
        row = await repo.get_by_id(canonical_id)
    except Exception as e:  # store failure (R3): honest 503
        raise _store_unavailable("review read", e) from e
    if not row:
        raise HTTPException(status_code=404, detail=f"Review {review_id} was not found.")
    dag_hash = row.get("dag_version_hash")
    history_rows: List[Dict[str, Any]] = []
    if dag_hash:
        try:
            history_rows = await repo.get_reviews_for_dag(
                dag_hash, include_expired=True, brand=row.get("brand")
            )
        except Exception as e:
            raise _store_unavailable("review history read", e) from e
    return ExpertReviewDetailResponse(
        review=ReviewRecord.model_validate(row),
        history=[ReviewRecord.model_validate(r) for r in history_rows],
    )
