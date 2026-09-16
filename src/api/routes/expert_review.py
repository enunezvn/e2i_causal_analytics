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
- GET  /expert-reviews/{review_id}        -> one review (any status) + estimand history + versions

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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ValidationError

from src.api.dependencies.auth import require_operator
from src.api.dependencies.inflight_lock import InflightLock
from src.api.errors import user_safe_503_detail
from src.api.schemas.errors import ErrorResponse, ValidationErrorResponse
from src.api.schemas.expert_review import (
    AgentAssessmentResponse,
    DagChanges,
    ExpertReviewDetailResponse,
    PendingReviewItem,
    PendingReviewsResponse,
    ResolveReviewRequest,
    ResolveReviewResponse,
    ReviewRecord,
    ReviewSummaryResponse,
    ReviewVersion,
    parse_json_column,
)
from src.causal_engine.dag_hash import effective_adjustment_hash, get_dag_changes

if TYPE_CHECKING:
    from src.repositories.expert_review import ExpertReviewRepository

logger = logging.getLogger(__name__)

# R3: the repository re-raises store errors instead of returning [] / zeros
# (a "0 pending" page with HTTP 200 during an outage is a plausible-wrong
# value). The app's catch-all classifies exceptions by message keyword
# ("connection"/"unavailable" -> 503, otherwise a generic 500), which is not a
# contract, so these routes map a store failure themselves with the existing
# HTTPException(503) pattern (routes/causal/catalog.py, routes/digital_twin.py).
# The app's StarletteHTTPException handler MASKS a 503 detail unless it is marked
# with errors.user_safe_503_detail(); this one names no internals, so it is
# marked and reaches the client as the response ``message``. The handler
# forwards HTTPException headers (#1999), so the 503 sends the ``Retry-After``
# its DependencyError body already promises ("try again in 30 seconds").
_STORE_UNAVAILABLE_DETAIL = "Expert-review store unavailable. Retry shortly."
_STORE_UNAVAILABLE_RETRY_AFTER_SECONDS = 30

# #1993: one LLM build per review id across BOTH gunicorn workers. Redis
# ``SET NX PX`` with a process-local fallback. The TTL (120 s) tracks nginx's
# ``proxy_read_timeout 120s`` for /api/ (docker/nginx/nginx.secure.conf:227),
# the longest a caller waits on a build; see dependencies/inflight_lock.py for
# the degradation and wait bounds and what happens when a build outlives it.
_ASSESSMENT_LOCK = InflightLock("expert_review:assessment:inflight")

# Strong references to in-flight build tasks: asyncio keeps only weak references
# to tasks, and a build whose request was cancelled is awaited by nobody
# (precedent: middleware/activity_tracking.py flush tasks).
_INFLIGHT_BUILDS: "set[asyncio.Task[Any]]" = set()

_BUILD_IN_PROGRESS_DETAIL = (
    "An assessment build for this review is still in progress; retry shortly."
)

# Codex round-3 MED-B: the replay signal is a per-build generation id stamped
# INTO the persisted payload (assessment-specific, cross-worker, survives a
# byte-identical regeneration). ``updated_at`` was rejected as the signal:
# unrelated writes (submit_review/resolve, update_dag_structure) move it, so a
# forced request that snapshotted the OLD assessment, lost the race to such a
# write and then took an idle lock would replay the OLD value and the requested
# regeneration would never run. Extra keys are tolerated end to end:
# ``AgentAssessmentResponse.assessment`` and ``agent_assessment_json`` are open
# dicts, and the frontend reads only items/is_fallback/evidence (ResolveForm.tsx,
# PrepareAssessmentsButton.tsx; no Zod parse on this response).
_GENERATION_ID_KEY = "generation_id"
_GENERATED_AT_KEY = "generated_at"

# Codex round-3 MED-A: bounded re-read retry before the honest 503 (never build
# from the stale snapshot over a result persisted meanwhile).
_REREAD_ATTEMPTS = 2
_REREAD_BACKOFF_SECONDS = 0.2


def _store_unavailable(operation: str, exc: Exception) -> HTTPException:
    logger.error(f"Expert-review {operation} failed: {exc}", exc_info=True)
    return HTTPException(
        status_code=503,
        detail=user_safe_503_detail(_STORE_UNAVAILABLE_DETAIL),
        headers={"Retry-After": str(_STORE_UNAVAILABLE_RETRY_AFTER_SECONDS)},
    )


_ReviewModelT = TypeVar("_ReviewModelT", bound=BaseModel)


def _validate_review_row(model: Type[_ReviewModelT], row: Dict[str, Any]) -> _ReviewModelT:
    """``model.model_validate(row)``, converting a schema violation in the
    STORED row (#1991 debt 4: a snapshot with an out-of-vocabulary
    ``discovery_gate_decision``, or any other field the typed schema now
    rejects) into an honest 500 naming the review id and the failing field --
    never an unhandled ``ValidationError`` (a bare 500 with no detail), and
    never the ``_store_unavailable`` 503 (the store answered fine; the ROW
    it returned is malformed)."""
    try:
        return model.model_validate(row)
    except ValidationError as exc:
        review_id = row.get("review_id") or row.get("id")
        raise HTTPException(
            status_code=500,
            detail=(
                f"expert review {review_id} has a malformed dag_structure_json: "
                f"{exc.errors()[0].get('loc')}"
            ),
        ) from exc


def _snapshot_or_500(raw: Any, review_id: Any) -> Optional[Dict[str, Any]]:
    """A stored ``dag_structure_json`` as a dict, ``None`` when the column is NULL.

    The raw value is sometimes a ``json.dumps`` string (the repo's write path),
    so it goes through ``parse_json_column`` -- the schema's single definition of
    how this column is read.

    ``expert_reviews.dag_structure_json`` carries NO CHECK constraint (migration
    137 added none; only the versions table's column is constrained to
    object-or-NULL), so a stored ARRAY or number reaches here intact and
    ``get_dag_changes`` would call ``.get`` on it -- an ``AttributeError``, which
    surfaces as a bare 500 with no detail. Raise the same honest 500
    ``_validate_review_row`` raises instead: the store answered fine, the ROW is
    malformed, and the message names the row and the column so an operator can
    find it.
    """
    if raw is None:
        return None
    parsed = parse_json_column(raw)
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=500,
            detail=(
                f"expert review {review_id} has a malformed dag_structure_json: "
                "expected a JSON object"
            ),
        )
    return parsed


def _changes_between(
    prev_snapshot: Any,
    cur_snapshot: Any,
    *,
    prev_id: Any = None,
    cur_id: Any = None,
) -> DagChanges:
    """The structural delta between two stored snapshots (#1991 debt 3).

    A NULL snapshot diffs as ``{}``, so the delta against "no recorded
    structure" is everything-added -- what the timeline should show, not a
    crash. Its HASH, though, is reported as None rather than the sha256 of the
    empty graph: that digest is a real-looking 64-character value matching no
    row in the database, and an operator comparing it to the version's stored
    ``dag_version_hash`` would be reading a fabricated one. No recorded
    structure has no hash.
    """
    prev = _snapshot_or_500(prev_snapshot, prev_id)
    cur = _snapshot_or_500(cur_snapshot, cur_id)
    changes = get_dag_changes(prev or {}, cur or {})
    if prev is None:
        changes["old_hash"] = None
    if cur is None:
        changes["new_hash"] = None
    return DagChanges(**changes)


def _current_version_index(
    entries: Sequence[Tuple[Optional[str], Optional[str], Any]],
    dag_version_hash: Optional[str],
    adjustment_set_hash: Optional[str],
    review_snapshot: Any,
) -> Optional[int]:
    """Which timeline entry is the review's CURRENT version (codex round 3).

    ``expert_review_versions`` is a TIMELINE of facts; the review row is the
    authority on which structure it covers, and the two can disagree. A run
    that recorded its version and then LOST the compare-and-set advance leaves
    its row after the winner's (timeline ``... C, B`` while the review is on
    C), so "the last entry" is not "the version under review" -- rendering its
    delta would show the reviewer a change they are not being asked to approve,
    and reporting its timestamp would date a move that never happened.

    The review's identity is the PAIR: ``compute_dag_hash`` excludes adjustment
    sets, so the DAG hash alone cannot see a covariate-only advance. Among the
    entries carrying the review's hash we therefore take the LAST whose
    covariate set is equal as well. The LAST, because a revert A -> B -> A
    appends A a second time and the review then points at the NEWER A -- whose
    ``changes`` is the B -> A delta the reviewer is actually being shown.

    THE RULE: compare the EFFECTIVE adjustment on BOTH sides (codex round 5).
    ``effective_adjustment_hash`` reads the stored hash when there is one and
    otherwise derives it from the snapshot beside it, so each side is asked the
    same question -- "which covariate set does this actually name?" -- rather
    than one side being taken at face value and the other cross-examined. Two
    sides are equal when their effective values are, and None == None matches
    only when BOTH are genuinely unknown: unknown matches unknown, it is not a
    wildcard.

    Why EFFECTIVE and not the raw columns: a NULL adjustment half means "not
    recorded", NOT "conditions on nothing" -- the canonical empty set is a real
    hash, ``sha256("[]")``. Migration 141 backfilled every pre-existing version
    with a NULL hash while COPYING its snapshot, so on those rows the snapshot
    is the only place the covariate set is written down. Reading NULL as a value
    equal to another NULL therefore matched rows that name different structures.

    This one rule subsumes the round-4 fallback (a review that has learned its
    hash still resolves to the backfilled row whose snapshot derives it) and
    closes round 5's finding 1 from the other side: a review whose OWN adjustment
    column is NULL but whose snapshot names A is not unknown, so an
    information-free ``(h, NULL, NULL)`` row -- which a structureless run used to
    record -- no longer matches it. Naming that row would render "the previous
    snapshot disappeared" beside the reviewer's real graph A, and would date the
    queue's "last changed" to a move that never happened. It also works in the
    direction the fallback could not: a NULL-adjustment review whose snapshot
    derives C matches a row that states C outright.

    A LEGITIMATE no-structure recording still matches: when the review carries
    neither a hash nor a snapshot and the row carries neither either, both
    effective values are None and the row IS the review's version.

    An EMPTY ``dag_version_hash`` is treated as absent, like None: the column
    the entries carry is NOT NULL and non-empty, so no entry could match it
    anyway, and "" is the review row saying nothing about its structure.

    None when nothing matches: a review minted before the versions table, or an
    old-image mint whose first version was never recorded. Callers show no delta
    and no change date rather than guess.

    Takes normalized ``(dag_version_hash, adjustment_set_hash, snapshot)``
    triples plus the REVIEW's own snapshot, not the entries themselves, so the
    ONE definition of this rule serves both the detail route (validated
    ``ReviewVersion`` models) and the pending queue (raw stored rows, which it
    deliberately never validates -- it reads one timestamp off them, and a
    malformed row must not 500 the whole queue; a snapshot that is not a dict,
    or whose adjustment data is malformed, simply proves nothing).
    """
    if not dag_version_hash:
        return None
    wanted = effective_adjustment_hash(adjustment_set_hash, review_snapshot)
    for index in reversed(range(len(entries))):
        entry_hash, entry_adjustment, snapshot = entries[index]
        if entry_hash != dag_version_hash:
            continue
        if effective_adjustment_hash(entry_adjustment, snapshot) == wanted:
            return index
    return None


def _current_version_id(
    versions: List[ReviewVersion],
    dag_version_hash: Optional[str],
    adjustment_set_hash: Optional[str],
    review_snapshot: Any,
) -> Optional[str]:
    """The ``version_id`` of the review's current version, or None.

    See ``_current_version_index`` for the selection rule.
    """
    index = _current_version_index(
        [
            (
                v.dag_version_hash,
                v.adjustment_set_hash,
                # The VALIDATED snapshot as a plain dict, so the shared
                # derivation reads it exactly as it reads the queue's raw
                # column. ``DagStructureSnapshot`` keeps unknown keys
                # (extra="allow"), so nothing is lost on the way through.
                None if v.dag_structure_json is None else v.dag_structure_json.model_dump(),
            )
            for v in versions
        ],
        dag_version_hash,
        adjustment_set_hash,
        review_snapshot,
    )
    return None if index is None else versions[index].version_id


def _last_changed_at(row: Dict[str, Any], versions: List[Dict[str, Any]]) -> Any:
    """When the structure the review is CURRENTLY on was recorded.

    RAW stored version rows (the pending queue never validates them), so the
    pair is read with ``.get``. Falls back to the review's own ``created_at``
    when nothing matches -- including the empty timeline of a review minted
    before migration 141.

    Every snapshot the rule weighs -- the version rows' AND the review's own,
    which is what names its covariate set when its adjustment column is NULL --
    goes through ``parse_json_column``, the module's single definition of how
    this column is read, and what the detail route's validated model applies too, so
    the two callers read the same stored row the same way. The detail
    additionally validates it and turns a malformed snapshot into a named 500
    rather than a derivation; migration 141's object-or-NULL CHECK makes that
    unreachable for rows this table can hold. It only parses; it validates
    nothing and cannot raise, and a value that is not an object comes back as
    something the derivation reads as "proves nothing".
    """
    index = _current_version_index(
        [
            (
                v.get("dag_version_hash"),
                v.get("adjustment_set_hash"),
                parse_json_column(v.get("dag_structure_json")),
            )
            for v in versions
        ],
        row.get("dag_version_hash"),
        row.get("adjustment_set_hash"),
        parse_json_column(row.get("dag_structure_json")),
    )
    return row.get("created_at") if index is None else versions[index].get("created_at")


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

    ``version_count`` / ``last_changed_at`` (#1991 debt 3) come from ONE batched
    ``expert_review_versions`` read, so an operator can see that a pending
    review's structure moved under them since it was queued. The count is every
    recorded version; the date is the CURRENT version's, never the timeline's
    last, which can belong to a version the review is not on.
    """
    repo = await _get_expert_review_repo()
    try:
        rows = await repo.get_pending_reviews(brand=brand, reviewer_id=reviewer_id, limit=limit)
    except Exception as e:  # store failure (R3): honest 503, never an empty queue
        raise _store_unavailable("pending-queue read", e) from e
    # ONE batched versions read for the whole page (#1991 debt 3): the queue
    # serves up to 200 rows, and a per-row get_versions would be 200 round trips.
    try:
        versions_by_review = await repo.get_versions_for_reviews(
            [row["review_id"] for row in rows if row.get("review_id")]
        )
    except Exception as e:  # store failure (R3): honest 503, never "never changed"
        raise _store_unavailable("pending-queue versions read", e) from e
    reviews = []
    for row in rows:
        # The batched read keys its mapping by ``str(review_id)`` (``get_versions_for_reviews``),
        # so a row whose id is missing or not a string can never match a key -- the
        # empty list is the same answer the lookup would give, said without asking.
        review_id = row.get("review_id")
        versions = versions_by_review.get(review_id, []) if isinstance(review_id, str) else []
        reviews.append(
            _validate_review_row(
                PendingReviewItem,
                {
                    **row,
                    # No version rows means the review was minted before the
                    # versions table (mig 141) -- it is at version 1, changed
                    # when it was created. Never 0: that reads as "no structure".
                    "version_count": len(versions) or 1,
                    # WHEN the structure under review landed -- the CURRENT
                    # version's timestamp, not the timeline's last (codex round
                    # 3). The tail can belong to a run that recorded its version
                    # and then lost the compare-and-set advance, and its
                    # timestamp is the NEWEST: reporting it would tell the
                    # operator their review moved at a moment it did not. When
                    # no recorded version carries the review's pair the date is
                    # unknown, and the review's own creation is the honest
                    # answer -- exactly as for a review with no timeline at all.
                    # ``version_count`` stays len(versions): the rows WERE
                    # recorded, orphan included.
                    "last_changed_at": _last_changed_at(row, versions),
                },
            )
        )
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

    Version binding (codex rounds 1 and 2): the PAIR
    ``(request.dag_version_hash, request.adjustment_set_hash)`` is the structure
    the reviewer's form displayed, and ``submit_review`` filters the UPDATE on
    BOTH, so a review a concurrent run advanced (migration 141) is NOT resolved
    by a form opened on the old version. Both halves are needed because
    ``compute_dag_hash`` excludes adjustment sets: an ADJUSTMENT-ONLY advance
    leaves the hash equal, and the hash-only filter let a reviewer sign off
    covariates they were never shown. The repo keeps its boolean; a False is
    disambiguated HERE by ONE extra read: still pending on a DIFFERENT pair ->
    409 (reload and resolve the current version), anything else -> the existing
    404. The read is only on the failure path, so the happy path still costs one
    write. A re-read that itself fails is treated as the 404 case -- fail-closed,
    never a fabricated 200.
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
        expected_dag_version_hash=request.dag_version_hash,
        expected_adjustment_set_hash=request.adjustment_set_hash,
    )
    if not success:
        current = await _current_review_row(repo, review_id)
        # EITHER half having moved is an advance (codex round-2 HIGH 1): an
        # adjustment-only advance leaves the DAG hash equal, so comparing it
        # alone reported 404 "already resolved" for a row that is pending on a
        # structure this form never displayed. The adjustment comparison is
        # NULL-aware by construction -- both sides are Optional[str] and None ==
        # None is the "still unknown" case, which is a match, not a mismatch.
        if (
            current is not None
            and current.get("approval_status") == "pending"
            and (
                current.get("dag_version_hash") != request.dag_version_hash
                or current.get("adjustment_set_hash") != request.adjustment_set_hash
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Review {review_id} has advanced to a new structure version since this "
                    "form was opened; reload the review and resolve the current version."
                ),
            )
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


async def _current_review_row(
    repo: "ExpertReviewRepository", review_id: str
) -> Optional[Dict[str, Any]]:
    """The review row as it stands, or None -- the failure-path read that tells a
    STALE resolution (409) from a gone/resolved one (404).

    A read that raises answers None: the caller then reports the conservative
    404 rather than propagating a store outage as a 500 on a request whose write
    already did nothing. The distinction is only ever used to pick an error code.
    """
    try:
        return await repo.get_by_id(review_id)
    except Exception as read_err:  # noqa: BLE001 - only the error CODE depends on this
        logger.warning(
            f"Resolve of review {review_id} failed and the disambiguating re-read also "
            f"failed ({read_err}); reporting 404 rather than guessing 409"
        )
        return None


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

    Concurrency (#1993): one build per review id across workers (Redis in-flight
    lock, ``dependencies/inflight_lock.py``). A request that waited replays the
    winner's stored result as ``cached=True``; a request whose bounded wait (the
    TTL) is exhausted answers 409 with ``Retry-After`` instead of building
    unlocked. The build runs in its own shielded task, so a cancelled request
    (client gone, nginx 504) still persists and releases the lock in order.
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

    # Codex HIGH 1: the build runs in its OWN task and the request only shields
    # it. Cancelling ``await asyncio.to_thread(...)`` does not stop the builder
    # thread, so an unshielded request that was cancelled would release the lock
    # while the thread kept building, and a second caller would build
    # concurrently. Shielded, the task finishes the persist and releases at the
    # right moment; waiters then replay its result. (#1999 measured that a
    # client disconnect alone does NOT cancel the request task on this stack;
    # uvicorn cancels request tasks only past a ``timeout_graceful_shutdown``,
    # which the UvicornWorker does not set. The shield covers any such path.)
    task = asyncio.create_task(
        _build_under_lock(repo, review_id, review, cached, force),
        name=f"expert-review-assessment:{review_id}",
    )
    _INFLIGHT_BUILDS.add(task)
    task.add_done_callback(_INFLIGHT_BUILDS.discard)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if not task.done():
            logger.info(
                f"Expert-review {review_id}: request cancelled mid-build; the build "
                "continues so its result is persisted and the lock released in order"
            )
            task.add_done_callback(_log_orphaned_build)
        raise


async def _build_under_lock(
    repo: "ExpertReviewRepository",
    review_id: str,
    review: Dict[str, Any],
    cached: Optional[Dict[str, Any]],
    force: bool,
) -> AgentAssessmentResponse:
    """The lock-guarded half of ``generate_review_assessment`` (its own task; see there).

    ``review``/``cached`` are the request's INITIAL snapshot, read outside the
    lock. After EVERY acquisition the row is re-read under the lock and judged
    against that snapshot (codex round-2 HIGH: a request that snapshotted
    before another's acquire and acquired only after its release would see
    ``waited=False`` and build again). A stored assessment whose
    ``generation_id`` differs from the snapshot's (or any stored assessment when
    the snapshot had none) is another request's finished build and is replayed
    as ``cached=True`` regardless of ``force``. Every build stamps the payload
    with a fresh ``generation_id``/``generated_at`` before persisting, so a
    byte-identical forced regeneration (deterministic fallbacks exist) is still
    recognised, while two legacy payloads without the key compare as NOT new
    and a forced request on a pre-stamp row builds.

    A failed re-read after the acquire is retried once and then answered with
    the SAFE 503 (``_reread_row``): building from the stale snapshot could
    overwrite a result persisted meanwhile.

    The persist is bound to the STRUCTURE VERSION the build graded (codex
    round-1 HIGH): the row's ``dag_version_hash`` is captured before the build
    and passed to ``update_agent_assessment``, so a build that finishes after a
    refutation run advanced the review (migration 141) writes nothing and the
    response says ``persisted: false`` rather than filing a grading of the old
    DAG and the old evidence under the new structure. The advance itself already
    CLEARS the cached assessment (``append_version``), so the request-level cache
    short-circuit above cannot serve the previous structure's grading either.

    Worker shutdown (#1999, measured): uvicorn waits for every request task
    before it sends the lifespan shutdown, and the request task awaits this one
    (a disconnect does not cancel it), so a graceful stop (SIGTERM, a
    ``--max-requests`` recycle) lets the build persist before Redis/Supabase are
    closed; test_expert_review_shutdown_ordering_1999.py pins it. The api's
    compose ``stop_grace_period: 35s`` lets a deploy recreate wait out gunicorn's
    ``--graceful-timeout 30`` (docker's 10 s default killed it first). What still
    loses a build is a KILL: a build still running at gunicorn's 30 s graceful
    timeout, or a ``--max-requests`` recycle blocked behind OTHER in-flight
    requests, which stops heartbeating and can be aborted by the arbiter's worker
    timeout (from source, not measured). Then the key clears at its TTL and
    nothing is corrupted.
    """
    async with _ASSESSMENT_LOCK.hold(review_id) as lease:
        if lease.mode == "none":
            # Codex HIGH 2: the bounded wait (the TTL) is exhausted. An unlocked
            # build could overlap a legitimate holder, and a client that waited
            # the full TTL has already received nginx's 504, so it would serve
            # nobody. Holds in EITHER mode (a local holder exceeding the bound
            # is the same case); Redis being DOWN never lands here, ``hold``
            # degrades that to the process-local lock and yields normally.
            logger.warning(f"Expert-review {review_id}: in-flight lock wait exhausted; 409")
            raise HTTPException(
                status_code=409,
                detail=_BUILD_IN_PROGRESS_DETAIL,
                headers={"Retry-After": "5"},
            )
        row = await _reread_row(repo, review_id)  # SAFE 503 on failure; lease released
        stored = _as_json_object(row.get("agent_assessment_json"))
        if stored is not None and (
            cached is None or stored.get(_GENERATION_ID_KEY) != cached.get(_GENERATION_ID_KEY)
        ):
            logger.info(
                f"Expert-review {review_id}: a build finished between this request's read "
                f"and its lock (waited={lease.waited}, lock mode={lease.mode}); replaying "
                "the stored result"
            )
            return AgentAssessmentResponse(
                review_id=review_id, assessment=stored, cached=True, persisted=True
            )
        if lease.waited:
            logger.info(
                f"Expert-review {review_id}: assessment waited on an in-flight build "
                f"(lock mode={lease.mode}) but no new stored result appeared (the winner "
                "failed to persist or died); building"
            )
        # Build from the row as it is NOW (the snapshot can be up to a full wait old).
        source = row or review
        # The structure this build GRADES, captured before it starts (codex
        # round-1 HIGH). A refutation run can append a version and advance the
        # review while the (blocking, LM-backed) build runs; persisting then
        # would file a grading of h1's DAG and h1's evidence under h2. The
        # persist below carries this hash as a filter, so a stale build is
        # refused by the row rather than by a re-read that could itself race.
        source_hash = source.get("dag_version_hash")
        # The adjustment half of that same captured version (codex round-2 HIGH
        # 2). Without it the guard missed an ADJUSTMENT-ONLY advance: the DAG
        # hash never moves, so a build that graded (h1, adj-W) still persisted
        # after the advance to (h1, adj-Z) had cleared the cache, and the review
        # UI showed a grading of covariates the review no longer covers, marked
        # persisted=true.
        source_adj = source.get("adjustment_set_hash")
        validation_ids = source.get("related_validation_ids") or []
        validations = await _get_validation_rows(validation_ids)
        # run_signature is a BLOCKING LM call; keep the event loop free.
        started = time.monotonic()
        assessment = await asyncio.to_thread(_build_assessment, source, validations)
        elapsed = time.monotonic() - started
        assessment = {
            **assessment,
            _GENERATION_ID_KEY: uuid.uuid4().hex,
            _GENERATED_AT_KEY: datetime.now(timezone.utc).isoformat(),
        }
        # A None ``source_hash`` OMITS the filter rather than matching NULL: the
        # column is nullable (a pre-141 row that never carried a hash), and such
        # a row has no version to bind to, so filtering on it would refuse every
        # write instead of guarding one. That row gets the pre-#1991 behaviour.
        # The two halves travel together: the repository applies the adjustment
        # filter only when the hash one is given, so a NULL ``source_adj`` under
        # a real ``source_hash`` still guards -- it asserts "the review's
        # adjustment set was unknown when this build started", which is a
        # precondition, not an absence of one.
        persisted = await repo.update_agent_assessment(
            review_id,
            assessment,
            for_dag_version_hash=source_hash,
            for_adjustment_set_hash=source_adj,
        )
        # No p99 exists for this build anywhere; record the elapsed seconds so the
        # lock TTL (120 s, nginx proxy_read_timeout) can be revisited on data.
        logger.info(
            f"Expert-review {review_id}: assessment built in {elapsed:.1f}s "
            f"(lock mode={lease.mode}, force={force}, persisted={persisted})"
        )
    return AgentAssessmentResponse(
        review_id=review_id, assessment=assessment, cached=False, persisted=persisted
    )


async def _reread_row(repo: "ExpertReviewRepository", review_id: str) -> Dict[str, Any]:
    """The review row as it is NOW, read under the lock (a vanished row reads as ``{}``).

    Codex round-3 MED-A: a failed re-read must NOT fall back to building from
    the initial snapshot, which could overwrite a result persisted meanwhile.
    Retry once after a short backoff; if the store still fails, raise the same
    SAFE 503 the detail route uses for a failed row read -- the caller's
    ``async with`` releases the lease, and nothing is built.
    """
    failure: Exception = RuntimeError("re-read not attempted")
    for attempt in range(1, _REREAD_ATTEMPTS + 1):
        try:
            return await repo.get_by_id(review_id) or {}
        except Exception as e:
            failure = e
            logger.warning(
                f"Expert-review {review_id}: post-acquire re-read failed "
                f"(attempt {attempt}/{_REREAD_ATTEMPTS}): {e}"
            )
            if attempt < _REREAD_ATTEMPTS:
                await asyncio.sleep(_REREAD_BACKOFF_SECONDS)
    raise _store_unavailable("assessment re-read", failure)


def _log_orphaned_build(task: "asyncio.Task[Any]") -> None:
    """Done-callback for a build whose request was cancelled (nobody awaits the
    task any more, so its outcome would otherwise be silent)."""
    if task.cancelled():
        logger.warning(f"{task.get_name()}: orphaned build was cancelled; nothing persisted")
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(f"{task.get_name()}: orphaned build failed: {exc!r}")
    else:
        logger.info(
            f"{task.get_name()}: orphaned build completed after its client left; the "
            "result is persisted for the next caller"
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
    """Return status counts (pending/approved/rejected/superseded/expired/expiring_soon).

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
        # Partition member since migration 140 (BLOCK-band reviews resolved to
        # ``superseded``); a resolution, never a queue item.
        superseded=summary.get("superseded", 0),
        expired=summary.get("expired", 0),
        expiring_soon=summary.get("expiring_soon", 0),
    )


@router.get(
    "/{review_id}",
    response_model=ExpertReviewDetailResponse,
    summary="One expert review (any status) with its estimand history and version timeline",
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
    """Return one review row in any status, its ESTIMAND history and its structure timeline.

    Powers the linked-review card the causal drill-down deep-links to
    (``/expert-reviews?review=<id>``), so a run whose structure is pending,
    approved or rejected always resolves to its record.

    ``history`` is every review of the same ESTIMAND (migration 140), newest
    first, expired included. A row minted before that migration carries no
    ``estimand_key`` and falls back to the same-hash (and same-brand) list --
    the read ``ExpertReviewGate.check_rejection`` performs.

    ``versions`` is this review's ``expert_review_versions`` timeline (migration
    141), oldest first, each row carrying the diff against the one before it.
    ``current_version_id`` names the entry the review row itself is on, which is
    not always the last one: a client renders THAT row's delta, never the
    tail's.

    Declared LAST in this module so it cannot shadow ``/pending`` and ``/summary``.
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
    estimand_key = row.get("estimand_key")
    dag_hash = row.get("dag_version_hash")
    history_rows: List[Dict[str, Any]] = []
    try:
        if estimand_key:
            # The estimand IS the review's identity since migration 140: a
            # structure change advances the SAME review, so the hash-keyed read
            # would drop every earlier structure of this question.
            history_rows = await repo.get_reviews_for_estimand(estimand_key)
        elif dag_hash:
            # Pre-140 rows carry no estimand_key; the same-hash history (the
            # read the gate's rejection probe performs) is all there is.
            history_rows = await repo.get_reviews_for_dag(
                dag_hash, include_expired=True, brand=row.get("brand")
            )
    except Exception as e:
        raise _store_unavailable("review history read", e) from e

    try:
        version_rows = await repo.get_versions(canonical_id)
    except Exception as e:  # store failure (R3): honest 503, never "never changed"
        raise _store_unavailable("review versions read", e) from e

    # Oldest-first timeline: each version diffs against the one before it, and
    # the FIRST carries None -- it has no predecessor, and an everything-added
    # diff there would read as a real structure change.
    versions: List[ReviewVersion] = []
    for index, version_row in enumerate(version_rows):
        changes = (
            None
            if index == 0
            else _changes_between(
                version_rows[index - 1].get("dag_structure_json"),
                version_row.get("dag_structure_json"),
                prev_id=version_rows[index - 1].get("review_id"),
                cur_id=version_row.get("review_id"),
            )
        )
        versions.append(_validate_review_row(ReviewVersion, {**version_row, "changes": changes}))

    # ``history`` is newest-first, so each row's predecessor is the NEXT element;
    # the last (oldest) review of the estimand has none.
    history: List[ReviewRecord] = []
    for index, history_row in enumerate(history_rows):
        changes_from_previous = (
            None
            if index + 1 >= len(history_rows)
            else _changes_between(
                history_rows[index + 1].get("dag_structure_json"),
                history_row.get("dag_structure_json"),
                prev_id=history_rows[index + 1].get("review_id"),
                cur_id=history_row.get("review_id"),
            )
        )
        history.append(
            _validate_review_row(
                ReviewRecord, {**history_row, "changes_from_previous": changes_from_previous}
            )
        )

    review = _validate_review_row(ReviewRecord, row)
    return ExpertReviewDetailResponse(
        review=review,
        history=history,
        versions=versions,
        # The review row -- not the tail of the timeline -- is the authority on
        # which version is under review; see ``_current_version_id``.
        current_version_id=_current_version_id(
            versions,
            review.dag_version_hash,
            review.adjustment_set_hash,
            # The review's OWN snapshot, as a plain dict: when its adjustment
            # column is NULL this is the only place its covariate set is
            # written down, and the rule needs the EFFECTIVE value on this side
            # too. ``model_dump()`` matches how the entries' snapshots are read.
            None if review.dag_structure_json is None else review.dag_structure_json.model_dump(),
        ),
    )
