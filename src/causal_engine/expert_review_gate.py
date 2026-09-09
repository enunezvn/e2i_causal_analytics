"""
Expert Review Gate for Causal DAG Validation.

Provides workflow gating based on domain expert approval status of causal DAGs.
Integrates with ExpertReviewRepository for persistence.

Version: 4.3
"""

import json
import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from src.repositories.expert_review import ExpertReviewRepository

logger = logging.getLogger(__name__)

# Keys of the in-state CausalGraph that are persisted with an auto-created
# review (mig 097). Bounded on purpose: enough to RENDER the DAG in the review
# UI and GROUND the advisory agent assessment — not the dag_dot blob or any
# other transient state.
_DAG_SNAPSHOT_KEYS = (
    "treatment_nodes",
    "outcome_nodes",
    "adjustment_sets",
    "confidence",
    "discovery_gate_decision",
    "dag_version_hash",
)


def sanitize_dag_structure(causal_graph: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Reduce a rich in-state CausalGraph to a JSON-serializable render snapshot.

    Keeps nodes/edges (edge tuples coerced to lists for JSONB round-trip) plus
    the treatment/outcome/adjustment/discovery keys; drops ``dag_dot`` and
    anything else. Returns None when there are no nodes — an empty snapshot
    would render as a blank graph posing as a real one.
    """
    if not causal_graph or not causal_graph.get("nodes"):
        return None
    structure: Dict[str, Any] = {
        "nodes": [str(n) for n in causal_graph.get("nodes", [])],
        "edges": [[str(e[0]), str(e[1])] for e in causal_graph.get("edges", [])],
    }
    if causal_graph.get("augmented_edges"):
        structure["augmented_edges"] = [
            [str(e[0]), str(e[1])] for e in causal_graph["augmented_edges"]
        ]
    for key in _DAG_SNAPSHOT_KEYS:
        if causal_graph.get(key) is not None:
            structure[key] = causal_graph[key]
    return structure


class ReviewGateDecision(Enum):
    """Expert review gate decisions.

    ``is_approved`` is True only for PROCEED / RENEWAL_REQUIRED (a real, active
    approval row). Every other value means "no usable approval", and they are
    deliberately distinct so a consumer never has to guess WHY:

    * PENDING_REVIEW -- a queue row exists; a human can resolve it
      (``POST /expert-reviews/{review_id}/resolve``).
    * REJECTED -- a human adjudicated this structure and turned it down.
      Durable: never re-queued on a later run (#1970); honoured on EVERY
      refutation band, PROCEED included (#1971).
    * BLOCKED -- no approval, no pending row, and no review could be queued.
    * UNAVAILABLE -- the gate could not be consulted at all (no repository /
      review store unreachable). Nothing was checked and nothing was queued.
      Replaces the old no-repository bypass that answered PROCEED with
      ``is_approved=True`` for a DAG nobody had looked at (#1969, #1971).
    """

    PROCEED = "proceed"  # DAG has active approval
    PENDING_REVIEW = "pending_review"  # Review request created, awaiting approval
    RENEWAL_REQUIRED = "renewal_required"  # Approval expiring soon, needs renewal
    REJECTED = "rejected"  # A human rejected this structure (durable verdict)
    BLOCKED = "blocked"  # No approval, no pending review, none could be created
    UNAVAILABLE = "unavailable"  # Gate could not be consulted; nothing checked/queued


# Upper bound on a rejection reason carried into caveats / error messages.
_REASON_MAX_CHARS = 300


def rejection_reason_from_row(row: Mapping[str, Any]) -> Optional[str]:
    """The reviewer's stated reason for a rejection, read from the review row.

    Precedence mirrors what ``ExpertReviewRepository.submit_review`` writes:
    ``concerns_raised`` (the specific concerns) first, then ``comments_json``
    (free-form notes; stored via ``json.dumps`` so it may come back as a JSON
    string -- a dict is rendered ``key: value``), then ``conditions``.
    Bounded so a caveat stays readable. ``None`` when the reviewer recorded
    nothing -- never a placeholder.
    """
    reason: Optional[str] = None

    concerns = row.get("concerns_raised")
    if isinstance(concerns, (list, tuple)):
        parts = [str(c).strip() for c in concerns if str(c).strip()]
        if parts:
            reason = "; ".join(parts)

    if reason is None:
        comments: Any = row.get("comments_json")
        if isinstance(comments, str) and comments.strip():
            try:
                comments = json.loads(comments)
            except ValueError:
                comments = comments.strip()
        if isinstance(comments, Mapping):
            parts = [f"{k}: {v}" for k, v in comments.items() if str(v).strip()]
            if parts:
                reason = "; ".join(parts)
        elif isinstance(comments, str) and comments.strip():
            reason = comments.strip()

    if reason is None:
        conditions = row.get("conditions")
        if isinstance(conditions, str) and conditions.strip():
            reason = conditions.strip()

    if reason is None:
        return None
    if len(reason) > _REASON_MAX_CHARS:
        reason = reason[: _REASON_MAX_CHARS - 3] + "..."
    return reason


@dataclass
class ReviewGateResult:
    """Result of expert review gate check."""

    decision: ReviewGateDecision
    dag_hash: str
    is_approved: bool
    review_id: Optional[str] = None
    approved_at: Optional[str] = None
    valid_until: Optional[str] = None
    days_until_expiry: Optional[int] = None
    reviewer_name: Optional[str] = None
    message: str = ""
    requires_action: bool = False
    # REJECTED only: the reviewer's recorded reason (see rejection_reason_from_row).
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "dag_hash": self.dag_hash,
            "is_approved": self.is_approved,
            "review_id": self.review_id,
            "approved_at": self.approved_at,
            "valid_until": self.valid_until,
            "days_until_expiry": self.days_until_expiry,
            "reviewer_name": self.reviewer_name,
            "message": self.message,
            "requires_action": self.requires_action,
            "rejection_reason": self.rejection_reason,
        }


class ExpertReviewGate:
    """
    Workflow gate for expert review of causal DAGs.

    Usage:
        gate = ExpertReviewGate(expert_review_repo)
        result = await gate.check_approval(dag_hash, brand)

        if result.decision == ReviewGateDecision.PROCEED:
            # Continue with causal analysis
            pass
        elif result.decision == ReviewGateDecision.PENDING_REVIEW:
            # Analysis blocked until review complete
            pass

    Live path (#1971): the causal_impact RefutationNode calls ``check_rejection``
    on every refutation band (read-only) and ``check_approval`` on REVIEW/BLOCK
    (queue-or-lookup). Whether a missing approval HALTS a run is the node's
    ``CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL`` switch; a human REJECTION always does.
    A gate without a repository answers UNAVAILABLE -- it never vouches for a
    DAG it could not look up.
    """

    # Default renewal warning threshold (days before expiry)
    RENEWAL_WARNING_DAYS = 14

    def __init__(
        self,
        repository: Optional[ExpertReviewRepository] = None,
        renewal_warning_days: int = RENEWAL_WARNING_DAYS,
        auto_create_review: bool = False,
    ):
        """
        Initialize expert review gate.

        Args:
            repository: ExpertReviewRepository instance
            renewal_warning_days: Days before expiry to warn about renewal
            auto_create_review: Whether to auto-create a `pending` review row for a new
                DAG. Defaults False (fail-closed): DEFERRED until the review-queue consumer
                + admin UI exist (R6-F2) so no orphan pending rows are created that no human
                can clear. Pass True explicitly once a human-in-the-loop consumer is wired.
        """
        self.repository = repository
        self.renewal_warning_days = renewal_warning_days
        self.auto_create_review = auto_create_review

    async def check_approval(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        requester_id: Optional[str] = None,
        analysis_context: Optional[str] = None,
        # Mapping (not Dict): callers pass the in-state CausalGraph TypedDict,
        # which mypy only accepts through a read-only Mapping.
        dag_structure: Optional[Mapping[str, Any]] = None,
        related_validation_ids: Optional[List[str]] = None,
    ) -> ReviewGateResult:
        """
        Check if a DAG has expert approval and is valid.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            brand: Brand context for filtering
            treatment: Treatment variable name
            outcome: Outcome variable name
            requester_id: User ID requesting the analysis
            analysis_context: Description of the analysis
            dag_structure: In-state CausalGraph (rich form accepted; sanitized
                via ``sanitize_dag_structure`` before persistence) so an
                auto-created review row is renderable in the review UI (097)
            related_validation_ids: causal_validations row ids from the
                refutation run that triggered this consult — links the
                review to its statistical evidence

        Returns:
            ReviewGateResult with decision and metadata
        """
        if not self.repository:
            # #1971: no repository means nothing CAN be checked. Say so. The
            # old answer here (PROCEED / is_approved=True, "development mode")
            # was exactly what a prod ServiceConnectionError degraded to, and
            # it told the refutation node an unreachable Supabase had cleared
            # the DAG (#1969). An honest "unavailable" lets the caller decide
            # (advisory: caveat; enforcing: halt).
            logger.warning(
                "No repository configured; expert review gate could not be consulted "
                "(decision: unavailable)"
            )
            return ReviewGateResult(
                decision=ReviewGateDecision.UNAVAILABLE,
                dag_hash=dag_hash,
                is_approved=False,
                message=(
                    "Expert review gate could not be consulted: no review repository is "
                    "configured (Supabase unavailable or unset). The DAG holds no approval "
                    "and was not queued for review."
                ),
                requires_action=True,
            )

        # Chronology first (#1971, codex iter-3): the MOST RECENT adjudication
        # of this structure wins. An approval that is still unexpired but OLDER
        # than a rejection (approve A, renew as B, reject B while A's 90 days
        # run) is superseded -- ``get_dag_approval`` would still return A, so
        # it must not be consulted on its own. The whole history is needed
        # for that ordering, expired approvals included.
        history = await self.repository.get_reviews_for_dag(
            dag_hash, include_expired=True, brand=brand
        )
        latest_verdict, reopened = self._latest_adjudication(history)
        superseded_by_rejection = (
            latest_verdict is not None and latest_verdict.get("approval_status") == "rejected"
        )

        # Check for active approval -- only if no newer rejection supersedes it.
        approval = (
            None
            if superseded_by_rejection
            else await self.repository.get_dag_approval(dag_hash, brand)
        )

        if approval:
            # DAG has active approval - check expiry
            valid_until = approval.get("valid_until")
            days_until_expiry = None

            if valid_until:
                try:
                    expiry_date = date.fromisoformat(valid_until)
                    days_until_expiry = (expiry_date - date.today()).days
                except (ValueError, TypeError):
                    pass

            # Check if renewal warning needed
            if days_until_expiry is not None and days_until_expiry <= self.renewal_warning_days:
                return ReviewGateResult(
                    decision=ReviewGateDecision.RENEWAL_REQUIRED,
                    dag_hash=dag_hash,
                    is_approved=True,
                    review_id=approval.get("review_id"),
                    approved_at=approval.get("approved_at"),
                    valid_until=valid_until,
                    days_until_expiry=days_until_expiry,
                    reviewer_name=approval.get("reviewer_name"),
                    message=f"DAG approval expiring in {days_until_expiry} days. Renewal required.",
                    requires_action=True,
                )

            # Approval is valid and not expiring soon
            return ReviewGateResult(
                decision=ReviewGateDecision.PROCEED,
                dag_hash=dag_hash,
                is_approved=True,
                review_id=approval.get("review_id"),
                approved_at=approval.get("approved_at"),
                valid_until=valid_until,
                days_until_expiry=days_until_expiry,
                reviewer_name=approval.get("reviewer_name"),
                message="DAG has active expert approval",
            )

        # A REJECTED verdict is durable (#1970). ``get_reviews_for_dag`` orders
        # created_at DESC, so if the most recent adjudication of this DAG is
        # 'rejected' a human already turned this structure down.
        # Auto-creating a fresh pending row on the next REVIEW/BLOCK band would
        # silently undo that decision. A NEWER approval or pending row wins:
        # ``reopened`` comes from the ``_latest_adjudication`` call above; a
        # genuinely newer pending row sets it and skips this block, reaching
        # the pending branch below. A reviewer who wants to re-open the
        # structure does so from the review UI, not by re-running. #1971
        # gives the verdict its own decision value
        # (REJECTED, not BLOCKED) so consumers can tell "a human said no" from
        # "nobody has looked yet and no row could be queued".
        if superseded_by_rejection and not reopened:
            assert latest_verdict is not None  # narrowed by superseded_by_rejection
            return self._rejection_result(latest_verdict, dag_hash)

        # No usable approval - check for pending review (a pending row NEWER
        # than a rejection is a reviewer re-opening the structure).
        pending = [r for r in history if r.get("approval_status") == "pending"]

        if pending:
            # Review already pending. Backfill-on-encounter (097): this
            # short-circuit is the ONLY consult a pre-097 (structure-less)
            # pending row will ever see for its DAG, so attach the renderable
            # snapshot now. Best-effort — a backfill failure must never break
            # the gate.
            pending_row = pending[0]
            review_id = pending_row.get("review_id")
            structure = sanitize_dag_structure(dag_structure)
            if structure and review_id and not pending_row.get("dag_structure_json"):
                try:
                    await self.repository.update_dag_structure(
                        str(review_id),
                        structure,
                        related_validation_ids=related_validation_ids,
                    )
                except Exception as backfill_err:  # noqa: BLE001 - gate must not break
                    logger.warning(
                        f"DAG-structure backfill failed for review {review_id}: {backfill_err}"
                    )
            return ReviewGateResult(
                decision=ReviewGateDecision.PENDING_REVIEW,
                dag_hash=dag_hash,
                is_approved=False,
                review_id=review_id,
                message="DAG review pending expert approval",
                requires_action=True,
            )

        # No approval and no pending review
        if self.auto_create_review and requester_id:
            # Auto-create review request.
            # KNOWN FOLLOW-UP (R6-F2, DEFERRED): the get_dag_approval +
            # get_reviews_for_dag pre-checks above guard against duplicate rows
            # within a single run, but two concurrent runs on the SAME dag_hash
            # can both pass the check and both INSERT, creating duplicate pending
            # rows. The robust fix is a DB UNIQUE constraint on
            # (dag_version_hash, approval_status='pending') — out of scope here
            # (needs a migration); tracked separately. No behavior change.
            review_id = await self.repository.create_review(
                reviewer_id=requester_id,
                # C1 (R6-F2): MUST be a valid ``expert_review_type`` ENUM member.
                # 'initial_dag' is NOT a member (valid: dag_approval,
                # methodology_review, quarterly_audit, ad_hoc_validation); with
                # auto_create_review=True the bad value fails the Postgres enum
                # cast -> create_review returns None -> gate falls to BLOCKED
                # (silent hard-block, zero rows). 'dag_approval' is the new-DAG
                # sign-off type (010 :53-58).
                review_type="dag_approval",
                dag_version_hash=dag_hash,
                brand=brand,
                treatment_variable=treatment,
                outcome_variable=outcome,
                analysis_context=analysis_context,
                dag_structure=sanitize_dag_structure(dag_structure),
                related_validation_ids=related_validation_ids,
            )

            if review_id:
                return ReviewGateResult(
                    decision=ReviewGateDecision.PENDING_REVIEW,
                    dag_hash=dag_hash,
                    is_approved=False,
                    review_id=review_id,
                    message="New DAG detected. Expert review request created.",
                    requires_action=True,
                )

        # Blocked - no approval and couldn't create review
        return ReviewGateResult(
            decision=ReviewGateDecision.BLOCKED,
            dag_hash=dag_hash,
            is_approved=False,
            message="DAG requires expert approval before analysis can proceed",
            requires_action=True,
        )

    @staticmethod
    def _latest_adjudication(
        history: List[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """``(most recent non-pending row, a pending row is newer than it)``.

        ``history`` is newest-first (``get_reviews_for_dag`` orders created_at
        DESC; rows are created and resolved in order because the unique-pending
        index (migration 062) allows one open review per structure at a time,
        so creation order is adjudication order). An exact ``created_at`` tie
        between a pending row and the adjudication after it is not a reopen
        (migration 134 reads it the same way). ``None`` when nothing has been
        adjudicated yet.
        """
        reopened = False
        for idx, row in enumerate(history):
            if row.get("approval_status") == "pending":
                # Tie-break (lane 1): a pending row that shares its created_at
                # with the adjudication that follows it is NOT newer than it --
                # the reading migration 134's strict ``>`` gives -- so the probe
                # and the promote can never disagree on a tie. Rows without a
                # timestamp keep the repository's order (unchanged behaviour).
                nxt = next(
                    (r for r in history[idx + 1 :] if r.get("approval_status") != "pending"),
                    None,
                )
                if (
                    nxt is not None
                    and row.get("created_at")
                    and row.get("created_at") == nxt.get("created_at")
                ):
                    continue
                reopened = True
                continue
            return row, reopened
        return None, reopened

    @staticmethod
    def _rejection_result(latest: Mapping[str, Any], dag_hash: str) -> ReviewGateResult:
        """The REJECTED result for the most recent (rejected) review row."""
        reviewer = latest.get("reviewer_name")
        reason = rejection_reason_from_row(latest)
        return ReviewGateResult(
            decision=ReviewGateDecision.REJECTED,
            dag_hash=dag_hash,
            is_approved=False,
            review_id=latest.get("review_id"),
            reviewer_name=reviewer,
            rejection_reason=reason,
            message=(
                "DAG structure was rejected by expert review"
                + (f" by {reviewer}" if reviewer else "")
                + (f": {reason}" if reason else "")
                + "; not re-queued"
            ),
            requires_action=True,
        )

    async def check_rejection(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
    ) -> Optional[ReviewGateResult]:
        """READ-ONLY: did a human REJECT this DAG structure? (#1971)

        Consulted by the refutation node on EVERY band -- a PROCEED band
        included, where ``check_approval`` is deliberately not called because
        a robust estimate does not need a queue row. Same chronology as
        ``check_approval`` so the two can never disagree: the most recent
        adjudication of the structure wins (an older still-unexpired approval
        never masks a newer rejection -- codex iter-3), and a pending row
        newer than that rejection means a reviewer re-opened the structure.
        Never creates a review row. One read (the full history).

        Returns:
            The REJECTED ``ReviewGateResult`` (reviewer, review id, reason) when
            the most recent adjudication of this structure is a rejection and
            nothing re-opened it; ``None`` otherwise -- including when there is
            no repository, in which case the answer is "cannot tell", not "not
            rejected".
        """
        if not self.repository:
            return None

        history = await self.repository.get_reviews_for_dag(
            dag_hash, include_expired=True, brand=brand
        )
        latest_verdict, reopened = self._latest_adjudication(history)
        if (
            latest_verdict is not None
            and latest_verdict.get("approval_status") == "rejected"
            and not reopened
        ):
            return self._rejection_result(latest_verdict, dag_hash)
        return None

    async def can_proceed(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
        allow_pending: bool = False,
        allow_expiring: bool = True,
    ) -> bool:
        """
        Simple check if analysis can proceed.

        Args:
            dag_hash: SHA256 hash of the DAG
            brand: Brand context
            allow_pending: If True, allow proceeding with pending reviews
            allow_expiring: If True, allow proceeding with expiring approvals

        Returns:
            True if analysis can proceed. REJECTED, BLOCKED and UNAVAILABLE are
            never a yes: a gate that could not look is not a gate that cleared.
        """
        result = await self.check_approval(dag_hash, brand)

        if result.decision == ReviewGateDecision.PROCEED:
            return True
        elif result.decision == ReviewGateDecision.RENEWAL_REQUIRED and allow_expiring:
            return True
        elif result.decision == ReviewGateDecision.PENDING_REVIEW and allow_pending:
            return True

        return False

    async def request_renewal(
        self,
        dag_hash: str,
        requester_id: str,
        brand: Optional[str] = None,
        requester_name: Optional[str] = None,
        requester_email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Request renewal of an expiring DAG approval.

        Args:
            dag_hash: SHA256 hash of the DAG
            requester_id: User ID requesting renewal
            brand: Brand context
            requester_name: Display name
            requester_email: Contact email

        Returns:
            New review_id or None on failure
        """
        if not self.repository:
            return None

        # Find the existing approval to renew
        approval = await self.repository.get_dag_approval(dag_hash, brand)

        if not approval:
            logger.warning(f"No existing approval found for DAG {dag_hash}")
            return None

        # Create renewal review
        review_id = approval.get("review_id")
        if review_id is None:
            logger.warning(f"No review_id found in approval for DAG {dag_hash}")
            return None
        return await self.repository.renew_review(
            original_review_id=review_id,
            reviewer_id=requester_id,
            reviewer_name=requester_name,
            reviewer_email=requester_email,
        )

    async def get_pending_review_count(
        self,
        brand: Optional[str] = None,
    ) -> int:
        """
        Get count of pending reviews for monitoring.

        Args:
            brand: Optional brand filter

        Returns:
            Count of pending reviews
        """
        if not self.repository:
            return 0

        pending = await self.repository.get_pending_reviews(brand=brand)
        return len(pending)

    async def get_expiring_dag_count(
        self,
        days: int = 14,
        brand: Optional[str] = None,
    ) -> int:
        """
        Get count of DAG approvals expiring soon.

        Args:
            days: Days threshold for expiry warning
            brand: Optional brand filter

        Returns:
            Count of expiring approvals
        """
        if not self.repository:
            return 0

        expiring = await self.repository.get_expiring_reviews(days, brand)
        return len(expiring)

    async def get_gate_status(
        self,
        brand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get overall gate status for monitoring dashboard.

        Args:
            brand: Optional brand filter

        Returns:
            Dict with gate health metrics
        """
        if not self.repository:
            return {
                "healthy": True,
                "pending_reviews": 0,
                "expiring_soon": 0,
                "total_approved": 0,
                "message": "Expert review gate not configured",
            }

        summary = await self.repository.get_review_summary(brand)

        pending = summary.get("pending", 0)
        expiring = summary.get("expiring_soon", 0)
        approved = summary.get("approved", 0)

        # Gate is unhealthy if many reviews pending or expiring
        healthy = pending < 5 and expiring < 3

        return {
            "healthy": healthy,
            "pending_reviews": pending,
            "expiring_soon": expiring,
            "total_approved": approved,
            "total_rejected": summary.get("rejected", 0),
            "total_expired": summary.get("expired", 0),
            "message": (
                "Gate healthy"
                if healthy
                else "Attention needed: pending reviews or expiring approvals"
            ),
        }


# Convenience function for integration with causal workflow
async def check_dag_approval(
    dag_hash: str,
    brand: Optional[str] = None,
    repository: Optional[ExpertReviewRepository] = None,
) -> ReviewGateResult:
    """
    Check DAG approval status (standalone function).

    Args:
        dag_hash: SHA256 hash of the DAG
        brand: Brand context
        repository: ExpertReviewRepository instance

    Returns:
        ReviewGateResult with decision
    """
    gate = ExpertReviewGate(repository=repository)
    return await gate.check_approval(dag_hash, brand)
