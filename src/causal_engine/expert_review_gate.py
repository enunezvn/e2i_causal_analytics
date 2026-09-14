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

from src.causal_engine.dag_hash import compute_adjustment_set_hash
from src.repositories.expert_review import (
    ExpertReviewRepository,
    approval_validity,
    estimand_key_for,
)

logger = logging.getLogger(__name__)


class _VersionMatch(Enum):
    """How a run's structure compares with a review's last RECORDED version.

    Three states, not a boolean, because the pending branch and the mint path
    need different answers for "the review has no timeline yet": a mint records
    version 1 there, while an unchanged hash with nothing recorded appends
    nothing (there is no recorded pair to contradict it).
    """

    SAME = "same"
    DIFFERENT = "different"
    NOT_RECORDED = "not_recorded"


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
    on every refutation band (read-only) and ``check_approval`` on REVIEW alone
    (queue-or-lookup; a BLOCK band queues nothing -- #1991 debt 3, it already
    failed statistically). Whether a missing approval HALTS a run is the node's
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
        Check whether this ESTIMAND's review clears the DAG under analysis.

        Identity is the estimand (``lower(brand):treatment:outcome``, migration
        140): an estimand holds at most one pending review, and a structure
        change UPDATES it. Version is ``dag_hash``: a differing hash appends a
        row to the review's version timeline (migration 141) instead of minting
        a sibling row, and an approval is only ever an approval OF the hash it
        was granted on.

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

        # The ESTIMAND's history, newest first (#1991 debt 3). Keying the read
        # on the estimand instead of on one hash is what lets a structure
        # change land on the review that already exists: a hash-keyed read
        # returns nothing for a new hash, which is how the old gate minted a
        # second row for the same question.
        estimand_key = estimand_key_for(brand, treatment, outcome)
        history = await self.repository.get_reviews_for_estimand(estimand_key, include_expired=True)

        # Chronology first (#1971, codex iter-3): the MOST RECENT adjudication
        # wins. An approval that is still unexpired but OLDER than a rejection
        # (approve A, renew as B, reject B while A's 90 days run) is superseded,
        # so the approval must not be read on its own. The whole history is
        # needed for that ordering, expired approvals included.
        # A rejection covers the VERSION it was given on, not the estimand
        # (spec §7): the reviewer said "THIS structure is wrong", which is
        # precisely what a revised structure answers -- so a rejected estimand
        # stays re-openable by revising the DAG. The chronology that decides it
        # is therefore ranked over the SAME-HASH rows, which is exactly the row
        # set the hash-keyed ``check_rejection`` probe reads: the two readers
        # share one input and cannot disagree. Filtering the VERDICT instead of
        # the INPUT is not the same thing -- rank the whole estimand and an
        # approval of a revised hash sits on top, hiding the rejection of this
        # one, and the gate would queue a structure a human already refused.
        same_hash_history = [r for r in history if r.get("dag_version_hash") == dag_hash]
        latest_verdict, reopened = self._latest_adjudication(same_hash_history)
        superseded_by_rejection = (
            latest_verdict is not None and latest_verdict.get("approval_status") == "rejected"
        )

        # Check for active approval -- only if no newer rejection supersedes it.
        # Read from the history rather than from ``get_dag_approval``: that
        # query answers "is THIS hash approved" without the estimand's
        # chronology, and the history already carries the answer. An approval
        # is scoped to the hash it was granted on -- the same estimand on a
        # DIFFERENT structure is NOT approved by it (spec §7); it re-opens a
        # review below, and the old approval keeps its validity for its own
        # hash.
        today = date.today()
        approval = (
            None
            if superseded_by_rejection
            else next(
                (
                    row
                    for row in history
                    if row.get("approval_status") == "approved"
                    and row.get("dag_version_hash") == dag_hash
                    and approval_validity(row, today) != "expired"
                ),
                None,
            )
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

        # A REJECTED verdict is durable (#1970). The estimand's history comes
        # back created_at DESC, so if the most recent adjudication is a
        # rejection OF THIS HASH (the scoping above) a human already turned
        # this exact structure down. Auto-creating a fresh pending row for it
        # on the next REVIEW band would silently undo that decision. A NEWER
        # approval or pending row wins: ``reopened`` comes from the
        # ``_latest_adjudication`` call above; a genuinely newer pending row
        # sets it and skips this block, reaching the pending branch below. A
        # reviewer who wants to re-open the SAME structure does so from the
        # review UI, not by re-running -- but a DIFFERENT structure of the same
        # estimand is a new question and falls through to the mint below. #1971
        # gives the verdict its own decision value (REJECTED, not BLOCKED) so
        # consumers can tell "a human said no" from "nobody has looked yet and
        # no row could be queued".
        if superseded_by_rejection and not reopened:
            assert latest_verdict is not None  # narrowed by superseded_by_rejection
            return self._rejection_result(latest_verdict, dag_hash)

        # No usable approval - check for pending review (a pending row NEWER
        # than a rejection is a reviewer re-opening the structure).
        pending = [r for r in history if r.get("approval_status") == "pending"]
        structure = sanitize_dag_structure(dag_structure)
        # No structure in scope means the adjustment set is UNKNOWN, which
        # migration 141 records as NULL. ``sha256("[]")`` is the canonical
        # EMPTY set -- a different fact, and one this call cannot assert.
        adjustment_set_hash = (
            compute_adjustment_set_hash(list(dag_structure.get("adjustment_sets") or []))
            if dag_structure
            else None
        )

        if pending:
            pending_row = pending[0]
            review_id = pending_row.get("review_id")

            current_hash = pending_row.get("dag_version_hash")
            match = await self._match_last_recorded_version(
                review_id, dag_hash, adjustment_set_hash
            )
            # What "the structure changed" MEANS is the PAIR (codex round-1
            # HIGH): ``compute_dag_hash`` deliberately EXCLUDES adjustment sets,
            # so comparing the hash alone kept every covariate-only change off
            # the timeline -- the reviewer never saw the estimand's adjustment
            # set move under an unchanged DAG. The pair is compared against the
            # LAST RECORDED version, not against the review row, because
            # ``expert_reviews`` has no adjustment-set column.
            # The review's own hash still counts on its own: when it disagrees
            # with this run, the review must be advanced even if the timeline
            # already holds the pair (a previous advance lost its race), or it
            # would stay on a structure nothing will move it off.
            # With no timeline at all (recorded is None) an unchanged hash
            # appends nothing, exactly as before: there is no recorded pair to
            # contradict it.
            hash_changed = current_hash != dag_hash
            if review_id and (hash_changed or match is _VersionMatch.DIFFERENT):
                # Same question, new structure (#1991 debt 3): APPEND a version
                # to the open review and advance it, instead of minting a
                # sibling row. ``expert_review_versions`` is a timeline, so a
                # revert (A -> B -> A) legitimately appends a third row.
                # Best-effort: a failed append must not withhold the review id
                # the caller needs (the repository logs the detail, and the
                # review is still pending either way).
                appended = await self.repository.append_version(
                    str(review_id),
                    dag_version_hash=dag_hash,
                    dag_structure=structure,
                    adjustment_set_hash=adjustment_set_hash,
                    query_id=requester_id,
                    # The review now carries THIS run's hash, so it must carry
                    # this run's evidence too -- the detail route renders the
                    # two side by side.
                    related_validation_ids=related_validation_ids,
                    # Compare-and-set on the hash READ above: two interleaved
                    # appends must not let the loser's advance overwrite the
                    # winner's (insert h2, insert h3, advance h3, advance h2).
                    expected_current_hash=current_hash,
                )
                if not appended:
                    logger.warning(
                        f"Could not record structure version {dag_hash} on pending review "
                        f"{review_id} (estimand {estimand_key}); the review stays pending "
                        "on its previous version."
                    )
                return ReviewGateResult(
                    decision=ReviewGateDecision.PENDING_REVIEW,
                    dag_hash=dag_hash,
                    is_approved=False,
                    review_id=review_id,
                    message=(
                        "DAG review pending expert approval (structure updated to a new version)"
                    ),
                    requires_action=True,
                )

            # Review already pending on THIS structure. Backfill-on-encounter
            # (097): this short-circuit is the ONLY consult a pre-097
            # (structure-less) pending row will ever see for its DAG, so attach
            # the renderable snapshot now. Best-effort — a backfill failure must
            # never break the gate.
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

        # No approval and no pending review. A VALID approval of this estimand
        # on a DIFFERENT structure means the question was signed off once and
        # the structure has since changed: re-open, recording which approval the
        # new review replaces. The old approval is NOT revoked -- it keeps its
        # validity for its own hash (spec §7), so a re-run of that structure
        # still clears.
        # ``supersedes_review_id`` links a superseded APPROVAL only (§7). A
        # rejection of another hash is not superseded by this review -- it
        # stands as the verdict on its own structure.
        superseded_approval = next(
            (
                row
                for row in history
                if row.get("approval_status") == "approved"
                and row.get("dag_version_hash") != dag_hash
                and approval_validity(row, today) != "expired"
            ),
            None,
        )
        supersedes_review_id = (
            str(superseded_approval["review_id"])
            if superseded_approval and superseded_approval.get("review_id")
            else None
        )

        if self.auto_create_review and requester_id:
            # Auto-create review request. The duplicate-row race the old
            # hash-keyed pre-check could not close is now closed in the DB:
            # migration 140's partial UNIQUE index uq_er_pending_estimand allows
            # ONE pending review per estimand, and create_review recovers the
            # winner's row on the 23505 -- so ``review_id`` below is the id of a
            # review this call may or may not have inserted, and nothing here
            # may claim it was newly created.
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
                dag_structure=structure,
                related_validation_ids=related_validation_ids,
                supersedes_review_id=supersedes_review_id,
            )

            if review_id:
                # On the insert path this is version 1 of the new review's
                # timeline, so every review minted after migration 141 has a
                # first version and ``get_versions`` is never empty (the
                # backfill covers older rows). Its review UPDATE rewrites the
                # hash and snapshot ``create_review`` just stored -- redundant,
                # not a second distinct write. On the 23505-recovery path the
                # append is the part that MATTERS: this run's structure becomes
                # a version on the WINNER's review rather than being lost.
                # Best-effort either way, like the append above: the review
                # exists and is pending whatever the timeline write did.
                #
                # ... but ONLY when it is not already recorded (codex round-1
                # HIGH): on that recovery path ``review_id`` is a review this
                # call may not have inserted, and two concurrent mints of the
                # SAME structure both recover the winner's id. Appending
                # unconditionally gave the winner's timeline two identical rows,
                # which reads as "the DAG changed twice" in the review UI. A
                # fresh insert has no timeline (recorded is None) and still
                # records version 1.
                match = await self._match_last_recorded_version(
                    review_id, dag_hash, adjustment_set_hash
                )
                if match is not _VersionMatch.SAME:
                    appended = await self.repository.append_version(
                        str(review_id),
                        dag_version_hash=dag_hash,
                        dag_structure=structure,
                        adjustment_set_hash=adjustment_set_hash,
                        query_id=requester_id,
                        # Same value ``create_review`` just stored; passed for
                        # symmetry with the append above, so the two call sites
                        # cannot drift on what an advance carries.
                        related_validation_ids=related_validation_ids,
                        # ``create_review`` stored this hash a moment ago, so the
                        # compare-and-set normally holds; it refuses only if
                        # something advanced the review in between, which is
                        # exactly the write this must not undo.
                        expected_current_hash=dag_hash,
                    )
                    if not appended:
                        logger.warning(
                            f"Could not record the first structure version {dag_hash} on new "
                            f"review {review_id} (estimand {estimand_key})."
                        )
                return ReviewGateResult(
                    decision=ReviewGateDecision.PENDING_REVIEW,
                    dag_hash=dag_hash,
                    is_approved=False,
                    review_id=review_id,
                    # Neutral on WHO created the row: on the 23505-recovery
                    # path a concurrent run inserted it and this one only
                    # added its structure version.
                    message="Expert review pending for this DAG structure",
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

    async def _match_last_recorded_version(
        self, review_id: Any, dag_hash: str, adjustment_set_hash: Optional[str]
    ) -> "_VersionMatch":
        """How this run's ``(dag_version_hash, adjustment_set_hash)`` pair compares
        with the review's LAST RECORDED structure version.

        This is the thing that decides whether anything actually changed.
        ``expert_reviews`` carries no adjustment-set column -- that hash lives on
        the version row (migration 141) -- so the timeline is the only place the
        previous adjustment set can be read.

        A version row written by migration 141's BACKFILL has
        ``adjustment_set_hash = NULL``, which is not "no adjustment set": the
        hash was simply never computed for it. Where the row kept a structure
        snapshot, the hash is DERIVED from it with the same function the writer
        uses, so the first re-encounter of an unchanged DAG is recognised as
        unchanged instead of appending a spurious version. A NULL hash with no
        usable snapshot stays None -- genuinely unknown, which differs from every
        computed hash and from the canonical empty set ``sha256("[]")``.

        A read FAILURE is reported as SAME, not as NOT_RECORDED: it is logged and
        read as "nothing new to record". Skipping an append cannot corrupt
        anything -- the review stays pending on the version it already carries,
        that version is what a resolution binds to (``submit_review``), and the
        next run re-reads and appends -- whereas appending on a comparison that
        never happened writes the duplicate row this check exists to prevent. The
        consult itself stays available, which a propagated error would not leave.

        Returns:
            SAME when the last recorded version is this pair (or is unreadable),
            DIFFERENT when it is another pair, NOT_RECORDED when the review has
            no timeline at all
        """
        try:
            latest = await self.repository.get_latest_version(str(review_id))
        except Exception as read_err:  # noqa: BLE001 - the consult must not break
            logger.warning(
                f"Could not read the latest structure version of review {review_id} "
                f"({read_err}); treating this run's structure as already recorded, so "
                "nothing is appended and the review keeps the version it carries"
            )
            return _VersionMatch.SAME
        if not latest:
            return _VersionMatch.NOT_RECORDED
        adjustment_hash = latest.get("adjustment_set_hash")
        if adjustment_hash is None:
            snapshot = latest.get("dag_structure_json")
            if isinstance(snapshot, dict):
                adjustment_hash = compute_adjustment_set_hash(
                    list(snapshot.get("adjustment_sets") or [])
                )
        recorded = (str(latest.get("dag_version_hash") or ""), adjustment_hash)
        if recorded == (dag_hash, adjustment_set_hash):
            return _VersionMatch.SAME
        return _VersionMatch.DIFFERENT

    @staticmethod
    def _latest_adjudication(
        history: List[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """``(most recent non-pending row, a pending row is newer than it)``.

        ``history`` is newest-first (``get_reviews_for_estimand`` orders
        created_at DESC; rows are created and resolved in order because the
        unique-pending index (migration 140) allows one pending review per
        ESTIMAND at a time, so creation order is adjudication order -- 062's
        per-structure index, which this replaced, gave the same property for
        the hash-keyed read ``check_rejection`` still uses). An exact
        ``created_at`` tie
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
