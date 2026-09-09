"""
Expert Review Repository.

Handles persistence of domain expert reviews for causal DAG validation.

Version: 4.3
Database: expert_reviews table (010_causal_validation_tables.sql)

Validity semantics (#1972) -- ONE definition, owned by the schema:
    active approval  ==  approval_status = 'approved'
                         AND (valid_until IS NULL OR valid_until >= today)
A NULL ``valid_until`` is a PERMANENT approval (``v_active_expert_approvals``
labels it ``'permanent'``). ``expired`` is never stored; it is derived from
``valid_until`` at read time. Every reader in this module routes through the
helpers below so the gate, the summary and SQL ``is_dag_approved()`` agree.
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Mapping, Optional

from src.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONE definition of "active approval" (#1972)
#
# The schema owns it. 010_causal_validation_tables.sql's is_dag_approved()
# tests
#     approval_status = 'approved' AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
# and the v_active_expert_approvals view (which filters only on
# approval_status = 'approved' and returns expired rows too) classifies each
# row's validity_status the same way: NULL -> 'permanent', >= CURRENT_DATE ->
# 'active', else 'expired'. approval_validity() mirrors that classification;
# is_active_approval() / _apply_active_validity() mirror the function.
#
# Why the helpers exist (measured on the live droplet, 2026-09-08): three
# readers here used ``.gte("valid_until", today)``. Under SQL three-valued
# logic ``NULL >= date`` is not true, so a NULL row was invisible to them
# (PostgREST ``valid_until=gte.<today>`` -> 0 of 39 rows; the
# ``or=(valid_until.gte.<today>,valid_until.is.null)`` form -> 39 of 39) while
# SQL and get_reviews_for_dag called the same row permanent, and the summary
# counted it approved. Nothing writes NULL on approval today (submit_review
# always sets today + validity_days), so the harm was latent -- but a
# SQL-side write, a backfill or a future permanent approval would have been
# re-queued by ExpertReviewGate as "not approved". The AST guard in
# tests/unit/test_repositories/test_approval_expiry_readers_1972.py rejects
# any raw valid_until filter outside these helpers.
# ---------------------------------------------------------------------------

ApprovalValidity = Literal["permanent", "active", "expired"]


def _valid_until_date(row: Mapping[str, Any]) -> Optional[date]:
    """Parse a row's ``valid_until``; ``None`` means the approval never expires.

    PostgREST returns the DATE column as an ISO date string; rows built in
    Python may carry a ``date`` (or ``datetime``). Anything else -- including
    an empty string -- raises ``ValueError``: a malformed value is never
    silently classified as permanent or active (codex iter-1 LOW). In
    ``get_review_summary`` that surfaces as the logged error + zero counts
    the outer handler already produces for any malformed row.
    """
    raw = row.get("valid_until")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw))


def approval_validity(row: Mapping[str, Any], today: Optional[date] = None) -> ApprovalValidity:
    """Classify a row exactly as ``v_active_expert_approvals.validity_status`` does.

    ``'permanent'`` when ``valid_until`` is NULL; ``'active'`` while
    ``valid_until >= today`` (inclusive, like SQL ``>= CURRENT_DATE``);
    ``'expired'`` after that. Pure -- pass ``today`` for determinism.
    Meaningful only for ``approval_status == 'approved'`` rows: pending and
    rejected rows carry a NULL ``valid_until`` but are not approvals.
    """
    expires = _valid_until_date(row)
    if expires is None:
        return "permanent"
    return "active" if expires >= (today or date.today()) else "expired"


def is_active_approval(row: Mapping[str, Any], today: Optional[date] = None) -> bool:
    """Per-row mirror of SQL ``is_dag_approved()``:
    ``approval_status = 'approved' AND (valid_until IS NULL OR valid_until >= today)``.
    """
    return row.get("approval_status") == "approved" and approval_validity(row, today) != "expired"


def _apply_active_validity(query: Any, today: Optional[date] = None) -> Any:
    """Query-side form of the one definition: ``valid_until >= today OR valid_until IS NULL``.

    The ``or=(...)`` form is required. A bare ``.gte("valid_until", today)``
    drops NULL rows under three-valued logic -- that was the drift.
    """
    day = (today or date.today()).isoformat()
    return query.or_(f"valid_until.gte.{day},valid_until.is.null")


def _apply_expiring_window(query: Any, days: int, today: Optional[date] = None) -> Any:
    """``today <= valid_until <= today + days`` (both ends inclusive).

    A NULL ``valid_until`` (permanent approval) never matches -- by design, a
    permanent approval is never "expiring soon".
    """
    start = today or date.today()
    end = start + timedelta(days=days)
    return query.gte("valid_until", start.isoformat()).lte("valid_until", end.isoformat())


class ExpertReviewRepository(BaseRepository):
    """
    Repository for expert_reviews table.

    Supports:
    - Creating and updating expert reviews
    - Checking DAG approval status
    - Querying pending reviews for admin UI
    - Managing review validity periods

    Validity: see the module docstring -- NULL ``valid_until`` is permanent,
    ``expired`` is derived at read time, and every reader goes through
    ``_apply_active_validity`` / ``_apply_expiring_window`` /
    ``approval_validity`` (#1972).

    Database Schema (expert_reviews):
    - review_id: UUID PRIMARY KEY
    - review_type: expert_review_type ENUM
    - dag_version_hash: VARCHAR(64)
    - reviewer_id: VARCHAR(100) NOT NULL
    - reviewer_name: VARCHAR(200)
    - reviewer_role: VARCHAR(100)
    - reviewer_email: VARCHAR(200)
    - approval_status: VARCHAR(30) DEFAULT 'pending'
    - checklist_json: JSONB
    - comments_json: JSONB
    - concerns_raised: TEXT[]
    - conditions: TEXT
    - brand: VARCHAR(50)
    - analysis_context: TEXT
    - treatment_variable: VARCHAR(100)
    - outcome_variable: VARCHAR(100)
    - valid_from: DATE
    - valid_until: DATE
    - created_at: TIMESTAMPTZ
    - updated_at: TIMESTAMPTZ
    - approved_at: TIMESTAMPTZ
    - related_validation_ids: UUID[]
    - supersedes_review_id: UUID
    """

    table_name = "expert_reviews"
    id_column = "review_id"  # live PK (#894 codex R1: create_renewal_review's
    # get_by_id queried a nonexistent "id" column -> latent 42703)
    model_class = None  # Using raw dicts

    # Default validity period for expert reviews (90 days = quarterly)
    DEFAULT_VALIDITY_DAYS = 90

    async def create_review(
        self,
        reviewer_id: str,
        review_type: str,
        dag_version_hash: Optional[str] = None,
        reviewer_name: Optional[str] = None,
        reviewer_role: Optional[str] = None,
        reviewer_email: Optional[str] = None,
        brand: Optional[str] = None,
        treatment_variable: Optional[str] = None,
        outcome_variable: Optional[str] = None,
        analysis_context: Optional[str] = None,
        checklist: Optional[Dict[str, Any]] = None,
        related_validation_ids: Optional[List[str]] = None,
        dag_structure: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Create a new expert review request.

        Args:
            reviewer_id: User ID of the reviewer
            review_type: Type of review ('initial_dag', 'quarterly_audit', 'methodology_change', 'edge_dispute')
            dag_version_hash: SHA256 hash of the DAG being reviewed
            reviewer_name: Display name of reviewer
            reviewer_role: Role (commercial_ops, medical_affairs, data_science, etc.)
            reviewer_email: Contact email
            brand: Brand context
            treatment_variable: Treatment being analyzed
            outcome_variable: Outcome being measured
            analysis_context: Description of what analysis this covers
            checklist: Initial checklist items (to be completed during review)
            related_validation_ids: Related causal_validations records
            dag_structure: Sanitized causal-graph snapshot (nodes/edges/
                treatment/outcome) persisted as dag_structure_json (mig 097) so
                the review UI can render the DAG under review — the hash alone
                is one-way and not renderable

        Returns:
            Created review_id or None on failure
        """
        if not self.client:
            logger.warning("No Supabase client, skipping review creation")
            return None

        row = {
            "reviewer_id": reviewer_id,
            "review_type": review_type,
            "dag_version_hash": dag_version_hash,
            "reviewer_name": reviewer_name,
            "reviewer_role": reviewer_role,
            "reviewer_email": reviewer_email,
            "approval_status": "pending",
            "brand": brand,
            "treatment_variable": treatment_variable,
            "outcome_variable": outcome_variable,
            "analysis_context": analysis_context,
            "checklist_json": json.dumps(checklist) if checklist else None,
            "related_validation_ids": related_validation_ids,
            "dag_structure_json": json.dumps(dag_structure) if dag_structure else None,
        }

        # Remove None values
        row = {k: v for k, v in row.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(row).execute()
            review_id = result.data[0]["review_id"] if result.data else None
            logger.info(f"Created expert review {review_id} for DAG {dag_version_hash}")
            return review_id
        except Exception as e:
            # M-reach1: a concurrent creator may have won the race and inserted the
            # pending row first — the partial UNIQUE index uq_er_pending_dag_brand
            # (mig 062) then rejects THIS duplicate with a 23505 unique violation.
            # Recover ONLY for that case (return the winner's pending review); let
            # any other failure (transient connection/timeout, schema error) surface
            # via the error log rather than masking it behind a possibly-stale
            # pending row (codex MEDIUM).
            err = str(e).lower()
            if "23505" in err or "unique" in err or "duplicate key" in err:
                existing = await self._find_pending_review_id(dag_version_hash, brand)
                if existing is not None:
                    logger.info(
                        "create_review: a pending review already exists for DAG "
                        f"{dag_version_hash} (brand={brand}); returning it "
                        "(concurrent create / unique-violation recovery)."
                    )
                    return existing
            logger.error(f"Failed to create expert review: {e}")
            return None

    async def _find_pending_review_id(
        self, dag_version_hash: Optional[str], brand: Optional[str]
    ) -> Optional[str]:
        """M-reach1: return the review_id of an existing PENDING review for this DAG
        (+brand), if any. Brand matching mirrors the uq_er_pending_dag_brand index
        (NULL brand normalized): a None brand matches the NULL-brand pending row, a
        set brand matches that brand exactly.

        Note (codex LOW): the index keys on ``COALESCE(brand, '')``, so it also
        collapses an explicit empty-string brand into the NULL bucket; this lookup
        uses ``IS NULL`` and would NOT match a ``brand=''`` winner. Callers populate
        brand from a real brand name or None — never an explicit empty string — so
        this boundary is not reachable in practice.
        """
        if not self.client or not dag_version_hash:
            return None
        try:
            query = (
                self.client.table(self.table_name)
                .select("review_id")
                .eq("dag_version_hash", dag_version_hash)
                .eq("approval_status", "pending")
            )
            query = query.eq("brand", brand) if brand else query.is_("brand", "null")
            result = await query.limit(1).execute()
            if result.data:
                return str(result.data[0]["review_id"])
        except Exception as e:
            logger.warning(f"_find_pending_review_id lookup failed: {e}")
        return None

    async def submit_review(
        self,
        review_id: str,
        approval_status: str,
        checklist: Dict[str, Any],
        comments: Optional[Dict[str, Any]] = None,
        concerns_raised: Optional[List[str]] = None,
        conditions: Optional[str] = None,
        validity_days: int = DEFAULT_VALIDITY_DAYS,
        reviewer_name: Optional[str] = None,
        reviewer_email: Optional[str] = None,
    ) -> bool:
        """
        Submit a completed expert review.

        Only a PENDING row can be resolved (R2, lane-1971 audit): the UPDATE
        itself carries ``approval_status = 'pending'``, so an already-resolved
        row -- including an OLDER approval while a NEWER one exists -- matches
        zero rows and returns False (the route surfaces that as 404). Before
        this the filter was ``review_id`` alone and the pending-only claim was
        documentation, not enforcement.

        Resolution provenance (lane 1, codex whole-diff HIGH F1): BOTH statuses
        stamp ``resolved_at = now()`` (migration 136; ``approved_at`` stays
        approval-only) and record the RESOLVER's ``reviewer_name`` /
        ``reviewer_email`` when the caller knows them. ``reviewer_id`` is
        deliberately NOT written here: the gate stores the REQUESTER in it --
        the originating query id (src/causal_engine/expert_review_gate.py
        create_review(reviewer_id=requester_id), ~:392) -- and that breadcrumb
        must survive the resolution. An absent identity is left absent (the
        None-strip below drops it): unknown stays unknown, never a placeholder.

        Args:
            review_id: UUID of the review to complete
            approval_status: 'approved' or 'rejected'
            checklist: Completed checklist with responses
            comments: Reviewer notes and feedback
            concerns_raised: List of specific concerns
            conditions: Any conditions on approval
            validity_days: Days until review expires (default 90)
            reviewer_name: Display name of the resolving operator, if known
            reviewer_email: Email of the resolving operator, if known

        Returns:
            True if exactly this pending row was resolved, False otherwise
            (nonexistent, already resolved, or persistence error)
        """
        if not self.client:
            return False

        if approval_status not in ("approved", "rejected"):
            logger.error(f"Invalid approval_status: {approval_status}")
            return False

        update_data = {
            "approval_status": approval_status,
            "checklist_json": json.dumps(checklist),
            "comments_json": json.dumps(comments) if comments else None,
            "concerns_raised": concerns_raised,
            "conditions": conditions,
            # Decision time for BOTH statuses (migration 136).
            "resolved_at": "now()",
            "reviewer_name": reviewer_name,
            "reviewer_email": reviewer_email,
        }

        if approval_status == "approved":
            valid_until = date.today() + timedelta(days=validity_days)
            update_data["valid_from"] = date.today().isoformat()
            update_data["valid_until"] = valid_until.isoformat()
            update_data["approved_at"] = "now()"

        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}

        try:
            result = await (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("review_id", review_id)
                .eq("approval_status", "pending")
                .execute()
            )
            # FIX B (codex HIGH): a zero-row update (nonexistent or already-resolved
            # review_id) matches nothing — supabase-py returns the updated rows in
            # ``result.data`` (same convention as base.py:131), so empty data means
            # nothing was touched. Returning True there is a fabricated success that
            # would make the route 200 a record it never changed.
            if not result.data:
                logger.warning(
                    f"submit_review matched no rows for {review_id} "
                    "(nonexistent or already-resolved); returning False"
                )
                return False
            logger.info(f"Submitted review {review_id} with status {approval_status}")
            return True
        except Exception as e:
            logger.error(f"Failed to submit review {review_id}: {e}")
            return False

    async def update_agent_assessment(
        self,
        review_id: str,
        assessment: Dict[str, Any],
    ) -> bool:
        """Cache an advisory agent assessment on the review row (mig 097).

        Writes ``agent_assessment_json`` ONLY — never touches
        ``checklist_json``, which remains the human reviewer's own record.

        Returns:
            True when exactly this row was updated; False on zero-row match
            (nonexistent review) or persistence error — fail-closed, mirroring
            ``submit_review``.
        """
        if not self.client:
            return False

        try:
            result = await (
                self.client.table(self.table_name)
                .update({"agent_assessment_json": json.dumps(assessment)})
                .eq("review_id", review_id)
                .execute()
            )
            if not result.data:
                logger.warning(
                    f"update_agent_assessment matched no rows for {review_id}; returning False"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to cache agent assessment for {review_id}: {e}")
            return False

    async def update_dag_structure(
        self,
        review_id: str,
        dag_structure: Dict[str, Any],
        related_validation_ids: Optional[List[str]] = None,
    ) -> bool:
        """Backfill the DAG snapshot on an existing review row (097).

        Pre-097 pending rows carry only the one-way hash; when the same DAG is
        re-encountered by the gate, this attaches the renderable structure (and
        the fresh run's evidence ids, when given) to the EXISTING row instead of
        losing it to the pending-row short-circuit. Zero-row match or error
        returns False (fail-closed), mirroring ``submit_review``.
        """
        if not self.client or not dag_structure:
            return False

        update_data: Dict[str, Any] = {"dag_structure_json": json.dumps(dag_structure)}
        if related_validation_ids:
            update_data["related_validation_ids"] = related_validation_ids

        try:
            result = await (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("review_id", review_id)
                .execute()
            )
            if not result.data:
                logger.warning(
                    f"update_dag_structure matched no rows for {review_id}; returning False"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to backfill DAG structure for {review_id}: {e}")
            return False

    async def is_dag_approved(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
    ) -> bool:
        """
        Check if a DAG has an active expert approval.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            brand: Optional brand filter

        Returns:
            True if DAG has active approval (permanent or unexpired), False
            otherwise -- including when there is no client to ask (#1972:
            "assuming approved" was a plausible-wrong value a caller could not
            tell apart from a verified one; fail-closed, like every other
            no-client path in this repository).

        Raises:
            The underlying client error on a query failure, after logging it
            (R3). An outage must not read as "not approved" any more than as
            "approved" -- only the no-client early return keeps its False.
        """
        if not self.client:
            logger.warning(
                "No Supabase client; cannot verify DAG approval, treating it as NOT approved"
            )
            return False

        try:
            query = _apply_active_validity(
                self.client.table(self.table_name)
                .select("review_id")
                .eq("dag_version_hash", dag_hash)
                .eq("approval_status", "approved")
            )

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return len(result.data) > 0
        except Exception as e:
            # R3: log, then re-raise -- never turn an outage into "not approved".
            logger.error(f"Failed to check DAG approval: {e}")
            raise

    async def get_dag_approval(
        self,
        dag_hash: str,
        brand: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the active approval record for a DAG.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            brand: Optional brand filter

        Returns:
            The newest active approval record (permanent or unexpired --
            same predicate as ``is_dag_approved``), or None if not approved

        Raises:
            The underlying client error on a query failure, after logging it
            (R1, lane-1971 audit). A store outage must not read as "no
            approval on file": the gate maps the raise to ``unavailable``
            and the refutation node to ``unknown`` instead of proceeding.
            The no-client early return (None) is unchanged.
        """
        if not self.client:
            return None

        try:
            query = _apply_active_validity(
                self.client.table(self.table_name)
                .select("*")
                .eq("dag_version_hash", dag_hash)
                .eq("approval_status", "approved")
            )
            query = query.order("approved_at", desc=True).limit(1)

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data[0] if result.data else None
        except Exception as e:
            # R1: log, then re-raise -- never turn an outage into "not approved".
            logger.error(f"Failed to get DAG approval: {e}")
            raise

    async def get_pending_reviews(
        self,
        brand: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get pending reviews for admin UI.

        Args:
            brand: Optional brand filter
            reviewer_id: Optional filter by assigned reviewer
            limit: Maximum records to return

        Returns:
            List of pending review records, oldest first

        Raises:
            The underlying client error on a query failure, after logging it
            (R3). This feeds the Expert Reviews page: an empty queue on an
            outage would render "0 pending" with HTTP 200, a plausible-wrong
            value. The route maps the raise to 503. The no-client early
            return ([]) is unchanged.
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("approval_status", "pending")
                .order("created_at", desc=False)
                .limit(limit)
            )

            if brand:
                query = query.eq("brand", brand)
            if reviewer_id:
                query = query.eq("reviewer_id", reviewer_id)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            # R3: log, then re-raise -- never serve an empty queue on an outage.
            logger.error(f"Failed to get pending reviews: {e}")
            raise

    async def get_expiring_reviews(
        self,
        days_until_expiry: int = 14,
        brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get approved reviews whose ``valid_until`` falls within the next
        ``days_until_expiry`` days (today and the horizon both inclusive).

        Used for proactive renewal notifications. A PERMANENT approval
        (NULL ``valid_until``) is never "expiring soon" and is never returned
        here -- renewal reminders are only for time-limited approvals. This is
        the one reader that deliberately does NOT use the active-approval
        predicate: "active" includes permanent, "expiring" excludes it.

        Args:
            days_until_expiry: Number of days until expiration
            brand: Optional brand filter

        Returns:
            List of soon-to-expire review records, soonest first

        Raises:
            The underlying client error on a query failure, after logging it
            (R3): "nothing expiring" must be a measured fact, not an outage.
            The no-client early return ([]) is unchanged.
        """
        if not self.client:
            return []

        try:
            query = _apply_expiring_window(
                self.client.table(self.table_name).select("*").eq("approval_status", "approved"),
                days_until_expiry,
            ).order("valid_until", desc=False)

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            # R3: log, then re-raise -- never serve "nothing expiring" on an outage.
            logger.error(f"Failed to get expiring reviews: {e}")
            raise

    async def get_reviews_for_dag(
        self,
        dag_hash: str,
        include_expired: bool = False,
        brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all reviews for a specific DAG version.

        Args:
            dag_hash: SHA256 hash of the DAG structure
            include_expired: When False (default), rows whose ``valid_until``
                has passed are dropped using the same predicate as
                ``is_dag_approved``; a NULL ``valid_until`` (pending and
                rejected rows, and permanent approvals) is kept
            brand: Optional brand filter; when set, only reviews for this brand are returned

        Returns:
            List of review records for the DAG

        Raises:
            The underlying client error on a query failure, after logging it
            (R1, lane-1971 audit). The gate's rejection probe reads an empty
            list as "structure clear"; an outage must not look like that.
            The no-client early return ([]) is unchanged.
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("dag_version_hash", dag_hash)
                .order("created_at", desc=True)
            )

            if not include_expired:
                query = _apply_active_validity(query)

            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            # R1: log, then re-raise -- never turn an outage into "no reviews".
            logger.error(f"Failed to get reviews for DAG: {e}")
            raise

    async def renew_review(
        self,
        original_review_id: str,
        reviewer_id: str,
        reviewer_name: Optional[str] = None,
        reviewer_role: Optional[str] = None,
        reviewer_email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create a renewal review that supersedes an existing one.

        Validity (#1972): the renewal is a ``pending`` row carrying
        ``supersedes_review_id`` and NO validity of its own -- ``submit_review``
        assigns ``valid_until = today + validity_days`` when it is approved.
        The original row is not modified and nothing filters on
        ``supersedes_review_id``, so approving a renewal NEVER revokes the
        original: ``is_dag_approved`` stays True while EITHER row is active,
        and ``get_dag_approval`` (newest active ``approved_at`` first) reports
        the renewal while it is active, then the original again -- but only
        while the original is itself still active (always, for a permanent
        one; a time-limited original that has also lapsed leaves nothing, and
        the DAG is unapproved). Renewing a PERMANENT approval (NULL
        ``valid_until``) therefore does not make the DAG time-limited -- it
        only changes which record the gate reports, and the gate's renewal
        warning follows that record's ``valid_until``. The original is not
        checked for being approved or active -- any existing row may be
        renewed.

        Args:
            original_review_id: UUID of the review to renew
            reviewer_id: User ID of the new reviewer
            reviewer_name: Display name
            reviewer_role: Role
            reviewer_email: Contact email

        Returns:
            New review_id or None on failure
        """
        if not self.client:
            return None

        # Get the original review
        original = await self.get_by_id(original_review_id)
        if not original:
            logger.error(f"Original review {original_review_id} not found")
            return None

        # Create renewal review with context from original
        row = {
            "reviewer_id": reviewer_id,
            "review_type": "quarterly_audit",
            "dag_version_hash": original.get("dag_version_hash"),
            "reviewer_name": reviewer_name,
            "reviewer_role": reviewer_role,
            "reviewer_email": reviewer_email,
            "approval_status": "pending",
            "brand": original.get("brand"),
            "treatment_variable": original.get("treatment_variable"),
            "outcome_variable": original.get("outcome_variable"),
            "analysis_context": f"Renewal of review {original_review_id}",
            "supersedes_review_id": original_review_id,
        }

        # Remove None values
        row = {k: v for k, v in row.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(row).execute()
            review_id = result.data[0]["review_id"] if result.data else None
            logger.info(f"Created renewal review {review_id} superseding {original_review_id}")
            return review_id
        except Exception as e:
            logger.error(f"Failed to create renewal review: {e}")
            return None

    async def get_review_summary(
        self,
        brand: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get summary statistics of expert reviews.

        Args:
            brand: Optional brand filter

        Returns:
            Summary dict with counts by status. ``pending`` / ``approved`` /
            ``rejected`` / ``expired`` partition the rows; ``expiring_soon`` is
            a SUBSET of ``approved`` (within 14 days). ``expired`` is derived
            from ``valid_until`` via ``approval_validity`` -- the same predicate
            the gate's queries use -- and a NULL ``valid_until`` is a permanent
            approval: counted in ``approved``, never in ``expired`` or
            ``expiring_soon`` (#1972).

        Raises:
            The underlying client error on a query failure, after logging it
            (R3). All-zero counts on an outage rendered as real counts on the
            Expert Reviews page with HTTP 200; the route now maps the raise
            to 503. Only the no-client early return keeps the zero dict.
        """
        if not self.client:
            return {
                "pending": 0,
                "approved": 0,
                "rejected": 0,
                "expired": 0,
                "expiring_soon": 0,
            }

        try:
            # Get all reviews for counting
            query = self.client.table(self.table_name).select("approval_status, valid_until")
            if brand:
                query = query.eq("brand", brand)

            result = await query.execute()
            reviews = result.data or []

            today = date.today()
            soon = today + timedelta(days=14)

            pending = 0
            approved = 0
            rejected = 0
            expired = 0
            expiring_soon = 0

            for r in reviews:
                status = r.get("approval_status")

                if status == "pending":
                    pending += 1
                elif status == "rejected":
                    rejected += 1
                elif status == "approved":
                    # ONE definition (#1972): approval_validity() is the same
                    # predicate _apply_active_validity() sends to PostgREST.
                    exp_date = _valid_until_date(r)
                    if exp_date is None:
                        # NULL valid_until = permanent: approved, never expiring.
                        approved += 1
                    elif approval_validity(r, today) == "expired":
                        expired += 1
                    elif exp_date <= soon:
                        expiring_soon += 1
                        approved += 1
                    else:
                        approved += 1

            return {
                "pending": pending,
                "approved": approved,
                "rejected": rejected,
                "expired": expired,
                "expiring_soon": expiring_soon,
            }
        except Exception as e:
            # R3: log, then re-raise -- zero counts on an outage are a
            # plausible-wrong value on a user-facing page.
            logger.error(f"Failed to get review summary: {e}")
            raise
