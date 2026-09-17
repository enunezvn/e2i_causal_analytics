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

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Mapping, Optional

from src.repositories.expert_review_versions import ExpertReviewVersionTimeline
from src.repositories.json_utils import to_plain_json
from src.repositories.query_utils import match_nullable_column

# ``match_nullable_column`` is the ONE definition of "the review is still on this value", shared by
# ``submit_review``, ``update_agent_assessment`` and ``advance_review`` across both halves of the
# pair -- ``adjustment_set_hash`` (migration 142) and ``dag_version_hash`` (nullable, migration 141)
# -- so the three guards cannot drift apart (codex round 2).

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


def estimand_key_for(brand: Optional[str], treatment: Optional[str], outcome: Optional[str]) -> str:
    """Review identity (migration 140): ``lower(brand):treatment:outcome``, null-safe.

    Mirrors the STORED GENERATED column defined in
    ``database/migrations/140_expert_reviews_estimand_key.sql`` operand for
    operand (``lower()`` over each field, ``COALESCE`` to ``''``, joined by
    ``':'``), so a Python lookup lands on the same row PostgREST derived --
    change one and change the other. Used ONLY for lookups: the column is
    generated, so writers never send it -- an insert that supplied it would be
    rejected outright.

    ASCII assumption: Postgres ``lower()`` folds case under the database
    collation while Python ``str.lower()`` follows Unicode, so the two can
    disagree on non-ASCII text (a Turkish dotless I, say). Brands and variable
    names are ASCII identifiers today, which is the only case
    test_estimand_key_matches_migration_140_expression pins; a non-ASCII brand
    would need that equivalence re-measured against the live collation.

    A DAG hash is deliberately absent. The hash is the structure VERSION a
    review currently covers (expert_review_versions, migration 141), not the
    review's identity: a covariate or structure change must UPDATE the pending
    review of an estimand, not mint a sibling.
    """
    return f"{(brand or '').lower()}:{(treatment or '').lower()}:{(outcome or '').lower()}"


class ExpertReviewRepository(ExpertReviewVersionTimeline):
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

    Structure versions and the pair compare-and-set that moves a review onto
    one live in :class:`ExpertReviewVersionTimeline`, which this class
    inherits: ``record_version``, ``append_version``, ``advance_review``,
    ``get_latest_version``, ``get_versions``, ``get_versions_for_reviews``.

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
        supersedes_review_id: Optional[str] = None,
        adjustment_set_hash: Optional[str] = None,
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
            supersedes_review_id: The earlier review of this estimand that this
                one replaces, when the caller is minting a successor (existing
                column; ``renew_review`` sets it the same way)
            adjustment_set_hash: The adjustment-set half of the structure's
                version identity (migration 142). Stored beside
                ``dag_version_hash`` so the minted row carries its WHOLE
                identity from the start -- the version-1 append's
                compare-and-set has a pair to expect, and every guard keyed on
                the row (resolution, the assessment persist, the advance) can
                bind to both halves. None means the adjustment set is UNKNOWN
                (no structure in scope), and is omitted from the payload so the
                column's NULL stands for exactly that.

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
            # #1992: JSON OBJECTS, not json.dumps'ed strings, so PostgREST
            # stores a jsonb object (queryable) instead of a jsonb string
            # scalar -- mirrors causal_validation.py (lane 1, migration 135).
            # to_plain_json also coerces datetime/enum/numpy values and maps
            # NaN to null where json.dumps used to raise (today's inputs here
            # are already plain dicts, so nothing observable changes).
            "checklist_json": to_plain_json(checklist) if checklist else None,
            "related_validation_ids": related_validation_ids,
            "dag_structure_json": to_plain_json(dag_structure) if dag_structure else None,
            "supersedes_review_id": supersedes_review_id,
            # The adjustment-set half of the version identity (migration 142).
            # Stripped when None by the rule below, like every other unset
            # column: an omitted key leaves the column NULL, which is what
            # "unknown adjustment set" IS -- and it keeps the payload minimal.
            "adjustment_set_hash": adjustment_set_hash,
        }

        # estimand_key is NOT in this row and must never be: migration 140 made
        # it GENERATED ALWAYS AS ... STORED, and PostgREST rejects an insert
        # that supplies a generated column.

        # Remove None values
        row = {k: v for k, v in row.items() if v is not None}

        try:
            result = await self.client.table(self.table_name).insert(row).execute()
            review_id = result.data[0]["review_id"] if result.data else None
            logger.info(f"Created expert review {review_id} for DAG {dag_version_hash}")
            return review_id
        except Exception as e:
            # M-reach1: a concurrent creator may have won the race and inserted the
            # pending row first — the partial UNIQUE index uq_er_pending_estimand
            # (mig 140, keyed on the ESTIMAND; it replaced 062's
            # uq_er_pending_dag_brand) then rejects THIS duplicate with a 23505
            # unique violation. Recover ONLY for that case (return the winner's
            # pending review); let any other failure (transient connection/timeout,
            # schema error) surface via the error log rather than masking it behind
            # a possibly-stale pending row (codex MEDIUM).
            err = str(e).lower()
            if "23505" in err or "unique" in err or "duplicate key" in err:
                key = estimand_key_for(brand, treatment_variable, outcome_variable)
                existing = await self._find_pending_review_id(key)
                if existing is not None:
                    logger.info(
                        "create_review: a pending review already exists for estimand "
                        f"{key}; returning it (concurrent create / unique-violation "
                        "recovery)."
                    )
                    return existing
            logger.error(f"Failed to create expert review: {e}")
            return None

    async def _find_pending_review_id(self, estimand_key: str) -> Optional[str]:
        """M-reach1: return the review_id of the existing PENDING review of this
        ESTIMAND, if any -- the row that won the uq_er_pending_estimand race.

        The lookup is a plain equality on the generated ``estimand_key`` column,
        which is exactly what the partial unique index keys on, so it cannot
        disagree with the index that rejected our insert. The old brand
        normalisation note is obsolete: migration 140's expression already
        COALESCEs every operand to ``''``, so a NULL brand and an explicit
        empty-string brand derive the SAME key and both are matched here (the
        pre-140 ``IS NULL`` lookup could not match a ``brand=''`` winner).
        Compute the argument with :func:`estimand_key_for`.
        """
        if not self.client:
            return None
        try:
            query = (
                self.client.table(self.table_name)
                .select("review_id")
                .eq("estimand_key", estimand_key)
                .eq("approval_status", "pending")
            )
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
        *,
        expected_dag_version_hash: str,
        expected_adjustment_set_hash: Optional[str],
    ) -> bool:
        """
        Submit a completed expert review.

        Only a PENDING row can be resolved (R2, lane-1971 audit): the UPDATE
        itself carries ``approval_status = 'pending'``, so an already-resolved
        row -- including an OLDER approval while a NEWER one exists -- matches
        zero rows and returns False (the route surfaces that as 404). Before
        this the filter was ``review_id`` alone and the pending-only claim was
        documentation, not enforcement.

        The resolution is also bound to the STRUCTURE VERSION the reviewer saw
        (codex round-1 HIGH): the UPDATE carries
        ``dag_version_hash = expected_dag_version_hash``, so an approval of the
        form opened on h1 can never land on a review a concurrent run has since
        advanced to h2 (``append_version``). Pending-only and hash-bound are the
        same mechanism -- one UPDATE whose filters ARE the precondition -- and a
        mismatch matches zero rows and returns False, which the route
        disambiguates into 409 (pending but advanced) vs 404 (gone or already
        resolved). A row whose ``dag_version_hash`` is NULL (the column is
        nullable; no such row exists live, measured 2026-09-14) therefore
        matches no hash and is refused rather than resolved blind.

        That binding is to the PAIR, not to the hash alone (codex round-2 HIGH
        1). ``compute_dag_hash`` EXCLUDES adjustment sets, so an ADJUSTMENT-ONLY
        advance -- (h1, adj-W) to (h1, adj-Z), the same DAG with different
        covariates -- leaves the hash untouched, and a hash-only filter matched
        it: a reviewer approved covariates their form never displayed. The
        UPDATE now also carries ``adjustment_set_hash`` (migration 142), matched
        the way the value requires: ``eq`` for a known hash, ``is_`` (IS NULL)
        for None. None is matched EXPLICITLY rather than by omitting the filter,
        because an omitted filter matches every row and would restore exactly
        the defect this closes -- "the review's adjustment set is unknown" is a
        precondition to check, not one to skip.

        Resolution provenance (lane 1, codex whole-diff HIGH F1 + iter-2 HIGH
        F1): BOTH statuses stamp ``resolved_at = now()`` (migration 136;
        ``approved_at`` stays approval-only), and on resolution the reviewer
        display fields ``reviewer_name`` / ``reviewer_email`` describe the
        RESOLVER -- they are written EXPLICITLY on every resolution, JSON null
        when unknown, overwriting whatever a renewal pre-filled
        (``renew_review`` inserts the REQUESTER's name/email into these same
        columns). ``reviewer_id`` is deliberately NOT written here: the gate
        stores the REQUESTER in it -- the originating query id
        (src/causal_engine/expert_review_gate.py
        create_review(reviewer_id=requester_id), ~:392) -- and that breadcrumb
        must survive the resolution. Unknown stays unknown: a null, never a
        placeholder and never a leftover requester.

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
            expected_dag_version_hash: The structure version the reviewer's form
                displayed; the resolution applies only while the review still
                carries it (keyword-only and required -- a caller that cannot
                name the version it is resolving must not resolve)
            expected_adjustment_set_hash: The other half of that version, the
                adjustment-set hash the form displayed. Keyword-only and
                REQUIRED, though its VALUE may be None: None asserts "the review
                carried no known adjustment set", which is a precondition the
                UPDATE checks with IS NULL, not an absence of one

        Returns:
            True if exactly this pending row was resolved AT THE EXPECTED
            VERSION, False otherwise (nonexistent, already resolved, advanced to
            another structure -- including one whose DAG is unchanged and only
            the adjustment set moved -- or persistence error)
        """
        if not self.client:
            return False

        if approval_status not in ("approved", "rejected"):
            logger.error(f"Invalid approval_status: {approval_status}")
            return False

        update_data = {
            "approval_status": approval_status,
            # #1992: JSON OBJECTS, not json.dumps'ed strings.
            "checklist_json": to_plain_json(checklist),
            "comments_json": to_plain_json(comments) if comments else None,
            "concerns_raised": concerns_raised,
            "conditions": conditions,
            # Decision time for BOTH statuses (migration 136). The literal is cast
            # by Postgres to the current timestamp (measured 2026-09-09:
            # select 'now()'::timestamptz, and json_populate_record over
            # {"approved_at":"now()"}, both return now()).
            "resolved_at": "now()",
        }

        if approval_status == "approved":
            valid_until = date.today() + timedelta(days=validity_days)
            update_data["valid_from"] = date.today().isoformat()
            update_data["valid_until"] = valid_until.isoformat()
            update_data["approved_at"] = "now()"

        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}
        # ... except the resolver's identity, which is written on EVERY
        # resolution -- null when unknown -- so a requester's name/email that a
        # renewal pre-filled can never stand as the reviewer (iter-2 HIGH F1).
        update_data["reviewer_name"] = reviewer_name
        update_data["reviewer_email"] = reviewer_email

        try:
            query = (
                self.client.table(self.table_name)
                .update(update_data)
                .eq("review_id", review_id)
                .eq("approval_status", "pending")
                .eq("dag_version_hash", expected_dag_version_hash)
            )
            result = await match_nullable_column(
                query, "adjustment_set_hash", expected_adjustment_set_hash
            ).execute()
            # FIX B (codex HIGH): a zero-row update (nonexistent or already-resolved
            # review_id) matches nothing — supabase-py returns the updated rows in
            # ``result.data`` (same convention as base.py:131), so empty data means
            # nothing was touched. Returning True there is a fabricated success that
            # would make the route 200 a record it never changed.
            if not result.data:
                logger.warning(
                    f"submit_review matched no rows for {review_id} at version "
                    f"({expected_dag_version_hash}, {expected_adjustment_set_hash}) "
                    "(nonexistent, already-resolved, or advanced to another structure); "
                    "returning False"
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
        *,
        for_dag_version_hash: Optional[str] = None,
        for_adjustment_set_hash: Optional[str] = None,
    ) -> bool:
        """Cache an advisory agent assessment on the review row (mig 097).

        Writes ``agent_assessment_json`` ONLY — never touches
        ``checklist_json``, which remains the human reviewer's own record.

        ``for_dag_version_hash`` binds the write to the structure the assessment
        actually GRADED (codex round-1 HIGH): the build reads a DAG snapshot and
        its linked refutation evidence, and a concurrent run can advance the
        review while it runs. Filtering the UPDATE on the hash makes a build of
        the superseded structure match zero rows instead of overwriting the new
        version's cache with a grading of the old one. Zero rows is the existing
        False contract, which the route reports as ``persisted: false`` — the
        assessment is still returned to whoever asked for it.

        The guard is on the PAIR (codex round-2 HIGH 2). A build that graded
        (h1, adj-W) still persisted after an ADJUSTMENT-ONLY advance to
        (h1, adj-Z) had cleared the cache, because the DAG hash never moved --
        the review UI then showed a grading of covariates the review no longer
        covers, marked ``persisted: true``. When ``for_dag_version_hash`` is
        given, BOTH halves filter the UPDATE (``for_adjustment_set_hash`` via
        eq-or-IS-NULL, see :func:`match_nullable_column`).

        The two are given TOGETHER or not at all. ``for_dag_version_hash=None``
        means "this row has no version to bind to" -- a pre-141 row that never
        carried a hash -- and applies NO version filter, the pre-#1991
        behaviour. Filtering the adjustment half alone there would refuse every
        write instead of guarding one.

        Args:
            review_id: The review whose cache is written
            assessment: The advisory grading payload
            for_dag_version_hash: When given, write only while the review still
                carries this structure version
            for_adjustment_set_hash: The adjustment-set half of that version.
                Read together with ``for_dag_version_hash``: it applies only
                when that one is given, and a None value there means the review
                must still carry an UNKNOWN adjustment set (IS NULL), not that
                the half is unchecked

        Returns:
            True when exactly this row was updated; False on zero-row match
            (nonexistent review, or one whose structure moved) or persistence
            error — fail-closed, mirroring ``submit_review``.
        """
        if not self.client:
            return False

        try:
            query = (
                self.client.table(self.table_name)
                # #1992: JSON OBJECT, not a json.dumps'ed string.
                .update({"agent_assessment_json": to_plain_json(assessment)})
                .eq("review_id", review_id)
            )
            if for_dag_version_hash is not None:
                query = query.eq("dag_version_hash", for_dag_version_hash)
                query = match_nullable_column(query, "adjustment_set_hash", for_adjustment_set_hash)
            result = await query.execute()
            if not result.data:
                logger.warning(
                    f"update_agent_assessment matched no rows for {review_id} "
                    f"(version filter: {for_dag_version_hash}, {for_adjustment_set_hash}); "
                    "returning False"
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

        # #1992: JSON OBJECT, not a json.dumps'ed string.
        update_data: Dict[str, Any] = {"dag_structure_json": to_plain_json(dag_structure)}
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

    async def get_reviews_for_estimand(
        self,
        estimand_key: str,
        include_expired: bool = True,
    ) -> List[Dict[str, Any]]:
        """Every review of this ESTIMAND, newest first -- the history the gate and
        the detail route read.

        Sibling of ``get_reviews_for_dag``, keyed on identity instead of on one
        structure version (migration 140). ``include_expired`` defaults to True
        here: this is the audit view of an estimand, and a lapsed approval is
        part of that history, including expired rows by design. Pass False for
        the active-approval predicate (``_apply_active_validity``).

        Args:
            estimand_key: The generated key -- build it with ``estimand_key_for``
            include_expired: When True (default) every row is returned, expired
                approvals included; when False, rows whose ``valid_until`` has
                passed are dropped (a NULL ``valid_until`` is kept: permanent)

        Returns:
            Review records for the estimand, newest ``created_at`` first

        Raises:
            The underlying client error on a query failure, after logging it
            (R1/R3 convention of this module): an empty history reads as "this
            estimand was never reviewed", which an outage must not fake. The
            no-client early return ([]) is unchanged.
        """
        if not self.client:
            return []

        try:
            query = (
                self.client.table(self.table_name)
                .select("*")
                .eq("estimand_key", estimand_key)
                .order("created_at", desc=True)
            )

            if not include_expired:
                query = _apply_active_validity(query)

            result = await query.execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get reviews for estimand: {e}")
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
            # #2090: the renewal is a pending row of the ORIGINAL's estimand, so
            # uq_er_pending_estimand (mig 140) rejects it with a 23505 whenever
            # that estimand already has a pending review (a concurrent mint or
            # renewal). Recover exactly as create_review does: return the pending
            # row, and only for a unique violation.
            err = str(e).lower()
            if "23505" in err or "unique" in err or "duplicate key" in err:
                key = estimand_key_for(
                    original.get("brand"),
                    original.get("treatment_variable"),
                    original.get("outcome_variable"),
                )
                existing = await self._find_pending_review_id(key)
                if existing is not None:
                    logger.info(
                        "renew_review: a pending review already exists for estimand "
                        f"{key}; returning it (unique-violation recovery)."
                    )
                    return existing
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
            ``rejected`` / ``superseded`` / ``expired`` partition the rows;
            ``superseded`` is a STORED status (migration 140): a review that can
            no longer change an outcome -- written today only by migration 140's
            BLOCK-band resolution; a review re-opened on a new hash stays
            ``approved`` (spec §7) and is linked from its successor's
            ``supersedes_review_id``, not by this status. It is a resolution,
            never a queue item, so it is counted apart from ``pending``. ``expiring_soon`` is
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
                "superseded": 0,
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
            superseded = 0
            expired = 0
            expiring_soon = 0

            for r in reviews:
                status = r.get("approval_status")

                if status == "pending":
                    pending += 1
                elif status == "rejected":
                    rejected += 1
                elif status == "superseded":
                    superseded += 1
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
                "superseded": superseded,
                "expired": expired,
                "expiring_soon": expiring_soon,
            }
        except Exception as e:
            # R3: log, then re-raise -- zero counts on an outage are a
            # plausible-wrong value on a user-facing page.
            logger.error(f"Failed to get review summary: {e}")
            raise
