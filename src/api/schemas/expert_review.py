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
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _add_tuple_items_sibling(schema: Dict[str, Any]) -> None:
    """OpenAPI-lint fixup (#1991 debt 4, spectral ``array-items``): pydantic
    renders ``Tuple[str, str]`` as ``{"type": "array", "prefixItems": [...],
    "minItems": 2, "maxItems": 2}`` with NO sibling ``items`` key, which the
    spectral ruleset flags on any ``type: array`` schema. Add one here rather
    than widen the type -- OpenAPI 3.1's ``items`` governs elements past the
    prefix, which ``maxItems == len(prefixItems)`` already makes unreachable,
    so this documents the schema for the linter without loosening the arity
    contract. Handles both the plain field (``schema["items"]``) and the
    ``Optional``/``anyOf``-wrapped one (each ``schema["anyOf"][i]["items"]``)."""
    for candidate in [schema, *schema.get("anyOf", [])]:
        inner = candidate.get("items")
        if isinstance(inner, dict) and inner.get("type") == "array" and "items" not in inner:
            inner["items"] = {"type": "string"}


class DagStructureSnapshot(BaseModel):
    """The sanitized causal-graph snapshot (mig 097) with the DISCOVERY gate typed.

    Mirrors ``sanitize_dag_structure`` (src/causal_engine/expert_review_gate.py):
    ``nodes``/``edges`` are always coerced to plain lists (edge tuples ->
    2-element string lists) before the JSONB write, and ``_DAG_SNAPSHOT_KEYS``
    is exactly the optional field set below. ``extra="allow"`` keeps this
    forward-compatible with a future snapshot key without a schema change.
    """

    model_config = ConfigDict(extra="allow")

    nodes: List[str] = []
    edges: List[Tuple[str, str]] = Field(default=[], json_schema_extra=_add_tuple_items_sibling)
    treatment_nodes: Optional[List[str]] = None
    outcome_nodes: Optional[List[str]] = None
    adjustment_sets: Optional[List[List[str]]] = None
    augmented_edges: Optional[List[Tuple[str, str]]] = Field(
        default=None, json_schema_extra=_add_tuple_items_sibling
    )
    discovery_gate_decision: Optional[Literal["accept", "review", "reject", "augment"]] = None
    confidence: Optional[float] = None
    dag_version_hash: Optional[str] = None


def parse_json_column(value: Any) -> Any:
    """Shared ``mode="before"`` body for JSONB columns the repo writes as
    ``json.dumps`` strings: parse a string, keep only a dict, pass anything
    else through untouched.

    Public because ``routes/expert_review.py`` reuses it on the RAW column value
    before diffing two snapshots -- one definition of "how this column is read",
    rather than a second copy in the route.

    A non-string is returned AS IS, so a caller that needs a dict must check:
    ``expert_reviews.dag_structure_json`` carries no CHECK constraint, and a
    stored array reaches the caller unchanged.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return value


class DagChanges(BaseModel):
    """The structural delta between two DAG snapshots (``get_dag_changes``).

    Rendered on the review's version timeline and on its estimand history, so
    an operator can see WHAT moved between two versions rather than only that
    the hash changed. Every list is sorted by the engine -- a set-difference
    order would render the same delta differently between two calls.

    ``adjustment_sets_*`` carries the covariate delta (#1991 debt 3, spec §7): a
    covariate change with an unchanged graph is still a new version of the
    estimand, and reports ``is_changed`` True.

    No ``json_schema_extra`` sibling is needed here (unlike
    ``DagStructureSnapshot.edges``): these are ``List[List[str]]``, which
    pydantic renders with a plain ``items`` key, not the ``prefixItems``-only
    shape ``Tuple[str, str]`` produces and the spectral ``array-items`` rule
    flags (#1991 debt 4).
    """

    nodes_added: List[str] = []
    nodes_removed: List[str] = []
    edges_added: List[List[str]] = []
    edges_removed: List[List[str]] = []
    adjustment_sets_added: List[List[str]] = []
    adjustment_sets_removed: List[List[str]] = []
    is_changed: bool
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None


class ReviewVersion(BaseModel):
    """One ``expert_review_versions`` row (migration 141) -- a structure version
    of a review, in timeline order.

    The table is a TIMELINE, not a set: a revert (A -> B -> A) appends a third
    row rather than being suppressed, so two versions may carry the same hash.
    ``adjustment_set_hash`` is NULL on rows the migration backfilled, and
    ``dag_structure_json`` is object-or-NULL by CHECK constraint.
    """

    version_id: str
    dag_version_hash: str
    adjustment_set_hash: Optional[str] = None
    dag_structure_json: Optional[DagStructureSnapshot] = None
    query_id: Optional[str] = None
    created_at: Optional[datetime] = None
    #: The delta against the version before this one. None on the FIRST version:
    #: it has no predecessor, and an everything-added diff there would read as a
    #: real structure change.
    changes: Optional[DagChanges] = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("dag_structure_json", mode="before")
    @classmethod
    def _parse_json_string(cls, value: Any) -> Any:
        return parse_json_column(value)


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
    #: The adjustment-set half of the review's current version identity
    #: (migration 142); null = unknown. Paired with ``dag_version_hash`` it names
    #: the structure the review covers, and the resolve form echoes BOTH -- the
    #: DAG hash alone cannot see a covariate-only change, because
    #: ``compute_dag_hash`` excludes adjustment sets.
    adjustment_set_hash: Optional[str] = None
    brand: Optional[str] = None
    treatment_variable: Optional[str] = None
    outcome_variable: Optional[str] = None
    analysis_context: Optional[str] = None
    created_at: Optional[datetime] = None
    days_pending: Optional[float] = None
    dag_structure_json: Optional[DagStructureSnapshot] = None
    agent_assessment_json: Optional[Dict[str, Any]] = None
    #: How many structure versions this review has (#1991 debt 3). Set by the
    #: route from the batched ``expert_review_versions`` read, NOT a column. A
    #: review that never moved has ONE version (the mint), never zero -- zero
    #: would read as "this review has no structure".
    version_count: int = 1
    #: When the structure last moved: the newest version's ``created_at``, or
    #: the review's own ``created_at`` when it never moved.
    last_changed_at: Optional[datetime] = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("dag_structure_json", "agent_assessment_json", mode="before")
    @classmethod
    def _parse_json_string(cls, value: Any) -> Any:
        return parse_json_column(value)


class ReviewRecord(PendingReviewItem):
    """One ``expert_reviews`` row in ANY status (lane 1, spec §4.2).

    Extends ``PendingReviewItem`` with the resolution columns so the queue
    page's linked-review card can show who decided what, and until when.

    Provenance (codex whole-diff HIGH F1): ``reviewer_id`` holds the REQUESTER
    (the originating query id the gate wrote), never the resolver. The
    resolver is ``reviewer_name`` / ``reviewer_email`` and the decision time is
    ``resolved_at`` -- the resolution time for BOTH statuses since migration
    136, NULL for rows resolved before it (``approved_at`` is approval-only).
    """

    approval_status: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewer_name: Optional[str] = None
    reviewer_email: Optional[str] = None
    approved_at: Optional[datetime] = None
    #: Resolution time for both statuses (migration 136); None before it.
    resolved_at: Optional[datetime] = None
    valid_from: Optional[date] = None
    valid_until: Optional[date] = None
    concerns_raised: Optional[List[str]] = None
    conditions: Optional[str] = None
    checklist_json: Optional[Dict[str, Any]] = None
    comments_json: Optional[Dict[str, Any]] = None
    supersedes_review_id: Optional[str] = None
    changes_from_previous: Optional[DagChanges] = Field(
        default=None,
        description=(
            "Structural delta between this review's snapshot and that of the "
            "next-OLDER review of the same estimand. Populated on ``history`` "
            "entries only: it is None on the top-level ``review`` (validated "
            "from the stored row, which has no predecessor in scope) and on the "
            "OLDEST history entry (nothing to diff against). Computed by the "
            "detail route, never a stored column."
        ),
    )

    @field_validator("checklist_json", "comments_json", mode="before")
    @classmethod
    def _parse_resolution_json(cls, value: Any) -> Any:
        return parse_json_column(value)


class ExpertReviewDetailResponse(BaseModel):
    """``GET /expert-reviews/{review_id}``: the row, its ESTIMAND history, and its
    structure timeline.

    ``history`` is every review of the same ESTIMAND (migration 140), newest
    first, expired rows included. It was keyed on the DAG hash before #1991
    debt 3; since a covariate or structure change now ADVANCES the pending
    review of an estimand instead of minting a sibling, a hash-keyed history
    would drop every earlier structure of the same question. A row older than
    migration 140 carries no ``estimand_key``, and falls back to the same-hash
    history (the read the gate's rejection probe performs).

    ``versions`` is the review's own ``expert_review_versions`` timeline
    (migration 141), OLDEST first, each row carrying ``changes`` against the one
    before it. Empty for a review minted before the versions table. The timeline
    is a set of FACTS and may end on a row the review is not on, so
    ``current_version_id`` -- not ``versions[-1]`` -- names the version under
    review.
    """

    review: ReviewRecord
    history: List[ReviewRecord]
    versions: List[ReviewVersion] = []
    current_version_id: Optional[str] = Field(
        default=None,
        description=(
            "The ``version_id`` of the timeline row that carries the review's "
            "CURRENT version identity -- the pair (``dag_version_hash``, "
            "``adjustment_set_hash``) the review row itself holds. The timeline "
            "is a set of facts that MAY end on a different row: a run that "
            "recorded its version and then lost the compare-and-set advance "
            "leaves an orphan after the current one. Clients must render the "
            "delta of THIS row, never of the last entry. None when no row "
            "carries the review's pair (a review minted before the versions "
            "table, or one whose first version was never recorded) -- the "
            "honest answer is then no delta at all."
        ),
    )


class PendingReviewsResponse(BaseModel):
    """Response for ``GET /expert-reviews/pending``."""

    reviews: List[PendingReviewItem]
    total: int


class ResolveReviewRequest(BaseModel):
    """Request body for ``POST /expert-reviews/{review_id}/resolve``.

    ``approval_status`` is constrained to the SAME vocabulary
    ``submit_review`` validates against (repo :157) so a mismatched value is a
    422 (FastAPI validation) rather than a silent repo ``False``.

    ``dag_version_hash`` is REQUIRED (codex round-1 HIGH): a review's structure
    can ADVANCE while a reviewer's form is open (migration 141's timeline), and
    a resolution filtered on ``(review_id, pending)`` alone would apply the
    verdict to whatever structure the row carries NOW -- one nobody looked at.
    The form echoes the hash it displayed and the resolution applies only if
    the review still carries it; a mismatch is a 409, not a silent sign-off.

    ``adjustment_set_hash`` is the OTHER half of that version, and the key is
    required too (codex round-2 HIGH 1) -- though its VALUE may be null.
    ``compute_dag_hash`` EXCLUDES adjustment sets, so an ADJUSTMENT-ONLY advance
    leaves the DAG hash untouched: a form opened on (h1, adj-W) still resolved a
    review advanced to (h1, adj-Z), and the reviewer signed off covariates they
    were never shown. A MISSING key is a 422 rather than a null default,
    because "the form did not send this" and "the review had no adjustment set"
    are different facts and only the second may resolve a null-carrying row.
    """

    approval_status: Literal["approved", "rejected"]
    dag_version_hash: str = Field(
        min_length=1,
        description=(
            "The DAG version hash the reviewer's form displayed. The resolution "
            "applies only if the review still carries it; if the structure has "
            "advanced since the form was opened, the request is rejected with 409."
        ),
    )
    adjustment_set_hash: Optional[str] = Field(
        ...,
        description=(
            "The adjustment-set hash the reviewer's form displayed; null when the "
            "review carried none. Required as a KEY (a missing one is 422). The "
            "resolution applies only if the review still carries this exact pair: "
            "the DAG hash alone cannot see a covariate-only change, so a form "
            "opened before an adjustment-only advance is rejected with 409."
        ),
    )
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

    Mirrors ``ExpertReviewRepository.get_review_summary``.

    These counts are NOT all disjoint (#1972). ``pending`` / ``approved`` /
    ``rejected`` / ``superseded`` / ``expired`` partition the rows, but
    **``expiring_soon`` is a subset of ``approved``** -- a row approved and
    within 14 days of ``valid_until`` is counted in both. Summing all six
    double-counts those rows.
    ``expired`` is derived from ``valid_until`` at read time and is never
    a stored ``approval_status``. A NULL ``valid_until`` is a PERMANENT
    approval (the schema's ``v_active_expert_approvals`` labels it
    ``'permanent'``): counted in ``approved``, never in ``expired`` or
    ``expiring_soon``.
    """

    pending: int
    approved: int
    rejected: int
    #: STORED status (migration 140): a review that can no longer change an
    #: outcome. A partition member alongside pending/approved/rejected/expired,
    #: never a queue item -- written today only by migration 140's resolution of
    #: the BLOCK-band backlog.
    superseded: int
    #: Derived from ``valid_until`` at read time -- never a stored status.
    expired: int
    #: SUBSET of ``approved``, not a peer bucket. Do not add it to a total.
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
