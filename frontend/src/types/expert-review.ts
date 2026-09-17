/**
 * Expert Review API Types
 * =======================
 *
 * TypeScript types for the E2I expert-review queue (R6-F2).
 * Based on src/api/schemas/expert_review.py backend schemas.
 *
 * The human-in-the-loop loop for causal-DAG validation:
 * - A REVIEW-band causal estimate creates a `pending` expert_reviews row.
 * - An operator reads the queue and resolves (approves/rejects) it.
 *
 * This file is a HAND-WRITTEN MIRROR of the expert-review schemas in
 * `src/types/generated/api.ts`. Any backend schema change must be copied here by
 * hand in the same PR that regenerates api.ts; `expert-review.drift.test.ts`
 * (#2092) fails the typecheck when the two disagree, and lists the deliberate
 * relaxations (e.g. `version_count?`).
 *
 * @module types/expert-review
 */

/** DAG-discovery gate vocabulary (#1991 debt 4) — its own vocabulary, distinct
 * from `RefutationGate` / `ExpertReviewDecision` (types/causal.ts). */
export type DiscoveryGate = 'accept' | 'review' | 'reject' | 'augment';

/**
 * Sanitized causal-graph snapshot captured when the review was created
 * (mig 097). Null/absent for rows created before snapshot capture existed —
 * the hash is one-way, so those cannot be rendered.
 */
export interface DagStructure {
  nodes: string[];
  /** Arity-enforced (#1991 debt 4): mirrors the backend's Tuple[str, str]. */
  edges: [string, string][];
  treatment_nodes?: string[] | null;
  outcome_nodes?: string[] | null;
  adjustment_sets?: string[][] | null;
  augmented_edges?: [string, string][] | null;
  discovery_gate_decision?: DiscoveryGate | null;
  confidence?: number | null;
  dag_version_hash?: string | null;
}

/**
 * The structural delta between two DAG snapshots of the same estimand
 * (#1991 debt 3, migration 141). Every list is sorted by the engine, so the
 * same delta renders identically across calls.
 *
 * `adjustment_sets_*` carries the covariate delta: a covariate change with an
 * unchanged graph is still a new version and reports `is_changed` true.
 *
 * Edges and adjustment sets are `string[][]` (NOT the arity-enforced 2-tuple
 * `DagStructure.edges` uses) — that mirrors the backend's `List[List[str]]`.
 */
export interface DagChanges {
  nodes_added: string[];
  nodes_removed: string[];
  edges_added: string[][];
  edges_removed: string[][];
  adjustment_sets_added: string[][];
  adjustment_sets_removed: string[][];
  is_changed: boolean;
  old_hash?: string | null;
  new_hash?: string | null;
}

/**
 * One `expert_review_versions` row (migration 141): a structure version of a
 * review, in timeline order.
 *
 * The table is a TIMELINE, not a set — a revert (A → B → A) appends a third
 * row rather than being suppressed, so two versions may carry the same hash.
 * `changes` is the delta against the version BEFORE it, null on the oldest.
 */
export interface ReviewVersion {
  version_id: string;
  dag_version_hash: string;
  adjustment_set_hash?: string | null;
  dag_structure_json?: DagStructure | null;
  query_id?: string | null;
  created_at?: string | null;
  changes?: DagChanges | null;
}

/** Verdict vocabulary of the advisory agent assessment. */
export type AssessmentVerdict = 'supports' | 'concern' | 'unclear' | 'no_evidence';

/** One checklist question graded by the agent (advisory only). */
export interface AssessmentItem {
  id: string;
  question: string;
  verdict: AssessmentVerdict;
  rationale: string;
}

/**
 * Advisory agent assessment of the reviewer checklist. `is_fallback` marks the
 * deterministic (no-LLM) grading; evidence counts say what it was grounded in.
 */
export interface AgentAssessment {
  items: AssessmentItem[];
  is_fallback: boolean;
  evidence?: {
    refutation_tests: number;
    has_dag_structure: boolean;
  };
}

/**
 * A single pending expert review.
 *
 * Mirrors the `PendingReviewItem` backend schema / the
 * `v_pending_expert_reviews` view columns.
 */
export interface PendingReviewItem {
  review_id: string;
  review_type?: string | null;
  dag_version_hash?: string | null;
  /**
   * The adjustment-set half of the review's current version identity
   * (#1991 debt 3, migration 142); null = unknown. The DAG hash alone cannot
   * see a covariate-only change, because the backend's DAG hash excludes
   * adjustment sets — so the resolve form echoes BOTH halves.
   */
  adjustment_set_hash?: string | null;
  brand?: string | null;
  treatment_variable?: string | null;
  outcome_variable?: string | null;
  analysis_context?: string | null;
  created_at?: string | null;
  days_pending?: number | null;
  dag_structure_json?: DagStructure | null;
  agent_assessment_json?: AgentAssessment | null;
  /**
   * How many structure versions this review's estimand has (#1991 debt 3).
   * The backend always sends it (server default 1); kept OPTIONAL here so a
   * partial row literal stays valid — read it as `version_count ?? 1`.
   */
  version_count?: number;
  /** When the estimand's newest version was appended; null if never changed. */
  last_changed_at?: string | null;
}

/**
 * Response for GET /expert-reviews/pending.
 */
export interface PendingReviewsResponse {
  reviews: PendingReviewItem[];
  total: number;
}

/**
 * Approval status vocabulary — matches the backend `submit_review` validation.
 */
export type ReviewApprovalStatus = 'approved' | 'rejected';

/**
 * Request body for POST /expert-reviews/{review_id}/resolve.
 */
export interface ResolveReviewRequest {
  approval_status: ReviewApprovalStatus;
  /**
   * The structure version the reviewer's form displayed (#1991 debt 3). REQUIRED:
   * a review's DAG can ADVANCE while the form is open, and the backend applies
   * the resolution only while the review still carries this hash — a mismatch is
   * a 409 telling the reviewer to reload, never a silent sign-off of a structure
   * nobody looked at.
   */
  dag_version_hash: string;
  /**
   * The adjustment-set half of that same version (#1991 debt 3, codex round 2).
   * The KEY is required; the VALUE may be null when the review carried no known
   * adjustment set. An ADJUSTMENT-ONLY advance leaves the DAG hash untouched, so
   * sending the hash alone let a form opened on the previous covariates resolve
   * a structure nobody looked at. Omitting the key is a 422, deliberately: "the
   * form did not send this" and "the review had no adjustment set" are different
   * facts, and only the second may resolve a null-carrying row.
   */
  adjustment_set_hash: string | null;
  checklist: Record<string, unknown>;
  comments?: Record<string, unknown> | null;
  concerns_raised?: string[] | null;
  conditions?: string | null;
  validity_days?: number;
}

/**
 * Response for POST /expert-reviews/{review_id}/resolve.
 */
export interface ResolveReviewResponse {
  review_id: string;
  approval_status: string;
  success: boolean;
}

/**
 * Response for GET /expert-reviews/summary.
 */
export interface ReviewSummaryResponse {
  pending: number;
  approved: number;
  rejected: number;
  /** BLOCK-band reviews resolved by migration 140 (#1991 debt 3). */
  superseded: number;
  expired: number;
  expiring_soon: number;
}

/**
 * Response for POST /expert-reviews/{review_id}/assessment.
 *
 * `cached` marks a replay of the stored assessment; `persisted` is honest
 * about whether a fresh assessment reached the DB.
 */
export interface AgentAssessmentResponse {
  review_id: string;
  assessment: AgentAssessment;
  cached: boolean;
  persisted: boolean;
}

/**
 * One expert_reviews row in ANY status (GET /expert-reviews/{review_id}).
 * Extends the pending shape with the resolution columns.
 */
export interface ReviewRecord extends PendingReviewItem {
  /** pending / approved / rejected (expired is derived at read time from valid_until) */
  approval_status?: string | null;
  /**
   * The REQUESTER, not the resolver: the gate stores the originating query id
   * here (expert_review_gate.py create_review(reviewer_id=requester_id)).
   * Never render it as the reviewer.
   */
  reviewer_id?: string | null;
  /** The resolving operator's name, written on resolve; null when not recorded. */
  reviewer_name?: string | null;
  /** The resolving operator's email, written on resolve; null when not recorded. */
  reviewer_email?: string | null;
  /** Approval-only timestamp; a rejection never sets it. */
  approved_at?: string | null;
  /**
   * When the review was resolved, for BOTH statuses (migration 136). Null for
   * rows resolved before the migration — unknown stays unknown.
   */
  resolved_at?: string | null;
  valid_from?: string | null;
  valid_until?: string | null;
  concerns_raised?: string[] | null;
  conditions?: string | null;
  checklist_json?: Record<string, unknown> | null;
  comments_json?: Record<string, unknown> | null;
  supersedes_review_id?: string | null;
  /**
   * Structural delta between this review's snapshot and that of the
   * next-OLDER review of the same estimand. Populated on `history` entries
   * only: null on the top-level `review` and on the OLDEST history entry
   * (nothing to diff against). Computed by the detail route, never stored.
   */
  changes_from_previous?: DagChanges | null;
}

/**
 * Response for GET /expert-reviews/{review_id}: the row plus every review of
 * the same DAG structure (newest first, expired included).
 */
export interface ExpertReviewDetailResponse {
  review: ReviewRecord;
  history: ReviewRecord[];
  /**
   * This review's own `expert_review_versions` timeline (migration 141),
   * OLDEST first, each row carrying `changes` against the one before it.
   * Empty for a review minted before the versions table.
   *
   * A set of FACTS, not a queue: it may END on a row the review is NOT on (a
   * run that recorded its version and then lost the compare-and-set advance
   * leaves an orphan after the winner), so `versions[versions.length - 1]` is
   * not "the version under review". Use `current_version_id`.
   */
  versions: ReviewVersion[];
  /**
   * The `version_id` of the timeline entry carrying the review's CURRENT
   * version identity — the pair (`dag_version_hash`, `adjustment_set_hash`) the
   * review row itself holds. The ONLY entry whose `changes` describes what the
   * reviewer is being asked to approve.
   *
   * Null when no entry carries that pair (a review minted before the versions
   * table, or one whose first version was never recorded): render no delta at
   * all rather than a plausible-wrong one.
   */
  current_version_id?: string | null;
}
