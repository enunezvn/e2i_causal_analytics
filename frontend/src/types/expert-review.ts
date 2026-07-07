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
 * @module types/expert-review
 */

/**
 * Sanitized causal-graph snapshot captured when the review was created
 * (mig 097). Null/absent for rows created before snapshot capture existed —
 * the hash is one-way, so those cannot be rendered.
 */
export interface DagStructure {
  nodes: string[];
  edges: string[][];
  treatment_nodes?: string[];
  outcome_nodes?: string[];
  adjustment_sets?: string[][];
  augmented_edges?: string[][];
  discovery_gate_decision?: string | null;
  confidence?: number | null;
  dag_version_hash?: string | null;
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
  brand?: string | null;
  treatment_variable?: string | null;
  outcome_variable?: string | null;
  analysis_context?: string | null;
  created_at?: string | null;
  days_pending?: number | null;
  dag_structure_json?: DagStructure | null;
  agent_assessment_json?: AgentAssessment | null;
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
