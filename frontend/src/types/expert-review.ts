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
