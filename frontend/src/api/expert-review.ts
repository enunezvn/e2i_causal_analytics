/**
 * Expert Review API Client
 * ========================
 *
 * TypeScript API client functions for the E2I expert-review queue endpoints.
 * Uses the shared apiClient (get/post) for consistent error handling.
 *
 * Endpoints:
 * - GET  /expert-reviews/pending            : List pending reviews
 * - POST /expert-reviews/{review_id}/resolve : Approve/reject a review
 * - GET  /expert-reviews/summary            : Status counts
 *
 * @module api/expert-review
 */

import { get, post } from '@/lib/api-client';
import type {
  AgentAssessmentResponse,
  PendingReviewsResponse,
  ResolveReviewRequest,
  ResolveReviewResponse,
  ReviewSummaryResponse,
} from '@/types/expert-review';

const EXPERT_REVIEW_BASE = '/expert-reviews';

/**
 * List pending expert reviews (oldest-first).
 *
 * @param params - Optional brand / reviewer_id / limit filters
 * @returns The pending review queue and total count
 */
export async function getPendingReviews(params?: {
  brand?: string;
  reviewer_id?: string;
  limit?: number;
}): Promise<PendingReviewsResponse> {
  return get<PendingReviewsResponse>(`${EXPERT_REVIEW_BASE}/pending`, params);
}

/**
 * Resolve (approve/reject) a pending expert review.
 *
 * @param reviewId - The review identifier
 * @param body - Resolution payload (approval_status, checklist, comments?)
 * @returns The persisted resolution result
 */
export async function resolveReview(
  reviewId: string,
  body: ResolveReviewRequest
): Promise<ResolveReviewResponse> {
  return post<ResolveReviewResponse, ResolveReviewRequest>(
    `${EXPERT_REVIEW_BASE}/${reviewId}/resolve`,
    body
  );
}

/**
 * Get expert-review status counts.
 *
 * @param params - Optional brand filter
 * @returns Status counts (pending/approved/rejected/expired/expiring_soon)
 */
export async function getReviewSummary(params?: {
  brand?: string;
}): Promise<ReviewSummaryResponse> {
  return get<ReviewSummaryResponse>(`${EXPERT_REVIEW_BASE}/summary`, params);
}

/**
 * Generate (or fetch cached) the advisory agent assessment for a review.
 *
 * @param reviewId - The review identifier
 * @param force - Regenerate even when a cached assessment exists
 * @returns The assessment plus cached/persisted honesty flags
 */
export async function generateReviewAssessment(
  reviewId: string,
  force = false
): Promise<AgentAssessmentResponse> {
  return post<AgentAssessmentResponse>(
    `${EXPERT_REVIEW_BASE}/${reviewId}/assessment`,
    undefined,
    force ? { params: { force: true } } : undefined
  );
}
