/**
 * Expert Review API Query Hooks
 * =============================
 *
 * TanStack Query hooks for the E2I expert-review queue (R6-F2).
 * Provides type-safe data fetching for the admin review-queue UI.
 *
 * - usePendingReviews: read the oldest-first pending queue
 * - useReviewSummary:  read status counts
 * - useResolveReview:  approve/reject a review, then invalidate the queue/summary
 *
 * @module hooks/api/use-expert-review
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import {
  generateReviewAssessment,
  getPendingReviews,
  getReviewSummary,
  resolveReview,
} from '@/api/expert-review';
import type {
  AgentAssessmentResponse,
  PendingReviewsResponse,
  ResolveReviewRequest,
  ResolveReviewResponse,
  ReviewSummaryResponse,
} from '@/types/expert-review';
import type { ApiError } from '@/lib/api-client';

/**
 * Hook to fetch the pending expert-review queue.
 *
 * @param params - Optional brand / reviewer_id / limit filters
 * @param options - Additional TanStack Query options
 */
export function usePendingReviews(
  params?: { brand?: string; reviewer_id?: string; limit?: number },
  options?: Omit<
    UseQueryOptions<PendingReviewsResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<PendingReviewsResponse, ApiError>({
    queryKey: queryKeys.expertReviews.pending(params),
    queryFn: () => getPendingReviews(params),
    ...options,
  });
}

/**
 * Hook to fetch expert-review status counts.
 *
 * @param params - Optional brand filter
 * @param options - Additional TanStack Query options
 */
export function useReviewSummary(
  params?: { brand?: string },
  options?: Omit<
    UseQueryOptions<ReviewSummaryResponse, ApiError>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery<ReviewSummaryResponse, ApiError>({
    queryKey: queryKeys.expertReviews.summary(params),
    queryFn: () => getReviewSummary(params),
    ...options,
  });
}

/** Variables for the resolve mutation. */
export interface ResolveReviewVariables {
  reviewId: string;
  body: ResolveReviewRequest;
}

/**
 * Hook to resolve (approve/reject) an expert review.
 *
 * On success, invalidates BOTH the pending queue and the summary so the UI
 * reflects the resolution immediately.
 *
 * @param options - Additional TanStack mutation options
 */
export function useResolveReview(
  options?: Omit<
    UseMutationOptions<ResolveReviewResponse, ApiError, ResolveReviewVariables>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<ResolveReviewResponse, ApiError, ResolveReviewVariables>({
    mutationFn: ({ reviewId, body }) => resolveReview(reviewId, body),
    onSuccess: () => {
      // A resolution changes the pending queue AND the status counts. Use the
      // bare ['…', 'pending'] / ['…', 'summary'] prefixes so partial (prefix)
      // matching covers every param-keyed cache entry.
      queryClient.invalidateQueries({
        queryKey: [...queryKeys.expertReviews.all(), 'pending'],
      });
      queryClient.invalidateQueries({
        queryKey: [...queryKeys.expertReviews.all(), 'summary'],
      });
    },
    ...options,
  });
}

/** Variables for the assessment mutation. */
export interface ReviewAssessmentVariables {
  reviewId: string;
  /** Regenerate even when a cached assessment exists. */
  force?: boolean;
}

/**
 * Hook to generate (or fetch cached) the advisory agent assessment.
 *
 * On success, invalidates the pending queue so the row's cached
 * `agent_assessment_json` stays in sync on the next refetch.
 *
 * @param options - Additional TanStack mutation options
 */
export function useReviewAssessment(
  options?: Omit<
    UseMutationOptions<AgentAssessmentResponse, ApiError, ReviewAssessmentVariables>,
    'mutationFn'
  >
) {
  const queryClient = useQueryClient();

  return useMutation<AgentAssessmentResponse, ApiError, ReviewAssessmentVariables>({
    mutationFn: ({ reviewId, force }) => generateReviewAssessment(reviewId, force),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [...queryKeys.expertReviews.all(), 'pending'],
      });
    },
    ...options,
  });
}
