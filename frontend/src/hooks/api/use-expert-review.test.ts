/**
 * Expert Review API Query Hooks Tests (R6-F2 Phase B3)
 * ====================================================
 *
 * Tests the TanStack Query hooks for the E2I expert-review queue:
 * - usePendingReviews fetches the pending queue
 * - useResolveReview posts a resolution + invalidates pending/summary keys
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import type {
  PendingReviewsResponse,
  ResolveReviewResponse,
  ReviewSummaryResponse,
} from '@/types/expert-review';

// Mock the API functions
vi.mock('@/api/expert-review', () => ({
  getPendingReviews: vi.fn(),
  resolveReview: vi.fn(),
  getReviewSummary: vi.fn(),
}));

// Mock query-client
vi.mock('@/lib/query-client', () => ({
  queryKeys: {
    all: ['e2i'] as const,
    expertReviews: {
      all: () => ['e2i', 'expert-reviews'] as const,
      pending: (params?: { brand?: string; reviewer_id?: string; limit?: number }) =>
        [
          'e2i',
          'expert-reviews',
          'pending',
          params?.brand ?? null,
          params?.reviewer_id ?? null,
          params?.limit ?? 50,
        ] as const,
      summary: (params?: { brand?: string }) =>
        ['e2i', 'expert-reviews', 'summary', params?.brand ?? null] as const,
    },
  },
}));

import {
  usePendingReviews,
  useResolveReview,
  useReviewSummary,
} from './use-expert-review';
import * as expertReviewApi from '@/api/expert-review';

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
      mutations: { retry: false },
    },
  });
}

function createWrapper() {
  const queryClient = createTestQueryClient();
  return {
    queryClient,
    wrapper: ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children),
  };
}

const mockPendingResponse: PendingReviewsResponse = {
  reviews: [
    {
      review_id: '11111111-1111-1111-1111-111111111111',
      review_type: 'dag_approval',
      dag_version_hash: 'abc123',
      brand: 'Remibrutinib',
      treatment_variable: 'email_frequency',
      outcome_variable: 'trx',
      analysis_context: 'confidence=0.60, gate=review',
      created_at: '2026-06-01T00:00:00Z',
      days_pending: 5,
    },
  ],
  total: 1,
};

const mockSummaryResponse: ReviewSummaryResponse = {
  pending: 1,
  approved: 4,
  rejected: 0,
  expired: 0,
  expiring_soon: 0,
};

const mockResolveResponse: ResolveReviewResponse = {
  review_id: '11111111-1111-1111-1111-111111111111',
  approval_status: 'approved',
  success: true,
};

describe('usePendingReviews', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches the pending queue successfully', async () => {
    vi.mocked(expertReviewApi.getPendingReviews).mockResolvedValueOnce(mockPendingResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePendingReviews(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockPendingResponse);
    expect(expertReviewApi.getPendingReviews).toHaveBeenCalledWith(undefined);
  });

  it('passes params to the API', async () => {
    vi.mocked(expertReviewApi.getPendingReviews).mockResolvedValueOnce(mockPendingResponse);
    const { wrapper } = createWrapper();
    const params = { brand: 'Remibrutinib', limit: 10 };

    const { result } = renderHook(() => usePendingReviews(params), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(expertReviewApi.getPendingReviews).toHaveBeenCalledWith(params);
  });

  it('handles an empty queue', async () => {
    vi.mocked(expertReviewApi.getPendingReviews).mockResolvedValueOnce({ reviews: [], total: 0 });
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePendingReviews(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.total).toBe(0);
  });

  it('handles an error state', async () => {
    vi.mocked(expertReviewApi.getPendingReviews).mockRejectedValueOnce(new Error('boom'));
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePendingReviews(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});

describe('useReviewSummary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('fetches the summary successfully', async () => {
    vi.mocked(expertReviewApi.getReviewSummary).mockResolvedValueOnce(mockSummaryResponse);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useReviewSummary(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockSummaryResponse);
    expect(expertReviewApi.getReviewSummary).toHaveBeenCalledWith(undefined);
  });
});

describe('useResolveReview', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('posts a resolution and invalidates pending + summary queries', async () => {
    vi.mocked(expertReviewApi.resolveReview).mockResolvedValueOnce(mockResolveResponse);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useResolveReview(), { wrapper });

    result.current.mutate({
      reviewId: '11111111-1111-1111-1111-111111111111',
      body: { approval_status: 'approved', checklist: { conf_complete: true } },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual(mockResolveResponse);
    expect(expertReviewApi.resolveReview).toHaveBeenCalledWith(
      '11111111-1111-1111-1111-111111111111',
      { approval_status: 'approved', checklist: { conf_complete: true } }
    );
    // invalidate both the pending queue and the summary
    expect(invalidateSpy).toHaveBeenCalledTimes(2);
  });

  it('handles a resolve error', async () => {
    vi.mocked(expertReviewApi.resolveReview).mockRejectedValueOnce(new Error('nope'));
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useResolveReview(), { wrapper });

    result.current.mutate({
      reviewId: 'bad',
      body: { approval_status: 'rejected', checklist: {} },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
  });

  it('calls onSuccess callback', async () => {
    vi.mocked(expertReviewApi.resolveReview).mockResolvedValueOnce(mockResolveResponse);
    const { wrapper } = createWrapper();
    const onSuccess = vi.fn();

    const { result } = renderHook(() => useResolveReview({ onSuccess }), { wrapper });

    result.current.mutate({
      reviewId: '11111111-1111-1111-1111-111111111111',
      body: { approval_status: 'approved', checklist: {} },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(onSuccess).toHaveBeenCalled();
  });
});
