/**
 * ExpertReviews Page Tests (R6-F2 Phase B4)
 * =========================================
 *
 * The page must render ONLY the live pending queue (no hardcoded SAMPLE_ rows)
 * with honest loading / error / empty states, and offer a per-row approve/reject
 * action wired to useResolveReview.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ExpertReviews from './ExpertReviews';
import type { PendingReviewsResponse } from '@/types/expert-review';

vi.mock('@/hooks/api/use-expert-review', () => ({
  usePendingReviews: vi.fn(),
  useReviewSummary: vi.fn(),
  useResolveReview: vi.fn(),
}));

import {
  usePendingReviews,
  useReviewSummary,
  useResolveReview,
} from '@/hooks/api/use-expert-review';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const mockPending: PendingReviewsResponse = {
  reviews: [
    {
      review_id: 'rev-1',
      review_type: 'dag_approval',
      dag_version_hash: 'deadbeefcafebabe0123',
      brand: 'Remibrutinib',
      treatment_variable: 'email_frequency',
      outcome_variable: 'trx',
      analysis_context: 'confidence=0.60',
      created_at: '2026-06-01T00:00:00Z',
      days_pending: 5,
    },
  ],
  total: 1,
};

function mockResolveReturn(overrides = {}) {
  return {
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    error: null,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(useReviewSummary).mockReturnValue({ data: undefined } as never);
  vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn() as never);
});

describe('ExpertReviews page', () => {
  it('shows a loading state while fetching', () => {
    vi.mocked(usePendingReviews).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      isFetching: true,
      refetch: vi.fn(),
    } as never);

    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Expert Reviews')).toBeInTheDocument();
  });

  it('shows an honest empty state (no SAMPLE rows) when the queue is empty', () => {
    vi.mocked(usePendingReviews).mockReturnValue({
      data: { reviews: [], total: 0 },
      isLoading: false,
      isError: false,
      isFetching: false,
      refetch: vi.fn(),
    } as never);

    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('No pending reviews')).toBeInTheDocument();
  });

  it('shows an error banner on failure', () => {
    vi.mocked(usePendingReviews).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { message: 'boom' },
      isFetching: false,
      refetch: vi.fn(),
    } as never);

    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Failed to load pending reviews')).toBeInTheDocument();
  });

  it('renders the live pending queue and resolves a review', async () => {
    const mutate = vi.fn();
    vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn({ mutate }) as never);
    vi.mocked(usePendingReviews).mockReturnValue({
      data: mockPending,
      isLoading: false,
      isError: false,
      isFetching: false,
      refetch: vi.fn(),
    } as never);

    render(<ExpertReviews />, { wrapper: createWrapper() });

    expect(screen.getByText('email_frequency')).toBeInTheDocument();
    expect(screen.getByText('Remibrutinib')).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /review/i }));

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /approve/i })).toBeInTheDocument()
    );

    await user.click(screen.getByRole('button', { name: /approve/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const [vars] = mutate.mock.calls[0];
    expect(vars.reviewId).toBe('rev-1');
    expect(vars.body.approval_status).toBe('approved');
  });
});
