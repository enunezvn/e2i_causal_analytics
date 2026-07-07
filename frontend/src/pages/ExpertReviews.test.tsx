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
  useReviewAssessment: vi.fn(),
}));

// The DAG renderer is D3-heavy; the page test only asserts it is MOUNTED with
// the right graph (its own rendering is covered by causal.test.tsx).
vi.mock('@/components/visualizations/causal/CausalDAG', () => {
  const FakeDag = ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  );
  return { CausalDAG: FakeDag, default: FakeDag };
});

import {
  usePendingReviews,
  useReviewSummary,
  useResolveReview,
  useReviewAssessment,
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

function mockAssessmentReturn(overrides = {}) {
  return {
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    error: null,
    data: undefined,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(useReviewSummary).mockReturnValue({ data: undefined } as never);
  vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn() as never);
  vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn() as never);
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

const STRUCTURE = {
  nodes: ['t', 'y', 'c'],
  edges: [
    ['t', 'y'],
    ['c', 't'],
    ['c', 'y'],
  ],
  treatment_nodes: ['t'],
  outcome_nodes: ['y'],
};

const ASSESSMENT = {
  items: [
    {
      id: 'conf_complete',
      question: 'Are all known confounders included?',
      verdict: 'supports',
      rationale: 'confounder refuters passed',
    },
    {
      id: 'positivity',
      question: 'Is there sufficient overlap in treatment groups?',
      verdict: 'concern',
      rationale: 'data_subset failed',
    },
  ],
  is_fallback: true,
  evidence: { refutation_tests: 2, has_dag_structure: true },
};

function renderWithRow(row: Record<string, unknown>) {
  vi.mocked(usePendingReviews).mockReturnValue({
    data: { reviews: [{ ...mockPending.reviews[0], ...row }], total: 1 },
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
  } as never);
  return render(<ExpertReviews />, { wrapper: createWrapper() });
}

describe('ExpertReviews DAG snapshot (mig 097)', () => {
  it('renders the stored DAG in the expanded row', async () => {
    renderWithRow({ dag_structure_json: STRUCTURE });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /review/i }));

    const dag = await screen.findByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-nodes', '3');
    expect(dag).toHaveAttribute('data-edges', '3');
  });

  it('shows an honest fallback when the structure was never captured', async () => {
    renderWithRow({ dag_structure_json: null });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /review/i }));

    expect(
      await screen.findByText(/DAG structure not captured for this review/i)
    ).toBeInTheDocument();
    expect(screen.queryByTestId('causal-dag')).not.toBeInTheDocument();
  });
});

describe('ExpertReviews agent assessment (advisory)', () => {
  it('offers a generate button when no assessment is cached', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(
      mockAssessmentReturn({ mutate }) as never
    );
    renderWithRow({ dag_structure_json: STRUCTURE });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /review/i }));

    const generate = await screen.findByRole('button', {
      name: /agent assessment/i,
    });
    await user.click(generate);

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(mutate.mock.calls[0][0].reviewId).toBe('rev-1');
  });

  it('renders cached verdict chips beside the checklist, labeled advisory', async () => {
    renderWithRow({
      dag_structure_json: STRUCTURE,
      agent_assessment_json: ASSESSMENT,
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /review/i }));

    expect(await screen.findByText('supports')).toBeInTheDocument();
    expect(screen.getByText('concern')).toBeInTheDocument();
    expect(screen.getByText(/confounder refuters passed/i)).toBeInTheDocument();
    // Advisory, never a substitute for the human's own answers.
    expect(screen.getAllByText(/advisory/i).length).toBeGreaterThan(0);
    // Chips must NOT pre-check the human checklist.
    const checkboxes = screen.getAllByRole('checkbox');
    checkboxes.forEach((cb) => expect(cb).not.toBeChecked());
  });

  it('renders assessment returned by the mutation', async () => {
    vi.mocked(useReviewAssessment).mockReturnValue(
      mockAssessmentReturn({
        data: {
          review_id: 'rev-1',
          assessment: ASSESSMENT,
          cached: false,
          persisted: true,
        },
      }) as never
    );
    renderWithRow({ dag_structure_json: STRUCTURE });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /review/i }));

    expect(await screen.findByText('supports')).toBeInTheDocument();
    expect(screen.getByText(/data_subset failed/i)).toBeInTheDocument();
  });
});
