/**
 * LinkedReviewCard tests — the deep-linked review's DAG panel.
 *
 * The card already holds the detail response's version timeline, so its panel
 * must show the same structure diff the expanded queue row does; a reviewer
 * arriving from the causal drill-down should not see LESS than one who opened
 * the row from the queue.
 */
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { LinkedReviewCard } from './LinkedReviewCard';
import type { ExpertReviewDetailResponse } from '@/types/expert-review';

vi.mock('@/hooks/api/use-expert-review', () => ({
  useExpertReview: vi.fn(),
  useReviewAssessment: vi.fn(() => ({
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    error: null,
    data: undefined,
  })),
  useResolveReview: vi.fn(() => ({ mutate: vi.fn(), isPending: false, isError: false, error: null })),
}));

// The DAG renderer is D3-heavy; this test only asserts it is MOUNTED.
vi.mock('@/components/visualizations/causal/CausalDAG', () => {
  const FakeDag = ({ nodes, edges }: { nodes: unknown[]; edges: unknown[] }) => (
    <div data-testid="causal-dag" data-nodes={nodes.length} data-edges={edges.length} />
  );
  return { CausalDAG: FakeDag, default: FakeDag };
});

import { useExpertReview } from '@/hooks/api/use-expert-review';

const DETAIL: ExpertReviewDetailResponse = {
  review: {
    review_id: 'rev-linked',
    approval_status: 'rejected',
    brand: 'Kisqali',
    treatment_variable: 'treatment_arm',
    outcome_variable: 'persistent_180d',
    created_at: '2026-09-01T00:00:00Z',
    dag_structure_json: {
      nodes: ['T', 'Y', 'W'],
      edges: [
        ['T', 'Y'],
        ['W', 'T'],
      ],
      treatment_nodes: ['T'],
      outcome_nodes: ['Y'],
    },
  },
  history: [],
  current_version_id: 'v2',
  versions: [
    { version_id: 'v1', dag_version_hash: 'aaaa1111', changes: null },
    {
      version_id: 'v2',
      dag_version_hash: 'bbbb2222',
      changes: {
        nodes_added: ['W'],
        nodes_removed: [],
        edges_added: [['W', 'T']],
        edges_removed: [],
        adjustment_sets_added: [],
        adjustment_sets_removed: [],
        is_changed: true,
      },
    },
  ],
};

function renderCard() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <LinkedReviewCard reviewId="rev-linked" />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('LinkedReviewCard DAG panel', () => {
  it('passes the detail response versions to the panel, so the structure diff renders', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: DETAIL,
      isLoading: false,
      isError: false,
    } as never);
    renderCard();
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.getByText('Changed since the previous version')).toBeInTheDocument();
    expect(screen.getByText('+ W')).toBeInTheDocument();
    expect(screen.getByText('+ W → T')).toBeInTheDocument();
  });

  it('renders the delta of the version the DETAIL names, not the last entry', () => {
    // The card's panel must honour `current_version_id`: with the review on v1
    // (the FIRST version, nothing to diff against) the v2 delta is a timeline
    // fact belonging to a version the review is not on.
    vi.mocked(useExpertReview).mockReturnValue({
      data: { ...DETAIL, current_version_id: 'v1' },
      isLoading: false,
      isError: false,
    } as never);
    renderCard();
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
    expect(screen.queryByText('+ W')).toBeNull();
  });

  it('renders no diff for a review with a single version', () => {
    // The review is ON that one version (`v1`), so the silence is the
    // FIRST-version case — nothing to diff against — and not the unrelated
    // "the named id is missing from the timeline" path.
    vi.mocked(useExpertReview).mockReturnValue({
      data: { ...DETAIL, current_version_id: 'v1', versions: [DETAIL.versions[0]] },
      isLoading: false,
      isError: false,
    } as never);
    renderCard();
    expect(screen.getByTestId('causal-dag')).toBeInTheDocument();
    expect(screen.queryByText('Changed since the previous version')).toBeNull();
  });
});
