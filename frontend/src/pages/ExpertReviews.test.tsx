/**
 * ExpertReviews Page Tests (R6-F2 Phase B4 + lane 1)
 * ===================================================
 *
 * The page renders ONLY the live pending queue (no hardcoded SAMPLE_ rows) with
 * honest loading / error / empty states, resolves a review, follows the global
 * brand filter, shows a summary error instead of dropping the counts, opens a
 * linked review from `?review=`, auto-generates a missing assessment when a row
 * is expanded, and prefetches assessments one row at a time.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import ExpertReviews from './ExpertReviews';
import type {
  AgentAssessment,
  ExpertReviewDetailResponse,
  PendingReviewsResponse,
} from '@/types/expert-review';

vi.mock('@/hooks/api/use-expert-review', () => ({
  usePendingReviews: vi.fn(),
  useReviewSummary: vi.fn(),
  useResolveReview: vi.fn(),
  useReviewAssessment: vi.fn(),
  useExpertReview: vi.fn(),
}));

const mockSetBrand = vi.fn();
const filtersState = { brand: 'All' as string };
vi.mock('@/hooks/use-e2i-filters', () => ({
  useE2IFilters: () => ({ filters: { brand: filtersState.brand }, setBrand: mockSetBrand }),
}));

vi.mock('@/api/expert-review', () => ({
  generateReviewAssessment: vi.fn(),
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
  useExpertReview,
  usePendingReviews,
  useReviewAssessment,
  useReviewSummary,
  useResolveReview,
} from '@/hooks/api/use-expert-review';
import { generateReviewAssessment } from '@/api/expert-review';

function createWrapper(initialPath = '/expert-reviews') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialPath]}>{children}</MemoryRouter>
    </QueryClientProvider>
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
  return { mutate: vi.fn(), isPending: false, isError: false, error: null, ...overrides };
}

function mockAssessmentReturn(overrides = {}) {
  return { mutate: vi.fn(), isPending: false, isError: false, error: null, data: undefined, ...overrides };
}

function mockQueue(response: PendingReviewsResponse | undefined, extra = {}) {
  vi.mocked(usePendingReviews).mockReturnValue({
    data: response,
    isLoading: false,
    isError: false,
    isFetching: false,
    refetch: vi.fn(),
    ...extra,
  } as never);
}

beforeEach(() => {
  vi.clearAllMocks();
  filtersState.brand = 'All';
  vi.mocked(useReviewSummary).mockReturnValue({ data: undefined, isError: false } as never);
  vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn() as never);
  vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn() as never);
  vi.mocked(useExpertReview).mockReturnValue({ data: undefined, isLoading: false, isError: false } as never);
});

describe('ExpertReviews page', () => {
  it('shows a loading state while fetching', () => {
    mockQueue(undefined, { isLoading: true, isFetching: true });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Expert Reviews')).toBeInTheDocument();
  });

  it('shows an honest empty state (no SAMPLE rows) when the queue is empty', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('No pending reviews')).toBeInTheDocument();
  });

  it('shows an error banner on failure', () => {
    mockQueue(undefined, { isError: true, error: { message: 'boom' } });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Failed to load pending reviews')).toBeInTheDocument();
  });

  it('renders the live pending queue and resolves a review', async () => {
    const mutate = vi.fn();
    vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn({ mutate }) as never);
    mockQueue(mockPending);

    render(<ExpertReviews />, { wrapper: createWrapper() });

    expect(screen.getByText('email_frequency')).toBeInTheDocument();
    expect(screen.getByText('Remibrutinib')).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^review$/i }));
    await waitFor(() => expect(screen.getByRole('button', { name: /approve/i })).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: /approve/i }));

    expect(mutate).toHaveBeenCalledTimes(1);
    const [vars] = mutate.mock.calls[0];
    expect(vars.reviewId).toBe('rev-1');
    expect(vars.body.approval_status).toBe('approved');
  });
});

describe('ExpertReviews brand filter and summary (lane 1)', () => {
  it('passes the global brand to BOTH the queue and the summary; All sends no brand', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(vi.mocked(usePendingReviews)).toHaveBeenLastCalledWith(undefined);
    expect(vi.mocked(useReviewSummary)).toHaveBeenLastCalledWith(undefined);
    expect(screen.getByText(/including reviews with no brand/i)).toBeInTheDocument();
  });

  it('scopes to the selected brand', () => {
    filtersState.brand = 'Kisqali';
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(vi.mocked(usePendingReviews)).toHaveBeenLastCalledWith({ brand: 'Kisqali' });
    expect(vi.mocked(useReviewSummary)).toHaveBeenLastCalledWith({ brand: 'Kisqali' });
    expect(screen.getByText(/brand: Kisqali/)).toBeInTheDocument();
  });

  it('shows a banner instead of silently dropping the counts when the summary fails', () => {
    vi.mocked(useReviewSummary).mockReturnValue({
      data: undefined,
      isError: true,
      error: { message: 'Expert-review store unavailable. Retry shortly.' },
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.getByText('Review counts unavailable')).toBeInTheDocument();
    expect(screen.queryByText(/Pending:/)).not.toBeInTheDocument();
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

// Typed so the literal verdicts stay `AssessmentVerdict` when the fixture is
// handed to the typed queue mock (untyped, they widen to `string`).
const ASSESSMENT: AgentAssessment = {
  items: [
    { id: 'conf_complete', question: 'Are all known confounders included?', verdict: 'supports', rationale: 'confounder refuters passed' },
    { id: 'positivity', question: 'Is there sufficient overlap in treatment groups?', verdict: 'concern', rationale: 'data_subset failed' },
  ],
  is_fallback: true,
  evidence: { refutation_tests: 2, has_dag_structure: true },
};

function renderWithRow(row: Record<string, unknown>, path?: string) {
  mockQueue({ reviews: [{ ...mockPending.reviews[0], ...row }], total: 1 });
  return render(<ExpertReviews />, { wrapper: createWrapper(path) });
}

describe('ExpertReviews DAG snapshot (mig 097)', () => {
  it('renders the stored DAG in the expanded row', async () => {
    renderWithRow({ dag_structure_json: STRUCTURE });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    const dag = await screen.findByTestId('causal-dag');
    expect(dag).toHaveAttribute('data-nodes', '3');
    expect(dag).toHaveAttribute('data-edges', '3');
  });

  it('shows an honest fallback when the structure was never captured', async () => {
    renderWithRow({ dag_structure_json: null });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect(await screen.findByText(/DAG structure not captured for this review/i)).toBeInTheDocument();
    expect(screen.queryByTestId('causal-dag')).not.toBeInTheDocument();
  });
});

describe('ExpertReviews agent assessment (advisory)', () => {
  it('generates the assessment once when a row without a cache is expanded, then regenerates on click', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    renderWithRow({ dag_structure_json: STRUCTURE });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^review$/i }));

    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    expect(mutate.mock.calls[0][0]).toEqual({ reviewId: 'rev-1' });

    await user.click(await screen.findByRole('button', { name: /agent assessment/i }));
    expect(mutate).toHaveBeenCalledTimes(2);
  });

  it('does not auto-generate when a cached assessment exists', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    renderWithRow({ dag_structure_json: STRUCTURE, agent_assessment_json: ASSESSMENT });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect(await screen.findByText('supports')).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it('generates ONCE when the linked card and the queue row show the same pending review', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    // A ReviewRecord always carries approval_status (GET /expert-reviews/{id});
    // the card only mounts a form for a PENDING record.
    vi.mocked(useExpertReview).mockReturnValue({
      data: {
        review: { ...mockPending.reviews[0], approval_status: 'pending', dag_structure_json: STRUCTURE },
        history: [],
      },
      isLoading: false,
      isError: false,
    } as never);
    mockQueue({ reviews: [{ ...mockPending.reviews[0], dag_structure_json: STRUCTURE }], total: 1 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-1') });
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect((await screen.findAllByRole('button', { name: /approve/i })).length).toBe(2);
    expect(mutate).toHaveBeenCalledTimes(1);
    // Two forms for one review must not share element ids (labels would target the other form).
    const ids = Array.from(document.querySelectorAll('[id]')).map((el) => el.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('renders cached verdict chips beside the checklist, labeled advisory, never pre-checked', async () => {
    renderWithRow({ dag_structure_json: STRUCTURE, agent_assessment_json: ASSESSMENT });
    await userEvent.setup().click(screen.getByRole('button', { name: /^review$/i }));
    expect(await screen.findByText('supports')).toBeInTheDocument();
    expect(screen.getByText('concern')).toBeInTheDocument();
    expect(screen.getAllByText(/advisory/i).length).toBeGreaterThan(0);
    screen.getAllByRole('checkbox').forEach((cb) => expect(cb).not.toBeChecked());
  });

  it('prepares the missing assessments one row at a time', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    vi.mocked(generateReviewAssessment).mockResolvedValue({
      review_id: 'x', assessment: ASSESSMENT, cached: false, persisted: true,
    } as never);
    mockQueue({
      reviews: [
        { ...mockPending.reviews[0], review_id: 'rev-1' },
        { ...mockPending.reviews[0], review_id: 'rev-2', agent_assessment_json: ASSESSMENT },
        { ...mockPending.reviews[0], review_id: 'rev-3' },
      ],
      total: 3,
    });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    const button = screen.getByRole('button', { name: /prepare assessments/i });
    expect(button).toHaveTextContent('2 missing');
    await userEvent.setup().click(button);
    await waitFor(() => expect(generateReviewAssessment).toHaveBeenCalledTimes(2));
    expect(vi.mocked(generateReviewAssessment).mock.calls.map((c) => c[0])).toEqual(['rev-1', 'rev-3']);
    // The bulk run marked rev-1 in the shared guard: expanding it must not start a second generation.
    await userEvent.setup().click(screen.getAllByRole('button', { name: /^review$/i })[0]);
    await screen.findByRole('button', { name: /approve/i });
    expect(mutate).not.toHaveBeenCalled();
  });

  it('skips a review whose generation an expanded form already started', async () => {
    const mutate = vi.fn();
    vi.mocked(useReviewAssessment).mockReturnValue(mockAssessmentReturn({ mutate }) as never);
    vi.mocked(generateReviewAssessment).mockResolvedValue({
      review_id: 'x', assessment: ASSESSMENT, cached: false, persisted: true,
    } as never);
    mockQueue({
      reviews: [
        { ...mockPending.reviews[0], review_id: 'rev-1' },
        { ...mockPending.reviews[0], review_id: 'rev-3' },
      ],
      total: 2,
    });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    const user = userEvent.setup();
    await user.click(screen.getAllByRole('button', { name: /^review$/i })[0]); // rev-1's form fires once
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: /prepare assessments/i }));
    await waitFor(() => expect(generateReviewAssessment).toHaveBeenCalledTimes(1));
    expect(vi.mocked(generateReviewAssessment).mock.calls[0][0]).toBe('rev-3');
  });

  it('stops the prefetch on the first error and says how far it got', async () => {
    vi.mocked(generateReviewAssessment)
      .mockResolvedValueOnce({ review_id: 'rev-1', assessment: ASSESSMENT, cached: false, persisted: true } as never)
      .mockRejectedValueOnce(new Error('LM unavailable'));
    mockQueue({
      reviews: [
        { ...mockPending.reviews[0], review_id: 'rev-1' },
        { ...mockPending.reviews[0], review_id: 'rev-2' },
        { ...mockPending.reviews[0], review_id: 'rev-3' },
      ],
      total: 3,
    });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    await userEvent.setup().click(screen.getByRole('button', { name: /prepare assessments/i }));
    expect(await screen.findByText('Stopped after 1 of 3')).toBeInTheDocument();
    expect(screen.getByText('LM unavailable')).toBeInTheDocument();
    expect(generateReviewAssessment).toHaveBeenCalledTimes(2);
  });

  it('lets a later Prepare retry a review whose earlier attempt failed', async () => {
    vi.mocked(generateReviewAssessment)
      .mockRejectedValueOnce(new Error('LM unavailable'))
      .mockResolvedValue({ review_id: 'rev-1', assessment: ASSESSMENT, cached: false, persisted: true } as never);
    mockQueue({ reviews: [{ ...mockPending.reviews[0], review_id: 'rev-1' }], total: 1 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /prepare assessments/i }));
    expect(await screen.findByText('LM unavailable')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /prepare assessments/i }));
    await waitFor(() => expect(generateReviewAssessment).toHaveBeenCalledTimes(2));
  });
});

const DETAIL: ExpertReviewDetailResponse = {
  review: {
    review_id: 'rev-rejected',
    review_type: 'dag_approval',
    dag_version_hash: 'deadbeefcafebabe0123',
    brand: null,
    treatment_variable: 'treatment_arm',
    outcome_variable: 'persistent_180d',
    approval_status: 'rejected',
    reviewer_name: 'Dr. No',
    concerns_raised: ['collider'],
    created_at: '2026-07-13T10:00:00Z',
    dag_structure_json: STRUCTURE,
  },
  history: [
    { review_id: 'rev-rejected', approval_status: 'rejected', created_at: '2026-07-13T10:00:00Z', reviewer_name: 'Dr. No' },
    { review_id: 'rev-older', approval_status: 'pending', created_at: '2026-07-01T10:00:00Z' },
  ],
};

describe('ExpertReviews linked review (lane 1)', () => {
  it('renders nothing extra without the review param', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('linked-review')).not.toBeInTheDocument();
    // The hook lives in LinkedReviewCard, which is not mounted without the param
    // (pre-execution review 2026-09-08, codex MED: the old assertion could not pass).
    expect(vi.mocked(useExpertReview)).not.toHaveBeenCalled();
  });

  it('shows a resolved linked review with its decision and same-structure history', () => {
    vi.mocked(useExpertReview).mockReturnValue({ data: DETAIL, isLoading: false, isError: false } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-rejected') });
    expect(vi.mocked(useExpertReview)).toHaveBeenLastCalledWith('rev-rejected');
    const card = screen.getByTestId('linked-review');
    expect(card).toHaveTextContent('rejected');
    expect(card).toHaveTextContent('Dr. No');
    expect(card).toHaveTextContent('collider');
    expect(card).toHaveTextContent('This review is resolved');
    expect(screen.getAllByTestId('causal-dag').length).toBe(1);
    expect(card).toHaveTextContent('rev-older');
    // The backend history INCLUDES the linked review itself: it is marked exactly
    // once, on its own row, and never on the sibling rows (dispatcher deviation 1).
    expect(screen.getAllByText('(this review)')).toHaveLength(1);
    const current = card.querySelectorAll('[data-current="true"]');
    expect(current).toHaveLength(1);
    expect(current[0]).toHaveTextContent('rev-rejected');
    expect(current[0]).toHaveTextContent('(this review)');
    expect(current[0]).not.toHaveTextContent('rev-older');
  });

  it('resolves a pending linked review in place', async () => {
    const mutate = vi.fn();
    vi.mocked(useResolveReview).mockReturnValue(mockResolveReturn({ mutate }) as never);
    vi.mocked(useExpertReview).mockReturnValue({
      data: { ...DETAIL, review: { ...DETAIL.review, review_id: 'rev-p', approval_status: 'pending' }, history: [] },
      isLoading: false,
      isError: false,
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-p') });
    await userEvent.setup().click(screen.getByRole('button', { name: /reject/i }));
    expect(mutate.mock.calls[0][0].reviewId).toBe('rev-p');
    expect(mutate.mock.calls[0][0].body.approval_status).toBe('rejected');
  });

  it('says so when the linked review no longer exists (404), and keeps the queue usable', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { status: 404, message: 'Review nope was not found.' },
    } as never);
    mockQueue(mockPending);
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=nope') });
    expect(screen.getByText('This review no longer exists')).toBeInTheDocument();
    expect(screen.getByText('Review nope was not found.')).toBeInTheDocument();
    expect(screen.queryByText('Failed to load the linked review')).not.toBeInTheDocument();
    expect(screen.getByText('email_frequency')).toBeInTheDocument();
  });

  it('shows the generic failure title with the backend message when the store is unavailable (503)', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { status: 503, message: 'Expert-review store unavailable. Retry shortly.' },
    } as never);
    mockQueue(mockPending);
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=rev-1') });
    expect(screen.getByText('Failed to load the linked review')).toBeInTheDocument();
    expect(screen.getByText('Expert-review store unavailable. Retry shortly.')).toBeInTheDocument();
    expect(screen.queryByText('This review no longer exists')).not.toBeInTheDocument();
    expect(screen.getByText('email_frequency')).toBeInTheDocument();
  });

  it('falls back to the message text only when the error carries no status', () => {
    vi.mocked(useExpertReview).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: { message: 'Review nope was not found.' },
    } as never);
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=nope') });
    expect(screen.getByText('This review no longer exists')).toBeInTheDocument();
  });

  it('ignores a blank review param', () => {
    mockQueue({ reviews: [], total: 0 });
    render(<ExpertReviews />, { wrapper: createWrapper('/expert-reviews?review=%20%20') });
    expect(screen.queryByTestId('linked-review')).not.toBeInTheDocument();
    expect(vi.mocked(useExpertReview)).not.toHaveBeenCalled();
  });
});
