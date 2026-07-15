/**
 * FeedbackLearning Page — Warning Banner Coverage
 * ===============================================
 *
 * Regression tests for F-010-frontend: API-reported `warnings[]` from
 * the learning-cycle mutation are surfaced to the user as a banner.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import FeedbackLearning from './FeedbackLearning';

vi.mock('recharts', async () => {
  const actual = await vi.importActual('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive-container">{children}</div>
    ),
  };
});

vi.mock('@/hooks/api', () => ({
  useFeedbackHealth: vi.fn(),
  usePatterns: vi.fn(),
  useKnowledgeUpdates: vi.fn(),
  useQuickLearningCycle: vi.fn(),
  useApplyUpdate: vi.fn(),
  useRollbackUpdate: vi.fn(),
  useFeedbackLearningInsight: vi.fn(),
}));

import {
  useFeedbackHealth,
  usePatterns,
  useKnowledgeUpdates,
  useQuickLearningCycle,
  useApplyUpdate,
  useRollbackUpdate,
  useFeedbackLearningInsight,
} from '@/hooks/api';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('FeedbackLearning — F-002 empty state', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, cycles_24h: 0 },
      refetch: vi.fn().mockResolvedValue({}),
    });
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { updates: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    });
    (useApplyUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useRollbackUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false, data: undefined, error: null });
  });

  it('does not render fabricated SAMPLE_PATTERNS / SAMPLE_UPDATES when API empty', () => {
    render(<FeedbackLearning />, { wrapper: createWrapper() });

    // Former SAMPLE_PATTERNS descriptions must not appear in DOM.
    expect(
      screen.queryByText(/Causal Impact agent showing increased latency during peak hours/),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Refine causal impact explanation template/),
    ).not.toBeInTheDocument();
  });

  it('explains WHY the patterns/updates tabs can be empty (window-bounded cycles)', async () => {
    render(<FeedbackLearning />, { wrapper: createWrapper() });

    await userEvent.click(screen.getByRole('tab', { name: /^Patterns$/i }));
    expect(screen.getByText(/bounded lookback window/i)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('tab', { name: /Knowledge Updates/i }));
    expect(screen.getByText(/wait here for manual review/i)).toBeInTheDocument();
  });

});

describe('FeedbackLearning — warnings rendering (F-010-frontend)', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, cycles_24h: 4 },
      refetch: vi.fn().mockResolvedValue({}),
    });

    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { updates: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useApplyUpdate as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    });

    (useRollbackUpdate as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    });

    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      data: undefined,
      error: null,
    });
  });

  it('does not render WarningBanner before the cycle has run', () => {
    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });
    expect(screen.queryByTestId('warning-banner')).not.toBeInTheDocument();
  });

  it('renders WarningBanner with onSuccess warnings after running cycle', async () => {
    // Capture the mutation's onSuccess callback so we can simulate API
    // returning warnings without spinning up a real fetch.
    let capturedOnSuccess: ((data: unknown) => void) | undefined;
    const mutateMock = vi.fn((_args: unknown, opts: { onSuccess?: (d: unknown) => void } = {}) => {
      capturedOnSuccess = opts.onSuccess;
    });

    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mutateMock,
      isPending: false,
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('button', { name: /Run Learning Cycle/i }));

    // Now invoke the captured callback as the mutation hook would.
    expect(capturedOnSuccess).toBeDefined();
    act(() => {
      capturedOnSuccess?.({
        warnings: ['Cycle ran in degraded mode; pattern detector skipped'],
        batch_id: 'b1',
      });
    });

    await waitFor(() => {
      expect(screen.getByTestId('warning-banner')).toBeInTheDocument();
    });
    expect(
      screen.getByText('Cycle ran in degraded mode; pattern detector skipped'),
    ).toBeInTheDocument();
    expect(screen.getByText('Learning cycle warnings')).toBeInTheDocument();
  });
});

describe('FeedbackLearning — no fabricated health defaults', () => {
  // Regression: the summary cards used `cycles_24h ?? 12` and
  // `agent_available ?? true`, so a loading/failed health query rendered a
  // FAKE "12" cycles + "Online". The fix shows an honest "Checking…" / "—"
  // while loading and conservative real defaults (Offline / 0) otherwise —
  // never a fabricated plausible value. These would FAIL before the fix.
  beforeEach(() => {
    vi.clearAllMocks();
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [] }, isLoading: false, refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { updates: [] }, isLoading: false, refetch: vi.fn().mockResolvedValue({}),
    });
    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useApplyUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useRollbackUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false, data: undefined, error: null });
  });

  it('shows "Checking…" / "—" while health is loading (not fabricated 12 / Online)', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined, isLoading: true, refetch: vi.fn().mockResolvedValue({}),
    });
    render(<FeedbackLearning />, { wrapper: createWrapper() });
    expect(screen.getByText(/Checking/)).toBeInTheDocument();
    expect(screen.queryByText('12')).not.toBeInTheDocument();
    expect(screen.queryByText('Online')).not.toBeInTheDocument();
  });

  it('shows conservative Offline / 0 when health is unavailable (not fabricated 12 / Online)', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined, isLoading: false, refetch: vi.fn().mockResolvedValue({}),
    });
    render(<FeedbackLearning />, { wrapper: createWrapper() });
    expect(screen.queryByText('12')).not.toBeInTheDocument();
    expect(screen.queryByText('Online')).not.toBeInTheDocument();
    expect(screen.getByText('Offline')).toBeInTheDocument();
  });

  it('reflects the real health values when present', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, cycles_24h: 7 }, isLoading: false, refetch: vi.fn().mockResolvedValue({}),
    });
    render(<FeedbackLearning />, { wrapper: createWrapper() });
    expect(screen.getByText('Online')).toBeInTheDocument();
    expect(screen.getByText('7')).toBeInTheDocument();
    expect(screen.queryByText(/Checking/)).not.toBeInTheDocument();
  });
});

describe('FeedbackLearning — StrategicInsightCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, cycles_24h: 4 },
      refetch: vi.fn().mockResolvedValue({}),
    });
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [] }, isLoading: false, refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { updates: [] }, isLoading: false, refetch: vi.fn().mockResolvedValue({}),
    });
    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useApplyUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useRollbackUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
  });

  it('renders the Strategic Interpretation card with a generate action', () => {
    const mutateMock = vi.fn();
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: mutateMock, isPending: false, data: undefined, error: null,
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    expect(screen.getByText(/Strategic Interpretation/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Generate strategic insight/i }));
    // Grounding is server-derived; the page only picks the 7-day inflow window.
    expect(mutateMock).toHaveBeenCalledWith({ days: 7 });
  });

  it('renders the returned insight with grounding chips and fallback badge state', () => {
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
      data: {
        insight: 'The loop is actively learning from 39 feedback items.',
        key_takeaways: ['Review the pending update'],
        grounding: [{ label: 'Cycles 24h', value: '4' }],
        is_fallback: false,
        provenance: 'Live feedback-learning loop data (server-derived)',
        generated_at: '2026-07-07T00:00:00Z',
      },
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    expect(
      screen.getByText('The loop is actively learning from 39 feedback items.'),
    ).toBeInTheDocument();
    expect(screen.getByText('Review the pending update')).toBeInTheDocument();
    expect(screen.getByText('Cycles 24h')).toBeInTheDocument();
  });
});

describe('FeedbackLearning — #1244 Recent Activity pattern attribution', () => {
  // Shaped like the REAL prod pattern row (fb_c7189882e96d cycle): the API's
  // DetectedPattern carries affected_agents + detected_at, but never
  // agent_name / last_seen (those are UI/sample-era fields).
  const realPattern = {
    pattern_id: 'P1',
    pattern_type: 'relevance_issue',
    severity: 'high',
    description: "Agent 'cognitive_investigator' has high negative feedback rate",
    frequency: 10,
    confidence: 0.7,
    affected_agents: ['cognitive_investigator'],
    example_feedback_ids: ['d36dcc05'],
    root_cause_hypothesis: 'May need retraining or prompt updates',
    detected_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { agent_available: true, cycles_24h: 1 },
      refetch: vi.fn().mockResolvedValue({}),
    });
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [realPattern] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { updates: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useQuickLearningCycle as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    });
    (useApplyUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useRollbackUpdate as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false });
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({ mutate: vi.fn(), isPending: false, data: undefined, error: null });
  });

  it('falls back to affected_agents[0] and detected_at instead of N/A • N/A', () => {
    render(<FeedbackLearning />, { wrapper: createWrapper() });

    // Recent Activity (Overview tab, default) renders the pattern row with
    // real attribution, not the N/A • N/A placeholder.
    expect(screen.getByText('Pattern Detected')).toBeInTheDocument();
    expect(screen.queryByText(/N\/A\s*•\s*N\/A/)).not.toBeInTheDocument();
    expect(screen.getByText(/cognitive_investigator\s*•/)).toBeInTheDocument();
    // detected_at (2h old) renders as a relative time, not N/A.
    expect(screen.getByText(/•\s*2h ago/)).toBeInTheDocument();
  });
});
