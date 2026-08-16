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

  it('sorts a weeks-old pattern below a fresh Knowledge Update (detected_at-aware)', () => {
    // codex-1244 MEDIUM: the Recent Activity comparator only knew
    // last_seen/created_at; real API patterns carry neither, so an old
    // pattern fell through to "now" and displaced genuinely-recent updates
    // — self-contradicting the row's own "3 weeks ago" label.
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        patterns: [
          {
            ...realPattern,
            detected_at: new Date(Date.now() - 21 * 24 * 60 * 60 * 1000).toISOString(),
          },
        ],
      },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        updates: [
          {
            update_id: 'U1',
            update_type: 'prompt_refinement',
            status: 'proposed',
            target_agent: 'cognitive_investigator',
            description: 'Refine investigator prompt for relevance',
            created_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
          },
        ],
      },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    const rowLabels = screen
      .getAllByText(/^(Pattern Detected|Knowledge Update)$/)
      .map((el) => el.textContent);
    expect(rowLabels).toEqual(['Knowledge Update', 'Pattern Detected']);
  });

  it('update rows attribute via target_agent when agent_name is absent (#1263)', () => {
    // The learner's real KnowledgeUpdate rows carry target_agent only
    // (agent_name is a UI/sample-era field). The fallback chain
    // agent_name ?? target_agent ?? 'N/A' was rendered by the sort test
    // above but never discriminated — regressing it to agent_name ?? 'N/A'
    // re-shows the #1244 "N/A" symptom with the suite green.
    (usePatterns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { patterns: [] },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
    (useKnowledgeUpdates as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        updates: [
          {
            update_id: 'U2',
            update_type: 'prompt_refinement',
            status: 'proposed',
            target_agent: 'gap_analyzer',
            description: 'Refine gap analyzer prompt',
            created_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
          },
        ],
      },
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    expect(screen.getByText('Knowledge Update')).toBeInTheDocument();
    expect(screen.getByText(/gap_analyzer\s*•/)).toBeInTheDocument();
    expect(screen.queryByText(/N\/A\s*•/)).not.toBeInTheDocument();
  });
});

describe('FeedbackLearning — optimizer gate visibility (#1661)', () => {
  const baseMocks = () => {
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
    (useFeedbackLearningInsight as ReturnType<typeof vi.fn>).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      data: undefined,
      error: null,
    });
  };

  beforeEach(() => {
    vi.clearAllMocks();
    baseMocks();
  });

  it('surfaces an inert optimizer with its yield denominator, not just the shortfall', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        agent_available: true,
        cycles_24h: 1,
        optimizer: {
          eligible_signals: 8,
          total_signals: 218,
          last_eligible_signal_at: '2026-08-08T07:09:02.686027+00:00',
          optimization_runs: 0,
          min_signals: 20,
          min_reward: 0.5,
          would_trigger: false,
          // Verbatim from the beat's own trigger — not a re-worded copy.
          reason: 'Insufficient signals: 8 < 20',
        },
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    // The shortfall AND the denominator — "8" alone reads as a volume problem.
    expect(screen.getByText('8 / 20')).toBeInTheDocument();
    expect(screen.getByText(/218 signals/i)).toBeInTheDocument();
    // "Never optimized" is the fact the page currently hides behind "Online".
    expect(screen.getByText(/never optimized/i)).toBeInTheDocument();
    expect(screen.getByText('Insufficient signals: 8 < 20')).toBeInTheDocument();
    expect(screen.getByText('Inert')).toBeInTheDocument();
  });

  it('surfaces the cooldown gate once the signal gate opens', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        agent_available: true,
        cycles_24h: 1,
        optimizer: {
          eligible_signals: 25,
          total_signals: 300,
          last_eligible_signal_at: '2026-08-16T00:00:00+00:00',
          optimization_runs: 1,
          min_signals: 20,
          min_reward: 0.5,
          would_trigger: false,
          reason: 'Cooldown active: 2.0h < 24h',
        },
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    // Count gate satisfied, yet still not running — the page must say WHY.
    expect(screen.getByText('25 / 20')).toBeInTheDocument();
    expect(screen.getByText('Cooldown active: 2.0h < 24h')).toBeInTheDocument();
    expect(screen.getByText('Inert')).toBeInTheDocument();
  });

  it('shows a ready optimizer once the gate is satisfied', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        agent_available: true,
        cycles_24h: 1,
        optimizer: {
          eligible_signals: 25,
          total_signals: 300,
          last_eligible_signal_at: '2026-08-16T00:00:00+00:00',
          optimization_runs: 3,
          min_signals: 20,
          min_reward: 0.5,
          would_trigger: true,
          reason: 'Reward improved: 0.600 >= 0.05',
        },
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    expect(screen.getByText('25 / 20')).toBeInTheDocument();
    expect(screen.getByText(/3 optimization runs/i)).toBeInTheDocument();
    expect(screen.queryByText(/never optimized/i)).not.toBeInTheDocument();
  });

  it('renders unknown — never a fabricated zero — when the gate read failed', () => {
    (useFeedbackHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        agent_available: true,
        cycles_24h: 1,
        optimizer: {
          eligible_signals: null,
          total_signals: null,
          last_eligible_signal_at: null,
          optimization_runs: null,
          min_signals: 20,
          min_reward: 0.5,
          would_trigger: null,
          reason: 'Optimizer gate status unavailable (db down)',
        },
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    expect(screen.getByText('— / 20')).toBeInTheDocument();
    expect(screen.getByText(/status unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/never optimized/i)).not.toBeInTheDocument();
  });
});
