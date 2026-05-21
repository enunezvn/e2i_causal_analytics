/**
 * FeedbackLearning Page — Warning Banner Coverage
 * ===============================================
 *
 * Regression tests for F-010-frontend: API-reported `warnings[]` from
 * the learning-cycle mutation are surfaced to the user as a banner.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
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
}));

import {
  useFeedbackHealth,
  usePatterns,
  useKnowledgeUpdates,
  useQuickLearningCycle,
  useApplyUpdate,
  useRollbackUpdate,
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
  it('does not render fabricated SAMPLE_PATTERNS / SAMPLE_UPDATES when API empty', () => {
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

    render(<FeedbackLearning />, { wrapper: createWrapper() });

    // Former SAMPLE_PATTERNS descriptions must not appear in DOM.
    expect(
      screen.queryByText(/Causal Impact agent showing increased latency during peak hours/),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Refine causal impact explanation template/),
    ).not.toBeInTheDocument();
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
