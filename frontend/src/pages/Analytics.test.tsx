/**
 * Analytics Page Tests
 * ====================
 * Covers: real metric rendering, the poller-exclusion transparency note,
 * the latency/volume trend charts, and the honest error state.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

vi.mock('@/hooks/api/use-analytics', () => ({
  useAnalyticsDashboard: vi.fn(),
}));
vi.mock('@/hooks/use-data-freshness', () => ({
  useDataFreshness: vi.fn(() => ({})),
}));
vi.mock('@/components/ui/data-freshness-indicator', () => ({
  DataFreshnessIndicator: () => null,
}));

import { useAnalyticsDashboard } from '@/hooks/api/use-analytics';
import Analytics from './Analytics';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function dashboard(overrides: Record<string, unknown> = {}) {
  return {
    summary: {
      period_start: '2026-06-09T00:00:00Z',
      period_end: '2026-06-16T00:00:00Z',
      total_queries: 27,
      successful_queries: 25,
      failed_queries: 2,
      success_rate: 92.59,
      avg_latency_ms: 1122.67,
      p50_latency_ms: 900,
      p95_latency_ms: 2000,
      p99_latency_ms: 3000,
      intent_distribution: {},
      top_agents: ['heterogeneous_optimizer', 'gap_analyzer'],
    },
    agent_metrics: [
      {
        agent_name: 'heterogeneous_optimizer',
        agent_tier: 2,
        total_invocations: 30,
        successful_invocations: 28,
        failed_invocations: 2,
        success_rate: 93.3,
        avg_latency_ms: 1100,
        p50_latency_ms: 900,
        p95_latency_ms: 2000,
        p99_latency_ms: 3000,
        min_latency_ms: 100,
        max_latency_ms: 3500,
        avg_confidence: 0.8,
      },
    ],
    latency_trend: [{ timestamp: '2026-06-14T06:00:00Z', value: 1100, label: null }],
    query_volume_trend: [{ timestamp: '2026-06-14T06:00:00Z', value: 6, label: null }],
    latency_breakdown: {
      classification_ms: 10,
      rag_retrieval_ms: 0,
      routing_ms: 5,
      agent_dispatch_ms: 1000,
      synthesis_ms: 50,
      total_ms: 1122,
    },
    generated_at: '2026-06-16T00:00:00Z',
    excluded_background_count: 6906,
    excluded_agents: ['health_score_quick'],
    ...overrides,
  };
}

const mockDashboard = (overrides: Record<string, unknown> = {}) => {
  (useAnalyticsDashboard as ReturnType<typeof vi.fn>).mockReturnValue({
    data: dashboard(),
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    isFetching: false,
    dataUpdatedAt: Date.now(),
    ...overrides,
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  mockDashboard();
});

describe('Analytics', () => {
  it('renders real query metrics (not poller-inflated)', () => {
    render(<Analytics />, { wrapper: createWrapper() });
    // Total Queries reflects the analytical count (27), not the ~2,373 poller total.
    expect(screen.getByText('27')).toBeInTheDocument();
    // The real analytical agent surfaces in Top Agents / the table.
    expect(screen.getAllByText('heterogeneous_optimizer').length).toBeGreaterThan(0);
  });

  it('discloses excluded automated background agents (transparency, not silent drop)', () => {
    render(<Analytics />, { wrapper: createWrapper() });
    const note = screen.getByRole('note');
    expect(note).toHaveTextContent(/automated health-poll/i);
    // The excluded poller is named so the filtering is transparent.
    expect(note).toHaveTextContent('health_score_quick');
  });

  it('does NOT show the transparency note when nothing was excluded', () => {
    mockDashboard({ data: dashboard({ excluded_background_count: 0, excluded_agents: [] }) });
    render(<Analytics />, { wrapper: createWrapper() });
    expect(screen.queryByRole('note')).not.toBeInTheDocument();
  });

  it('renders the latency and query-volume trend charts', () => {
    render(<Analytics />, { wrapper: createWrapper() });
    expect(screen.getByText('Query Volume')).toBeInTheDocument();
    expect(screen.getByText('Latency Trend')).toBeInTheDocument();
  });

  it('renders an honest error state on failure', () => {
    mockDashboard({ data: undefined, error: new Error('metrics store unavailable') });
    render(<Analytics />, { wrapper: createWrapper() });
    expect(screen.getByText(/Failed to load analytics/i)).toBeInTheDocument();
    expect(screen.getByText(/metrics store unavailable/i)).toBeInTheDocument();
  });
});
