/**
 * AgentOrchestration — Tier Metrics wiring tests.
 *
 * The Tier Metrics tab previously rendered "—" for every per-tier Avg Response
 * and Tasks (the values were hardcoded null). They are now wired to GET
 * /analytics/tier-metrics (audit_chain_entries, automated health poller
 * excluded). These tests verify real per-tier values RENDER, that an idle tier
 * shows an honest "no activity" state (not a fabricated 0), and that per-tier
 * success rate stays "—" (validation is too sparse to report honestly).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AgentOrchestration from './AgentOrchestration';

// getValidated is called for BOTH /agents/status and /analytics/tier-metrics;
// dispatch by path so each query gets its own payload.
const statusState: { data: unknown } = { data: undefined };
const tierState: { data: unknown } = { data: undefined };

vi.mock('@/lib/api-client', () => ({
  getValidated: vi.fn((_schema: unknown, path: string) =>
    Promise.resolve(
      path.startsWith('/analytics/tier-metrics') ? tierState.data : statusState.data,
    ),
  ),
}));

vi.mock('@/hooks/api/use-analytics', () => ({
  useMetricsSummary: () => ({ data: undefined, isLoading: false, refetch: vi.fn() }),
}));

vi.mock('@/providers/E2ICopilotProvider', () => ({
  useE2ICopilot: vi.fn(() => ({
    agents: [],
    filters: { brand: 'All' },
    preferences: { detailLevel: 'detailed' },
  })),
}));

vi.mock('@/components/visualizations/agents/AgentTierBadge', () => ({
  TierOverview: () => <div data-testid="tier-overview" />,
  AgentTierBadge: ({ tier }: { tier: number }) => <span>Tier {tier}</span>,
}));

vi.mock('@/components/chat/AgentStatusPanel', () => ({
  AgentStatusPanel: () => <div data-testid="agent-status-panel" />,
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// Faithful to the GET /analytics/tier-metrics wire shape (snake_case): tier 2
// has real activity, tier 0 is idle, success_rate is null for all.
const realTierMetrics = {
  tiers: [
    { tier: 0, tasks_completed: 0, avg_response_time_ms: null, success_rate: null },
    { tier: 1, tasks_completed: 0, avg_response_time_ms: null, success_rate: null },
    { tier: 2, tasks_completed: 26, avg_response_time_ms: 1234, success_rate: null },
    { tier: 3, tasks_completed: 54, avg_response_time_ms: 200, success_rate: null },
    { tier: 4, tasks_completed: 2, avg_response_time_ms: 1, success_rate: null },
    { tier: 5, tasks_completed: 0, avg_response_time_ms: null, success_rate: null },
  ],
  window_hours: 24,
  generated_at: '2026-06-16T13:00:00Z',
};

async function openTierMetricsTab() {
  const user = userEvent.setup();
  render(<AgentOrchestration />, { wrapper: createWrapper() });
  await user.click(await screen.findByRole('tab', { name: /tier metrics/i }));
}

describe('AgentOrchestration — Tier Metrics wiring', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    statusState.data = { agents: [], total: 0 };
    tierState.data = realTierMetrics;
  });

  it('renders REAL per-tier avg response and task counts from the endpoint', async () => {
    await openTierMetricsTab();
    // Tier 2's real average latency and a tier's real task count render.
    expect(await screen.findByText('1234ms')).toBeInTheDocument();
    expect(screen.getByText('200ms')).toBeInTheDocument();
    expect(screen.getByText('54')).toBeInTheDocument();
  });

  it('shows an honest empty state for an idle tier (not a fabricated 0)', async () => {
    await openTierMetricsTab();
    // Idle tiers (0/1/5) have no activity -> honest note, never "0ms".
    expect(
      (await screen.findAllByText(/No agent activity recorded for this tier/i)).length,
    ).toBeGreaterThan(0);
  });

  it('keeps per-tier success rate as "—" (validation too sparse to report)', async () => {
    await openTierMetricsTab();
    // Active tiers surface the honest caveat rather than a fabricated rate.
    expect(
      (await screen.findAllByText(/Success rate is not recorded per tier/i)).length,
    ).toBeGreaterThan(0);
  });
});
