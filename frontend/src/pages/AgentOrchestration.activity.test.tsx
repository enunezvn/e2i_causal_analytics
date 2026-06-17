/**
 * AgentOrchestration — Activity Feed wiring tests.
 *
 * The Activity Feed was a hardcoded `ACTIVITIES = []` (always "No activity to
 * display"). It is now wired to GET /agents/activity (real rows from
 * audit_chain_entries, automated health poller excluded server-side). These
 * tests verify real activity renders, and that an empty response is an honest
 * empty state rather than a fabricated row.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AgentOrchestration from './AgentOrchestration';

// getValidated is called for BOTH /agents/status and /agents/activity; dispatch
// by path so each query gets its own payload.
const statusState: { data: unknown } = { data: undefined };
const activityState: { data: unknown } = { data: undefined };

vi.mock('@/lib/api-client', () => ({
  getValidated: vi.fn((_schema: unknown, path: string) =>
    Promise.resolve(path.startsWith('/agents/activity') ? activityState.data : statusState.data),
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

// Faithful to the GET /agents/activity wire shape (snake_case).
const realActivity = {
  activities: [
    {
      entry_id: 'e1',
      agent_id: 'gap-analyzer',
      agent_name: 'Gap Analyzer',
      tier: 2,
      action: 'Gap Detector',
      action_type: 'gap_detector',
      timestamp: '2026-06-16T11:00:00Z',
      duration_ms: 42,
      status: 'completed',
      details: 'find gaps for Remibrutinib',
    },
  ],
  total: 1,
  window_hours: 24,
  timestamp: '2026-06-16T13:00:00Z',
};

describe('AgentOrchestration — Activity Feed wiring', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    statusState.data = undefined;
    activityState.data = realActivity;
  });

  it('renders a REAL activity item from /agents/activity in the overview preview', async () => {
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    // The Recent Activity preview (Overview tab, default) shows the action.
    expect(await screen.findByText('Gap Detector')).toBeInTheDocument();
    expect(screen.getAllByText('Gap Analyzer').length).toBeGreaterThan(0);
  });

  it('shows an honest empty state (not a fabricated row) when the feed is empty', async () => {
    activityState.data = { activities: [], total: 0, window_hours: 24 };
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    // Overview preview renders the empty state; never a fake activity.
    expect(await screen.findByText('No recent activity')).toBeInTheDocument();
    expect(screen.queryByText('Gap Detector')).not.toBeInTheDocument();
  });
});
