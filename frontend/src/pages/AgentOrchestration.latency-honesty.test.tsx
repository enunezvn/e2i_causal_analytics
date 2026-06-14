/**
 * AgentOrchestration — Problem A honesty regression tests.
 *
 * Two distinct defects, both verified against the real prod responses:
 *
 *  1. Field mismatch: GET /api/agents/status returns `total_agents`, but the
 *     frontend reads `agentStatus?.total` and api-schemas only declares `total`.
 *     The backend now also emits `total` (alias) so the real count resolves
 *     instead of silently falling back.
 *
 *  2. Fake "0ms": /analytics/summary returned `avg_latency_ms: 0.0` for a
 *     genesis-only window (no audit entry carried a real duration_ms). The UI
 *     rendered "0ms" — reads like "instant", which is false. A 0 (or null)
 *     avg with no real timed entries must render "—" (not measured), not "0ms".
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AgentOrchestration from './AgentOrchestration';

// ---- Mock the live agent-status fetch (getValidated) -----------------------
// Faithful to the real /api/agents/status shape: it carries an `agents` array
// AND a `total` alias for `total_agents`.
const agentStatusState: { data: unknown } = { data: undefined };

vi.mock('@/lib/api-client', () => ({
  getValidated: vi.fn(() => Promise.resolve(agentStatusState.data)),
}));

// ---- Mock the telemetry summary hook --------------------------------------
const summaryState: { data: unknown; isLoading: boolean } = {
  data: undefined,
  isLoading: false,
};

vi.mock('@/hooks/api/use-analytics', () => ({
  useMetricsSummary: () => ({
    data: summaryState.data,
    isLoading: summaryState.isLoading,
    refetch: vi.fn(),
  }),
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

// A real 21-agent roster (only 3 shown for brevity); `total` is the alias the
// backend now emits and the frontend reads.
const realAgentStatus = {
  agents: [
    { id: 'orchestrator', name: 'Orchestrator', tier: 1, status: 'active', capabilities: [] },
    { id: 'gap-analyzer', name: 'Gap Analyzer', tier: 2, status: 'idle', capabilities: [] },
    { id: 'explainer', name: 'Explainer', tier: 5, status: 'idle', capabilities: [] },
  ],
  total: 21,
  timestamp: '2026-06-14T09:00:00Z',
};

// A StatCard renders the title in a CardTitle and the value in a sibling div
// inside the same Card. Walk up from the title to the nearest ancestor that
// also contains the value (i.e. the Card root), so we can assert on the value.
async function findStatCard(title: string): Promise<HTMLElement | null> {
  const titleEl = await screen.findByText(title);
  let node: HTMLElement | null = titleEl.parentElement;
  // Climb at most a few levels to the Card root (CardHeader -> Card).
  for (let i = 0; i < 5 && node; i += 1) {
    // The Card root contains the value div (text-2xl font-bold) as well.
    if (node.querySelector('.text-2xl')) {
      return node;
    }
    node = node.parentElement;
  }
  return node;
}

describe('AgentOrchestration latency/total honesty', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    agentStatusState.data = realAgentStatus;
    summaryState.data = undefined;
    summaryState.isLoading = false;
  });

  it('renders "—" (not "0ms") when avg_latency_ms is 0 with no timed entries', async () => {
    // genesis-only window: queries happened (total_queries=24) but no entry
    // carried a duration_ms, so the backend averaged an empty list -> 0.
    summaryState.data = {
      period_start: '2026-06-13T09:00:00Z',
      period_end: '2026-06-14T09:00:00Z',
      total_queries: 24,
      successful_queries: 24,
      failed_queries: 0,
      success_rate: 100,
      avg_latency_ms: 0,
      p50_latency_ms: 0,
      p95_latency_ms: 0,
      p99_latency_ms: 0,
      intent_distribution: {},
      top_agents: ['health_score', 'experiment_monitor'],
    };

    render(<AgentOrchestration />, { wrapper: createWrapper() });

    const container = await findStatCard('Avg Response Time');
    expect(container).toBeTruthy();
    // Honest: shows the em dash, never a fabricated "0ms".
    expect(container?.textContent).toContain('—');
    expect(container?.textContent).not.toContain('0ms');
  });

  it('renders "—" when avg_latency_ms is null (unmeasured)', async () => {
    summaryState.data = {
      period_start: '2026-06-13T09:00:00Z',
      period_end: '2026-06-14T09:00:00Z',
      total_queries: 24,
      successful_queries: 24,
      failed_queries: 0,
      success_rate: 100,
      avg_latency_ms: null,
      p50_latency_ms: null,
      p95_latency_ms: null,
      p99_latency_ms: null,
      intent_distribution: {},
      top_agents: [],
    };

    render(<AgentOrchestration />, { wrapper: createWrapper() });
    const container = await findStatCard('Avg Response Time');
    expect(container?.textContent).toContain('—');
    expect(container?.textContent).not.toContain('0ms');
  });

  it('renders a REAL measured latency when present', async () => {
    summaryState.data = {
      period_start: '2026-06-13T09:00:00Z',
      period_end: '2026-06-14T09:00:00Z',
      total_queries: 24,
      successful_queries: 24,
      failed_queries: 0,
      success_rate: 100,
      avg_latency_ms: 142.6,
      p50_latency_ms: 120,
      p95_latency_ms: 300,
      p99_latency_ms: 410,
      intent_distribution: {},
      top_agents: [],
    };

    render(<AgentOrchestration />, { wrapper: createWrapper() });
    const container = await findStatCard('Avg Response Time');
    // Rounded real value, with ms suffix.
    expect(container?.textContent).toContain('143ms');
  });

  it('resolves the real total from the `total` alias (field mismatch fix)', async () => {
    summaryState.data = undefined;
    render(<AgentOrchestration />, { wrapper: createWrapper() });

    // Must resolve to the real total (21), not the 3-agent array length.
    await waitFor(async () => {
      const container = await findStatCard('Total Agents');
      expect(container?.textContent).toContain('21');
    });
  });
});
