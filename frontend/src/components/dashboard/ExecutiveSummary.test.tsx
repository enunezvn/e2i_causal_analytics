/**
 * ExecutiveSummary Tests
 * ======================
 *
 * Verified finding (fix/fe-home-fake-data task 3): the component fabricated
 * numeric fallbacks (142 relationships / 847 nodes / 12 communities / 1.47M
 * journeys), a hardcoded activeAgents=8, a fabricated healthScore (84/72
 * derived from a status string), an invented dollar-impact formula
 * (totalRelationships * 0.167 rendered as "$X.XM Est. Impact" + prose), and
 * three hardcoded "causal insight" cards presented as real-time analysis.
 *
 * Real substrate exists and must be wired instead:
 * - useGraphStats()           -> relationships/nodes/communities/episodes
 * - useQuickHealthCheck()     -> real overall_health_score + grade
 * - GET /agents/status        -> real agent roster (active/total)
 *
 * Everything without substrate must be honestly absent.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '@/test/utils';
import { ExecutiveSummary } from './ExecutiveSummary';

vi.mock('@/hooks/api/use-graph', () => ({
  useGraphStats: vi.fn(),
}));
vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIHealth: vi.fn(),
}));
vi.mock('@/hooks/api/use-health-score', () => ({
  useQuickHealthCheck: vi.fn(),
}));
vi.mock('@/lib/api-client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api-client')>();
  return { ...actual, getValidated: vi.fn() };
});

import { useGraphStats } from '@/hooks/api/use-graph';
import { useKPIHealth } from '@/hooks/api/use-kpi';
import { useQuickHealthCheck } from '@/hooks/api/use-health-score';
import { getValidated } from '@/lib/api-client';

function setDefaults() {
  (useGraphStats as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({ data: undefined });
  (useQuickHealthCheck as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
    error: null,
  });
  (getValidated as ReturnType<typeof vi.fn>).mockRejectedValue(
    new Error('agents service unavailable')
  );
}

describe('ExecutiveSummary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setDefaults();
  });

  it('fabricates NOTHING when no live data is available (no 142/847/$M/8 agents/84%)', () => {
    renderWithProviders(<ExecutiveSummary />);

    // The old hardcoded numeric fallbacks must not render.
    expect(screen.queryByText('142')).not.toBeInTheDocument();
    expect(screen.queryByText('847')).not.toBeInTheDocument();
    expect(screen.queryByText('1.5M')).not.toBeInTheDocument();
    // The invented dollar-impact formula (142 * 0.167 = $23.7M) must not render.
    expect(screen.queryByText(/\$23\.7M/)).not.toBeInTheDocument();
    expect(screen.queryByText('Est. Impact')).not.toBeInTheDocument();
    // The fabricated health/agent prose must not render.
    expect(screen.queryByText(/72% health/)).not.toBeInTheDocument();
    expect(screen.queryByText(/84% health/)).not.toBeInTheDocument();
    expect(screen.queryByText(/8 active AI agents/)).not.toBeInTheDocument();
    expect(screen.queryByText(/worth \$/)).not.toBeInTheDocument();
    // Honest absence instead.
    expect(
      screen.getByText(/live system metrics are currently unavailable/i)
    ).toBeInTheDocument();
  });

  it('never renders the fabricated healthScore (84) derived from a KPI status string', () => {
    (useKPIHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { status: 'healthy' },
    });

    renderWithProviders(<ExecutiveSummary />);

    expect(screen.queryByText('84%')).not.toBeInTheDocument();
    expect(screen.queryByText(/84% health/)).not.toBeInTheDocument();
  });

  it('renders REAL graph stats, REAL health score, and the REAL agent roster', async () => {
    (useGraphStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        total_nodes: 210,
        total_relationships: 33,
        total_episodes: 5200,
        total_communities: 4,
      },
      isLoading: false,
      error: null,
    });
    (useQuickHealthCheck as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        check_id: 'c1',
        check_scope: 'quick',
        overall_health_score: 91,
        health_grade: 'A',
        component_health_score: 0.95,
        model_health_score: 0.88,
        pipeline_health_score: 0.82,
        agent_health_score: 0.92,
      },
      isLoading: false,
      error: null,
    });
    (getValidated as ReturnType<typeof vi.fn>).mockResolvedValue({
      agents: [
        { id: 'a1', name: 'Orchestrator', tier: 1, status: 'active', capabilities: [] },
        { id: 'a2', name: 'Causal Impact', tier: 2, status: 'active', capabilities: [] },
        { id: 'a3', name: 'Scope Definer', tier: 0, status: 'idle', capabilities: [] },
      ],
      total: 3,
    });

    renderWithProviders(<ExecutiveSummary />);

    // Real graph stats.
    expect(screen.getByText('33')).toBeInTheDocument();
    expect(screen.getByText('210')).toBeInTheDocument();
    // Real health score from the Health Score agent (not 84/72).
    expect(screen.getByText('91%')).toBeInTheDocument();
    // Real roster (async query).
    expect(await screen.findByText('2/3')).toBeInTheDocument();
    // The invented $-impact (33 * 0.167 = $5.5M) must never come back.
    expect(screen.queryByText(/\$5\.5M/)).not.toBeInTheDocument();
    expect(screen.queryByText('Est. Impact')).not.toBeInTheDocument();
  });

  it('does NOT render the three hardcoded "causal insight" cards', () => {
    renderWithProviders(<ExecutiveSummary />);

    expect(screen.queryByText('3.2x ROI')).not.toBeInTheDocument();
    expect(screen.queryByText('58% → 75%')).not.toBeInTheDocument();
    expect(screen.queryByText('4.2pp gap')).not.toBeInTheDocument();
    expect(screen.queryByText('Data-to-Value Pipeline')).not.toBeInTheDocument();
    expect(screen.queryByText('Model-to-Impact Bridge')).not.toBeInTheDocument();
    expect(screen.queryByText('Fairness & Trust Nexus')).not.toBeInTheDocument();
    expect(screen.queryByText('Causal Intelligence Finding')).not.toBeInTheDocument();
  });

  it('shows the loading skeleton while graph stats load', () => {
    (useGraphStats as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    });

    const { container } = renderWithProviders(<ExecutiveSummary />);

    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
  });
});
