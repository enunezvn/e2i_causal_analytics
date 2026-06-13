/**
 * SystemHealthScore Tests
 * =======================
 *
 * Red-first guards for the fabricated-health finding: the widget formerly
 * booted SAMPLE_METRICS / SAMPLE_SUMMARY (87% score, "1,247 predictions/min",
 * "P95 latency slightly elevated (245ms)", 8 models) and mapped the API's
 * health status onto invented scores (healthy->95, warning->75, critical->45).
 *
 * Desired behavior: the real /health-score/full measured score (0-100),
 * real dimension scores (component/model/pipeline/agent, "—" when null),
 * real alert counts; honest empty/error states otherwise.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SystemHealthScore } from './SystemHealthScore';
import * as useHealthScore from '@/hooks/api/use-health-score';
import * as useMonitoring from '@/hooks/api/use-monitoring';
import type { HealthScoreResponse } from '@/types/health-score';

vi.mock('@/hooks/api/use-health-score');
vi.mock('@/hooks/api/use-monitoring');

type FullCheckQuery = ReturnType<typeof useHealthScore.useFullHealthCheck>;
type AlertsQuery = ReturnType<typeof useMonitoring.useAlerts>;
type ModelHealthQuery = ReturnType<typeof useMonitoring.useModelHealth>;

function mockFullCheck(overrides: Partial<FullCheckQuery> = {}) {
  vi.mocked(useHealthScore.useFullHealthCheck).mockReturnValue({
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  } as unknown as FullCheckQuery);
}

function mockAlerts(overrides: Partial<AlertsQuery> = {}) {
  vi.mocked(useMonitoring.useAlerts).mockReturnValue({
    data: undefined,
    isLoading: false,
    ...overrides,
  } as unknown as AlertsQuery);
}

function mockModelHealth(overrides: Partial<ModelHealthQuery> = {}) {
  vi.mocked(useMonitoring.useModelHealth).mockReturnValue({
    data: undefined,
    isLoading: false,
    refetch: vi.fn(),
    ...overrides,
  } as unknown as ModelHealthQuery);
}

const REAL_HEALTH: HealthScoreResponse = {
  check_id: 'chk_1',
  check_scope: 'full',
  overall_health_score: 73.4,
  health_grade: 'C',
  component_health_score: 0.91,
  model_health_score: 0.55,
  pipeline_health_score: null as unknown as number,
  agent_health_score: 0.8,
  critical_issues: ['Model registry degraded'],
  warnings: [],
  recommendations: ['Retrain churn model'],
  health_summary: 'Degraded: model dimension below threshold.',
  check_latency_ms: 412,
  timestamp: '2026-06-12T02:00:00Z',
  data_provenance: 'measured',
} as unknown as HealthScoreResponse;

beforeEach(() => {
  vi.clearAllMocks();
  mockFullCheck();
  mockAlerts();
  mockModelHealth();
});

describe('SystemHealthScore — no fabricated metrics', () => {
  it('renders an honest empty state (not the 87% SAMPLE_SUMMARY) when no data', () => {
    render(<SystemHealthScore />);

    // Fabricated SAMPLE values must never render.
    expect(screen.queryByText('87%')).not.toBeInTheDocument();
    expect(screen.queryByText(/1,247 predictions\/min/)).not.toBeInTheDocument();
    expect(screen.queryByText(/P95 latency slightly elevated/)).not.toBeInTheDocument();
    expect(screen.queryByText('Inference Throughput')).not.toBeInTheDocument();

    expect(screen.getByTestId('empty-state')).toBeInTheDocument();
  });

  it('renders the real measured health score and dimensions from /health-score/full', () => {
    mockFullCheck({ data: REAL_HEALTH } as unknown as Partial<FullCheckQuery>);
    mockAlerts({
      data: { total_count: 4, active_count: 4, alerts: [] },
    } as unknown as Partial<AlertsQuery>);

    render(<SystemHealthScore />);

    // Real overall score, not invented 95/75/45 mapping.
    expect(screen.getByText('73%')).toBeInTheDocument();
    expect(screen.getByText(/grade c/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Degraded: model dimension below threshold/)
    ).toBeInTheDocument();

    // Real dimension scores.
    expect(screen.getByText('Component Health')).toBeInTheDocument();
    expect(screen.getByText('91%')).toBeInTheDocument();
    expect(screen.getByText('55%')).toBeInTheDocument();
    // Unmeasured dimension renders an em-dash, never a fake number.
    expect(screen.getByText('Pipeline Health')).toBeInTheDocument();
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);

    // Real active alert count from the monitoring API.
    expect(screen.getByText('4')).toBeInTheDocument();

    // Real recommendations surface.
    expect(screen.getByText(/Retrain churn model/)).toBeInTheDocument();
  });

  it('shows a labeled error state when the health check fails', () => {
    mockFullCheck({
      isError: true,
      error: new Error('health service down'),
    } as unknown as Partial<FullCheckQuery>);

    render(<SystemHealthScore />);
    expect(screen.getByText(/unable to load system health/i)).toBeInTheDocument();
    expect(screen.queryByText('87%')).not.toBeInTheDocument();
  });

  it('labels non-measured provenance honestly instead of presenting it as real', () => {
    mockFullCheck({
      data: { ...REAL_HEALTH, data_provenance: 'placeholder' },
    } as unknown as Partial<FullCheckQuery>);

    render(<SystemHealthScore />);
    expect(screen.getByText(/placeholder/i)).toBeInTheDocument();
  });
});
