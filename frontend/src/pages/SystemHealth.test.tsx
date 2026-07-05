/**
 * SystemHealth Page Tests
 * =======================
 *
 * Tests for the System Health monitoring dashboard page.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SystemHealth from './SystemHealth';
import { AlertStatus } from '@/types/monitoring';

// Mock the monitoring hooks
vi.mock('@/hooks/api/use-monitoring', () => ({
  useAlerts: vi.fn(),
  useMonitoringRuns: vi.fn(),
}));

// Mock the health-score hooks (imported from the @/hooks/api barrel by the page).
// Each hook gets a default no-data implementation in beforeEach so existing
// empty-state tests keep their (undefined data) behaviour, and the new wiring
// tests override the two we care about.
vi.mock('@/hooks/api', () => ({
  useFullHealthCheck: vi.fn(),
  usePipelineHealth: vi.fn(),
  useAgentHealth: vi.fn(),
  useHealthHistory: vi.fn(),
  useComponentHealth: vi.fn(),
  useModelHealth: vi.fn(),
}));

import { useAlerts, useMonitoringRuns } from '@/hooks/api/use-monitoring';
import {
  useFullHealthCheck,
  usePipelineHealth,
  useAgentHealth,
  useHealthHistory,
  useComponentHealth,
  useModelHealth,
} from '@/hooks/api';

type MockFn = ReturnType<typeof vi.fn>;

// Create wrapper with QueryClientProvider
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// Mock data for alerts
const mockAlertsData = {
  alerts: [
    {
      id: 'alert-1',
      title: 'Data Drift Detected',
      description: 'Feature distribution has shifted significantly.',
      severity: 'high',
      model_version: 'churn_v1.5.2',
      triggered_at: new Date().toISOString(),
      status: AlertStatus.ACTIVE,
    },
    {
      id: 'alert-2',
      title: 'Performance Degradation',
      description: 'Model accuracy dropped below threshold.',
      severity: 'medium',
      model_version: 'propensity_v2.1.0',
      triggered_at: new Date().toISOString(),
      status: AlertStatus.ACTIVE,
    },
  ],
  active_count: 2,
  total_count: 5,
};

// Mock data for monitoring runs
const mockRunsData = {
  runs: [
    { id: 'run-1', model_id: 'propensity_v2.1.0', started_at: new Date().toISOString(), status: 'completed' },
    { id: 'run-2', model_id: 'churn_v1.5.2', started_at: new Date().toISOString(), status: 'completed' },
  ],
  total_runs: 15,
};

// Real-shaped /components payload (mirrors ComponentHealthResponse from the
// backend health_score.py route, data_provenance="measured"). Counts are kept
// internally consistent with the components list (1 healthy + 1 degraded), as
// the real backend computes them from the same list.
const measuredComponentHealth = {
  component_health_score: 0.75,
  total_components: 2,
  healthy_count: 1,
  degraded_count: 1,
  unhealthy_count: 0,
  components: [
    {
      component_name: 'Database',
      status: 'healthy',
      latency_ms: 12,
      last_check: new Date().toISOString(),
    },
    {
      component_name: 'Cache (Redis)',
      status: 'degraded',
      latency_ms: 240,
      last_check: new Date().toISOString(),
    },
  ],
  check_latency_ms: 30,
  data_provenance: 'measured',
};

// Real-shaped /models payload (mirrors ModelHealthResponse). The backend
// model_health domain reports status (measured) plus accuracy / error_rate /
// predictions_last_24h, each of which is left null when ml_performance_metrics
// has no source row. It does NOT return drift scores or a performance trend.
// First model = performance sub-fields populated; second = partial (sub-fields
// null) — exactly the two states the live endpoint emits.
const measuredModelHealth = {
  model_health_score: 0.75,
  total_models: 2,
  healthy_count: 1,
  degraded_count: 1,
  unhealthy_count: 0,
  models: [
    {
      model_id: 'mdl-001',
      model_name: 'CSU Initiation Model',
      accuracy: 0.83,
      error_rate: 0.04,
      predictions_last_24h: 1500,
      status: 'healthy',
    },
    {
      model_id: 'mdl-002',
      model_name: 'Remission Propensity Model',
      // unmeasured performance sub-fields stay null (partial provenance)
      accuracy: null,
      error_rate: null,
      predictions_last_24h: null,
      status: 'degraded',
    },
  ],
  check_latency_ms: 45,
  data_provenance: 'partial',
};

describe('SystemHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    (useAlerts as MockFn).mockReturnValue({
      data: mockAlertsData,
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useMonitoringRuns as MockFn).mockReturnValue({
      data: mockRunsData,
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    // Default: health hooks return no data (undefined) so empty states render.
    (useFullHealthCheck as MockFn).mockReturnValue({ data: undefined, refetch: vi.fn().mockResolvedValue({}) });
    (usePipelineHealth as MockFn).mockReturnValue({ data: undefined });
    (useAgentHealth as MockFn).mockReturnValue({ data: undefined });
    (useHealthHistory as MockFn).mockReturnValue({ data: undefined });
    (useComponentHealth as MockFn).mockReturnValue({ data: undefined, refetch: vi.fn().mockResolvedValue({}) });
    (useModelHealth as MockFn).mockReturnValue({ data: undefined, refetch: vi.fn().mockResolvedValue({}) });
  });

  it('renders page header with title', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText('System Health')).toBeInTheDocument();
    expect(screen.getByText(/Comprehensive system monitoring with health scores/)).toBeInTheDocument();
  });

  it('renders empty state for services when no API data (F-002)', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText('Service Status')).toBeInTheDocument();
    // F-002: services no longer fabricated. Empty state surfaced instead.
    expect(
      screen.getByText(/No service status available/),
    ).toBeInTheDocument();
    // Verify the former fabricated service names are NOT in the DOM.
    expect(screen.queryByText('API Gateway')).not.toBeInTheDocument();
    expect(screen.queryByText('PostgreSQL')).not.toBeInTheDocument();
    expect(screen.queryByText('BentoML')).not.toBeInTheDocument();
  });

  it('renders empty state for model health when no API data (F-002)', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // "Model Health" appears in both overview card and section title
    const modelHealthTexts = screen.getAllByText('Model Health');
    expect(modelHealthTexts.length).toBeGreaterThanOrEqual(1);
    // F-002: models no longer fabricated.
    expect(screen.getByText(/No model health data/)).toBeInTheDocument();
    expect(screen.queryByText('Propensity Model')).not.toBeInTheDocument();
    expect(screen.queryByText('Churn Prediction')).not.toBeInTheDocument();
    expect(screen.queryByText('Conversion Model')).not.toBeInTheDocument();
  });

  // ===========================================================================
  // OVERALL HEALTH CARD: consolidated home for the composer's summary +
  // provenance flag (ported from the retired /ai-insights System Health Score
  // card — /system-health is now the single system-health surface).
  // ===========================================================================

  const fullHealthBase = {
    overall_health_score: 87.5,
    health_grade: 'B',
    critical_issues: [],
    warnings: [],
    recommendations: [],
    check_latency_ms: 10,
    timestamp: new Date().toISOString(),
  };

  it('renders the composer health_summary in the Overall Health card', () => {
    (useFullHealthCheck as MockFn).mockReturnValue({
      data: {
        ...fullHealthBase,
        health_summary: 'All systems nominal\n4 of 4 dimensions measured',
        data_provenance: 'measured',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText(/All systems nominal/)).toBeInTheDocument();
    // A fully measured check needs no provenance caveat.
    expect(screen.queryByText(/provenance:/)).not.toBeInTheDocument();
    // The backend check time renders (label only — the HH:MM rendering is
    // locale/timezone dependent, so the exact time is not asserted).
    expect(screen.getByText(/Health check:/)).toBeInTheDocument();
  });

  it('flags a partial (non-fully-measured) check with a provenance badge', () => {
    (useFullHealthCheck as MockFn).mockReturnValue({
      data: {
        ...fullHealthBase,
        health_summary: 'Component + model measured; pipeline/agent skipped',
        data_provenance: 'partial',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText('provenance: partial')).toBeInTheDocument();
  });

  it('does not surface a summary from placeholder-provenance data', () => {
    (useFullHealthCheck as MockFn).mockReturnValue({
      data: {
        ...fullHealthBase,
        health_summary: 'Dev placeholder summary',
        data_provenance: 'placeholder',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.queryByText(/Dev placeholder summary/)).not.toBeInTheDocument();
    expect(screen.getByText(/Awaiting health check/)).toBeInTheDocument();
    // Placeholder data must not surface a backend check time either.
    expect(screen.queryByText(/Health check:/)).not.toBeInTheDocument();
  });

  it('treats fail-closed "unknown" provenance as untrusted (codex PR-4 round 2)', () => {
    // The backend defaults data_provenance to "unknown" precisely so paths
    // that forget to tag it fail CLOSED. The page must honor that: unknown is
    // untrusted, so no score, summary, or check time may render from it.
    (useFullHealthCheck as MockFn).mockReturnValue({
      data: {
        ...fullHealthBase,
        health_summary: 'Composer forgot to tag provenance',
        data_provenance: 'unknown',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText(/Awaiting health check/)).toBeInTheDocument();
    expect(screen.queryByText(/Composer forgot/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Health check:/)).not.toBeInTheDocument();
  });

  it('suppresses untrusted issues/warnings/recommendations on the Alerts tab (codex PR-4 round 3)', async () => {
    // An untrusted payload's issue and recommendation STRINGS are just as
    // fabricated as its score — the backend's dev-offline mock emits
    // placeholder warnings/recommendations, so gating only the headline
    // number would still hand operators fake action items.
    const user = (await import('@testing-library/user-event')).default.setup();
    (useFullHealthCheck as MockFn).mockReturnValue({
      data: {
        ...fullHealthBase,
        critical_issues: ['Placeholder critical issue - restart the composer'],
        warnings: ['Placeholder warning - check adapter wiring'],
        recommendations: ['Placeholder recommendation - scale workers'],
        data_provenance: 'placeholder',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Alerts/i }));

    expect(screen.queryByText(/restart the composer/)).not.toBeInTheDocument();
    expect(screen.queryByText(/check adapter wiring/)).not.toBeInTheDocument();
    expect(screen.queryByText(/scale workers/)).not.toBeInTheDocument();
    expect(screen.queryByText('Critical Issues')).not.toBeInTheDocument();
    expect(screen.queryByText('Recommendations')).not.toBeInTheDocument();
  });

  it('renders trusted issues/warnings/recommendations on the Alerts tab', async () => {
    // Positive control for the trust gate: measured data must still surface —
    // the gate suppresses fabricated actions, not real ones.
    const user = (await import('@testing-library/user-event')).default.setup();
    (useFullHealthCheck as MockFn).mockReturnValue({
      data: {
        ...fullHealthBase,
        critical_issues: ['Redis connection pool exhausted'],
        warnings: ['Model staleness above threshold'],
        recommendations: ['Increase pool size to 50'],
        data_provenance: 'measured',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Alerts/i }));

    expect(screen.getByText(/Redis connection pool exhausted/)).toBeInTheDocument();
    expect(screen.getByText(/Model staleness above threshold/)).toBeInTheDocument();
    expect(screen.getByText(/Increase pool size to 50/)).toBeInTheDocument();
  });

  // ===========================================================================
  // AGENT / PIPELINE PROVENANCE GATE (codex PR-4 round 4): the /agents and
  // /pipelines wrappers also default provenance to "placeholder" fail-closed;
  // their raw arrays are sample data unless the backend tagged them trusted.
  // ===========================================================================

  const placeholderAgent = {
    agent_name: 'sample_orchestrator',
    tier: 0,
    available: true,
    avg_latency_ms: 120,
    success_rate: 0.99,
    invocations_24h: 42,
  };

  const placeholderPipeline = {
    pipeline_name: 'sample_etl_pipeline',
    last_run: new Date().toISOString(),
    last_success: new Date().toISOString(),
    rows_processed: 10000,
    freshness_hours: 0.5,
    status: 'healthy',
  };

  it('suppresses untrusted (placeholder) agent health on the Agents tab', async () => {
    const user = (await import('@testing-library/user-event')).default.setup();
    (useAgentHealth as MockFn).mockReturnValue({
      data: {
        agent_health_score: 1,
        total_agents: 1,
        available_count: 1,
        unavailable_count: 0,
        agents: [placeholderAgent],
        by_tier: { '0': 1 },
        check_latency_ms: 5,
        data_provenance: 'placeholder',
      },
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Agents/i }));

    expect(screen.queryByText('sample_orchestrator')).not.toBeInTheDocument();
    expect(screen.getByText(/No agent health data/)).toBeInTheDocument();
  });

  it('renders trusted (partial) agent health on the Agents tab', async () => {
    const user = (await import('@testing-library/user-event')).default.setup();
    (useAgentHealth as MockFn).mockReturnValue({
      data: {
        agent_health_score: 1,
        total_agents: 1,
        available_count: 1,
        unavailable_count: 0,
        agents: [placeholderAgent],
        by_tier: { '0': 1 },
        check_latency_ms: 5,
        data_provenance: 'partial',
      },
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Agents/i }));

    expect(screen.getByText('sample_orchestrator')).toBeInTheDocument();
    expect(screen.queryByText(/No agent health data/)).not.toBeInTheDocument();
  });

  it('suppresses untrusted (placeholder) pipeline health on the Pipelines tab', async () => {
    const user = (await import('@testing-library/user-event')).default.setup();
    (usePipelineHealth as MockFn).mockReturnValue({
      data: {
        pipeline_health_score: 1,
        total_pipelines: 1,
        healthy_count: 1,
        stale_count: 0,
        failed_count: 0,
        pipelines: [placeholderPipeline],
        check_latency_ms: 5,
        data_provenance: 'placeholder',
      },
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Pipelines/i }));

    expect(screen.queryByText('sample_etl_pipeline')).not.toBeInTheDocument();
    expect(screen.getByText(/No pipeline health data/)).toBeInTheDocument();
  });

  it('renders trusted (measured) pipeline health on the Pipelines tab', async () => {
    const user = (await import('@testing-library/user-event')).default.setup();
    (usePipelineHealth as MockFn).mockReturnValue({
      data: {
        pipeline_health_score: 1,
        total_pipelines: 1,
        healthy_count: 1,
        stale_count: 0,
        failed_count: 0,
        pipelines: [placeholderPipeline],
        check_latency_ms: 5,
        data_provenance: 'measured',
      },
    });
    render(<SystemHealth />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Pipelines/i }));

    expect(screen.getByText('sample_etl_pipeline')).toBeInTheDocument();
    expect(screen.queryByText(/No pipeline health data/)).not.toBeInTheDocument();
  });

  // ===========================================================================
  // WIRING TESTS (this PR): the Service Status / Model Health cards must render
  // REAL data from useComponentHealth / useModelHealth, and degrade to honest
  // empty states (never fabricated values) when data is absent or placeholder.
  // ===========================================================================

  it('renders REAL service status from useComponentHealth when measured', () => {
    (useComponentHealth as MockFn).mockReturnValue({
      data: measuredComponentHealth,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // Real component names from the /components endpoint.
    expect(screen.getByText('Database')).toBeInTheDocument();
    expect(screen.getByText('Cache (Redis)')).toBeInTheDocument();
    // Real latency surfaced.
    expect(screen.getByText('12ms')).toBeInTheDocument();
    expect(screen.getByText('240ms')).toBeInTheDocument();
    // Empty-state copy must be gone.
    expect(screen.queryByText(/No service status available/)).not.toBeInTheDocument();
  });

  it('renders REAL model health from useModelHealth when measured', () => {
    (useModelHealth as MockFn).mockReturnValue({
      data: measuredModelHealth,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // Real model names from the /models endpoint.
    expect(screen.getByText('CSU Initiation Model')).toBeInTheDocument();
    expect(screen.getByText('Remission Propensity Model')).toBeInTheDocument();
    // Empty-state copy must be gone.
    expect(screen.queryByText(/No model health data/)).not.toBeInTheDocument();
    // Anti-fabrication: the page must NOT invent a "drift" metric (the real
    // /models endpoint returns no drift score).
    expect(screen.queryByText(/Drift/i)).not.toBeInTheDocument();
  });

  it('shows honest "—" for unmeasured model performance sub-fields (no fabricated zeros)', () => {
    (useModelHealth as MockFn).mockReturnValue({
      data: measuredModelHealth,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // The 2nd model has null accuracy/error_rate/predictions: those must render
    // as "—", never as a fabricated 0 / 0% / 0.00.
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });

  it('does not fabricate a zero avg latency when no service latency is measured', () => {
    (useComponentHealth as MockFn).mockReturnValue({
      data: {
        ...measuredComponentHealth,
        component_health_score: 1.0,
        healthy_count: 1,
        degraded_count: 0,
        total_components: 1,
        // Single component with NO latency reported (latency_ms omitted).
        components: [
          {
            component_name: 'Message Queue',
            status: 'healthy',
            last_check: new Date().toISOString(),
          },
        ],
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // Overview avg latency must render "—", never "0ms".
    expect(screen.getByText(/Avg latency: —/)).toBeInTheDocument();
    expect(screen.queryByText(/Avg latency: 0ms/)).not.toBeInTheDocument();
  });

  it('treats placeholder-provenance component data as honest empty (no fake services)', () => {
    (useComponentHealth as MockFn).mockReturnValue({
      data: {
        ...measuredComponentHealth,
        data_provenance: 'placeholder',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // Placeholder dev data must NOT be presented as real measured services.
    expect(screen.queryByText('Database')).not.toBeInTheDocument();
    expect(screen.queryByText('Cache (Redis)')).not.toBeInTheDocument();
    expect(screen.getByText(/No service status available/)).toBeInTheDocument();
  });

  it('treats untrusted/absent provenance as honest empty (only measured|partial render)', () => {
    // 'unknown' provenance and a response with NO provenance field must both be
    // treated as no-data — only 'measured'/'partial' are surfaced as real.
    (useComponentHealth as MockFn).mockReturnValue({
      data: { ...measuredComponentHealth, data_provenance: 'unknown' },
      refetch: vi.fn().mockResolvedValue({}),
    });
    // Model response with the data_provenance field omitted entirely.
    const { data_provenance: _omitted, ...modelNoProv } = measuredModelHealth;
    void _omitted;
    (useModelHealth as MockFn).mockReturnValue({
      data: modelNoProv,
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.queryByText('Database')).not.toBeInTheDocument();
    expect(screen.queryByText('CSU Initiation Model')).not.toBeInTheDocument();
    expect(screen.getByText(/No service status available/)).toBeInTheDocument();
    expect(screen.getByText(/No model health data/)).toBeInTheDocument();
  });

  it('treats placeholder-provenance model data as honest empty (no fake models)', () => {
    (useModelHealth as MockFn).mockReturnValue({
      data: {
        ...measuredModelHealth,
        data_provenance: 'placeholder',
      },
      refetch: vi.fn().mockResolvedValue({}),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.queryByText('CSU Initiation Model')).not.toBeInTheDocument();
    expect(screen.queryByText('Remission Propensity Model')).not.toBeInTheDocument();
    expect(screen.getByText(/No model health data/)).toBeInTheDocument();
  });

  it('degrades to honest empty when a hook crashes / returns null data', () => {
    // Simulate a hook whose query errored: data is undefined. The page must
    // render the empty state, never a fabricated default service/model.
    (useComponentHealth as MockFn).mockReturnValue({ data: null, refetch: vi.fn() });
    (useModelHealth as MockFn).mockReturnValue({ data: null, refetch: vi.fn() });

    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText(/No service status available/)).toBeInTheDocument();
    expect(screen.getByText(/No model health data/)).toBeInTheDocument();
  });

  it('displays overview stat cards with neutral defaults (F-002)', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // Services card - shows 0/0 when no API data
    expect(screen.getByText('Services')).toBeInTheDocument();
    expect(screen.getByText('0/0')).toBeInTheDocument();

    // Active Alerts card - appears in both overview and alerts tab
    const activeAlertsTexts = screen.getAllByText('Active Alerts');
    expect(activeAlertsTexts.length).toBeGreaterThanOrEqual(1);

    // Agents card - appears in both overview and agents tab
    const agentsTexts = screen.getAllByText('Agents');
    expect(agentsTexts.length).toBeGreaterThanOrEqual(1);
  });

  it('displays active alerts section', async () => {
    const user = (await import('@testing-library/user-event')).default.setup();
    render(<SystemHealth />, { wrapper: createWrapper() });

    // "Active Alerts" appears in overview card
    const activeAlertsTexts = screen.getAllByText('Active Alerts');
    expect(activeAlertsTexts.length).toBeGreaterThanOrEqual(1);

    // Switch to Alerts tab to see the description
    const alertsTab = screen.getByRole('tab', { name: /Alerts/i });
    await user.click(alertsTab);

    await waitFor(() => {
      expect(screen.getByText(/Recent alerts requiring attention/)).toBeInTheDocument();
    });
  });

  it('shows refresh button and last updated time', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
  });

  it('triggers refresh when refresh button clicked', async () => {
    const mockRefetchAlerts = vi.fn().mockResolvedValue({});
    const mockRefetchRuns = vi.fn().mockResolvedValue({});

    (useAlerts as MockFn).mockReturnValue({
      data: mockAlertsData,
      isLoading: false,
      refetch: mockRefetchAlerts,
    });

    (useMonitoringRuns as MockFn).mockReturnValue({
      data: mockRunsData,
      isLoading: false,
      refetch: mockRefetchRuns,
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    const refreshButton = screen.getByRole('button', { name: /Refresh/i });
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(mockRefetchAlerts).toHaveBeenCalled();
      expect(mockRefetchRuns).toHaveBeenCalled();
    });
  });

  it('shows loading state while fetching data', () => {
    (useAlerts as MockFn).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: vi.fn(),
    });

    (useMonitoringRuns as MockFn).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: vi.fn(),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // Page should still render with sample data
    expect(screen.getByText('System Health')).toBeInTheDocument();
    expect(screen.getByText('Service Status')).toBeInTheDocument();
  });

  // F-002: removed assertions on fabricated SAMPLE_MODELS performance
  // trends, drift scores, and SAMPLE_SERVICES latency values. The page
  // now renders empty states for these sections when no API data is
  // available, so these tests no longer have a meaningful target.

  it('displays empty alerts message when no active alerts', () => {
    (useAlerts as MockFn).mockReturnValue({
      data: { alerts: [], active_count: 0, total_count: 0 },
      isLoading: false,
      refetch: vi.fn(),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    // "Active Alerts" appears in both overview card and section title
    const activeAlertsTexts = screen.getAllByText('Active Alerts');
    expect(activeAlertsTexts.length).toBeGreaterThanOrEqual(1);
    // With 0 alerts, should show "0 critical" in model health section and "All clear" text
    expect(screen.getByText('All clear')).toBeInTheDocument();
  });

  it('does not fabricate alerts when the API returns zero active alerts (M3)', async () => {
    const user = (await import('@testing-library/user-event')).default.setup();
    (useAlerts as MockFn).mockReturnValue({
      data: { alerts: [], active_count: 0, total_count: 0 },
      isLoading: false,
      refetch: vi.fn(),
    });
    (useMonitoringRuns as MockFn).mockReturnValue({
      data: mockRunsData,
      isLoading: false,
      refetch: vi.fn(),
    });

    render(<SystemHealth />, { wrapper: createWrapper() });

    const alertsTab = screen.getByRole('tab', { name: /Alerts/i });
    await user.click(alertsTab);

    // The former fabricated fallback alert titles must NOT appear.
    expect(screen.queryByText('Model Retraining Scheduled')).not.toBeInTheDocument();
    expect(screen.queryByText('Data Drift Detected')).not.toBeInTheDocument();
    // The honest empty message is shown instead.
    expect(screen.getByText(/No active alerts - all systems operational/i)).toBeInTheDocument();
  });
});
