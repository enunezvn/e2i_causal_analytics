/**
 * Monitoring Page Tests
 * =====================
 *
 * Tests for the Monitoring page — verifies it pulls live data
 * via the monitoring hooks rather than rendering hard-coded mock arrays.
 *
 * Reference: GitHub issue #297 — "Wire Monitoring page to live backend".
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Monitoring from './Monitoring';
import { AlertStatus } from '@/types/monitoring';

// Mock the monitoring hooks BEFORE importing them so the page renders against
// these stubs instead of hitting the real API.
vi.mock('@/hooks/api/use-monitoring', () => ({
  useAlerts: vi.fn(),
  useMonitoringRuns: vi.fn(),
  useModelHealth: vi.fn(),
}));

// The model selector is driven from the registry-backed /api/models/status
// endpoint via useModelsStatus — mock it so the page resolves a deterministic
// model list (no more hardcoded handles).
vi.mock('@/hooks/api/use-predictions', () => ({
  useModelsStatus: vi.fn(),
}));

import {
  useAlerts,
  useMonitoringRuns,
  useModelHealth,
} from '@/hooks/api/use-monitoring';
import { useModelsStatus } from '@/hooks/api/use-predictions';

// First registered production model the page should default its selection to.
const PRIMARY_MODEL = 'csu_treatment_initiation_lr_balanced_v1';

// QueryClient wrapper for tests (required because the page uses TanStack
// Query under the hood through these hooks).
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0 },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// Mock URL.createObjectURL/revokeObjectURL so export-button tests do not blow up
// when the test environment lacks Blob URL support.
const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();
global.URL.createObjectURL = mockCreateObjectURL;
global.URL.revokeObjectURL = mockRevokeObjectURL;

// Distinctive fixture values so we can differentiate from the previous
// hard-coded SAMPLE_* arrays that used to live in Monitoring.tsx.
//
// Timestamps are RELATIVE to test wall-clock (computed at module-load time)
// so the suite stays stable as the calendar advances — Monitoring.tsx
// filters runs against `Date.now() - windowMs` (codex iter-5 LOW).
const ONE_HOUR_MS = 60 * 60 * 1000;
const ONE_DAY_MS = 24 * ONE_HOUR_MS;
const NOW = Date.now();
const HOUR_AGO_ISO = new Date(NOW - ONE_HOUR_MS).toISOString();
const TWO_HOURS_AGO_ISO = new Date(NOW - 2 * ONE_HOUR_MS).toISOString();
const ONE_WEEK_AGO_ISO = new Date(NOW - 7 * ONE_DAY_MS).toISOString();

const mockAlertsData = {
  total_count: 7,
  active_count: 3,
  alerts: [
    {
      id: 'alert-distinct-001',
      model_version: 'churn_v1.5.2',
      alert_type: 'data_drift',
      severity: 'high',
      title: 'LIVE_API_ALERT_TITLE_DISTINCT',
      description: 'Live-API description that should appear in the DOM.',
      status: AlertStatus.ACTIVE,
      triggered_at: HOUR_AGO_ISO,
    },
  ],
};

const mockRunsData = {
  model_id: 'propensity_v2.1.0',
  total_runs: 42,
  runs: [
    {
      id: 'run-001',
      model_version: 'propensity_v2.1.0',
      run_type: 'scheduled',
      started_at: TWO_HOURS_AGO_ISO,
      completed_at: HOUR_AGO_ISO,
      features_checked: 47,
      drift_detected_count: 2,
      alerts_generated: 1,
      duration_ms: 150_000,
    },
  ],
};

const mockHealthData = {
  model_id: 'propensity_v2.1.0',
  overall_health: 'warning' as const,
  last_check: HOUR_AGO_ISO,
  drift_score: 0.42,
  active_alerts: 3,
  last_retrained: ONE_WEEK_AGO_ISO,
  performance_trend: 'degrading' as const,
  recommendations: ['Increase training frequency'],
};

// Recharts ResponsiveContainer + jsdom can stretch render time well past
// the default 5000 ms timeout when many cards/charts are in the tree. Bump
// per-suite to keep this stable in CI.
describe('Monitoring page — live-backend wiring (issue #297)', { timeout: 20_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks();

    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockAlertsData,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockRunsData,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useModelHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockHealthData,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useModelsStatus as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        total_models: 2,
        healthy_count: 2,
        unhealthy_count: 0,
        models: [
          {
            model_name: PRIMARY_MODEL,
            status: 'healthy',
            endpoint: 'http://localhost:3000',
            last_check: HOUR_AGO_ISO,
          },
          {
            model_name: 'csu_treatment_initiation_lr_full_v1',
            status: 'healthy',
            endpoint: 'http://localhost:3000',
            last_check: HOUR_AGO_ISO,
          },
        ],
        timestamp: HOUR_AGO_ISO,
      },
      isLoading: false,
      isError: false,
      error: null,
    });
  });

  it('calls useAlerts, useMonitoringRuns, and useModelHealth (live-data wiring)', () => {
    render(<Monitoring />, { wrapper: createWrapper() });

    expect(useAlerts).toHaveBeenCalled();
    expect(useMonitoringRuns).toHaveBeenCalled();
    expect(useModelHealth).toHaveBeenCalled();
  });

  it('defaults selection to the first REGISTERED model and passes it + days into the hooks', async () => {
    render(<Monitoring />, { wrapper: createWrapper() });

    // The page resolves its model list from useModelsStatus and defaults the
    // selection (via effect) to the first registered production model — NOT a
    // hardcoded fictional handle. The "24h" time range maps to days=1.
    await waitFor(() => {
      const healthCalls = (useModelHealth as ReturnType<typeof vi.fn>).mock.calls;
      const lastHealth = healthCalls[healthCalls.length - 1]?.[0];
      expect(lastHealth).toBe(PRIMARY_MODEL);
    });

    const alertsCalls = (useAlerts as ReturnType<typeof vi.fn>).mock.calls;
    const lastAlerts = alertsCalls[alertsCalls.length - 1]?.[0];
    expect(lastAlerts).toMatchObject({
      model_id: PRIMARY_MODEL,
      status: AlertStatus.ACTIVE,
    });

    const runsCalls = (useMonitoringRuns as ReturnType<typeof vi.fn>).mock.calls;
    const lastRuns = runsCalls[runsCalls.length - 1]?.[0];
    expect(lastRuns).toMatchObject({
      model_id: PRIMARY_MODEL,
      days: 1,
    });
  });

  it('"Total Runs" KPI reflects displayed runs.length (not unfiltered total_runs)', () => {
    // The mock fixture has total_runs=42 (server-side total) but only 1
    // run in the runs[] array. The "Total Runs" KPI MUST reflect the
    // narrowed/displayed count, otherwise it disagrees with the chart and
    // table (codex iter-2 MED finding).
    render(<Monitoring />, { wrapper: createWrapper() });

    // 42 from the fixture must NOT appear as a Total Runs KPI value.
    expect(screen.queryByText('42')).not.toBeInTheDocument();

    // Scope the value assertion to the Total Runs KPI card so we don't
    // accidentally match `1` from `alerts_generated` etc. (codex iter-4 LOW).
    // The KPI renders the label "Total Runs" inside the card, so the
    // value `1` must be present in the same KPI card.
    const totalRunsLabel = screen.getByText('Total Runs');
    const card = totalRunsLabel.closest('[class*="rounded"]') as HTMLElement | null;
    expect(card).not.toBeNull();
    // Within that scoped card, the displayed value must be 1.
    expect(card!.textContent ?? '').toMatch(/(?:^|\D)1(?:\D|$)/);
  });

  it('renders alert content from the live hook (not hard-coded SAMPLE_ERROR_LOGS)', async () => {
    const user = userEvent.setup();
    render(<Monitoring />, { wrapper: createWrapper() });

    // Switch to the Errors tab so live-API alert content is visible.
    await user.click(screen.getByRole('tab', { name: /Errors/i }));

    await waitFor(() => {
      expect(screen.getByText(/LIVE_API_ALERT_TITLE_DISTINCT/i)).toBeInTheDocument();
    });

    // And ensure the prior hard-coded error messages no longer appear.
    expect(
      screen.queryByText(/Causal discovery timeout: operation exceeded 30s limit/i)
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Database connection pool exhausted/i)
    ).not.toBeInTheDocument();
  });

  it('renders model selector for selecting model_id', () => {
    render(<Monitoring />, { wrapper: createWrapper() });

    // At minimum, multiple combobox elements should exist
    // (time range + model selector).
    const comboboxes = screen.getAllByRole('combobox');
    expect(comboboxes.length).toBeGreaterThanOrEqual(2);
  });

  it('shows loading state when hooks are still fetching', () => {
    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    (useModelHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    const { container } = render(<Monitoring />, { wrapper: createWrapper() });

    // Loading state must show some loading affordance
    // (spinner via animate-pulse / animate-spin / aria-busy / role=status / "Loading" text).
    const loadingMarkers = container.querySelectorAll(
      '[aria-busy="true"], [role="status"], .animate-pulse, .animate-spin'
    );
    const hasLoadingText = !!screen.queryByText(/loading/i);
    expect(loadingMarkers.length + (hasLoadingText ? 1 : 0)).toBeGreaterThan(0);
  });

  it('shows error state when a monitoring hook errors', () => {
    const apiErr = new Error('Network unreachable');
    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: apiErr,
      refetch: vi.fn(),
    });
    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });
    (useModelHealth as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    });

    render(<Monitoring />, { wrapper: createWrapper() });

    // QueryErrorState renders the message text from the Error.
    expect(screen.getByText(/Network unreachable/i)).toBeInTheDocument();
  });

  it('does NOT render the hard-coded SAMPLE_ERROR_LOGS messages anywhere', () => {
    render(<Monitoring />, { wrapper: createWrapper() });

    // The 6 SAMPLE_ERROR_LOGS messages from the old mock data
    const hardcodedMessages = [
      /Causal discovery timeout: operation exceeded 30s limit/i,
      /Database connection pool exhausted/i,
      /Rate limit exceeded for user usr-006/i,
      /Model inference failed: insufficient memory/i,
      /Slow query detected: 2.5s response time/i,
      /Authentication token expired/i,
    ];

    for (const re of hardcodedMessages) {
      expect(screen.queryByText(re)).not.toBeInTheDocument();
    }
  });

  it('does NOT render the hard-coded SAMPLE_ENDPOINT_STATS endpoint paths', () => {
    render(<Monitoring />, { wrapper: createWrapper() });

    // Endpoints from SAMPLE_ENDPOINT_STATS that were inline literals.
    expect(screen.queryByText('/api/v1/causal/discover')).not.toBeInTheDocument();
    expect(screen.queryByText('/api/v1/agents/orchestrate')).not.toBeInTheDocument();
  });
});
