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

import { useAlerts, useMonitoringRuns } from '@/hooks/api/use-monitoring';

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

describe('SystemHealth', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockAlertsData,
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });

    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockRunsData,
      isLoading: false,
      refetch: vi.fn().mockResolvedValue({}),
    });
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

    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockAlertsData,
      isLoading: false,
      refetch: mockRefetchAlerts,
    });

    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
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
    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: vi.fn(),
    });

    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
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
    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
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
    (useAlerts as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { alerts: [], active_count: 0, total_count: 0 },
      isLoading: false,
      refetch: vi.fn(),
    });
    (useMonitoringRuns as ReturnType<typeof vi.fn>).mockReturnValue({
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
