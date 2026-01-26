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

  it('displays service status section with 5 services', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    expect(screen.getByText('Service Status')).toBeInTheDocument();
    expect(screen.getByText('API Gateway')).toBeInTheDocument();
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument();
    expect(screen.getByText('Redis Cache')).toBeInTheDocument();
    expect(screen.getByText('FalkorDB')).toBeInTheDocument();
    expect(screen.getByText('BentoML')).toBeInTheDocument();
  });

  it('displays model health section with 3 models', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // "Model Health" appears in both overview card and section title
    const modelHealthTexts = screen.getAllByText('Model Health');
    expect(modelHealthTexts.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Propensity Model')).toBeInTheDocument();
    expect(screen.getByText('Churn Prediction')).toBeInTheDocument();
    expect(screen.getByText('Conversion Model')).toBeInTheDocument();
  });

  it('displays overview stat cards', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // Services card - shows X/Y format
    expect(screen.getByText('Services')).toBeInTheDocument();
    expect(screen.getByText('5/5')).toBeInTheDocument();

    // Models card - shows X / Y format with sample data (2 healthy out of 3)
    expect(screen.getByText('Models')).toBeInTheDocument();
    expect(screen.getByText('2 / 3')).toBeInTheDocument();

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

  it('displays model performance trends', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // Check for trend indicators
    expect(screen.getByText('stable')).toBeInTheDocument();
    expect(screen.getByText('degrading')).toBeInTheDocument();
    expect(screen.getByText('improving')).toBeInTheDocument();
  });

  it('shows drift scores for models', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // Drift label should be present for each model
    const driftLabels = screen.getAllByText('Drift');
    expect(driftLabels.length).toBe(3);
  });

  it('displays model health scores via ProgressRing', () => {
    const { container } = render(<SystemHealth />, { wrapper: createWrapper() });

    // Should have ProgressRing components with SVG circles
    const progressRings = container.querySelectorAll('svg circle');
    expect(progressRings.length).toBeGreaterThan(0);
  });

  it('shows infrastructure latency for services', () => {
    render(<SystemHealth />, { wrapper: createWrapper() });

    // Check for latency values (in ms format)
    expect(screen.getByText('45ms')).toBeInTheDocument();
    expect(screen.getByText('12ms')).toBeInTheDocument();
    expect(screen.getByText('3ms')).toBeInTheDocument();
  });

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
});
