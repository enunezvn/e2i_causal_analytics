/**
 * TimeSeries Page Tests
 * =====================
 *
 * Tests that the Time Series page is wired to live monitoring/KPI hooks
 * (issue #302) and renders zero `sample*` constants.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as fs from 'node:fs';
import * as path from 'node:path';
import TimeSeries from './TimeSeries';

// Mock the live hooks the page should call.
vi.mock('@/hooks/api/use-monitoring', () => ({
  usePerformanceTrend: vi.fn(),
  useDriftHistory: vi.fn(),
}));

vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIValue: vi.fn(),
  useKPIMetadata: vi.fn(),
  useKPIList: vi.fn(),
}));

import { usePerformanceTrend, useDriftHistory } from '@/hooks/api/use-monitoring';
import { useKPIValue, useKPIMetadata, useKPIList } from '@/hooks/api/use-kpi';

const mockUsePerformanceTrend = usePerformanceTrend as unknown as ReturnType<typeof vi.fn>;
const mockUseDriftHistory = useDriftHistory as unknown as ReturnType<typeof vi.fn>;
const mockUseKPIValue = useKPIValue as unknown as ReturnType<typeof vi.fn>;
const mockUseKPIMetadata = useKPIMetadata as unknown as ReturnType<typeof vi.fn>;
const mockUseKPIList = useKPIList as unknown as ReturnType<typeof vi.fn>;

const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const samplePerformanceTrend = {
  model_id: 'propensity_v2.1.0',
  metric_name: 'accuracy',
  current_value: 0.91,
  baseline_value: 0.88,
  change_percent: 3.4,
  trend: 'improving' as const,
  is_significant: true,
  alert_threshold_breached: false,
  history: [
    { metric_name: 'accuracy', metric_value: 0.85, recorded_at: '2024-01-01T00:00:00Z' },
    { metric_name: 'accuracy', metric_value: 0.87, recorded_at: '2024-01-02T00:00:00Z' },
    { metric_name: 'accuracy', metric_value: 0.91, recorded_at: '2024-01-03T00:00:00Z' },
  ],
};

const sampleKPIMetadata = {
  id: 'WS1-DQ-001',
  name: 'Source Coverage',
  definition: 'Coverage metric',
  formula: 'covered/total',
  calculation_type: 'direct',
  workstream: 'ws1_data_quality',
  tables: [],
  columns: [],
  frequency: 'daily',
  primary_causal_library: 'none',
};

const sampleKPIValue = {
  kpi_id: 'WS1-DQ-001',
  value: 0.87,
  status: 'good' as const,
  calculated_at: '2024-01-03T00:00:00Z',
  cached: false,
  metadata: {
    history: [
      { recorded_at: '2024-01-01T00:00:00Z', value: 0.80 },
      { recorded_at: '2024-01-02T00:00:00Z', value: 0.84 },
      { recorded_at: '2024-01-03T00:00:00Z', value: 0.87 },
    ],
  },
};

beforeEach(() => {
  vi.clearAllMocks();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;

  mockUsePerformanceTrend.mockReturnValue({
    data: samplePerformanceTrend,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
  mockUseDriftHistory.mockReturnValue({
    data: { model_id: 'propensity_v2.1.0', total_records: 0, records: [] },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
  mockUseKPIValue.mockReturnValue({
    data: sampleKPIValue,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
  mockUseKPIMetadata.mockReturnValue({
    data: sampleKPIMetadata,
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
  mockUseKPIList.mockReturnValue({
    data: { kpis: [sampleKPIMetadata], total: 1 },
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn(),
    isRefetching: false,
  });
});

describe('TimeSeries (live data wiring — issue #302)', () => {
  it('renders the page header and the two-mode toggle', () => {
    render(<TimeSeries />, { wrapper: createWrapper() });

    expect(screen.getByText('Time Series Analysis')).toBeInTheDocument();
    // Mode toggle: Model performance | KPI history
    expect(screen.getByRole('tab', { name: /Model performance/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /KPI history/i })).toBeInTheDocument();
  });

  it('calls usePerformanceTrend in Model performance mode (default)', () => {
    render(<TimeSeries />, { wrapper: createWrapper() });

    expect(mockUsePerformanceTrend).toHaveBeenCalled();
    const lastCall = mockUsePerformanceTrend.mock.calls[
      mockUsePerformanceTrend.mock.calls.length - 1
    ];
    const params = lastCall[0];
    expect(params).toMatchObject({
      model_id: expect.any(String),
      metric_name: expect.any(String),
      days: expect.any(Number),
    });
  });

  it('feeds time-range filter into hook params (not just UI state)', async () => {
    const user = userEvent.setup();
    render(<TimeSeries />, { wrapper: createWrapper() });

    const defaultDaysParam = mockUsePerformanceTrend.mock.calls.at(-1)?.[0]?.days;
    expect(defaultDaysParam).toBe(90);

    const timeRangeTrigger = screen.getByRole('combobox', { name: /time range/i });
    await user.click(timeRangeTrigger);
    const option = await screen.findByRole('option', { name: /30 Days/i });
    await user.click(option);

    await waitFor(() => {
      const lastDaysParam = mockUsePerformanceTrend.mock.calls.at(-1)?.[0]?.days;
      expect(lastDaysParam).toBe(30);
    });
  });

  it('switches to KPI history mode and calls useKPIValue', async () => {
    const user = userEvent.setup();
    render(<TimeSeries />, { wrapper: createWrapper() });

    const kpiTab = screen.getByRole('tab', { name: /KPI history/i });
    await user.click(kpiTab);

    await waitFor(() => {
      expect(mockUseKPIValue).toHaveBeenCalled();
    });
    const lastCall = mockUseKPIValue.mock.calls.at(-1);
    expect(lastCall?.[0]).toEqual(expect.any(String));
  });

  it('renders a recharts container fed by live hook history data', () => {
    const { container } = render(<TimeSeries />, { wrapper: createWrapper() });
    expect(container.querySelector('.recharts-responsive-container')).toBeInTheDocument();
  });

  it('shows QueryErrorState when performance trend errors', () => {
    const networkErr = new Error('Network unreachable');
    mockUsePerformanceTrend.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: networkErr,
      refetch: vi.fn(),
      isRefetching: false,
    });

    render(<TimeSeries />, { wrapper: createWrapper() });

    // QueryErrorState renders an Alert with role=alert
    const alerts = screen.getAllByRole('alert');
    expect(alerts.length).toBeGreaterThan(0);
  });

  it('shows loading indicator while performance trend is loading', () => {
    mockUsePerformanceTrend.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
      refetch: vi.fn(),
      isRefetching: false,
    });

    render(<TimeSeries />, { wrapper: createWrapper() });

    // Use accessible-name based assertion — generic loading indicator
    expect(screen.getByTestId('timeseries-loading')).toBeInTheDocument();
  });

  it('source file contains NO `sample*` constants (sample data fully removed)', () => {
    const sourcePath = path.resolve(__dirname, 'TimeSeries.tsx');
    const source = fs.readFileSync(sourcePath, 'utf-8');

    // Strip block + line comments to avoid false positives in docstrings.
    const stripped = source
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/.*$/gm, '');

    // Forbid identifiers of the form sample*, SAMPLE_*, generate*Data
    // (the 38 constants enumerated in the original file).
    const forbiddenPatterns = [
      /\bSAMPLE_[A-Z_]+\b/,
      /\bsample[A-Z]\w*\b/,
      /\bgenerateTimeSeriesData\b/,
      /\bgenerateForecastData\b/,
      /\bgenerateSeasonalityData\b/,
    ];

    for (const pattern of forbiddenPatterns) {
      const match = stripped.match(pattern);
      if (match) {
        throw new Error(
          `TimeSeries.tsx still contains forbidden mock-data identifier: ${match[0]} (pattern ${pattern})`,
        );
      }
    }
  });
});
