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
  useKPIHistory: vi.fn(),
  useKPIMetadata: vi.fn(),
  useKPIList: vi.fn(),
}));

import { usePerformanceTrend, useDriftHistory } from '@/hooks/api/use-monitoring';
import { useKPIValue, useKPIHistory, useKPIMetadata, useKPIList } from '@/hooks/api/use-kpi';

const mockUsePerformanceTrend = usePerformanceTrend as unknown as ReturnType<typeof vi.fn>;
const mockUseDriftHistory = useDriftHistory as unknown as ReturnType<typeof vi.fn>;
const mockUseKPIValue = useKPIValue as unknown as ReturnType<typeof vi.fn>;
const mockUseKPIHistory = useKPIHistory as unknown as ReturnType<typeof vi.fn>;
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
  // Default: a real monthly KPI history series from the /history endpoint.
  mockUseKPIHistory.mockReturnValue({
    data: {
      kpi_id: 'WS3-BI-010',
      brand: '',
      region: '',
      count: 3,
      points: [
        { metric_date: '2026-04-01', value: 1.83, status: 'warning' },
        { metric_date: '2026-05-01', value: 1.84, status: 'warning' },
        { metric_date: '2026-06-01', value: 1.85, status: 'warning' },
      ],
    },
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

    const defaultCalls = mockUsePerformanceTrend.mock.calls;
    const defaultDaysParam = defaultCalls[defaultCalls.length - 1]?.[0]?.days;
    expect(defaultDaysParam).toBe(1825);

    const timeRangeTrigger = screen.getByRole('combobox', { name: /time range/i });
    await user.click(timeRangeTrigger);
    const option = await screen.findByRole('option', { name: /30 Days/i });
    await user.click(option);

    await waitFor(() => {
      const allCalls = mockUsePerformanceTrend.mock.calls;
      const lastDaysParam = allCalls[allCalls.length - 1]?.[0]?.days;
      expect(lastDaysParam).toBe(30);
    });
  });

  it('renders KPI history mode content after switching tabs (mock-boundary)', async () => {
    const user = userEvent.setup();
    render(<TimeSeries />, { wrapper: createWrapper() });

    // Pre-click: KPI panel content not visible. The "Current KPI Status"
    // card only renders inside the KPI tab panel.
    expect(screen.queryByText('Current KPI Status')).not.toBeInTheDocument();
    // Performance panel is visible by default.
    expect(screen.getByText('Performance Trend')).toBeInTheDocument();

    const kpiTab = screen.getByRole('tab', { name: /KPI history/i });
    await user.click(kpiTab);

    // Post-click: the KPI status panel becomes visible.
    await waitFor(() => {
      expect(screen.getByText('Current KPI Status')).toBeInTheDocument();
    });

    // And useKPIValue was called with a string KPI ID.
    expect(mockUseKPIValue).toHaveBeenCalled();
    const kpiCalls = mockUseKPIValue.mock.calls;
    const lastCall = kpiCalls[kpiCalls.length - 1];
    expect(typeof lastCall?.[0]).toBe('string');
    expect((lastCall?.[0] as string).length).toBeGreaterThan(0);
  });

  it('time-range filter applies in KPI history mode too (filters embedded history)', async () => {
    const user = userEvent.setup();
    // Build history that spans 2000 days so the 1825d default window (5 Years)
    // still trims the earliest ~175 points, proving the filter is applied.
    const today = new Date();
    const longHistory = Array.from({ length: 2000 }, (_, i) => {
      const d = new Date(today.getTime() - (2000 - i) * 24 * 60 * 60 * 1000);
      return { metric_date: d.toISOString().slice(0, 10), value: 0.5 + i * 0.001, status: 'warning' };
    });
    mockUseKPIHistory.mockReturnValue({
      data: { kpi_id: 'WS3-BI-010', brand: '', region: '', count: longHistory.length, points: longHistory },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
      isRefetching: false,
    });

    const { container } = render(<TimeSeries />, { wrapper: createWrapper() });

    // Switch to KPI mode.
    await user.click(screen.getByRole('tab', { name: /KPI history/i }));
    await waitFor(() => {
      expect(screen.getByText('Current KPI Status')).toBeInTheDocument();
    });

    // The recharts container is present. The "Data Points" KPI card reflects
    // the filtered series length; default range is 1825d (5 Years) so we expect a
    // count strictly less than the full 2000-point history.
    const recharts = container.querySelector('.recharts-responsive-container');
    expect(recharts).toBeInTheDocument();

    // Find the Data Points card by its title's parent card.
    const dataPointsLabel = screen.getByText('Data Points');
    const card = dataPointsLabel.closest('[class*="rounded"]');
    expect(card).not.toBeNull();
    const txt = card!.textContent ?? '';
    // Extract digits; must be a positive integer less than 2000.
    const m = txt.match(/(\d{1,4})/);
    expect(m).not.toBeNull();
    const count = Number(m![1]);
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThan(2000);
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

  it('KPICard usages never trigger the SAMPLE_SPARKLINE fallback (codex MED iter-2)', () => {
    // KPICard.tsx falls back to a `SAMPLE_SPARKLINE = [45, 52, 48, ...]` when
    // `sparklineData` is undefined. Every KPICard usage in TimeSeries.tsx
    // MUST pass `sparklineData=` explicitly (real series or [] to opt out).
    const sourcePath = path.resolve(__dirname, 'TimeSeries.tsx');
    const source = fs.readFileSync(sourcePath, 'utf-8');

    // Find every `<KPICard ... />` invocation and check it includes
    // `sparklineData=` as a prop.
    const kpiCardRegex = /<KPICard\b([^>]*?)\/>/gs;
    const matches = [...source.matchAll(kpiCardRegex)];
    expect(matches.length).toBeGreaterThan(0);

    const offenders: string[] = [];
    for (const match of matches) {
      const propsBlob = match[1];
      if (!/\bsparklineData\s*=/.test(propsBlob)) {
        offenders.push(match[0].slice(0, 120));
      }
    }

    if (offenders.length > 0) {
      throw new Error(
        `KPICard usages missing explicit \`sparklineData=\` (would trigger SAMPLE_SPARKLINE fallback):\n` +
          offenders.join('\n'),
      );
    }
  });

  it('default selection yields persistence_remibrutinib_goldstd_lr_v1', () => {
    render(<TimeSeries />, { wrapper: createWrapper() });

    // After initial render the hook should have been called with the default per-brand model id.
    expect(mockUsePerformanceTrend).toHaveBeenCalled();
    const lastCall = mockUsePerformanceTrend.mock.calls[
      mockUsePerformanceTrend.mock.calls.length - 1
    ];
    expect(lastCall[0].model_id).toBe('persistence_remibrutinib_goldstd_lr_v1');
  });

  it('cohort+brand dropdowns update the queried model id (initiation + Kisqali)', async () => {
    const user = userEvent.setup();
    render(<TimeSeries />, { wrapper: createWrapper() });

    // Change cohort to initiation
    const cohortSelect = screen.getByRole('combobox', { name: /cohort/i });
    await user.selectOptions(cohortSelect, 'initiation');

    // Change brand to Kisqali
    const brandSelect = screen.getByRole('combobox', { name: /brand/i });
    await user.selectOptions(brandSelect, 'Kisqali');

    await waitFor(() => {
      const allCalls = mockUsePerformanceTrend.mock.calls;
      const lastModelId = allCalls[allCalls.length - 1]?.[0]?.model_id;
      expect(lastModelId).toBe('initiation_kisqali_goldstd_lr_v1');
    });
  });

  it('HCP Adoption cohort option resolves to hcp_adoption_{brand}_goldstd_lr_v1', async () => {
    const user = userEvent.setup();
    render(<TimeSeries />, { wrapper: createWrapper() });

    // Select the HCP Adoption cohort
    const cohortSelect = screen.getByRole('combobox', { name: /cohort/i });
    await user.selectOptions(cohortSelect, 'hcp_adoption');

    // Default brand is Remibrutinib; the model handle should reflect the template
    await waitFor(() => {
      const allCalls = mockUsePerformanceTrend.mock.calls;
      const lastModelId = allCalls[allCalls.length - 1]?.[0]?.model_id;
      expect(lastModelId).toBe('hcp_adoption_remibrutinib_goldstd_lr_v1');
    });

    // Also verify with a different brand (Fabhalta) to prove the template wiring
    const brandSelect = screen.getByRole('combobox', { name: /brand/i });
    await user.selectOptions(brandSelect, 'Fabhalta');

    await waitFor(() => {
      const allCalls = mockUsePerformanceTrend.mock.calls;
      const lastModelId = allCalls[allCalls.length - 1]?.[0]?.model_id;
      expect(lastModelId).toBe('hcp_adoption_fabhalta_goldstd_lr_v1');
    });
  });

  it('removes the free-text Model ID advanced override', () => {
    const { container } = render(<TimeSeries />, { wrapper: createWrapper() });
    // The advanced-override input is gone; cohort+brand selects fully drive the model id.
    expect(screen.queryByText(/advanced override/i)).not.toBeInTheDocument();
    expect(container.querySelector('#ts-model-id')).toBeNull();
    // The selection card is still present at the top with cohort + brand.
    expect(screen.getByRole('combobox', { name: /cohort/i })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /brand/i })).toBeInTheDocument();
  });

  it('Model Selection appears above the Performance Trend chart (top placement)', () => {
    render(<TimeSeries />, { wrapper: createWrapper() });
    const selection = screen.getByText('Model Selection');
    const chart = screen.getByText('Performance Trend');
    // DOCUMENT_POSITION_FOLLOWING (4) => selection comes before chart in DOM order.
    expect(selection.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('KPI history shows an honest empty-state when a KPI has no time series', async () => {
    const user = userEvent.setup();
    // A point-in-time KPI: a current value, but the /history endpoint returns
    // no points (it isn't backfillable) — never a fabricated flat series.
    mockUseKPIValue.mockReturnValue({
      data: {
        kpi_id: 'WS1-DQ-001',
        value: 0.0576,
        status: 'critical' as const,
        calculated_at: '2026-06-15T00:00:00Z',
        cached: false,
        metadata: { include_synthetic: true },
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
      isRefetching: false,
    });
    mockUseKPIHistory.mockReturnValue({
      data: { kpi_id: 'WS1-DQ-001', brand: '', region: '', count: 0, points: [] },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
      isRefetching: false,
    });

    render(<TimeSeries />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /KPI history/i }));

    await waitFor(() => {
      expect(screen.getByTestId('kpi-history-empty')).toBeInTheDocument();
    });
    expect(screen.getByText(/No historical data available for this KPI/i)).toBeInTheDocument();
    // The current KPI value is still shown.
    expect(screen.getByText('Current KPI Status')).toBeInTheDocument();
  });

  it('source file contains NO sample/mock data — by identifier AND by behavior', () => {
    const sourcePath = path.resolve(__dirname, 'TimeSeries.tsx');
    const source = fs.readFileSync(sourcePath, 'utf-8');

    // Strip block + line comments to avoid false positives in docstrings.
    const stripped = source
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/.*$/gm, '');

    // (a) Identifier-level — forbid the 38 enumerated constants + common
    //     bypass renames (MOCK_*, DEMO_*, FAKE_*, FIXTURE_*).
    const forbiddenIdentifiers = [
      /\bSAMPLE_[A-Z_]+\b/,
      /\bsample[A-Z]\w*\b/,
      /\bMOCK_[A-Z_]+\b/,
      /\bDEMO_[A-Z_]+\b/,
      /\bFAKE_[A-Z_]+\b/,
      /\bFIXTURE_[A-Z_]+\b/,
      /\bgenerateTimeSeriesData\b/,
      /\bgenerateForecastData\b/,
      /\bgenerateSeasonalityData\b/,
    ];
    for (const pattern of forbiddenIdentifiers) {
      const match = stripped.match(pattern);
      if (match) {
        throw new Error(
          `TimeSeries.tsx still contains forbidden mock-data identifier: ${match[0]} (pattern ${pattern})`,
        );
      }
    }

    // (b) Behavior-level — forbid large static array literals of
    //     `{date: ..., value: ...}` shape OR `{date: ..., trend: ...}` shape,
    //     which are how the original sample arrays were materialised. Caps
    //     are conservative; live test fixtures live in the .test.tsx file,
    //     not in the page source.
    const bigDateValueArray =
      /\[\s*(\{\s*[a-zA-Z_]+\s*:\s*['"`][\d-]+['"`]\s*,\s*value\s*:[^}]+\}\s*,\s*){5,}/;
    if (bigDateValueArray.test(stripped)) {
      throw new Error(
        'TimeSeries.tsx contains a large static `[{date, value}, ...]` array literal — looks like inlined sample data.',
      );
    }

    // (c) Behavior-level — forbid for-loops that synthesise time-series
    //     data inside the page (the original used `for (let i = 0; i < 90; i++)`
    //     plus `new Date(startDate)` to build mock arrays).
    const syntheticDateLoop =
      /for\s*\(\s*let\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*\d{2,}\s*;\s*\w+\+\+\s*\)\s*\{[\s\S]{0,400}new\s+Date\s*\(/;
    if (syntheticDateLoop.test(stripped)) {
      throw new Error(
        'TimeSeries.tsx contains a synthetic Date-loop — looks like client-side mock-data generation.',
      );
    }
  });
});
