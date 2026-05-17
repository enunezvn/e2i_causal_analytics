/**
 * DataQuality Page Tests
 * ======================
 *
 * Tests for the Data Quality monitoring dashboard page.
 * Verifies live wiring to KPI workstream `ws1_data_quality` + drift detection backends
 * (issue #301) and preserves the tab/refresh DOM contract the Playwright spec expects
 * (issue #306: dataProfilingTab, qualityIssuesTab, validationRulesTab, refreshButton).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { KPIListResponse, KPIMetadata } from '@/types/kpi';
import type { DriftDetectionResponse, DriftHistoryResponse } from '@/types/monitoring';
import DataQuality from './DataQuality';

// =============================================================================
// HOOK MOCKS
// =============================================================================

vi.mock('@/hooks/api/use-kpi', () => ({
  useKPIList: vi.fn(),
  useKPIDetail: vi.fn(),
  useKPIMetadata: vi.fn(),
  useKPIValue: vi.fn(),
}));

vi.mock('@/hooks/api/use-monitoring', () => ({
  useLatestDriftStatus: vi.fn(),
  useDriftHistory: vi.fn(),
  useTriggerDriftDetection: vi.fn(),
}));

// Toast mock for #324 (mutation success/error feedback). Captures every call
// so assertions can check that triggerDrift's onSuccess/onError emit toasts.
const toastMock = vi.fn();
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: toastMock, toasts: [], dismiss: vi.fn() }),
  toast: (...args: unknown[]) => toastMock(...args),
}));

// -----------------------------------------------------------------------------
// VISUALIZATION STUBS — record KPICard props so we can assert no fabricated
// sparkline fallback leaks (HIGH-1 from adversarial review of PR #320; mirrors
// the same anti-pattern PR #313 captured at memory
// `feedback_mock_data_scans_must_check_imported_defaults`).
// -----------------------------------------------------------------------------
const kpiCardCalls: Array<Record<string, unknown>> = [];

vi.mock('@/components/visualizations', async () => {
  const actual = await vi.importActual<typeof import('@/components/visualizations')>(
    '@/components/visualizations'
  );
  return {
    ...actual,
    KPICard: (props: Record<string, unknown>) => {
      kpiCardCalls.push(props);
      return <div data-testid="kpi-card-stub">{String(props.title)}</div>;
    },
  };
});

import { useKPIList, useKPIDetail } from '@/hooks/api/use-kpi';
import {
  useLatestDriftStatus,
  useDriftHistory,
  useTriggerDriftDetection,
} from '@/hooks/api/use-monitoring';

// =============================================================================
// FIXTURES
// =============================================================================

const dqKpis: KPIMetadata[] = [
  {
    id: 'WS1-DQ-001',
    name: 'Source Coverage - Patients',
    definition: 'Percentage of eligible patients present in source vs reference universe',
    formula: 'covered_patients / reference_patients',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['patient_journeys'],
    columns: ['patient_id'],
    threshold: { target: 85, warning: 70, critical: 50 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
  },
  {
    id: 'WS1-DQ-002',
    name: 'Completeness - HCP Master',
    definition: 'HCP master record completeness',
    formula: 'non_null_hcp / total_hcp',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['hcp_master'],
    columns: ['npi'],
    threshold: { target: 98, warning: 90, critical: 80 },
    unit: '%',
    frequency: 'daily',
    primary_causal_library: 'none',
  },
];

const kpiListResponse: KPIListResponse = {
  kpis: dqKpis,
  total: dqKpis.length,
  workstream: 'ws1_data_quality',
};

const driftResponse: DriftDetectionResponse = {
  task_id: 'task-123',
  model_id: 'data_quality_pipeline',
  status: 'completed',
  overall_drift_score: 0.12,
  features_checked: 8,
  features_with_drift: ['hcp_id', 'npi'],
  results: [],
  drift_summary: '2 features show drift',
  recommended_actions: ['Investigate hcp_id'],
  detection_latency_ms: 250,
  timestamp: '2026-01-02T08:30:00Z',
};

const driftHistoryResponse: DriftHistoryResponse = {
  model_id: 'data_quality_pipeline',
  total_records: 1,
  records: [
    {
      id: 'dh-1',
      model_version: 'v1.0',
      feature_name: 'hcp_id',
      drift_type: 'data_drift',
      drift_score: 0.18,
      severity: 'medium',
      detected_at: '2026-01-01T00:00:00Z',
      baseline_start: '2025-12-25T00:00:00Z',
      baseline_end: '2025-12-31T00:00:00Z',
      current_start: '2026-01-01T00:00:00Z',
      current_end: '2026-01-02T00:00:00Z',
    },
  ],
};

// =============================================================================
// SETUP
// =============================================================================

const mockCreateObjectURL = vi.fn(() => 'blob:mock-url');
const mockRevokeObjectURL = vi.fn();

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

const mockMutate = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  kpiCardCalls.length = 0;
  toastMock.mockClear();
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;

  // Default: KPI list returns wired DQ KPIs from ws1_data_quality
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: kpiListResponse,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  });

  (useKPIDetail as ReturnType<typeof vi.fn>).mockReturnValue({
    metadata: dqKpis[0],
    value: {
      kpi_id: 'WS1-DQ-001',
      value: 94.5,
      status: 'good',
      calculated_at: '2026-01-02T08:30:00Z',
      cached: false,
      metadata: {},
    },
    isLoading: false,
    error: null,
    isMetadataLoading: false,
    isValueLoading: false,
    refetch: vi.fn(),
  });

  (useLatestDriftStatus as ReturnType<typeof vi.fn>).mockReturnValue({
    data: driftResponse,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  });

  (useDriftHistory as ReturnType<typeof vi.fn>).mockReturnValue({
    data: driftHistoryResponse,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  });

  (useTriggerDriftDetection as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate: mockMutate,
    isPending: false,
    error: null,
  });
});

// =============================================================================
// TESTS
// =============================================================================

describe('DataQuality (live wiring + Playwright contract)', () => {
  // ===========================================================================
  // Page chrome that the Playwright spec asserts (issue #306)
  // ===========================================================================

  it('renders page header (Playwright: pageHeader)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // h1 must include "Data Quality"
    expect(
      screen.getByRole('heading', { level: 1, name: /Data Quality/i })
    ).toBeInTheDocument();
  });

  it('renders page description matching Playwright pageDescription regex', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // Spec (data-quality.page.ts): getByText(/profiling|completeness|accuracy|validation/i).first()
    // The Playwright spec uses `.first()` because the regex matches multiple substrings
    // (KPI card titles, descriptions, etc). We mirror that by asserting >= 1 match.
    const matches = screen.getAllByText(/profiling|completeness|accuracy|validation/i);
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders Validation Rules tab (Playwright: validationRulesTab)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      screen.getByRole('tab', { name: /Validation Rules/i })
    ).toBeInTheDocument();
  });

  it('renders Data Profiling tab (Playwright: dataProfilingTab)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      screen.getByRole('tab', { name: /Data Profiling/i })
    ).toBeInTheDocument();
  });

  it('renders Quality Issues tab (Playwright: qualityIssuesTab)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      screen.getByRole('tab', { name: /Quality Issues/i })
    ).toBeInTheDocument();
  });

  it('renders Refresh button (Playwright: refreshButton)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      screen.getByRole('button', { name: /refresh/i })
    ).toBeInTheDocument();
  });

  // ===========================================================================
  // Dimension cards / KPI cards (Playwright: dimensionCards)
  // ===========================================================================

  it('renders the four quality dimension cards', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(screen.getByText('Completeness')).toBeInTheDocument();
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Consistency')).toBeInTheDocument();
    expect(screen.getByText('Timeliness')).toBeInTheDocument();
  });

  it('renders Overall Quality card', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(screen.getByText('Overall Quality')).toBeInTheDocument();
  });

  // ===========================================================================
  // ISSUE #301 — Live KPI wiring (replaces 17 mock column blocks)
  // ===========================================================================

  it('calls useKPIList with workstream=ws1_data_quality (#301 AC)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(useKPIList).toHaveBeenCalled();
    // First call's first arg must include workstream filter
    const calls = (useKPIList as ReturnType<typeof vi.fn>).mock.calls;
    const allParams = calls.map((c) => c[0]).filter(Boolean);
    expect(
      allParams.some(
        (p) => p && (p.workstream === 'ws1_data_quality' || p.workstream === 'WS1_DATA_QUALITY')
      )
    ).toBe(true);
  });

  it('renders KPI rows from useKPIList (NOT hard-coded HCP IDs / NPI mock blocks)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // Live KPIs from useKPIList must render by name. The default tab (Validation
    // Rules) renders a row per KPI from kpiList.kpis, each pulling metadata via
    // useKPIDetail. Names from the kpiList fixture must appear in the DOM.
    expect(screen.getAllByText('Source Coverage - Patients').length).toBeGreaterThanOrEqual(1);
    // Definition text also proves we're rendering from the live KPI metadata
    expect(
      screen.getAllByText(/Percentage of eligible patients present in source/i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('does NOT render the deleted hard-coded mock data (no SAMPLE_* identifiers in DOM)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // None of the old mock-only validation rule names should appear
    expect(screen.queryByText('HCP ID Not Null')).not.toBeInTheDocument();
    expect(screen.queryByText('Valid NPI Format')).not.toBeInTheDocument();
    expect(screen.queryByText('Sales Amount Range')).not.toBeInTheDocument();
    // None of the old fabricated HCP rows (with sample 125.4K, 2.5M etc.)
    expect(screen.queryByText('125.4K rows')).not.toBeInTheDocument();
    expect(screen.queryByText('2.5M rows')).not.toBeInTheDocument();
  });

  // ===========================================================================
  // ISSUE #301 — Drift wiring
  // ===========================================================================

  it('wires drift section to useLatestDriftStatus and useDriftHistory (#301 AC)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(useLatestDriftStatus).toHaveBeenCalled();
    expect(useDriftHistory).toHaveBeenCalled();
  });

  it('surfaces drift summary from the live drift response', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // drift_summary or features_with_drift content visible at least once
    expect(
      screen.getAllByText(/2 features show drift|features with drift/i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('Refresh button triggers useTriggerDriftDetection mutation', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    const refreshBtn = screen.getByRole('button', { name: /refresh/i });
    fireEvent.click(refreshBtn);

    expect(mockMutate).toHaveBeenCalled();
  });

  // ===========================================================================
  // ISSUE #301 — Loading / error states via QueryErrorState
  // ===========================================================================

  it('renders error state when KPI list fails (#301 AC: QueryErrorState)', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('KPI service unavailable'),
      refetch: vi.fn(),
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // QueryErrorState renders the error message or a generic friendly title
    expect(
      screen.getByText(/Something went wrong|KPI service unavailable|unable to/i)
    ).toBeInTheDocument();
  });

  it('renders loading state when KPI list is loading (#301 AC)', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // Loading indicator (skeleton / spinner / "Loading" text) — possibly multiple sections
    expect(
      screen.getAllByText(/Loading/i).length
    ).toBeGreaterThanOrEqual(1);
  });

  // ===========================================================================
  // Preserve prior behavior - export still works
  // ===========================================================================

  it('handles export button click', () => {
    const mockClick = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'a') {
        const link = originalCreateElement('a');
        link.click = mockClick;
        return link;
      }
      return originalCreateElement(tag);
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    const exportButton = screen.getByRole('button', { name: /Export/i });
    fireEvent.click(exportButton);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();

    vi.restoreAllMocks();
  });

  // ===========================================================================
  // ADVERSARIAL REVIEW HIGH-1 — KPICard SAMPLE_SPARKLINE fallback leak
  // (mirrors A5/PR #313's `feedback_mock_data_scans_must_check_imported_defaults`).
  // Each dimension <KPICard> must pass an explicit sparklineData prop and it
  // must NOT equal the SAMPLE_SPARKLINE fabricated constant from KPICard.tsx:70
  // `[45, 52, 48, 55, 60, 58, 62, 65, 63, 68]`.
  // ===========================================================================

  it('passes explicit sparklineData to every dimension KPICard (no SAMPLE_SPARKLINE leak)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // 5 dimension cards: Overall Quality, Completeness, Accuracy, Consistency, Timeliness
    expect(kpiCardCalls.length).toBeGreaterThanOrEqual(5);

    const SAMPLE_SPARKLINE = [45, 52, 48, 55, 60, 58, 62, 65, 63, 68];

    for (const props of kpiCardCalls) {
      // The prop MUST be defined (so KPICard's `??` fallback to SAMPLE_SPARKLINE
      // never triggers).
      expect(
        props.sparklineData,
        `KPICard "${String(props.title)}" must pass sparklineData; got undefined (would fall back to SAMPLE_SPARKLINE)`
      ).toBeDefined();

      // And the value must not be the fabricated SAMPLE_SPARKLINE constant.
      expect(
        props.sparklineData,
        `KPICard "${String(props.title)}" must not pass the fabricated SAMPLE_SPARKLINE array`
      ).not.toEqual(SAMPLE_SPARKLINE);
    }
  });
});

// =============================================================================
// PR #322-326,328 — adversarial-review fixes (additive describe block; left
// existing blocks for #327 / Agent C). See issues #322 #323 #324 #325 #326 #328.
// =============================================================================

describe('PR #322-326,328 — adversarial-review fixes', () => {
  it('#322 wires status filter to rule.status field', async () => {
    // Override useKPIDetail so the two KPIs produce DIFFERENT computed statuses:
    //   WS1-DQ-001 -> value=94.5 vs threshold {target:85, warning:70, critical:50} = 'pass'
    //   WS1-DQ-002 -> value=85   vs threshold {target:98, warning:90, critical:80} = 'warning'
    (useKPIDetail as ReturnType<typeof vi.fn>).mockReset();
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      if (kpiId === 'WS1-DQ-002') {
        return {
          metadata: dqKpis[1],
          value: {
            kpi_id: 'WS1-DQ-002',
            value: 85,
            status: 'warning',
            calculated_at: '2026-01-02T08:30:00Z',
            cached: false,
            metadata: {},
          },
          isLoading: false,
          error: null,
          isMetadataLoading: false,
          isValueLoading: false,
          refetch: vi.fn(),
        };
      }
      return {
        metadata: dqKpis[0],
        value: {
          kpi_id: 'WS1-DQ-001',
          value: 94.5,
          status: 'good',
          calculated_at: '2026-01-02T08:30:00Z',
          cached: false,
          metadata: {},
        },
        isLoading: false,
        error: null,
        isMetadataLoading: false,
        isValueLoading: false,
        refetch: vi.fn(),
      };
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // Both rows visible under 'all'
    expect(screen.getAllByText('Source Coverage - Patients').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Completeness - HCP Master').length).toBeGreaterThanOrEqual(1);

    // Open the status filter and pick "Warning"
    const statusTrigger = screen.getByRole('combobox');
    fireEvent.click(statusTrigger);
    const warningOpt = screen.getByRole('option', { name: /^Warning$/ });
    fireEvent.click(warningOpt);

    // Only the WS1-DQ-002 'warning' row remains in the Validation Rules table
    expect(screen.queryByText('Source Coverage - Patients')).not.toBeInTheDocument();
    expect(screen.getAllByText('Completeness - HCP Master').length).toBeGreaterThanOrEqual(1);
  });

  it('#323 surfaces driftHistoryError on default tab (rules)', () => {
    (useDriftHistory as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('drift history 503'),
      refetch: vi.fn(),
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // Default tab is 'rules'. The driftHistoryError banner must render at the
    // page level (NOT only inside the Quality Issues tab), so it's visible from
    // the default tab. There may also be an in-tab banner; getAllByText covers
    // both. The page-level banner must specifically use the "30-day" title to
    // distinguish from the latest-status `driftError` banner above.
    const matches = screen.getAllByText(/Could not load 30-day drift history/i);
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('#324 toasts on triggerDrift success and error', () => {
    let capturedOptions:
      | { onSuccess?: (data: unknown) => void; onError?: (err: unknown) => void }
      | undefined;
    (useTriggerDriftDetection as ReturnType<typeof vi.fn>).mockImplementation((opts) => {
      capturedOptions = opts;
      return { mutate: mockMutate, isPending: false, error: null };
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // Production code must pass onSuccess + onError callbacks
    expect(capturedOptions).toBeDefined();
    expect(typeof capturedOptions?.onSuccess).toBe('function');
    expect(typeof capturedOptions?.onError).toBe('function');

    // Invoke onSuccess -> a toast fires
    capturedOptions!.onSuccess!({ task_id: 'task-xyz' });
    expect(toastMock).toHaveBeenCalled();
    const successCalls = toastMock.mock.calls;
    const successCall = successCalls[successCalls.length - 1]?.[0] as
      | Record<string, unknown>
      | undefined;
    expect(JSON.stringify(successCall).toLowerCase()).toMatch(/drift|trigger|success|task-xyz/);

    toastMock.mockClear();

    // Invoke onError -> a (destructive) toast fires
    capturedOptions!.onError!({ message: 'queue full' });
    expect(toastMock).toHaveBeenCalled();
    const errorCalls = toastMock.mock.calls;
    const errorCall = errorCalls[errorCalls.length - 1]?.[0] as
      | Record<string, unknown>
      | undefined;
    expect(JSON.stringify(errorCall).toLowerCase()).toMatch(/fail|error|queue full/);
  });

  it('#325 disables Export button while data is loading', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    const exportBtn = screen.getByRole('button', { name: /Export/i });
    expect(exportBtn).toBeDisabled();
  });

  it('#326 triggerDrift sends time_window=30d', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    const refreshBtn = screen.getByRole('button', { name: /refresh/i });
    fireEvent.click(refreshBtn);

    expect(mockMutate).toHaveBeenCalled();
    const mutateCalls = mockMutate.mock.calls;
    const callArg = mutateCalls[mutateCalls.length - 1]?.[0] as
      | { request?: { time_window?: string } }
      | undefined;
    expect(callArg?.request?.time_window).toBe('30d');
  });

  it('#328 has aria-label / label on search input and status select', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // Search input must be reachable by accessible name "Search validation rules"
    // (either via <Label htmlFor> association or aria-label on the input).
    const searchInput = screen.getByRole('textbox', {
      name: /search.*(rules|validation)/i,
    });
    expect(searchInput).toBeInTheDocument();

    // Status select trigger must carry an aria-label
    const statusTrigger = screen.getByRole('combobox', { name: /filter.*status|status/i });
    expect(statusTrigger).toBeInTheDocument();
  });
});
