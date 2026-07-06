/**
 * DataQuality Page Tests
 * ======================
 *
 * Tests for the Data Quality monitoring dashboard page.
 * Verifies live wiring to KPI workstream `ws1_data_quality` (issue #301) and
 * preserves the tab/refresh DOM contract the Playwright spec expects (issue
 * #306: dataProfilingTab, qualityIssuesTab, validationRulesTab, refreshButton).
 *
 * Dimension cards derive from MEASURED WS1-DQ KPI values (portfolio scope) —
 * the old drift-based derivation read a `data_quality_pipeline` model id that
 * no sweep monitors, so the cards could structurally never leave "No data".
 * Model & data drift are consolidated on /monitoring (the page links there).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import type { KPIListResponse, KPIMetadata } from '@/types/kpi';
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

/**
 * Dimension-source KPI fixtures (portfolio values the dimension cards derive
 * from). Chosen to make each derivation's arithmetic assertable:
 *   completeness = 0.94 × 100                       = 94    (status warning)
 *   accuracy     = 0.80 × 100                       = 80    (status good/healthy —
 *                  beats its 0.75 target; the OLD generic ≥85 client cut would
 *                  have mispainted this healthy value as critical)
 *   consistency  = (1 − 0.105) × 100                = 89.5  (status critical —
 *                  DQ-006 is lower-is-better; backend status is authoritative)
 *   timeliness   = mean(min(100, 3/1.25×100), min(100, 24/21×100)) = 100
 *   overall      = (94 + 80 + 89.5 + 100) / 4       = 90.875 (status critical =
 *                  worst measured dimension)
 */
const dimensionFixtures: Record<
  string,
  { value: number; status: string; target: number; name: string }
> = {
  'WS1-DQ-003': { value: 0.8, status: 'good', target: 0.75, name: 'Cross-source Match Rate' },
  'WS1-DQ-005': { value: 0.94, status: 'warning', target: 0.95, name: 'Completeness Pass Rate' },
  'WS1-DQ-006': { value: 0.105, status: 'critical', target: 0.05, name: 'Geographic Consistency' },
  'WS1-DQ-007': { value: 1.25, status: 'good', target: 3, name: 'Data Lag (Median)' },
  'WS1-DQ-009': { value: 21, status: 'good', target: 24, name: 'Time-to-Release (TTR)' },
};

function dimensionMeta(kpiId: string): KPIMetadata {
  const fx = dimensionFixtures[kpiId];
  return {
    id: kpiId,
    name: fx?.name ?? kpiId,
    definition: `${fx?.name ?? kpiId} definition`,
    formula: 'x / y',
    calculation_type: 'direct',
    workstream: 'ws1_data_quality',
    tables: ['patient_journeys'],
    columns: ['patient_id'],
    threshold: fx ? { target: fx.target } : undefined,
    unit: undefined,
    value_format: 'percent',
    frequency: 'daily',
    primary_causal_library: 'none',
  } as KPIMetadata;
}

/** Default per-id useKPIDetail implementation: dimension ids resolve to the
 * dimension fixtures; list-row ids resolve to their dqKpis metadata with a
 * healthy value. */
function defaultKPIDetailImpl(kpiId: string) {
  const fx = dimensionFixtures[kpiId];
  if (fx) {
    return {
      metadata: dimensionMeta(kpiId),
      value: {
        kpi_id: kpiId,
        value: fx.value,
        status: fx.status,
        calculated_at: '2026-01-02T08:30:00Z',
        cached: false,
        metadata: {},
      },
      isLoading: false,
      isFetching: false,
      error: null,
      metadataError: null,
      valueError: null,
      isMetadataLoading: false,
      isValueLoading: false,
      refetch: vi.fn(),
    };
  }
  const meta = dqKpis.find((k) => k.id === kpiId) ?? dqKpis[0];
  return {
    metadata: meta,
    value: {
      kpi_id: meta.id,
      value: 94.5,
      status: 'good',
      calculated_at: '2026-01-02T08:30:00Z',
      cached: false,
      metadata: {},
    },
    isLoading: false,
    isFetching: false,
    error: null,
    metadataError: null,
    valueError: null,
    isMetadataLoading: false,
    isValueLoading: false,
    refetch: vi.fn(),
  };
}

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
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

const mockRefetchKpis = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  kpiCardCalls.length = 0;
  global.URL.createObjectURL = mockCreateObjectURL;
  global.URL.revokeObjectURL = mockRevokeObjectURL;

  // Default: KPI list returns wired DQ KPIs from ws1_data_quality
  (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
    data: kpiListResponse,
    isLoading: false,
    error: null,
    refetch: mockRefetchKpis,
    isRefetching: false,
  });

  (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation(defaultKPIDetailImpl);
});

// =============================================================================
// TESTS
// =============================================================================

describe('DataQuality (live wiring + Playwright contract)', () => {
  // ===========================================================================
  // Page chrome that the Playwright spec asserts (issue #306)
  // ===========================================================================

  it('renders page header and the /monitoring consolidation link (drift lives on Monitoring)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // h1 must include "Data Quality"
    expect(
      screen.getByRole('heading', { level: 1, name: /Data Quality/i })
    ).toBeInTheDocument();

    // Vacuity gate: the page must NOT wire a per-page drift section for the
    // unmonitored `data_quality_pipeline` id; instead it links to /monitoring
    // where model & data drift actually live.
    expect(screen.queryByText(/data_quality_pipeline/)).not.toBeInTheDocument();
    const monitoringLinks = screen.getAllByRole('link', { name: /monitoring/i });
    expect(monitoringLinks.length).toBeGreaterThanOrEqual(1);
    expect(monitoringLinks[0]).toHaveAttribute('href', '/monitoring');
  });

  it('renders page description + live workstream id (Playwright regex + #301 wiring)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    // Spec (data-quality.page.ts): getByText(/profiling|completeness|accuracy|validation/i).first()
    const matches = screen.getAllByText(/profiling|completeness|accuracy|validation/i);
    expect(matches.length).toBeGreaterThanOrEqual(1);

    // Vacuity gate (#327): live KPI wiring surfaces the workstream name as a
    // <code>ws1_data_quality</code> token inside the Validation Rules
    // CardDescription.
    expect(screen.getAllByText('ws1_data_quality').length).toBeGreaterThanOrEqual(1);
  });

  it('renders Validation Rules tab with live KPI rows (Playwright + #301 wiring)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      screen.getByRole('tab', { name: /Validation Rules/i })
    ).toBeInTheDocument();

    // Vacuity gate (#327): default tab is "rules" and the body renders a row
    // per `kpiList.kpis` (per-id metadata via the default useKPIDetail impl).
    expect(screen.getByText('Source Coverage - Patients')).toBeInTheDocument();
    expect(screen.getByText('Completeness - HCP Master')).toBeInTheDocument();
  });

  it('renders Data Profiling tab with live KPI table data (Playwright + #301 wiring)', async () => {
    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });

    const profilingTab = screen.getByRole('tab', { name: /Data Profiling/i });
    expect(profilingTab).toBeInTheDocument();

    await user.click(profilingTab);

    await waitFor(() => {
      expect(screen.getByText('WS1-DQ-002')).toBeInTheDocument();
    });
    expect(screen.getByText('hcp_master')).toBeInTheDocument();
  });

  it('renders Quality Issues tab with KPI threshold breaches (drift consolidation)', async () => {
    // WS1-DQ-002 breaches critical; WS1-DQ-001 is healthy. The issues tab must
    // list ONLY the breaching KPI, sourced from the backend-authoritative
    // status — no drift records involved.
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        return {
          ...base,
          value: { ...base.value, value: 55, status: 'critical' },
        };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });

    const issuesTab = screen.getByRole('tab', { name: /Quality Issues/i });
    await user.click(issuesTab);

    // The breaching KPI renders with a critical badge; the healthy one does not.
    await waitFor(() => {
      expect(screen.getByText('Completeness - HCP Master')).toBeInTheDocument();
    });
    expect(screen.getByText('critical')).toBeInTheDocument();
    expect(screen.queryByText('Source Coverage - Patients')).not.toBeInTheDocument();
  });

  it('Quality Issues tab renders the no-issues empty state with the /monitoring link', async () => {
    // Default impl: both list KPIs are 'good' → zero issues.
    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });

    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    expect(
      await screen.findByText(/No data-quality KPIs are breaching their thresholds/i)
    ).toBeInTheDocument();
  });

  it('Refresh button reflects the KPI list isRefetching state', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: kpiListResponse,
      isLoading: false,
      error: null,
      refetch: mockRefetchKpis,
      isRefetching: true,
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    const refreshBtn = screen.getByRole('button', { name: /refresh/i });
    expect(refreshBtn).toBeInTheDocument();
    expect(refreshBtn).toBeDisabled();
  });

  it('Refresh button refetches the KPI list (no drift POST — drift lives on /monitoring)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    const refreshBtn = screen.getByRole('button', { name: /refresh/i });
    fireEvent.click(refreshBtn);

    expect(mockRefetchKpis).toHaveBeenCalled();
  });

  // ===========================================================================
  // Dimension cards / KPI cards (Playwright: dimensionCards) — derived from
  // MEASURED WS1-DQ KPI values, statuses from the backend (direction-aware).
  // ===========================================================================

  it('derives the four dimension cards from measured KPI values (not drift)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(screen.getByText('Completeness')).toBeInTheDocument();
    expect(screen.getByText('Accuracy')).toBeInTheDocument();
    expect(screen.getByText('Consistency')).toBeInTheDocument();
    expect(screen.getByText('Timeliness')).toBeInTheDocument();

    const byTitle = (t: string) => kpiCardCalls.find((c) => c.title === t);

    // Values — see dimensionFixtures docstring for the arithmetic.
    expect(byTitle('Completeness')!.value).toBeCloseTo(94, 1);
    expect(byTitle('Accuracy')!.value).toBeCloseTo(80, 1);
    expect(byTitle('Consistency')!.value).toBeCloseTo(89.5, 1);
    expect(byTitle('Timeliness')!.value).toBeCloseTo(100, 1);

    // Statuses come from the backend's direction-aware KPI statuses — NOT a
    // generic ≥95/≥85 client-side cut. An 80% match rate beating its 75%
    // target is healthy; an 89.5% consistency complement of a critical
    // geographic gap is critical.
    expect(byTitle('Accuracy')!.status).toBe('healthy');
    expect(byTitle('Completeness')!.status).toBe('warning');
    expect(byTitle('Consistency')!.status).toBe('critical');
    expect(byTitle('Timeliness')!.status).toBe('healthy');
  });

  it('renders Overall Quality as the mean of measured dimensions, status = worst dimension', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(screen.getByText('Overall Quality')).toBeInTheDocument();

    const overallCall = kpiCardCalls.find((c) => c.title === 'Overall Quality');
    expect(overallCall).toBeDefined();
    // (94 + 80 + 89.5 + 100) / 4 = 90.875 → rounded to 90.9 for display
    expect(overallCall!.value).toBeCloseTo(90.9, 1);
    // Worst measured dimension is critical (Consistency) → the composite must
    // not read healthy while a component dimension is critical.
    expect(overallCall!.status).toBe('critical');
  });

  // ===========================================================================
  // ISSUE #301 — Live KPI wiring (replaces 17 mock column blocks)
  // ===========================================================================

  it('calls useKPIList with workstream=ws1_data_quality (#301 AC)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(useKPIList).toHaveBeenCalled();
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

    expect(screen.getAllByText('Source Coverage - Patients').length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText(/Percentage of eligible patients present in source/i).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('does NOT render the deleted hard-coded mock data (no SAMPLE_* identifiers in DOM)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    expect(screen.queryByText('HCP ID Not Null')).not.toBeInTheDocument();
    expect(screen.queryByText('Valid NPI Format')).not.toBeInTheDocument();
    expect(screen.queryByText('Sales Amount Range')).not.toBeInTheDocument();
    expect(screen.queryByText('125.4K rows')).not.toBeInTheDocument();
    expect(screen.queryByText('2.5M rows')).not.toBeInTheDocument();
  });

  // ===========================================================================
  // ISSUE #301 — Loading / error states via QueryErrorState
  // ===========================================================================

  it('renders error state when KPI list fails (#301 AC: QueryErrorState)', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('KPI service unavailable'),
      refetch: mockRefetchKpis,
      isRefetching: false,
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      screen.getByText(/Something went wrong|KPI service unavailable|unable to/i)
    ).toBeInTheDocument();
  });

  it('renders loading state when KPI list is loading (#301 AC)', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: mockRefetchKpis,
      isRefetching: false,
    });

    render(<DataQuality />, { wrapper: createWrapper() });

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
      expect(
        props.sparklineData,
        `KPICard "${String(props.title)}" must pass sparklineData; got undefined (would fall back to SAMPLE_SPARKLINE)`
      ).toBeDefined();

      expect(
        props.sparklineData,
        `KPICard "${String(props.title)}" must not pass the fabricated SAMPLE_SPARKLINE array`
      ).not.toEqual(SAMPLE_SPARKLINE);
    }
  });
});

// =============================================================================
// PR #322-328 — adversarial-review fixes that remain in the drift-free design.
// (#323/#324/#326 were drift-section behaviors; the drift section moved to
// /monitoring, so those tests moved out with it.)
// =============================================================================

describe('PR #322-328 — adversarial-review fixes', () => {
  it('#322 shows empty-state when status filter hides every row (codex MED-1)', async () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    const statusTrigger = screen.getByRole('combobox', { name: /filter.*status|status/i });
    fireEvent.click(statusTrigger);
    const failOpt = screen.getByRole('option', { name: /^Fail$/ });
    fireEvent.click(failOpt);

    // Empty state appears (no rows match 'fail')
    expect(
      await screen.findByText(/No data quality KPIs match your filters/i)
    ).toBeInTheDocument();
  });

  it('#322 wires status filter to rule.status field', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        return { ...base, value: { ...base.value, value: 85, status: 'warning' } };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // Both rows visible under 'all'
    expect(screen.getAllByText('Source Coverage - Patients').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Completeness - HCP Master').length).toBeGreaterThanOrEqual(1);

    // Open the status filter and pick "Warning"
    const statusTrigger = screen.getByRole('combobox', { name: /filter.*status|status/i });
    fireEvent.click(statusTrigger);
    const warningOpt = screen.getByRole('option', { name: /^Warning$/ });
    fireEvent.click(warningOpt);

    // Only the WS1-DQ-002 'warning' row remains in the Validation Rules table
    expect(screen.queryByText('Source Coverage - Patients')).not.toBeInTheDocument();
    expect(screen.getAllByText('Completeness - HCP Master').length).toBeGreaterThanOrEqual(1);
  });

  it('#325 disables Export button while data is loading', () => {
    (useKPIList as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: mockRefetchKpis,
      isRefetching: false,
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    const exportBtn = screen.getByRole('button', { name: /Export/i });
    expect(exportBtn).toBeDisabled();
  });

  it('#328 has aria-label / label on search input and status select', () => {
    render(<DataQuality />, { wrapper: createWrapper() });

    const searchInput = screen.getByRole('textbox', {
      name: /search.*(rules|validation)/i,
    });
    expect(searchInput).toBeInTheDocument();

    const statusTrigger = screen.getByRole('combobox', { name: /filter.*status|status/i });
    expect(statusTrigger).toBeInTheDocument();
  });
});

// =============================================================================
// HONESTY — a dimension whose source KPI has no value must read "No data",
// never a fabricated healthy default. (The pre-095 prod reality: 4 of 8 WS1-DQ
// KPIs were blocked by the synthetic-exclusion gate → unknown/no value.)
// =============================================================================

describe('DataQuality honesty — missing KPI values', () => {
  beforeEach(() => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => ({
      metadata: dimensionFixtures[kpiId] ? dimensionMeta(kpiId) : (dqKpis[0] as KPIMetadata),
      value: {
        kpi_id: kpiId,
        value: undefined,
        status: 'unknown',
        error: 'KPI unavailable: no data',
        calculated_at: '2026-01-02T08:30:00Z',
        cached: false,
        metadata: {},
      },
      isLoading: false,
      error: null,
      isMetadataLoading: false,
      isValueLoading: false,
      refetch: vi.fn(),
    }));
  });

  it('shows "No data" (not a fabricated score) on every dimension card when values are absent', () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    for (const title of [
      'Overall Quality',
      'Completeness',
      'Accuracy',
      'Consistency',
      'Timeliness',
    ]) {
      const call = kpiCardCalls.find((c) => c.title === title);
      expect(call, `${title} card must render`).toBeDefined();
      expect(call!.value, `${title} must read "No data" when its source KPI has no value`).toBe(
        'No data'
      );
      expect(typeof call!.value).not.toBe('number');
      expect(call!.status).toBe('neutral');
    }
  });
});

// =============================================================================
// HONESTY — validation rule status comes from the backend `value.status`
// (good/warning/critical/unknown), NOT a naive client-side higher-is-better
// recompute. A null/UNKNOWN value is "No data", NOT a fail-X.
// =============================================================================

describe('DataQuality honesty — backend rule status (no null→X)', () => {
  function mockDetail(unknownId: string) {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === unknownId) {
        return {
          ...base,
          value: {
            kpi_id: kpiId,
            value: undefined,
            status: 'unknown',
            error: 'KPI unavailable: no data for cross-source match',
            calculated_at: '2026-01-02T08:30:00Z',
            cached: false,
            metadata: {},
          },
        };
      }
      return base;
    });
  }

  it('renders an UNKNOWN (null-value) KPI as "No data", not a value', () => {
    mockDetail('WS1-DQ-002');
    render(<DataQuality />, { wrapper: createWrapper() });
    // KPICard is stubbed to render only its title, so "No data" here is
    // unambiguously the unknown rule row's value cell (the dimension-source
    // KPIs have values → cards pass numbers to the stub).
    expect(screen.getByText('No data')).toBeInTheDocument();
  });

  it('does NOT classify an UNKNOWN (no-data) KPI as Fail — the old null→X bug', async () => {
    mockDetail('WS1-DQ-002');
    render(<DataQuality />, { wrapper: createWrapper() });

    // Select "Fail". DQ-001='good'(pass), DQ-002='unknown'. Under the OLD bug a
    // null value scored as 'fail' and DQ-002 stayed visible. Now NEITHER is a
    // fail → the empty-state appears and the unknown row is hidden.
    const statusTrigger = screen.getByRole('combobox', { name: /filter.*status|status/i });
    fireEvent.click(statusTrigger);
    fireEvent.click(screen.getByRole('option', { name: /^Fail$/ }));

    expect(
      await screen.findByText(/No data quality KPIs match your filters/i)
    ).toBeInTheDocument();
    expect(screen.queryByText('Completeness - HCP Master')).not.toBeInTheDocument();
  });
});

// =============================================================================
// CODEX ITER-1 MED FINDINGS — loading/error honesty in the rewritten render
// logic.
//  MED-1: the Quality Issues no-issues empty-state must not render while any
//         row's detail query is still in flight (a slow /api/kpis/{id} would
//         read as a clean bill of health seconds before a breach pops in).
//  MED-2: a failed dimension-source fetch must read "Error", not "No data" —
//         an API outage must be distinguishable from a genuine data gap.
// =============================================================================

describe('DataQuality — codex iter-1 loading/error honesty', () => {
  it('MED-2: a dimension whose source fetch errors reads "Error" (rose), not "No data"', () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-003') {
        const err = new Error('500 Internal Server Error');
        return { ...base, value: undefined, error: err, valueError: err };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    const accuracy = kpiCardCalls.find((c) => c.title === 'Accuracy');
    expect(accuracy).toBeDefined();
    expect(accuracy!.value).toBe('Error');
    expect(accuracy!.valueColor).toBe('text-rose-500');
    // The other dimensions still render their measured values (one outage must
    // not blank the whole grid).
    expect(kpiCardCalls.find((c) => c.title === 'Completeness')!.value).toBeCloseTo(94, 1);
    // Overall stays the honest mean of the still-measured dimensions.
    expect(typeof kpiCardCalls.find((c) => c.title === 'Overall Quality')!.value).toBe(
      'number'
    );
  });

  it('MED-2: Overall reads "Error" when nothing measured and a source errored', () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (dimensionFixtures[kpiId]) {
        const err = new Error('boom');
        return { ...base, value: undefined, error: err, valueError: err };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    for (const title of [
      'Overall Quality',
      'Completeness',
      'Accuracy',
      'Consistency',
      'Timeliness',
    ]) {
      const call = kpiCardCalls.find((c) => c.title === title);
      expect(call!.value, `${title} must read "Error" on a total outage`).toBe('Error');
    }
  });

  it('MED-1: issues tab shows a checking indicator, NOT the no-issues empty-state, while a detail query is in flight', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        return { ...base, value: undefined, isLoading: true };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    expect(await screen.findByText(/Checking KPI thresholds/i)).toBeInTheDocument();
    expect(
      screen.queryByText(/No data-quality KPIs are breaching their thresholds/i)
    ).not.toBeInTheDocument();
  });
});

// =============================================================================
// Codex iter-2 — background-refetch honesty on the Quality Issues tab.
// In react-query v5, isLoading = isPending && isFetching: it covers only the
// pre-first-data window. A background refetch of cached data (Refresh click,
// prod window-focus refetch — query-client.ts sets refetchOnWindowFocus in
// prod) keeps isLoading false, so gating only on it let a stale "no issues"
// claim stand while a recheck was in flight. Fix: rows report a separate
// fetching flag; the last settled status keeps driving rows/count (stale
// real data beats a blank — real issues must never blink out on refetch),
// while the "no issues" empty-state downgrades to the checking indicator.
// =============================================================================

describe('DataQuality — codex iter-2 background-refetch honesty', () => {
  it('downgrades "no issues" to the checking indicator while a cached row background-refetches', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        // Cached healthy value present; refetch in flight (isLoading false).
        return { ...base, isFetching: true };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    expect(await screen.findByText(/Checking KPI thresholds/i)).toBeInTheDocument();
    expect(
      screen.queryByText(/No data-quality KPIs are breaching their thresholds/i)
    ).not.toBeInTheDocument();
  });

  it('keeps a breaching row visible with its stale status during a background refetch', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        // Last settled value breaches; refetch in flight. The row and its
        // critical badge must stay rendered — never blank mid-recheck.
        return {
          ...base,
          value: { ...base.value, value: 42.0, status: 'critical' },
          isFetching: true,
        };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    expect(await screen.findByText(/\(WS1-DQ-002\)/)).toBeInTheDocument();
    expect(screen.getByText(/^critical$/i)).toBeInTheDocument();
    // The recheck indicator shows alongside the stale-listed issue.
    expect(screen.getByText(/Checking KPI thresholds/i)).toBeInTheDocument();
  });
});

// =============================================================================
// Codex iter-3 — a FAILED per-KPI fetch must not read as a clean check.
// ruleStatusFromKPI maps both no-data and errored to 'unknown', so an errored
// detail query (no cached value, error settled, isLoading/isFetching false)
// used to render identically to a healthy KPI — "No data-quality KPIs are
// breaching" could show while a check silently 500'd. Fix: error-with-no-value
// reports 'error', renders a visible failed-check row, and downgrades the
// clean empty-state to a caveated one. A stale cached value still drives the
// row through the error (same policy as the dimension cards) — a transient
// refetch blip must not flip real data into alarm — but per codex iter-4 the
// unqualified clean CLAIM no longer stands on it (see the iter-4 block below).
// =============================================================================

describe('DataQuality — codex iter-3 failed-check honesty', () => {
  it('renders a visible failed-check row and caveats the empty-state when a fetch errors with no data', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        const err = new Error('500 Internal Server Error');
        return { ...base, value: undefined, error: err, valueError: err };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    // The failed check is visible, named, and labeled — never silently clean.
    expect(await screen.findByText(/\(WS1-DQ-002\)/)).toBeInTheDocument();
    expect(screen.getByText(/^check failed$/i)).toBeInTheDocument();
    expect(
      screen.getByText(/thresholds could not be verified/i)
    ).toBeInTheDocument();
    // The unqualified clean claim is replaced by the caveated one.
    expect(
      screen.queryByText(/No data-quality KPIs are breaching their thresholds/i)
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/1 quality check failed to load and could not be verified/i)
    ).toBeInTheDocument();
  });

  it('a stale cached value beats a failed refetch — no failed-check row, but the clean claim is qualified (iter-4)', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        // Cached healthy value present; the refetch errored. The last real
        // check said healthy — a transient blip must not read as a FAILURE
        // (no check-failed row, no alarm), but per codex iter-4 the strong
        // "no issues" claim must not stand unqualified on a value that can't
        // currently be re-verified.
        const err = new Error('refetch blip');
        return { ...base, error: err, valueError: err };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    expect(
      await screen.findByText(/1 check could not be re-verified on the latest refresh/i)
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/No data-quality KPIs are breaching their thresholds/i)
    ).not.toBeInTheDocument();
    // Still no alarm: the healthy row stays hidden and nothing reads "failed".
    expect(screen.queryByText(/^check failed$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Latest recheck failed/i)).not.toBeInTheDocument();
  });
});

// =============================================================================
// Codex iter-4 — stale verification must be visible, and combined states must
// not hide each other. react-query keeps .data through a failed refetch, so a
// cached-HEALTHY KPI whose rechecks keep failing never reaches 'error' — the
// unqualified "no issues" claim could stand indefinitely after verification
// silently stopped (the risky asymmetry: a stale BREACH aging preserves the
// signal; a stale HEALTHY aging actively asserts all-clear). Fix: rows report
// a staleError signal; the clean claim is qualified while any contributing
// value can't be re-verified; stale breach rows and stale dimension cards get
// a visible note. No data disappears and nothing flips to alarm.
// =============================================================================

describe('DataQuality — codex iter-4 stale-verification honesty', () => {
  it('a stale BREACHING row stays visible with its badge plus a recheck-failed note', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        // Last settled value breaches; the latest refetch FAILED (settled
        // error, not in flight). The breach must stay listed — stale beats
        // blank — but flagged as no-longer-verified.
        const err = new Error('refetch failed');
        return {
          ...base,
          value: { ...base.value, value: 42.0, status: 'critical' },
          error: err,
          valueError: err,
        };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    expect(await screen.findByText(/\(WS1-DQ-002\)/)).toBeInTheDocument();
    expect(screen.getByText(/^critical$/i)).toBeInTheDocument();
    expect(
      screen.getByText(/Latest recheck failed — showing last known value/i)
    ).toBeInTheDocument();
    // Settled state: no checking spinner, and no empty-state of either kind.
    expect(screen.queryByText(/Checking KPI thresholds/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No breaches detected/i)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/No data-quality KPIs are breaching their thresholds/i)
    ).not.toBeInTheDocument();
  });

  it('a breach and a failed check render side by side — neither empty-state appears', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        return {
          ...base,
          value: { ...base.value, value: 42.0, status: 'critical' },
        };
      }
      if (kpiId === 'WS1-DQ-001') {
        const err = new Error('500 Internal Server Error');
        return { ...base, value: undefined, error: err, valueError: err };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    // Both the real breach and the unverifiable check are visible, named rows.
    expect(await screen.findByText(/\(WS1-DQ-002\)/)).toBeInTheDocument();
    expect(screen.getByText(/^critical$/i)).toBeInTheDocument();
    expect(await screen.findByText(/\(WS1-DQ-001\)/)).toBeInTheDocument();
    expect(screen.getByText(/^check failed$/i)).toBeInTheDocument();
    // With a breach present, no empty-state text (clean OR caveated) renders.
    expect(screen.queryByText(/No breaches detected/i)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/No data-quality KPIs are breaching their thresholds/i)
    ).not.toBeInTheDocument();
  });

  it('a dimension card showing a cached score through a failed refetch surfaces the stale note', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-005') {
        // Completeness source: cached fixture value present, refetch errored.
        // The card keeps its score (stale beats blank) but the page must say
        // the score is no longer verified.
        const err = new Error('refetch failed');
        return { ...base, error: err, valueError: err };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      await screen.findByText(/Some dimension scores could not be re-verified/i)
    ).toBeInTheDocument();
  });
});

// =============================================================================
// Codex iter-5 — staleness/failure signals must key off the VALUE query only.
// useKPIDetail().error merges metadataQuery.error || valueQuery.error, but the
// issues tab's verification claim IS the value fetch (ruleStatusFromKPI reads
// value.status; metadata is display-only with a static list fallback). A
// metadata-only failure alongside a fresh, successful value fetch must not
// fire the INVERSE false alarm — a genuinely verified check reading "could
// not be re-verified". Exception: the Timeliness dimension score is
// attainment vs the METADATA threshold target, so metadata errors still count
// there. Also pins the combined dual-clause caveat (failed + stale, zero
// breaches), which no earlier test exercised.
// =============================================================================

describe('DataQuality — codex iter-5 metadata-vs-value error separation', () => {
  it('a metadata-only failure with a fresh healthy value does NOT qualify the clean claim', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-002') {
        // The value fetch — the actual threshold check — succeeded fresh;
        // only the metadata read failed. The check DID verify this cycle.
        const err = new Error('metadata 500');
        return { ...base, metadata: undefined, error: err, metadataError: err, valueError: null };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    // The unqualified clean claim stands — nothing is stale or failed.
    expect(
      await screen.findByText(/No data-quality KPIs are breaching their thresholds/i)
    ).toBeInTheDocument();
    expect(screen.queryByText(/could not be re-verified/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^check failed$/i)).not.toBeInTheDocument();
  });

  it('a metadata-only failure does not mark a value-computed dimension stale or errored', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-005') {
        // Completeness computes from the value payload only; its metadata
        // read failing must not flag the card.
        const err = new Error('metadata 500');
        return { ...base, error: err, metadataError: err, valueError: null };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // The card still shows its measured value (positive control), with no
    // error chrome and no page-level stale note.
    const completeness = kpiCardCalls.find((c) => c.title === 'Completeness');
    expect(completeness).toBeDefined();
    expect(typeof completeness!.value).toBe('number');
    expect(
      screen.queryByText(/Some dimension scores could not be re-verified/i)
    ).not.toBeInTheDocument();
  });

  it('a metadata-only failure DOES mark Timeliness stale — its target is a score input', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-007') {
        // Data-lag source: cached metadata (with the threshold target the
        // attainment score divides by) whose refetch failed. The score still
        // computes from the cached target, but that input is unverified.
        const err = new Error('metadata 500');
        return { ...base, error: err, metadataError: err, valueError: null };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      await screen.findByText(/Some dimension scores could not be re-verified/i)
    ).toBeInTheDocument();
  });

  it('renders BOTH caveat clauses when a failed check and a stale check coexist with zero breaches', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-001') {
        // Never loaded: failed with no cached value.
        const err = new Error('500 Internal Server Error');
        return { ...base, value: undefined, error: err, valueError: err };
      }
      if (kpiId === 'WS1-DQ-002') {
        // Cached healthy value whose recheck failed.
        const err = new Error('refetch failed');
        return { ...base, error: err, valueError: err };
      }
      return base;
    });

    const user = userEvent.setup();
    render(<DataQuality />, { wrapper: createWrapper() });
    await user.click(screen.getByRole('tab', { name: /Quality Issues/i }));

    // Full assembled sentence: both clauses, em-dash joins, independent
    // singular pluralization, one terminal period.
    const caveat = await screen.findByText(
      /No breaches detected among the KPIs that could be checked/i
    );
    expect(caveat).toHaveTextContent(
      'No breaches detected among the KPIs that could be checked — 1 quality check failed to load and could not be verified — 1 check could not be re-verified on the latest refresh and shows last known values.'
    );
    // The failed check still renders as its own visible row.
    expect(screen.getByText(/\(WS1-DQ-001\)/)).toBeInTheDocument();
    expect(screen.getByText(/^check failed$/i)).toBeInTheDocument();
  });
});

// =============================================================================
// PR #1154 documented limitation, now fixed — a single-input Timeliness
// average is PARTIAL, not stale. On a cold load where one input's metadata
// (threshold target) or value never arrives, attainment() yields undefined
// for that input and the dimension averages only the survivor. That fresh
// partial score must not be worded as "showing last known values"; it gets
// its own partial note. When NEITHER input computes there is no score at all
// — no partial note, the card reads its error state.
// =============================================================================

describe('DataQuality — Timeliness partial average vs stale', () => {
  it('cold-load metadata failure on one input renders the partial note, not the stale note', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-007') {
        // Metadata NEVER loaded (no cached threshold target), value fresh:
        // the lag attainment cannot compute; only TTR survives the average.
        const err = new Error('metadata 500');
        return { ...base, metadata: undefined, error: err, metadataError: err, valueError: null };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      await screen.findByText(/The Timeliness score is a partial average/i)
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Some dimension scores could not be re-verified/i)
    ).not.toBeInTheDocument();
    // The partial score itself still renders as a number, not error chrome.
    const timeliness = kpiCardCalls.find((c) => c.title === 'Timeliness');
    expect(timeliness).toBeDefined();
    expect(typeof timeliness!.value).toBe('number');
  });

  it('cold-load value failure on one input also reads partial, not stale', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-007') {
        // Value never loaded: lag attainment cannot compute either way.
        const err = new Error('500 Internal Server Error');
        return { ...base, value: undefined, error: err, valueError: err };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    expect(
      await screen.findByText(/The Timeliness score is a partial average/i)
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/Some dimension scores could not be re-verified/i)
    ).not.toBeInTheDocument();
  });

  it('no partial note when neither input computes — the card reads Error, not a score', async () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-007' || kpiId === 'WS1-DQ-009') {
        const err = new Error('metadata 500');
        return { ...base, metadata: undefined, error: err, metadataError: err, valueError: null };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // A wholly absent score is neither partial nor stale — it is an error.
    expect(
      screen.queryByText(/The Timeliness score is a partial average/i)
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText(/Some dimension scores could not be re-verified/i)
    ).not.toBeInTheDocument();
    const timeliness = kpiCardCalls.find((c) => c.title === 'Timeliness');
    expect(timeliness).toBeDefined();
    expect(timeliness!.value).toBe('Error');
  });
});

// =============================================================================
// F3 — Brand / Region cut selectors.
// The KPI calculators are brand/region-aware (mig 078) and the value endpoint
// accepts both, but the page never passed either, so every rule value read the
// portfolio aggregate ("why only aggregated metrics?"). The selectors must (a)
// render with the real gold-standard brands + US regions, (b) default to the
// portfolio (no brand/region), and (c) forward the selection — region lowercased
// to match the backend's case — down to the per-row useKPIDetail fetch.
// =============================================================================

describe('DataQuality — F3 brand/region cut selectors', () => {
  it('renders Brand and Region selectors alongside the status filter', () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    expect(screen.getByRole('combobox', { name: /brand/i })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /region/i })).toBeInTheDocument();
    // The pre-existing status filter must remain distinct (not swallowed).
    expect(screen.getByRole('combobox', { name: /status/i })).toBeInTheDocument();
  });

  it('Brand selector lists the three gold-standard brands (+ All Brands)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('combobox', { name: /brand/i }));
    expect(screen.getByRole('option', { name: 'All Brands' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Remibrutinib' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Fabhalta' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Kisqali' })).toBeInTheDocument();
  });

  it('Region selector lists the four US regions (+ All US Regions)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('combobox', { name: /region/i }));
    expect(screen.getByRole('option', { name: 'All US Regions' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Northeast' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'South' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Midwest' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'West' })).toBeInTheDocument();
  });

  it('defaults to the portfolio aggregate (no brand/region passed to useKPIDetail)', () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    const calls = (useKPIDetail as ReturnType<typeof vi.fn>).mock.calls.filter(
      (c) => c[0] === 'WS1-DQ-001'
    );
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.every((c) => c[1] === undefined && c[2] === undefined)).toBe(true);
  });

  it('selecting a brand forwards it to the per-rule value fetch', async () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('combobox', { name: /brand/i }));
    fireEvent.click(screen.getByRole('option', { name: 'Remibrutinib' }));
    await waitFor(() =>
      expect(useKPIDetail).toHaveBeenCalledWith('WS1-DQ-001', 'Remibrutinib', undefined)
    );
  });

  it('selecting a region forwards the lowercased region to the per-rule value fetch', async () => {
    render(<DataQuality />, { wrapper: createWrapper() });
    fireEvent.click(screen.getByRole('combobox', { name: /region/i }));
    fireEvent.click(screen.getByRole('option', { name: 'West' }));
    await waitFor(() =>
      expect(useKPIDetail).toHaveBeenCalledWith('WS1-DQ-001', undefined, 'west')
    );
  });
});

// =============================================================================
// value_format='percent' — ratio KPIs (value is 0-1) must render as NN.N%, not
// a raw fraction. toFixed(1) on a fraction collapses 0.87 -> "0.9" AND hides the
// per-cut differences F3 exposes (DQ-006 0.1053 vs 0.1095 both -> "0.1"). The
// backend stamps value_format='percent'; the row must honor it.
// =============================================================================

describe('DataQuality — value_format=percent rendering', () => {
  it('renders a percent KPI as NN.N% (value*100), not the raw 0-1 fraction', () => {
    (useKPIDetail as ReturnType<typeof vi.fn>).mockImplementation((kpiId: string) => {
      const base = defaultKPIDetailImpl(kpiId);
      if (kpiId === 'WS1-DQ-001') {
        return {
          ...base,
          metadata: {
            ...dqKpis[0],
            unit: undefined,
            value_format: 'percent',
            threshold: { target: 0.85, warning: 0.7, critical: 0.5 },
          },
          value: { ...base.value, value: 0.870049, status: 'good' },
        };
      }
      return base;
    });

    render(<DataQuality />, { wrapper: createWrapper() });

    // The percent form appears; the raw fraction "0.9" must NOT.
    expect(screen.getAllByText('87.0%').length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText('0.9')).not.toBeInTheDocument();
  });
});
