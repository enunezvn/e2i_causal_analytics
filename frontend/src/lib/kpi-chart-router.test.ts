/**
 * kpi-chart-router tests — 2026-07-30
 * ===================================
 *
 * Why this exists: the chat could chart a KPI only if it had a materialized
 * monthly series, which most of the registry does not. Asking to plot ROC-AUC
 * or Cross-source Match Rate produced an empty frame even though the value was
 * one `/api/kpis/{id}` call away. The router picks the endpoint that can serve
 * each KPI, so "no trend" becomes a current-value chart rather than a dead end.
 *
 * The coverage test is the load-bearing one: it drives every KPI in the
 * registry through the router and asserts each produces either real rows or a
 * stated reason — never a silent blank.
 *
 * Every branch here reads mocked API responses standing in for real ones. No
 * test asserts a fabricated value is charted, because the router must never
 * produce one.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockGetKPIHistory = vi.fn();
const mockGetKPIHistorySegmented = vi.fn();
const mockGetKPIValue = vi.fn();
const mockBatchCalculateKPIs = vi.fn();

vi.mock('@/api/kpi', () => ({
  getKPIHistory: (...args: unknown[]) => mockGetKPIHistory(...args),
  getKPIHistorySegmented: (...args: unknown[]) => mockGetKPIHistorySegmented(...args),
  getKPIValue: (...args: unknown[]) => mockGetKPIValue(...args),
  batchCalculateKPIs: (...args: unknown[]) => mockBatchCalculateKPIs(...args),
}));

const { routeKpiChart } = await import('./kpi-chart-router');
const { KPI_CATALOG } = await import('./kpi-catalog.generated');
const { assembleKpiFigure, encodingsFor } = await import('./flint-chart');

const NO_HISTORY = { kpi_id: '', brand: '', region: '', count: 0, points: [] };

beforeEach(() => {
  vi.clearAllMocks();
  mockGetKPIHistory.mockResolvedValue({ ...NO_HISTORY });
  mockGetKPIValue.mockResolvedValue({
    kpi_id: 'X',
    value: 0.87,
    status: 'good',
    calculated_at: '2026-07-30T00:00:00Z',
    cached: false,
    metadata: {},
  });
});

describe('history routing', () => {
  it('charts a materialized monthly series as a time series', async () => {
    mockGetKPIHistory.mockResolvedValue({
      kpi_id: 'WS3-BI-005',
      brand: '',
      region: '',
      count: 2,
      points: [
        { metric_date: '2026-05-01', value: 1200 },
        { metric_date: '2026-06-01', value: 1310 },
      ],
    });

    const data = await routeKpiChart({ kpis: ['trx'] });

    expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', undefined, undefined);
    expect(data.emptyReason).toBeUndefined();
    expect(data.chartType).toBe('Line Chart');
    // Real API values reach the rows unchanged.
    expect(data.rows).toEqual([
      { month: '2026-05-01', value: 1200 },
      { month: '2026-06-01', value: 1310 },
    ]);
    expect(data.semanticTypes.value).toBe('Count');
  });

  it('canonicalizes the brand before fetching', async () => {
    await routeKpiChart({ kpis: ['trx'], brand: 'remi' });
    expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', 'Remibrutinib', undefined);
  });

  it('honours an explicit chart type over the routed default', async () => {
    mockGetKPIHistory.mockResolvedValue({
      kpi_id: 'WS3-BI-005',
      brand: '',
      region: '',
      count: 1,
      points: [{ metric_date: '2026-06-01', value: 1310 }],
    });
    const data = await routeKpiChart({ kpis: ['trx'], chartType: 'Bar Chart' });
    expect(data.chartType).toBe('Bar Chart');
  });
});

describe('patient-axis routing', () => {
  it('fetches the segmented endpoint and colours by bucket', async () => {
    mockGetKPIHistorySegmented.mockResolvedValue({
      kpi_id: 'WS3-BI-005',
      axis: 'segment',
      brand: '',
      data_through: '2026-06-01',
      series: [
        {
          key: 'high_severity',
          label: 'High severity',
          points: [{ metric_date: '2026-06-01', value: 300 }],
        },
        {
          key: 'low_severity',
          label: 'Low severity',
          points: [{ metric_date: '2026-06-01', value: 900 }],
        },
      ],
    });

    const data = await routeKpiChart({ kpis: ['trx'], compareBy: 'severity' });

    expect(mockGetKPIHistorySegmented).toHaveBeenCalledWith(
      'WS3-BI-005',
      'segment',
      undefined,
      undefined
    );
    expect(data.encoding.series).toBe('bucket');
    expect(data.rows).toHaveLength(2);
    expect(data.subtitle).toContain('by severity tier');
  });

  it('explains that a non-axis KPI cannot be split, instead of 422-ing', async () => {
    // The segmented endpoint 422s for anything but TRx/NRx/NBRx; saying so is
    // more useful than a failed request the user cannot interpret.
    const data = await routeKpiChart({ kpis: ['roc_auc'], compareBy: 'severity' });
    expect(mockGetKPIHistorySegmented).not.toHaveBeenCalled();
    expect(data.emptyReason).toMatch(/not tracked by severity tier/);
  });
});

describe('current-value routing', () => {
  it('charts the current value when a KPI has no series', async () => {
    const data = await routeKpiChart({ kpis: ['roc_auc'] });

    expect(mockGetKPIHistory).toHaveBeenCalledWith('WS1-MP-001', undefined, undefined);
    // Third arg = region (#1538), explicitly undefined when none was asked.
    expect(mockGetKPIValue).toHaveBeenCalledWith('WS1-MP-001', undefined, undefined);
    expect(data.emptyReason).toBeUndefined();
    expect(data.chartType).toBe('KPI Card');
    expect(data.rows[0].value).toBe(0.87);
    expect(encodingsFor(data.chartType, data.encoding).metric).toBe('kpi');
    expect(encodingsFor(data.chartType, data.encoding).value).toBe('value');
  });

  it('draws the registry threshold as the KPI Card goal', async () => {
    const data = await routeKpiChart({ kpis: ['roc_auc'] });
    // WS1-MP-001 declares a target in the registry; it must come from there.
    const target = KPI_CATALOG.find((e) => e.id === 'WS1-MP-001')?.target;
    expect(target).toBeDefined();
    expect(data.rows[0].target).toBe(target);
    expect(encodingsFor(data.chartType, data.encoding).goal).toBe('target');
  });

  it('adds no goal for a KPI the registry gives no target', async () => {
    // Causal metrics have no threshold; inventing one would put a fake
    // benchmark on a real chart.
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'CM-001',
      value: 0.12,
      status: 'informational',
      calculated_at: '2026-07-30T00:00:00Z',
      cached: false,
      metadata: {},
    });
    const data = await routeKpiChart({ kpis: ['ate'] });
    expect(data.rows[0].target).toBeUndefined();
    expect(encodingsFor(data.chartType, data.encoding).goal).toBeUndefined();
  });

  it('draws a confidence interval rather than only captioning it', async () => {
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'CM-001',
      value: 0.12,
      status: 'informational',
      confidence_interval: [-0.02, 0.26],
      calculated_at: '2026-07-30T00:00:00Z',
      cached: false,
      metadata: {},
    });
    const data = await routeKpiChart({ kpis: ['CM-001'] });

    // A KPI Card is a Plotly `indicator` and cannot carry whiskers, so a KPI
    // reporting an interval switches to a bar.
    expect(data.chartType).toBe('Bar Chart');
    expect(data.errorBars).toEqual({ low: 'ci_low', high: 'ci_high' });
    expect(data.rows[0].ci_low).toBe(-0.02);
    expect(data.rows[0].ci_high).toBe(0.26);
    // Still stated in the caption too — the chart shows it, the text says it.
    expect(data.subtitle).toContain('95% CI');
    expect(data.subtitle).toContain('-0.02');
  });

  it('keeps the KPI Card when there is no interval to draw', async () => {
    const data = await routeKpiChart({ kpis: ['roc_auc'] });
    expect(data.chartType).toBe('KPI Card');
    expect(data.errorBars).toBeUndefined();
  });

  it('reports a calculation error as an error, not as empty', async () => {
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'WS1-MP-001',
      status: 'unknown',
      error: 'no scored cohort in window',
      calculated_at: '2026-07-30T00:00:00Z',
      cached: false,
      metadata: {},
    });
    const data = await routeKpiChart({ kpis: ['roc_auc'] });
    expect(data.emptyReason).toMatch(/no scored cohort in window/);
    expect(data.rows).toEqual([]);
  });
});

describe('multi-KPI comparison', () => {
  it('batches several KPIs into one comparison chart', async () => {
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        { kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {} },
        { kpi_id: 'WS1-MP-002', value: 0.64, status: 'warning', calculated_at: '', cached: false, metadata: {} },
      ],
      calculated_at: '',
      total_kpis: 2,
    });

    const data = await routeKpiChart({ kpis: ['roc_auc', 'pr_auc'] });

    expect(mockBatchCalculateKPIs).toHaveBeenCalledWith({
      kpi_ids: ['WS1-MP-001', 'WS1-MP-002'],
      context: undefined,
    });
    expect(data.chartType).toBe('Bar Chart');
    expect(data.rows).toEqual([
      { kpi: 'ROC-AUC', value: 0.87 },
      { kpi: 'PR-AUC', value: 0.64 },
    ]);
  });

  it('refuses to format an axis shared by KPIs with different units', async () => {
    // A percentage and a raw count cannot share formatted ticks; labelling
    // both as the first KPI's unit would misstate one of them.
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        { kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {} },
        { kpi_id: 'WS3-BI-005', value: 41000, status: 'good', calculated_at: '', cached: false, metadata: {} },
      ],
      calculated_at: '',
      total_kpis: 2,
    });
    const data = await routeKpiChart({ kpis: ['roc_auc', 'trx'] });
    expect(data.semanticTypes.value).toBe('Number');
    expect(data.subtitle).toContain('mixed units');
  });

  it('draws a forest plot when every compared KPI reports an interval', async () => {
    // Several causal metrics side by side, each with its CI, IS the forest plot.
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        { kpi_id: 'CM-001', value: 0.18, confidence_interval: [0.12, 0.24], status: 'informational', calculated_at: '', cached: false, metadata: {} },
        { kpi_id: 'CM-002', value: 0.04, confidence_interval: [-0.01, 0.09], status: 'informational', calculated_at: '', cached: false, metadata: {} },
      ],
      calculated_at: '',
      total_kpis: 2,
    });
    const data = await routeKpiChart({ kpis: ['ate', 'cate'] });

    expect(data.errorBars).toEqual({ low: 'ci_low', high: 'ci_high' });
    expect(data.rows.map((r) => r.ci_low)).toEqual([0.12, -0.01]);
    expect(data.subtitle).toContain('95% CI');
  });

  it('omits intervals when only some compared KPIs report one', async () => {
    // Whiskers on two bars but not the third would read as "the third is
    // certain", which is the opposite of what a missing interval means.
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        { kpi_id: 'CM-001', value: 0.18, confidence_interval: [0.12, 0.24], status: 'informational', calculated_at: '', cached: false, metadata: {} },
        { kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {} },
      ],
      calculated_at: '',
      total_kpis: 2,
    });
    const data = await routeKpiChart({ kpis: ['ate', 'roc_auc'] });

    expect(data.errorBars).toBeUndefined();
    expect(data.rows[0].ci_low).toBeUndefined();
    expect(data.subtitle).toContain('intervals omitted');
  });

  it('says how many KPIs returned nothing', async () => {
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        { kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {} },
        { kpi_id: 'WS1-MP-002', status: 'unknown', calculated_at: '', cached: false, metadata: {} },
      ],
      calculated_at: '',
      total_kpis: 2,
    });
    const data = await routeKpiChart({ kpis: ['roc_auc', 'pr_auc'] });
    expect(data.rows).toHaveLength(1);
    expect(data.subtitle).toContain('1 returned no value');
  });
});

describe('registry coverage', () => {
  it('routes and compiles a chart for every KPI in the registry', async () => {
    // The requirement this whole change exists for: no KPI in the registry is
    // unreachable from the chat. Each one must either compile to a real figure
    // or state why it cannot — never a silent blank frame.
    const failures: string[] = [];

    for (const entry of KPI_CATALOG) {
      vi.clearAllMocks();
      mockGetKPIHistory.mockResolvedValue({ ...NO_HISTORY, kpi_id: entry.id });
      mockGetKPIValue.mockResolvedValue({
        kpi_id: entry.id,
        value: 0.5,
        status: 'good',
        calculated_at: '2026-07-30T00:00:00Z',
        cached: false,
        metadata: {},
      });

      const data = await routeKpiChart({ kpis: [entry.id] });

      if (data.emptyReason) {
        failures.push(`${entry.id}: unexpectedly empty (${data.emptyReason})`);
        continue;
      }
      const compiled = assembleKpiFigure({
        rows: data.rows,
        semanticTypes: data.semanticTypes,
        chartType: data.chartType,
        encodings: encodingsFor(data.chartType, data.encoding),
      });
      if (!compiled.ok) {
        failures.push(`${entry.id}: ${compiled.reason}`);
      }
    }

    expect(failures).toEqual([]);
  });

  it('resolves every KPI by display name too, not just by code', async () => {
    const unresolved: string[] = [];
    for (const entry of KPI_CATALOG) {
      vi.clearAllMocks();
      mockGetKPIHistory.mockResolvedValue({ ...NO_HISTORY });
      mockGetKPIValue.mockResolvedValue({
        kpi_id: entry.id,
        value: 1,
        status: 'good',
        calculated_at: '',
        cached: false,
        metadata: {},
      });
      await routeKpiChart({ kpis: [entry.name] });
      const requested = mockGetKPIHistory.mock.calls[0]?.[0];
      if (requested !== entry.id) {
        unresolved.push(`${entry.name} -> ${requested}`);
      }
    }
    expect(unresolved).toEqual([]);
  });

  it('names no KPI at all as an explicit reason', async () => {
    const data = await routeKpiChart({ kpis: [] });
    expect(data.emptyReason).toMatch(/No KPI was named/);
  });
});

describe('region scope (#1536)', () => {
  it('threads region into the history fetch and captions the chart with it', async () => {
    mockGetKPIHistory.mockResolvedValue({
      kpi_id: 'WS3-BI-005',
      brand: 'Kisqali',
      region: 'northeast',
      count: 1,
      points: [{ metric_date: '2026-06-01', value: 249 }],
    });

    const data = await routeKpiChart({ kpis: ['trx'], brand: 'kisqali', region: 'Northeast' });

    expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', 'Kisqali', 'northeast');
    expect(data.emptyReason).toBeUndefined();
    expect(data.subtitle).toContain('northeast');
  });

  it('falls through to the region-aware current value when the region has no series (#1538)', async () => {
    // #1536 refused here because the frontend could not verify whether a
    // calculator had a live region variant. The backend now attests it via
    // region provenance, so the fall-through is safe: the response says
    // whether the figure is region-scoped, and the router obeys.
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'WS3-BI-009',
      value: 0.31,
      status: 'good',
      calculated_at: '2026-08-11T00:00:00Z',
      cached: false,
      metadata: {},
      region_requested: 'northeast',
      region_applied: 'northeast',
      region_status: 'applied',
    });

    const data = await routeKpiChart({ kpis: ['conversion_rate'], region: 'northeast' });

    expect(mockGetKPIValue).toHaveBeenCalledWith('WS3-BI-009', undefined, 'northeast');
    expect(data.emptyReason).toBeUndefined();
    expect(data.rows[0].value).toBe(0.31);
    expect(data.subtitle).toContain('northeast');
  });

  it('refuses a region-scoped segmented request — the segmented endpoint is global-only', async () => {
    const data = await routeKpiChart({
      kpis: ['trx'],
      compareBy: 'severity',
      region: 'northeast',
    });

    expect(data.emptyReason).toMatch(/global-only/i);
    expect(mockGetKPIHistorySegmented).not.toHaveBeenCalled();
  });

  it('keeps the exact pre-region fetch when no region is passed', async () => {
    await routeKpiChart({ kpis: ['trx'] });
    expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', undefined, undefined);
  });
});

describe('region vocabulary (#1538)', () => {
  it('resolves a region synonym before fetching', async () => {
    await routeKpiChart({ kpis: ['trx'], region: 'North East' });
    expect(mockGetKPIHistory).toHaveBeenCalledWith('WS3-BI-005', undefined, 'northeast');
  });

  it('refuses an unmappable region with the known labels, before any fetch', async () => {
    // Passing junk through would produce a 0-value figure under a region
    // caption — the same misleading shape the backend tool fails fast on.
    const data = await routeKpiChart({ kpis: ['trx'], region: 'EMEA' });

    expect(data.emptyReason).toMatch(/EMEA/);
    expect(data.emptyReason).toMatch(/northeast.*south.*midwest.*west/i);
    expect(mockGetKPIHistory).not.toHaveBeenCalled();
    expect(mockGetKPIValue).not.toHaveBeenCalled();
  });
});

describe('region-aware current values (#1538)', () => {
  it('refuses to caption a global value with the region when provenance says not applied', async () => {
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'WS3-BI-004',
      value: 0.42,
      status: 'good',
      calculated_at: '2026-08-11T00:00:00Z',
      cached: false,
      metadata: {},
      region_requested: 'northeast',
      region_applied: null,
      region_status: 'not_applicable',
    });

    const data = await routeKpiChart({ kpis: ['hcp_coverage'], region: 'northeast' });

    expect(data.emptyReason).toMatch(/global/i);
    expect(data.emptyReason).toMatch(/northeast/);
    expect(data.rows).toEqual([]);
  });

  it('refuses when the backend reports no region provenance at all', async () => {
    // A pre-#1538 backend (rolling deploy) cannot attest the scope; charting
    // its value under the region caption would be the exact mislabel this
    // change removes.
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'WS3-BI-009',
      value: 0.31,
      status: 'good',
      calculated_at: '2026-08-11T00:00:00Z',
      cached: false,
      metadata: {},
    });

    const data = await routeKpiChart({ kpis: ['conversion_rate'], region: 'northeast' });

    expect(data.emptyReason).toMatch(/region/i);
    expect(data.rows).toEqual([]);
  });

  it('surfaces the calculator error verbatim for an unsupported region combination', async () => {
    mockGetKPIValue.mockResolvedValue({
      kpi_id: 'WS3-BI-009',
      status: 'unknown',
      error: 'brand and region cannot be combined for conversion rate',
      calculated_at: '2026-08-11T00:00:00Z',
      cached: false,
      metadata: {},
    });

    const data = await routeKpiChart({
      kpis: ['conversion_rate'],
      brand: 'kisqali',
      region: 'northeast',
    });

    expect(data.emptyReason).toMatch(/brand and region cannot be combined/);
  });

  it('charts only the region-scoped results in a region comparison and says what was omitted', async () => {
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        {
          kpi_id: 'WS3-BI-005', value: 249, status: 'informational', calculated_at: '', cached: false, metadata: {},
          region_requested: 'northeast', region_applied: 'northeast', region_status: 'applied',
        },
        {
          kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {},
          region_requested: 'northeast', region_applied: null, region_status: 'not_applicable',
        },
      ],
      calculated_at: '',
      total_kpis: 2,
    });

    const data = await routeKpiChart({ kpis: ['trx', 'roc_auc'], region: 'northeast' });

    expect(mockBatchCalculateKPIs).toHaveBeenCalledWith({
      kpi_ids: ['WS3-BI-005', 'WS1-MP-001'],
      context: { region: 'northeast' },
    });
    // Mixing a northeast figure and a global figure on one labeled axis would
    // mislabel the global one — only attested region values are drawn.
    expect(data.rows).toHaveLength(1);
    expect(data.rows[0].kpi).toContain('TRx');
    expect(data.subtitle).toContain('northeast');
    expect(data.subtitle).toMatch(/1 .*(global|no northeast)/i);
    expect(data.emptyReason).toBeUndefined();
  });

  it('refuses the comparison when no compared KPI is region-scoped', async () => {
    mockBatchCalculateKPIs.mockResolvedValue({
      results: [
        {
          kpi_id: 'WS1-MP-001', value: 0.87, status: 'good', calculated_at: '', cached: false, metadata: {},
          region_requested: 'northeast', region_applied: null, region_status: 'not_applicable',
        },
        {
          kpi_id: 'WS1-MP-002', value: 0.64, status: 'warning', calculated_at: '', cached: false, metadata: {},
          region_requested: 'northeast', region_applied: null, region_status: 'not_applicable',
        },
      ],
      calculated_at: '',
      total_kpis: 2,
    });

    const data = await routeKpiChart({ kpis: ['roc_auc', 'pr_auc'], region: 'northeast' });

    expect(data.rows).toEqual([]);
    expect(data.emptyReason).toMatch(/northeast/);
    expect(data.emptyReason).toMatch(/global/i);
  });
});
