/**
 * flint-chart tests — 2026-07-30
 * ==============================
 *
 * Why this exists: flint-chart@0.4.1's `assemblePlotly` performs no validation.
 * Measured directly against the package — an encoding naming a field that is
 * absent from the data compiles to a spec with that field typed nominal, with
 * no throw and no `warnings` entry, and renders as a blank plot. An
 * unrecognised semantic type is accepted the same way. Both of those values
 * come from an LLM in this integration, so the shim in front of the compiler is
 * the only thing standing between a model typo and a chart that looks broken
 * for no stated reason.
 *
 * The first test pins the upstream behaviour that motivates the shim, so if a
 * later flint version starts validating, we find out here rather than carrying
 * a redundant layer forever.
 */

import { describe, it, expect } from 'vitest';
import { assemblePlotly } from 'flint-chart';
import {
  assembleKpiFigure,
  attachErrorBars,
  encodingsFor,
  validateChartRequest,
  SUPPORTED_CHART_TYPES,
} from './flint-chart';
import { KPI_CATALOG } from './kpi-catalog.generated';
import type { ChartRequest, ChartRow, LogicalEncoding } from './flint-chart';

const ROWS = [
  { month: '2026-01-01', value: 1200 },
  { month: '2026-02-01', value: 1310 },
  { month: '2026-03-01', value: 1288 },
];

function request(overrides: Partial<ChartRequest> = {}): ChartRequest {
  return {
    rows: ROWS,
    semanticTypes: { month: 'Date', value: 'Count' },
    chartType: 'Line Chart',
    encodings: { x: 'month', y: 'value' },
    ...overrides,
  };
}

describe('upstream behaviour the shim exists for', () => {
  it('flint compiles a nonexistent field without throwing or warning', () => {
    const spec = assemblePlotly({
      data: { values: ROWS },
      semantic_types: { month: 'Date', value: 'Count' },
      chart_spec: {
        chartType: 'Line Chart',
        encodings: { x: { field: 'month' }, y: { field: 'total_scripts' } },
        baseSize: { width: 400, height: 200 },
      },
    }) as { data?: unknown[]; warnings?: unknown[] };

    // No exception, and nothing in warnings — this is the silent failure.
    expect(spec).toBeTruthy();
    expect(spec.warnings ?? []).toEqual([]);
  });
});

describe('validateChartRequest', () => {
  it('accepts a well-formed request', () => {
    expect(validateChartRequest(request())).toBeNull();
  });

  it('rejects a field that is not in the data', () => {
    const reason = validateChartRequest(
      request({ encodings: { x: 'month', y: 'total_scripts' } })
    );
    expect(reason).toMatch(/total_scripts/);
  });

  it('rejects an unknown chart type', () => {
    expect(validateChartRequest(request({ chartType: 'Forest Plot' }))).toMatch(
      /Forest Plot/
    );
  });

  it('rejects a semantic type outside flint’s vocabulary', () => {
    // 'Ratio' reads as valid and is not — flint has Percentage/Number, no Ratio.
    const reason = validateChartRequest(
      request({ semanticTypes: { month: 'Date', value: 'Ratio' } })
    );
    expect(reason).toMatch(/Ratio/);
  });

  it('rejects an empty dataset rather than compiling an empty plot', () => {
    expect(validateChartRequest(request({ rows: [] }))).toMatch(/No data/i);
  });

  it('rejects a request with no channels mapped', () => {
    expect(validateChartRequest(request({ encodings: {} }))).toMatch(/axes/i);
  });

  it('accepts a field missing from some rows but present in others', () => {
    // Segmented series legitimately leave a bucket absent from an early month.
    const sparse: ChartRow[] = [
      { month: '2026-01-01', bucket: 'high', value: 10 },
      { month: '2026-02-01', bucket: 'high', value: 12, extra: 1 },
    ];
    expect(
      validateChartRequest(
        request({
          rows: sparse,
          semanticTypes: { month: 'Date', bucket: 'Category', value: 'Count' },
          encodings: { x: 'month', y: 'value', color: 'bucket' },
        })
      )
    ).toBeNull();
  });
});

describe('assembleKpiFigure', () => {
  it('compiles real rows into a Plotly figure', () => {
    const result = assembleKpiFigure(request());
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(Array.isArray(result.figure.data)).toBe(true);
    expect(result.figure.data.length).toBeGreaterThan(0);
  });

  it('carries the real values through unchanged', () => {
    // The whole point of the data rule: what is charted is what was fetched.
    const result = assembleKpiFigure(request());
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const trace = result.figure.data[0] as { y?: unknown[] };
    expect(trace.y).toEqual([1200, 1310, 1288]);
  });

  it('returns a reason instead of a figure for a bad field', () => {
    const result = assembleKpiFigure(
      request({ encodings: { x: 'month', y: 'nope' } })
    );
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toMatch(/nope/);
  });

  it('compiles every advertised chart type against every routed shape and KPI unit', () => {
    // The question this answers: can the chat plot ANY registry KPI at ANY of
    // the chart types it advertises?
    //
    // The 45 registry KPIs do not need 45 cases. Flint compiles from the DATA
    // — a row shape plus a semantic type per field — and never sees a KPI id.
    // So the registry collapses to (shapes the router emits) x (semantic types
    // the catalog assigns), and that cross-product is the real surface.
    //
    // Both axes are DERIVED, not listed: semantic types come from KPI_CATALOG,
    // so a registry KPI introducing a new unit widens this test automatically
    // rather than slipping through untested.
    //
    // A combination that compiles to zero traces would be a dead option the
    // model can pick and the user would watch fail. This is what caught Range
    // Area Chart, which needs a y2 channel no routing branch produces.
    const semanticTypes = [...new Set(KPI_CATALOG.map((e) => e.semanticType))].sort();

    // One case per shape `kpi-chart-router` can emit. `value` is the measure in
    // every one, so the KPI's unit is the only thing that varies per iteration.
    const shapes: Array<[string, LogicalEncoding, ChartRow[], Record<string, string>]> = [
      [
        'monthly series',
        { axis: 'month', value: 'value' },
        ROWS,
        { month: 'Date' },
      ],
      [
        'segmented series',
        { axis: 'month', value: 'value', series: 'bucket' },
        [
          { month: '2026-01-01', bucket: 'high', value: 10 },
          { month: '2026-01-01', bucket: 'low', value: 40 },
          { month: '2026-02-01', bucket: 'high', value: 12 },
          { month: '2026-02-01', bucket: 'low', value: 44 },
        ],
        { month: 'Date', bucket: 'Rank' },
      ],
      [
        'multi-KPI comparison',
        { axis: 'kpi', value: 'value' },
        [
          { kpi: 'A', value: 3 },
          { kpi: 'B', value: 5 },
        ],
        { kpi: 'Category' },
      ],
      [
        'single current value',
        { axis: 'kpi', value: 'value' },
        [{ kpi: 'A', value: 3 }],
        { kpi: 'Category' },
      ],
    ];

    const failures: string[] = [];
    let checked = 0;

    for (const [shapeName, logical, rows, baseTypes] of shapes) {
      for (const semanticType of semanticTypes) {
        for (const chartType of SUPPORTED_CHART_TYPES) {
          checked++;
          const result = assembleKpiFigure({
            rows,
            semanticTypes: { ...baseTypes, value: semanticType },
            chartType,
            encodings: encodingsFor(chartType, logical),
          });
          if (!result.ok) {
            failures.push(`${chartType} / ${shapeName} / ${semanticType}: ${result.reason}`);
          }
        }
      }
    }

    expect(failures).toEqual([]);
    // Guards the derivation itself: if SUPPORTED_CHART_TYPES or the catalog's
    // unit set shrinks silently, the matrix would still "pass" while covering
    // less. 4 shapes x 5 units x 12 types = 240 today.
    expect(checked).toBe(shapes.length * semanticTypes.length * SUPPORTED_CHART_TYPES.length);
    expect(checked).toBeGreaterThanOrEqual(240);
  });
});

describe('catalog semantic types are in flint’s vocabulary', () => {
  it('accepts every semantic type the KPI catalog assigns', () => {
    // The catalog's union is hand-authored in the generator; flint accepts an
    // unknown type silently, so nothing but this test would catch a typo like
    // 'Ratio' until a chart came out unformatted.
    const used = new Set(KPI_CATALOG.map((e) => e.semanticType));
    for (const semanticType of used) {
      const reason = validateChartRequest({
        rows: [{ kpi: 'A', value: 1 }],
        semanticTypes: { kpi: 'Category', value: semanticType },
        chartType: 'Bar Chart',
        encodings: { x: 'kpi', y: 'value' },
      });
      expect(reason, `${semanticType} rejected: ${reason}`).toBeNull();
    }
  });
});

describe('confidence intervals as error bars', () => {
  // Flint ships no error-bar template and no way to register one, but its
  // output is a plain Plotly figure and Plotly draws error bars natively — so
  // the interval is attached after assembly rather than captioned away.
  const ateRows: ChartRow[] = [
    { kpi: 'High severity, 2L+', value: 0.182, ci_low: 0.121, ci_high: 0.243 },
    { kpi: 'Medium severity, 1L', value: 0.044, ci_low: -0.012, ci_high: 0.1 },
    { kpi: 'Low severity, 1L', value: 0.008, ci_low: -0.041, ci_high: 0.057 },
  ];

  function forestRequest(overrides: Partial<ChartRequest> = {}): ChartRequest {
    return {
      rows: ateRows,
      semanticTypes: {
        kpi: 'Category',
        value: 'Number',
        ci_low: 'Number',
        ci_high: 'Number',
      },
      chartType: 'Bar Chart',
      encodings: { x: 'kpi', y: 'value' },
      errorBars: { low: 'ci_low', high: 'ci_high' },
      ...overrides,
    };
  }

  it('draws the interval as offsets from each point', () => {
    const result = assembleKpiFigure(forestRequest());
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const trace = result.figure.data[0] as {
      error_y?: { array: number[]; arrayminus: number[]; symmetric: boolean };
    };
    expect(trace.error_y).toBeDefined();
    expect(trace.error_y!.symmetric).toBe(false);
    // Plotly wants deltas, not absolute bounds.
    expect(trace.error_y!.array[0]).toBeCloseTo(0.243 - 0.182, 6);
    expect(trace.error_y!.arrayminus[0]).toBeCloseTo(0.182 - 0.121, 6);
  });

  it('keeps a negative lower bound, so a CI crossing zero stays visible', () => {
    // The whole reason to draw the interval on a causal metric.
    const result = assembleKpiFigure(forestRequest());
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    const trace = result.figure.data[0] as { y: number[]; error_y: { arrayminus: number[] } };
    // Low severity: 0.008 with a lower bound of -0.041 -> a 0.049 downward whisker
    // that reaches below zero.
    expect(trace.y[2] - trace.error_y.arrayminus[2]).toBeLessThan(0);
  });

  it('draws no error bars when none were requested', () => {
    const result = assembleKpiFigure(forestRequest({ errorBars: undefined }));
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect((result.figure.data[0] as { error_y?: unknown }).error_y).toBeUndefined();
  });

  it('omits the interval entirely rather than pairing it with the wrong point', () => {
    // Guards the row-order assumption: if a future Flint version sorts or
    // aggregates, index-keyed bounds would silently attach each interval to
    // some other row. Simulated here with a figure whose y is reordered.
    const figure = {
      data: [{ x: ['a', 'b', 'c'], y: [0.008, 0.044, 0.182] }],
      layout: {},
    };
    attachErrorBars(figure, ateRows, { low: 'ci_low', high: 'ci_high' }, 'value');
    expect((figure.data[0] as { error_y?: unknown }).error_y).toBeUndefined();
  });

  it('omits the interval when a bound is missing on any row', () => {
    // Whiskers on some points and not others reads as "these are certain".
    const partial: ChartRow[] = [
      { kpi: 'A', value: 1, ci_low: 0.5, ci_high: 1.5 },
      { kpi: 'B', value: 2, ci_low: null, ci_high: null },
    ];
    const figure = { data: [{ x: ['A', 'B'], y: [1, 2] }], layout: {} };
    attachErrorBars(figure, partial, { low: 'ci_low', high: 'ci_high' }, 'value');
    expect((figure.data[0] as { error_y?: unknown }).error_y).toBeUndefined();
  });

  it('whiskers the value axis whichever side it is on', () => {
    // Horizontal (categories on y, effect on x) is the forest-plot convention.
    const figure = {
      data: [{ y: ['A', 'B', 'C'], x: ateRows.map((r) => r.value) }],
      layout: {},
    };
    attachErrorBars(figure, ateRows, { low: 'ci_low', high: 'ci_high' }, 'value');
    const trace = figure.data[0] as { error_x?: unknown; error_y?: unknown };
    expect(trace.error_x).toBeDefined();
    expect(trace.error_y).toBeUndefined();
  });
});
