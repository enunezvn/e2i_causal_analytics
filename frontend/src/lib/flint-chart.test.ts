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

  it('compiles every advertised chart type against both routed data shapes', () => {
    // A type in the action description that compiles to zero traces would be a
    // dead option the model can pick and the user would see fail. This caught
    // Range Area Chart, which needs a y2 channel no routing branch produces.
    const shapes: Array<[string, LogicalEncoding, Omit<ChartRequest, 'encodings'>]> = [
      [
        'categorical comparison',
        { axis: 'kpi', value: 'value' },
        {
          rows: [
            { kpi: 'A', value: 3 },
            { kpi: 'B', value: 5 },
          ],
          semanticTypes: { kpi: 'Category', value: 'Number' },
          chartType: 'Bar Chart',
        },
      ],
      [
        'monthly series',
        { axis: 'month', value: 'value' },
        {
          rows: ROWS,
          semanticTypes: { month: 'Date', value: 'Count' },
          chartType: 'Line Chart',
        },
      ],
      [
        'segmented series',
        { axis: 'month', value: 'value', series: 'bucket' },
        {
          rows: [
            { month: '2026-01-01', bucket: 'high', value: 10 },
            { month: '2026-01-01', bucket: 'low', value: 40 },
            { month: '2026-02-01', bucket: 'high', value: 12 },
            { month: '2026-02-01', bucket: 'low', value: 44 },
          ],
          semanticTypes: { month: 'Date', bucket: 'Category', value: 'Count' },
          chartType: 'Line Chart',
        },
      ],
    ];

    for (const [shapeName, logical, base] of shapes) {
      for (const chartType of SUPPORTED_CHART_TYPES) {
        const result = assembleKpiFigure({
          ...base,
          chartType,
          encodings: encodingsFor(chartType, logical),
        });
        expect(
          result.ok,
          `${chartType} / ${shapeName}: ${result.ok ? '' : result.reason}`
        ).toBe(true);
      }
    }
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
