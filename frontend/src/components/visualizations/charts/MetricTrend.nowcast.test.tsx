/**
 * MetricTrend — claims-lag provisional / nowcast overlay (backlog #45, PR-C)
 * ==========================================================================
 *
 * Spec for the ADDITIVE provisional/nowcast capability on MetricTrend:
 *  (a) points flagged `provisional` render as a dashed tail (its own line
 *      layer, class `metric-trend-provisional`) with hollow markers while
 *      the mature segment stays solid;
 *  (b) the tooltip disclosure for a provisional point carries the completion
 *      percentage and the nowcast estimate (pure helper — jsdom cannot
 *      activate the recharts hover tooltip, verified by probe);
 *  (c) the nowcast overlay (faint line, class `metric-trend-nowcast`, + CI
 *      band) renders ONLY when `showNowcast` is true — the default stays
 *      the honest mature view;
 *  (f) REGRESSION PIN: consumers passing only the legacy props render
 *      exactly as before (single line, no band, no provisional layers).
 *
 * jsdom constraints probed before writing this spec:
 *  - ResponsiveContainer must be mocked with fixed dimensions (the
 *    ResizeObserver stub never delivers a size);
 *  - recharts' draw-in animation freezes at frame 0 in jsdom and overwrites
 *    `stroke-dasharray` on EVERY line curve with "0px 0px", so dashedness
 *    is asserted via the stable class hook + a source pin, never via the
 *    rendered attribute.
 */

import { describe, it, expect, vi } from 'vitest';
import * as React from 'react';
import { render } from '@testing-library/react';
import * as fs from 'node:fs';
import * as path from 'node:path';

vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <div style={{ width: 800, height: 400 }}>
        {React.cloneElement(children, { width: 800, height: 400 })}
      </div>
    ),
  };
});

import * as MetricTrendModule from './MetricTrend';
import type { MetricDataPoint } from './MetricTrend';

const { MetricTrend } = MetricTrendModule;
// Runtime lookup so the WHOLE file does not fail on a missing named export
// while the helper is unimplemented (red phase) — assertions stay attributable.
const provisionalTooltipText = (
  MetricTrendModule as unknown as Record<string, unknown>
)['provisionalTooltipText'] as
  | ((
      point: Pick<MetricDataPoint, 'provisional' | 'completionFactor' | 'nowcastValue'>,
      valueFormatter?: (v: number) => string,
      unit?: string
    ) => string | null)
  | undefined;

// =============================================================================
// FIXTURES
// =============================================================================

/** 4 mature months + a 2-month provisional tail (claims still arriving). */
const RX_SERIES: MetricDataPoint[] = [
  { timestamp: '2026-01-01', value: 1200 },
  { timestamp: '2026-02-01', value: 1250 },
  { timestamp: '2026-03-01', value: 1290 },
  { timestamp: '2026-04-01', value: 1310 },
  {
    timestamp: '2026-05-01',
    value: 1322,
    provisional: true,
    completionFactor: 0.8,
    nowcastValue: 1321.25,
    nowcastCiLower: 1274,
    nowcastCiUpper: 1369,
  },
  {
    timestamp: '2026-06-01',
    value: 1340,
    provisional: true,
    completionFactor: 0.55,
    nowcastValue: 1352.7,
    nowcastCiLower: 1281,
    nowcastCiUpper: 1420,
  },
];

/** Legacy shape only — what every pre-existing consumer passes. */
const LEGACY_SERIES: MetricDataPoint[] = [
  { timestamp: '2024-01-01', value: 0.85 },
  { timestamp: '2024-01-08', value: 0.87 },
  { timestamp: '2024-01-15', value: 0.84 },
  { timestamp: '2024-01-22', value: 0.91, annotation: 'Model updated' },
];

const HOLLOW_FILL = 'var(--color-background)';

function lineCurveCount(container: HTMLElement): number {
  return container.querySelectorAll('.recharts-line-curve').length;
}

function hollowDotCount(container: HTMLElement): number {
  return [...container.querySelectorAll('.recharts-line-dots circle')].filter(
    (c) => c.getAttribute('fill') === HOLLOW_FILL
  ).length;
}

function readSource(): string {
  return fs.readFileSync(path.resolve(__dirname, 'MetricTrend.tsx'), 'utf-8');
}

// =============================================================================
// (a) provisional tail styling
// =============================================================================

describe('MetricTrend provisional tail (a)', () => {
  it('splits the series: solid mature segment + a dedicated provisional tail line', () => {
    const { container } = render(<MetricTrend name="TRx" data={RX_SERIES} />);

    // Two line series: the solid mature segment and the provisional tail.
    expect(lineCurveCount(container)).toBe(2);
    expect(container.querySelectorAll('.metric-trend-provisional').length).toBe(1);
    expect(
      container.querySelectorAll('.metric-trend-provisional .recharts-line-curve').length
    ).toBe(1);
  });

  it('dashes the provisional tail (source pin — jsdom freezes draw-in animation at frame 0)', () => {
    const source = readSource();
    // The provisional Line must carry BOTH the class hook and a real dash
    // pattern; the rendered stroke-dasharray attribute cannot be asserted
    // under jsdom (animation frame 0 overwrites it on every curve).
    expect(source).toMatch(
      /className="metric-trend-provisional"[\s\S]{0,400}strokeDasharray="\d[\d ]*"|strokeDasharray="\d[\d ]*"[\s\S]{0,400}className="metric-trend-provisional"/
    );
  });

  it('renders hollow markers on exactly the provisional points', () => {
    const { container } = render(<MetricTrend name="TRx" data={RX_SERIES} />);

    // 2 provisional months -> 2 hollow dots (the mature connector point that
    // anchors the dashed tail must NOT read as provisional).
    expect(hollowDotCount(container)).toBe(2);
  });
});

// =============================================================================
// (b) tooltip disclosure
// =============================================================================

describe('provisionalTooltipText (b)', () => {
  it('is exported from the MetricTrend module', () => {
    expect(provisionalTooltipText).toBeTypeOf('function');
  });

  it('discloses the completion percentage and the nowcast estimate', () => {
    const text = provisionalTooltipText?.(
      { provisional: true, completionFactor: 0.8, nowcastValue: 1321.25 },
      (v) => v.toLocaleString('en-US'),
      ''
    );
    expect(text).toBe(
      "Provisional — claims still maturing (~80% of this month's claims have arrived). " +
        'Nowcast estimate: 1,321.25.'
    );
  });

  it('returns null for a mature point (no disclosure on truth)', () => {
    expect(
      provisionalTooltipText?.({ provisional: false, completionFactor: 1, nowcastValue: null })
    ).toBeNull();
    expect(provisionalTooltipText?.({ provisional: undefined })).toBeNull();
  });

  it('never fabricates: omits the percentage and estimate when unknown', () => {
    // A month younger than the observed lag support carries NO completion
    // factor and NO nowcast — the disclosure must not invent either.
    const text = provisionalTooltipText?.({
      provisional: true,
      completionFactor: null,
      nowcastValue: null,
    });
    expect(text).toBe('Provisional — claims still maturing.');
  });

  it('is wired into the chart tooltip (source pin)', () => {
    const source = readSource();
    // Definition + at least one call site inside the component.
    expect((source.match(/provisionalTooltipText/g) ?? []).length).toBeGreaterThanOrEqual(2);
    expect(source).toMatch(/export function provisionalTooltipText/);
  });
});

// =============================================================================
// (c) opt-in nowcast overlay
// =============================================================================

describe('MetricTrend nowcast overlay (c)', () => {
  it('hides the overlay by default — the honest mature view is the default', () => {
    const { container } = render(<MetricTrend name="TRx" data={RX_SERIES} />);

    expect(container.querySelector('.recharts-area')).toBeNull();
    expect(container.querySelectorAll('.metric-trend-nowcast').length).toBe(0);
    expect(lineCurveCount(container)).toBe(2);
  });

  it('showNowcast renders the nowcast line and its CI band over the tail', () => {
    const { container } = render(<MetricTrend name="TRx" data={RX_SERIES} showNowcast />);

    // CI band (range Area) + a third (nowcast) line.
    expect(container.querySelector('.recharts-area')).not.toBeNull();
    expect(container.querySelectorAll('.metric-trend-nowcast').length).toBe(1);
    expect(lineCurveCount(container)).toBe(3);
  });

  it('showNowcast on an all-mature series renders no overlay artifacts', () => {
    const { container } = render(<MetricTrend name="Acc" data={LEGACY_SERIES} showNowcast />);

    expect(container.querySelector('.recharts-area')).toBeNull();
    expect(container.querySelectorAll('.metric-trend-nowcast').length).toBe(0);
    expect(lineCurveCount(container)).toBe(1);
  });
});

// =============================================================================
// (f) regression pin — legacy consumers render byte-identically
// =============================================================================

describe('MetricTrend legacy rendering (f)', () => {
  it('legacy props render a single line with no provisional/nowcast artifacts', () => {
    const { container } = render(<MetricTrend name="Accuracy" data={LEGACY_SERIES} />);

    expect(lineCurveCount(container)).toBe(1);
    expect(container.querySelectorAll('.metric-trend-provisional').length).toBe(0);
    expect(container.querySelectorAll('.metric-trend-nowcast').length).toBe(0);
    expect(container.querySelector('.recharts-area')).toBeNull();
    expect(hollowDotCount(container)).toBe(0);
    // The annotation dot contract survives: one filled r=5 marker.
    const annotationDotCount = [
      ...container.querySelectorAll('.recharts-line-dots circle'),
    ].filter((c) => c.getAttribute('r') === '5').length;
    expect(annotationDotCount).toBe(1);
  });
});
