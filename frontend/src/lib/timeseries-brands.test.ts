/**
 * Tests for multi-brand time-series merging.
 * ==========================================
 *
 * "All brands" overlays the per-brand performance-trend lines on one chart.
 * mergeBrandSeries() aligns the per-brand dated series into recharts rows keyed
 * by date ({ date, Remibrutinib, Fabhalta, Kisqali }), leaving a brand's cell
 * absent on dates it has no point (recharts skips gaps). meanPerDate() collapses
 * the available brands to a single mean-per-date series for the summary cards.
 */

import { describe, it, expect } from 'vitest';
import { mergeBrandSeries, meanPerDate, type DatedValue } from './timeseries-brands';

const PER_BRAND: Record<string, DatedValue[]> = {
  Remibrutinib: [
    { date: '2026-01-01', value: 0.8 },
    { date: '2026-02-01', value: 0.82 },
  ],
  Fabhalta: [
    { date: '2026-01-01', value: 0.7 },
    { date: '2026-03-01', value: 0.75 }, // a date Remibrutinib lacks
  ],
  Kisqali: [{ date: '2026-02-01', value: 0.6 }],
};
const BRANDS = ['Remibrutinib', 'Fabhalta', 'Kisqali'] as const;

describe('mergeBrandSeries', () => {
  it('aligns per-brand points into one row per date, sorted ascending', () => {
    const rows = mergeBrandSeries(PER_BRAND, BRANDS);
    expect(rows.map((r) => r.date)).toEqual(['2026-01-01', '2026-02-01', '2026-03-01']);
  });

  it('keys each brand value under its brand name', () => {
    const rows = mergeBrandSeries(PER_BRAND, BRANDS);
    const jan = rows.find((r) => r.date === '2026-01-01')!;
    expect(jan.Remibrutinib).toBe(0.8);
    expect(jan.Fabhalta).toBe(0.7);
  });

  it('leaves a brand cell ABSENT on dates it has no point (gap, not zero)', () => {
    const rows = mergeBrandSeries(PER_BRAND, BRANDS);
    const jan = rows.find((r) => r.date === '2026-01-01')!;
    // Kisqali only has a Feb point — its Jan cell must be undefined, not 0.
    expect(jan.Kisqali).toBeUndefined();
    const mar = rows.find((r) => r.date === '2026-03-01')!;
    expect(mar.Remibrutinib).toBeUndefined();
    expect(mar.Fabhalta).toBe(0.75);
  });

  it('only includes the requested brands (single-brand selection)', () => {
    const rows = mergeBrandSeries(PER_BRAND, ['Fabhalta']);
    expect(rows.map((r) => r.date)).toEqual(['2026-01-01', '2026-03-01']);
    expect(rows.every((r) => !('Remibrutinib' in r) && !('Kisqali' in r))).toBe(true);
  });

  it('returns [] for empty input', () => {
    expect(mergeBrandSeries({}, BRANDS)).toEqual([]);
  });
});

describe('meanPerDate', () => {
  it('averages only the brands present on each date', () => {
    const mean = meanPerDate(PER_BRAND, BRANDS);
    const byDate = Object.fromEntries(mean.map((p) => [p.date, p.value]));
    expect(byDate['2026-01-01']).toBeCloseTo((0.8 + 0.7) / 2, 6); // Remi + Fab
    expect(byDate['2026-02-01']).toBeCloseTo((0.82 + 0.6) / 2, 6); // Remi + Kisqali
    expect(byDate['2026-03-01']).toBeCloseTo(0.75, 6); // Fab only
  });

  it('is sorted ascending by date', () => {
    const mean = meanPerDate(PER_BRAND, BRANDS);
    expect(mean.map((p) => p.date)).toEqual(['2026-01-01', '2026-02-01', '2026-03-01']);
  });

  it('returns [] for empty input', () => {
    expect(meanPerDate({}, BRANDS)).toEqual([]);
  });
});
