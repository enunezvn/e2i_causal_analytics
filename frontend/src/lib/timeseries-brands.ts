/**
 * Multi-brand time-series merging for the Time Series page.
 * =========================================================
 *
 * The performance trend is per-model (`{cohort}_{brand}_goldstd_lr_v1`). The
 * "All brands" view fetches one trend per brand and overlays them on a single
 * chart. {@link mergeBrandSeries} aligns the per-brand dated series into recharts
 * rows keyed by date (one numeric cell per brand; absent — NOT zero — on dates a
 * brand has no point, so recharts renders a gap). {@link meanPerDate} collapses
 * the available brands into one mean-per-date series for the summary cards.
 *
 * Pure functions; no rendering. Raw metric values are never altered.
 *
 * @module lib/timeseries-brands
 */

/** A single dated metric observation. */
export interface DatedValue {
  date: string;
  value: number;
}

/** A recharts row: a date plus one numeric cell per brand that has a point. */
export type BrandTrendRow = { date: string } & Record<string, number | string>;

/**
 * Align per-brand dated series into rows keyed by date, one cell per brand.
 *
 * @param perBrand map of brand → its dated series
 * @param brands   brands to include (single-element for one brand; all for "All")
 * @returns rows `{ date, <brand>: value, ... }` sorted ascending by date; a
 *          brand's cell is omitted on dates it has no observation (a gap).
 */
export function mergeBrandSeries(
  perBrand: Record<string, DatedValue[]>,
  brands: readonly string[]
): BrandTrendRow[] {
  const byDate = new Map<string, BrandTrendRow>();
  for (const brand of brands) {
    for (const { date, value } of perBrand[brand] ?? []) {
      const row = byDate.get(date) ?? { date };
      row[brand] = value;
      byDate.set(date, row);
    }
  }
  return Array.from(byDate.values()).sort(
    (a, b) => Date.parse(a.date) - Date.parse(b.date)
  );
}

/**
 * Collapse the per-brand series to a single mean-per-date series (the summary
 * cards / sparkline source for the "All brands" view). Only the brands that
 * actually have a point on a given date contribute to that date's mean.
 */
export function meanPerDate(
  perBrand: Record<string, DatedValue[]>,
  brands: readonly string[]
): DatedValue[] {
  const byDate = new Map<string, number[]>();
  for (const brand of brands) {
    for (const { date, value } of perBrand[brand] ?? []) {
      const arr = byDate.get(date) ?? [];
      arr.push(value);
      byDate.set(date, arr);
    }
  }
  return Array.from(byDate.entries())
    .map(([date, vals]) => ({
      date,
      value: vals.reduce((a, b) => a + b, 0) / vals.length,
    }))
    .sort((a, b) => Date.parse(a.date) - Date.parse(b.date));
}
