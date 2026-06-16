/**
 * KPI value formatting.
 *
 * `valueFormat === 'percent'` means the backend value is a 0-1 RATIO to be shown
 * as a percentage (×100) — e.g. 0.87 → "87.0%". This reads correctly AND keeps
 * per-cut differences that `toFixed(1)` on a raw fraction collapses (0.1053 vs
 * 0.1095 → both "0.1", but "10.5%" vs "11.0%"). Any other format renders the
 * number as-is with its `unit` (no scaling), preserving prior behavior
 * (`value.toFixed(digits) + unit`, e.g. 2.5 + "days" → "2.5days").
 *
 * Centralized so every KPI surface (DataQuality, Home, TimeSeries, …) renders a
 * value identically. The backend declares the format via the KPI metadata
 * `value_format` field (config/kpi_definitions.yaml).
 *
 * @module lib/kpi-format
 */

export interface KpiValueFormatOpts {
  /** Unit suffix for non-percent values (e.g. 'days', 'hours'). */
  unit?: string | null;
  /** Backend display-format hint; 'percent' scales a 0-1 ratio to NN.N%. */
  valueFormat?: string | null;
  /** Decimal places (default 1). */
  digits?: number;
}

export function formatKpiValue(value: number, opts?: KpiValueFormatOpts): string {
  const digits = opts?.digits ?? 1;
  if (opts?.valueFormat === 'percent') {
    return `${(value * 100).toFixed(digits)}%`;
  }
  return `${value.toFixed(digits)}${opts?.unit ?? ''}`;
}
