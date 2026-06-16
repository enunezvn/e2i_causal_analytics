import { describe, it, expect } from 'vitest';
import { formatKpiValue } from './kpi-format';

describe('formatKpiValue', () => {
  it('renders value_format=percent as (value*100) + "%" (value is a 0-1 ratio)', () => {
    expect(formatKpiValue(0.870049, { valueFormat: 'percent' })).toBe('87.0%');
    expect(formatKpiValue(0.105318, { valueFormat: 'percent' })).toBe('10.5%');
    expect(formatKpiValue(1.0, { valueFormat: 'percent' })).toBe('100.0%');
  });

  it('surfaces a per-cut difference that toFixed(1) on a raw fraction would hide', () => {
    // DQ-006 global 0.105318 vs Remibrutinib 0.109541 BOTH render "0.1" as a raw
    // fraction; the percent form keeps them distinct — the point of the F3 cut.
    expect(formatKpiValue(0.105318, { valueFormat: 'percent' })).toBe('10.5%');
    expect(formatKpiValue(0.109541, { valueFormat: 'percent' })).not.toBe('10.5%');
    expect(formatKpiValue(0.105318, { valueFormat: 'percent' })).not.toBe(
      formatKpiValue(0.109541, { valueFormat: 'percent' })
    );
  });

  it('formats a small negative ratio cleanly, not "-0.0"', () => {
    expect(formatKpiValue(-0.004372, { valueFormat: 'percent' })).toBe('-0.4%');
  });

  it('back-compat: non-percent renders value.toFixed(1) + unit with NO scaling', () => {
    expect(formatKpiValue(2.5, { unit: 'days' })).toBe('2.5days');
    expect(formatKpiValue(94.5, { unit: '%' })).toBe('94.5%');
    expect(formatKpiValue(0.87, {})).toBe('0.9');
  });

  it('honors a custom digits option', () => {
    expect(formatKpiValue(0.87349, { valueFormat: 'percent', digits: 2 })).toBe('87.35%');
  });
});
