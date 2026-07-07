/**
 * kpi-alias tests — 2026-07-07
 * ============================
 *
 * Why this exists: the `renderKpiTrend` chat action told the model to pass
 * friendly ids like `nrx`, but `kpi_history` (and `/api/kpis/{id}/history`)
 * key on registry codes (NRx = WS3-BI-006). "plot NRX trends" fetched
 * history for the nonexistent id "nrx" and rendered an honest-empty chart
 * even though 35 monthly points exist. The alias map translates the friendly
 * ids the model naturally uses into the registry codes the substrate speaks.
 *
 * Brand values in kpi_history are canonical-cased ('Remibrutinib', …) and
 * the backend matches exactly, so lowercase brand names from the model must
 * be canonicalized too.
 */

import { describe, it, expect } from 'vitest';
import { resolveKpiId, resolveBrand } from './kpi-alias';

describe('resolveKpiId', () => {
  it('maps friendly commercial-KPI ids to registry codes', () => {
    expect(resolveKpiId('trx')).toBe('WS3-BI-005');
    expect(resolveKpiId('nrx')).toBe('WS3-BI-006');
    expect(resolveKpiId('nbrx')).toBe('WS3-BI-007');
    expect(resolveKpiId('trx_share')).toBe('WS3-BI-008');
    expect(resolveKpiId('market_share')).toBe('WS3-BI-008');
    expect(resolveKpiId('conversion_rate')).toBe('WS3-BI-009');
    expect(resolveKpiId('roi')).toBe('WS3-BI-010');
  });

  it('is case- and separator-insensitive for friendly ids', () => {
    expect(resolveKpiId('NRx')).toBe('WS3-BI-006');
    expect(resolveKpiId('TRX')).toBe('WS3-BI-005');
    expect(resolveKpiId('market share')).toBe('WS3-BI-008');
    expect(resolveKpiId('TRx-Share')).toBe('WS3-BI-008');
    expect(resolveKpiId(' roi ')).toBe('WS3-BI-010');
  });

  it('passes registry codes through, normalized to upper case', () => {
    expect(resolveKpiId('WS3-BI-005')).toBe('WS3-BI-005');
    expect(resolveKpiId('ws3-bi-006')).toBe('WS3-BI-006');
    expect(resolveKpiId('br-001')).toBe('BR-001');
    expect(resolveKpiId('WS2-TR-004')).toBe('WS2-TR-004');
  });

  it('returns unknown ids unchanged (honest-empty downstream)', () => {
    expect(resolveKpiId('bogus_metric')).toBe('bogus_metric');
  });
});

describe('resolveBrand', () => {
  it('canonicalizes brand casing and the remi shorthand', () => {
    expect(resolveBrand('remibrutinib')).toBe('Remibrutinib');
    expect(resolveBrand('remi')).toBe('Remibrutinib');
    expect(resolveBrand('FABHALTA')).toBe('Fabhalta');
    expect(resolveBrand('kisqali')).toBe('Kisqali');
  });

  it('passes canonical and unknown brands through', () => {
    expect(resolveBrand('Remibrutinib')).toBe('Remibrutinib');
    expect(resolveBrand('OtherBrand')).toBe('OtherBrand');
  });

  it('returns undefined for missing/blank brand', () => {
    expect(resolveBrand(undefined)).toBeUndefined();
    expect(resolveBrand('')).toBeUndefined();
    expect(resolveBrand('  ')).toBeUndefined();
  });
});
