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
import {
  resolveBrand,
  resolveCompareAxis,
  resolveKpiId,
  resolveRegion,
  resolveSegment,
  resolveTherapyLine,
} from './kpi-alias';

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

describe('resolveSegment', () => {
  it('maps friendly severity names to segment_assignment values', () => {
    expect(resolveSegment('low')).toBe('low_severity');
    expect(resolveSegment('medium')).toBe('medium_severity');
    expect(resolveSegment('high')).toBe('high_severity');
    expect(resolveSegment('High Severity')).toBe('high_severity');
    expect(resolveSegment('moderate')).toBe('medium_severity');
    expect(resolveSegment('severe')).toBe('high_severity');
  });

  it('passes canonical and unknown values through (API 422s honestly)', () => {
    expect(resolveSegment('medium_severity')).toBe('medium_severity');
    expect(resolveSegment('extreme')).toBe('extreme');
  });

  it('returns undefined for missing/blank input', () => {
    expect(resolveSegment(undefined)).toBeUndefined();
    expect(resolveSegment('  ')).toBeUndefined();
  });
});

describe('resolveTherapyLine', () => {
  it('extracts the line number from friendly phrasings', () => {
    expect(resolveTherapyLine('2')).toBe('2');
    expect(resolveTherapyLine(2)).toBe('2');
    expect(resolveTherapyLine('LOT 3')).toBe('3');
    expect(resolveTherapyLine('line 1')).toBe('1');
  });

  it('returns undefined for missing/blank input', () => {
    expect(resolveTherapyLine(undefined)).toBeUndefined();
    expect(resolveTherapyLine('')).toBeUndefined();
  });
});

describe('resolveCompareAxis', () => {
  it('maps severity-ish names to the segment axis', () => {
    expect(resolveCompareAxis('severity')).toBe('segment');
    expect(resolveCompareAxis('segment')).toBe('segment');
    expect(resolveCompareAxis('Severity Tier')).toBe('segment');
  });

  it('maps LOT-ish names to the therapy_line axis', () => {
    expect(resolveCompareAxis('lot')).toBe('therapy_line');
    expect(resolveCompareAxis('LOT')).toBe('therapy_line');
    expect(resolveCompareAxis('line of therapy')).toBe('therapy_line');
    expect(resolveCompareAxis('therapy_line')).toBe('therapy_line');
  });

  it('returns undefined for unknown/missing axes', () => {
    expect(resolveCompareAxis(undefined)).toBeUndefined();
    expect(resolveCompareAxis('region')).toBeUndefined();
  });
});

describe('resolveRegion (#1538)', () => {
  it('returns undefined for a missing or blank region', () => {
    expect(resolveRegion(undefined)).toBeUndefined();
    expect(resolveRegion('')).toBeUndefined();
    expect(resolveRegion('   ')).toBeUndefined();
  });

  it('canonicalizes the enum labels case-insensitively', () => {
    expect(resolveRegion('northeast')).toBe('northeast');
    expect(resolveRegion('Northeast')).toBe('northeast');
    expect(resolveRegion('SOUTH')).toBe('south');
  });

  it('resolves the platform synonyms (enum_labels REGION_ALIASES mirror)', () => {
    expect(resolveRegion('North East')).toBe('northeast');
    expect(resolveRegion('NE')).toBe('northeast');
    expect(resolveRegion('new england')).toBe('northeast');
    expect(resolveRegion('Pacific')).toBe('west');
    expect(resolveRegion('mid-west')).toBe('midwest');
    expect(resolveRegion('Southern')).toBe('south');
  });

  it('returns null for an unmappable region so callers can refuse honestly', () => {
    // Passing junk through would put a 0-value figure under a region caption;
    // null tells the router to refuse with the known-label list instead —
    // the same fail-fast the backend chat tool applies.
    expect(resolveRegion('EMEA')).toBeNull();
    expect(resolveRegion('narnia')).toBeNull();
  });
});
