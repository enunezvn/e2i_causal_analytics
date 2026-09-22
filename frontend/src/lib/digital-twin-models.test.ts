import { describe, it, expect } from 'vitest';
import type { TwinModelSummary } from '@/types/digital-twin';
import {
  describeModelCensus,
  explainModelCensus,
  fidelityStatusLabel,
  summarizeModelCensus,
} from './digital-twin-models';

function model(brand: string, over: Partial<TwinModelSummary> = {}): TwinModelSummary {
  return {
    model_id: `m-${brand}`,
    model_name: `hcp_twin_${brand}`,
    twin_type: 'hcp',
    brand,
    algorithm: 'random_forest',
    r2_score: 0.8104,
    training_samples: 2000,
    is_active: true,
    created_at: '2026-06-16T00:00:00Z',
    fidelity_status: 'unvalidated',
    fidelity_score: null,
    fidelity_sample_count: 0,
    data_provenance: 'synthetic',
    r2_score_basis: 'synthetic_target',
    brand_is_feature: false,
    training_fingerprint: 'fp-shared',
    shared_fit_model_count: 3,
    shared_fit_with: [],
    ...over,
  };
}

describe('digital-twin model census (#2206)', () => {
  it('states three brand labels over one shared synthetic fit, unvalidated', () => {
    const rows = [model('Remibrutinib'), model('Fabhalta'), model('Kisqali')];
    expect(describeModelCensus(rows)).toBe(
      '3 brand labels over 1 shared synthetic fit · unvalidated'
    );
    const why = explainModelCensus(rows) ?? '';
    expect(why).toMatch(/Brand is routing metadata/);
    expect(why).toMatch(/Remibrutinib, Fabhalta, Kisqali/);
    expect(why).toMatch(/self-generated target/);
    expect(why).toMatch(/no experiment outcome has been compared/);
  });

  it('does not call distinct fits shared, and counts validated models honestly', () => {
    const rows = [
      model('Remibrutinib'),
      model('Kisqali', {
        training_fingerprint: 'fp-other',
        r2_score_basis: 'rwd_target',
        fidelity_status: 'validated',
        fidelity_score: 0.9,
        fidelity_sample_count: 4,
      }),
    ];
    expect(describeModelCensus(rows)).toBe('2 brand labels over 2 fits · 1/2 validated');
    expect(summarizeModelCensus(rows)?.sharedBrands).toEqual([]);
    expect(explainModelCensus(rows) ?? '').not.toMatch(/routing metadata/);
  });

  it('is null with no models and labels unknown states as unknown', () => {
    expect(describeModelCensus([])).toBeNull();
    expect(fidelityStatusLabel('unvalidated')).toBe('Unvalidated');
    expect(fidelityStatusLabel('below_threshold')).toBe('Below threshold');
    expect(fidelityStatusLabel(undefined)).toBe('Unknown');
  });
});
