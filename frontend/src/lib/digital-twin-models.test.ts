import { describe, it, expect } from 'vitest';
import type { TwinModelSummary } from '@/types/digital-twin';
import {
  allModelsUnvalidated,
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
      model('Remibrutinib', { shared_fit_model_count: 1 }),
      model('Kisqali', {
        shared_fit_model_count: 1,
        training_fingerprint: 'fp-other',
        r2_score_basis: 'rwd_target',
        fidelity_status: 'validated',
        fidelity_score: 0.9,
        fidelity_sample_count: 4,
      }),
    ];
    expect(describeModelCensus(rows)).toBe('2 brand labels over 2 fits · 1 validated · 1 unvalidated');
    expect(summarizeModelCensus(rows)?.sharedBrands).toEqual([]);
    expect(explainModelCensus(rows) ?? '').not.toMatch(/routing metadata/);
  });

  it('tells a brand-scoped viewer that their one visible fit is shared, without other brand names (codex r1 #3)', () => {
    const rows = [model('Kisqali', { shared_fit_model_count: 3, shared_fit_with: [] })];
    expect(describeModelCensus(rows)).toBe(
      '3 brand labels over 1 shared synthetic fit · Kisqali unvalidated'
    );
    const why = explainModelCensus(rows) ?? '';
    expect(why).toMatch(/Brand is routing metadata: Kisqali \(2 other brand models outside your brand grant\)/);
    expect(why).toMatch(/brand is not a model feature/);
    expect(why).not.toMatch(/Remibrutinib|Fabhalta/);
    expect(summarizeModelCensus(rows)).toMatchObject({ visible: 1, labels: 3, hiddenSharedLabels: 2 });
  });

  it('does not call brand routing metadata when the backend says brand IS a feature (codex r2 #3)', () => {
    const rows = [
      model('Remibrutinib', { brand_is_feature: true, shared_fit_model_count: 2 }),
      model('Kisqali', { brand_is_feature: true, shared_fit_model_count: 2 }),
    ];
    const why = explainModelCensus(rows) ?? '';
    expect(why).toMatch(/brand is a model feature\./);
    expect(why).toMatch(/Remibrutinib, Kisqali share one identical recorded fit/);
    expect(why).not.toMatch(/routing metadata/);
  });

  it('never labels a below-threshold model as validated (codex r2 #2)', () => {
    const low = [
      model('Kisqali', {
        shared_fit_model_count: 1,
        fidelity_status: 'below_threshold',
        fidelity_score: 0.4,
        fidelity_sample_count: 2,
      }),
    ];
    expect(describeModelCensus(low)).toBe('1 brand label over 1 synthetic fit · below threshold');
    expect(explainModelCensus(low) ?? '').toMatch(/1 of 1 models? (is|are) below the fidelity threshold/);

    const mixed = [
      model('Remibrutinib', { shared_fit_model_count: 1, training_fingerprint: 'a' }),
      model('Fabhalta', {
        shared_fit_model_count: 1,
        training_fingerprint: 'b',
        fidelity_status: 'validated',
        fidelity_score: 0.9,
        fidelity_sample_count: 1,
      }),
      model('Kisqali', {
        shared_fit_model_count: 1,
        training_fingerprint: 'c',
        fidelity_status: 'below_threshold',
        fidelity_score: 0.4,
        fidelity_sample_count: 2,
      }),
    ];
    expect(describeModelCensus(mixed)).toBe(
      '3 brand labels over 3 synthetic fits · 1 validated · 1 below threshold · 1 unvalidated'
    );
  });

  it('is null with no models and labels unknown states as unknown', () => {
    expect(allModelsUnvalidated([])).toBe(false);
    expect(allModelsUnvalidated([model('Kisqali')])).toBe(true);
    expect(allModelsUnvalidated([model('Kisqali', { fidelity_status: 'validated' })])).toBe(false);
    expect(describeModelCensus([])).toBeNull();
    expect(fidelityStatusLabel('unvalidated')).toBe('Unvalidated');
    expect(fidelityStatusLabel('below_threshold')).toBe('Below threshold');
    expect(fidelityStatusLabel(undefined)).toBe('Unknown');
  });
});
