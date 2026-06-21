import { describe, it, expect } from 'vitest';
import type { ConfusionMatrixResponse } from '@/types/monitoring';
import { aucBand, describeModel, interpretConfusion, interpretRoc } from './interpret';

describe('describeModel', () => {
  it('maps initiation cohort to patient/initiated treatment', () => {
    const m = describeModel('initiation_remibrutinib_goldstd_lr_v1');
    expect(m).toEqual({
      subject: 'patient',
      subjectPlural: 'patients',
      positiveEvent: 'initiated treatment',
      known: true,
    });
  });

  it('maps hcp_adoption to HCP/adopted the brand (not a patient cohort)', () => {
    const m = describeModel('hcp_adoption_kisqali_goldstd_lr_v1');
    expect(m.subject).toBe('HCP');
    expect(m.subjectPlural).toBe('HCPs');
    expect(m.positiveEvent).toBe('adopted the brand');
    expect(m.known).toBe(true);
  });

  it('maps persistence and discontinuation cohorts', () => {
    expect(describeModel('persistence_fabhalta_goldstd_lr_v1').positiveEvent).toBe(
      'persisted ≥180 days'
    );
    expect(describeModel('discontinuation_remibrutinib_goldstd_lr_v1').positiveEvent).toBe(
      'discontinued within 180 days'
    );
  });

  it('matches legacy names (csu_initiation, pnh_persistence)', () => {
    expect(describeModel('csu_initiation_goldstd_lr_v1').positiveEvent).toBe('initiated treatment');
    expect(describeModel('pnh_persistence_goldstd_lr_v1').positiveEvent).toBe('persisted ≥180 days');
  });

  it('falls back to generic for unknown names without throwing', () => {
    const m = describeModel('some_unknown_model');
    expect(m.known).toBe(false);
    expect(m.subjectPlural).toBe('cases');
  });
});

function cm(partial: Partial<ConfusionMatrixResponse>): ConfusionMatrixResponse {
  return {
    model_id: 'm',
    available: true,
    tn: 0,
    fp: 0,
    fn: 0,
    tp: 0,
    threshold: 0.5,
    sample_size: null,
    measured_at: null,
    ...partial,
  } as ConfusionMatrixResponse;
}

describe('interpretConfusion', () => {
  const meaning = describeModel('initiation_remibrutinib_goldstd_lr_v1');

  it('computes metrics from real initiation_remibrutinib counts', () => {
    const r = interpretConfusion(cm({ tn: 2946, fp: 346, fn: 1277, tp: 506 }), meaning);
    expect(r.precision.value).toBeCloseTo(0.5939, 3);
    expect(r.recall.value).toBeCloseTo(0.2838, 3);
    expect(r.specificity.value).toBeCloseTo(0.8949, 3);
    expect(r.accuracy.value).toBeCloseTo(0.6802, 3);
    expect(r.f1.value).toBeCloseTo(0.3841, 3);
    expect(r.precision.pct).toBe('59%');
    expect(r.recall.pct).toBe('28%');
  });

  it('selects the conservative archetype and includes real counts + domain event', () => {
    const r = interpretConfusion(cm({ tn: 2946, fp: 346, fn: 1277, tp: 506 }), meaning);
    expect(r.verdict).toContain('conservative');
    expect(r.verdict).toContain('initiated treatment');
    expect(r.verdict).toContain('506');
    expect(r.verdict).toContain('1,783'); // tp + fn
  });

  it('selects the aggressive archetype on high-recall/low-precision', () => {
    const r = interpretConfusion(cm({ tn: 100, fp: 200, fn: 10, tp: 90 }), meaning);
    expect(r.verdict).toContain('aggressive');
  });

  it('selects the balanced archetype', () => {
    const r = interpretConfusion(cm({ tn: 70, fp: 30, fn: 30, tp: 70 }), meaning);
    expect(r.verdict).toContain('balanced');
  });

  it('selects the weak archetype when recall and precision are both moderate-low', () => {
    // R=0.55, P=0.55, S=0.55 -> not conservative (R>=0.5), not aggressive (R<0.7),
    // not balanced (P<0.6) -> weak
    const r = interpretConfusion(cm({ tn: 55, fp: 45, fn: 45, tp: 55 }), meaning);
    expect(r.verdict).toContain('limited discrimination');
  });

  it('reads n/a (never a fake 0/100%) when a denominator is zero', () => {
    const r = interpretConfusion(cm({ tn: 50, fp: 0, fn: 0, tp: 0 }), meaning);
    expect(r.precision.value).toBeNull(); // tp+fp == 0
    expect(r.precision.pct).toBe('n/a');
    expect(r.recall.value).toBeNull(); // tp+fn == 0
  });

  it('returns the undetermined verdict when recall and precision are both n/a', () => {
    const r = interpretConfusion(cm({ tn: 50, fp: 0, fn: 0, tp: 0 }), meaning);
    expect(r.verdict.toLowerCase()).toContain('not enough');
  });
});

describe('aucBand', () => {
  it('maps every band boundary', () => {
    expect(aucBand(0.59)).toBe('near-random');
    expect(aucBand(0.6)).toBe('weak');
    expect(aucBand(0.69)).toBe('weak');
    expect(aucBand(0.7)).toBe('acceptable');
    expect(aucBand(0.79)).toBe('acceptable');
    expect(aucBand(0.8)).toBe('good');
    expect(aucBand(0.89)).toBe('good');
    expect(aucBand(0.9)).toBe('excellent');
  });
});

describe('interpretRoc', () => {
  const meaning = describeModel('initiation_remibrutinib_goldstd_lr_v1');

  it('formats AUC, band, ranking % and domain framing', () => {
    const r = interpretRoc(0.671, meaning);
    expect(r.band).toBe('weak');
    expect(r.text).toContain('AUC 0.671 (weak)');
    expect(r.text).toContain('67%');
    expect(r.text).toContain('patient who initiated treatment');
    expect(r.text).toContain('coin-flip');
  });

  it('uses each comparison branch', () => {
    expect(interpretRoc(0.55, meaning).text).toContain('barely above');
    expect(interpretRoc(0.65, meaning).text).toContain('modestly better');
    expect(interpretRoc(0.78, meaning).text).toContain('clearly better than chance');
    expect(interpretRoc(0.9, meaning).text).toContain('strong separation');
  });

  it('uses generic framing when the cohort is unknown', () => {
    const r = interpretRoc(0.72, describeModel('mystery_model'));
    expect(r.text).toContain('a random positive case above a random negative case');
  });

  it('keeps band and comparison consistent at the 0.7 boundary', () => {
    const r = interpretRoc(0.7, meaning);
    expect(r.band).toBe('acceptable');
    expect(r.text).toContain('clearly better than chance');
  });

  it('keeps band and comparison consistent in the near-random upper range', () => {
    const r = interpretRoc(0.58, meaning);
    expect(r.band).toBe('near-random');
    expect(r.text).toContain('barely above');
  });
});
