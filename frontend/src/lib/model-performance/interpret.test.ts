import { describe, it, expect } from 'vitest';
import { describeModel } from './interpret';

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
