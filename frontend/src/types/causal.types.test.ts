import { describe, it, expect } from 'vitest';
import type { DiscoveredEffect } from './causal';

describe('DiscoveredEffect carries the P1 SSOT fields', () => {
  it('accepts brand + adjustment_set + summary (compile-time shape lock)', () => {
    const e: DiscoveredEffect = {
      treatment: 'treatment_arm',
      outcome: 'persistent_180d',
      status: 'completed',
      statistical_significance: true,
      confidence_score: 0.9,
      n_rows: 1500,
      brand: 'Kisqali',
      adjustment_set: ['disease_severity', 'academic_hcp'],
      summary: 'treatment_arm raises persistent_180d by +0.088.',
    };
    expect(e.brand).toBe('Kisqali');
    expect(e.adjustment_set).toEqual(['disease_severity', 'academic_hcp']);
    expect(e.summary).toContain('+0.088');
  });
});
