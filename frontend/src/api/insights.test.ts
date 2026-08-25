import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as apiClient from '@/lib/api-client';
import {
  getCausalDiscoveryInsight,
  getHomeKpiInsight,
  getTreatmentEffectInsight,
  getClinicalNarrativeInsight,
} from './insights';

describe('insights api', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('POSTs to /insights/causal-discovery and returns the response', async () => {
    const resp = {
      insight: 'x', key_takeaways: [], grounding: [], is_fallback: true,
      generated_at: 't', provenance: 'p',
    };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getCausalDiscoveryInsight({ brand: 'Kisqali', grain: 'patient', effects: [] });
    expect(spy).toHaveBeenCalledWith('/insights/causal-discovery', expect.any(Object));
    expect(out.is_fallback).toBe(true);
  });

  it('POSTs to /insights/treatment-effect and returns the response', async () => {
    const resp = {
      insight: 'x', key_takeaways: [], grounding: [], is_fallback: true,
      generated_at: 't', provenance: 'p',
    };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getTreatmentEffectInsight({
      cohort: 'hcp_adoption', brand: 'Remibrutinib', treatment_var: 'treatment_arm',
      outcome_var: 'adopted', confounders: [], ate: 0.14, n: 5000,
    });
    expect(spy).toHaveBeenCalledWith('/insights/treatment-effect', expect.any(Object));
    expect(out.is_fallback).toBe(true);
  });

  it('POSTs only the scope to /insights/home-kpis (figures are server-derived)', async () => {
    const resp = {
      insight: 'x', key_takeaways: [], grounding: [], is_fallback: true,
      generated_at: 't', provenance: 'p',
    };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getHomeKpiInsight({ brand: 'Fabhalta', region: 'northeast' });
    expect(spy).toHaveBeenCalledWith(
      '/insights/home-kpis',
      { brand: 'Fabhalta', region: 'northeast' },
      // Extended timeout: cold scopes recompute the KPI batch + run the LM.
      { timeout: 95000 }
    );
    expect(out.is_fallback).toBe(true);
  });

  it('POSTs scope+result to /insights/clinical-narrative with the extended timeout', async () => {
    const resp = {
      insight: 'x', key_takeaways: [], grounding: [], is_fallback: false,
      generated_at: 't', provenance: 'p',
    };
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue(resp as never);
    const out = await getClinicalNarrativeInsight({
      brand: 'Remibrutinib', grain: 'hcp', treatment: 'treatment_arm', outcome: 'adopted',
      ate: 0.14, ate_ci_lower: 0.05, ate_ci_upper: 0.23, gate_decision: 'proceed',
    });
    expect(spy).toHaveBeenCalledWith(
      '/insights/clinical-narrative',
      {
        brand: 'Remibrutinib', grain: 'hcp', treatment: 'treatment_arm', outcome: 'adopted',
        ate: 0.14, ate_ci_lower: 0.05, ate_ci_upper: 0.23, gate_decision: 'proceed',
      },
      // Cold scope = clinical fan-out server-side + LM; Redis caches per grounding.
      { timeout: 95000 }
    );
    expect(out.is_fallback).toBe(false);
  });
});
