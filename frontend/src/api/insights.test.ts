import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as apiClient from '@/lib/api-client';
import {
  getCausalDiscoveryInsight,
  getHomeKpiInsight,
  getTreatmentEffectInsight,
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
});
