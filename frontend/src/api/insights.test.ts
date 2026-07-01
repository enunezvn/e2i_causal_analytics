import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as apiClient from '@/lib/api-client';
import { getCausalDiscoveryInsight } from './insights';

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
});
