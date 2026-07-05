import { describe, it, expect } from 'vitest';

import type { OpportunityListResponse, PrioritizedOpportunity } from '@/types/gaps';
import { ImplementationDifficulty } from '@/types/gaps';

import { buildExecutiveBriefRequest } from './brief-request';

function makeOpp(overrides: Partial<PrioritizedOpportunity> = {}): PrioritizedOpportunity {
  return {
    rank: 1,
    gap: {
      gap_id: 'g1',
      metric: 'trx',
      segment: 'region',
      segment_value: 'Northeast',
      current_value: 85,
      target_value: 100,
      gap_size: 15,
      gap_percentage: 15,
      gap_type: 'vs_target',
    },
    roi_estimate: {
      gap_id: 'g1',
      estimated_revenue_impact: 2_400_000,
      estimated_cost_to_close: 300_000,
      expected_roi: 4,
      risk_adjusted_roi: 3,
      payback_period_months: 6,
      attribution_level: 'partial',
      attribution_rate: 0.65,
      confidence: 0.8,
    },
    recommended_action: 'Expand specialty coverage in the Northeast',
    implementation_difficulty: ImplementationDifficulty.MEDIUM,
    time_to_impact: '3-6 months',
    category: 'steady_play',
    ...overrides,
  };
}

function makeContext(overrides: Partial<OpportunityListResponse> = {}): OpportunityListResponse {
  return {
    total_count: 3,
    quick_wins_count: 1,
    steady_plays_count: 1,
    strategic_bets_count: 1,
    suppressed_count: 2,
    opportunities: [makeOpp()],
    total_addressable_value: 5_000_000,
    ...overrides,
  };
}

describe('buildExecutiveBriefRequest', () => {
  it('maps the real ROI/gap/bucket figures 1:1 onto the insight request', () => {
    const r = buildExecutiveBriefRequest('Kisqali', makeContext());

    expect(r).not.toBeNull();
    expect(r!.brand).toBe('Kisqali');
    expect(r!.total_addressable_value).toBe(5_000_000);
    expect(r!.quick_wins_count).toBe(1);
    expect(r!.steady_plays_count).toBe(1);
    expect(r!.strategic_bets_count).toBe(1);
    expect(r!.suppressed_count).toBe(2);

    // The opportunity carries every figure the server-side grounding needs —
    // faithfully, no reformatting and no fabrication.
    expect(r!.opportunities).toHaveLength(1);
    const o = r!.opportunities![0];
    expect(o.rank).toBe(1);
    expect(o.recommended_action).toBe('Expand specialty coverage in the Northeast');
    expect(o.expected_roi).toBe(4);
    expect(o.revenue_impact).toBe(2_400_000);
    expect(o.gap_metric).toBe('trx');
    expect(o.gap_percentage).toBe(15);
    expect(o.segment_value).toBe('Northeast');
    expect(o.implementation_difficulty).toBe(ImplementationDifficulty.MEDIUM);
  });

  it('returns null when no context is available — the endpoint must not be called', () => {
    expect(buildExecutiveBriefRequest('Fabhalta', undefined)).toBeNull();
  });

  it('returns null when the feed has NO real signal (nothing surfaced, nothing suppressed)', () => {
    const r = buildExecutiveBriefRequest(
      'Remibrutinib',
      makeContext({
        opportunities: [],
        total_count: 0,
        quick_wins_count: 0,
        steady_plays_count: 0,
        strategic_bets_count: 0,
        suppressed_count: 0,
        total_addressable_value: 0,
      })
    );
    expect(r).toBeNull();
  });

  it('builds a request on the suppressed count alone (all below break-even is real signal)', () => {
    // Live finding: Fabhalta surfaces 0 opportunities but suppressed 2
    // money-losers. The honest brief is "don't invest now" (T6 narrative), so
    // the request IS built and carries the suppression signal.
    const r = buildExecutiveBriefRequest(
      'Fabhalta',
      makeContext({
        opportunities: [],
        total_count: 0,
        quick_wins_count: 0,
        steady_plays_count: 0,
        strategic_bets_count: 0,
        suppressed_count: 2,
        total_addressable_value: 0,
      })
    );
    expect(r).not.toBeNull();
    expect(r!.suppressed_count).toBe(2);
    expect(r!.opportunities).toHaveLength(0);
  });

  it('passes every surfaced opportunity through (rank ordering and top-5 cap are server-side)', () => {
    const opps = [
      makeOpp({ rank: 2, recommended_action: 'Action two' }),
      makeOpp({ rank: 1, recommended_action: 'Action one' }),
      makeOpp({ rank: 3, recommended_action: 'Action three' }),
    ];
    const r = buildExecutiveBriefRequest('Kisqali', makeContext({ opportunities: opps }));
    expect(r!.opportunities!.map((o) => o.recommended_action)).toEqual([
      'Action two',
      'Action one',
      'Action three',
    ]);
  });
});
