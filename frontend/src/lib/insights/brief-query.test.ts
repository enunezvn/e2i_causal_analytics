import { describe, it, expect } from 'vitest';

import type { OpportunityListResponse, PrioritizedOpportunity } from '@/types/gaps';
import { ImplementationDifficulty } from '@/types/gaps';

import { buildExecutiveBriefQuery } from './brief-query';

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

describe('buildExecutiveBriefQuery', () => {
  it('grounds the query in real ROI/gap/bucket figures from the opportunity context', () => {
    const q = buildExecutiveBriefQuery('Kisqali', makeContext());

    // Names the brand and preserves the original brief intent.
    expect(q).toContain('Kisqali');
    expect(q.toLowerCase()).toMatch(/performance trend|opportunit|risk/);

    // Real total addressable value (formatted, not raw).
    expect(q).toMatch(/\$5\.0M/);

    // Real top-opportunity detail: action, ROI multiple, and revenue impact.
    expect(q).toContain('Expand specialty coverage in the Northeast');
    expect(q).toMatch(/4\.0×|4\.0x/);
    expect(q).toMatch(/\$2\.4M/);

    // Real gap detail.
    expect(q).toContain('Northeast');
    expect(q).toMatch(/15(\.0)?%/);

    // Real bucket mix.
    expect(q).toMatch(/1 quick win/i);
    expect(q).toMatch(/1 steady play/i);
    expect(q).toMatch(/1 strategic bet/i);

    // Suppressed transparency.
    expect(q).toMatch(/2 .*below break-even|2 low-value|2 .*suppress/i);
  });

  it('does NOT fabricate numbers — falls back to a basic prompt when no context is available', () => {
    const q = buildExecutiveBriefQuery('Fabhalta', undefined);
    expect(q).toContain('Fabhalta');
    expect(q.toLowerCase()).toContain('executive brief');
    // No invented dollar figures when there is no real data.
    expect(q).not.toMatch(/\$\d/);
  });

  it('grounds the query on the suppressed count even when none are surfaced (all below break-even)', () => {
    // Live finding: Fabhalta surfaces 0 opportunities but suppressed 2
    // money-losers. The brief must convey that honestly (mirrors the T6
    // gap-analyzer narrative fix) rather than fall back to a generic prompt.
    const q = buildExecutiveBriefQuery('Fabhalta', makeContext({
      opportunities: [],
      total_count: 0,
      quick_wins_count: 0,
      steady_plays_count: 0,
      strategic_bets_count: 0,
      suppressed_count: 2,
      total_addressable_value: 0,
    }));
    expect(q).toContain('Fabhalta');
    expect(q).toMatch(/2 low-value/i);
    expect(q).toMatch(/below break-even/i);
    // No fabricated top-opportunity detail when none were surfaced.
    expect(q).not.toMatch(/Top opportunities by ROI:/);
  });

  it('falls back to the basic prompt when the opportunity list is empty (no real signal)', () => {
    const q = buildExecutiveBriefQuery('Remibrutinib', makeContext({
      opportunities: [],
      total_count: 0,
      quick_wins_count: 0,
      steady_plays_count: 0,
      strategic_bets_count: 0,
      suppressed_count: 0,
      total_addressable_value: 0,
    }));
    expect(q).toContain('Remibrutinib');
    expect(q).not.toMatch(/\$\d/);
  });

  it('summarizes up to the top three opportunities by rank', () => {
    const opps = [
      makeOpp({ rank: 1, recommended_action: 'Action one' }),
      makeOpp({ rank: 2, recommended_action: 'Action two' }),
      makeOpp({ rank: 3, recommended_action: 'Action three' }),
      makeOpp({ rank: 4, recommended_action: 'Action four' }),
    ];
    const q = buildExecutiveBriefQuery('Kisqali', makeContext({ opportunities: opps }));
    expect(q).toContain('Action one');
    expect(q).toContain('Action three');
    expect(q).not.toContain('Action four');
  });
});
