import { describe, it, expect } from 'vitest';

import type { PrioritizedOpportunity } from '@/types/gaps';
import { ImplementationDifficulty } from '@/types/gaps';

import {
  BUCKET_META,
  explainBucket,
  explainRank,
  explainTimeline,
  formatValueByDriver,
} from './interpret';

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
      estimated_revenue_impact: 500000,
      estimated_cost_to_close: 100000,
      expected_roi: 4,
      risk_adjusted_roi: 3,
      payback_period_months: 6,
      attribution_level: 'partial',
      attribution_rate: 0.65,
      confidence: 0.8,
    },
    recommended_action: 'Do the thing',
    implementation_difficulty: ImplementationDifficulty.MEDIUM,
    time_to_impact: '3-6 months',
    category: 'steady_play',
    ...overrides,
  };
}

describe('BUCKET_META', () => {
  it('covers all three buckets with a label, color, and blurb — no "other"', () => {
    expect(Object.keys(BUCKET_META).sort()).toEqual([
      'quick_win',
      'steady_play',
      'strategic_bet',
    ]);
    for (const meta of Object.values(BUCKET_META)) {
      expect(meta.label.length).toBeGreaterThan(0);
      expect(meta.color).toMatch(/^#/);
      expect(meta.blurb.length).toBeGreaterThan(0);
    }
  });
});

describe('explainBucket', () => {
  it('explains a quick win by low effort + ROI', () => {
    const s = explainBucket(makeOpp({ category: 'quick_win', implementation_difficulty: ImplementationDifficulty.LOW }));
    expect(s.toLowerCase()).toContain('quick win');
    expect(s).toContain('4.0×');
  });

  it('explains a steady play as the middle ground', () => {
    const s = explainBucket(makeOpp({ category: 'steady_play' }));
    expect(s.toLowerCase()).toContain('steady play');
    expect(s.toLowerCase()).toContain('middle');
  });

  it('explains a strategic bet as high effort, high impact', () => {
    const s = explainBucket(makeOpp({ category: 'strategic_bet', implementation_difficulty: ImplementationDifficulty.HIGH }));
    expect(s.toLowerCase()).toContain('strategic bet');
    expect(s.toLowerCase()).toContain('impact');
  });
});

describe('explainTimeline', () => {
  it('cites the time range and effort, and appends the difficulty rationale when present', () => {
    const s = explainTimeline(
      makeOpp({ time_to_impact: '6-12 months', difficulty_rationale: 'Rated high effort: high cost to close.' })
    );
    expect(s).toContain('6-12 months');
    expect(s.toLowerCase()).toContain('effort');
    expect(s).toContain('high cost to close');
  });

  it('still produces a sentence without a rationale', () => {
    const s = explainTimeline(makeOpp({ difficulty_rationale: undefined }));
    expect(s).toContain('3-6 months');
  });
});

describe('explainRank', () => {
  it('explains the rank by ROI ordering', () => {
    const s = explainRank(makeOpp({ rank: 2, roi_estimate: { ...makeOpp().roi_estimate, expected_roi: 4 } }));
    expect(s).toContain('#2');
    expect(s.toLowerCase()).toContain('roi');
  });

  it('notes off-label demotion when the opportunity is off-label', () => {
    const s = explainRank(
      makeOpp({ roi_estimate: { ...makeOpp().roi_estimate, off_label: true } })
    );
    expect(s.toLowerCase()).toContain('off-label');
    expect(s.toLowerCase()).toContain('demoted');
  });

  it('counts how many opportunities rank higher when given the full list', () => {
    const a = makeOpp({ rank: 3, roi_estimate: { ...makeOpp().roi_estimate, expected_roi: 2 } });
    const all = [
      makeOpp({ rank: 1, roi_estimate: { ...makeOpp().roi_estimate, expected_roi: 5 } }),
      makeOpp({ rank: 2, roi_estimate: { ...makeOpp().roi_estimate, expected_roi: 3 } }),
      a,
    ];
    const s = explainRank(a, all);
    expect(s).toContain('2');
  });
});

describe('formatValueByDriver', () => {
  it('returns [] for missing data', () => {
    expect(formatValueByDriver(undefined)).toEqual([]);
  });

  it('sorts drivers descending and humanizes keys', () => {
    const rows = formatValueByDriver({ trx_lift: 425000, patient_id: 75000 });
    expect(rows[0].key).toBe('trx_lift');
    expect(rows[0].label).toBe('TRx lift');
    expect(rows[0].value).toBe(425000);
    expect(rows[1].label.toLowerCase()).toContain('patient');
  });
});
