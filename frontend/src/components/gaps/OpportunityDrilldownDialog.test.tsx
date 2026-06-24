/**
 * OpportunityDrilldownDialog — shared "why" drill-down (T7).
 *
 * The drill-down that T6 built inline on the Gap-Analysis page is extracted
 * here so the AI-Insights Priority-Actions card can surface the SAME rich
 * explanation (why ranked, why this timeline, full ROI rationale) without
 * duplicating the rule-based logic. These tests pin the shared contract.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import type { PrioritizedOpportunity } from '@/types/gaps';
import { ImplementationDifficulty } from '@/types/gaps';
import { OpportunityDrilldownDialog } from './OpportunityDrilldownDialog';

function makeOpp(overrides: Partial<PrioritizedOpportunity> = {}): PrioritizedOpportunity {
  return {
    rank: 2,
    gap: {
      gap_id: 'g1',
      metric: 'trx',
      segment: 'specialty',
      segment_value: 'Rheumatology',
      current_value: 120,
      target_value: 180,
      gap_size: 60,
      gap_percentage: 33.3,
      gap_type: 'vs_target',
    },
    roi_estimate: {
      gap_id: 'g1',
      estimated_revenue_impact: 2_400_000,
      estimated_cost_to_close: 300_000,
      expected_roi: 4,
      risk_adjusted_roi: 3.2,
      payback_period_months: 6,
      attribution_level: 'partial',
      attribution_rate: 0.65,
      confidence: 0.82,
      total_risk_adjustment: 0.8,
      value_by_driver: { trx_lift: 1_800_000, patient_id: 600_000 },
      assumptions: ['Detailing reach holds at current levels'],
    },
    recommended_action: 'Expand specialty coverage in Rheumatology',
    implementation_difficulty: ImplementationDifficulty.MEDIUM,
    time_to_impact: '3-6 months',
    category: 'steady_play',
    difficulty_rationale: 'Medium effort: requires field-force realignment.',
    ...overrides,
  };
}

describe('OpportunityDrilldownDialog', () => {
  it('renders nothing when no opportunity is selected (closed)', () => {
    render(<OpportunityDrilldownDialog opp={null} allOpps={[]} onClose={vi.fn()} />);
    expect(screen.queryByText('Why this rank')).not.toBeInTheDocument();
    expect(screen.queryByText('ROI breakdown')).not.toBeInTheDocument();
  });

  it('surfaces the rank, category, action, and the rule-based "why" sections', () => {
    const opp = makeOpp();
    render(<OpportunityDrilldownDialog opp={opp} allOpps={[opp]} onClose={vi.fn()} />);

    // Header: rank + curated bucket label + recommended action.
    expect(screen.getByText('#2')).toBeInTheDocument();
    expect(screen.getByText('Steady Play')).toBeInTheDocument();
    expect(screen.getByText('Expand specialty coverage in Rheumatology')).toBeInTheDocument();

    // Rule-based explanations.
    expect(screen.getByText('Why this rank')).toBeInTheDocument();
    expect(screen.getByText('Why this timeline')).toBeInTheDocument();
    expect(screen.getByText('ROI breakdown')).toBeInTheDocument();
  });

  it('shows the full ROI rationale: figures, value-by-driver, and assumptions', () => {
    const opp = makeOpp();
    render(<OpportunityDrilldownDialog opp={opp} allOpps={[opp]} onClose={vi.fn()} />);

    // Formatted ROI figures.
    expect(screen.getByText('$2.4M')).toBeInTheDocument();
    expect(screen.getByText('4.0x')).toBeInTheDocument();

    // Value-by-driver breakdown (T6 field previously dropped before the FE).
    expect(screen.getByText('Revenue by value driver')).toBeInTheDocument();
    expect(screen.getByText('TRx lift')).toBeInTheDocument();

    // Assumptions list.
    expect(screen.getByText('Assumptions')).toBeInTheDocument();
    expect(
      screen.getByText('Detailing reach holds at current levels')
    ).toBeInTheDocument();

    // Gap detail (uniquely identified by the current-vs-target sentence — the
    // segment value also appears in the action title).
    expect(screen.getByText('Gap detail')).toBeInTheDocument();
    expect(screen.getByText(/Current 120 vs target/)).toBeInTheDocument();
  });
});
