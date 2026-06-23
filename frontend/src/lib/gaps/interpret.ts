/**
 * Gap-opportunity interpretation helpers (T6).
 *
 * Pure, rule-based functions that turn a prioritized opportunity into
 * human-readable "why" explanations for the drill-down drawer: why it's in its
 * bucket, why it's ranked where it is, why the timeline, and how its ROI breaks
 * down by value driver. No network, no fabrication — mirrors the
 * model-performance/interpret.ts pattern.
 *
 * @module lib/gaps/interpret
 */

import type { OpportunityCategory, PrioritizedOpportunity } from '@/types/gaps';

export interface BucketMeta {
  /** Display label, e.g. "Steady Play". */
  label: string;
  /** Brand color for badges/charts. */
  color: string;
  /** One-line description of what the bucket means. */
  blurb: string;
}

/**
 * Presentation + meaning for each of the three buckets. The TOTAL partition (no
 * residual "other") is the single source for labels, colors, and blurbs across
 * the page.
 */
export const BUCKET_META: Record<OpportunityCategory, BucketMeta> = {
  quick_win: {
    label: 'Quick Win',
    color: '#10b981', // green — low effort, fast profitable return
    blurb: 'Low implementation effort and a solid return — capture it fast.',
  },
  steady_play: {
    label: 'Steady Play',
    color: '#3b82f6', // blue — the dependable middle ground
    blurb:
      'The dependable middle ground: a solid, worthwhile earner that is neither a quick win nor a big strategic bet.',
  },
  strategic_bet: {
    label: 'Strategic Bet',
    color: '#8b5cf6', // purple — high effort/cost, high impact
    blurb: 'High effort and a significant investment, but high impact — a deliberate, larger bet.',
  },
};

/** Resolve bucket meta with a safe fallback (treat unknown as steady_play). */
export function bucketMeta(category?: string): BucketMeta {
  if (category && category in BUCKET_META) {
    return BUCKET_META[category as OpportunityCategory];
  }
  return BUCKET_META.steady_play;
}

function roiX(opp: PrioritizedOpportunity): string {
  return `${opp.roi_estimate.expected_roi.toFixed(1)}×`;
}

/** Why this opportunity sits in its bucket. */
export function explainBucket(opp: PrioritizedOpportunity): string {
  const roi = roiX(opp);
  const diff = opp.implementation_difficulty;
  switch (opp.category) {
    case 'quick_win':
      return `Quick Win — low implementation effort with a strong ${roi} ROI. Capture it fast for an early return.`;
    case 'strategic_bet':
      return `Strategic Bet — high effort and a significant investment, but a ${roi} ROI makes it a deliberate, high-impact play.`;
    case 'steady_play':
    default:
      return `Steady Play — the dependable middle ground (${diff} effort, ${roi} ROI): neither a quick win nor a big strategic bet, but a solid, worthwhile earner.`;
  }
}

/** Why this time-to-impact — difficulty drives the range; cite the rationale. */
export function explainTimeline(opp: PrioritizedOpportunity): string {
  const base = `Estimated ${opp.time_to_impact} to see results — driven by ${opp.implementation_difficulty} implementation effort.`;
  return opp.difficulty_rationale ? `${base} ${opp.difficulty_rationale}` : base;
}

/**
 * Why this rank — opportunities are ordered by expected ROI (highest first),
 * and off-label opportunities are demoted below all on-label ones. When the full
 * list is supplied, also state how many opportunities rank above this one.
 */
export function explainRank(
  opp: PrioritizedOpportunity,
  allOpps?: PrioritizedOpportunity[]
): string {
  const roi = opp.roi_estimate.expected_roi;
  const off = opp.roi_estimate.off_label === true;
  let s = `Ranked #${opp.rank}: opportunities are ordered by expected ROI (highest first), and this one returns ${roi.toFixed(1)}× its cost`;

  if (allOpps && allOpps.length > 1) {
    // Count what actually ranks ABOVE this one, honoring the off-label demotion
    // (on-label always outranks off-label; within a partition, higher ROI wins).
    // A plain ROI count would mislead for an off-label opportunity.
    const ranksAbove = (o: PrioritizedOpportunity): boolean => {
      if (o === opp) return false;
      const oOff = o.roi_estimate.off_label === true;
      if (oOff !== off) return !oOff;
      return o.roi_estimate.expected_roi > roi;
    };
    const higher = allOpps.filter(ranksAbove).length;
    if (higher > 0) {
      s += ` — ${higher} ${higher === 1 ? 'opportunity ranks' : 'opportunities rank'} higher`;
    } else {
      s += ' — the top-ranked opportunity in this view';
    }
  }

  if (off) {
    s +=
      '. It is off-label (its segment falls outside the FDA-indicated population), so it is demoted below all on-label opportunities';
  }
  return `${s}.`;
}

const DRIVER_LABELS: Record<string, string> = {
  trx_lift: 'TRx lift',
  patient_id: 'Patient identification',
  patient_identification: 'Patient identification',
  action_rate: 'Action rate',
  intent_to_prescribe: 'Intent to prescribe',
  data_quality: 'Data quality',
  drift_prevention: 'Drift prevention',
};

export interface DriverRow {
  key: string;
  label: string;
  value: number;
}

/**
 * Turn the ROI value_by_driver map into a sorted, human-labeled breakdown.
 * Drops non-finite values; returns [] when absent.
 */
export function formatValueByDriver(vbd?: Record<string, number>): DriverRow[] {
  if (!vbd) return [];
  return Object.entries(vbd)
    .filter(([, v]) => Number.isFinite(v))
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => ({
      key,
      label: DRIVER_LABELS[key] ?? key.replace(/_/g, ' '),
      value,
    }));
}
