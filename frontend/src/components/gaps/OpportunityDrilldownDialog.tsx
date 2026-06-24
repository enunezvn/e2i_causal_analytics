/**
 * OpportunityDrilldownDialog (T7)
 * ===============================
 *
 * The shared "why" drill-down for a prioritized gap opportunity: why it is
 * ranked where it is, why the timeline, and its full ROI rationale (figures,
 * value-by-driver breakdown, assumptions, gap detail, market landscape).
 *
 * T6 built this inline on the Gap-Analysis page. It is extracted here so the
 * AI-Insights "Priority Actions by ROI" card can open the IDENTICAL drawer
 * without duplicating the rule-based explanation logic — both surfaces share
 * one source of truth (this component + `lib/gaps/interpret`). All explanations
 * are pure/rule-based; nothing is fabricated.
 *
 * @module components/gaps/OpportunityDrilldownDialog
 */

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { CompetitorDensityBadge } from '@/components/insights/CompetitorDensityBadge';
import type { PrioritizedOpportunity } from '@/types/gaps';
import {
  bucketMeta,
  explainBucket,
  explainRank,
  explainTimeline,
  formatValueByDriver,
} from '@/lib/gaps/interpret';

// Default bucket when a (pre-T6) row carries no category.
const DEFAULT_CATEGORY = 'steady_play';

/** Compact USD formatting, matching the gap-analysis page ($2.4M / $300K). */
function formatCurrency(value: number): string {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

// Primary curated-category badge (Quick Win / Steady Play / Strategic Bet).
function getCategoryBadge(category?: string) {
  const meta = bucketMeta(category ?? DEFAULT_CATEGORY);
  const color = meta.color;
  return (
    <Badge style={{ backgroundColor: `${color}20`, color, borderColor: color }} variant="outline">
      {meta.label}
    </Badge>
  );
}

export interface OpportunityDrilldownDialogProps {
  /** The opportunity to explain; `null` keeps the dialog closed. */
  opp: PrioritizedOpportunity | null;
  /** The full list, so "why this rank" can count what ranks above it. */
  allOpps: PrioritizedOpportunity[];
  /** Called when the dialog requests to close. */
  onClose: () => void;
}

export function OpportunityDrilldownDialog({
  opp,
  allOpps,
  onClose,
}: OpportunityDrilldownDialogProps) {
  return (
    <Dialog open={!!opp} onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        {opp && (
          <>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 flex-wrap">
                <span className="text-primary">#{opp.rank}</span>
                {getCategoryBadge(opp.category)}
                <span className="text-base font-semibold">{opp.recommended_action}</span>
              </DialogTitle>
              <DialogDescription>{explainBucket(opp)}</DialogDescription>
            </DialogHeader>

            <div className="space-y-4 text-sm">
              <section>
                <h4 className="font-medium mb-1">Why this rank</h4>
                <p className="text-muted-foreground">{explainRank(opp, allOpps)}</p>
              </section>

              <section>
                <h4 className="font-medium mb-1">Why this timeline</h4>
                <p className="text-muted-foreground">{explainTimeline(opp)}</p>
              </section>

              <section>
                <h4 className="font-medium mb-1">ROI breakdown</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div>
                    <p className="text-muted-foreground text-xs">Revenue impact</p>
                    <p className="font-semibold text-emerald-600">
                      {formatCurrency(opp.roi_estimate.estimated_revenue_impact)}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Investment</p>
                    <p className="font-semibold text-amber-600">
                      {formatCurrency(opp.roi_estimate.estimated_cost_to_close)}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Expected ROI</p>
                    <p className="font-semibold text-primary">
                      {opp.roi_estimate.expected_roi.toFixed(1)}x
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Risk-adjusted ROI</p>
                    <p className="font-semibold">
                      {opp.roi_estimate.risk_adjusted_roi.toFixed(1)}x
                    </p>
                  </div>
                </div>
                {typeof opp.roi_estimate.total_risk_adjustment === 'number' && (
                  <p className="text-muted-foreground text-xs mt-2">
                    Risk adjustment retains{' '}
                    {(opp.roi_estimate.total_risk_adjustment * 100).toFixed(0)}% of the
                    unadjusted value.
                  </p>
                )}
              </section>

              {formatValueByDriver(opp.roi_estimate.value_by_driver).length > 0 && (
                <section>
                  <h4 className="font-medium mb-1">Revenue by value driver</h4>
                  <ul className="space-y-1">
                    {formatValueByDriver(opp.roi_estimate.value_by_driver).map((row) => (
                      <li key={row.key} className="flex justify-between text-muted-foreground">
                        <span>{row.label}</span>
                        <span className="font-medium">{formatCurrency(row.value)}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              {opp.roi_estimate.assumptions &&
                opp.roi_estimate.assumptions.length > 0 && (
                  <section>
                    <h4 className="font-medium mb-1">Assumptions</h4>
                    <ul className="list-disc list-inside text-muted-foreground space-y-0.5">
                      {opp.roi_estimate.assumptions.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </section>
                )}

              <section>
                <h4 className="font-medium mb-1">Gap detail</h4>
                <p className="text-muted-foreground">
                  {opp.gap.metric.toUpperCase()} — {opp.gap.segment}:{' '}
                  {opp.gap.segment_value}. Current {opp.gap.current_value} vs target{' '}
                  {opp.gap.target_value} ({opp.gap.gap_percentage.toFixed(1)}% gap).
                </p>
                <CompetitorDensityBadge
                  competitor_products_count={opp.roi_estimate.competitor_products_count}
                  competitor_density_label={opp.roi_estimate.competitor_density_label}
                  competitor_drug_names={opp.roi_estimate.competitor_drug_names}
                  className="mt-2"
                />
              </section>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default OpportunityDrilldownDialog;
