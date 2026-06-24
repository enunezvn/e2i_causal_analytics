/**
 * Priority Actions by ROI Component
 * ==================================
 *
 * Displays ranked recommendations based on ROI potential.
 * Shows actionable insights prioritized by business impact.
 *
 * @module components/insights/PriorityActionsROI
 */

import { useState } from 'react';
import { ArrowUpRight, DollarSign, Target, Users, Clock, TrendingUp } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useOpportunities } from '@/hooks/api';
import { EmptyState } from '@/components/ui/EmptyState';
import { OpportunityDrilldownDialog } from '@/components/gaps/OpportunityDrilldownDialog';
import type { PrioritizedOpportunity } from '@/types/gaps';

// =============================================================================
// TYPES
// =============================================================================

interface PriorityAction {
  id: string;
  title: string;
  description: string;
  estimatedROI: number;
  effort: 'low' | 'medium' | 'high';
  timeframe: string;
  impactArea: string;
  confidence: number;
  icon: React.ReactNode;
}

interface PriorityActionsROIProps {
  className?: string;
  /** Brand filter forwarded to the opportunities feed. */
  brand?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

function formatROI(value: number): string {
  if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `$${(value / 1000).toFixed(0)}K`;
  return `$${value}`;
}

function getEffortConfig(effort: PriorityAction['effort']) {
  const config = {
    low: { label: 'Low Effort', className: 'bg-emerald-500/10 text-emerald-600' },
    medium: { label: 'Medium Effort', className: 'bg-amber-500/10 text-amber-600' },
    high: { label: 'High Effort', className: 'bg-rose-500/10 text-rose-600' },
  };
  return config[effort];
}

const RANK_ICONS = [
  <Users className="h-4 w-4 text-blue-500" />,
  <Target className="h-4 w-4 text-emerald-500" />,
  <TrendingUp className="h-4 w-4 text-purple-500" />,
  <DollarSign className="h-4 w-4 text-amber-500" />,
];

/**
 * Map a live ROI-ranked opportunity into the card's PriorityAction shape.
 * `implementation_difficulty` is the ImplementationDifficulty string enum
 * ('low' | 'medium' | 'high'), matching the card's `effort` union; we narrow
 * defensively so an unexpected value falls back to 'medium' rather than
 * rendering an undefined effort badge.
 */
function toEffort(difficulty: string): PriorityAction['effort'] {
  return difficulty === 'low' || difficulty === 'high' ? difficulty : 'medium';
}

function toPriorityAction(opp: PrioritizedOpportunity): PriorityAction {
  return {
    id: opp.gap.gap_id,
    title: opp.recommended_action,
    description: `${opp.gap.metric} gap in ${opp.gap.segment_value} (${opp.gap.gap_percentage.toFixed(0)}% vs target).`,
    estimatedROI: opp.roi_estimate.estimated_revenue_impact,
    effort: toEffort(opp.implementation_difficulty),
    timeframe: opp.time_to_impact,
    impactArea: opp.gap.metric,
    confidence: opp.roi_estimate.confidence,
    icon: RANK_ICONS[(opp.rank - 1) % RANK_ICONS.length],
  };
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function ActionCard({
  action,
  rank,
  onSelect,
}: {
  action: PriorityAction;
  rank: number;
  /** Open the shared "why" drill-down for this opportunity. */
  onSelect: () => void;
}) {
  const effortConfig = getEffortConfig(action.effort);

  return (
    <button
      type="button"
      onClick={onSelect}
      className="w-full text-left p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] hover:border-[var(--color-primary)]/30 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-primary)]/40"
    >
      <div className="flex items-start gap-3">
        {/* Rank Badge */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-[var(--color-primary)]/10 flex items-center justify-center">
          <span className="text-sm font-bold text-[var(--color-primary)]">#{rank}</span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded bg-[var(--color-muted)]">{action.icon}</div>
              <h4 className="text-sm font-medium text-[var(--color-foreground)]">
                {action.title}
              </h4>
            </div>
            <div className="text-right flex-shrink-0">
              <div className="text-lg font-bold text-emerald-600">
                {formatROI(action.estimatedROI)}
              </div>
              <div className="text-xs text-[var(--color-muted-foreground)]">Est. ROI</div>
            </div>
          </div>

          <p className="text-sm text-[var(--color-muted-foreground)] mb-3">
            {action.description}
          </p>

          {/* Metadata Row */}
          <div className="flex items-center gap-3 flex-wrap">
            <Badge variant="outline" className={cn('text-xs', effortConfig.className)}>
              {effortConfig.label}
            </Badge>
            <div className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
              <Clock className="h-3 w-3" />
              <span>{action.timeframe}</span>
            </div>
            <div className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]">
              <Target className="h-3 w-3" />
              <span>{action.impactArea}</span>
            </div>
          </div>

          {/* Confidence Bar */}
          <div className="mt-3 flex items-center gap-2">
            <span className="text-xs text-[var(--color-muted-foreground)]">Confidence:</span>
            <Progress value={action.confidence * 100} className="h-1.5 flex-1" />
            <span className="text-xs font-medium">{(action.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </button>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function PriorityActionsROI({ className, brand }: PriorityActionsROIProps) {
  const { data, isLoading, isError, error } = useOpportunities({ brand, limit: 4 });
  // Keep the full opportunities alongside the lossy display shape so the
  // drill-down can surface the complete ROI rationale (T7b).
  const opportunities = data?.opportunities ?? [];
  const actions = opportunities.map(toPriorityAction);
  const totalROI = data?.total_addressable_value ?? 0;
  const navigate = useNavigate();
  // The opportunity whose "why" drawer is open (null = closed).
  const [drillOpp, setDrillOpp] = useState<PrioritizedOpportunity | null>(null);

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-emerald-500/10">
              <DollarSign className="h-5 w-5 text-emerald-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Priority Actions by ROI</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Ranked recommendations based on causal analysis
              </p>
            </div>
          </div>
          {actions.length > 0 && (
            <div className="text-right">
              <div className="text-xl font-bold text-emerald-600">{formatROI(totalROI)}</div>
              <div className="text-xs text-[var(--color-muted-foreground)]">Total Opportunity</div>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {isLoading ? (
          <EmptyState title="Loading opportunities…" />
        ) : isError ? (
          <EmptyState
            title="Could not load opportunities"
            description={error?.message ?? 'The opportunities feed returned an error.'}
          />
        ) : actions.length === 0 ? (
          <EmptyState
            title="No prioritized opportunities"
            description="Run a gap analysis to surface ROI-ranked recommendations."
          />
        ) : (
          <>
            {actions.map((action, idx) => (
              <ActionCard
                key={action.id}
                action={action}
                rank={idx + 1}
                onSelect={() => setDrillOpp(opportunities[idx])}
              />
            ))}
            <Button
              variant="outline"
              className="w-full mt-2"
              onClick={() => navigate('/gap-analysis')}
            >
              <span>View All Recommendations</span>
              <ArrowUpRight className="h-4 w-4 ml-2" />
            </Button>
          </>
        )}
      </CardContent>

      {/* Shared "why" drill-down — identical to the Gap-Analysis page (T7b). */}
      <OpportunityDrilldownDialog
        opp={drillOpp}
        allOpps={opportunities}
        onClose={() => setDrillOpp(null)}
      />
    </Card>
  );
}

export default PriorityActionsROI;
