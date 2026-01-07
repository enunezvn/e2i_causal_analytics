/**
 * Priority Actions by ROI Component
 * ==================================
 *
 * Displays ranked recommendations based on ROI potential.
 * Shows actionable insights prioritized by business impact.
 *
 * @module components/insights/PriorityActionsROI
 */

import { ArrowUpRight, DollarSign, Target, Users, Clock, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

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
}

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_ACTIONS: PriorityAction[] = [
  {
    id: 'action-1',
    title: 'Increase NE Region Call Frequency',
    description:
      'Causal analysis shows +1 call/month in NE region drives 2.3x TRx uplift for Remibrutinib.',
    estimatedROI: 2300000,
    effort: 'low',
    timeframe: '4-6 weeks',
    impactArea: 'TRx Volume',
    confidence: 0.91,
    icon: <Users className="h-4 w-4 text-blue-500" />,
  },
  {
    id: 'action-2',
    title: 'Optimize Formulary Access Strategy',
    description:
      'Gap analyzer identified 12 high-volume accounts with suboptimal formulary positioning.',
    estimatedROI: 1800000,
    effort: 'medium',
    timeframe: '8-12 weeks',
    impactArea: 'Market Access',
    confidence: 0.87,
    icon: <Target className="h-4 w-4 text-emerald-500" />,
  },
  {
    id: 'action-3',
    title: 'Deploy Targeted Digital Campaign',
    description:
      'Heterogeneous treatment effects suggest high-value segment for digital engagement.',
    estimatedROI: 950000,
    effort: 'low',
    timeframe: '2-3 weeks',
    impactArea: 'HCP Engagement',
    confidence: 0.82,
    icon: <TrendingUp className="h-4 w-4 text-purple-500" />,
  },
  {
    id: 'action-4',
    title: 'Address South Region Equity Gap',
    description:
      'DiD analysis shows 4.2pp fairness gap causing trust erosion and acceptance drop.',
    estimatedROI: 750000,
    effort: 'high',
    timeframe: '12-16 weeks',
    impactArea: 'Regional Equity',
    confidence: 0.78,
    icon: <DollarSign className="h-4 w-4 text-amber-500" />,
  },
];

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

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function ActionCard({ action, rank }: { action: PriorityAction; rank: number }) {
  const effortConfig = getEffortConfig(action.effort);

  return (
    <div className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] hover:border-[var(--color-primary)]/30 transition-colors">
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
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function PriorityActionsROI({ className }: PriorityActionsROIProps) {
  const totalROI = SAMPLE_ACTIONS.reduce((sum, a) => sum + a.estimatedROI, 0);

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
          <div className="text-right">
            <div className="text-xl font-bold text-emerald-600">{formatROI(totalROI)}</div>
            <div className="text-xs text-[var(--color-muted-foreground)]">Total Opportunity</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {SAMPLE_ACTIONS.map((action, idx) => (
          <ActionCard key={action.id} action={action} rank={idx + 1} />
        ))}

        {/* View All Button */}
        <Button variant="outline" className="w-full mt-2">
          <span>View All Recommendations</span>
          <ArrowUpRight className="h-4 w-4 ml-2" />
        </Button>
      </CardContent>
    </Card>
  );
}

export default PriorityActionsROI;
