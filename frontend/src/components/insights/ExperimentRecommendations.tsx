/**
 * Experiment Health Monitor Component
 * ====================================
 *
 * Surfaces the live health of currently running experiments from the monitoring
 * sweep (enrollment, information fraction, SRM, open alerts), ranked worst-first
 * with the sweep's own recommended actions. Digital-twin pre-screening of
 * proposed experiments is not yet wired, so no twin score or lift estimate is
 * fabricated here — and no "Recommended/Simulated/Approved" pipeline states are
 * invented for what is purely a monitoring feed.
 *
 * @module components/insights/ExperimentRecommendations
 */

import { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FlaskConical, Users, AlertCircle, ListChecks, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { useTriggerMonitoring } from '@/hooks/api';
import { EmptyState } from '@/components/ui/EmptyState';
import type { ExperimentHealthSummary } from '@/types/experiments';

// =============================================================================
// TYPES
// =============================================================================

interface ExperimentRecommendationsProps {
  className?: string;
}

interface Experiment {
  id: string;
  title: string;
  /** Total enrolled to date (real). */
  enrolled: number;
  /** Current information fraction 0..1 from the sequential test (real).
   *  Null = the experiment carries no recorded enrollment plan, so progress
   *  is unknowable — the card omits the progress bar instead of showing 0%. */
  infoFraction: number | null;
  /** Open monitoring alerts (real). */
  openAlerts: number;
  /** Whether a sample-ratio mismatch was detected (real). */
  hasSrm: boolean;
  /** Live health from the monitoring sweep (real). */
  health: 'healthy' | 'warning' | 'critical' | 'unknown';
}

// Worst-first ranking for the card list; the card shows the top slice and
// links to /experiments for the full monitored set.
const HEALTH_RANK: Record<Experiment['health'], number> = {
  critical: 0,
  warning: 1,
  unknown: 2,
  healthy: 3,
};

const MAX_CARDS = 5;

// =============================================================================
// HELPERS
// =============================================================================

function getHealthConfig(health: Experiment['health']) {
  const config = {
    healthy: {
      label: 'Healthy',
      className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
    },
    warning: {
      label: 'Warning',
      className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
    },
    critical: {
      label: 'Critical',
      className: 'bg-rose-500/10 text-rose-600 border-rose-500/20',
    },
    unknown: {
      label: 'Unknown',
      className: 'bg-slate-500/10 text-slate-600 border-slate-500/20',
    },
  };
  return config[health];
}

/**
 * Map a live experiment health summary into the card's Experiment shape.
 * The live monitor returns experiment HEALTH (enrollment, information fraction,
 * SRM, open alerts) — it has no Digital-Twin prescreen score or expected-lift —
 * so we surface those real fields only. We never fabricate a twin score or a
 * lift estimate that the backend does not produce.
 */
function toExperimentCard(summary: ExperimentHealthSummary): Experiment {
  const health: Experiment['health'] =
    summary.health_status === 'critical' ||
    summary.health_status === 'warning' ||
    summary.health_status === 'healthy'
      ? summary.health_status
      : 'unknown';
  return {
    id: summary.experiment_id,
    title: summary.experiment_name,
    enrolled: summary.total_enrolled,
    infoFraction: summary.current_information_fraction ?? null,
    openAlerts: summary.active_alerts,
    hasSrm: summary.has_srm,
    health,
  };
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function ExperimentCard({ experiment }: { experiment: Experiment }) {
  const healthConfig = getHealthConfig(experiment.health);

  return (
    <div className="p-4 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
      {/* Header */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded bg-purple-500/10">
            <FlaskConical className="h-4 w-4 text-purple-500" />
          </div>
          <h4 className="text-sm font-medium text-[var(--color-foreground)]">
            {experiment.title}
          </h4>
        </div>
        <Badge variant="outline" className={cn('text-xs', healthConfig.className)}>
          {healthConfig.label}
        </Badge>
      </div>

      {/* Metrics Grid — real experiment-health fields (no Digital-Twin prescreen wired) */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Enrolled</div>
          <div className="text-lg font-bold text-[var(--color-foreground)]">
            {experiment.enrolled.toLocaleString()}
          </div>
        </div>
        <div className="p-2 rounded bg-[var(--color-muted)]/30">
          <div className="text-xs text-[var(--color-muted-foreground)]">Open Alerts</div>
          <div className={cn(
            'text-lg font-bold',
            experiment.openAlerts > 0 ? 'text-amber-600' : 'text-emerald-600'
          )}>
            {experiment.openAlerts}
          </div>
        </div>
      </div>

      {/* Information fraction — real sequential-test progress toward a decision.
          Omitted (not zeroed) when the experiment has no recorded plan. */}
      {experiment.infoFraction != null && (
        <div className="mb-3">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-[var(--color-muted-foreground)]">Information fraction</span>
            <span className="font-medium">{(experiment.infoFraction * 100).toFixed(0)}%</span>
          </div>
          <Progress value={experiment.infoFraction * 100} className="h-1.5" />
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center gap-3 pt-3 border-t border-[var(--color-border)] text-xs text-[var(--color-muted-foreground)]">
        <div className="flex items-center gap-1">
          <Users className="h-3 w-3" />
          <span>n={experiment.enrolled.toLocaleString()}</span>
        </div>
        {experiment.hasSrm && (
          <span className="text-rose-600 font-medium">SRM detected</span>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExperimentRecommendations({ className }: ExperimentRecommendationsProps) {
  const { data, isPending, mutate } = useTriggerMonitoring();

  // Trigger a one-shot monitoring sweep on mount — there is no GET-list
  // endpoint; the monitor endpoint returns the live experiment summaries.
  useEffect(() => {
    mutate({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const allExperiments = (data?.experiments ?? [])
    .map(toExperimentCard)
    .sort(
      (a, b) => HEALTH_RANK[a.health] - HEALTH_RANK[b.health] || b.openAlerts - a.openAlerts
    );
  const experiments = allExperiments.slice(0, MAX_CARDS);
  // Sweep-level recommended actions from the monitor agent (real, not invented).
  const recommendedActions = data?.recommended_actions ?? [];
  // Non-empty when the monitor agent hit errors (e.g. a node could not reach the
  // DB). Distinguishes a backend failure from a genuine empty dataset so the
  // panel does NOT show a misleading "no experiments" state over a crash.
  const monitorErrors = data?.errors ?? [];
  // Provenance honesty (#894): drive the synthetic-substrate note off
  // forced-OR-included (goldstd experiments are deliberately is_synthetic=False,
  // so per-row flags under-report; the deployment flag is the reliable signal).
  const syntheticSubstrate =
    (data?.synthetic_data_forced ?? false) || (data?.synthetic_data_included ?? false);

  return (
    <Card className={cn('bg-[var(--color-card)] border-[var(--color-border)]', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <FlaskConical className="h-5 w-5 text-purple-500" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">Experiment Health Monitor</CardTitle>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                Live health of running experiments
              </p>
            </div>
          </div>
          {allExperiments.length > 0 && (
            <Badge variant="outline" className="text-xs bg-amber-500/10 text-amber-600">
              {allExperiments.length} Monitored
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {isPending ? (
          <EmptyState title="Loading experiments…" />
        ) : allExperiments.length === 0 && monitorErrors.length > 0 ? (
          <EmptyState
            title="Couldn’t load experiments"
            description={`The monitoring service reported ${monitorErrors.length} error(s): ${monitorErrors
              .slice(0, 2)
              .join('; ')}`}
          />
        ) : allExperiments.length === 0 ? (
          <EmptyState
            title="No running experiments"
            description="Once experiments are running, the monitoring sweep will surface their health here."
          />
        ) : (
          <>
            {experiments.map((experiment) => (
              <ExperimentCard key={experiment.id} experiment={experiment} />
            ))}
            {allExperiments.length > MAX_CARDS && (
              <Link
                to="/experiments"
                className="flex items-center justify-center gap-1 p-2 rounded-lg border border-[var(--color-border)] text-xs font-medium text-purple-600 hover:bg-purple-500/5"
              >
                View all {allExperiments.length} monitored experiments
                <ArrowRight className="h-3 w-3" />
              </Link>
            )}
            {recommendedActions.length > 0 && (
              <div className="p-3 rounded-lg bg-[var(--color-muted)]/30 border border-[var(--color-border)]">
                <div className="flex items-center gap-1.5 mb-2 text-xs font-medium text-[var(--color-foreground)]">
                  <ListChecks className="h-3.5 w-3.5 text-purple-500" />
                  Recommended actions
                </div>
                <ul className="space-y-1 text-xs text-[var(--color-muted-foreground)] list-disc list-inside">
                  {recommendedActions.slice(0, 5).map((action) => (
                    <li key={action}>{action}</li>
                  ))}
                </ul>
              </div>
            )}
            <div className="flex items-start gap-2 p-3 rounded-lg bg-purple-500/5 border border-purple-500/20">
              <AlertCircle className="h-4 w-4 text-purple-500 mt-0.5" />
              <div className="text-xs text-[var(--color-muted-foreground)]">
                <span className="font-medium text-purple-600">Live experiment monitoring:</span> these
                are health summaries of currently running experiments (enrollment, information fraction,
                SRM and open alerts){syntheticSubstrate ? (
                  <>
                    {' '}on a <span className="font-medium">synthetic-gold substrate</span> — freshness
                    and enrollment alerts reflect the seeded dataset, not a live feed
                  </>
                ) : null}. Digital-twin pre-screening of proposed experiments is not yet wired.
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default ExperimentRecommendations;
