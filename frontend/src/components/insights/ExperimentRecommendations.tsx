/**
 * Experiment Health Monitor + Proposed Experiments Component
 * ===========================================================
 *
 * Two real feeds, never mixed:
 *
 * 1. The live health of currently running experiments from the monitoring
 *    sweep (enrollment, information fraction, SRM, open alerts), ranked
 *    worst-first with the sweep's own recommended actions. No twin score or
 *    lift estimate is fabricated for a monitoring row.
 * 2. Proposed experiments (#2206): completed digital-twin simulations whose
 *    recommendation is deploy or refine and which are not yet linked to an
 *    experiment — each with the twin's own predicted lift and interval,
 *    recommended sample size and duration, the model's fidelity state and the
 *    estimate's provenance. An admin can turn one into a linked `draft`
 *    experiment; it stays a draft until promoted. Nothing runs by itself.
 *
 * @module components/insights/ExperimentRecommendations
 */

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  FlaskConical,
  Users,
  AlertCircle,
  ListChecks,
  ArrowRight,
  Lightbulb,
  Clock,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useTriggerMonitoring } from '@/hooks/api';
import {
  useCreateDraftExperiment,
  useProposedExperiments,
} from '@/hooks/api/use-digital-twin';
import { useAuth } from '@/hooks/use-auth';
import { toast } from '@/hooks/use-toast';
import { EmptyState } from '@/components/ui/EmptyState';
import type { ExperimentHealthSummary } from '@/types/experiments';
import { FidelityStatus } from '@/types/digital-twin';
import { provenanceLabel } from '@/lib/digital-twin-provenance';
import type { ProposedExperimentItem } from '@/types/digital-twin';

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
const MAX_PROPOSALS = 6;

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

/** `digital_engagement` → `Digital engagement` */
function humanize(value: string): string {
  const spaced = value.replace(/_/g, ' ').trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * The twin's effect is an ABSOLUTE difference in the outcome column's units
 * (effect_scale 'absolute'); the outcome is unbounded and not a rate, so it is never
 * shown as a percentage (codex r1 #5).
 */
function formatEffect(ate: number, lower?: number | null, upper?: number | null): string {
  const abs = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(3)}`;
  if (lower == null || upper == null) return abs(ate);
  return `${abs(ate)} [${abs(lower)}, ${abs(upper)}]`;
}

function fidelityConfig(status: ProposedExperimentItem['fidelity_status']) {
  switch (status) {
    case FidelityStatus.VALIDATED:
      return {
        label: 'Model validated',
        className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
      };
    case FidelityStatus.BELOW_THRESHOLD:
      return {
        label: 'Model below threshold',
        className: 'bg-rose-500/10 text-rose-600 border-rose-500/20',
      };
    case FidelityStatus.UNVALIDATED:
    default:
      return {
        label: 'Model unvalidated',
        className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
      };
  }
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

      {/* Metrics Grid — real experiment-health fields */}
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

function ProposalRow({
  proposal,
  canDraft,
  isDrafting,
  draftedExperimentId,
  onDraft,
}: {
  proposal: ProposedExperimentItem;
  canDraft: boolean;
  isDrafting: boolean;
  draftedExperimentId?: string;
  onDraft: (proposal: ProposedExperimentItem) => void;
}) {
  const fidelity = fidelityConfig(proposal.fidelity_status);
  const isDeploy = proposal.recommendation === 'deploy';

  return (
    <div
      className="p-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]"
      data-testid={`proposal-${proposal.simulation_id}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="text-sm font-medium text-[var(--color-foreground)]">
              {proposal.brand} · {humanize(proposal.intervention_type)}
            </h4>
            <Badge
              variant="outline"
              className={cn(
                'text-xs',
                isDeploy
                  ? 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20'
                  : 'bg-amber-500/10 text-amber-600 border-amber-500/20'
              )}
            >
              {isDeploy ? 'Twin says deploy' : 'Twin says refine'}
            </Badge>
            <Badge variant="outline" className={cn('text-xs', fidelity.className)}>
              {fidelity.label}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
            Predicted effect{' '}
            <span className="font-medium text-[var(--color-foreground)]">
              {formatEffect(
                proposal.simulated_ate,
                proposal.simulated_ci_lower,
                proposal.simulated_ci_upper
              )}
            </span>{' '}
            on {proposal.outcome_column} (absolute, outcome units)
            {' · '}
            {proposal.recommended_sample_size != null
              ? `n=${proposal.recommended_sample_size.toLocaleString()}`
              : 'n not recommended'}
            {' · '}
            {proposal.recommended_duration_weeks != null
              ? `${proposal.recommended_duration_weeks} weeks`
              : 'duration not recommended'}
            {' · '}
            {provenanceLabel(proposal.data_provenance)}
          </p>
          {proposal.recommendation_rationale && (
            <p className="mt-1 text-xs text-[var(--color-muted-foreground)] line-clamp-2">
              {proposal.recommendation_rationale}
            </p>
          )}
        </div>
        {canDraft && !draftedExperimentId && (
          <Button
            size="sm"
            variant="outline"
            disabled={isDrafting}
            onClick={() => onDraft(proposal)}
            aria-label={`Create draft experiment from ${proposal.brand} ${humanize(proposal.intervention_type)}`}
          >
            {isDrafting ? 'Creating…' : 'Create draft experiment'}
          </Button>
        )}
      </div>
      {draftedExperimentId && (
        <p className="mt-2 text-xs text-emerald-600">
          Draft experiment <span className="font-mono">{draftedExperimentId}</span> created and
          linked. It stays a draft until promoted to running and enrolled.
        </p>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExperimentRecommendations({ className }: ExperimentRecommendationsProps) {
  const { data, isPending, mutate } = useTriggerMonitoring();
  const { isAdmin } = useAuth();
  const proposalsQuery = useProposedExperiments();
  const createDraft = useCreateDraftExperiment();
  // simulation_id -> experiment_id created in this session (the row leaves the
  // list on refetch; until then the row says what happened).
  const [drafted, setDrafted] = useState<Record<string, string>>({});
  // The feed shows a short list; "Show all" expands in place (there is no other page
  // that renders proposals, so a link elsewhere would lead nowhere — codex r1 #8).
  const [showAllProposals, setShowAllProposals] = useState(false);

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

  const proposals = proposalsQuery.data?.proposals ?? [];
  const shownProposals = showAllProposals ? proposals : proposals.slice(0, MAX_PROPOSALS);
  const outcomeMeasurable = proposalsQuery.data?.outcome_measurable_in_real_mode ?? false;
  const windowTruncated = proposalsQuery.data?.truncated ?? false;
  const outcomeColumn = proposalsQuery.data?.outcome_column;
  const totalProposed = proposalsQuery.data?.total_proposed ?? 0;
  const totalLinked = proposalsQuery.data?.total_linked ?? 0;
  const realRunning = proposalsQuery.data?.real_experiments_running ?? 0;
  const modelsState = (() => {
    if (proposals.length === 0) return null;
    const states = new Set(proposals.map((p) => p.fidelity_status));
    if (states.size === 1 && states.has(FidelityStatus.UNVALIDATED)) return 'models unvalidated';
    if (states.size === 1 && states.has(FidelityStatus.VALIDATED)) return 'models validated';
    return 'model fidelity mixed';
  })();

  const handleDraft = (proposal: ProposedExperimentItem) => {
    const confirmed = window.confirm(
      `Create a draft experiment for ${proposal.brand} · ${humanize(proposal.intervention_type)}?\n\n` +
        `It is written with status "draft" (n=${proposal.recommended_sample_size ?? '—'}, ` +
        `${proposal.recommended_duration_weeks ?? '—'} weeks) and linked to this simulation. ` +
        'Nothing runs until it is promoted.'
    );
    if (!confirmed) return;
    createDraft.mutate(proposal.simulation_id, {
      onSuccess: (created) => {
        setDrafted((prev) => ({ ...prev, [proposal.simulation_id]: created.experiment_id }));
        toast({
          title: 'Draft experiment created',
          description: `${created.experiment_name} (${created.experiment_id}) is linked to the simulation and stays a draft until promoted.`,
        });
      },
      onError: (error) => {
        toast({
          title: 'Could not create the draft experiment',
          description: error.message,
          variant: 'destructive',
        });
      },
    });
  };

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
                ) : null}. Proposed experiments below come from completed digital-twin simulations;
                nothing runs until a draft is promoted.
              </div>
            </div>
          </>
        )}

        {/* Proposed experiments (#2206) — a second real feed, never merged into the monitor list */}
        <div className="pt-3 border-t border-[var(--color-border)]" data-testid="proposed-experiments">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5 text-sm font-medium text-[var(--color-foreground)]">
              <Lightbulb className="h-4 w-4 text-purple-500" />
              Proposed experiments
            </div>
            {totalProposed > 0 && (
              <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-600">
                {totalProposed} proposed
              </Badge>
            )}
          </div>
          {proposalsQuery.isPending ? (
            <EmptyState title="Loading proposals…" />
          ) : proposalsQuery.isError ? (
            <EmptyState
              title="Couldn’t load proposed experiments"
              description={proposalsQuery.error?.message ?? 'The proposals request failed.'}
            />
          ) : proposals.length === 0 ? (
            <EmptyState
              title="No proposed experiments"
              description={
                totalLinked > 0
                  ? `All ${totalLinked} deploy/refine simulations are already linked to an experiment.`
                  : 'No completed digital-twin simulation recommends deploy or refine yet. Run one on the Digital Twin page.'
              }
            />
          ) : (
            <div className="space-y-2">
              <p className="text-xs text-[var(--color-muted-foreground)]" data-testid="proposals-envelope">
                {totalProposed} {totalProposed === 1 ? 'proposal' : 'proposals'} from twin simulations
                {' · '}
                {totalLinked} linked
                {' · '}
                {realRunning} real {realRunning === 1 ? 'experiment' : 'experiments'} running
                {modelsState ? ` · ${modelsState}` : ''}
                {windowTruncated
                  ? ` · showing the top ${proposals.length} of ${totalProposed} (deploy first, then predicted effect)`
                  : ''}
              </p>
              {shownProposals.map((proposal) => (
                <ProposalRow
                  key={proposal.simulation_id}
                  proposal={proposal}
                  canDraft={isAdmin}
                  isDrafting={
                    createDraft.isPending && createDraft.variables === proposal.simulation_id
                  }
                  draftedExperimentId={drafted[proposal.simulation_id]}
                  onDraft={handleDraft}
                />
              ))}
              {proposals.length > MAX_PROPOSALS && (
                <button
                  type="button"
                  onClick={() => setShowAllProposals((v) => !v)}
                  className="flex w-full items-center justify-center gap-1 p-2 rounded-lg border border-[var(--color-border)] text-xs font-medium text-purple-600 hover:bg-purple-500/5"
                >
                  {showAllProposals
                    ? `Show the top ${MAX_PROPOSALS} only`
                    : `Show all ${proposals.length} proposals`}
                  <ArrowRight className="h-3 w-3" />
                </button>
              )}
              {!outcomeMeasurable && (
                <div
                  className="flex items-start gap-2 p-2 rounded-lg bg-amber-500/5 border border-amber-500/20 text-xs text-[var(--color-muted-foreground)]"
                  data-testid="proposals-outcome-note"
                >
                  <AlertCircle className="h-3.5 w-3.5 mt-0.5 text-amber-600" />
                  <span>
                    The outcome these effects are stated on ({outcomeColumn ?? 'the twin outcome'}) is
                    recorded only on the synthetic-gold cohort rows today, so a real experiment drafted
                    from a proposal cannot yet be measured against the twin — the real endpoint is an
                    owner decision.
                  </span>
                </div>
              )}
              <div className="flex items-start gap-2 text-xs text-[var(--color-muted-foreground)]">
                <Clock className="h-3.5 w-3.5 mt-0.5 text-purple-500" />
                <span>
                  A draft is written with the twin’s recommended sample size and duration and linked
                  to its simulation. Promotion to running and enrollment stay manual; the daily sweep,
                  final analysis and fidelity roll-up then close the loop.
                </span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default ExperimentRecommendations;
