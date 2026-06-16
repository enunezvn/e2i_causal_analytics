/**
 * Executive Intelligence Summary Component
 * =========================================
 *
 * Displays a live system summary for the E2I dashboard, built EXCLUSIVELY
 * from real API substrate:
 *
 * - Graph metrics ............ useGraphStats() (relationships, nodes,
 *                              communities, episodes)
 * - System health ............ useQuickHealthCheck() (Health Score agent's
 *                              real overall_health_score + grade)
 * - Agent roster ............. GET /agents/status (real active/total counts)
 *
 * Anything without live data renders an honest placeholder ('—') or is
 * omitted. The former fabrications — numeric fallbacks (142/847/12/1.47M),
 * hardcoded activeAgents=8, a healthScore invented from a KPI status string
 * (84/72), an undocumented dollar-impact formula (total_relationships *
 * 0.167 shown as "$X.XM Est. Impact"), and three hardcoded "causal insight"
 * cards ported from the static mock design (commit fbf84df7) — were removed:
 * they presented plausible-wrong values as real-time analysis.
 *
 * @module components/dashboard/ExecutiveSummary
 */

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, Brain, Target, CheckCircle2, Users } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useGraphStats } from '@/hooks/api/use-graph';
import { useFullHealthCheck } from '@/hooks/api/use-health-score';
import { getValidated } from '@/lib/api-client';
import { AgentStatusResponseSchema } from '@/lib/api-schemas';

// =============================================================================
// TYPES
// =============================================================================

interface ExecutiveSummaryProps {
  className?: string;
}

// =============================================================================
// HELPERS
// =============================================================================

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

interface QuickStat {
  label: string;
  /** Pre-formatted display value; '—' is the honest "no data" placeholder. */
  display: string;
  icon: React.ReactNode;
  iconBg: string;
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExecutiveSummary({ className }: ExecutiveSummaryProps) {
  // Real graph statistics (FalkorDB-backed).
  const {
    data: graphStats,
    isLoading: graphLoading,
    error: graphError,
  } = useGraphStats();

  // Real system health from the Health Score agent (same source as the Home
  // System Health card) — NOT a number invented from a KPI status string. Uses
  // the FULL (all-dimension) check, not the component-only quick check whose
  // overall score is a misleading component-only 100/A.
  const { data: health, error: healthError } = useFullHealthCheck({
    refetchInterval: 60000,
  });

  // Real agent roster (same source as the Home Agent Status card).
  const { data: agentStatus, error: agentsError } = useQuery({
    queryKey: ['agent-status'],
    queryFn: () => getValidated(AgentStatusResponseSchema, '/agents/status'),
    refetchInterval: 30000,
    retry: false,
  });

  // Failed sources get a LABELED degraded notice — a query error must be
  // distinguishable from honest "no data yet" ('—').
  const failedSources = [
    ...(graphError ? ['graph metrics'] : []),
    ...(healthError ? ['health score'] : []),
    ...(agentsError ? ['agent roster'] : []),
  ];

  const totalAgents = agentStatus?.agents?.length ?? null;
  const activeAgents = useMemo(
    () =>
      agentStatus
        ? agentStatus.agents.filter((a) => a.status === 'active').length
        : null,
    [agentStatus]
  );
  const healthScore =
    health?.overall_health_score != null
      ? Math.round(health.overall_health_score)
      : null;

  // Quick stats — real values only; '—' when the source has no data.
  const quickStats: QuickStat[] = [
    {
      label: 'Causal Paths',
      display:
        graphStats?.total_relationships != null
          ? formatNumber(graphStats.total_relationships)
          : '—',
      icon: <Brain className="h-3.5 w-3.5 text-purple-500" />,
      iconBg: 'bg-purple-500/10',
    },
    {
      label: 'Graph Nodes',
      display:
        graphStats?.total_nodes != null
          ? formatNumber(graphStats.total_nodes)
          : '—',
      icon: <Target className="h-3.5 w-3.5 text-blue-500" />,
      iconBg: 'bg-blue-500/10',
    },
    {
      label: 'System Health',
      display: healthScore != null ? `${healthScore}%` : '—',
      icon: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />,
      iconBg: 'bg-emerald-500/10',
    },
    {
      label: 'Agents Active',
      display:
        activeAgents != null && totalAgents != null
          ? `${activeAgents}/${totalAgents}`
          : '—',
      icon: <Users className="h-3.5 w-3.5 text-amber-500" />,
      iconBg: 'bg-amber-500/10',
    },
  ];

  // System-status prose assembled ONLY from clauses with real data behind
  // them; honest fallback sentence when nothing is available yet.
  const statusClauses = useMemo(() => {
    const clauses: string[] = [];
    if (healthScore != null) {
      clauses.push(
        `The platform health score is ${healthScore}%${
          health?.health_grade ? ` (grade ${health.health_grade})` : ''
        }.`
      );
    }
    if (activeAgents != null && totalAgents != null) {
      clauses.push(`${activeAgents} of ${totalAgents} AI agents are active.`);
    }
    if (graphStats) {
      const parts: string[] = [];
      if (graphStats.total_relationships != null) {
        parts.push(`${formatNumber(graphStats.total_relationships)} causal relationships`);
      }
      if (graphStats.total_nodes != null) {
        parts.push(`${formatNumber(graphStats.total_nodes)} nodes`);
      }
      if (graphStats.total_communities != null) {
        parts.push(`${formatNumber(graphStats.total_communities)} communities`);
      }
      if (graphStats.total_episodes != null) {
        parts.push(`${formatNumber(graphStats.total_episodes)} episodes`);
      }
      if (parts.length > 0) {
        clauses.push(`The causal knowledge graph tracks ${parts.join(', ')}.`);
      }
    }
    return clauses;
  }, [healthScore, health?.health_grade, activeAgents, totalAgents, graphStats]);

  if (graphLoading) {
    return (
      <div className={cn('space-y-4', className)}>
        <div className="h-32 bg-[var(--color-muted)] animate-pulse rounded-lg" />
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Main Summary Card */}
      <Card className="bg-gradient-to-br from-[var(--color-card)] to-[var(--color-muted)]/30 border-[var(--color-border)]">
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-purple-500" />
            <CardTitle className="text-lg">
              Executive Intelligence Summary
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-[var(--color-muted-foreground)]">
              <Activity className="h-4 w-4" />
              <span className="font-medium">Current System Status</span>
            </div>
            <p className="text-sm leading-relaxed text-[var(--color-foreground)]">
              {statusClauses.length > 0
                ? statusClauses.join(' ')
                : 'Live system metrics are currently unavailable.'}
            </p>

            {failedSources.length > 0 && (
              <p role="alert" className="text-xs text-amber-600 dark:text-amber-400">
                Degraded: {failedSources.join(', ')}{' '}
                {failedSources.length === 1 ? 'request' : 'requests'} failed —
                affected figures show '—'.
              </p>
            )}

            {/* Quick Stats Row — real values or honest '—'. */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
              {quickStats.map((stat) => (
                <div key={stat.label} className="flex items-center gap-2 text-sm">
                  <div className={cn('p-1.5 rounded', stat.iconBg)}>{stat.icon}</div>
                  <div>
                    <div className="font-medium">{stat.display}</div>
                    <div className="text-xs text-[var(--color-muted-foreground)]">
                      {stat.label}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default ExecutiveSummary;
