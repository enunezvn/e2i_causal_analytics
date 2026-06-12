/**
 * Primary Causal Value Chains Component
 * =====================================
 *
 * Displays live tracking of causal value chains for the E2I dashboard.
 * Shows interactive causal chain cards with impact indicators and
 * confidence scores.
 *
 * Features:
 * - Interactive causal chain cards
 * - Visual chain path representation
 * - Impact indicators (high/medium/low)
 * - Confidence scores and methods
 * - Real-time API integration via useCausalChains
 *
 * @module components/dashboard/CausalValueChains
 */

import { useEffect, useMemo } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  GitBranch,
  TrendingUp,
  Zap,
  BarChart3,
  Users,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { useCausalChains } from '@/hooks/api/use-graph';
import type { GraphPath, GraphNode, CausalChainRequest } from '@/types/graph';

// =============================================================================
// TYPES
// =============================================================================

interface ChainCardData {
  id: string;
  title: string;
  status: 'active' | 'in-progress' | 'monitored';
  nodes: string[];
  result: string;
  /** Combined path confidence from the API; null when not reported. */
  confidence: number | null;
  /** Causal method recorded on the relationship; null when not reported. */
  method: string | null;
  /** Derived impact band; null when confidence is unknown (no claim made). */
  impact: 'high' | 'medium' | 'low' | null;
  icon: React.ReactNode;
}

interface CausalValueChainsProps {
  className?: string;
}

// NOTE: the former SAMPLE_CHAINS fabricated fallback ("+12% TRx Accuracy",
// confidence 0.92, "DoWhy", "2 min ago" — ported from the static mock design
// in commit fbf84df7) was removed. It rendered unlabeled fake chains on BOTH
// an empty-but-successful response AND a mutation error. The only states now
// are: real data, honest empty (EmptyState), labeled error.

// =============================================================================
// HELPERS
// =============================================================================

function getStatusConfig(status: ChainCardData['status']) {
  const config = {
    active: {
      label: 'ACTIVE',
      className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
    },
    'in-progress': {
      label: 'IN PROGRESS',
      className: 'bg-blue-500/10 text-blue-600 border-blue-500/20',
    },
    monitored: {
      label: 'MONITORED',
      className: 'bg-slate-500/10 text-slate-600 border-slate-500/20',
    },
  };
  return config[status];
}

function getImpactConfig(impact: NonNullable<ChainCardData['impact']>) {
  const config = {
    high: { label: 'High Impact', className: 'text-emerald-600' },
    medium: { label: 'Medium Impact', className: 'text-amber-600' },
    low: { label: 'Low Impact', className: 'text-slate-500' },
  };
  return config[impact];
}

/**
 * Transform API GraphPath to ChainCardData
 */
function transformGraphPathToCard(
  path: GraphPath,
  index: number
): ChainCardData {
  const nodeNames = path.nodes.map((n: GraphNode) => n.name);
  const lastNode = path.nodes[path.nodes.length - 1];
  // Honest confidence: only what the API reports — never fabricate a default.
  const confidence = path.total_confidence ?? null;

  // Determine status from REPORTED confidence only; unknown stays 'monitored'.
  let status: ChainCardData['status'] = 'monitored';
  if (confidence != null) {
    if (confidence >= 0.9) status = 'active';
    else if (confidence >= 0.7) status = 'in-progress';
  }

  // Impact band only when confidence is reported; otherwise make no claim.
  let impact: ChainCardData['impact'] = null;
  if (confidence != null) {
    if (confidence >= 0.85 && path.path_length >= 3) impact = 'high';
    else if (confidence >= 0.7) impact = 'medium';
    else impact = 'low';
  }

  // Method only when recorded on the relationship — never default to 'DoWhy'.
  const method =
    path.relationships.length > 0
      ? ((path.relationships[0].properties?.method as string | undefined) ?? null)
      : null;

  // Create title from first and last node
  const title =
    nodeNames.length >= 2
      ? `${nodeNames[0]} → ${nodeNames[nodeNames.length - 1]}`
      : 'Causal Chain';

  // Result text from the real terminal-node value; honest when absent.
  // `!= null` (not truthiness): a quantified ZERO effect is real data.
  const resultValue = lastNode?.properties?.value as number | undefined;
  const result = resultValue != null
    ? `${resultValue > 0 ? '+' : ''}${resultValue.toFixed(1)}% Impact`
    : 'Impact not quantified';

  return {
    id: `chain-${index}`,
    title: title.length > 30 ? title.substring(0, 27) + '...' : title,
    status,
    nodes: nodeNames.slice(0, -1), // All but last (result)
    result,
    confidence,
    method,
    // NOTE: no timestamp — GraphPath carries none, and inventing a recency
    // claim ('Just now' / '2 min ago') would be silently-fake data.
    impact,
    icon:
      index === 0 ? (
        <BarChart3 className="h-4 w-4 text-blue-500" />
      ) : index === 1 ? (
        <Users className="h-4 w-4 text-emerald-500" />
      ) : (
        <TrendingUp className="h-4 w-4 text-amber-500" />
      ),
  };
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function ChainCard({ chain }: { chain: ChainCardData }) {
  const statusConfig = getStatusConfig(chain.status);
  const impactConfig = chain.impact != null ? getImpactConfig(chain.impact) : null;

  return (
    <Card className="bg-[var(--color-card)] border-[var(--color-border)] hover:border-[var(--color-primary)]/30 transition-colors">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-[var(--color-muted)]">
              {chain.icon}
            </div>
            <CardTitle className="text-sm font-medium">{chain.title}</CardTitle>
          </div>
          <Badge variant="outline" className={cn('text-xs', statusConfig.className)}>
            {statusConfig.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        {/* Chain Visualization */}
        <div className="flex items-center gap-1 py-3 overflow-x-auto">
          {chain.nodes.map((node, idx) => (
            <div key={idx} className="flex items-center">
              <div className="px-2 py-1 rounded bg-[var(--color-muted)] text-xs font-medium whitespace-nowrap">
                {node}
              </div>
              <ArrowRight className="h-3 w-3 mx-1 text-[var(--color-muted-foreground)] flex-shrink-0" />
            </div>
          ))}
          <div
            className={cn(
              'px-2 py-1 rounded text-xs font-semibold whitespace-nowrap',
              chain.result === 'Impact not quantified'
                ? 'bg-[var(--color-muted)] text-[var(--color-muted-foreground)]'
                : 'bg-emerald-500/10 text-emerald-600'
            )}
          >
            {chain.result}
          </div>
        </div>

        {/* Metadata Row */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--color-border)]">
          <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
            <div className="flex items-center gap-1">
              <Zap className="h-3 w-3" />
              <span>
                {chain.confidence != null
                  ? `${(chain.confidence * 100).toFixed(0)}% confidence`
                  : 'confidence unavailable'}
              </span>
            </div>
            {chain.method && (
              <div className="flex items-center gap-1">
                <GitBranch className="h-3 w-3" />
                <span>{chain.method}</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            {impactConfig && (
              <span className={cn('text-xs font-medium', impactConfig.className)}>
                {impactConfig.label}
              </span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function CausalValueChains({ className }: CausalValueChainsProps) {
  // Use mutation hook to fetch causal chains
  const {
    mutate: fetchChains,
    data: chainsResponse,
    isPending,
    isError,
  } = useCausalChains();

  // Fetch chains on mount
  useEffect(() => {
    const request: CausalChainRequest = {
      min_confidence: 0.5,
      max_chain_length: 5,
    };
    fetchChains(request);
  }, [fetchChains]);

  // Real API data ONLY — no sample fallback on empty or error.
  const chains = useMemo((): ChainCardData[] => {
    if (!chainsResponse?.chains) return [];
    return chainsResponse.chains
      .slice(0, 3)
      .map((path, idx) => transformGraphPathToCard(path, idx));
  }, [chainsResponse]);

  // Pending covers both the in-flight mutation and the first-mount frame
  // before the useEffect-triggered mutate() settles (no fabricated flash).
  const isLoading = isPending || (!chainsResponse && !isError);

  // Loading state
  if (isLoading) {
    return (
      <div className={cn('space-y-4', className)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-purple-500" />
            <h2 className="text-lg font-semibold">
              Primary Causal Value Chains - Live Tracking
            </h2>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-40 bg-[var(--color-muted)] animate-pulse rounded-lg"
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch className="h-5 w-5 text-purple-500" />
          <h2 className="text-lg font-semibold">
            Primary Causal Value Chains - Live Tracking
          </h2>
        </div>
        <div className="flex items-center gap-2">
          {chainsResponse?.total_chains !== undefined && (
            <Badge variant="secondary" className="text-xs">
              {chainsResponse.total_chains} chains discovered
            </Badge>
          )}
          {chainsResponse?.aggregate_effect !== undefined && (
            <Badge variant="outline" className="text-xs">
              {(chainsResponse.aggregate_effect * 100).toFixed(1)}% aggregate effect
            </Badge>
          )}
        </div>
      </div>

      {/* Error: clearly labeled degraded state — never unlabeled fakes. */}
      {isError ? (
        <div
          role="alert"
          className="flex items-center gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-400"
        >
          <AlertTriangle className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
          <span>
            Causal chains unavailable — the graph service request failed. Live
            chain data cannot be displayed.
          </span>
        </div>
      ) : chains.length === 0 ? (
        /* Honest empty state on a successful-but-empty response. */
        <EmptyState
          title="No causal chains discovered"
          description="The causal graph returned no value chains yet. Chains appear here once causal discovery has produced validated paths."
          icon={<GitBranch className="h-8 w-8" aria-hidden="true" />}
        />
      ) : (
        /* Chain Cards Grid — real API data only. */
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {chains.map((chain) => (
            <ChainCard key={chain.id} chain={chain} />
          ))}
        </div>
      )}
    </div>
  );
}

export default CausalValueChains;
