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
import { useCausalChains } from '@/hooks/api/use-graph';
import type { GraphPath, GraphNode, CausalChainRequest } from '@/types/graph';

// =============================================================================
// TYPES
// =============================================================================

interface ChainCardData {
  id: string;
  title: string;
  status: 'active' | 'high-roi' | 'in-progress' | 'monitored';
  nodes: string[];
  result: string;
  confidence: number;
  method: string;
  timestamp: string;
  impact: 'high' | 'medium' | 'low';
  icon: React.ReactNode;
}

interface CausalValueChainsProps {
  className?: string;
}

// =============================================================================
// SAMPLE DATA (Fallback when API unavailable)
// =============================================================================

const SAMPLE_CHAINS: ChainCardData[] = [
  {
    id: 'chain-1',
    title: 'Data Quality Impact',
    status: 'active',
    nodes: ['Source Integration', 'Validation Score', 'Data Completeness'],
    result: '+12% TRx Accuracy',
    confidence: 0.92,
    method: 'DoWhy',
    timestamp: '2 min ago',
    impact: 'high',
    icon: <BarChart3 className="h-4 w-4 text-blue-500" />,
  },
  {
    id: 'chain-2',
    title: 'HCP Engagement Path',
    status: 'high-roi',
    nodes: ['Call Frequency', 'Detailing Quality', 'Rx Propensity'],
    result: '+8.5% Conversion',
    confidence: 0.87,
    method: 'EconML',
    timestamp: '5 min ago',
    impact: 'high',
    icon: <Users className="h-4 w-4 text-emerald-500" />,
  },
  {
    id: 'chain-3',
    title: 'Coverage Equity Chain',
    status: 'in-progress',
    nodes: ['Regional Access', 'Formulary Status', 'Prior Auth'],
    result: '-4.2pp Gap',
    confidence: 0.78,
    method: 'CausalForest',
    timestamp: '12 min ago',
    impact: 'medium',
    icon: <TrendingUp className="h-4 w-4 text-amber-500" />,
  },
];

// =============================================================================
// HELPERS
// =============================================================================

function getStatusConfig(status: ChainCardData['status']) {
  const config = {
    active: {
      label: 'ACTIVE',
      className: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20',
    },
    'high-roi': {
      label: 'HIGH ROI',
      className: 'bg-amber-500/10 text-amber-600 border-amber-500/20',
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

function getImpactConfig(impact: ChainCardData['impact']) {
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
  const confidence = path.total_confidence ?? 0.8;

  // Determine status based on confidence
  let status: ChainCardData['status'] = 'monitored';
  if (confidence >= 0.9) status = 'active';
  else if (confidence >= 0.8) status = 'high-roi';
  else if (confidence >= 0.7) status = 'in-progress';

  // Determine impact based on path length and confidence
  let impact: ChainCardData['impact'] = 'low';
  if (confidence >= 0.85 && path.path_length >= 3) impact = 'high';
  else if (confidence >= 0.7) impact = 'medium';

  // Get method from relationship properties if available
  const method =
    path.relationships.length > 0
      ? (path.relationships[0].properties?.method as string) ?? 'DoWhy'
      : 'DoWhy';

  // Create title from first and last node
  const title =
    nodeNames.length >= 2
      ? `${nodeNames[0]} → ${nodeNames[nodeNames.length - 1]}`
      : 'Causal Chain';

  // Generate result text
  const resultValue = lastNode?.properties?.value as number | undefined;
  const result = resultValue
    ? `${resultValue > 0 ? '+' : ''}${resultValue.toFixed(1)}% Impact`
    : '+X% Impact';

  return {
    id: `chain-${index}`,
    title: title.length > 30 ? title.substring(0, 27) + '...' : title,
    status,
    nodes: nodeNames.slice(0, -1), // All but last (result)
    result,
    confidence,
    method,
    timestamp: 'Just now',
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
  const impactConfig = getImpactConfig(chain.impact);

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
          <div className="px-2 py-1 rounded bg-emerald-500/10 text-emerald-600 text-xs font-semibold whitespace-nowrap">
            {chain.result}
          </div>
        </div>

        {/* Metadata Row */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--color-border)]">
          <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
            <div className="flex items-center gap-1">
              <Zap className="h-3 w-3" />
              <span>{(chain.confidence * 100).toFixed(0)}% confidence</span>
            </div>
            <div className="flex items-center gap-1">
              <GitBranch className="h-3 w-3" />
              <span>{chain.method}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn('text-xs font-medium', impactConfig.className)}>
              {impactConfig.label}
            </span>
            <span className="text-xs text-[var(--color-muted-foreground)]">
              {chain.timestamp}
            </span>
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
    isPending: isLoading,
  } = useCausalChains();

  // Fetch chains on mount
  useEffect(() => {
    const request: CausalChainRequest = {
      min_confidence: 0.5,
      max_chain_length: 5,
    };
    fetchChains(request);
  }, [fetchChains]);

  // Transform API data or use sample data
  const chains = useMemo((): ChainCardData[] => {
    if (chainsResponse?.chains && chainsResponse.chains.length > 0) {
      return chainsResponse.chains
        .slice(0, 3)
        .map((path, idx) => transformGraphPathToCard(path, idx));
    }
    return SAMPLE_CHAINS;
  }, [chainsResponse]);

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

      {/* Chain Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {chains.map((chain) => (
          <ChainCard key={chain.id} chain={chain} />
        ))}
      </div>
    </div>
  );
}

export default CausalValueChains;
