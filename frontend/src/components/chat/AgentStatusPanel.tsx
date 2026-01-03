/**
 * Agent Status Panel Component
 * ============================
 *
 * Displays the status of all 18 E2I agents organized by tier.
 * Shows real-time status indicators and capabilities.
 *
 * @module components/chat/AgentStatusPanel
 */

import * as React from 'react';
import {
  Bot,
  Zap,
  CircleDot,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import type { AgentInfo } from '@/providers/E2ICopilotProvider';

// =============================================================================
// TYPES
// =============================================================================

export interface AgentStatusPanelProps {
  /** List of agents to display */
  agents: AgentInfo[];
  /** Compact mode for sidebar */
  compact?: boolean;
  /** Show capabilities */
  showCapabilities?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

const TIER_INFO: Record<number, { label: string; color: string; bgColor: string }> = {
  0: { label: 'ML Foundation', color: 'text-slate-600', bgColor: 'bg-slate-100 dark:bg-slate-800' },
  1: { label: 'Orchestration', color: 'text-purple-600', bgColor: 'bg-purple-100 dark:bg-purple-900/30' },
  2: { label: 'Causal Analytics', color: 'text-blue-600', bgColor: 'bg-blue-100 dark:bg-blue-900/30' },
  3: { label: 'Monitoring', color: 'text-amber-600', bgColor: 'bg-amber-100 dark:bg-amber-900/30' },
  4: { label: 'ML Predictions', color: 'text-emerald-600', bgColor: 'bg-emerald-100 dark:bg-emerald-900/30' },
  5: { label: 'Self-Improvement', color: 'text-rose-600', bgColor: 'bg-rose-100 dark:bg-rose-900/30' },
};

function getStatusConfig(status: AgentInfo['status']) {
  switch (status) {
    case 'active':
      return {
        icon: Zap,
        color: 'text-emerald-500',
        bgColor: 'bg-emerald-500',
        label: 'Active',
      };
    case 'processing':
      return {
        icon: Loader2,
        color: 'text-blue-500',
        bgColor: 'bg-blue-500',
        label: 'Processing',
        animate: true,
      };
    case 'error':
      return {
        icon: AlertCircle,
        color: 'text-rose-500',
        bgColor: 'bg-rose-500',
        label: 'Error',
      };
    case 'idle':
    default:
      return {
        icon: CircleDot,
        color: 'text-slate-400',
        bgColor: 'bg-slate-400',
        label: 'Idle',
      };
  }
}

function groupAgentsByTier(agents: AgentInfo[]): Map<number, AgentInfo[]> {
  const grouped = new Map<number, AgentInfo[]>();

  agents.forEach((agent) => {
    const tier = agent.tier;
    if (!grouped.has(tier)) {
      grouped.set(tier, []);
    }
    grouped.get(tier)!.push(agent);
  });

  return grouped;
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

interface AgentItemProps {
  agent: AgentInfo;
  compact?: boolean;
  showCapabilities?: boolean;
}

function AgentItem({ agent, compact = false, showCapabilities = false }: AgentItemProps) {
  const statusConfig = getStatusConfig(agent.status);
  const StatusIcon = statusConfig.icon;

  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-lg transition-colors',
        compact ? 'p-1.5' : 'p-2',
        agent.status === 'active' && 'bg-emerald-50 dark:bg-emerald-900/10',
        agent.status === 'processing' && 'bg-blue-50 dark:bg-blue-900/10',
        agent.status === 'error' && 'bg-rose-50 dark:bg-rose-900/10'
      )}
    >
      {/* Status indicator */}
      <div className="relative">
        <StatusIcon
          className={cn(
            'h-3.5 w-3.5',
            statusConfig.color,
            statusConfig.animate && 'animate-spin'
          )}
        />
        {(agent.status === 'active' || agent.status === 'processing') && (
          <span
            className={cn(
              'absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full',
              statusConfig.bgColor,
              'animate-pulse'
            )}
          />
        )}
      </div>

      {/* Agent info */}
      <div className="flex-1 min-w-0">
        <p className={cn('font-medium truncate', compact ? 'text-xs' : 'text-sm')}>
          {agent.name}
        </p>
        {showCapabilities && !compact && agent.capabilities.length > 0 && (
          <p className="text-xs text-muted-foreground truncate">
            {agent.capabilities.slice(0, 2).join(', ')}
          </p>
        )}
      </div>

      {/* Status badge */}
      {!compact && (
        <Badge
          variant="outline"
          className={cn('text-[10px] px-1.5 py-0', statusConfig.color)}
        >
          {statusConfig.label}
        </Badge>
      )}
    </div>
  );
}

interface TierGroupProps {
  tier: number;
  agents: AgentInfo[];
  compact?: boolean;
  showCapabilities?: boolean;
  defaultExpanded?: boolean;
}

function TierGroup({
  tier,
  agents,
  compact = false,
  showCapabilities = false,
  defaultExpanded = true,
}: TierGroupProps) {
  const [expanded, setExpanded] = React.useState(defaultExpanded);
  const tierInfo = TIER_INFO[tier];
  const activeCount = agents.filter((a) => a.status === 'active' || a.status === 'processing').length;

  return (
    <div className="space-y-1">
      {/* Tier header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className={cn(
          'w-full flex items-center justify-between rounded-lg transition-colors',
          compact ? 'p-1.5' : 'p-2',
          tierInfo.bgColor
        )}
      >
        <div className="flex items-center gap-2">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
          )}
          <span className={cn('font-semibold', compact ? 'text-xs' : 'text-sm', tierInfo.color)}>
            Tier {tier}
          </span>
          {!compact && (
            <span className="text-xs text-muted-foreground">
              {tierInfo.label}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {activeCount > 0 && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0 bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
              {activeCount} active
            </Badge>
          )}
          <span className="text-xs text-muted-foreground">
            {agents.length}
          </span>
        </div>
      </button>

      {/* Agent list */}
      {expanded && (
        <div className={cn('space-y-0.5', compact ? 'pl-4' : 'pl-6')}>
          {agents.map((agent) => (
            <AgentItem
              key={agent.id}
              agent={agent}
              compact={compact}
              showCapabilities={showCapabilities}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * AgentStatusPanel displays the status of all E2I agents organized by tier.
 *
 * @example
 * ```tsx
 * <AgentStatusPanel agents={agents} compact showCapabilities />
 * ```
 */
export function AgentStatusPanel({
  agents,
  compact = false,
  showCapabilities = false,
  className,
}: AgentStatusPanelProps) {
  const groupedAgents = React.useMemo(() => groupAgentsByTier(agents), [agents]);
  const totalActive = agents.filter((a) => a.status === 'active' || a.status === 'processing').length;

  return (
    <div className={cn('p-3 space-y-2', className)}>
      {/* Summary header */}
      <div className="flex items-center justify-between pb-2 border-b">
        <div className="flex items-center gap-2">
          <Bot className={cn('text-muted-foreground', compact ? 'h-4 w-4' : 'h-5 w-5')} />
          <span className={cn('font-semibold', compact ? 'text-xs' : 'text-sm')}>
            Agent Status
          </span>
        </div>
        <Badge variant="outline" className="text-xs">
          {totalActive} / {agents.length} active
        </Badge>
      </div>

      {/* Tier groups */}
      <div className="space-y-2 max-h-[300px] overflow-y-auto">
        {Array.from(groupedAgents.entries())
          .sort(([a], [b]) => a - b)
          .map(([tier, tierAgents]) => (
            <TierGroup
              key={tier}
              tier={tier}
              agents={tierAgents}
              compact={compact}
              showCapabilities={showCapabilities}
              defaultExpanded={!compact}
            />
          ))}
      </div>
    </div>
  );
}

export default AgentStatusPanel;
