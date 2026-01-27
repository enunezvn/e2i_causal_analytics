/**
 * Agent Tier Badge Component
 * ==========================
 *
 * Badge component for displaying E2I agent tier levels.
 * Each tier represents a different level of agent sophistication.
 *
 * @module components/visualizations/agents/AgentTierBadge
 */

import * as React from 'react';
import {
  Cpu,
  Brain,
  GitBranch,
  Activity,
  LineChart,
  Lightbulb,
  Layers,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// =============================================================================
// TYPES
// =============================================================================

export type AgentTier = 0 | 1 | 2 | 3 | 4 | 5;

export interface AgentTierBadgeProps {
  /** Agent tier level (0-5) */
  tier: AgentTier;
  /** Whether to show the tier label */
  showLabel?: boolean;
  /** Whether to show tier description on hover */
  showTooltip?: boolean;
  /** Badge size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// TIER CONFIGURATION
// =============================================================================

interface TierConfig {
  name: string;
  description: string;
  icon: React.ElementType;
  color: string;
  bgColor: string;
  borderColor: string;
  agents: string[];
}

const TIER_CONFIG: Record<AgentTier, TierConfig> = {
  0: {
    name: 'Foundation',
    description: 'ML Foundation agents for data preparation and model training',
    icon: Cpu,
    color: 'text-slate-600',
    bgColor: 'bg-slate-100 dark:bg-slate-800',
    borderColor: 'border-slate-300 dark:border-slate-600',
    agents: ['Scope Definer', 'Cohort Constructor', 'Data Preparer', 'Feature Analyzer', 'Model Selector', 'Model Trainer', 'Model Deployer', 'Observability Connector'],
  },
  1: {
    name: 'Orchestration',
    description: 'Coordination agents for routing and multi-tool orchestration',
    icon: GitBranch,
    color: 'text-violet-600',
    bgColor: 'bg-violet-100 dark:bg-violet-900/30',
    borderColor: 'border-violet-300 dark:border-violet-700',
    agents: ['Orchestrator', 'Tool Composer'],
  },
  2: {
    name: 'Causal',
    description: 'Causal inference and impact analysis agents',
    icon: Brain,
    color: 'text-emerald-600',
    bgColor: 'bg-emerald-100 dark:bg-emerald-900/30',
    borderColor: 'border-emerald-300 dark:border-emerald-700',
    agents: ['Causal Impact', 'Gap Analyzer', 'Heterogeneous Optimizer'],
  },
  3: {
    name: 'Monitoring',
    description: 'System monitoring and experiment design agents',
    icon: Activity,
    color: 'text-blue-600',
    bgColor: 'bg-blue-100 dark:bg-blue-900/30',
    borderColor: 'border-blue-300 dark:border-blue-700',
    agents: ['Drift Monitor', 'Experiment Designer', 'Health Score'],
  },
  4: {
    name: 'ML Predictions',
    description: 'Machine learning prediction and optimization agents',
    icon: LineChart,
    color: 'text-amber-600',
    bgColor: 'bg-amber-100 dark:bg-amber-900/30',
    borderColor: 'border-amber-300 dark:border-amber-700',
    agents: ['Prediction Synthesizer', 'Resource Optimizer'],
  },
  5: {
    name: 'Self-Improvement',
    description: 'Explainability and continuous learning agents',
    icon: Lightbulb,
    color: 'text-rose-600',
    bgColor: 'bg-rose-100 dark:bg-rose-900/30',
    borderColor: 'border-rose-300 dark:border-rose-700',
    agents: ['Explainer', 'Feedback Learner'],
  },
};

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * AgentTierBadge displays the tier level of an E2I agent.
 *
 * @example
 * ```tsx
 * <AgentTierBadge tier={2} showLabel />
 * <AgentTierBadge tier={5} size="lg" showTooltip />
 * ```
 */
export const AgentTierBadge = React.forwardRef<HTMLSpanElement, AgentTierBadgeProps>(
  (
    {
      tier,
      showLabel = true,
      showTooltip = true,
      size = 'md',
      className,
    },
    ref
  ) => {
    const config = TIER_CONFIG[tier];
    const Icon = config.icon;

    // Size styles
    const sizeStyles = {
      sm: {
        badge: 'px-1.5 py-0.5 text-xs gap-1',
        icon: 'h-3 w-3',
      },
      md: {
        badge: 'px-2 py-1 text-sm gap-1.5',
        icon: 'h-3.5 w-3.5',
      },
      lg: {
        badge: 'px-3 py-1.5 text-base gap-2',
        icon: 'h-4 w-4',
      },
    }[size];

    const badge = (
      <span
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full font-medium border',
          config.bgColor,
          config.borderColor,
          config.color,
          sizeStyles.badge,
          className
        )}
      >
        <Icon className={sizeStyles.icon} />
        {showLabel && (
          <>
            <span>Tier {tier}</span>
            <span className="opacity-70">•</span>
            <span>{config.name}</span>
          </>
        )}
      </span>
    );

    if (!showTooltip) return badge;

    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{badge}</TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <div className="space-y-2">
              <p className="font-medium">
                Tier {tier}: {config.name}
              </p>
              <p className="text-sm text-[var(--color-muted-foreground)]">
                {config.description}
              </p>
              <div className="text-xs">
                <span className="font-medium">Agents: </span>
                <span className="text-[var(--color-muted-foreground)]">
                  {config.agents.join(', ')}
                </span>
              </div>
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }
);

AgentTierBadge.displayName = 'AgentTierBadge';

// =============================================================================
// TIER OVERVIEW
// =============================================================================

export interface TierOverviewProps {
  /** Currently active/selected tier */
  activeTier?: AgentTier;
  /** Callback when a tier is selected */
  onTierSelect?: (tier: AgentTier) => void;
  /** Whether to show in compact mode */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * TierOverview displays all agent tiers in a visual hierarchy.
 *
 * @example
 * ```tsx
 * <TierOverview
 *   activeTier={2}
 *   onTierSelect={(tier) => setSelectedTier(tier)}
 * />
 * ```
 */
export const TierOverview = React.forwardRef<HTMLDivElement, TierOverviewProps>(
  ({ activeTier, onTierSelect, compact = false, className }, ref) => {
    const tiers = Object.keys(TIER_CONFIG).map(Number) as AgentTier[];

    if (compact) {
      return (
        <div ref={ref} className={cn('flex items-center gap-2', className)}>
          {tiers.map((tier) => {
            const config = TIER_CONFIG[tier];
            const isActive = tier === activeTier;
            return (
              <button
                key={tier}
                onClick={() => onTierSelect?.(tier)}
                className={cn(
                  'w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all',
                  isActive
                    ? cn(config.bgColor, config.borderColor, config.color, 'ring-2 ring-offset-2')
                    : 'bg-[var(--color-muted)] border-transparent text-[var(--color-muted-foreground)] hover:border-[var(--color-border)]'
                )}
              >
                {tier}
              </button>
            );
          })}
        </div>
      );
    }

    return (
      <div ref={ref} className={cn('space-y-2', className)}>
        {tiers.map((tier) => {
          const config = TIER_CONFIG[tier];
          const Icon = config.icon;
          const isActive = tier === activeTier;

          return (
            <button
              key={tier}
              onClick={() => onTierSelect?.(tier)}
              className={cn(
                'w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left',
                isActive
                  ? cn(config.bgColor, config.borderColor, 'ring-2 ring-offset-2')
                  : 'bg-[var(--color-card)] border-[var(--color-border)] hover:border-[var(--color-border)]'
              )}
            >
              <div className={cn('p-2 rounded-lg', config.bgColor)}>
                <Icon className={cn('h-5 w-5', config.color)} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={cn('font-semibold', isActive ? config.color : '')}>
                    Tier {tier}
                  </span>
                  <span className={cn('text-sm', isActive ? config.color : 'text-[var(--color-muted-foreground)]')}>
                    {config.name}
                  </span>
                </div>
                <p className="text-xs text-[var(--color-muted-foreground)] truncate">
                  {config.agents.length} agents
                </p>
              </div>
              <Layers className="h-4 w-4 text-[var(--color-muted-foreground)]" />
            </button>
          );
        })}
      </div>
    );
  }
);

TierOverview.displayName = 'TierOverview';

export default AgentTierBadge;
