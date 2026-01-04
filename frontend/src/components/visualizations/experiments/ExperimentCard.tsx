/**
 * ExperimentCard Component
 * ========================
 *
 * Displays A/B test experiment information in a card format.
 * Shows experiment status, metrics, treatment/control groups, and results.
 */

import * as React from 'react';
import {
  Beaker,
  TrendingUp,
  TrendingDown,
  Users,
  Calendar,
  BarChart3,
  CheckCircle,
  Clock,
  AlertTriangle,
  XCircle,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export type ExperimentStatus = 'draft' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface ExperimentVariant {
  name: string;
  sample_size: number;
  conversion_rate?: number;
  mean_outcome?: number;
}

export interface ExperimentResult {
  lift_percentage: number;
  confidence_level: number;
  p_value: number;
  is_significant: boolean;
  winner?: 'treatment' | 'control' | 'none';
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  status: ExperimentStatus;
  start_date: string;
  end_date?: string;
  treatment: ExperimentVariant;
  control: ExperimentVariant;
  primary_metric: string;
  result?: ExperimentResult;
  brand?: string;
  tags?: string[];
}

interface ExperimentCardProps {
  experiment: Experiment;
  onClick?: (experiment: Experiment) => void;
  className?: string;
  compact?: boolean;
}

// =============================================================================
// STATUS CONFIGURATION
// =============================================================================

const statusConfig: Record<ExperimentStatus, { label: string; icon: React.ElementType; bgClass: string; textClass: string }> = {
  draft: {
    label: 'Draft',
    icon: Clock,
    bgClass: 'bg-gray-100 dark:bg-gray-800',
    textClass: 'text-gray-700 dark:text-gray-300',
  },
  running: {
    label: 'Running',
    icon: Beaker,
    bgClass: 'bg-blue-100 dark:bg-blue-900/30',
    textClass: 'text-blue-700 dark:text-blue-300',
  },
  paused: {
    label: 'Paused',
    icon: AlertTriangle,
    bgClass: 'bg-yellow-100 dark:bg-yellow-900/30',
    textClass: 'text-yellow-700 dark:text-yellow-300',
  },
  completed: {
    label: 'Completed',
    icon: CheckCircle,
    bgClass: 'bg-green-100 dark:bg-green-900/30',
    textClass: 'text-green-700 dark:text-green-300',
  },
  failed: {
    label: 'Failed',
    icon: XCircle,
    bgClass: 'bg-red-100 dark:bg-red-900/30',
    textClass: 'text-red-700 dark:text-red-300',
  },
  cancelled: {
    label: 'Cancelled',
    icon: XCircle,
    bgClass: 'bg-gray-100 dark:bg-gray-800',
    textClass: 'text-gray-500 dark:text-gray-400',
  },
};

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function StatusBadge({ status }: { status: ExperimentStatus }) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium',
        config.bgClass,
        config.textClass
      )}
    >
      <Icon className="h-3 w-3" />
      {config.label}
    </span>
  );
}

function ResultIndicator({ result }: { result: ExperimentResult }) {
  const isPositive = result.lift_percentage > 0;
  const TrendIcon = isPositive ? TrendingUp : TrendingDown;
  const colorClass = result.is_significant
    ? isPositive
      ? 'text-green-600 dark:text-green-400'
      : 'text-red-600 dark:text-red-400'
    : 'text-gray-500 dark:text-gray-400';

  return (
    <div className={cn('flex items-center gap-2', colorClass)}>
      <TrendIcon className="h-4 w-4" />
      <span className="font-semibold">
        {isPositive ? '+' : ''}{result.lift_percentage.toFixed(1)}%
      </span>
      {result.is_significant && (
        <span className="text-xs bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 px-1.5 py-0.5 rounded">
          Significant
        </span>
      )}
    </div>
  );
}

function VariantStat({ label, value, subvalue }: { label: string; value: string | number; subvalue?: string }) {
  return (
    <div className="text-center">
      <p className="text-xs text-[var(--color-text-tertiary)]">{label}</p>
      <p className="text-sm font-semibold text-[var(--color-text-primary)]">{value}</p>
      {subvalue && <p className="text-xs text-[var(--color-text-secondary)]">{subvalue}</p>}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function ExperimentCard({ experiment, onClick, className, compact = false }: ExperimentCardProps) {
  const { treatment, control, result } = experiment;
  const totalSampleSize = treatment.sample_size + control.sample_size;

  const handleClick = () => {
    if (onClick) onClick(experiment);
  };

  return (
    <div
      onClick={handleClick}
      className={cn(
        'bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-4 transition-all',
        onClick && 'cursor-pointer hover:border-[var(--color-primary)] hover:shadow-md',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Beaker className="h-4 w-4 text-[var(--color-primary)]" />
            <h3 className="font-medium text-[var(--color-text-primary)] truncate">{experiment.name}</h3>
          </div>
          {!compact && experiment.description && (
            <p className="text-xs text-[var(--color-text-secondary)] line-clamp-2">{experiment.description}</p>
          )}
        </div>
        <StatusBadge status={experiment.status} />
      </div>

      {/* Tags and Brand */}
      {!compact && (experiment.brand || experiment.tags?.length) && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {experiment.brand && (
            <span className="px-2 py-0.5 bg-[var(--color-primary)]/10 text-[var(--color-primary)] text-xs rounded-full">
              {experiment.brand}
            </span>
          )}
          {experiment.tags?.map((tag) => (
            <span key={tag} className="px-2 py-0.5 bg-[var(--color-border)] text-[var(--color-text-secondary)] text-xs rounded-full">
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-3">
        <div className="space-y-2">
          <div className="flex items-center gap-1 text-xs text-[var(--color-text-tertiary)]">
            <Users className="h-3 w-3" />
            Total Sample
          </div>
          <p className="text-lg font-bold text-[var(--color-text-primary)]">{totalSampleSize.toLocaleString()}</p>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-1 text-xs text-[var(--color-text-tertiary)]">
            <BarChart3 className="h-3 w-3" />
            Primary Metric
          </div>
          <p className="text-sm font-medium text-[var(--color-text-primary)] truncate">{experiment.primary_metric}</p>
        </div>
      </div>

      {/* Variant Comparison */}
      {!compact && (
        <div className="grid grid-cols-2 gap-4 p-3 bg-[var(--color-background)] rounded-lg mb-3">
          <div className="text-center border-r border-[var(--color-border)]">
            <p className="text-xs font-medium text-blue-600 dark:text-blue-400 mb-2">Treatment</p>
            <VariantStat
              label="Sample Size"
              value={treatment.sample_size.toLocaleString()}
              subvalue={treatment.conversion_rate ? `${(treatment.conversion_rate * 100).toFixed(1)}% conv.` : undefined}
            />
          </div>
          <div className="text-center">
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">Control</p>
            <VariantStat
              label="Sample Size"
              value={control.sample_size.toLocaleString()}
              subvalue={control.conversion_rate ? `${(control.conversion_rate * 100).toFixed(1)}% conv.` : undefined}
            />
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="flex items-center justify-between pt-3 border-t border-[var(--color-border)]">
          <ResultIndicator result={result} />
          <span className="text-xs text-[var(--color-text-tertiary)]">
            {result.confidence_level}% confidence
          </span>
        </div>
      )}

      {/* Date Range */}
      <div className="flex items-center gap-2 mt-3 pt-3 border-t border-[var(--color-border)] text-xs text-[var(--color-text-tertiary)]">
        <Calendar className="h-3 w-3" />
        <span>{new Date(experiment.start_date).toLocaleDateString()}</span>
        {experiment.end_date && (
          <>
            <span>-</span>
            <span>{new Date(experiment.end_date).toLocaleDateString()}</span>
          </>
        )}
      </div>
    </div>
  );
}

// Export compact list item variant
export function ExperimentListItem({ experiment, onClick }: { experiment: Experiment; onClick?: (e: Experiment) => void }) {
  return (
    <div
      onClick={() => onClick?.(experiment)}
      className={cn(
        'flex items-center justify-between p-3 bg-[var(--color-card)] rounded-lg border border-[var(--color-border)]',
        onClick && 'cursor-pointer hover:border-[var(--color-primary)] transition-colors'
      )}
    >
      <div className="flex items-center gap-3">
        <Beaker className="h-4 w-4 text-[var(--color-primary)]" />
        <div>
          <p className="text-sm font-medium text-[var(--color-text-primary)]">{experiment.name}</p>
          <p className="text-xs text-[var(--color-text-tertiary)]">{experiment.primary_metric}</p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        {experiment.result && <ResultIndicator result={experiment.result} />}
        <StatusBadge status={experiment.status} />
      </div>
    </div>
  );
}

export default ExperimentCard;
