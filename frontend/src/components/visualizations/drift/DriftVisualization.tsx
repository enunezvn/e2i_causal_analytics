/**
 * DriftVisualization Component
 * ============================
 *
 * Visualizes model and data drift metrics over time.
 * Includes trend lines, threshold indicators, and alert status.
 */

import * as React from 'react';
import { useState } from 'react';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
  BarChart3,
  ArrowRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// =============================================================================
// TYPES
// =============================================================================

export type DriftType = 'data' | 'model' | 'concept' | 'feature';
export type DriftSeverity = 'none' | 'low' | 'medium' | 'high' | 'critical';

export interface DriftDataPoint {
  timestamp: string;
  value: number;
  threshold?: number;
}

export interface DriftMetric {
  name: string;
  type: DriftType;
  current_value: number;
  threshold: number;
  severity: DriftSeverity;
  trend: 'increasing' | 'decreasing' | 'stable';
  history: DriftDataPoint[];
  description?: string;
  last_updated: string;
}

export interface DriftSummary {
  total_features: number;
  drifting_features: number;
  model_drift_detected: boolean;
  data_drift_percentage: number;
  last_check: string;
  alerts: Array<{ feature: string; severity: DriftSeverity; message: string }>;
}

interface DriftVisualizationProps {
  metrics: DriftMetric[];
  summary?: DriftSummary;
  className?: string;
  showHistory?: boolean;
  onMetricClick?: (metric: DriftMetric) => void;
}

// =============================================================================
// SEVERITY CONFIGURATION
// =============================================================================

const severityConfig: Record<DriftSeverity, { label: string; color: string; bgColor: string; borderColor: string }> = {
  none: {
    label: 'No Drift',
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-100 dark:bg-green-900/30',
    borderColor: 'border-green-200 dark:border-green-800',
  },
  low: {
    label: 'Low',
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-100 dark:bg-blue-900/30',
    borderColor: 'border-blue-200 dark:border-blue-800',
  },
  medium: {
    label: 'Medium',
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-100 dark:bg-yellow-900/30',
    borderColor: 'border-yellow-200 dark:border-yellow-800',
  },
  high: {
    label: 'High',
    color: 'text-orange-600 dark:text-orange-400',
    bgColor: 'bg-orange-100 dark:bg-orange-900/30',
    borderColor: 'border-orange-200 dark:border-orange-800',
  },
  critical: {
    label: 'Critical',
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-100 dark:bg-red-900/30',
    borderColor: 'border-red-200 dark:border-red-800',
  },
};

const driftTypeIcons: Record<DriftType, React.ElementType> = {
  data: BarChart3,
  model: Activity,
  concept: TrendingUp,
  feature: ArrowRight,
};

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

function SeverityBadge({ severity }: { severity: DriftSeverity }) {
  const config = severityConfig[severity];
  const Icon = severity === 'none' ? CheckCircle : severity === 'critical' || severity === 'high' ? AlertTriangle : Clock;

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium',
        config.bgColor,
        config.color
      )}
    >
      <Icon className="h-3 w-3" />
      {config.label}
    </span>
  );
}

function TrendIndicator({ trend, value }: { trend: 'increasing' | 'decreasing' | 'stable'; value: number }) {
  const TrendIcon = trend === 'increasing' ? TrendingUp : trend === 'decreasing' ? TrendingDown : Activity;
  const colorClass = trend === 'increasing' ? 'text-red-500' : trend === 'decreasing' ? 'text-green-500' : 'text-gray-500';

  return (
    <div className={cn('flex items-center gap-1', colorClass)}>
      <TrendIcon className="h-4 w-4" />
      <span className="text-sm font-medium">{value.toFixed(3)}</span>
    </div>
  );
}

function MiniSparkline({ data, threshold }: { data: DriftDataPoint[]; threshold: number }) {
  if (data.length < 2) return null;

  const values = data.map((d) => d.value);
  const max = Math.max(...values, threshold * 1.2);
  const min = Math.min(...values, 0);
  const range = max - min || 1;

  const width = 120;
  const height = 32;
  const padding = 2;

  const points = data
    .map((d, i) => {
      const x = padding + (i / (data.length - 1)) * (width - padding * 2);
      const y = height - padding - ((d.value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(' ');

  const thresholdY = height - padding - ((threshold - min) / range) * (height - padding * 2);

  return (
    <svg width={width} height={height} className="flex-shrink-0">
      {/* Threshold line */}
      <line
        x1={padding}
        y1={thresholdY}
        x2={width - padding}
        y2={thresholdY}
        stroke="var(--color-destructive)"
        strokeDasharray="4 2"
        strokeWidth="1"
        opacity="0.5"
      />
      {/* Data line */}
      <polyline
        points={points}
        fill="none"
        stroke="var(--color-primary)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Latest point */}
      {data.length > 0 && (
        <circle
          cx={width - padding}
          cy={height - padding - ((data[data.length - 1].value - min) / range) * (height - padding * 2)}
          r="3"
          fill="var(--color-primary)"
        />
      )}
    </svg>
  );
}

function DriftMetricCard({ metric, onClick, showHistory }: { metric: DriftMetric; onClick?: () => void; showHistory?: boolean }) {
  const TypeIcon = driftTypeIcons[metric.type];
  const isAboveThreshold = metric.current_value > metric.threshold;
  const percentOfThreshold = (metric.current_value / metric.threshold) * 100;

  return (
    <div
      onClick={onClick}
      className={cn(
        'p-4 rounded-lg border transition-all',
        severityConfig[metric.severity].bgColor,
        severityConfig[metric.severity].borderColor,
        onClick && 'cursor-pointer hover:shadow-md'
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <TypeIcon className="h-4 w-4 text-[var(--color-text-secondary)]" />
          <h4 className="font-medium text-[var(--color-text-primary)] text-sm">{metric.name}</h4>
        </div>
        <SeverityBadge severity={metric.severity} />
      </div>

      {metric.description && (
        <p className="text-xs text-[var(--color-text-secondary)] mb-3">{metric.description}</p>
      )}

      <div className="flex items-center justify-between">
        <div>
          <TrendIndicator trend={metric.trend} value={metric.current_value} />
          <p className="text-xs text-[var(--color-text-tertiary)] mt-1">
            Threshold: {metric.threshold.toFixed(3)} ({percentOfThreshold.toFixed(0)}%)
          </p>
        </div>
        {showHistory && metric.history.length > 1 && (
          <MiniSparkline data={metric.history} threshold={metric.threshold} />
        )}
      </div>

      {/* Progress bar showing current vs threshold */}
      <div className="mt-3">
        <div className="h-1.5 bg-[var(--color-border)] rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all',
              isAboveThreshold ? 'bg-red-500' : 'bg-green-500'
            )}
            style={{ width: `${Math.min(percentOfThreshold, 100)}%` }}
          />
        </div>
      </div>

      <p className="text-xs text-[var(--color-text-tertiary)] mt-2">
        Updated {new Date(metric.last_updated).toLocaleString()}
      </p>
    </div>
  );
}

function DriftSummaryPanel({ summary }: { summary: DriftSummary }) {
  const driftPercentage = (summary.drifting_features / summary.total_features) * 100;
  const statusColor = summary.model_drift_detected
    ? 'text-red-600 dark:text-red-400'
    : driftPercentage > 20
    ? 'text-yellow-600 dark:text-yellow-400'
    : 'text-green-600 dark:text-green-400';

  return (
    <div className="bg-[var(--color-card)] rounded-lg border border-[var(--color-border)] p-4 mb-4">
      <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3 flex items-center gap-2">
        <Activity className="h-4 w-4" />
        Drift Summary
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="text-center p-3 bg-[var(--color-background)] rounded-lg">
          <p className="text-2xl font-bold text-[var(--color-text-primary)]">{summary.total_features}</p>
          <p className="text-xs text-[var(--color-text-tertiary)]">Total Features</p>
        </div>
        <div className="text-center p-3 bg-[var(--color-background)] rounded-lg">
          <p className={cn('text-2xl font-bold', summary.drifting_features > 0 ? 'text-orange-500' : 'text-green-500')}>
            {summary.drifting_features}
          </p>
          <p className="text-xs text-[var(--color-text-tertiary)]">Drifting</p>
        </div>
        <div className="text-center p-3 bg-[var(--color-background)] rounded-lg">
          <p className={cn('text-2xl font-bold', statusColor)}>{summary.data_drift_percentage.toFixed(1)}%</p>
          <p className="text-xs text-[var(--color-text-tertiary)]">Data Drift</p>
        </div>
        <div className="text-center p-3 bg-[var(--color-background)] rounded-lg">
          <p
            className={cn(
              'text-sm font-semibold flex items-center justify-center gap-1',
              summary.model_drift_detected ? 'text-red-500' : 'text-green-500'
            )}
          >
            {summary.model_drift_detected ? (
              <>
                <AlertTriangle className="h-4 w-4" />
                Detected
              </>
            ) : (
              <>
                <CheckCircle className="h-4 w-4" />
                Stable
              </>
            )}
          </p>
          <p className="text-xs text-[var(--color-text-tertiary)]">Model Drift</p>
        </div>
      </div>

      {summary.alerts.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-[var(--color-text-secondary)]">Active Alerts</p>
          {summary.alerts.slice(0, 3).map((alert, idx) => (
            <div
              key={idx}
              className={cn(
                'flex items-center gap-2 p-2 rounded text-xs',
                severityConfig[alert.severity].bgColor,
                severityConfig[alert.severity].color
              )}
            >
              <AlertTriangle className="h-3 w-3 flex-shrink-0" />
              <span className="font-medium">{alert.feature}:</span>
              <span className="truncate">{alert.message}</span>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-[var(--color-text-tertiary)] mt-3">
        Last check: {new Date(summary.last_check).toLocaleString()}
      </p>
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function DriftVisualization({
  metrics,
  summary,
  className,
  showHistory = true,
  onMetricClick,
}: DriftVisualizationProps) {
  const [filterType, setFilterType] = useState<DriftType | 'all'>('all');
  const [filterSeverity, setFilterSeverity] = useState<DriftSeverity | 'all'>('all');

  const filteredMetrics = metrics.filter((m) => {
    if (filterType !== 'all' && m.type !== filterType) return false;
    if (filterSeverity !== 'all' && m.severity !== filterSeverity) return false;
    return true;
  });

  const groupedByType = filteredMetrics.reduce<Record<DriftType, DriftMetric[]>>((acc, metric) => {
    if (!acc[metric.type]) acc[metric.type] = [];
    acc[metric.type].push(metric);
    return acc;
  }, {} as Record<DriftType, DriftMetric[]>);

  return (
    <div className={cn('space-y-4', className)}>
      {/* Summary Panel */}
      {summary && <DriftSummaryPanel summary={summary} />}

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-4">
        <select
          value={filterType}
          onChange={(e) => setFilterType(e.target.value as DriftType | 'all')}
          className="px-3 py-1.5 text-sm border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] text-[var(--color-text-primary)]"
        >
          <option value="all">All Types</option>
          <option value="data">Data Drift</option>
          <option value="model">Model Drift</option>
          <option value="concept">Concept Drift</option>
          <option value="feature">Feature Drift</option>
        </select>
        <select
          value={filterSeverity}
          onChange={(e) => setFilterSeverity(e.target.value as DriftSeverity | 'all')}
          className="px-3 py-1.5 text-sm border border-[var(--color-border)] rounded-lg bg-[var(--color-background)] text-[var(--color-text-primary)]"
        >
          <option value="all">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
          <option value="none">None</option>
        </select>
      </div>

      {/* Metrics Grid */}
      {Object.entries(groupedByType).map(([type, typeMetrics]) => (
        <div key={type}>
          <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] mb-2 capitalize flex items-center gap-2">
            {React.createElement(driftTypeIcons[type as DriftType], { className: 'h-4 w-4' })}
            {type} Drift
            <span className="text-xs font-normal text-[var(--color-text-tertiary)]">({typeMetrics.length})</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {typeMetrics.map((metric) => (
              <DriftMetricCard
                key={`${metric.type}-${metric.name}`}
                metric={metric}
                showHistory={showHistory}
                onClick={onMetricClick ? () => onMetricClick(metric) : undefined}
              />
            ))}
          </div>
        </div>
      ))}

      {filteredMetrics.length === 0 && (
        <div className="text-center py-8 text-[var(--color-text-secondary)]">
          <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p>No drift metrics match the current filters</p>
        </div>
      )}
    </div>
  );
}

export default DriftVisualization;
