/**
 * Analytics Types
 * ================
 *
 * TypeScript types for the analytics dashboard.
 *
 * @module types/analytics
 */

// =============================================================================
// TIME SERIES
// =============================================================================

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  label?: string;
}

// =============================================================================
// AGENT METRICS
// =============================================================================

export interface AgentMetrics {
  agent_name: string;
  agent_tier: number;
  total_invocations: number;
  successful_invocations: number;
  failed_invocations: number;
  success_rate: number;

  // Latency metrics (milliseconds). null == UNMEASURED (agent has entries in the
  // window but none carried a real duration_ms), DISTINCT from a measured 0ms.
  // The per-agent table renders "—" for null, never a fabricated "0ms".
  avg_latency_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  p99_latency_ms: number | null;
  min_latency_ms: number | null;
  max_latency_ms: number | null;

  // Confidence
  avg_confidence: number | null;
}

// =============================================================================
// LATENCY BREAKDOWN
// =============================================================================

export interface LatencyBreakdown {
  classification_ms: number;
  rag_retrieval_ms: number;
  routing_ms: number;
  agent_dispatch_ms: number;
  synthesis_ms: number;
  // null == UNMEASURED (no timed entries in window), distinct from a real 0ms.
  total_ms: number | null;
}

// =============================================================================
// QUERY METRICS SUMMARY
// =============================================================================

export interface QueryMetricsSummary {
  period_start: string;
  period_end: string;
  total_queries: number;
  successful_queries: number;
  failed_queries: number;
  success_rate: number;

  // Latency summary. null == UNMEASURED (no audit entry in the window carried a
  // duration_ms) — DISTINCT from a real measured 0ms. The UI renders "—" for
  // null rather than a misleading "0ms / instant". Mirrors the backend
  // QueryMetricsSummary, whose latency fields are now Optional[float] = None.
  avg_latency_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  p99_latency_ms: number | null;

  // Distribution
  intent_distribution: Record<string, number>;
  top_agents: string[];
}

// =============================================================================
// DASHBOARD RESPONSE
// =============================================================================

export interface AnalyticsDashboardResponse {
  summary: QueryMetricsSummary;
  agent_metrics: AgentMetrics[];
  latency_trend: TimeSeriesPoint[];
  query_volume_trend: TimeSeriesPoint[];
  latency_breakdown: LatencyBreakdown;
  generated_at: string;
  /** Count of audit entries excluded as automated background (polling/monitoring). */
  excluded_background_count: number;
  /** Names of the automated background agents excluded from the metrics. */
  excluded_agents: string[];
}

// =============================================================================
// AGENT PERFORMANCE TREND
// =============================================================================

export interface AgentPerformanceTrend {
  agent_name: string;
  data_points: TimeSeriesPoint[];
  period: string;
}

// =============================================================================
// PERIOD OPTIONS
// =============================================================================

export type AnalyticsPeriod = '1d' | '7d' | '30d' | '90d';
export type SummaryPeriod = '1h' | '6h' | '24h' | '7d';

export const PERIOD_LABELS: Record<AnalyticsPeriod, string> = {
  '1d': 'Last 24 Hours',
  '7d': 'Last 7 Days',
  '30d': 'Last 30 Days',
  '90d': 'Last 90 Days',
};

export const SUMMARY_PERIOD_LABELS: Record<SummaryPeriod, string> = {
  '1h': 'Last Hour',
  '6h': 'Last 6 Hours',
  '24h': 'Last 24 Hours',
  '7d': 'Last 7 Days',
};

// =============================================================================
// TIER LABELS
// =============================================================================

export const TIER_LABELS: Record<number, string> = {
  0: 'Foundation',
  1: 'Orchestrator',
  2: 'Causal',
  3: 'Monitoring',
  4: 'ML',
  5: 'Learning',
};

export function getTierLabel(tier: number): string {
  return TIER_LABELS[tier] ?? `Tier ${tier}`;
}

// =============================================================================
// FORMATTING HELPERS
// =============================================================================

export function formatLatency(ms: number | null | undefined): string {
  // null/undefined == UNMEASURED (no timed entries in window). Render an em
  // dash rather than a fabricated "0ms" that reads like "instant".
  if (ms == null) {
    return '—';
  }
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function formatNumber(value: number): string {
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M`;
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}K`;
  }
  return value.toString();
}
