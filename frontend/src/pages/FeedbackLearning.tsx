/**
 * Feedback Learning Page
 * ======================
 *
 * Dashboard for the Tier 5 self-improvement system including:
 * - Service health monitoring
 * - Pattern detection and listing
 * - Knowledge updates management
 * - Learning cycle execution
 *
 * @module pages/FeedbackLearning
 */

import { useState, useMemo, useCallback } from 'react';
import {
  Brain,
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  TrendingUp,
  Lightbulb,
  BookOpen,
  Play,
  Undo2,
  Check,
  Sparkles,
  Target,
  Zap,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { WarningBanner } from '@/components/ui/WarningBanner';
import {
  useFeedbackHealth,
  usePatterns,
  useKnowledgeUpdates,
  useQuickLearningCycle,
  useApplyUpdate,
  useRollbackUpdate,
  useFeedbackLearningInsight,
} from '@/hooks/api';
import { StrategicInsightCard } from '@/components/insights';
import { StatusDot } from '@/components/visualizations/dashboard/StatusBadge';
import type { StatusType } from '@/components/visualizations/dashboard/StatusBadge';
import { PatternSeverity, UpdateStatus, UpdateType } from '@/types/feedback';

// =============================================================================
// TYPES
// =============================================================================

// Extended pattern type for UI (matches API DetectedPattern + extra fields for display)
interface PatternItem {
  pattern_id: string;
  pattern_type: string;
  severity: PatternSeverity;
  description: string;
  // API fields
  frequency?: number;
  affected_agents?: string[];
  confidence?: number;
  /** When the pattern was detected (persistence created_at; #1244) */
  detected_at?: string | null;
  // UI/sample fields
  agent_name?: string;
  occurrences?: number;
  first_seen?: string;
  last_seen?: string;
}

// Extended update type for UI (matches API KnowledgeUpdate + extra fields for display)
interface UpdateItem {
  update_id: string;
  update_type: UpdateType;
  status: UpdateStatus;
  created_at: string;
  applied_at?: string;
  // API fields
  target_agent?: string;
  rationale?: string;
  expected_improvement?: string;
  // UI/sample fields
  description?: string;
  agent_name?: string;
  confidence_score?: number;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const SEVERITY_COLORS: Record<PatternSeverity, string> = {
  [PatternSeverity.CRITICAL]: '#ef4444', // red
  [PatternSeverity.HIGH]: '#f97316', // orange
  [PatternSeverity.MEDIUM]: '#f59e0b', // amber
  [PatternSeverity.LOW]: '#22c55e', // green
};

const SEVERITY_ORDER: Record<PatternSeverity, number> = {
  [PatternSeverity.CRITICAL]: 0,
  [PatternSeverity.HIGH]: 1,
  [PatternSeverity.MEDIUM]: 2,
  [PatternSeverity.LOW]: 3,
};


// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatDateTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatRelativeTime(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const hours = Math.floor(diffMs / (60 * 60 * 1000));
  const days = Math.floor(hours / 24);

  if (hours < 1) return 'Just now';
  if (hours < 24) return `${hours}h ago`;
  if (days < 7) return `${days}d ago`;
  return formatDateTime(d);
}

function getSeverityStatus(severity: PatternSeverity): StatusType {
  switch (severity) {
    case PatternSeverity.CRITICAL:
      return 'error';
    case PatternSeverity.HIGH:
      return 'error';
    case PatternSeverity.MEDIUM:
      return 'warning';
    case PatternSeverity.LOW:
      return 'healthy';
    default:
      return 'unknown';
  }
}

function getUpdateStatusBadgeVariant(status: UpdateStatus): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case UpdateStatus.APPLIED:
      return 'default';
    case UpdateStatus.APPROVED:
      return 'secondary';
    case UpdateStatus.PROPOSED:
      return 'outline';
    case UpdateStatus.ROLLED_BACK:
      return 'destructive';
    default:
      return 'outline';
  }
}

// =============================================================================
// COMPONENT
// =============================================================================

function FeedbackLearning() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isRefreshing, setIsRefreshing] = useState(false);
  // F-010: hold the latest learning-cycle warnings so we can surface them
  // to the user as a yellow banner. The mutation hook itself doesn't keep
  // a long-lived response cache, so we mirror it into local state here.
  const [cycleWarnings, setCycleWarnings] = useState<string[]>([]);

  // Fetch feedback health
  const {
    data: healthData,
    isLoading: isHealthLoading,
    refetch: refetchHealth,
  } = useFeedbackHealth({ refetchInterval: 30000 });

  // Fetch patterns
  const {
    data: patternsData,
    isLoading: isLoadingPatterns,
    refetch: refetchPatterns,
  } = usePatterns(undefined, { refetchInterval: 60000 });

  // Fetch knowledge updates
  const {
    data: updatesData,
    isLoading: isLoadingUpdates,
    refetch: refetchUpdates,
  } = useKnowledgeUpdates(undefined, { refetchInterval: 60000 });

  // Mutations
  const { mutate: runQuickCycle, isPending: isRunningCycle } = useQuickLearningCycle();
  const { mutate: applyUpdate, isPending: isApplying } = useApplyUpdate();
  const { mutate: rollbackUpdate, isPending: isRollingBack } = useRollbackUpdate();

  // Strategic interpretation (server-derived grounding; on-demand)
  const flInsight = useFeedbackLearningInsight();

  // F-002 fix: no fabricated `SAMPLE_PATTERNS`/`SAMPLE_UPDATES` fallback.
  // Data comes strictly from API; the page renders empty states when the
  // API has no patterns / updates yet.
  const patterns: PatternItem[] = (patternsData?.patterns ?? []) as PatternItem[];
  const updates: UpdateItem[] = (updatesData?.updates ?? []) as UpdateItem[];

  // Calculate stats
  const stats = useMemo(() => {
    const criticalCount = patterns.filter(p => p.severity === PatternSeverity.CRITICAL).length;
    const highCount = patterns.filter(p => p.severity === PatternSeverity.HIGH).length;
    const proposedCount = updates.filter(u => u.status === UpdateStatus.PROPOSED).length;
    const appliedCount = updates.filter(u => u.status === UpdateStatus.APPLIED).length;

    return {
      totalPatterns: patterns.length,
      criticalPatterns: criticalCount,
      highPatterns: highCount,
      totalUpdates: updates.length,
      pendingUpdates: proposedCount,
      appliedUpdates: appliedCount,
      // Honesty: default to 0 / not-available rather than fabricated plausible
      // values (was `?? 12` / `?? true`). Those hardcoded fallbacks rendered a
      // fake "12 cycles" + "Online" while the health query was loading or if it
      // failed. The card uses `healthPending` (below) to show "Checking…" / "—"
      // during the initial load so neither a fabricated nor a premature value
      // is shown; once health resolves these reflect the real response.
      cycles24h: healthData?.cycles_24h ?? 0,
      agentAvailable: healthData?.agent_available ?? false,
    };
  }, [patterns, updates, healthData]);

  // True only on the first health load (no response yet). Used to render an
  // honest "Checking…" / "—" placeholder instead of a fabricated value. A
  // resolved-but-failed health check (isHealthLoading=false, no data) falls
  // through to the conservative real defaults above (Offline / 0), never a
  // fabricated "Online" / "12".
  const healthPending = !healthData && isHealthLoading;

  // #1661: optimizer-gate block from the health poll. Absent (older backend or
  // health still loading) renders nothing rather than a fabricated placeholder.
  const optimizer = healthData?.optimizer ?? null;

  // Prepare chart data
  const severityChartData = useMemo(() => {
    const counts: Record<PatternSeverity, number> = {
      [PatternSeverity.CRITICAL]: 0,
      [PatternSeverity.HIGH]: 0,
      [PatternSeverity.MEDIUM]: 0,
      [PatternSeverity.LOW]: 0,
    };
    patterns.forEach(p => {
      counts[p.severity] = (counts[p.severity] || 0) + 1;
    });
    return Object.entries(counts).map(([severity, count]) => ({
      severity,
      count,
      fill: SEVERITY_COLORS[severity as PatternSeverity],
    }));
  }, [patterns]);

  const updateTypeChartData = useMemo(() => {
    const counts: Record<string, number> = {};
    updates.forEach(u => {
      counts[u.update_type] = (counts[u.update_type] || 0) + 1;
    });
    return Object.entries(counts).map(([type, count]) => ({
      name: type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: count,
    }));
  }, [updates]);

  // Refresh handler
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await Promise.all([refetchHealth(), refetchPatterns(), refetchUpdates()]);
    setIsRefreshing(false);
  }, [refetchHealth, refetchPatterns, refetchUpdates]);

  // Run learning cycle — capture API warnings so the page surfaces them
  // (F-010-frontend).
  const handleRunCycle = useCallback(() => {
    runQuickCycle(undefined, {
      onSuccess: (data) => {
        setCycleWarnings(data?.warnings ?? []);
        handleRefresh();
      },
    });
  }, [runQuickCycle, handleRefresh]);

  // Apply update
  const handleApplyUpdate = useCallback((updateId: string) => {
    applyUpdate({ updateId }, {
      onSuccess: () => {
        refetchUpdates();
      },
    });
  }, [applyUpdate, refetchUpdates]);

  // Rollback update
  const handleRollbackUpdate = useCallback((updateId: string) => {
    rollbackUpdate(updateId, {
      onSuccess: () => {
        refetchUpdates();
      },
    });
  }, [rollbackUpdate, refetchUpdates]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Feedback Learning</h1>
          <p className="text-[var(--color-muted-foreground)]">
            Tier 5 self-improvement system - pattern detection and knowledge updates
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="default"
            size="sm"
            onClick={handleRunCycle}
            disabled={isRunningCycle}
          >
            <Play className={`h-4 w-4 mr-2 ${isRunningCycle ? 'animate-pulse' : ''}`} />
            {isRunningCycle ? 'Running...' : 'Run Learning Cycle'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* API-reported warnings from the most recent learning cycle (F-010). */}
      {cycleWarnings.length > 0 && (
        <div className="mb-6">
          <WarningBanner
            messages={cycleWarnings}
            title="Learning cycle warnings"
          />
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Service Status</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthPending ? 'Checking…' : stats.agentAvailable ? 'Online' : 'Offline'}
              {healthPending ? (
                <RefreshCw className="h-5 w-5 animate-spin text-[var(--color-muted-foreground)]" />
              ) : stats.agentAvailable ? (
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              ) : (
                <XCircle className="h-5 w-5 text-rose-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Feedback Learner
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Learning Cycles</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {healthPending ? '—' : stats.cycles24h}
              <Sparkles className="h-5 w-5 text-violet-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Last 24 hours
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Patterns</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.totalPatterns}
              <Lightbulb className="h-5 w-5 text-amber-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Detected issues
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Critical</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.criticalPatterns}
              {stats.criticalPatterns > 0 ? (
                <AlertTriangle className="h-5 w-5 text-rose-500" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              {stats.criticalPatterns > 0 ? 'Requires attention' : 'All clear'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Pending Updates</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.pendingUpdates}
              <BookOpen className="h-5 w-5 text-blue-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Awaiting approval
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Applied</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.appliedUpdates}
              <TrendingUp className="h-5 w-5 text-emerald-500" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Knowledge updates
            </p>
          </CardContent>
        </Card>
      </div>

      {/* #1661: the optimizer half of the loop.
          The daily prompt-optimization beat returns a legitimate
          `{"status":"skipped"}` at its trigger, so it can complete successfully
          for months without ever compiling anything — and every other signal on
          this page stays green while it does. This card is the one place that
          says so. The denominator is deliberate: "8" alone reads as a volume
          shortfall, "8 of 218" reads as the low-yield problem it actually is. */}
      {optimizer && (
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <div className="flex items-start justify-between gap-4">
              <div>
                <CardDescription>Prompt Optimizer</CardDescription>
                <CardTitle className="text-2xl flex items-center gap-2">
                  {`${optimizer.eligible_signals ?? '—'} / ${optimizer.min_signals}`}
                  {optimizer.would_trigger === true ? (
                    <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  ) : optimizer.would_trigger === false ? (
                    <AlertTriangle className="h-5 w-5 text-amber-500" />
                  ) : (
                    <XCircle className="h-5 w-5 text-[var(--color-muted-foreground)]" />
                  )}
                </CardTitle>
              </div>
              <Badge variant={optimizer.would_trigger === true ? 'default' : 'secondary'}>
                {optimizer.would_trigger === true
                  ? 'Ready'
                  : optimizer.would_trigger === false
                    ? 'Inert'
                    : 'Unknown'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-sm text-[var(--color-muted-foreground)]">{optimizer.reason}</p>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              {optimizer.total_signals !== null && optimizer.total_signals !== undefined
                ? `${optimizer.total_signals} signals recorded`
                : 'signal count unknown'}
              {' • '}
              {optimizer.optimization_runs === 0
                ? 'never optimized'
                : optimizer.optimization_runs
                  ? `${optimizer.optimization_runs} optimization runs`
                  : 'run count unknown'}
              {optimizer.last_eligible_signal_at
                ? ` • last eligible signal ${new Date(optimizer.last_eligible_signal_at).toLocaleString()}`
                : ''}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Strategic Interpretation — grounded in persisted cycles/patterns/updates
          and the real feedback inflow (server-derived) */}
      <div className="mb-6">
        <StrategicInsightCard
          onGenerate={() => flInsight.mutate({ days: 7 })}
          isLoading={flInsight.isPending}
          error={flInsight.error?.message ?? null}
          insight={flInsight.data?.insight}
          keyTakeaways={flInsight.data?.key_takeaways}
          grounding={flInsight.data?.grounding}
          isFallback={flInsight.data?.is_fallback}
          provenance={flInsight.data?.provenance}
          generatedAt={flInsight.data?.generated_at}
        />
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 lg:w-auto lg:inline-flex">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="patterns">Patterns</TabsTrigger>
          <TabsTrigger value="updates">Knowledge Updates</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Pattern Severity Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Pattern Severity Distribution
                </CardTitle>
                <CardDescription>Breakdown of detected patterns by severity</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={severityChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="severity" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {severityChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Update Types */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Update Types
                </CardTitle>
                <CardDescription>Distribution of knowledge update types</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={updateTypeChartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {updateTypeChartData.map((_, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'][index % 5]}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Recent Activity
              </CardTitle>
              <CardDescription>Latest patterns and updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[...patterns.slice(0, 3), ...updates.slice(0, 2)]
                  .sort((a, b) => {
                    // #1244: real API patterns carry detected_at (never
                    // last_seen/created_at) — an unknown timestamp sorts
                    // LAST, not as "just now" displacing fresh updates.
                    const ts = (item: PatternItem | UpdateItem): number => {
                      const raw =
                        'pattern_id' in item
                          ? (item.last_seen ?? item.detected_at)
                          : item.created_at;
                      return raw ? new Date(raw).getTime() : 0;
                    };
                    return ts(b) - ts(a);
                  })
                  .slice(0, 5)
                  .map((item) => {
                    const isPattern = 'pattern_id' in item;
                    return (
                      <div
                        key={isPattern ? (item as PatternItem).pattern_id : (item as UpdateItem).update_id}
                        className="flex items-start gap-4 p-3 rounded-lg border border-[var(--color-border)]"
                      >
                        <div className="flex-shrink-0">
                          {isPattern ? (
                            <Lightbulb className="h-5 w-5 text-amber-500" />
                          ) : (
                            <BookOpen className="h-5 w-5 text-blue-500" />
                          )}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-medium">
                              {isPattern ? 'Pattern Detected' : 'Knowledge Update'}
                            </span>
                            {isPattern ? (
                              <Badge
                                variant={getSeverityStatus((item as PatternItem).severity) === 'error' ? 'destructive' : 'outline'}
                              >
                                {(item as PatternItem).severity}
                              </Badge>
                            ) : (
                              <Badge variant={getUpdateStatusBadgeVariant((item as UpdateItem).status)}>
                                {(item as UpdateItem).status}
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-[var(--color-muted-foreground)]">
                            {isPattern ? (item as PatternItem).description : (item as UpdateItem).description}
                          </p>
                          <p className="text-xs text-[var(--color-muted-foreground)] mt-1">
                            {isPattern
                              ? (() => {
                                  // #1244: real API patterns carry affected_agents +
                                  // detected_at, never agent_name / last_seen (UI/sample-era
                                  // fields) — same fallback chain as the Patterns tab.
                                  const p = item as PatternItem;
                                  const agent = p.agent_name ?? p.affected_agents?.[0] ?? 'N/A';
                                  const ts = p.last_seen ?? p.detected_at;
                                  return `${agent} • ${ts ? formatRelativeTime(ts) : 'N/A'}`;
                                })()
                              : `${(item as UpdateItem).agent_name ?? (item as UpdateItem).target_agent ?? 'N/A'} • ${formatRelativeTime((item as UpdateItem).created_at)}`}
                          </p>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Patterns Tab */}
        <TabsContent value="patterns" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5" />
                Detected Patterns
              </CardTitle>
              <CardDescription>Patterns identified from agent feedback and performance data</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingPatterns ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-6 w-6 animate-spin text-[var(--color-muted-foreground)]" />
                </div>
              ) : patterns.length > 0 ? (
                <div className="space-y-4">
                  {patterns
                    .sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity])
                    .map((pattern) => (
                      <div
                        key={pattern.pattern_id}
                        className={`p-4 rounded-lg border ${
                          pattern.severity === PatternSeverity.CRITICAL
                            ? 'border-rose-300 bg-rose-50/50 dark:bg-rose-950/20'
                            : pattern.severity === PatternSeverity.HIGH
                            ? 'border-orange-300 bg-orange-50/50 dark:bg-orange-950/20'
                            : 'border-[var(--color-border)]'
                        }`}
                      >
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <StatusDot status={getSeverityStatus(pattern.severity)} />
                            <span className="font-medium">{pattern.pattern_type.replace(/_/g, ' ')}</span>
                            <Badge
                              style={{ backgroundColor: SEVERITY_COLORS[pattern.severity], color: 'white' }}
                            >
                              {pattern.severity}
                            </Badge>
                          </div>
                          <div className="text-sm text-[var(--color-muted-foreground)]">
                            {pattern.occurrences ?? pattern.frequency ?? 0} occurrences
                          </div>
                        </div>
                        <p className="text-sm text-[var(--color-muted-foreground)] mb-2">
                          {pattern.description}
                        </p>
                        <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
                          <span>Agent: <strong>{pattern.agent_name ?? pattern.affected_agents?.[0] ?? 'N/A'}</strong></span>
                          {pattern.first_seen && <span>First seen: {formatRelativeTime(pattern.first_seen)}</span>}
                          {pattern.last_seen && <span>Last seen: {formatRelativeTime(pattern.last_seen)}</span>}
                          {!pattern.first_seen && !pattern.last_seen && pattern.detected_at && (
                            <span>Detected: {formatRelativeTime(pattern.detected_at)}</span>
                          )}
                        </div>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 py-8 text-center text-[var(--color-muted-foreground)]">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                    No patterns detected in the analyzed window
                  </div>
                  <p className="max-w-md text-xs">
                    Feedback signals accrue per chat turn, and each learning cycle scans a
                    bounded lookback window — an empty tab can also mean the last cycle&apos;s
                    window contained no feedback (see cycle warnings above).
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Updates Tab */}
        <TabsContent value="updates" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Knowledge Updates
              </CardTitle>
              <CardDescription>Proposed and applied system improvements</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingUpdates ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-6 w-6 animate-spin text-[var(--color-muted-foreground)]" />
                </div>
              ) : updates.length > 0 ? (
                <div className="space-y-4">
                  {updates.map((update) => (
                    <div
                      key={update.update_id}
                      className="p-4 rounded-lg border border-[var(--color-border)]"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{update.update_type.replace(/_/g, ' ')}</span>
                          <Badge variant={getUpdateStatusBadgeVariant(update.status)}>
                            {update.status}
                          </Badge>
                          {update.confidence_score && (
                            <Badge variant="outline">
                              {(update.confidence_score * 100).toFixed(0)}% confidence
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          {update.status === UpdateStatus.PROPOSED && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleApplyUpdate(update.update_id)}
                              disabled={isApplying}
                            >
                              <Check className="h-4 w-4 mr-1" />
                              Apply
                            </Button>
                          )}
                          {update.status === UpdateStatus.APPLIED && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleRollbackUpdate(update.update_id)}
                              disabled={isRollingBack}
                            >
                              <Undo2 className="h-4 w-4 mr-1" />
                              Rollback
                            </Button>
                          )}
                        </div>
                      </div>
                      <p className="text-sm text-[var(--color-muted-foreground)] mb-2">
                        {update.description ?? update.rationale ?? 'No description'}
                      </p>
                      <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
                        <span>Agent: <strong>{update.agent_name ?? update.target_agent ?? 'N/A'}</strong></span>
                        <span>Created: {formatRelativeTime(update.created_at)}</span>
                        {update.applied_at && (
                          <span>Applied: {formatRelativeTime(update.applied_at)}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 py-8 text-center text-[var(--color-muted-foreground)]">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                    No knowledge updates proposed
                  </div>
                  <p className="max-w-md text-xs">
                    Updates are generated from patterns detected by a learning cycle and
                    wait here for manual review and apply.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default FeedbackLearning;
