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
import {
  useFeedbackHealth,
  usePatterns,
  useKnowledgeUpdates,
  useQuickLearningCycle,
  useApplyUpdate,
  useRollbackUpdate,
} from '@/hooks/api';
import { StatusBadge, StatusDot } from '@/components/visualizations/dashboard/StatusBadge';
import type { StatusType } from '@/components/visualizations/dashboard/StatusBadge';
import { PatternSeverity, UpdateStatus, UpdateType } from '@/types/feedback';

// =============================================================================
// TYPES
// =============================================================================

interface PatternItem {
  pattern_id: string;
  pattern_type: string;
  severity: PatternSeverity;
  description: string;
  agent_name: string;
  occurrences: number;
  first_seen: string;
  last_seen: string;
}

interface UpdateItem {
  update_id: string;
  update_type: UpdateType;
  status: UpdateStatus;
  description: string;
  agent_name: string;
  confidence_score: number;
  created_at: string;
  applied_at?: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const SEVERITY_COLORS: Record<PatternSeverity, string> = {
  [PatternSeverity.CRITICAL]: '#ef4444', // red
  [PatternSeverity.HIGH]: '#f97316', // orange
  [PatternSeverity.MEDIUM]: '#f59e0b', // amber
  [PatternSeverity.LOW]: '#22c55e', // green
  [PatternSeverity.INFO]: '#3b82f6', // blue
};

const SEVERITY_ORDER: Record<PatternSeverity, number> = {
  [PatternSeverity.CRITICAL]: 0,
  [PatternSeverity.HIGH]: 1,
  [PatternSeverity.MEDIUM]: 2,
  [PatternSeverity.LOW]: 3,
  [PatternSeverity.INFO]: 4,
};

const STATUS_COLORS: Record<UpdateStatus, string> = {
  [UpdateStatus.PROPOSED]: '#3b82f6', // blue
  [UpdateStatus.APPROVED]: '#22c55e', // green
  [UpdateStatus.APPLIED]: '#10b981', // emerald
  [UpdateStatus.REJECTED]: '#ef4444', // red
  [UpdateStatus.ROLLED_BACK]: '#6b7280', // gray
};

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_PATTERNS: PatternItem[] = [
  {
    pattern_id: 'pat-001',
    pattern_type: 'performance_degradation',
    severity: PatternSeverity.HIGH,
    description: 'Causal Impact agent showing increased latency during peak hours',
    agent_name: 'causal_impact',
    occurrences: 15,
    first_seen: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    last_seen: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    pattern_id: 'pat-002',
    pattern_type: 'low_confidence',
    severity: PatternSeverity.MEDIUM,
    description: 'Gap Analyzer producing below-threshold confidence scores for Remibrutinib queries',
    agent_name: 'gap_analyzer',
    occurrences: 8,
    first_seen: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    last_seen: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
  },
  {
    pattern_id: 'pat-003',
    pattern_type: 'positive_feedback',
    severity: PatternSeverity.INFO,
    description: 'Explainer receiving consistent high ratings for clarity',
    agent_name: 'explainer',
    occurrences: 42,
    first_seen: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
    last_seen: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
  },
];

const SAMPLE_UPDATES: UpdateItem[] = [
  {
    update_id: 'upd-001',
    update_type: UpdateType.PROMPT_REFINEMENT,
    status: UpdateStatus.PROPOSED,
    description: 'Refine causal impact explanation template for better clarity',
    agent_name: 'causal_impact',
    confidence_score: 0.85,
    created_at: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
  },
  {
    update_id: 'upd-002',
    update_type: UpdateType.TOOL_PRIORITY,
    status: UpdateStatus.APPLIED,
    description: 'Increase priority of DoWhy estimator for propensity-based queries',
    agent_name: 'gap_analyzer',
    confidence_score: 0.92,
    created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    applied_at: new Date(Date.now() - 20 * 60 * 60 * 1000).toISOString(),
  },
  {
    update_id: 'upd-003',
    update_type: UpdateType.CONFIDENCE_THRESHOLD,
    status: UpdateStatus.APPROVED,
    description: 'Lower confidence threshold for Fabhalta brand queries based on feedback',
    agent_name: 'orchestrator',
    confidence_score: 0.78,
    created_at: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
  },
];

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
    case PatternSeverity.INFO:
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
    case UpdateStatus.REJECTED:
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

  // Fetch feedback health
  const {
    data: healthData,
    isLoading: isLoadingHealth,
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

  // Use API data or fall back to samples
  const patterns = patternsData?.patterns ?? SAMPLE_PATTERNS;
  const updates = updatesData?.updates ?? SAMPLE_UPDATES;

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
      cycles24h: healthData?.cycles_24h ?? 12,
      agentAvailable: healthData?.agent_available ?? true,
    };
  }, [patterns, updates, healthData]);

  // Prepare chart data
  const severityChartData = useMemo(() => {
    const counts: Record<PatternSeverity, number> = {
      [PatternSeverity.CRITICAL]: 0,
      [PatternSeverity.HIGH]: 0,
      [PatternSeverity.MEDIUM]: 0,
      [PatternSeverity.LOW]: 0,
      [PatternSeverity.INFO]: 0,
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

  // Run learning cycle
  const handleRunCycle = useCallback(() => {
    runQuickCycle(undefined, {
      onSuccess: () => {
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

  const isLoading = isLoadingHealth || isLoadingPatterns || isLoadingUpdates;

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

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Service Status</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.agentAvailable ? 'Online' : 'Offline'}
              {stats.agentAvailable ? (
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
              {stats.cycles24h}
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
                      {updateTypeChartData.map((entry, index) => (
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
                    const aDate = 'last_seen' in a ? a.last_seen : a.created_at;
                    const bDate = 'last_seen' in b ? b.last_seen : b.created_at;
                    return new Date(bDate).getTime() - new Date(aDate).getTime();
                  })
                  .slice(0, 5)
                  .map((item, index) => {
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
                              ? `${(item as PatternItem).agent_name} • ${formatRelativeTime((item as PatternItem).last_seen)}`
                              : `${(item as UpdateItem).agent_name} • ${formatRelativeTime((item as UpdateItem).created_at)}`}
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
                            {pattern.occurrences} occurrences
                          </div>
                        </div>
                        <p className="text-sm text-[var(--color-muted-foreground)] mb-2">
                          {pattern.description}
                        </p>
                        <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
                          <span>Agent: <strong>{pattern.agent_name}</strong></span>
                          <span>First seen: {formatRelativeTime(pattern.first_seen)}</span>
                          <span>Last seen: {formatRelativeTime(pattern.last_seen)}</span>
                        </div>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2 py-8 text-[var(--color-muted-foreground)]">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  No patterns detected
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
                          <Badge variant="outline">
                            {(update.confidence_score * 100).toFixed(0)}% confidence
                          </Badge>
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
                        {update.description}
                      </p>
                      <div className="flex items-center gap-4 text-xs text-[var(--color-muted-foreground)]">
                        <span>Agent: <strong>{update.agent_name}</strong></span>
                        <span>Created: {formatRelativeTime(update.created_at)}</span>
                        {update.applied_at && (
                          <span>Applied: {formatRelativeTime(update.applied_at)}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2 py-8 text-[var(--color-muted-foreground)]">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  No knowledge updates available
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
