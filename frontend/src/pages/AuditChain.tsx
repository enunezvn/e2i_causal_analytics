/**
 * Audit Chain Page
 * ================
 *
 * Dashboard for monitoring workflow audit trails including:
 * - Recent workflows list with verification status
 * - Workflow detail view with agent execution path
 * - Chain verification status
 * - Tier distribution visualization
 * - Low confidence and failed validation entries
 *
 * @module pages/AuditChain
 */

import { useState, useMemo, useCallback } from 'react';
import {
  Shield,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Link as LinkIcon,
  AlertTriangle,
  ChevronRight,
  Hash,
  Layers,
  Activity,
  FileText,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  useRecentWorkflows,
  useWorkflowDetails,
  useTierDistribution,
  useFailedValidationEntries,
  useLowConfidenceEntries,
} from '@/hooks/api';
import { StatusBadge, StatusDot } from '@/components/visualizations/dashboard/StatusBadge';
import type { StatusType } from '@/components/visualizations/dashboard/StatusBadge';

// =============================================================================
// TYPES
// =============================================================================

// Extended workflow item for UI (includes computed fields)
interface WorkflowListItem {
  workflow_id: string;
  brand?: string;
  first_agent: string;
  last_agent: string;
  entry_count: number;
  started_at: string;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const TIER_NAMES: Record<number, string> = {
  0: 'Foundation',
  1: 'Orchestration',
  2: 'Causal',
  3: 'Monitoring',
  4: 'ML Predictions',
  5: 'Learning',
};

const TIER_COLORS: Record<number, string> = {
  0: '#6366f1', // indigo
  1: '#8b5cf6', // violet
  2: '#3b82f6', // blue
  3: '#10b981', // emerald
  4: '#f59e0b', // amber
  5: '#ef4444', // red
};

// =============================================================================
// SAMPLE DATA
// =============================================================================

const SAMPLE_WORKFLOWS: WorkflowListItem[] = [
  {
    workflow_id: 'wf-001-abc123',
    brand: 'Kisqali',
    first_agent: 'orchestrator',
    last_agent: 'explainer',
    entry_count: 8,
    started_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    workflow_id: 'wf-002-def456',
    brand: 'Fabhalta',
    first_agent: 'orchestrator',
    last_agent: 'prediction_synthesizer',
    entry_count: 6,
    started_at: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
  },
  {
    workflow_id: 'wf-003-ghi789',
    brand: 'Remibrutinib',
    first_agent: 'orchestrator',
    last_agent: 'gap_analyzer',
    entry_count: 5,
    started_at: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
  },
];

const SAMPLE_TIER_DISTRIBUTION: Record<number, number> = {
  0: 0,
  1: 2,
  2: 3,
  3: 2,
  4: 1,
  5: 2,
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

function formatDuration(startStr: string, endStr?: string): string {
  const start = new Date(startStr);
  const end = endStr ? new Date(endStr) : new Date();
  const diffMs = end.getTime() - start.getTime();
  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

function getVerificationStatus(chainVerified?: boolean): StatusType {
  return chainVerified === false ? 'error' : 'healthy';
}

// =============================================================================
// COMPONENT
// =============================================================================

function AuditChain() {
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('workflows');
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Fetch recent workflows
  const {
    data: recentWorkflowsData,
    isLoading: isLoadingWorkflows,
    refetch: refetchWorkflows,
  } = useRecentWorkflows({ limit: 20 }, { refetchInterval: 60000 });

  // Fetch workflow details when selected
  const {
    data: workflowDetails,
    isLoading: isLoadingDetails,
  } = useWorkflowDetails(selectedWorkflow || '', {
    enabled: !!selectedWorkflow,
  });

  // Fetch tier distribution for selected workflow
  const { data: tierDistribution } = useTierDistribution(selectedWorkflow || '', {
    enabled: !!selectedWorkflow,
  });

  // Fetch failed validation entries
  const { data: failedEntries } = useFailedValidationEntries(selectedWorkflow || '', {
    enabled: !!selectedWorkflow,
  });

  // Fetch low confidence entries
  const { data: lowConfidenceEntries } = useLowConfidenceEntries(selectedWorkflow || '', 0.7, {
    enabled: !!selectedWorkflow,
  });

  // Use API data or fall back to samples
  const workflows = recentWorkflowsData ?? SAMPLE_WORKFLOWS;
  const distribution = tierDistribution ?? SAMPLE_TIER_DISTRIBUTION;

  // Prepare chart data for tier distribution
  const tierChartData = useMemo(() => {
    return Object.entries(distribution).map(([tier, count]) => ({
      tier: `T${tier}`,
      name: TIER_NAMES[parseInt(tier)] || `Tier ${tier}`,
      count,
      fill: TIER_COLORS[parseInt(tier)] || '#6b7280',
    }));
  }, [distribution]);

  // Calculate stats (verification status comes from workflow details, not list)
  const stats = useMemo(() => {
    const totalEntries = workflows.reduce((sum, w) => sum + w.entry_count, 0);
    return {
      totalWorkflows: workflows.length,
      totalEntries,
      avgEntriesPerWorkflow: workflows.length > 0 ? Math.round(totalEntries / workflows.length) : 0,
    };
  }, [workflows]);

  // Refresh handler
  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await refetchWorkflows();
    setIsRefreshing(false);
  }, [refetchWorkflows]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Audit Chain</h1>
          <p className="text-[var(--color-muted-foreground)]">
            Cryptographic audit trail and workflow verification
          </p>
        </div>
        <div className="flex items-center gap-4">
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Workflows</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.totalWorkflows}
              <FileText className="h-5 w-5 text-[var(--color-muted-foreground)]" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Recent 24h activity
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Total Entries</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.totalEntries}
              <Hash className="h-5 w-5 text-[var(--color-muted-foreground)]" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Audit records
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Avg Entries</CardDescription>
            <CardTitle className="text-2xl flex items-center gap-2">
              {stats.avgEntriesPerWorkflow}
              <Layers className="h-5 w-5 text-[var(--color-muted-foreground)]" />
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-[var(--color-muted-foreground)]">
              Per workflow
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-flex">
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="details" disabled={!selectedWorkflow}>Details</TabsTrigger>
          <TabsTrigger value="verification" disabled={!selectedWorkflow}>Verification</TabsTrigger>
          <TabsTrigger value="issues" disabled={!selectedWorkflow}>Issues</TabsTrigger>
        </TabsList>

        {/* Workflows Tab */}
        <TabsContent value="workflows" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Recent Workflows
              </CardTitle>
              <CardDescription>Click a workflow to view details and verification status</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingWorkflows ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-6 w-6 animate-spin text-[var(--color-muted-foreground)]" />
                </div>
              ) : (
                <div className="space-y-3">
                  {workflows.map((workflow) => (
                    <div
                      key={workflow.workflow_id}
                      className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-colors ${
                        selectedWorkflow === workflow.workflow_id
                          ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/20'
                          : 'border-[var(--color-border)] hover:bg-[var(--color-muted)]/50'
                      }`}
                      onClick={() => {
                        setSelectedWorkflow(workflow.workflow_id);
                        setActiveTab('details');
                      }}
                    >
                      <div className="flex items-center gap-4">
                        <StatusDot status="healthy" />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-sm">{workflow.workflow_id}</span>
                            {workflow.brand && <Badge variant="outline">{workflow.brand}</Badge>}
                          </div>
                          <p className="text-sm text-[var(--color-muted-foreground)] mt-1">
                            {workflow.first_agent} <ChevronRight className="inline h-3 w-3" /> {workflow.last_agent}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-6 text-sm">
                        <div className="text-right">
                          <p className="font-medium">{workflow.entry_count} entries</p>
                          <p className="text-[var(--color-muted-foreground)]">
                            {formatDateTime(workflow.started_at)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Details Tab */}
        <TabsContent value="details" className="space-y-6">
          {selectedWorkflow && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Workflow Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5" />
                      Workflow Summary
                    </CardTitle>
                    <CardDescription>Execution details for {selectedWorkflow}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {isLoadingDetails ? (
                      <div className="flex items-center justify-center py-8">
                        <RefreshCw className="h-6 w-6 animate-spin" />
                      </div>
                    ) : workflowDetails ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                            <p className="text-sm text-[var(--color-muted-foreground)]">Total Entries</p>
                            <p className="text-2xl font-bold">{workflowDetails.summary.total_entries}</p>
                          </div>
                          <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                            <p className="text-sm text-[var(--color-muted-foreground)]">Duration</p>
                            <p className="text-2xl font-bold">
                              {Math.round(workflowDetails.summary.total_duration_ms / 1000)}s
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                            <p className="text-sm text-[var(--color-muted-foreground)]">Unique Agents</p>
                            <p className="text-2xl font-bold">{workflowDetails.summary.agents_involved?.length ?? 0}</p>
                          </div>
                          <div className="p-3 rounded-lg bg-[var(--color-muted)]/50">
                            <p className="text-sm text-[var(--color-muted-foreground)]">Brand</p>
                            <p className="text-2xl font-bold">{workflowDetails.summary.brand || 'N/A'}</p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-[var(--color-muted-foreground)]">No data available</p>
                    )}
                  </CardContent>
                </Card>

                {/* Tier Distribution */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Layers className="h-5 w-5" />
                      Tier Distribution
                    </CardTitle>
                    <CardDescription>Entries by agent tier</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={tierChartData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="tier" width={40} />
                        <Tooltip
                          formatter={(value, name) => [value, 'Entries']}
                          labelFormatter={(label) => tierChartData.find(t => t.tier === label)?.name || label}
                        />
                        <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                          {tierChartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              {/* Agent Execution Path */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Execution Timeline
                  </CardTitle>
                  <CardDescription>Audit entries in sequence order</CardDescription>
                </CardHeader>
                <CardContent>
                  {workflowDetails?.entries && workflowDetails.entries.length > 0 ? (
                    <div className="space-y-4">
                      {workflowDetails.entries.map((entry, index) => (
                        <div
                          key={entry.entry_id || index}
                          className="flex items-start gap-4 p-3 rounded-lg border border-[var(--color-border)]"
                        >
                          <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-[var(--color-muted)] text-sm font-bold">
                            {entry.sequence_number}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-medium">{entry.agent_name}</span>
                              <Badge variant="outline" className="text-xs">
                                Tier {entry.agent_tier}
                              </Badge>
                              {entry.confidence_score !== undefined && (
                                <Badge
                                  variant={entry.confidence_score >= 0.7 ? 'outline' : 'destructive'}
                                  className="text-xs"
                                >
                                  {(entry.confidence_score * 100).toFixed(0)}% conf
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-[var(--color-muted-foreground)]">
                              {formatDateTime(entry.created_at)}
                              {entry.duration_ms && ` • ${entry.duration_ms}ms`}
                            </p>
                          </div>
                          <div className="flex-shrink-0">
                            {entry.validation_passed ? (
                              <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                            ) : (
                              <XCircle className="h-5 w-5 text-rose-500" />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[var(--color-muted-foreground)] py-4 text-center">
                      No entries to display
                    </p>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Verification Tab */}
        <TabsContent value="verification" className="space-y-6">
          {selectedWorkflow && workflowDetails && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <LinkIcon className="h-5 w-5" />
                  Chain Verification
                </CardTitle>
                <CardDescription>Cryptographic integrity check results</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Verification Status */}
                  <div className={`p-6 rounded-lg border-2 ${
                    workflowDetails.verification.is_valid
                      ? 'border-emerald-300 bg-emerald-50 dark:bg-emerald-950/20'
                      : 'border-rose-300 bg-rose-50 dark:bg-rose-950/20'
                  }`}>
                    <div className="flex items-center gap-4">
                      {workflowDetails.verification.is_valid ? (
                        <CheckCircle2 className="h-12 w-12 text-emerald-500" />
                      ) : (
                        <XCircle className="h-12 w-12 text-rose-500" />
                      )}
                      <div>
                        <h3 className="text-xl font-bold">
                          {workflowDetails.verification.is_valid
                            ? 'Chain Verified'
                            : 'Chain Invalid'}
                        </h3>
                        <p className="text-[var(--color-muted-foreground)]">
                          {workflowDetails.verification.is_valid
                            ? 'All hash links are valid and unbroken'
                            : `Chain broken at entry ${workflowDetails.verification.first_invalid_entry}`}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Verification Details */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg bg-[var(--color-muted)]/50">
                      <p className="text-sm text-[var(--color-muted-foreground)]">Entries Checked</p>
                      <p className="text-2xl font-bold">{workflowDetails.verification.entries_checked}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-[var(--color-muted)]/50">
                      <p className="text-sm text-[var(--color-muted-foreground)]">Hash Algorithm</p>
                      <p className="text-lg font-mono">SHA-256</p>
                    </div>
                    <div className="p-4 rounded-lg bg-[var(--color-muted)]/50">
                      <p className="text-sm text-[var(--color-muted-foreground)]">Verified At</p>
                      <p className="text-lg">{formatDateTime(workflowDetails.verification.verified_at)}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Issues Tab */}
        <TabsContent value="issues" className="space-y-6">
          {selectedWorkflow && (
            <>
              {/* Failed Validation Entries */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <XCircle className="h-5 w-5 text-rose-500" />
                    Failed Validation Entries
                  </CardTitle>
                  <CardDescription>Entries that did not pass validation checks</CardDescription>
                </CardHeader>
                <CardContent>
                  {failedEntries && failedEntries.length > 0 ? (
                    <div className="space-y-3">
                      {failedEntries.map((entry, index) => (
                        <div
                          key={entry.entry_id || index}
                          className="p-4 rounded-lg border border-rose-200 bg-rose-50/50 dark:bg-rose-950/20"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{entry.agent_name}</span>
                              <Badge variant="outline">Seq #{entry.sequence_number}</Badge>
                            </div>
                            <span className="text-sm text-[var(--color-muted-foreground)]">
                              {formatDateTime(entry.created_at)}
                            </span>
                          </div>
                          <p className="text-sm text-rose-600">
                            Validation check failed
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center gap-2 py-8 text-[var(--color-muted-foreground)]">
                      <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                      No failed validation entries
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Low Confidence Entries */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-amber-500" />
                    Low Confidence Entries
                  </CardTitle>
                  <CardDescription>Entries with confidence score below 70%</CardDescription>
                </CardHeader>
                <CardContent>
                  {lowConfidenceEntries && lowConfidenceEntries.length > 0 ? (
                    <div className="space-y-3">
                      {lowConfidenceEntries.map((entry, index) => (
                        <div
                          key={entry.entry_id || index}
                          className="p-4 rounded-lg border border-amber-200 bg-amber-50/50 dark:bg-amber-950/20"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{entry.agent_name}</span>
                              <Badge variant="outline">Seq #{entry.sequence_number}</Badge>
                              <Badge variant="secondary" className="bg-amber-100 text-amber-700">
                                {((entry.confidence_score || 0) * 100).toFixed(0)}% confidence
                              </Badge>
                            </div>
                            <span className="text-sm text-[var(--color-muted-foreground)]">
                              {formatDateTime(entry.created_at)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-center justify-center gap-2 py-8 text-[var(--color-muted-foreground)]">
                      <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                      No low confidence entries
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default AuditChain;
