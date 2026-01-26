/**
 * Audit History Component
 * =======================
 *
 * Displays audit chain history with workflow details,
 * entry timeline, and chain verification status.
 *
 * Features:
 * - Recent workflows list
 * - Expandable workflow details
 * - Chain verification status
 * - Agent tier badges
 * - Time-based formatting
 *
 * @module components/audit/AuditHistory
 */

import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Shield,
  Clock,
  Activity,
  AlertCircle,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { getValidated } from '@/lib/api-client';
import {
  RecentWorkflowsResponseSchema,
  AuditEntriesResponseSchema,
  ChainVerificationSchema,
  WorkflowSummarySchema,
  type RecentWorkflowValidated,
  type AuditEntryValidated,
  type ChainVerificationValidated,
  type WorkflowSummaryValidated,
} from '@/lib/api-schemas';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

// =============================================================================
// TYPES
// =============================================================================

export interface AuditHistoryProps {
  /** Maximum workflows to display */
  limit?: number;
  /** Filter by brand */
  brand?: string;
  /** Show compact version */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function getTierColor(tier: number): string {
  const colors: Record<number, string> = {
    0: 'bg-slate-500',
    1: 'bg-blue-500',
    2: 'bg-purple-500',
    3: 'bg-amber-500',
    4: 'bg-emerald-500',
    5: 'bg-rose-500',
  };
  return colors[tier] || 'bg-gray-500';
}

function getTierName(tier: number): string {
  const names: Record<number, string> = {
    0: 'Foundation',
    1: 'Orchestration',
    2: 'Causal',
    3: 'Monitoring',
    4: 'ML Prediction',
    5: 'Learning',
  };
  return names[tier] || `Tier ${tier}`;
}

// =============================================================================
// SUB-COMPONENTS
// =============================================================================

interface WorkflowItemProps {
  workflow: RecentWorkflowValidated;
  isExpanded: boolean;
  onToggle: () => void;
}

function WorkflowItem({ workflow, isExpanded, onToggle }: WorkflowItemProps) {
  // Fetch entries when expanded
  const { data: entries, isLoading: entriesLoading } = useQuery({
    queryKey: ['audit-entries', workflow.workflow_id],
    queryFn: () =>
      getValidated(
        AuditEntriesResponseSchema,
        `/audit/workflow/${workflow.workflow_id}`
      ),
    enabled: isExpanded,
    staleTime: 30000,
  });

  // Fetch verification when expanded
  const { data: verification, isLoading: verifyLoading } = useQuery({
    queryKey: ['audit-verify', workflow.workflow_id],
    queryFn: () =>
      getValidated(
        ChainVerificationSchema,
        `/audit/workflow/${workflow.workflow_id}/verify`
      ),
    enabled: isExpanded,
    staleTime: 60000,
  });

  // Fetch summary when expanded
  const { data: summary } = useQuery({
    queryKey: ['audit-summary', workflow.workflow_id],
    queryFn: () =>
      getValidated(
        WorkflowSummarySchema,
        `/audit/workflow/${workflow.workflow_id}/summary`
      ),
    enabled: isExpanded,
    staleTime: 30000,
  });

  return (
    <Collapsible open={isExpanded} onOpenChange={onToggle}>
      <CollapsibleTrigger asChild>
        <div
          className={cn(
            'flex items-center justify-between p-3 rounded-lg cursor-pointer transition-colors',
            'hover:bg-muted/50',
            isExpanded && 'bg-muted/30'
          )}
        >
          <div className="flex items-center gap-3">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}

            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">
                  {workflow.first_agent}
                </span>
                {workflow.first_agent !== workflow.last_agent && (
                  <>
                    <span className="text-muted-foreground text-xs">→</span>
                    <span className="text-sm">{workflow.last_agent}</span>
                  </>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                <span>{formatRelativeTime(workflow.started_at)}</span>
                {workflow.brand && (
                  <>
                    <span>·</span>
                    <span>{workflow.brand}</span>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="text-xs">
              {workflow.entry_count} entries
            </Badge>
          </div>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="ml-7 pl-4 border-l-2 border-muted pb-2 space-y-3">
          {/* Verification Status */}
          {verifyLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Verifying chain integrity...
            </div>
          ) : verification ? (
            <div
              className={cn(
                'flex items-center gap-2 p-2 rounded-md text-sm',
                verification.is_valid
                  ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                  : 'bg-red-500/10 text-red-700 dark:text-red-400'
              )}
            >
              {verification.is_valid ? (
                <>
                  <Shield className="h-4 w-4" />
                  Chain verified ({verification.entries_checked} entries)
                </>
              ) : (
                <>
                  <AlertCircle className="h-4 w-4" />
                  Chain invalid: {verification.error_message}
                </>
              )}
            </div>
          ) : null}

          {/* Summary Stats */}
          {summary && (
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="p-2 rounded-md bg-muted/50">
                <div className="text-muted-foreground">Duration</div>
                <div className="font-medium">
                  {formatDuration(summary.total_duration_ms)}
                </div>
              </div>
              <div className="p-2 rounded-md bg-muted/50">
                <div className="text-muted-foreground">Avg Confidence</div>
                <div className="font-medium">
                  {summary.avg_confidence_score
                    ? `${(summary.avg_confidence_score * 100).toFixed(0)}%`
                    : 'N/A'}
                </div>
              </div>
              <div className="p-2 rounded-md bg-muted/50">
                <div className="text-muted-foreground">Validation</div>
                <div className="font-medium">
                  {summary.validation_passed_count}/{summary.total_entries} passed
                </div>
              </div>
            </div>
          )}

          {/* Entries Timeline */}
          {entriesLoading ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading entries...
            </div>
          ) : entries && entries.length > 0 ? (
            <div className="space-y-1">
              <div className="text-xs text-muted-foreground font-medium mb-2">
                Entry Timeline
              </div>
              {entries.slice(0, 10).map((entry, idx) => (
                <EntryItem key={entry.entry_id} entry={entry} index={idx} />
              ))}
              {entries.length > 10 && (
                <div className="text-xs text-muted-foreground pl-6 pt-1">
                  ... and {entries.length - 10} more entries
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground py-2">
              No entries found
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

interface EntryItemProps {
  entry: AuditEntryValidated;
  index: number;
}

function EntryItem({ entry, index }: EntryItemProps) {
  return (
    <div className="flex items-start gap-2 py-1.5 px-2 rounded hover:bg-muted/30 transition-colors">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger>
            <div
              className={cn(
                'w-5 h-5 rounded-full flex items-center justify-center text-[10px] text-white font-medium',
                getTierColor(entry.agent_tier)
              )}
            >
              {entry.sequence_number}
            </div>
          </TooltipTrigger>
          <TooltipContent side="left">
            <div className="text-xs">
              <div className="font-medium">{getTierName(entry.agent_tier)}</div>
              <div className="text-muted-foreground">Tier {entry.agent_tier}</div>
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">{entry.agent_name}</span>
          <span className="text-xs text-muted-foreground">{entry.action_type}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {entry.duration_ms !== null && entry.duration_ms !== undefined && (
            <span>{formatDuration(entry.duration_ms)}</span>
          )}
          {entry.confidence_score !== null && entry.confidence_score !== undefined && (
            <>
              <span>·</span>
              <span>{(entry.confidence_score * 100).toFixed(0)}% conf</span>
            </>
          )}
        </div>
      </div>

      {entry.validation_passed !== null && entry.validation_passed !== undefined && (
        <div className="flex-shrink-0">
          {entry.validation_passed ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          ) : (
            <XCircle className="h-4 w-4 text-red-500" />
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

/**
 * AuditHistory displays a list of recent audit workflows with
 * expandable details and chain verification status.
 *
 * @example
 * ```tsx
 * <AuditHistory limit={10} brand="Remibrutinib" />
 * ```
 */
export function AuditHistory({
  limit = 20,
  brand,
  compact = false,
  className,
}: AuditHistoryProps) {
  const [expandedId, setExpandedId] = React.useState<string | null>(null);

  // Fetch recent workflows
  const {
    data: workflows,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['audit-recent', limit, brand],
    queryFn: () =>
      getValidated(RecentWorkflowsResponseSchema, '/audit/recent', {
        limit,
        brand,
      }),
    staleTime: 30000,
    refetchInterval: 60000, // Auto-refresh every minute
  });

  const handleToggle = (workflowId: string) => {
    setExpandedId((prev) => (prev === workflowId ? null : workflowId));
  };

  if (compact) {
    return (
      <div className={cn('space-y-2', className)}>
        {isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground p-4">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading audit history...
          </div>
        ) : isError ? (
          <div className="text-sm text-red-500 p-4">
            Failed to load audit history
          </div>
        ) : workflows && workflows.length > 0 ? (
          workflows.slice(0, 5).map((workflow) => (
            <WorkflowItem
              key={workflow.workflow_id}
              workflow={workflow}
              isExpanded={expandedId === workflow.workflow_id}
              onToggle={() => handleToggle(workflow.workflow_id)}
            />
          ))
        ) : (
          <div className="text-sm text-muted-foreground p-4">
            No audit records found
          </div>
        )}
      </div>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-lg">Audit History</CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RefreshCw
              className={cn('h-4 w-4', isLoading && 'animate-spin')}
            />
          </Button>
        </div>
        <CardDescription>
          Recent agent workflows with chain verification
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-1">
        {isLoading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span>Loading audit history...</span>
          </div>
        ) : isError ? (
          <div className="flex flex-col items-center gap-2 py-8 text-red-500">
            <AlertCircle className="h-6 w-6" />
            <span>Failed to load audit history</span>
            <span className="text-xs text-muted-foreground">
              {error instanceof Error ? error.message : 'Unknown error'}
            </span>
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              Retry
            </Button>
          </div>
        ) : workflows && workflows.length > 0 ? (
          <>
            {workflows.map((workflow) => (
              <WorkflowItem
                key={workflow.workflow_id}
                workflow={workflow}
                isExpanded={expandedId === workflow.workflow_id}
                onToggle={() => handleToggle(workflow.workflow_id)}
              />
            ))}
            {workflows.length === limit && (
              <div className="text-center text-xs text-muted-foreground pt-2">
                Showing {limit} most recent workflows
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 py-8 text-muted-foreground">
            <Shield className="h-8 w-8" />
            <span>No audit records found</span>
            <span className="text-xs">
              Agent workflows will appear here once executed
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default AuditHistory;
