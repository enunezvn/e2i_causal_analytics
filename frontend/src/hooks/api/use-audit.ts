/**
 * Audit Chain React Query Hooks
 * =============================
 *
 * TanStack Query hooks for the Audit Chain API endpoints.
 * Provides typed query hooks for workflow audit trail, verification,
 * and compliance tracking.
 *
 * The audit chain provides cryptographic proof of agent execution
 * sequence and data integrity using hash chains.
 *
 * @module hooks/api/use-audit
 */

import { useQuery } from '@tanstack/react-query';
import type { UseQueryOptions } from '@tanstack/react-query';
import { queryKeys } from '@/lib/query-client';
import { ApiError } from '@/lib/api-client';
import {
  getWorkflowEntries,
  verifyWorkflowChain,
  getWorkflowSummary,
  getRecentWorkflows,
  getWorkflowDetails,
  getAgentExecutionPath,
  getTierDistribution,
  getFailedValidationEntries,
  getLowConfidenceEntries,
} from '@/api/audit';
import type {
  AuditEntryResponse,
  ChainVerificationResponse,
  WorkflowSummaryResponse,
  RecentWorkflowResponse,
  ListWorkflowEntriesParams,
  ListRecentWorkflowsParams,
} from '@/types/audit';

// =============================================================================
// QUERY HOOKS
// =============================================================================

/**
 * Hook to fetch audit entries for a workflow.
 *
 * @param workflowId - The workflow UUID
 * @param params - Optional pagination parameters
 * @param options - Additional query options
 * @returns Query result with audit entries in sequence order
 *
 * @example
 * ```tsx
 * const { data: entries, isLoading } = useWorkflowEntries('abc123-uuid');
 * entries?.forEach(entry => {
 *   console.log(`[${entry.sequence_number}] ${entry.agent_name}`);
 * });
 * ```
 */
export function useWorkflowEntries(
  workflowId: string,
  params?: ListWorkflowEntriesParams,
  options?: Omit<UseQueryOptions<AuditEntryResponse[], ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AuditEntryResponse[], ApiError>({
    queryKey: [...queryKeys.audit.workflow(workflowId), params?.limit, params?.offset],
    queryFn: () => getWorkflowEntries(workflowId, params),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes - audit data doesn't change
    ...options,
  });
}

/**
 * Hook to verify a workflow's audit chain integrity.
 *
 * @param workflowId - The workflow UUID to verify
 * @param options - Additional query options
 * @returns Query result with verification status
 *
 * @example
 * ```tsx
 * const { data: verification } = useWorkflowVerification('abc123-uuid');
 * if (verification?.is_valid) {
 *   console.log('Chain verified!');
 * } else {
 *   console.error(`Chain broken at: ${verification?.first_invalid_entry}`);
 * }
 * ```
 */
export function useWorkflowVerification(
  workflowId: string,
  options?: Omit<UseQueryOptions<ChainVerificationResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<ChainVerificationResponse, ApiError>({
    queryKey: queryKeys.audit.workflowVerification(workflowId),
    queryFn: () => verifyWorkflowChain(workflowId),
    enabled: !!workflowId,
    staleTime: 10 * 60 * 1000, // 10 minutes - verification result is stable
    ...options,
  });
}

/**
 * Hook to get a workflow's summary statistics.
 *
 * @param workflowId - The workflow UUID
 * @param options - Additional query options
 * @returns Query result with workflow summary
 *
 * @example
 * ```tsx
 * const { data: summary } = useWorkflowSummary('abc123-uuid');
 * console.log(`${summary?.total_entries} entries, ${summary?.total_duration_ms}ms`);
 * ```
 */
export function useWorkflowSummary(
  workflowId: string,
  options?: Omit<UseQueryOptions<WorkflowSummaryResponse, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<WorkflowSummaryResponse, ApiError>({
    queryKey: queryKeys.audit.workflowSummary(workflowId),
    queryFn: () => getWorkflowSummary(workflowId),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to list recent audit workflows.
 *
 * @param params - Optional filter parameters (brand, agent_name, limit)
 * @param options - Additional query options
 * @returns Query result with recent workflows
 *
 * @example
 * ```tsx
 * const { data: workflows } = useRecentWorkflows({ limit: 20, brand: 'Kisqali' });
 * workflows?.forEach(w => {
 *   console.log(`${w.workflow_id}: ${w.first_agent} -> ${w.last_agent}`);
 * });
 * ```
 */
export function useRecentWorkflows(
  params?: ListRecentWorkflowsParams,
  options?: Omit<UseQueryOptions<RecentWorkflowResponse[], ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<RecentWorkflowResponse[], ApiError>({
    queryKey: [...queryKeys.audit.recent(), params?.brand, params?.agent_name, params?.limit],
    queryFn: () => getRecentWorkflows(params),
    staleTime: 30 * 1000, // 30 seconds - recent list changes more often
    ...options,
  });
}

// =============================================================================
// COMPOSITE HOOKS
// =============================================================================

/**
 * Combined workflow details response type
 */
export interface WorkflowDetailsData {
  entries: AuditEntryResponse[];
  summary: WorkflowSummaryResponse;
  verification: ChainVerificationResponse;
}

/**
 * Hook to get complete workflow details (entries, summary, verification).
 *
 * @param workflowId - The workflow UUID
 * @param options - Additional query options
 * @returns Query result with all workflow data
 *
 * @example
 * ```tsx
 * const { data: details } = useWorkflowDetails('abc123-uuid');
 * console.log(`Entries: ${details?.entries.length}`);
 * console.log(`Valid: ${details?.verification.is_valid}`);
 * ```
 */
export function useWorkflowDetails(
  workflowId: string,
  options?: Omit<UseQueryOptions<WorkflowDetailsData, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<WorkflowDetailsData, ApiError>({
    queryKey: [...queryKeys.audit.workflow(workflowId), 'details'],
    queryFn: () => getWorkflowDetails(workflowId),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to get the agent execution path for a workflow.
 *
 * @param workflowId - The workflow UUID
 * @param options - Additional query options
 * @returns Query result with agent names in execution order
 *
 * @example
 * ```tsx
 * const { data: path } = useAgentExecutionPath('abc123-uuid');
 * console.log(`Path: ${path?.join(' -> ')}`);
 * // "orchestrator -> causal_impact -> explainer"
 * ```
 */
export function useAgentExecutionPath(
  workflowId: string,
  options?: Omit<UseQueryOptions<string[], ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<string[], ApiError>({
    queryKey: [...queryKeys.audit.workflow(workflowId), 'path'],
    queryFn: () => getAgentExecutionPath(workflowId),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to get tier distribution for a workflow.
 *
 * @param workflowId - The workflow UUID
 * @param options - Additional query options
 * @returns Query result with tier counts
 *
 * @example
 * ```tsx
 * const { data: distribution } = useTierDistribution('abc123-uuid');
 * Object.entries(distribution ?? {}).forEach(([tier, count]) => {
 *   console.log(`Tier ${tier}: ${count} entries`);
 * });
 * ```
 */
export function useTierDistribution(
  workflowId: string,
  options?: Omit<UseQueryOptions<Record<number, number>, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<Record<number, number>, ApiError>({
    queryKey: [...queryKeys.audit.workflow(workflowId), 'tiers'],
    queryFn: () => getTierDistribution(workflowId),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to get entries with failed validation.
 *
 * @param workflowId - The workflow UUID
 * @param options - Additional query options
 * @returns Query result with failed entries
 *
 * @example
 * ```tsx
 * const { data: failed } = useFailedValidationEntries('abc123-uuid');
 * if (failed && failed.length > 0) {
 *   console.warn(`${failed.length} entries failed validation`);
 * }
 * ```
 */
export function useFailedValidationEntries(
  workflowId: string,
  options?: Omit<UseQueryOptions<AuditEntryResponse[], ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AuditEntryResponse[], ApiError>({
    queryKey: [...queryKeys.audit.workflow(workflowId), 'failed'],
    queryFn: () => getFailedValidationEntries(workflowId),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Hook to get entries with low confidence scores.
 *
 * @param workflowId - The workflow UUID
 * @param threshold - Confidence threshold (0-1, default: 0.7)
 * @param options - Additional query options
 * @returns Query result with low confidence entries
 *
 * @example
 * ```tsx
 * const { data: lowConf } = useLowConfidenceEntries('abc123-uuid', 0.8);
 * lowConf?.forEach(e => {
 *   console.log(`${e.agent_name}: ${e.confidence_score}`);
 * });
 * ```
 */
export function useLowConfidenceEntries(
  workflowId: string,
  threshold: number = 0.7,
  options?: Omit<UseQueryOptions<AuditEntryResponse[], ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AuditEntryResponse[], ApiError>({
    queryKey: [...queryKeys.audit.workflow(workflowId), 'low-confidence', threshold],
    queryFn: () => getLowConfidenceEntries(workflowId, threshold),
    enabled: !!workflowId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

// =============================================================================
// DASHBOARD HOOKS
// =============================================================================

/**
 * Combined audit dashboard data type
 */
export interface AuditDashboardData {
  recentWorkflows: RecentWorkflowResponse[];
  totalWorkflows: number;
}

/**
 * Hook for audit dashboard - recent workflows with count.
 *
 * @param limit - Number of recent workflows to fetch
 * @param options - Additional query options
 * @returns Query result for dashboard display
 *
 * @example
 * ```tsx
 * const { data: dashboard } = useAuditDashboard(10);
 * console.log(`${dashboard?.totalWorkflows} total workflows`);
 * ```
 */
export function useAuditDashboard(
  limit: number = 20,
  options?: Omit<UseQueryOptions<AuditDashboardData, ApiError>, 'queryKey' | 'queryFn'>
) {
  return useQuery<AuditDashboardData, ApiError>({
    queryKey: [...queryKeys.audit.recent(), 'dashboard', limit],
    queryFn: async () => {
      const recentWorkflows = await getRecentWorkflows({ limit });
      return {
        recentWorkflows,
        totalWorkflows: recentWorkflows.length,
      };
    },
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}
