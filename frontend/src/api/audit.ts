/**
 * Audit Chain API Client
 * ======================
 *
 * TypeScript API client functions for the E2I Audit Chain endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * The audit chain provides cryptographic proof of agent execution
 * sequence and data integrity using hash chains.
 *
 * Endpoints:
 * - Workflow entry retrieval
 * - Chain integrity verification
 * - Workflow summary aggregation
 * - Recent workflow listing
 *
 * @module api/audit
 */

import { get } from '@/lib/api-client';
import type {
  AuditEntryResponse,
  ChainVerificationResponse,
  ListRecentWorkflowsParams,
  ListWorkflowEntriesParams,
  RecentWorkflowResponse,
  WorkflowSummaryResponse,
} from '@/types/audit';

// =============================================================================
// AUDIT CHAIN API ENDPOINTS
// =============================================================================

const AUDIT_BASE = '/audit';

// =============================================================================
// WORKFLOW ENTRY ENDPOINTS
// =============================================================================

/**
 * Get all audit entries for a workflow.
 *
 * Retrieves audit chain entries ordered by sequence number.
 * Each entry contains agent action, hash chain data, and metrics.
 *
 * @param workflowId - Workflow UUID
 * @param params - Optional pagination parameters
 * @returns List of audit entries in sequence order
 *
 * @example
 * ```typescript
 * const entries = await getWorkflowEntries('abc123-uuid');
 * entries.forEach(entry => {
 *   console.log(`[${entry.sequence_number}] ${entry.agent_name}: ${entry.action_type}`);
 * });
 * ```
 */
export async function getWorkflowEntries(
  workflowId: string,
  params?: ListWorkflowEntriesParams
): Promise<AuditEntryResponse[]> {
  return get<AuditEntryResponse[]>(
    `${AUDIT_BASE}/workflow/${encodeURIComponent(workflowId)}`,
    params as Record<string, unknown> | undefined
  );
}

/**
 * Verify the integrity of a workflow's audit chain.
 *
 * Performs cryptographic verification by validating hash chains
 * between consecutive entries.
 *
 * @param workflowId - Workflow UUID to verify
 * @returns Verification result with validity status
 *
 * @example
 * ```typescript
 * const result = await verifyWorkflowChain('abc123-uuid');
 * if (result.is_valid) {
 *   console.log(`Chain verified: ${result.entries_checked} entries`);
 * } else {
 *   console.error(`Chain broken at entry: ${result.first_invalid_entry}`);
 *   console.error(`Error: ${result.error_message}`);
 * }
 * ```
 */
export async function verifyWorkflowChain(
  workflowId: string
): Promise<ChainVerificationResponse> {
  return get<ChainVerificationResponse>(
    `${AUDIT_BASE}/workflow/${encodeURIComponent(workflowId)}/verify`
  );
}

/**
 * Get a summary of a workflow's audit chain.
 *
 * Returns aggregated metrics including agents involved, tiers,
 * total duration, and validation statistics.
 *
 * @param workflowId - Workflow UUID
 * @returns Workflow summary with aggregated metrics
 *
 * @example
 * ```typescript
 * const summary = await getWorkflowSummary('abc123-uuid');
 * console.log(`Workflow: ${summary.workflow_id}`);
 * console.log(`Entries: ${summary.total_entries}`);
 * console.log(`Agents: ${summary.agents_involved.join(', ')}`);
 * console.log(`Duration: ${summary.total_duration_ms}ms`);
 * console.log(`Chain valid: ${summary.chain_verified}`);
 * ```
 */
export async function getWorkflowSummary(
  workflowId: string
): Promise<WorkflowSummaryResponse> {
  return get<WorkflowSummaryResponse>(
    `${AUDIT_BASE}/workflow/${encodeURIComponent(workflowId)}/summary`
  );
}

// =============================================================================
// RECENT WORKFLOWS ENDPOINTS
// =============================================================================

/**
 * Get a list of recent audit workflows.
 *
 * Returns workflows with basic info, ordered by start time descending.
 * Can filter by brand or agent name.
 *
 * @param params - Optional filter and pagination parameters
 * @returns List of recent workflows
 *
 * @example
 * ```typescript
 * // Get all recent workflows
 * const recent = await getRecentWorkflows({ limit: 10 });
 * recent.forEach(w => {
 *   console.log(`${w.workflow_id}: ${w.first_agent} -> ${w.last_agent}`);
 * });
 *
 * // Filter by brand
 * const brandWorkflows = await getRecentWorkflows({
 *   brand: 'Remibrutinib',
 *   limit: 20,
 * });
 * ```
 */
export async function getRecentWorkflows(
  params?: ListRecentWorkflowsParams
): Promise<RecentWorkflowResponse[]> {
  return get<RecentWorkflowResponse[]>(`${AUDIT_BASE}/recent`, params as Record<string, unknown> | undefined);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Get workflow details including entries, summary, and verification.
 *
 * Fetches all workflow data in parallel for comprehensive view.
 *
 * @param workflowId - Workflow UUID
 * @returns Complete workflow data
 *
 * @example
 * ```typescript
 * const workflow = await getWorkflowDetails('abc123-uuid');
 * console.log(`Summary: ${workflow.summary.total_entries} entries`);
 * console.log(`Verified: ${workflow.verification.is_valid}`);
 * workflow.entries.forEach(e => console.log(e.agent_name));
 * ```
 */
export async function getWorkflowDetails(workflowId: string): Promise<{
  entries: AuditEntryResponse[];
  summary: WorkflowSummaryResponse;
  verification: ChainVerificationResponse;
}> {
  const [entries, summary, verification] = await Promise.all([
    getWorkflowEntries(workflowId),
    getWorkflowSummary(workflowId),
    verifyWorkflowChain(workflowId),
  ]);

  return { entries, summary, verification };
}

/**
 * Get workflow by first agent.
 *
 * Filters recent workflows to find those starting with a specific agent.
 *
 * @param agentName - Agent name to filter by
 * @param limit - Maximum workflows to return
 * @returns Workflows starting with the specified agent
 *
 * @example
 * ```typescript
 * const orchestratorWorkflows = await getWorkflowsByFirstAgent('orchestrator', 10);
 * console.log(`Found ${orchestratorWorkflows.length} workflows`);
 * ```
 */
export async function getWorkflowsByFirstAgent(
  agentName: string,
  limit: number = 20
): Promise<RecentWorkflowResponse[]> {
  const workflows = await getRecentWorkflows({ agent_name: agentName, limit });
  return workflows.filter((w) => w.first_agent === agentName);
}

/**
 * Get workflow by brand.
 *
 * Filters recent workflows by pharmaceutical brand.
 *
 * @param brand - Brand name to filter by
 * @param limit - Maximum workflows to return
 * @returns Workflows for the specified brand
 *
 * @example
 * ```typescript
 * const remibWorkflows = await getWorkflowsByBrand('Remibrutinib', 10);
 * console.log(`Found ${remibWorkflows.length} Remibrutinib workflows`);
 * ```
 */
export async function getWorkflowsByBrand(
  brand: string,
  limit: number = 20
): Promise<RecentWorkflowResponse[]> {
  return getRecentWorkflows({ brand, limit });
}

/**
 * Check if a workflow chain is valid.
 *
 * Simple boolean check for chain integrity.
 *
 * @param workflowId - Workflow UUID
 * @returns True if chain is valid, false otherwise
 *
 * @example
 * ```typescript
 * const isValid = await isWorkflowChainValid('abc123-uuid');
 * if (!isValid) {
 *   alert('Workflow audit chain has been tampered with!');
 * }
 * ```
 */
export async function isWorkflowChainValid(workflowId: string): Promise<boolean> {
  try {
    const result = await verifyWorkflowChain(workflowId);
    return result.is_valid;
  } catch {
    return false;
  }
}

/**
 * Get agent execution path for a workflow.
 *
 * Returns the sequence of agents that executed in the workflow.
 *
 * @param workflowId - Workflow UUID
 * @returns Array of agent names in execution order
 *
 * @example
 * ```typescript
 * const path = await getAgentExecutionPath('abc123-uuid');
 * console.log(`Execution path: ${path.join(' -> ')}`);
 * // "orchestrator -> causal_impact -> explainer"
 * ```
 */
export async function getAgentExecutionPath(workflowId: string): Promise<string[]> {
  const entries = await getWorkflowEntries(workflowId);
  return entries.map((e) => e.agent_name);
}

/**
 * Get tier distribution for a workflow.
 *
 * Returns count of entries by agent tier.
 *
 * @param workflowId - Workflow UUID
 * @returns Map of tier number to entry count
 *
 * @example
 * ```typescript
 * const distribution = await getTierDistribution('abc123-uuid');
 * Object.entries(distribution).forEach(([tier, count]) => {
 *   console.log(`Tier ${tier}: ${count} entries`);
 * });
 * ```
 */
export async function getTierDistribution(
  workflowId: string
): Promise<Record<number, number>> {
  const entries = await getWorkflowEntries(workflowId);
  const distribution: Record<number, number> = {};

  for (const entry of entries) {
    distribution[entry.agent_tier] = (distribution[entry.agent_tier] || 0) + 1;
  }

  return distribution;
}

/**
 * Get total duration for a workflow.
 *
 * Sums up all entry durations.
 *
 * @param workflowId - Workflow UUID
 * @returns Total duration in milliseconds
 *
 * @example
 * ```typescript
 * const duration = await getWorkflowDuration('abc123-uuid');
 * console.log(`Total duration: ${duration}ms`);
 * ```
 */
export async function getWorkflowDuration(workflowId: string): Promise<number> {
  const summary = await getWorkflowSummary(workflowId);
  return summary.total_duration_ms;
}

/**
 * Get workflow time range.
 *
 * Returns the start and end timestamps for a workflow.
 *
 * @param workflowId - Workflow UUID
 * @returns Start and end timestamps
 *
 * @example
 * ```typescript
 * const range = await getWorkflowTimeRange('abc123-uuid');
 * console.log(`Started: ${range.start}`);
 * console.log(`Ended: ${range.end}`);
 * ```
 */
export async function getWorkflowTimeRange(workflowId: string): Promise<{
  start: string | null;
  end: string | null;
}> {
  const summary = await getWorkflowSummary(workflowId);
  return {
    start: summary.first_entry_at || null,
    end: summary.last_entry_at || null,
  };
}

/**
 * Get entries with failed validation.
 *
 * Filters workflow entries to only those that failed validation.
 *
 * @param workflowId - Workflow UUID
 * @returns Entries that failed validation
 *
 * @example
 * ```typescript
 * const failed = await getFailedValidationEntries('abc123-uuid');
 * if (failed.length > 0) {
 *   console.warn(`${failed.length} entries failed validation`);
 *   failed.forEach(e => console.warn(`  ${e.agent_name}: ${e.action_type}`));
 * }
 * ```
 */
export async function getFailedValidationEntries(
  workflowId: string
): Promise<AuditEntryResponse[]> {
  const entries = await getWorkflowEntries(workflowId);
  return entries.filter((e) => e.validation_passed === false);
}

/**
 * Get entries with low confidence.
 *
 * Filters workflow entries to those with confidence below threshold.
 *
 * @param workflowId - Workflow UUID
 * @param threshold - Confidence threshold (0-1, default: 0.7)
 * @returns Entries with confidence below threshold
 *
 * @example
 * ```typescript
 * const lowConfidence = await getLowConfidenceEntries('abc123-uuid', 0.8);
 * lowConfidence.forEach(e => {
 *   console.log(`${e.agent_name}: ${e.confidence_score}`);
 * });
 * ```
 */
export async function getLowConfidenceEntries(
  workflowId: string,
  threshold: number = 0.7
): Promise<AuditEntryResponse[]> {
  const entries = await getWorkflowEntries(workflowId);
  return entries.filter(
    (e) => e.confidence_score !== undefined && e.confidence_score < threshold
  );
}

/**
 * Format duration for display.
 *
 * Converts milliseconds to human-readable format.
 *
 * @param durationMs - Duration in milliseconds
 * @returns Formatted duration string
 *
 * @example
 * ```typescript
 * console.log(formatDuration(1500)); // "1.5s"
 * console.log(formatDuration(150)); // "150ms"
 * console.log(formatDuration(65000)); // "1m 5s"
 * ```
 */
export function formatDuration(durationMs: number): string {
  if (durationMs < 1000) {
    return `${durationMs}ms`;
  }
  if (durationMs < 60000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  }
  const minutes = Math.floor(durationMs / 60000);
  const seconds = Math.round((durationMs % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Get entry hash chain status.
 *
 * Checks if each entry's hash chain is properly linked.
 *
 * @param entries - Array of audit entries
 * @returns Array of boolean indicating if each entry is properly linked
 *
 * @example
 * ```typescript
 * const entries = await getWorkflowEntries('abc123-uuid');
 * const chainStatus = getHashChainStatus(entries);
 * const brokenLinks = chainStatus.filter(s => !s.valid);
 * if (brokenLinks.length > 0) {
 *   console.error('Chain has broken links!');
 * }
 * ```
 */
export function getHashChainStatus(entries: AuditEntryResponse[]): Array<{
  entry_id: string;
  sequence_number: number;
  valid: boolean;
}> {
  return entries.map((entry, index) => {
    const valid =
      index === 0
        ? entry.previous_entry_id === undefined || entry.previous_entry_id === null
        : entry.previous_entry_id === entries[index - 1].entry_id;

    return {
      entry_id: entry.entry_id,
      sequence_number: entry.sequence_number,
      valid,
    };
  });
}
