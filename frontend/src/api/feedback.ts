/**
 * Feedback Learning API Client
 * ============================
 *
 * TypeScript API client functions for the E2I Feedback Learning endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Learning cycle execution
 * - Feedback processing
 * - Pattern listing and management
 * - Knowledge updates management
 * - Service health
 *
 * @module api/feedback
 */

import { get, post } from '@/lib/api-client';
import type {
  ApplyUpdateRequest,
  FeedbackHealthResponse,
  KnowledgeUpdate,
  LearningResponse,
  ListPatternsParams,
  ListUpdatesParams,
  PatternListResponse,
  ProcessFeedbackRequest,
  RunLearningRequest,
  UpdateListResponse,
} from '@/types/feedback';

// =============================================================================
// FEEDBACK API ENDPOINTS
// =============================================================================

const FEEDBACK_BASE = '/feedback';

// =============================================================================
// LEARNING CYCLE ENDPOINTS
// =============================================================================

/**
 * Run a feedback learning cycle.
 *
 * Processes accumulated feedback from the specified time range to detect
 * patterns, generate recommendations, and propose knowledge updates.
 *
 * @param request - Learning cycle parameters
 * @param asyncMode - If true, returns immediately with batch ID (default: true)
 * @returns Learning results or pending status
 *
 * @example
 * ```typescript
 * const result = await runLearningCycle({
 *   time_range_start: '2024-01-01T00:00:00Z',
 *   focus_agents: ['causal_impact', 'gap_analyzer'],
 *   min_feedback_count: 20,
 *   pattern_threshold: 0.15,
 * });
 *
 * if (result.status === LearningStatus.COMPLETED) {
 *   console.log(`Found ${result.patterns_detected} patterns`);
 *   result.detected_patterns.forEach(p => {
 *     console.log(`${p.pattern_type}: ${p.description}`);
 *   });
 * }
 * ```
 */
export async function runLearningCycle(
  request: RunLearningRequest,
  asyncMode: boolean = true
): Promise<LearningResponse> {
  return post<LearningResponse, RunLearningRequest>(
    `${FEEDBACK_BASE}/learn`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get learning cycle results by batch ID.
 *
 * Use this to poll for results when running learning asynchronously.
 *
 * @param batchId - Unique batch identifier
 * @returns Learning results
 *
 * @example
 * ```typescript
 * const result = await getLearningResults('fb_abc123');
 * if (result.status === LearningStatus.COMPLETED) {
 *   console.log(result.learning_summary);
 * }
 * ```
 */
export async function getLearningResults(
  batchId: string
): Promise<LearningResponse> {
  return get<LearningResponse>(
    `${FEEDBACK_BASE}/${encodeURIComponent(batchId)}`
  );
}

/**
 * Process specific feedback items.
 *
 * Allows processing individual feedback items without running a full
 * learning cycle. Can detect patterns and generate recommendations.
 *
 * @param request - Feedback items and processing options
 * @returns Processing results with any detected patterns
 *
 * @example
 * ```typescript
 * const result = await processFeedback({
 *   items: [
 *     {
 *       feedback_type: FeedbackType.RATING,
 *       source_agent: 'causal_impact',
 *       query: 'What drives TRx?',
 *       agent_response: 'Based on analysis...',
 *       user_feedback: { rating: 4, helpful: true },
 *     },
 *   ],
 *   detect_patterns: true,
 *   generate_recommendations: true,
 * });
 *
 * console.log(`Processed ${result.feedback_summary?.total_feedback_items} items`);
 * ```
 */
export async function processFeedback(
  request: ProcessFeedbackRequest
): Promise<LearningResponse> {
  return post<LearningResponse, ProcessFeedbackRequest>(
    `${FEEDBACK_BASE}/process`,
    request
  );
}

// =============================================================================
// PATTERN ENDPOINTS
// =============================================================================

/**
 * List detected patterns.
 *
 * Returns all detected patterns with optional filtering by severity,
 * type, or affected agent.
 *
 * @param params - Optional filter parameters
 * @returns List of patterns matching filters
 *
 * @example
 * ```typescript
 * // Get all critical patterns
 * const critical = await listPatterns({
 *   severity: PatternSeverity.CRITICAL,
 * });
 *
 * console.log(`${critical.critical_count} critical patterns found`);
 * critical.patterns.forEach(p => {
 *   console.log(`${p.description} (${p.frequency} occurrences)`);
 * });
 * ```
 */
export async function listPatterns(
  params?: ListPatternsParams
): Promise<PatternListResponse> {
  return get<PatternListResponse>(`${FEEDBACK_BASE}/patterns`, params as Record<string, unknown> | undefined);
}

// =============================================================================
// KNOWLEDGE UPDATE ENDPOINTS
// =============================================================================

/**
 * List knowledge updates.
 *
 * Returns all proposed and applied knowledge updates with optional filtering.
 *
 * @param params - Optional filter parameters
 * @returns List of updates matching filters
 *
 * @example
 * ```typescript
 * // Get pending updates
 * const pending = await listUpdates({
 *   status: UpdateStatus.PROPOSED,
 * });
 *
 * console.log(`${pending.proposed_count} updates awaiting approval`);
 * ```
 */
export async function listUpdates(
  params?: ListUpdatesParams
): Promise<UpdateListResponse> {
  return get<UpdateListResponse>(`${FEEDBACK_BASE}/updates`, params as Record<string, unknown> | undefined);
}

/**
 * Apply a knowledge update.
 *
 * Applies a proposed knowledge update to the system.
 *
 * @param updateId - Update identifier
 * @param force - Force apply even if not approved (default: false)
 * @returns Updated knowledge update record
 *
 * @example
 * ```typescript
 * const applied = await applyUpdate('upd_abc123');
 * console.log(`Update ${applied.update_id} applied at ${applied.applied_at}`);
 * ```
 */
export async function applyUpdate(
  updateId: string,
  force: boolean = false
): Promise<KnowledgeUpdate> {
  return post<KnowledgeUpdate, ApplyUpdateRequest>(
    `${FEEDBACK_BASE}/updates/${encodeURIComponent(updateId)}/apply`,
    { update_id: updateId, force }
  );
}

/**
 * Rollback a knowledge update.
 *
 * Reverts a previously applied knowledge update.
 *
 * @param updateId - Update identifier
 * @returns Updated knowledge update record
 *
 * @example
 * ```typescript
 * const rolledBack = await rollbackUpdate('upd_abc123');
 * console.log(`Update ${rolledBack.update_id} rolled back`);
 * ```
 */
export async function rollbackUpdate(
  updateId: string
): Promise<KnowledgeUpdate> {
  return post<KnowledgeUpdate, Record<string, never>>(
    `${FEEDBACK_BASE}/updates/${encodeURIComponent(updateId)}/rollback`,
    {}
  );
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Get health status of feedback learning service.
 *
 * Checks agent availability, recent activity, and pending items.
 *
 * @returns Service health information
 *
 * @example
 * ```typescript
 * const health = await getFeedbackHealth();
 * if (health.status === 'healthy') {
 *   console.log(`${health.cycles_24h} learning cycles in last 24h`);
 *   console.log(`${health.patterns_active} active patterns`);
 *   console.log(`${health.pending_updates} pending updates`);
 * }
 * ```
 */
export async function getFeedbackHealth(): Promise<FeedbackHealthResponse> {
  return get<FeedbackHealthResponse>(`${FEEDBACK_BASE}/health`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run learning cycle and poll until complete.
 *
 * Convenience function that handles async polling automatically.
 *
 * @param request - Learning parameters
 * @param pollIntervalMs - Polling interval in milliseconds (default: 2000)
 * @param maxWaitMs - Maximum wait time in milliseconds (default: 120000)
 * @returns Completed learning results
 * @throws Error if learning fails or times out
 *
 * @example
 * ```typescript
 * try {
 *   const result = await runLearningCycleAndWait({
 *     focus_agents: ['causal_impact'],
 *     min_feedback_count: 10,
 *   });
 *   console.log(result.learning_summary);
 * } catch (error) {
 *   console.error('Learning cycle failed:', error);
 * }
 * ```
 */
export async function runLearningCycleAndWait(
  request: RunLearningRequest,
  pollIntervalMs: number = 2000,
  maxWaitMs: number = 120000
): Promise<LearningResponse> {
  // Start learning asynchronously
  const initial = await runLearningCycle(request, true);

  // If already complete, return immediately
  if (initial.status === 'completed' || initial.status === 'failed') {
    if (initial.status === 'failed') {
      throw new Error(
        `Learning cycle failed: ${initial.errors.join(', ') || 'Unknown error'}`
      );
    }
    return initial;
  }

  // Poll until complete or timeout
  const startTime = Date.now();
  const batchId = initial.batch_id;

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

    const result = await getLearningResults(batchId);

    if (result.status === 'completed') {
      return result;
    }

    if (result.status === 'failed') {
      throw new Error(
        `Learning cycle failed: ${result.errors.join(', ') || 'Unknown error'}`
      );
    }
  }

  throw new Error(`Learning cycle timed out after ${maxWaitMs}ms`);
}

/**
 * Get comprehensive feedback status.
 *
 * Returns health status along with recent patterns and pending updates.
 *
 * @returns Combined status information
 *
 * @example
 * ```typescript
 * const status = await getFeedbackStatus();
 * console.log(`Health: ${status.health.status}`);
 * console.log(`Critical patterns: ${status.patterns.critical_count}`);
 * console.log(`Pending updates: ${status.updates.proposed_count}`);
 * ```
 */
export async function getFeedbackStatus(): Promise<{
  health: FeedbackHealthResponse;
  patterns: PatternListResponse;
  updates: UpdateListResponse;
}> {
  const [health, patterns, updates] = await Promise.all([
    getFeedbackHealth(),
    listPatterns({ limit: 10 }),
    listUpdates({ limit: 10 }),
  ]);

  return { health, patterns, updates };
}

/**
 * Run quick feedback analysis on recent items.
 *
 * Simplified interface for processing feedback from the last 24 hours.
 *
 * @param focusAgents - Optional agents to focus on
 * @returns Learning results
 *
 * @example
 * ```typescript
 * const result = await quickLearningCycle(['causal_impact', 'gap_analyzer']);
 * console.log(`Found ${result.patterns_detected} patterns`);
 * ```
 */
export async function quickLearningCycle(
  focusAgents?: string[]
): Promise<LearningResponse> {
  return runLearningCycle(
    {
      focus_agents: focusAgents,
      min_feedback_count: 5,
      pattern_threshold: 0.1,
      auto_apply: false,
    },
    false // Run synchronously for quick analysis
  );
}

/**
 * Apply all approved updates.
 *
 * Batch applies all updates that have been approved.
 *
 * @returns Array of applied updates
 *
 * @example
 * ```typescript
 * const applied = await applyAllApprovedUpdates();
 * console.log(`Applied ${applied.length} updates`);
 * ```
 */
export async function applyAllApprovedUpdates(): Promise<KnowledgeUpdate[]> {
  const { updates } = await listUpdates({ status: 'approved' as never });

  const applied: KnowledgeUpdate[] = [];
  for (const update of updates) {
    try {
      const result = await applyUpdate(update.update_id);
      applied.push(result);
    } catch (error) {
      console.error(`Failed to apply update ${update.update_id}:`, error);
    }
  }

  return applied;
}

/**
 * Get pattern details by ID.
 *
 * Finds a specific pattern from the list of detected patterns.
 *
 * @param patternId - Pattern identifier
 * @returns Pattern details or undefined if not found
 *
 * @example
 * ```typescript
 * const pattern = await getPattern('pat_abc123');
 * if (pattern) {
 *   console.log(`Pattern: ${pattern.description}`);
 *   console.log(`Affected agents: ${pattern.affected_agents.join(', ')}`);
 * }
 * ```
 */
export async function getPattern(
  patternId: string
): Promise<DetectedPattern | undefined> {
  const { patterns } = await listPatterns({ limit: 200 });
  return patterns.find((p) => p.pattern_id === patternId);
}

// Re-export DetectedPattern type for convenience
import type { DetectedPattern } from '@/types/feedback';
export type { DetectedPattern };
