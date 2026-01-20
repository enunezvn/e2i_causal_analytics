/**
 * Resource Optimization API Client
 * =================================
 *
 * TypeScript API client functions for the E2I Resource Optimization endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Resource optimization execution
 * - Scenario analysis
 * - Service health
 *
 * @module api/resources
 */

import { get, post } from '@/lib/api-client';
import type {
  ListScenariosParams,
  OptimizationResponse,
  RunOptimizationRequest,
  ResourceHealthResponse,
  ScenarioListResponse,
} from '@/types/resources';

// =============================================================================
// RESOURCE OPTIMIZATION API ENDPOINTS
// =============================================================================

const RESOURCES_BASE = '/resources';

// =============================================================================
// OPTIMIZATION ENDPOINTS
// =============================================================================

/**
 * Run resource optimization.
 *
 * Invokes the Resource Optimizer agent (Tier 4) to optimize resource
 * allocation across entities using mathematical optimization.
 *
 * @param request - Optimization parameters
 * @param asyncMode - If true, returns immediately with optimization ID (default: true)
 * @returns Optimization results or pending status
 *
 * @example
 * ```typescript
 * const result = await runOptimization({
 *   query: 'Optimize budget allocation across territories',
 *   resource_type: ResourceType.BUDGET,
 *   allocation_targets: [
 *     {
 *       entity_id: 'territory_northeast',
 *       entity_type: 'territory',
 *       current_allocation: 50000,
 *       min_allocation: 30000,
 *       max_allocation: 80000,
 *       expected_response: 1.3,
 *     },
 *   ],
 *   constraints: [
 *     { constraint_type: ConstraintType.BUDGET, value: 200000, scope: ConstraintScope.GLOBAL }
 *   ],
 *   objective: OptimizationObjective.MAXIMIZE_OUTCOME,
 * });
 *
 * if (result.status === OptimizationStatus.COMPLETED) {
 *   console.log(`Objective value: ${result.objective_value}`);
 *   console.log(`Projected ROI: ${result.projected_roi}`);
 *   result.optimal_allocations.forEach(alloc => {
 *     console.log(`${alloc.entity_id}: ${alloc.change_percentage}% change`);
 *   });
 * }
 * ```
 */
export async function runOptimization(
  request: RunOptimizationRequest,
  asyncMode: boolean = true
): Promise<OptimizationResponse> {
  return post<OptimizationResponse, RunOptimizationRequest>(
    `${RESOURCES_BASE}/optimize`,
    request,
    { params: { async_mode: asyncMode } }
  );
}

/**
 * Get optimization results by ID.
 *
 * Use this to poll for results when running optimization asynchronously.
 *
 * @param optimizationId - Unique optimization identifier
 * @returns Optimization results
 *
 * @example
 * ```typescript
 * const result = await getOptimization('opt_abc123');
 * if (result.status === OptimizationStatus.COMPLETED) {
 *   console.log(result.optimization_summary);
 *   result.recommendations.forEach(rec => console.log(rec));
 * }
 * ```
 */
export async function getOptimization(
  optimizationId: string
): Promise<OptimizationResponse> {
  return get<OptimizationResponse>(
    `${RESOURCES_BASE}/${encodeURIComponent(optimizationId)}`
  );
}

// =============================================================================
// SCENARIO ENDPOINTS
// =============================================================================

/**
 * List scenario analyses from optimizations.
 *
 * Returns scenario results across all completed optimizations.
 *
 * @param params - Optional filters for ROI and limit
 * @returns List of scenario analyses
 *
 * @example
 * ```typescript
 * const scenarios = await listScenarios({
 *   min_roi: 1.5,
 *   limit: 10,
 * });
 *
 * scenarios.scenarios.forEach(s => {
 *   console.log(`${s.scenario_name}: ROI ${s.roi}, Outcome ${s.projected_outcome}`);
 * });
 * ```
 */
export async function listScenarios(
  params?: ListScenariosParams
): Promise<ScenarioListResponse> {
  return get<ScenarioListResponse>(`${RESOURCES_BASE}/scenarios`, {
    min_roi: params?.min_roi,
    limit: params?.limit,
  });
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Get health status of resource optimization service.
 *
 * Checks agent and solver availability.
 *
 * @returns Service health information
 *
 * @example
 * ```typescript
 * const health = await getResourceHealth();
 * if (health.agent_available && health.scipy_available) {
 *   console.log(`Service healthy, ${health.optimizations_24h} optimizations in last 24h`);
 * } else {
 *   console.warn('Some components unavailable');
 * }
 * ```
 */
export async function getResourceHealth(): Promise<ResourceHealthResponse> {
  return get<ResourceHealthResponse>(`${RESOURCES_BASE}/health`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run optimization and poll until complete.
 *
 * Convenience function that handles async polling automatically.
 *
 * @param request - Optimization parameters
 * @param pollIntervalMs - Polling interval in milliseconds (default: 2000)
 * @param maxWaitMs - Maximum wait time in milliseconds (default: 120000)
 * @returns Completed optimization results
 * @throws Error if optimization fails or times out
 *
 * @example
 * ```typescript
 * try {
 *   const result = await runOptimizationAndWait({
 *     query: 'Optimize sales rep time allocation',
 *     resource_type: ResourceType.REP_TIME,
 *     allocation_targets: targets,
 *     constraints: constraints,
 *   });
 *   console.log(result.optimization_summary);
 * } catch (error) {
 *   console.error('Optimization failed:', error);
 * }
 * ```
 */
export async function runOptimizationAndWait(
  request: RunOptimizationRequest,
  pollIntervalMs: number = 2000,
  maxWaitMs: number = 120000
): Promise<OptimizationResponse> {
  // Start optimization asynchronously
  const initial = await runOptimization(request, true);

  // If already complete, return immediately
  if (initial.status === 'completed' || initial.status === 'failed') {
    return initial;
  }

  // Poll until complete or timeout
  const startTime = Date.now();
  const optimizationId = initial.optimization_id;

  while (Date.now() - startTime < maxWaitMs) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));

    const result = await getOptimization(optimizationId);

    if (result.status === 'completed') {
      return result;
    }

    if (result.status === 'failed') {
      throw new Error(
        `Optimization failed: ${result.warnings.join(', ') || 'Unknown error'}`
      );
    }
  }

  throw new Error(`Optimization timed out after ${maxWaitMs}ms`);
}

/**
 * Optimize budget allocation for maximum ROI.
 *
 * Convenience function for budget optimization use case.
 *
 * @param targets - Allocation targets
 * @param totalBudget - Total budget constraint
 * @param runScenarios - Whether to run scenario analysis
 * @returns Optimization results
 *
 * @example
 * ```typescript
 * const result = await optimizeBudget(
 *   [
 *     { entity_id: 'northeast', entity_type: 'territory', current_allocation: 50000, expected_response: 1.3 },
 *     { entity_id: 'southeast', entity_type: 'territory', current_allocation: 40000, expected_response: 0.9 },
 *   ],
 *   200000,
 *   true
 * );
 * console.log(`New allocation: ${result.optimal_allocations.map(a => `${a.entity_id}: $${a.optimized_allocation}`)}`);
 * ```
 */
export async function optimizeBudget(
  targets: RunOptimizationRequest['allocation_targets'],
  totalBudget: number,
  runScenarios: boolean = false
): Promise<OptimizationResponse> {
  return runOptimizationAndWait({
    query: `Optimize budget allocation to maximize ROI within $${totalBudget} budget`,
    resource_type: 'budget' as never,
    allocation_targets: targets,
    constraints: [
      { constraint_type: 'budget' as never, value: totalBudget, scope: 'global' as never }
    ],
    objective: 'maximize_roi' as never,
    run_scenarios: runScenarios,
  });
}

/**
 * Optimize resource allocation with scenario comparison.
 *
 * Runs optimization with multiple scenarios to compare outcomes.
 *
 * @param request - Base optimization parameters
 * @param scenarioCount - Number of scenarios to generate (default: 3)
 * @returns Optimization results with scenarios
 *
 * @example
 * ```typescript
 * const result = await optimizeWithScenarios(
 *   {
 *     query: 'Compare allocation strategies',
 *     resource_type: ResourceType.SAMPLES,
 *     allocation_targets: targets,
 *   },
 *   5
 * );
 *
 * result.scenarios.forEach(s => {
 *   console.log(`${s.scenario_name}: ${s.roi}x ROI`);
 * });
 * ```
 */
export async function optimizeWithScenarios(
  request: RunOptimizationRequest,
  scenarioCount: number = 3
): Promise<OptimizationResponse> {
  return runOptimizationAndWait({
    ...request,
    run_scenarios: true,
    scenario_count: scenarioCount,
  });
}
