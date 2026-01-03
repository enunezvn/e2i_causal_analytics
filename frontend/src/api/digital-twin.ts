/**
 * Digital Twin API Client
 * =======================
 *
 * TypeScript API client functions for the E2I Digital Twin simulation endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Run simulations
 * - Compare scenarios
 * - Simulation history
 * - Health check
 *
 * @module api/digital-twin
 */

import { get, post } from '@/lib/api-client';
import type {
  SimulationRequest,
  SimulationResponse,
  ScenarioComparisonRequest,
  ScenarioComparisonResult,
  SimulationHistoryResponse,
} from '@/types/digital-twin';

// =============================================================================
// DIGITAL TWIN API ENDPOINTS
// =============================================================================

const DIGITAL_TWIN_BASE = '/digital-twin';

// =============================================================================
// SIMULATION ENDPOINTS
// =============================================================================

/**
 * Run a digital twin simulation for an intervention.
 *
 * Executes the simulation model with the provided parameters
 * and returns predicted outcomes, fidelity metrics, and recommendations.
 *
 * @param request - Simulation request with intervention parameters
 * @returns Simulation results with outcomes and recommendations
 *
 * @example
 * ```typescript
 * const result = await runSimulation({
 *   intervention_type: InterventionType.HCP_ENGAGEMENT,
 *   brand: 'Remibrutinib',
 *   sample_size: 1000,
 *   duration_days: 90,
 * });
 * console.log(`ATE: ${result.outcomes.ate.estimate}`);
 * ```
 */
export async function runSimulation(
  request: SimulationRequest
): Promise<SimulationResponse> {
  return post<SimulationResponse, SimulationRequest>(
    `${DIGITAL_TWIN_BASE}/simulate`,
    request
  );
}

/**
 * Compare multiple intervention scenarios.
 *
 * Runs simulations for a base scenario and alternatives,
 * then provides comparative analysis.
 *
 * @param request - Comparison request with base and alternative scenarios
 * @returns Comparison results with all simulations and summary
 *
 * @example
 * ```typescript
 * const comparison = await compareScenarios({
 *   base_scenario: { ... },
 *   alternative_scenarios: [{ ... }, { ... }],
 * });
 * console.log(`Best scenario: ${comparison.comparison.best_scenario_index}`);
 * ```
 */
export async function compareScenarios(
  request: ScenarioComparisonRequest
): Promise<ScenarioComparisonResult> {
  return post<ScenarioComparisonResult, ScenarioComparisonRequest>(
    `${DIGITAL_TWIN_BASE}/compare`,
    request
  );
}

/**
 * Get a specific simulation by ID.
 *
 * @param simulationId - The simulation identifier
 * @returns The simulation result
 */
export async function getSimulation(
  simulationId: string
): Promise<SimulationResponse> {
  return get<SimulationResponse>(`${DIGITAL_TWIN_BASE}/simulations/${simulationId}`);
}

/**
 * Get simulation history with pagination.
 *
 * @param params - Pagination and filter parameters
 * @returns Paginated list of historical simulations
 *
 * @example
 * ```typescript
 * const history = await getSimulationHistory({
 *   brand: 'Kisqali',
 *   limit: 10,
 *   offset: 0,
 * });
 * ```
 */
export async function getSimulationHistory(params?: {
  brand?: string;
  intervention_type?: string;
  limit?: number;
  offset?: number;
}): Promise<SimulationHistoryResponse> {
  const searchParams = new URLSearchParams();
  if (params?.brand) searchParams.set('brand', params.brand);
  if (params?.intervention_type) searchParams.set('intervention_type', params.intervention_type);
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  if (params?.offset) searchParams.set('offset', params.offset.toString());

  const queryString = searchParams.toString();
  const url = queryString
    ? `${DIGITAL_TWIN_BASE}/simulations?${queryString}`
    : `${DIGITAL_TWIN_BASE}/simulations`;

  return get<SimulationHistoryResponse>(url);
}

/**
 * Health check for the digital twin service.
 *
 * @returns Health status
 */
export async function getDigitalTwinHealth(): Promise<{
  status: string;
  model_version: string;
  last_calibration: string;
}> {
  return get(`${DIGITAL_TWIN_BASE}/health`);
}
