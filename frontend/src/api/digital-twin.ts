/**
 * Digital Twin API Client
 * =======================
 *
 * TypeScript API client functions for the E2I Digital Twin simulation endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - POST /digital-twin/simulate: Run simulation
 * - GET /digital-twin/simulations: List simulations
 * - GET /digital-twin/simulations/{id}: Get simulation detail
 * - POST /digital-twin/validate: Validate against actual results
 * - GET /digital-twin/models: List twin models
 * - GET /digital-twin/models/{id}: Get model detail
 * - GET /digital-twin/models/{id}/fidelity: Get fidelity history
 * - GET /digital-twin/models/{id}/fidelity/report: Get fidelity report
 * - GET /digital-twin/health: Health check
 *
 * @module api/digital-twin
 */

import { get, post } from '@/lib/api-client';
import type {
  Brand,
  DigitalTwinHealthResponse,
  FidelityHistoryResponse,
  FidelityRecordResponse,
  FidelityReportResponse,
  ModelListResponse,
  SimulateRequest,
  SimulationDetailResponse,
  SimulationListResponse,
  SimulationResponse,
  SimulationStatus,
  TwinModelDetailResponse,
  TwinType,
  ValidateFidelityRequest,
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
 * and returns predicted ATE, recommendations, and confidence metrics.
 *
 * @param request - Simulation request with intervention parameters
 * @returns Simulation results with outcomes and recommendations
 *
 * @example
 * ```typescript
 * const result = await runSimulation({
 *   intervention: {
 *     intervention_type: 'email_campaign',
 *     channel: 'email',
 *     duration_weeks: 8,
 *   },
 *   brand: 'Remibrutinib',
 *   twin_count: 10000,
 * });
 * console.log(`ATE: ${result.simulated_ate} [${result.simulated_ci_lower}, ${result.simulated_ci_upper}]`);
 * console.log(`Recommendation: ${result.recommendation}`);
 * ```
 */
export async function runSimulation(
  request: SimulateRequest
): Promise<SimulationResponse> {
  return post<SimulationResponse, SimulateRequest>(
    `${DIGITAL_TWIN_BASE}/simulate`,
    request
  );
}

/**
 * List simulation results with filtering and pagination.
 *
 * @param params - Filter and pagination parameters
 * @returns Paginated list of simulations
 *
 * @example
 * ```typescript
 * const list = await listSimulations({
 *   brand: 'Kisqali',
 *   status: 'completed',
 *   page: 1,
 *   page_size: 20,
 * });
 * console.log(`Found ${list.total_count} simulations`);
 * ```
 */
export async function listSimulations(params?: {
  brand?: Brand | string;
  model_id?: string;
  status?: SimulationStatus | string;
  page?: number;
  page_size?: number;
}): Promise<SimulationListResponse> {
  return get<SimulationListResponse>(`${DIGITAL_TWIN_BASE}/simulations`, params);
}

/**
 * Get detailed information about a simulation.
 *
 * Returns full simulation results including heterogeneous effects.
 *
 * @param simulationId - The simulation identifier (UUID)
 * @returns Detailed simulation result
 *
 * @example
 * ```typescript
 * const detail = await getSimulation('550e8400-e29b-41d4-a716-446655440000');
 * console.log(`ATE: ${detail.simulated_ate}`);
 * console.log(`Top segments:`, detail.effect_heterogeneity.top_segments);
 * ```
 */
export async function getSimulation(
  simulationId: string
): Promise<SimulationDetailResponse> {
  return get<SimulationDetailResponse>(
    `${DIGITAL_TWIN_BASE}/simulations/${encodeURIComponent(simulationId)}`
  );
}

// =============================================================================
// FIDELITY VALIDATION ENDPOINTS
// =============================================================================

/**
 * Validate a simulation against actual experiment results.
 *
 * Updates the fidelity record with actual outcomes and calculates
 * prediction error and fidelity grade.
 *
 * @param request - Validation data including actual ATE
 * @returns Updated fidelity record with grade
 *
 * @example
 * ```typescript
 * const record = await validateSimulation({
 *   simulation_id: '550e8400-e29b-41d4-a716-446655440000',
 *   experiment_id: '660e8400-e29b-41d4-a716-446655440000',
 *   actual_ate: 0.072,
 *   actual_ci_lower: 0.045,
 *   actual_ci_upper: 0.099,
 * });
 * console.log(`Fidelity grade: ${record.fidelity_grade}`);
 * console.log(`Prediction error: ${record.prediction_error}`);
 * ```
 */
export async function validateSimulation(
  request: ValidateFidelityRequest
): Promise<FidelityRecordResponse> {
  return post<FidelityRecordResponse, ValidateFidelityRequest>(
    `${DIGITAL_TWIN_BASE}/validate`,
    request
  );
}

// =============================================================================
// MODEL ENDPOINTS
// =============================================================================

/**
 * List trained twin generator models.
 *
 * @param params - Optional filter parameters
 * @returns List of active models
 *
 * @example
 * ```typescript
 * const models = await listModels({
 *   brand: 'Remibrutinib',
 *   twin_type: 'hcp',
 * });
 * models.models.forEach(m => {
 *   console.log(`${m.model_name}: R²=${m.r2_score?.toFixed(3)}`);
 * });
 * ```
 */
export async function listModels(params?: {
  brand?: Brand | string;
  twin_type?: TwinType | string;
}): Promise<ModelListResponse> {
  return get<ModelListResponse>(`${DIGITAL_TWIN_BASE}/models`, params);
}

/**
 * Get detailed information about a twin model.
 *
 * @param modelId - Model UUID
 * @returns Model details including performance metrics
 *
 * @example
 * ```typescript
 * const model = await getModel('abc123-uuid');
 * console.log(`Algorithm: ${model.algorithm}`);
 * console.log(`R²: ${model.r2_score}`);
 * console.log(`Top features:`, model.top_features);
 * ```
 */
export async function getModel(modelId: string): Promise<TwinModelDetailResponse> {
  return get<TwinModelDetailResponse>(
    `${DIGITAL_TWIN_BASE}/models/${encodeURIComponent(modelId)}`
  );
}

/**
 * Get fidelity validation history for a model.
 *
 * @param modelId - Model UUID
 * @param params - Optional filter parameters
 * @returns Fidelity history with grade distribution
 *
 * @example
 * ```typescript
 * const history = await getModelFidelity('abc123-uuid', {
 *   limit: 10,
 *   validated_only: true,
 * });
 * console.log(`Average fidelity: ${history.average_fidelity_score}`);
 * console.log(`Grade distribution:`, history.grade_distribution);
 * ```
 */
export async function getModelFidelity(
  modelId: string,
  params?: {
    limit?: number;
    validated_only?: boolean;
  }
): Promise<FidelityHistoryResponse> {
  return get<FidelityHistoryResponse>(
    `${DIGITAL_TWIN_BASE}/models/${encodeURIComponent(modelId)}/fidelity`,
    params
  );
}

/**
 * Get aggregated fidelity report for a model.
 *
 * Analyzes fidelity trends and provides degradation warnings.
 *
 * @param modelId - Model UUID
 * @param lookbackDays - Number of days to analyze (default: 90)
 * @returns Fidelity report with trend analysis
 *
 * @example
 * ```typescript
 * const report = await getModelFidelityReport('abc123-uuid', 90);
 * console.log(`Trend: ${report.trend}`);
 * if (report.is_degrading) {
 *   console.warn(`Model degrading! ${report.recommendation}`);
 * }
 * ```
 */
export async function getModelFidelityReport(
  modelId: string,
  lookbackDays: number = 90
): Promise<FidelityReportResponse> {
  return get<FidelityReportResponse>(
    `${DIGITAL_TWIN_BASE}/models/${encodeURIComponent(modelId)}/fidelity/report`,
    { lookback_days: lookbackDays }
  );
}

// =============================================================================
// HEALTH ENDPOINTS
// =============================================================================

/**
 * Health check for the digital twin service.
 *
 * @returns Service health status
 *
 * @example
 * ```typescript
 * const health = await getDigitalTwinHealth();
 * console.log(`Status: ${health.status}`);
 * console.log(`Models available: ${health.models_available}`);
 * ```
 */
export async function getDigitalTwinHealth(): Promise<DigitalTwinHealthResponse> {
  return get<DigitalTwinHealthResponse>(`${DIGITAL_TWIN_BASE}/health`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Run simulation and get full details including heterogeneity.
 *
 * Combines runSimulation and getSimulation into one call.
 *
 * @param request - Simulation request
 * @returns Detailed simulation response
 *
 * @example
 * ```typescript
 * const detail = await runAndGetSimulationDetail({
 *   intervention: { intervention_type: 'email_campaign' },
 *   brand: 'Remibrutinib',
 * });
 * console.log(`Top responding segments:`, detail.effect_heterogeneity.top_segments);
 * ```
 */
export async function runAndGetSimulationDetail(
  request: SimulateRequest
): Promise<SimulationDetailResponse> {
  const result = await runSimulation(request);
  return getSimulation(result.simulation_id);
}

/**
 * Get recent simulations for a brand.
 *
 * @param brand - Brand to filter by
 * @param limit - Maximum number of results
 * @returns List of recent simulations
 *
 * @example
 * ```typescript
 * const recent = await getRecentSimulations('Kisqali', 5);
 * recent.simulations.forEach(s => console.log(s.intervention_type, s.recommendation));
 * ```
 */
export async function getRecentSimulations(
  brand: Brand | string,
  limit: number = 10
): Promise<SimulationListResponse> {
  return listSimulations({ brand, page: 1, page_size: limit });
}

/**
 * Get simulations by recommendation.
 *
 * Filters simulations to only those with specified recommendation.
 *
 * @param recommendation - Recommendation to filter by
 * @param brand - Optional brand filter
 * @returns Filtered simulation list
 *
 * @example
 * ```typescript
 * const deployable = await getSimulationsByRecommendation('deploy', 'Remibrutinib');
 * console.log(`${deployable.length} interventions ready to deploy`);
 * ```
 */
export async function getSimulationsByRecommendation(
  recommendation: string,
  brand?: Brand | string
): Promise<SimulationListResponse> {
  const result = await listSimulations({ brand, page: 1, page_size: 100 });
  return {
    ...result,
    simulations: result.simulations.filter((s) => s.recommendation === recommendation),
    total_count: result.simulations.filter((s) => s.recommendation === recommendation)
      .length,
  };
}

/**
 * Check if a model needs retraining based on fidelity.
 *
 * @param modelId - Model UUID
 * @returns Whether model needs retraining and reason
 *
 * @example
 * ```typescript
 * const check = await checkModelHealth('abc123-uuid');
 * if (check.needsRetraining) {
 *   console.warn(`Model needs retraining: ${check.reason}`);
 * }
 * ```
 */
export async function checkModelHealth(modelId: string): Promise<{
  needsRetraining: boolean;
  reason?: string;
  fidelityScore?: number;
  trend: string;
}> {
  try {
    const report = await getModelFidelityReport(modelId);
    return {
      needsRetraining: report.is_degrading || report.average_fidelity_score < 0.6,
      reason: report.recommendation,
      fidelityScore: report.average_fidelity_score,
      trend: report.trend,
    };
  } catch {
    return {
      needsRetraining: false,
      reason: 'Unable to fetch fidelity report',
      trend: 'unknown',
    };
  }
}

/**
 * Get model with fidelity information.
 *
 * Combines model details with fidelity history.
 *
 * @param modelId - Model UUID
 * @returns Model details with fidelity data
 *
 * @example
 * ```typescript
 * const modelWithFidelity = await getModelWithFidelity('abc123-uuid');
 * console.log(`Model: ${modelWithFidelity.model.model_name}`);
 * console.log(`Avg fidelity: ${modelWithFidelity.fidelity.average_fidelity_score}`);
 * ```
 */
export async function getModelWithFidelity(modelId: string): Promise<{
  model: TwinModelDetailResponse;
  fidelity: FidelityHistoryResponse;
}> {
  const [model, fidelity] = await Promise.all([
    getModel(modelId),
    getModelFidelity(modelId, { limit: 10 }),
  ]);
  return { model, fidelity };
}

/**
 * Get all active models for a brand.
 *
 * @param brand - Brand to filter by
 * @returns List of active models
 *
 * @example
 * ```typescript
 * const models = await getActiveModelsForBrand('Fabhalta');
 * console.log(`${models.length} active models for Fabhalta`);
 * ```
 */
export async function getActiveModelsForBrand(
  brand: Brand | string
): Promise<ModelListResponse> {
  const result = await listModels({ brand });
  return {
    ...result,
    models: result.models.filter((m) => m.is_active),
    total_count: result.models.filter((m) => m.is_active).length,
  };
}

/**
 * Get best model for simulation.
 *
 * Finds the active model with highest R² score for given brand/twin type.
 *
 * @param brand - Target brand
 * @param twinType - Twin type (default: 'hcp')
 * @returns Best model or undefined
 *
 * @example
 * ```typescript
 * const best = await getBestModel('Remibrutinib', 'hcp');
 * if (best) {
 *   console.log(`Best model: ${best.model_name} (R²=${best.r2_score})`);
 * }
 * ```
 */
export async function getBestModel(
  brand: Brand | string,
  twinType: TwinType | string = 'hcp'
): Promise<TwinModelDetailResponse | undefined> {
  const result = await listModels({ brand, twin_type: twinType });
  const activeModels = result.models.filter((m) => m.is_active);

  if (activeModels.length === 0) {
    return undefined;
  }

  // Sort by R² score descending
  const sorted = activeModels.sort((a, b) => (b.r2_score || 0) - (a.r2_score || 0));
  return getModel(sorted[0].model_id);
}

/**
 * Format ATE with confidence interval for display.
 *
 * @param ate - Average Treatment Effect
 * @param ciLower - CI lower bound
 * @param ciUpper - CI upper bound
 * @returns Formatted string
 *
 * @example
 * ```typescript
 * const result = await runSimulation({ ... });
 * console.log(formatATE(result.simulated_ate, result.simulated_ci_lower, result.simulated_ci_upper));
 * // "0.085 [0.052, 0.118]"
 * ```
 */
export function formatATE(ate: number, ciLower: number, ciUpper: number): string {
  return `${ate.toFixed(3)} [${ciLower.toFixed(3)}, ${ciUpper.toFixed(3)}]`;
}

/**
 * Get recommendation color for UI display.
 *
 * @param recommendation - Simulation recommendation
 * @returns Color string
 *
 * @example
 * ```typescript
 * const color = getRecommendationColor(result.recommendation);
 * // style={{ color }}
 * ```
 */
export function getRecommendationColor(
  recommendation: string
): 'green' | 'red' | 'yellow' {
  switch (recommendation) {
    case 'deploy':
      return 'green';
    case 'skip':
      return 'red';
    case 'refine':
    default:
      return 'yellow';
  }
}

/**
 * Get fidelity grade color for UI display.
 *
 * @param grade - Fidelity grade
 * @returns Color string
 *
 * @example
 * ```typescript
 * const color = getFidelityGradeColor(record.fidelity_grade);
 * ```
 */
export function getFidelityGradeColor(
  grade: string
): 'green' | 'lime' | 'yellow' | 'orange' | 'gray' {
  switch (grade) {
    case 'excellent':
      return 'green';
    case 'good':
      return 'lime';
    case 'fair':
      return 'yellow';
    case 'poor':
      return 'orange';
    case 'unvalidated':
    default:
      return 'gray';
  }
}

/**
 * Check if simulation is statistically significant.
 *
 * @param simulation - Simulation response
 * @returns Whether effect is significant
 *
 * @example
 * ```typescript
 * if (isSignificant(result)) {
 *   console.log('Effect is statistically significant!');
 * }
 * ```
 */
export function isSignificant(
  simulation: SimulationResponse | SimulationDetailResponse
): boolean {
  // Check if CI doesn't cross zero
  return (
    (simulation.simulated_ci_lower > 0 && simulation.simulated_ci_upper > 0) ||
    (simulation.simulated_ci_lower < 0 && simulation.simulated_ci_upper < 0)
  );
}

/**
 * Calculate effect size interpretation.
 *
 * @param cohensD - Cohen's d effect size
 * @returns Effect size interpretation
 *
 * @example
 * ```typescript
 * const interpretation = interpretEffectSize(result.effect_size_cohens_d);
 * // "medium"
 * ```
 */
export function interpretEffectSize(
  cohensD?: number
): 'negligible' | 'small' | 'medium' | 'large' | 'unknown' {
  if (cohensD === undefined || cohensD === null) {
    return 'unknown';
  }
  const absD = Math.abs(cohensD);
  if (absD < 0.2) return 'negligible';
  if (absD < 0.5) return 'small';
  if (absD < 0.8) return 'medium';
  return 'large';
}
