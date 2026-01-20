/**
 * Health Score API Client
 * =======================
 *
 * TypeScript API client functions for the E2I Health Score endpoints.
 * Uses the shared apiClient for consistent error handling and interceptors.
 *
 * Endpoints:
 * - Health check execution (full, quick, scoped)
 * - Component, model, pipeline, agent health
 * - Health history and trends
 * - Service status
 *
 * @module api/health-score
 */

import { get } from '@/lib/api-client';
import type {
  AgentHealthResponse,
  CheckScope,
  ComponentHealthResponse,
  HealthHistoryResponse,
  HealthScoreResponse,
  HealthServiceStatus,
  ModelHealthResponse,
  PipelineHealthResponse,
} from '@/types/health-score';

// =============================================================================
// HEALTH SCORE API ENDPOINTS
// =============================================================================

const HEALTH_SCORE_BASE = '/health-score';

// =============================================================================
// HEALTH CHECK ENDPOINTS
// =============================================================================

/**
 * Run a health check with specified scope.
 *
 * This endpoint invokes the Health Score agent (Tier 3) which is a
 * Fast Path agent with no LLM usage.
 *
 * @param scope - Check scope (full, quick, models, pipelines, agents)
 * @returns Health check results with scores and details
 *
 * @example
 * ```typescript
 * const result = await runHealthCheck('full');
 * console.log(`Health: ${result.health_grade} (${result.overall_health_score}/100)`);
 * if (result.critical_issues.length > 0) {
 *   console.warn('Critical issues:', result.critical_issues);
 * }
 * ```
 */
export async function runHealthCheck(
  scope: CheckScope | string = 'full'
): Promise<HealthScoreResponse> {
  return get<HealthScoreResponse>(`${HEALTH_SCORE_BASE}/check`, { scope });
}

/**
 * Run a quick health check (<1s target).
 *
 * Quick checks focus on component health only, making them
 * suitable for frequent dashboard updates.
 *
 * @returns Basic health check results
 *
 * @example
 * ```typescript
 * const quick = await quickHealthCheck();
 * console.log(`Quick check: ${quick.health_grade}`);
 * ```
 */
export async function quickHealthCheck(): Promise<HealthScoreResponse> {
  return get<HealthScoreResponse>(`${HEALTH_SCORE_BASE}/quick`);
}

/**
 * Run a full health check (<5s target).
 *
 * Full checks assess all four health dimensions:
 * components, models, pipelines, and agents.
 *
 * @returns Comprehensive health check results
 *
 * @example
 * ```typescript
 * const full = await fullHealthCheck();
 * console.log(`Component: ${full.component_health_score}`);
 * console.log(`Model: ${full.model_health_score}`);
 * console.log(`Pipeline: ${full.pipeline_health_score}`);
 * console.log(`Agent: ${full.agent_health_score}`);
 * ```
 */
export async function fullHealthCheck(): Promise<HealthScoreResponse> {
  return get<HealthScoreResponse>(`${HEALTH_SCORE_BASE}/full`);
}

// =============================================================================
// DIMENSION-SPECIFIC ENDPOINTS
// =============================================================================

/**
 * Get detailed component health information.
 *
 * Checks: Database, Cache (Redis), Vector Store, API, Message Queue
 *
 * @returns Component health details
 *
 * @example
 * ```typescript
 * const components = await getComponentHealth();
 * components.components.forEach(c => {
 *   console.log(`${c.component_name}: ${c.status} (${c.latency_ms}ms)`);
 * });
 * ```
 */
export async function getComponentHealth(): Promise<ComponentHealthResponse> {
  return get<ComponentHealthResponse>(`${HEALTH_SCORE_BASE}/components`);
}

/**
 * Get detailed model health information.
 *
 * Checks model accuracy, latency, error rates, and prediction volume.
 *
 * @returns Model health details
 *
 * @example
 * ```typescript
 * const models = await getModelHealth();
 * models.models.forEach(m => {
 *   console.log(`${m.model_name}: ${m.status}`);
 *   console.log(`  Accuracy: ${m.accuracy?.toFixed(2)}`);
 *   console.log(`  Error rate: ${(m.error_rate * 100).toFixed(1)}%`);
 * });
 * ```
 */
export async function getModelHealth(): Promise<ModelHealthResponse> {
  return get<ModelHealthResponse>(`${HEALTH_SCORE_BASE}/models`);
}

/**
 * Get detailed pipeline health information.
 *
 * Checks data freshness, processing success, and row counts.
 *
 * @returns Pipeline health details
 *
 * @example
 * ```typescript
 * const pipelines = await getPipelineHealth();
 * pipelines.pipelines.forEach(p => {
 *   console.log(`${p.pipeline_name}: ${p.status}`);
 *   console.log(`  Freshness: ${p.freshness_hours.toFixed(1)} hours`);
 *   console.log(`  Rows: ${p.rows_processed.toLocaleString()}`);
 * });
 * ```
 */
export async function getPipelineHealth(): Promise<PipelineHealthResponse> {
  return get<PipelineHealthResponse>(`${HEALTH_SCORE_BASE}/pipelines`);
}

/**
 * Get detailed agent health information.
 *
 * Checks agent availability, success rates, and latency.
 *
 * @returns Agent health details
 *
 * @example
 * ```typescript
 * const agents = await getAgentHealth();
 * console.log(`Available: ${agents.available_count}/${agents.total_agents}`);
 * agents.agents.forEach(a => {
 *   const status = a.available ? 'UP' : 'DOWN';
 *   console.log(`[Tier ${a.tier}] ${a.agent_name}: ${status}`);
 * });
 * ```
 */
export async function getAgentHealth(): Promise<AgentHealthResponse> {
  return get<AgentHealthResponse>(`${HEALTH_SCORE_BASE}/agents`);
}

// =============================================================================
// HISTORY ENDPOINTS
// =============================================================================

/**
 * Get historical health check records.
 *
 * Returns recent health check results with trend analysis.
 *
 * @param limit - Maximum number of records to return (default: 20)
 * @returns Historical health check data
 *
 * @example
 * ```typescript
 * const history = await getHealthHistory(10);
 * console.log(`Trend: ${history.trend}`);
 * console.log(`Average score: ${history.avg_health_score.toFixed(1)}`);
 * history.checks.forEach(c => {
 *   console.log(`${c.timestamp}: ${c.health_grade} (${c.overall_health_score})`);
 * });
 * ```
 */
export async function getHealthHistory(
  limit: number = 20
): Promise<HealthHistoryResponse> {
  return get<HealthHistoryResponse>(`${HEALTH_SCORE_BASE}/history`, { limit });
}

// =============================================================================
// STATUS ENDPOINTS
// =============================================================================

/**
 * Get health score service status.
 *
 * Checks agent availability and recent activity.
 *
 * @returns Service status information
 *
 * @example
 * ```typescript
 * const status = await getHealthServiceStatus();
 * if (status.status === 'healthy') {
 *   console.log(`${status.checks_24h} checks in last 24h`);
 *   console.log(`Avg latency: ${status.avg_check_latency_ms}ms`);
 * }
 * ```
 */
export async function getHealthServiceStatus(): Promise<HealthServiceStatus> {
  return get<HealthServiceStatus>(`${HEALTH_SCORE_BASE}/status`);
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Get comprehensive health status.
 *
 * Fetches health check results along with dimension-specific details.
 *
 * @returns Combined health information
 *
 * @example
 * ```typescript
 * const status = await getComprehensiveHealth();
 * console.log(`Overall: ${status.health.health_grade}`);
 * console.log(`Components: ${status.components.healthy_count}/${status.components.total_components}`);
 * console.log(`Models: ${status.models.healthy_count}/${status.models.total_models}`);
 * ```
 */
export async function getComprehensiveHealth(): Promise<{
  health: HealthScoreResponse;
  components: ComponentHealthResponse;
  models: ModelHealthResponse;
  pipelines: PipelineHealthResponse;
  agents: AgentHealthResponse;
}> {
  const [health, components, models, pipelines, agents] = await Promise.all([
    fullHealthCheck(),
    getComponentHealth(),
    getModelHealth(),
    getPipelineHealth(),
    getAgentHealth(),
  ]);

  return { health, components, models, pipelines, agents };
}

/**
 * Get health dashboard data.
 *
 * Fetches current health status, service status, and recent history
 * in a single call - suitable for dashboard displays.
 *
 * @returns Dashboard-ready health data
 *
 * @example
 * ```typescript
 * const dashboard = await getHealthDashboard();
 * console.log(`Current: ${dashboard.current.health_grade}`);
 * console.log(`Trend: ${dashboard.history.trend}`);
 * console.log(`Service: ${dashboard.status.status}`);
 * ```
 */
export async function getHealthDashboard(): Promise<{
  current: HealthScoreResponse;
  status: HealthServiceStatus;
  history: HealthHistoryResponse;
}> {
  const [current, status, history] = await Promise.all([
    fullHealthCheck(),
    getHealthServiceStatus(),
    getHealthHistory(10),
  ]);

  return { current, status, history };
}

/**
 * Monitor health with polling.
 *
 * Periodically checks health and calls the callback with results.
 * Returns a function to stop monitoring.
 *
 * @param callback - Function called with each health check result
 * @param intervalMs - Polling interval in milliseconds (default: 30000)
 * @param useQuickCheck - Use quick check instead of full (default: true)
 * @returns Stop function
 *
 * @example
 * ```typescript
 * const stopMonitoring = monitorHealth(
 *   (result) => {
 *     console.log(`Health: ${result.health_grade}`);
 *     if (result.critical_issues.length > 0) {
 *       alert('Critical issues detected!');
 *     }
 *   },
 *   60000, // Check every minute
 *   true   // Use quick check
 * );
 *
 * // Later, to stop monitoring:
 * stopMonitoring();
 * ```
 */
export function monitorHealth(
  callback: (result: HealthScoreResponse) => void,
  intervalMs: number = 30000,
  useQuickCheck: boolean = true
): () => void {
  let isRunning = true;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const check = async () => {
    if (!isRunning) return;

    try {
      const result = useQuickCheck
        ? await quickHealthCheck()
        : await fullHealthCheck();
      callback(result);
    } catch (error) {
      console.error('Health monitoring check failed:', error);
    }

    if (isRunning) {
      timeoutId = setTimeout(check, intervalMs);
    }
  };

  // Start immediately
  check();

  // Return stop function
  return () => {
    isRunning = false;
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  };
}

/**
 * Get health score grade color.
 *
 * Returns a color string suitable for UI display based on health grade.
 *
 * @param grade - Health grade (A-F)
 * @returns Color string for UI display
 *
 * @example
 * ```typescript
 * const result = await quickHealthCheck();
 * const color = getGradeColor(result.health_grade);
 * // Use color in UI: style={{ backgroundColor: color }}
 * ```
 */
export function getGradeColor(
  grade: string
): 'green' | 'lime' | 'yellow' | 'orange' | 'red' {
  switch (grade) {
    case 'A':
      return 'green';
    case 'B':
      return 'lime';
    case 'C':
      return 'yellow';
    case 'D':
      return 'orange';
    case 'F':
    default:
      return 'red';
  }
}

/**
 * Format health score for display.
 *
 * Returns a formatted string with score and grade.
 *
 * @param score - Health score (0-100)
 * @param grade - Health grade (A-F)
 * @returns Formatted display string
 *
 * @example
 * ```typescript
 * const result = await quickHealthCheck();
 * const display = formatHealthScore(result.overall_health_score, result.health_grade);
 * // "85.5/100 (B)"
 * ```
 */
export function formatHealthScore(score: number, grade: string): string {
  return `${score.toFixed(1)}/100 (${grade})`;
}

/**
 * Check if health is critical.
 *
 * Returns true if the health check indicates critical issues
 * that require immediate attention.
 *
 * @param result - Health check result
 * @returns True if health is critical
 *
 * @example
 * ```typescript
 * const result = await quickHealthCheck();
 * if (isHealthCritical(result)) {
 *   notifyOpsTeam(result.critical_issues);
 * }
 * ```
 */
export function isHealthCritical(result: HealthScoreResponse): boolean {
  return (
    result.health_grade === 'F' ||
    result.critical_issues.length > 0 ||
    result.overall_health_score < 60
  );
}

/**
 * Get agents by tier.
 *
 * Groups agents from a health response by their tier.
 *
 * @param agents - Agent health response
 * @returns Agents grouped by tier
 *
 * @example
 * ```typescript
 * const agentHealth = await getAgentHealth();
 * const byTier = getAgentsByTier(agentHealth);
 * console.log('Tier 1 agents:', byTier[1]?.map(a => a.agent_name));
 * ```
 */
export function getAgentsByTier(
  agents: AgentHealthResponse
): Record<number, AgentHealthResponse['agents']> {
  const byTier: Record<number, AgentHealthResponse['agents']> = {};

  for (const agent of agents.agents) {
    if (!byTier[agent.tier]) {
      byTier[agent.tier] = [];
    }
    byTier[agent.tier].push(agent);
  }

  return byTier;
}
