/**
 * Resource Optimization Types
 * ===========================
 *
 * TypeScript interfaces for the E2I Resource Optimization API.
 * Based on src/api/routes/resource_optimizer.py backend schemas.
 *
 * @module types/resources
 */

// =============================================================================
// ENUMS
// =============================================================================

/**
 * Optimization objectives
 */
export enum OptimizationObjective {
  MAXIMIZE_OUTCOME = 'maximize_outcome',
  MAXIMIZE_ROI = 'maximize_roi',
  MINIMIZE_COST = 'minimize_cost',
  BALANCE = 'balance',
}

/**
 * Available solver types
 */
export enum SolverType {
  LINEAR = 'linear',
  MILP = 'milp',
  NONLINEAR = 'nonlinear',
}

/**
 * Status of optimization
 */
export enum OptimizationStatus {
  PENDING = 'pending',
  FORMULATING = 'formulating',
  OPTIMIZING = 'optimizing',
  ANALYZING = 'analyzing',
  PROJECTING = 'projecting',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

/**
 * Types of resources to optimize
 */
export enum ResourceType {
  BUDGET = 'budget',
  REP_TIME = 'rep_time',
  SAMPLES = 'samples',
  CALLS = 'calls',
}

/**
 * Types of optimization constraints
 */
export enum ConstraintType {
  BUDGET = 'budget',
  CAPACITY = 'capacity',
  MIN_COVERAGE = 'min_coverage',
  MAX_FREQUENCY = 'max_frequency',
}

/**
 * Scope of constraints
 */
export enum ConstraintScope {
  GLOBAL = 'global',
  REGIONAL = 'regional',
  ENTITY = 'entity',
}

// =============================================================================
// REQUEST MODELS
// =============================================================================

/**
 * Target entity for resource allocation
 */
export interface AllocationTarget {
  /** Entity identifier */
  entity_id: string;
  /** Entity type (hcp, territory, region) */
  entity_type: string;
  /** Current allocation amount */
  current_allocation: number;
  /** Minimum allowed allocation */
  min_allocation?: number;
  /** Maximum allowed allocation */
  max_allocation?: number;
  /** Response coefficient */
  expected_response?: number;
}

/**
 * Optimization constraint
 */
export interface Constraint {
  /** Type of constraint */
  constraint_type: ConstraintType;
  /** Constraint value */
  value: number;
  /** Constraint scope */
  scope?: ConstraintScope;
}

/**
 * Request to run resource optimization
 */
export interface RunOptimizationRequest {
  /** Natural language query */
  query: string;
  /** Type of resource to optimize */
  resource_type: ResourceType;
  /** Entities to allocate resources to */
  allocation_targets: AllocationTarget[];
  /** Optimization constraints */
  constraints?: Constraint[];
  /** Optimization objective */
  objective?: OptimizationObjective;
  /** Solver type */
  solver_type?: SolverType;
  /** Solver time limit (1-300 seconds) */
  time_limit_seconds?: number;
  /** MILP gap tolerance (0-1) */
  gap_tolerance?: number;
  /** Run what-if scenarios */
  run_scenarios?: boolean;
  /** Number of scenarios (1-10) */
  scenario_count?: number;
}

/**
 * Parameters for listing scenarios
 */
export interface ListScenariosParams {
  /** Minimum ROI threshold */
  min_roi?: number;
  /** Maximum number of results */
  limit?: number;
}

// =============================================================================
// RESPONSE MODELS
// =============================================================================

/**
 * Optimized allocation result for an entity
 */
export interface AllocationResult {
  /** Entity identifier */
  entity_id: string;
  /** Entity type */
  entity_type: string;
  /** Current allocation */
  current_allocation: number;
  /** Optimized allocation */
  optimized_allocation: number;
  /** Change from current */
  change: number;
  /** Change percentage */
  change_percentage: number;
  /** Expected outcome impact */
  expected_impact: number;
}

/**
 * Result of a scenario analysis
 */
export interface ScenarioResult {
  /** Scenario name */
  scenario_name: string;
  /** Total allocation in scenario */
  total_allocation: number;
  /** Projected outcome */
  projected_outcome: number;
  /** Return on investment */
  roi: number;
  /** Any constraint violations */
  constraint_violations: string[];
}

/**
 * Response from resource optimization
 */
export interface OptimizationResponse {
  /** Unique optimization identifier */
  optimization_id: string;
  /** Optimization status */
  status: OptimizationStatus;
  /** Resource type optimized */
  resource_type: ResourceType;
  /** Objective used */
  objective: OptimizationObjective;

  // Optimization results
  /** Optimized allocations */
  optimal_allocations: AllocationResult[];
  /** Optimized objective value */
  objective_value?: number;
  /** Solver termination status */
  solver_status?: string;
  /** Solver time (ms) */
  solve_time_ms: number;

  // Scenario results
  /** Scenario analysis results */
  scenarios: ScenarioResult[];
  /** Sensitivity of objective to constraints */
  sensitivity_analysis?: Record<string, number>;

  // Impact projections
  /** Total projected outcome */
  projected_total_outcome?: number;
  /** Projected ROI */
  projected_roi?: number;
  /** Impact breakdown by segment */
  impact_by_segment?: Record<string, number>;

  // Summary
  /** Executive summary */
  optimization_summary?: string;
  /** Actionable recommendations */
  recommendations: string[];

  // Metadata
  /** Problem formulation time (ms) */
  formulation_latency_ms: number;
  /** Optimization time (ms) */
  optimization_latency_ms: number;
  /** Total workflow time (ms) */
  total_latency_ms: number;
  /** Optimization timestamp */
  timestamp: string;
  /** Warnings */
  warnings: string[];
}

/**
 * Response for listing scenario analyses
 */
export interface ScenarioListResponse {
  /** Total scenarios */
  total_count: number;
  /** Scenario results */
  scenarios: ScenarioResult[];
}

/**
 * Health check response for resource optimization service
 */
export interface ResourceHealthResponse {
  /** Service status */
  status: string;
  /** Resource Optimizer agent status */
  agent_available: boolean;
  /** scipy availability */
  scipy_available: boolean;
  /** Last optimization timestamp */
  last_optimization?: string;
  /** Optimizations in last 24 hours */
  optimizations_24h: number;
}
