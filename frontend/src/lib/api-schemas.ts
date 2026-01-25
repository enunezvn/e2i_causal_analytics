/**
 * API Response Schemas (Zod)
 * ==========================
 *
 * Runtime validation schemas for API responses using Zod.
 * Ensures type safety beyond TypeScript's compile-time checks.
 *
 * Phase 3 - Type Safety Enhancement
 *
 * @module lib/api-schemas
 */

import { z } from 'zod';

// =============================================================================
// COMMON SCHEMAS
// =============================================================================

/**
 * Standard API error response schema
 */
export const ApiErrorResponseSchema = z.object({
  error: z.string(),
  message: z.string(),
  details: z.record(z.string(), z.unknown()).optional(),
  timestamp: z.string().optional(),
});

// =============================================================================
// REQUEST BODY SCHEMAS
// =============================================================================

/**
 * Intervention type enum values for simulation requests
 */
export const InterventionTypeEnum = z.enum([
  'hcp_engagement',
  'patient_support',
  'pricing',
  'rep_training',
  'digital_marketing',
  'formulary_access',
]);

/**
 * Digital twin simulation request schema
 * Used by SimulationPanel and DigitalTwin page
 */
export const SimulationRequestSchema = z.object({
  intervention_type: InterventionTypeEnum,
  brand: z.string().min(1, 'Brand is required'),
  sample_size: z.number().int().min(100, 'Minimum 100 samples').max(10000, 'Maximum 10,000 samples'),
  duration_days: z.number().int().min(30, 'Minimum 30 days').max(365, 'Maximum 365 days'),
  target_regions: z.array(z.string()).optional(),
  target_segments: z.array(z.string()).optional(),
  budget: z.number().min(0, 'Budget must be positive').max(999_999_999, 'Budget too large').optional(),
  parameters: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Chat feedback submission request schema
 */
export const ChatFeedbackRequestSchema = z.object({
  messageId: z.number().int().positive(),
  sessionId: z.string().min(1),
  rating: z.enum(['thumbs_up', 'thumbs_down', 'star_1', 'star_2', 'star_3', 'star_4', 'star_5']),
  responsePreview: z.string().max(500).optional(),
  agentName: z.string().optional(),
  comment: z.string().max(1000).optional(),
});

/**
 * Drift detection request schema
 */
export const DriftDetectionRequestSchema = z.object({
  model_id: z.string().min(1, 'Model ID required'),
  baseline_start_date: z.string().optional(),
  baseline_end_date: z.string().optional(),
  current_start_date: z.string().optional(),
  current_end_date: z.string().optional(),
  features: z.array(z.string()).optional(),
  threshold: z.number().min(0).max(1).optional(),
});

/**
 * Graph traverse/search request schema
 */
export const GraphSearchRequestSchema = z.object({
  query: z.string().min(1, 'Query required'),
  max_depth: z.number().int().min(1).max(10).optional(),
  node_types: z.array(z.string()).optional(),
  edge_types: z.array(z.string()).optional(),
  limit: z.number().int().min(1).max(1000).optional(),
});

/**
 * Memory search request schema
 */
export const MemorySearchRequestSchema = z.object({
  query: z.string().min(1, 'Search query required'),
  memory_type: z.enum(['semantic', 'episodic', 'procedural', 'all']).optional(),
  limit: z.number().int().min(1).max(100).optional(),
  min_relevance: z.number().min(0).max(1).optional(),
  agent_filter: z.string().optional(),
});

/**
 * KPI calculation request schema
 */
export const KPICalculateRequestSchema = z.object({
  kpi_id: z.string().min(1, 'KPI ID required'),
  date_from: z.string().optional(),
  date_to: z.string().optional(),
  brand: z.string().optional(),
  territory: z.string().optional(),
  hcp_segment: z.string().optional(),
  force_refresh: z.boolean().optional(),
});

/**
 * Batch KPI calculation request schema
 */
export const BatchKPICalculateRequestSchema = z.object({
  kpi_ids: z.array(z.string()).min(1, 'At least one KPI required'),
  workstream: z.string().optional(),
  date_from: z.string().optional(),
  date_to: z.string().optional(),
  brand: z.string().optional(),
  force_refresh: z.boolean().optional(),
});

/**
 * Learning cycle request schema
 */
export const LearningCycleRequestSchema = z.object({
  focus_agents: z.array(z.string()).optional(),
  min_feedback_count: z.number().int().min(1).optional(),
  time_window_hours: z.number().int().min(1).max(720).optional(),
  include_patterns: z.boolean().optional(),
  auto_apply_updates: z.boolean().optional(),
});

/**
 * Experiment design request schema
 */
export const ExperimentDesignRequestSchema = z.object({
  name: z.string().min(1, 'Experiment name required').max(200),
  description: z.string().max(2000).optional(),
  hypothesis: z.string().min(1, 'Hypothesis required'),
  treatment: z.string().min(1, 'Treatment description required'),
  outcome_metric: z.string().min(1, 'Outcome metric required'),
  target_sample_size: z.number().int().min(10).optional(),
  target_power: z.number().min(0.5).max(0.99).optional(),
  alpha: z.number().min(0.01).max(0.1).optional(),
  brand: z.string().optional(),
});

/**
 * Pagination info schema
 */
export const PaginationSchema = z.object({
  page: z.number().int().positive(),
  page_size: z.number().int().positive(),
  total: z.number().int().nonnegative(),
  total_pages: z.number().int().nonnegative(),
});

// =============================================================================
// KPI SCHEMAS
// =============================================================================

/**
 * KPI threshold configuration
 */
export const KPIThresholdSchema = z.object({
  target: z.number().optional(),
  warning: z.number().optional(),
  critical: z.number().optional(),
});

/**
 * KPI metadata/definition
 */
export const KPIMetadataSchema = z.object({
  id: z.string(),
  name: z.string(),
  definition: z.string(),
  formula: z.string(),
  calculation_type: z.string(),
  workstream: z.string(),
  tables: z.array(z.string()),
  columns: z.array(z.string()),
  view: z.string().optional(),
  threshold: KPIThresholdSchema.optional(),
  unit: z.string().optional(),
  frequency: z.string(),
  primary_causal_library: z.string(),
  brand: z.string().optional(),
  note: z.string().optional(),
});

/**
 * KPI calculation result
 */
export const KPIResultSchema = z.object({
  kpi_id: z.string(),
  value: z.number().optional(),
  status: z.string(),
  calculated_at: z.string(),
  cached: z.boolean(),
  cache_expires_at: z.string().optional(),
  error: z.string().optional(),
  causal_library_used: z.string().optional(),
  confidence_interval: z.array(z.number()).length(2).optional(),
  p_value: z.number().optional(),
  effect_size: z.number().optional(),
  metadata: z.record(z.string(), z.unknown()),
});

/**
 * KPI list response
 */
export const KPIListResponseSchema = z.object({
  kpis: z.array(KPIMetadataSchema),
  total: z.number().int().nonnegative(),
  workstream: z.string().optional(),
  causal_library: z.string().optional(),
});

/**
 * Workstream info
 */
export const WorkstreamInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  kpi_count: z.number().int().nonnegative(),
  description: z.string().optional(),
});

/**
 * Workstream list response
 */
export const WorkstreamListResponseSchema = z.object({
  workstreams: z.array(WorkstreamInfoSchema),
  total: z.number().int().nonnegative(),
});

/**
 * Batch KPI calculation response
 */
export const BatchKPICalculationResponseSchema = z.object({
  workstream: z.string().optional(),
  results: z.array(KPIResultSchema),
  calculated_at: z.string(),
  total_kpis: z.number().int().nonnegative(),
  successful: z.number().int().nonnegative(),
  failed: z.number().int().nonnegative(),
});

/**
 * Cache invalidation response
 */
export const CacheInvalidationResponseSchema = z.object({
  invalidated_count: z.number().int().nonnegative(),
  message: z.string(),
});

/**
 * KPI health response
 */
export const KPIHealthResponseSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  registry_loaded: z.boolean(),
  total_kpis: z.number().int().nonnegative(),
  cache_enabled: z.boolean(),
  cache_size: z.number().int().nonnegative(),
  database_connected: z.boolean(),
  workstreams_available: z.array(z.string()),
  last_calculation: z.string().optional(),
  error: z.string().optional(),
});

// =============================================================================
// MONITORING SCHEMAS
// =============================================================================

/**
 * Drift metric schema
 */
export const DriftMetricSchema = z.object({
  feature: z.string(),
  drift_score: z.number(),
  threshold: z.number(),
  is_drifted: z.boolean(),
  drift_type: z.string().optional(),
  baseline_mean: z.number().optional(),
  current_mean: z.number().optional(),
  baseline_std: z.number().optional(),
  current_std: z.number().optional(),
});

/**
 * Drift detection response
 */
export const DriftDetectionResponseSchema = z.object({
  run_id: z.string(),
  model_id: z.string(),
  timestamp: z.string(),
  overall_drift_detected: z.boolean(),
  drift_score: z.number(),
  metrics: z.array(DriftMetricSchema),
  recommendation: z.string().optional(),
});

/**
 * Alert schema
 */
export const AlertSchema = z.object({
  id: z.string(),
  type: z.string(),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  message: z.string(),
  source: z.string(),
  timestamp: z.string(),
  acknowledged: z.boolean(),
  acknowledged_by: z.string().optional(),
  acknowledged_at: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Alert list response
 */
export const AlertListResponseSchema = z.object({
  alerts: z.array(AlertSchema),
  total: z.number().int().nonnegative(),
  unacknowledged_count: z.number().int().nonnegative(),
});

// =============================================================================
// PREDICTIONS SCHEMAS
// =============================================================================

/**
 * Model prediction schema
 */
export const PredictionSchema = z.object({
  prediction_id: z.string(),
  model_id: z.string(),
  input_features: z.record(z.string(), z.unknown()),
  predicted_value: z.union([z.number(), z.string(), z.array(z.number())]),
  confidence: z.number().min(0).max(1).optional(),
  confidence_interval: z.array(z.number()).length(2).optional(),
  timestamp: z.string(),
  explanation: z.string().optional(),
  feature_importance: z.record(z.string(), z.number()).optional(),
});

/**
 * Batch prediction response
 */
export const BatchPredictionResponseSchema = z.object({
  batch_id: z.string(),
  model_id: z.string(),
  predictions: z.array(PredictionSchema),
  total: z.number().int().nonnegative(),
  successful: z.number().int().nonnegative(),
  failed: z.number().int().nonnegative(),
  processing_time_ms: z.number(),
});

// =============================================================================
// GRAPH SCHEMAS
// =============================================================================

/**
 * Graph node schema
 */
export const GraphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string(),
  properties: z.record(z.string(), z.unknown()).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Graph edge schema
 */
export const GraphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  type: z.string(),
  weight: z.number().optional(),
  properties: z.record(z.string(), z.unknown()).optional(),
});

/**
 * Graph query response
 */
export const GraphQueryResponseSchema = z.object({
  nodes: z.array(GraphNodeSchema),
  edges: z.array(GraphEdgeSchema),
  query: z.string().optional(),
  execution_time_ms: z.number().optional(),
});

// =============================================================================
// CAUSAL SCHEMAS
// =============================================================================

/**
 * Causal effect schema
 */
export const CausalEffectSchema = z.object({
  treatment: z.string(),
  outcome: z.string(),
  effect: z.number(),
  effect_type: z.string(),
  confidence_interval: z.array(z.number()).length(2).optional(),
  p_value: z.number().optional(),
  sample_size: z.number().int().optional(),
  method: z.string(),
});

/**
 * Causal analysis response
 */
export const CausalAnalysisResponseSchema = z.object({
  analysis_id: z.string(),
  treatment: z.string(),
  outcome: z.string(),
  effects: z.array(CausalEffectSchema),
  confounders: z.array(z.string()).optional(),
  mediators: z.array(z.string()).optional(),
  model_used: z.string(),
  timestamp: z.string(),
  warnings: z.array(z.string()).optional(),
});

// =============================================================================
// HEALTH SCHEMAS
// =============================================================================

/**
 * General health response schema
 */
export const HealthResponseSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  version: z.string().optional(),
  timestamp: z.string(),
  services: z.record(z.string(), z.object({
    status: z.enum(['healthy', 'degraded', 'unhealthy']),
    latency_ms: z.number().optional(),
    error: z.string().optional(),
  })).optional(),
});

// =============================================================================
// AGENT SCHEMAS
// =============================================================================

/**
 * Agent status schema for individual agents
 */
export const AgentSchema = z.object({
  id: z.string(),
  name: z.string(),
  tier: z.number().int().min(0).max(5),
  status: z.enum(['idle', 'active', 'processing', 'complete', 'error']),
  capabilities: z.array(z.string()),
  lastActive: z.string().optional(),
  errorMessage: z.string().optional(),
});

/**
 * Agent status response schema
 */
export const AgentStatusResponseSchema = z.object({
  agents: z.array(AgentSchema),
  total: z.number().int().nonnegative().optional(),
  timestamp: z.string().optional(),
});

// =============================================================================
// CHAT/COPILOTKIT SCHEMAS
// =============================================================================

/**
 * Chat response schema (matches ChatResponse in copilotkit.py)
 */
export const ChatResponseSchema = z.object({
  success: z.boolean(),
  session_id: z.string(),
  response: z.string(),
  conversation_title: z.string().nullable().optional(),
  agent_name: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  // Dispatch observability fields
  orchestrator_used: z.boolean().default(false),
  agents_dispatched: z.array(z.string()).default([]),
  routed_agent: z.string().nullable().optional(),
  response_confidence: z.number().nullable().optional(),
  execution_time_ms: z.number().nullable().optional(),
  intent: z.string().nullable().optional(),
  intent_confidence: z.number().nullable().optional(),
});

// =============================================================================
// VALIDATION UTILITIES
// =============================================================================

/**
 * Validation error with structured details
 */
export class ApiValidationError extends Error {
  public readonly issues: z.ZodIssue[];
  public readonly endpoint: string;
  public readonly rawData: unknown;

  constructor(
    message: string,
    issues: z.ZodIssue[],
    endpoint: string,
    rawData: unknown
  ) {
    super(message);
    this.name = 'ApiValidationError';
    this.issues = issues;
    this.endpoint = endpoint;
    this.rawData = rawData;
  }

  /**
   * Get a formatted error message with all issues
   */
  get formattedMessage(): string {
    const issueMessages = this.issues.map(
      (issue) => `  - ${issue.path.join('.')}: ${issue.message}`
    );
    return `API validation failed for ${this.endpoint}:\n${issueMessages.join('\n')}`;
  }
}

/**
 * Validate API response against a Zod schema
 *
 * @param schema - Zod schema to validate against
 * @param data - Raw API response data
 * @param endpoint - Endpoint name for error reporting
 * @param options - Validation options
 * @returns Validated and typed data
 * @throws ApiValidationError if validation fails
 */
export function validateApiResponse<T extends z.ZodTypeAny>(
  schema: T,
  data: unknown,
  endpoint: string,
  options: {
    /** Log validation errors to console in development */
    logErrors?: boolean;
    /** Throw error on validation failure (default: true) */
    throwOnError?: boolean;
  } = {}
): z.infer<T> {
  const { logErrors = true, throwOnError = true } = options;

  const result = schema.safeParse(data);

  if (!result.success) {
    if (logErrors && import.meta.env.DEV) {
      console.error(`[API Validation] ${endpoint}:`, {
        issues: result.error.issues,
        data,
      });
    }

    if (throwOnError) {
      throw new ApiValidationError(
        `API response validation failed for ${endpoint}`,
        result.error.issues,
        endpoint,
        data
      );
    }

    // If not throwing, return data as-is (unsafe cast)
    return data as z.infer<T>;
  }

  return result.data;
}

/**
 * Create a validated API fetch wrapper
 *
 * @param schema - Zod schema for response validation
 * @param fetcher - Async function that fetches the data
 * @param endpoint - Endpoint name for error reporting
 * @returns Validated response data
 */
export async function fetchWithValidation<T extends z.ZodTypeAny>(
  schema: T,
  fetcher: () => Promise<unknown>,
  endpoint: string
): Promise<z.infer<T>> {
  const data = await fetcher();
  return validateApiResponse(schema, data, endpoint);
}

// =============================================================================
// SCHEMA REGISTRY
// =============================================================================

/**
 * Registry of all API response schemas
 * Used for dynamic validation based on endpoint
 */
export const schemaRegistry = {
  // KPI
  'kpi.list': KPIListResponseSchema,
  'kpi.detail': KPIMetadataSchema,
  'kpi.calculate': KPIResultSchema,
  'kpi.batch': BatchKPICalculationResponseSchema,
  'kpi.health': KPIHealthResponseSchema,
  'kpi.workstreams': WorkstreamListResponseSchema,
  'kpi.invalidate': CacheInvalidationResponseSchema,

  // Monitoring
  'monitoring.drift': DriftDetectionResponseSchema,
  'monitoring.alerts': AlertListResponseSchema,

  // Predictions
  'predictions.single': PredictionSchema,
  'predictions.batch': BatchPredictionResponseSchema,

  // Graph
  'graph.query': GraphQueryResponseSchema,
  'graph.node': GraphNodeSchema,

  // Causal
  'causal.analysis': CausalAnalysisResponseSchema,

  // Chat
  'chat.response': ChatResponseSchema,

  // Health
  'health': HealthResponseSchema,

  // Agents
  'agents.status': AgentStatusResponseSchema,
} as const;

/**
 * Registry of request body schemas
 * Used for validating data before API submission
 */
export const requestSchemaRegistry = {
  // Digital Twin
  'digitalTwin.simulate': SimulationRequestSchema,

  // Chat/Feedback
  'chat.feedback': ChatFeedbackRequestSchema,

  // Monitoring
  'monitoring.driftDetect': DriftDetectionRequestSchema,

  // Graph
  'graph.search': GraphSearchRequestSchema,

  // Memory
  'memory.search': MemorySearchRequestSchema,

  // KPI
  'kpi.calculate': KPICalculateRequestSchema,
  'kpi.batchCalculate': BatchKPICalculateRequestSchema,

  // Feedback Learning
  'feedback.learningCycle': LearningCycleRequestSchema,

  // Experiments
  'experiments.design': ExperimentDesignRequestSchema,
} as const;

export type RequestSchemaKey = keyof typeof requestSchemaRegistry;

/**
 * Get request schema by registry key
 */
export function getRequestSchema(key: RequestSchemaKey) {
  return requestSchemaRegistry[key];
}

/**
 * Validate request body before API submission
 *
 * @param schema - Zod schema to validate against
 * @param data - Request body data
 * @param endpoint - Endpoint name for error reporting
 * @returns Validated and typed data
 * @throws ApiValidationError if validation fails
 */
export function validateRequestBody<T extends z.ZodTypeAny>(
  schema: T,
  data: unknown,
  endpoint: string
): z.infer<T> {
  const result = schema.safeParse(data);

  if (!result.success) {
    if (import.meta.env.DEV) {
      console.error(`[Request Validation] ${endpoint}:`, {
        issues: result.error.issues,
        data,
      });
    }

    throw new ApiValidationError(
      `Request body validation failed for ${endpoint}`,
      result.error.issues,
      endpoint,
      data
    );
  }

  return result.data;
}

export type SchemaKey = keyof typeof schemaRegistry;

/**
 * Get schema by registry key
 */
export function getSchema(key: SchemaKey) {
  return schemaRegistry[key];
}

// =============================================================================
// TYPE EXPORTS
// =============================================================================

// Export inferred types from schemas for use in components
export type ApiErrorResponse = z.infer<typeof ApiErrorResponseSchema>;
export type KPIMetadataValidated = z.infer<typeof KPIMetadataSchema>;
export type KPIResultValidated = z.infer<typeof KPIResultSchema>;
export type KPIListResponseValidated = z.infer<typeof KPIListResponseSchema>;
export type DriftDetectionResponseValidated = z.infer<typeof DriftDetectionResponseSchema>;
export type AlertValidated = z.infer<typeof AlertSchema>;
export type PredictionValidated = z.infer<typeof PredictionSchema>;
export type GraphNodeValidated = z.infer<typeof GraphNodeSchema>;
export type CausalAnalysisResponseValidated = z.infer<typeof CausalAnalysisResponseSchema>;
export type ChatResponseValidated = z.infer<typeof ChatResponseSchema>;
export type AgentValidated = z.infer<typeof AgentSchema>;
export type AgentStatusResponseValidated = z.infer<typeof AgentStatusResponseSchema>;

// Request body types inferred from schemas
export type SimulationRequest = z.infer<typeof SimulationRequestSchema>;
export type ChatFeedbackRequest = z.infer<typeof ChatFeedbackRequestSchema>;
export type DriftDetectionRequest = z.infer<typeof DriftDetectionRequestSchema>;
export type GraphSearchRequest = z.infer<typeof GraphSearchRequestSchema>;
export type MemorySearchRequest = z.infer<typeof MemorySearchRequestSchema>;
export type KPICalculateRequest = z.infer<typeof KPICalculateRequestSchema>;
export type BatchKPICalculateRequest = z.infer<typeof BatchKPICalculateRequestSchema>;
export type LearningCycleRequest = z.infer<typeof LearningCycleRequestSchema>;
export type ExperimentDesignRequest = z.infer<typeof ExperimentDesignRequestSchema>;
export type InterventionType = z.infer<typeof InterventionTypeEnum>;
