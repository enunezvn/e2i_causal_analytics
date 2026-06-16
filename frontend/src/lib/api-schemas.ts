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
  suggested_action: z.string().optional(),
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
  value_format: z.string().optional(),
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
  // 'database' = real (synthetic-excluded) rows; 'synthetic' = computed over
  // synthetic-gold rows in E2I_KPI_INCLUDE_SYNTHETIC demo mode (badged in the UI).
  data_source: z.string().optional(),
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
// AUDIT SCHEMAS
// =============================================================================

/**
 * Audit chain entry schema - individual audit record
 */
export const AuditEntrySchema = z.object({
  entry_id: z.string().uuid(),
  workflow_id: z.string().uuid(),
  sequence_number: z.number().int().nonnegative(),
  agent_name: z.string(),
  agent_tier: z.number().int().min(0).max(5),
  action_type: z.string(),
  created_at: z.string(),
  duration_ms: z.number().int().nonnegative().nullable().optional(),
  validation_passed: z.boolean().nullable().optional(),
  confidence_score: z.number().min(0).max(1).nullable().optional(),
  refutation_results: z.record(z.string(), z.unknown()).nullable().optional(),
  previous_entry_id: z.string().uuid().nullable().optional(),
  previous_hash: z.string().nullable().optional(),
  entry_hash: z.string(),
  user_id: z.string().nullable().optional(),
  session_id: z.string().uuid().nullable().optional(),
  brand: z.string().nullable().optional(),
});

/**
 * Chain verification result schema
 */
export const ChainVerificationSchema = z.object({
  workflow_id: z.string().uuid(),
  is_valid: z.boolean(),
  entries_checked: z.number().int().nonnegative(),
  first_invalid_entry: z.string().uuid().nullable().optional(),
  error_message: z.string().nullable().optional(),
  verified_at: z.string(),
});

/**
 * Workflow summary schema
 */
export const WorkflowSummarySchema = z.object({
  workflow_id: z.string().uuid(),
  total_entries: z.number().int().nonnegative(),
  first_entry_at: z.string().nullable().optional(),
  last_entry_at: z.string().nullable().optional(),
  agents_involved: z.array(z.string()),
  tiers_involved: z.array(z.number().int()),
  chain_verified: z.boolean(),
  brand: z.string().nullable().optional(),
  total_duration_ms: z.number().int().nonnegative(),
  avg_confidence_score: z.number().nullable().optional(),
  validation_passed_count: z.number().int().nonnegative(),
  validation_failed_count: z.number().int().nonnegative(),
});

/**
 * Recent workflow item schema
 */
export const RecentWorkflowSchema = z.object({
  workflow_id: z.string().uuid(),
  started_at: z.string(),
  entry_count: z.number().int().nonnegative(),
  first_agent: z.string(),
  last_agent: z.string(),
  brand: z.string().nullable().optional(),
});

/**
 * Audit entries list response
 */
export const AuditEntriesResponseSchema = z.array(AuditEntrySchema);

/**
 * Recent workflows list response
 */
export const RecentWorkflowsResponseSchema = z.array(RecentWorkflowSchema);

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

/**
 * Per-tier performance item (matches TierMetricsItem in analytics.py).
 * avg_response_time_ms / success_rate are null when unmeasured (-> "—").
 */
export const TierMetricsItemSchema = z.object({
  tier: z.number().int().min(0).max(5),
  tasks_completed: z.number().int().nonnegative(),
  avg_response_time_ms: z.number().nullable().optional(),
  success_rate: z.number().nullable().optional(),
});

/**
 * Per-tier metrics response (matches TierMetricsResponse in analytics.py).
 */
export const TierMetricsResponseSchema = z.object({
  tiers: z.array(TierMetricsItemSchema),
  window_hours: z.number().int(),
  generated_at: z.string().optional(),
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
// WIRE SCHEMAS (C31 — opt-in response validation for per-client GET reads)
// =============================================================================
//
// These schemas FAITHFULLY mirror the canonical response interfaces in
// `frontend/src/types/*` (which themselves track the FastAPI backend schemas)
// and the shapes returned by the MSW mocks in `frontend/src/mocks`.
//
// They are intentionally SEPARATE from the older `*ResponseSchema` exports
// above: a number of those legacy schemas were aspirational and DO NOT match
// the live contract (e.g. `PredictionSchema` uses `prediction_id` /
// `predicted_value` while the real `/models/predict` response uses
// `model_name` / `prediction` / `latency_ms`; `KPIListResponseSchema.workstream`
// is non-nullable while the backend sends `null`). Wiring those legacy schemas
// would FALSELY reject correct responses. The `*WireSchema` set below is the
// validated contract that the per-domain clients (`src/api/*.ts`) opt into via
// the base client's `schema` parameter.
//
// Field nullability rule: fields the backend/mock may send as `null` use
// `.nullable()`; fields that may be absent use `.optional()`. We keep object
// schemas permissive about EXTRA keys (Zod's default is to strip unknown keys,
// not reject them) so additive backend changes never break the UI — only
// missing/mis-typed REQUIRED fields trip validation, which is the drift we want
// to catch.
//
// HOW TO WIRE MORE ENDPOINTS (the C31 pattern):
//   1. Add a `*WireSchema` here that mirrors the canonical interface in
//      `frontend/src/types/<domain>.ts` (use `.nullable()` for null-able fields).
//   2. In `src/api/<domain>.ts` import it and pass it via the base client's
//      opt-in `schema` param, e.g.
//        return get<Foo>('/foo', params, { schema: FooWireSchema });
//      (For helpers without a `params` arg, pass `undefined` as the 2nd arg.)
//   3. Add a red-first test: a malformed MSW response must throw
//      `ApiValidationError`; a valid one must pass through parsed.
//
// PRIMARY GET reads currently wired: kpi (list/workstreams/metadata/value/health),
// predictions (model health/status), monitoring (alerts/model-health/drift-
// history/runs), causal (health/estimators), graph (health/stats), segments
// (policies/health), resources (scenarios/health), memory (episodic list+single/
// semantic-paths), rag (extract/causal-subgraph/causal-path/health), health-score
// (check/quick/full/components/models/pipelines/agents/history/status).
//
// DEFERRED (intentionally not yet wired, with reasoning):
//   - GET /memory/stats and GET /v1/rag/stats: both routes return
//     `Dict[str, Any]` with NO `response_model`, so their shape is not
//     backend-anchored. A wire schema would risk false-rejecting a valid
//     passthrough payload (same reasoning as `getModelInfo` below).
//   - GET/POST `SegmentAnalysisResponse` (segments) and `OptimizationResponse`
//     (resources): large, deeply-nested, mostly-optional response shapes;
//     deferred per the heavy/volatile-schema policy below.
//   - POST/PUT/PATCH/DELETE mutations: schema-param works on every helper, but
//     C31 prioritised high-traffic GET *reads* where contract drift is most
//     likely to surface in dashboards. Mutations can be wired with the same
//     pattern when needed.
//   - `src/api/analytics.ts`: it calls the raw axios `apiClient.get(...)` with
//     the query string baked into the path, NOT the `get()` helper, so it has no
//     `schema` opt-in. Wiring it would require a small refactor to the helper
//     and is out of this slice's scope.
//   - Heavy async causal pipelines (hierarchical/sequential/parallel responses)
//     and graph node/relationship list shapes have large, more volatile schemas;
//     deferred to keep the wire schemas faithful and low-maintenance.
//   - `getModelInfo` (GET /models/{name}/info): the backend route has NO
//     response_model — it passes the BentoML /metadata payload through verbatim,
//     so `name` is not contract-guaranteed. A wire schema would false-reject
//     valid responses; wire it once the backend anchors the route with a
//     response_model.

// ----- KPI -----

/** Faithful mirror of `KPIThreshold` (types/kpi.ts). */
export const KPIThresholdWireSchema = z.object({
  target: z.number().optional(),
  warning: z.number().optional(),
  critical: z.number().optional(),
});

/** Faithful mirror of `KPIMetadata` (types/kpi.ts). */
export const KPIMetadataWireSchema = z.object({
  id: z.string(),
  name: z.string(),
  definition: z.string(),
  formula: z.string(),
  calculation_type: z.string(),
  workstream: z.string(),
  tables: z.array(z.string()),
  columns: z.array(z.string()),
  view: z.string().nullable().optional(),
  threshold: KPIThresholdWireSchema.nullable().optional(),
  unit: z.string().nullable().optional(),
  value_format: z.string().nullable().optional(),
  frequency: z.string(),
  primary_causal_library: z.string(),
  brand: z.string().nullable().optional(),
  note: z.string().nullable().optional(),
});

/** Faithful mirror of `KPIResult` (types/kpi.ts). */
export const KPIResultWireSchema = z.object({
  kpi_id: z.string(),
  value: z.number().nullable().optional(),
  status: z.string(),
  calculated_at: z.string(),
  cached: z.boolean(),
  cache_expires_at: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  // 'database' = real (synthetic-excluded) rows; 'synthetic' = computed over
  // synthetic-gold rows in E2I_KPI_INCLUDE_SYNTHETIC demo mode. MUST be declared
  // here or Zod strips it from getKPIValue()'s result before the FE can badge it.
  data_source: z.string().nullable().optional(),
  causal_library_used: z.string().nullable().optional(),
  confidence_interval: z.array(z.number()).nullable().optional(),
  p_value: z.number().nullable().optional(),
  effect_size: z.number().nullable().optional(),
  metadata: z.record(z.string(), z.unknown()),
});

/** Faithful mirror of `KPIListResponse` (types/kpi.ts). `workstream`/`causal_library` are sent as `null` by the backend when unset. */
export const KPIListResponseWireSchema = z.object({
  kpis: z.array(KPIMetadataWireSchema),
  total: z.number().int().nonnegative(),
  workstream: z.string().nullable().optional(),
  causal_library: z.string().nullable().optional(),
});

/** Faithful mirror of `WorkstreamInfo` (types/kpi.ts). */
export const WorkstreamInfoWireSchema = z.object({
  id: z.string(),
  name: z.string(),
  kpi_count: z.number().int().nonnegative(),
  description: z.string().nullable().optional(),
});

/** Faithful mirror of `WorkstreamListResponse` (types/kpi.ts). */
export const WorkstreamListResponseWireSchema = z.object({
  workstreams: z.array(WorkstreamInfoWireSchema),
  total: z.number().int().nonnegative(),
});

/** Faithful mirror of `KPIHealthResponse` (types/kpi.ts). */
export const KPIHealthResponseWireSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  registry_loaded: z.boolean(),
  total_kpis: z.number().int().nonnegative(),
  cache_enabled: z.boolean(),
  cache_size: z.number().int().nonnegative(),
  database_connected: z.boolean(),
  workstreams_available: z.array(z.string()),
  last_calculation: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
});

// ----- PREDICTIONS -----

/** Faithful mirror of `ModelEndpointHealth` (types/predictions.ts). */
export const ModelEndpointHealthWireSchema = z.object({
  model_name: z.string(),
  status: z.string(),
  endpoint: z.string(),
  last_check: z.string(),
  error: z.string().nullable().optional(),
});

/** Faithful mirror of `ModelsStatusResponse` (types/predictions.ts). */
export const ModelsStatusResponseWireSchema = z.object({
  total_models: z.number().int().nonnegative(),
  healthy_count: z.number().int().nonnegative(),
  unhealthy_count: z.number().int().nonnegative(),
  models: z.array(ModelEndpointHealthWireSchema),
  timestamp: z.string(),
});

// ----- MONITORING -----

/** Faithful mirror of `AlertItem` (types/monitoring.ts). */
export const AlertItemWireSchema = z.object({
  id: z.string(),
  model_version: z.string(),
  alert_type: z.string(),
  severity: z.string(),
  title: z.string(),
  description: z.string(),
  status: z.string(),
  triggered_at: z.string(),
  acknowledged_at: z.string().nullable().optional(),
  acknowledged_by: z.string().nullable().optional(),
  resolved_at: z.string().nullable().optional(),
  resolved_by: z.string().nullable().optional(),
});

/** Faithful mirror of `AlertListResponse` (types/monitoring.ts). */
export const AlertListResponseWireSchema = z.object({
  total_count: z.number().int().nonnegative(),
  active_count: z.number().int().nonnegative(),
  alerts: z.array(AlertItemWireSchema),
});

/** Faithful mirror of `DriftHistoryItem` (types/monitoring.ts). */
export const DriftHistoryItemWireSchema = z.object({
  id: z.string(),
  model_version: z.string(),
  feature_name: z.string(),
  drift_type: z.string(),
  drift_score: z.number(),
  severity: z.string(),
  detected_at: z.string(),
  baseline_start: z.string(),
  baseline_end: z.string(),
  current_start: z.string(),
  current_end: z.string(),
});

/** Faithful mirror of `DriftHistoryResponse` (types/monitoring.ts). */
export const DriftHistoryResponseWireSchema = z.object({
  model_id: z.string(),
  total_records: z.number().int().nonnegative(),
  records: z.array(DriftHistoryItemWireSchema),
});

/** Faithful mirror of `MonitoringRunItem` (types/monitoring.ts). */
export const MonitoringRunItemWireSchema = z.object({
  id: z.string(),
  model_version: z.string(),
  run_type: z.string(),
  started_at: z.string(),
  completed_at: z.string().nullable().optional(),
  features_checked: z.number().int().nonnegative(),
  drift_detected_count: z.number().int().nonnegative(),
  alerts_generated: z.number().int().nonnegative(),
  duration_ms: z.number(),
  error_message: z.string().nullable().optional(),
});

/** Faithful mirror of `MonitoringRunsResponse` (types/monitoring.ts). */
export const MonitoringRunsResponseWireSchema = z.object({
  model_id: z.string().nullable().optional(),
  total_runs: z.number().int().nonnegative(),
  runs: z.array(MonitoringRunItemWireSchema),
});

/** Faithful mirror of `ModelHealthSummary` (types/monitoring.ts). */
export const ModelHealthSummaryWireSchema = z.object({
  model_id: z.string(),
  overall_health: z.enum(['healthy', 'warning', 'critical']),
  last_check: z.string().nullable().optional(),
  drift_score: z.number(),
  active_alerts: z.number().int().nonnegative(),
  last_retrained: z.string().nullable().optional(),
  performance_trend: z.enum(['stable', 'improving', 'degrading']),
  recommendations: z.array(z.string()),
});

// ----- CAUSAL -----

/** Faithful mirror of `CausalHealthResponse` (types/causal.ts). */
export const CausalHealthResponseWireSchema = z.object({
  status: z.string(),
  libraries_available: z.record(z.string(), z.boolean()),
  estimators_loaded: z.number().int().nonnegative(),
  pipeline_orchestrator_ready: z.boolean(),
  hierarchical_analyzer_ready: z.boolean(),
  last_analysis: z.string().nullable().optional(),
  analysis_count_24h: z.number().int().nonnegative(),
  average_latency_ms: z.number().nullable().optional(),
  error: z.string().nullable().optional(),
});

/** Faithful mirror of `CausalAnalysisHistoryItem` (types/causal.ts). */
export const CausalAnalysisHistoryItemWireSchema = z.object({
  memory_id: z.string(),
  event_type: z.string(),
  description: z.string().nullable().optional(),
  occurred_at: z.string(),
  agent_name: z.string().nullable().optional(),
  ate_estimate: z.number().nullable().optional(),
  confidence: z.number().nullable().optional(),
  model_used: z.string().nullable().optional(),
});

/** Faithful mirror of `CausalAnalysisHistoryResponse` (types/causal.ts). */
export const CausalAnalysisHistoryResponseWireSchema = z.object({
  items: z.array(CausalAnalysisHistoryItemWireSchema),
  total: z.number().int().nonnegative(),
});

/** Faithful mirror of `EstimatorInfo` (types/causal.ts). */
export const EstimatorInfoWireSchema = z.object({
  name: z.string(),
  library: z.string(),
  estimator_type: z.string(),
  description: z.string(),
  best_for: z.array(z.string()),
  parameters: z.array(z.string()),
  supports_confidence_intervals: z.boolean(),
  supports_heterogeneous_effects: z.boolean(),
});

/** Faithful mirror of `EstimatorListResponse` (types/causal.ts). */
export const EstimatorListResponseWireSchema = z.object({
  estimators: z.array(EstimatorInfoWireSchema),
  total: z.number().int().nonnegative(),
  by_library: z.record(z.string(), z.array(z.string())),
});

// ----- GRAPH -----

/** Faithful mirror of `GraphHealthResponse` (types/graph.ts). */
export const GraphHealthResponseWireSchema = z.object({
  status: z.enum(['healthy', 'degraded']),
  graphiti: z.enum(['connected', 'unavailable']),
  falkordb: z.enum(['connected', 'unavailable']),
  websocket_connections: z.number().int().nonnegative(),
  timestamp: z.string(),
});

/** Faithful mirror of `GraphStatsResponse` (types/graph.ts). */
export const GraphStatsResponseWireSchema = z.object({
  total_nodes: z.number().int().nonnegative(),
  total_relationships: z.number().int().nonnegative(),
  nodes_by_type: z.record(z.string(), z.number()),
  relationships_by_type: z.record(z.string(), z.number()),
  total_episodes: z.number().int().nonnegative(),
  total_communities: z.number().int().nonnegative(),
  last_updated: z.string().nullable().optional(),
  timestamp: z.string(),
});

// ----- SEGMENTS -----
//
// Backend route: src/api/routes/segments.py
//   GET /segments/policies  -> response_model=PolicyListResponse
//   GET /segments/health    -> response_model=SegmentHealthResponse
// DEFERRED: GET /segments/{id} and POST /segments/analyze return
// `SegmentAnalysisResponse`, a large, deeply-nested, mostly-optional shape
// (per-segment CATE maps, responder profiles, uplift metrics). Per the C31
// deferral policy for heavy/volatile schemas, it is left unwired.

/** Faithful mirror of `PolicyRecommendation` (segments.py PolicyRecommendation). */
export const PolicyRecommendationWireSchema = z.object({
  segment: z.string(),
  current_treatment_rate: z.number(),
  recommended_treatment_rate: z.number(),
  expected_incremental_outcome: z.number(),
  confidence: z.number(),
});

/** Faithful mirror of `PolicyListResponse` (segments.py PolicyListResponse). */
export const PolicyListResponseWireSchema = z.object({
  total_count: z.number().int().nonnegative(),
  recommendations: z.array(PolicyRecommendationWireSchema),
  expected_total_lift: z.number(),
});

/** Faithful mirror of `SegmentHealthResponse` (segments.py SegmentHealthResponse). */
export const SegmentHealthResponseWireSchema = z.object({
  status: z.string(),
  agent_available: z.boolean(),
  econml_available: z.boolean(),
  causalml_available: z.boolean(),
  last_analysis: z.string().nullable().optional(),
  analyses_24h: z.number().int().nonnegative(),
});

// ----- RESOURCES -----
//
// Backend route: src/api/routes/resource_optimizer.py
//   GET /resources/scenarios -> response_model=ScenarioListResponse
//   GET /resources/health    -> response_model=ResourceHealthResponse
// DEFERRED: GET /resources/{id} and POST /resources/optimize return
// `OptimizationResponse`, a large allocation/scenario/impact shape; deferred
// per the C31 heavy-schema policy.

/** Faithful mirror of `ScenarioResult` (resource_optimizer.py ScenarioResult). */
export const ScenarioResultWireSchema = z.object({
  scenario_name: z.string(),
  total_allocation: z.number(),
  projected_outcome: z.number(),
  roi: z.number(),
  constraint_violations: z.array(z.string()),
});

/** Faithful mirror of `ScenarioListResponse` (resource_optimizer.py ScenarioListResponse). */
export const ScenarioListResponseWireSchema = z.object({
  total_count: z.number().int().nonnegative(),
  scenarios: z.array(ScenarioResultWireSchema),
});

/** Faithful mirror of `ResourceHealthResponse` (resource_optimizer.py ResourceHealthResponse). */
export const ResourceHealthResponseWireSchema = z.object({
  status: z.string(),
  agent_available: z.boolean(),
  scipy_available: z.boolean(),
  last_optimization: z.string().nullable().optional(),
  optimizations_24h: z.number().int().nonnegative(),
  // 'durable' (Redis, shared across workers) or 'degraded' (process-local
  // in-memory fallback — cross-worker reads can 404). Optional for backward
  // compatibility with older backends that predate the field.
  storage_mode: z.string().optional(),
});

// ----- MEMORY -----
//
// Backend route: src/api/routes/memory.py
//   GET /memory/episodic       -> response_model=List[EpisodicMemoryResponse]
//   GET /memory/episodic/{id}  -> response_model=EpisodicMemoryResponse
//   GET /memory/semantic/paths -> response_model=SemanticPathResponse
// DEFERRED: GET /memory/stats has NO response_model (returns Dict[str, Any]),
// so its shape is not backend-anchored — wiring a schema would risk
// false-rejecting a valid passthrough payload. Left unwired (same reasoning
// as C31's deferral of `getModelInfo`).

/** Faithful mirror of `EpisodicMemoryResponse` (memory.py EpisodicMemoryResponse). */
export const EpisodicMemoryResponseWireSchema = z.object({
  id: z.string(),
  content: z.string(),
  event_type: z.string(),
  session_id: z.string().nullable().optional(),
  agent_name: z.string().nullable().optional(),
  brand: z.string().nullable().optional(),
  region: z.string().nullable().optional(),
  created_at: z.string(),
  metadata: z.record(z.string(), z.unknown()),
});

/** Faithful mirror of `List[EpisodicMemoryResponse]` (memory.py GET /episodic). */
export const EpisodicMemoryListResponseWireSchema = z.array(
  EpisodicMemoryResponseWireSchema
);

/** Faithful mirror of `SemanticPathResponse` (memory.py SemanticPathResponse). */
export const SemanticPathResponseWireSchema = z.object({
  paths: z.array(z.record(z.string(), z.unknown())),
  total_paths: z.number().int().nonnegative(),
  max_depth_searched: z.number().int().nonnegative(),
  query_latency_ms: z.number(),
  timestamp: z.string(),
});

// ----- RAG -----
//
// Backend route: src/api/routes/rag.py
//   GET /v1/rag/graph/{entity} -> response_model=CausalSubgraphResponse
//   GET /v1/rag/causal-path    -> response_model=CausalPathResponse
//   GET /v1/rag/entities       -> response_model=ExtractedEntitiesResponse
//   GET /v1/rag/health         -> response_model=HealthResponse
// DEFERRED: GET /v1/rag/stats has NO response_model (returns Dict[str, Any]),
// so it is not backend-anchored and is left unwired.

/** Faithful mirror of `GraphNode` (rag.py GraphNode). */
export const RAGGraphNodeWireSchema = z.object({
  id: z.string(),
  label: z.string(),
  type: z.string(),
  properties: z.record(z.string(), z.unknown()),
});

/** Faithful mirror of `GraphEdge` (rag.py GraphEdge). */
export const RAGGraphEdgeWireSchema = z.object({
  source: z.string(),
  target: z.string(),
  relationship: z.string(),
  weight: z.number(),
  properties: z.record(z.string(), z.unknown()),
});

/** Faithful mirror of `CausalSubgraphResponse` (rag.py CausalSubgraphResponse). */
export const CausalSubgraphResponseWireSchema = z.object({
  entity: z.string(),
  nodes: z.array(RAGGraphNodeWireSchema),
  edges: z.array(RAGGraphEdgeWireSchema),
  depth: z.number().int().nonnegative(),
  node_count: z.number().int().nonnegative(),
  edge_count: z.number().int().nonnegative(),
  query_time_ms: z.number(),
});

/** Faithful mirror of `CausalPathResponse` (rag.py CausalPathResponse). */
export const CausalPathResponseWireSchema = z.object({
  source: z.string(),
  target: z.string(),
  paths: z.array(z.array(z.string())),
  shortest_path_length: z.number().int(),
  total_paths: z.number().int().nonnegative(),
  query_time_ms: z.number(),
});

/** Faithful mirror of `ExtractedEntitiesResponse` (rag.py ExtractedEntitiesResponse). */
export const ExtractedEntitiesResponseWireSchema = z.object({
  brands: z.array(z.string()),
  regions: z.array(z.string()),
  kpis: z.array(z.string()),
  agents: z.array(z.string()),
  journey_stages: z.array(z.string()),
  time_references: z.array(z.string()),
  hcp_segments: z.array(z.string()),
});

/** Faithful mirror of `BackendHealthStatus` (rag.py BackendHealthStatus). */
export const RAGBackendHealthStatusWireSchema = z.object({
  status: z.string(),
  latency_ms: z.number(),
  last_check: z.string(),
  consecutive_failures: z.number().int().nonnegative(),
  circuit_breaker_state: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
});

/** Faithful mirror of `HealthResponse` (rag.py HealthResponse). `status` is a plain backend `str`, not an enum. */
export const RAGHealthResponseWireSchema = z.object({
  status: z.string(),
  timestamp: z.string(),
  backends: z.record(z.string(), RAGBackendHealthStatusWireSchema),
  monitoring_enabled: z.boolean(),
});

// ----- HEALTH SCORE -----
//
// Backend route: src/api/routes/health_score.py — every GET read is anchored
// with a response_model:
//   GET /health-score/check|quick|full -> HealthScoreResponse
//   GET /health-score/components        -> ComponentHealthResponse
//   GET /health-score/models            -> ModelHealthResponse
//   GET /health-score/pipelines         -> PipelineHealthResponse
//   GET /health-score/agents            -> AgentHealthResponse
//   GET /health-score/history           -> HealthHistoryResponse
//   GET /health-score/status            -> HealthServiceStatus
// `data_provenance` is a backend-optional field (default "measured") absent
// from the FE types; included as optional so a present value validates.

/** Faithful mirror of `ComponentHealth` (health_score.py ComponentHealth). */
export const ComponentHealthWireSchema = z.object({
  component_name: z.string(),
  status: z.string(),
  latency_ms: z.number().int().nullable().optional(),
  last_check: z.string(),
  error_message: z.string().nullable().optional(),
  details: z.record(z.string(), z.unknown()).nullable().optional(),
});

/** Faithful mirror of `ModelHealth` (health_score.py ModelHealth). */
export const HealthScoreModelHealthWireSchema = z.object({
  model_id: z.string(),
  model_name: z.string(),
  accuracy: z.number().nullable().optional(),
  precision: z.number().nullable().optional(),
  recall: z.number().nullable().optional(),
  f1_score: z.number().nullable().optional(),
  auc_roc: z.number().nullable().optional(),
  prediction_latency_p50_ms: z.number().int().nullable().optional(),
  prediction_latency_p99_ms: z.number().int().nullable().optional(),
  // null = UNMEASURED (no ml_performance_metrics source), not a real zero.
  predictions_last_24h: z.number().int().nonnegative().nullable(),
  error_rate: z.number().nullable(),
  status: z.string(),
});

/** Faithful mirror of `PipelineHealth` (health_score.py PipelineHealth). */
export const PipelineHealthWireSchema = z.object({
  pipeline_name: z.string(),
  last_run: z.string(),
  last_success: z.string(),
  rows_processed: z.number().int().nonnegative(),
  freshness_hours: z.number(),
  status: z.string(),
});

/** Faithful mirror of `AgentHealth` (health_score.py AgentHealth). */
export const AgentHealthWireSchema = z.object({
  agent_name: z.string(),
  tier: z.number().int(),
  available: z.boolean(),
  // null = UNMEASURED (no recent telemetry; provenance "partial"), not a zero.
  avg_latency_ms: z.number().int().nullable(),
  success_rate: z.number().nullable(),
  last_invocation: z.string().nullable().optional(),
  invocations_24h: z.number().int().nonnegative(),
});

/** Faithful mirror of `HealthScoreResponse` (health_score.py HealthScoreResponse). */
export const HealthScoreResponseWireSchema = z.object({
  check_id: z.string(),
  check_scope: z.string(),
  overall_health_score: z.number(),
  health_grade: z.string(),
  // Per-dimension scores are Optional[float] on the backend (HealthScoreResult:
  // "None if unmeasured" — unmeasured dimensions must never be fabricated).
  // The widget renders null as "Not measured in this check".
  component_health_score: z.number().nullable(),
  model_health_score: z.number().nullable(),
  pipeline_health_score: z.number().nullable(),
  agent_health_score: z.number().nullable(),
  component_statuses: z.array(ComponentHealthWireSchema).nullable().optional(),
  model_metrics: z.array(HealthScoreModelHealthWireSchema).nullable().optional(),
  pipeline_statuses: z.array(PipelineHealthWireSchema).nullable().optional(),
  agent_statuses: z.array(AgentHealthWireSchema).nullable().optional(),
  critical_issues: z.array(z.string()),
  warnings: z.array(z.string()),
  recommendations: z.array(z.string()),
  health_summary: z.string(),
  check_latency_ms: z.number().int(),
  timestamp: z.string(),
});

/** Faithful mirror of `ComponentHealthResponse` (health_score.py). */
export const ComponentHealthResponseWireSchema = z.object({
  component_health_score: z.number(),
  total_components: z.number().int().nonnegative(),
  healthy_count: z.number().int().nonnegative(),
  degraded_count: z.number().int().nonnegative(),
  unhealthy_count: z.number().int().nonnegative(),
  components: z.array(ComponentHealthWireSchema),
  check_latency_ms: z.number().int(),
  data_provenance: z.string().optional(),
});

/** Faithful mirror of `ModelHealthResponse` (health_score.py — health-score domain). */
export const HealthScoreModelHealthResponseWireSchema = z.object({
  model_health_score: z.number(),
  total_models: z.number().int().nonnegative(),
  healthy_count: z.number().int().nonnegative(),
  degraded_count: z.number().int().nonnegative(),
  unhealthy_count: z.number().int().nonnegative(),
  models: z.array(HealthScoreModelHealthWireSchema),
  check_latency_ms: z.number().int(),
  data_provenance: z.string().optional(),
});

/** Faithful mirror of `PipelineHealthResponse` (health_score.py). */
export const PipelineHealthResponseWireSchema = z.object({
  pipeline_health_score: z.number(),
  total_pipelines: z.number().int().nonnegative(),
  healthy_count: z.number().int().nonnegative(),
  stale_count: z.number().int().nonnegative(),
  failed_count: z.number().int().nonnegative(),
  pipelines: z.array(PipelineHealthWireSchema),
  check_latency_ms: z.number().int(),
  data_provenance: z.string().optional(),
});

/** Faithful mirror of `AgentHealthResponse` (health_score.py). */
export const AgentHealthResponseWireSchema = z.object({
  agent_health_score: z.number(),
  total_agents: z.number().int().nonnegative(),
  available_count: z.number().int().nonnegative(),
  unavailable_count: z.number().int().nonnegative(),
  agents: z.array(AgentHealthWireSchema),
  by_tier: z.record(z.string(), z.number().int()),
  check_latency_ms: z.number().int(),
  data_provenance: z.string().optional(),
});

/** Faithful mirror of `HealthHistoryItem` (health_score.py HealthHistoryItem). */
export const HealthHistoryItemWireSchema = z.object({
  check_id: z.string(),
  timestamp: z.string(),
  overall_health_score: z.number(),
  health_grade: z.string(),
  critical_issues_count: z.number().int().nonnegative(),
});

/** Faithful mirror of `HealthHistoryResponse` (health_score.py HealthHistoryResponse). */
export const HealthHistoryResponseWireSchema = z.object({
  total_checks: z.number().int().nonnegative(),
  checks: z.array(HealthHistoryItemWireSchema),
  // null when there is no history (not a fabricated 0.0).
  avg_health_score: z.number().nullable(),
  trend: z.string(),
});

/** Faithful mirror of `HealthServiceStatus` (health_score.py HealthServiceStatus). */
export const HealthServiceStatusWireSchema = z.object({
  status: z.string(),
  agent_available: z.boolean(),
  last_check: z.string().nullable().optional(),
  checks_24h: z.number().int().nonnegative(),
  avg_check_latency_ms: z.number().int(),
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
  'analytics.tier-metrics': TierMetricsResponseSchema,

  // Audit
  'audit.entries': AuditEntriesResponseSchema,
  'audit.entry': AuditEntrySchema,
  'audit.verify': ChainVerificationSchema,
  'audit.summary': WorkflowSummarySchema,
  'audit.recent': RecentWorkflowsResponseSchema,
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
export type TierMetricsResponseValidated = z.infer<typeof TierMetricsResponseSchema>;
export type AuditEntryValidated = z.infer<typeof AuditEntrySchema>;
export type ChainVerificationValidated = z.infer<typeof ChainVerificationSchema>;
export type WorkflowSummaryValidated = z.infer<typeof WorkflowSummarySchema>;
export type RecentWorkflowValidated = z.infer<typeof RecentWorkflowSchema>;

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
