/**
 * API Schemas Test Suite
 * ======================
 *
 * Tests for Zod runtime validation schemas.
 *
 * @module lib/api-schemas.test
 */

import { describe, it, expect } from 'vitest';
import {
  KPIMetadataSchema,
  KPIResultSchema,
  KPIListResponseSchema,
  KPIHealthResponseSchema,
  ChatResponseSchema,
  DriftDetectionResponseSchema,
  AlertSchema,
  GraphNodeSchema,
  CausalEffectSchema,
  validateApiResponse,
  ApiValidationError,
  schemaRegistry,
  getSchema,
  // C31 wire schemas
  KPIResultWireSchema,
  KPIListResponseWireSchema,
  ModelsStatusResponseWireSchema,
  AlertListResponseWireSchema,
  CausalHealthResponseWireSchema,
  GraphHealthResponseWireSchema,
  // Disputed-sweep wire schemas (segments/resources/memory/rag/health-score)
  ApiErrorResponseSchema,
  PolicyListResponseWireSchema,
  SegmentHealthResponseWireSchema,
  ScenarioListResponseWireSchema,
  ResourceHealthResponseWireSchema,
  EpisodicMemoryListResponseWireSchema,
  EpisodicMemoryResponseWireSchema,
  SemanticPathResponseWireSchema,
  CausalSubgraphResponseWireSchema,
  CausalPathResponseWireSchema,
  ExtractedEntitiesResponseWireSchema,
  RAGHealthResponseWireSchema,
  HealthScoreResponseWireSchema,
  ComponentHealthResponseWireSchema,
  HealthScoreModelHealthResponseWireSchema,
  PipelineHealthResponseWireSchema,
  AgentHealthResponseWireSchema,
  HealthHistoryResponseWireSchema,
  HealthServiceStatusWireSchema,
} from './api-schemas';

// =============================================================================
// KPI SCHEMA TESTS
// =============================================================================

describe('KPI Schemas', () => {
  describe('KPIMetadataSchema', () => {
    it('should validate valid KPI metadata', () => {
      const validData = {
        id: 'WS1-DQ-001',
        name: 'Data Completeness Rate',
        definition: 'Percentage of complete records in the dataset',
        formula: 'complete_records / total_records * 100',
        calculation_type: 'direct',
        workstream: 'ws1_data_quality',
        tables: ['hcp_data', 'patient_data'],
        columns: ['id', 'name', 'status'],
        frequency: 'daily',
        primary_causal_library: 'none',
      };

      const result = KPIMetadataSchema.safeParse(validData);
      expect(result.success).toBe(true);
    });

    it('should reject invalid KPI metadata missing required fields', () => {
      const invalidData = {
        id: 'WS1-DQ-001',
        name: 'Data Completeness Rate',
        // missing: definition, formula, etc.
      };

      const result = KPIMetadataSchema.safeParse(invalidData);
      expect(result.success).toBe(false);
    });

    it('should accept optional fields', () => {
      const dataWithOptionals = {
        id: 'WS1-DQ-001',
        name: 'Data Completeness Rate',
        definition: 'Percentage of complete records',
        formula: 'complete / total * 100',
        calculation_type: 'direct',
        workstream: 'ws1_data_quality',
        tables: ['hcp_data'],
        columns: ['id'],
        frequency: 'daily',
        primary_causal_library: 'dowhy',
        threshold: { target: 95, warning: 85, critical: 70 },
        unit: '%',
        view: 'kpi_view',
        brand: 'remibrutinib',
        note: 'Important KPI',
      };

      const result = KPIMetadataSchema.safeParse(dataWithOptionals);
      expect(result.success).toBe(true);
    });
  });

  describe('KPIResultSchema', () => {
    it('should validate valid KPI result', () => {
      const validResult = {
        kpi_id: 'WS1-DQ-001',
        value: 92.5,
        status: 'good',
        calculated_at: '2024-01-15T10:30:00Z',
        cached: true,
        metadata: {},
      };

      const result = KPIResultSchema.safeParse(validResult);
      expect(result.success).toBe(true);
    });

    it('should accept optional statistical fields', () => {
      const resultWithStats = {
        kpi_id: 'WS1-DQ-001',
        value: 92.5,
        status: 'good',
        calculated_at: '2024-01-15T10:30:00Z',
        cached: false,
        metadata: { source: 'live' },
        confidence_interval: [88.0, 97.0],
        p_value: 0.03,
        effect_size: 0.45,
        causal_library_used: 'dowhy',
      };

      const result = KPIResultSchema.safeParse(resultWithStats);
      expect(result.success).toBe(true);
    });

    it('preserves data_source on both KPI result schemas (no Zod strip)', () => {
      // Regression: getKPIValue() validates with KPIResultWireSchema and the
      // batch path with KPIResultSchema. Zod strips unknown keys, so the
      // synthetic-mode provenance MUST be declared on BOTH or the FE silently
      // loses the "synthetic" label and renders synthetic figures as real.
      const synthetic = {
        kpi_id: 'WS1-MP-001',
        value: 0.7704,
        status: 'good',
        calculated_at: '2026-06-13T10:30:00Z',
        cached: false,
        data_source: 'synthetic',
        metadata: {},
      };
      const wire = KPIResultWireSchema.safeParse(synthetic);
      const batch = KPIResultSchema.safeParse(synthetic);
      expect(wire.success).toBe(true);
      expect(batch.success).toBe(true);
      expect(wire.success && wire.data.data_source).toBe('synthetic');
      expect(batch.success && batch.data.data_source).toBe('synthetic');
    });
  });

  describe('KPIListResponseSchema', () => {
    it('should validate valid list response', () => {
      const validResponse = {
        kpis: [
          {
            id: 'WS1-DQ-001',
            name: 'Data Completeness',
            definition: 'Complete records percentage',
            formula: 'complete/total*100',
            calculation_type: 'direct',
            workstream: 'ws1_data_quality',
            tables: ['data'],
            columns: ['id'],
            frequency: 'daily',
            primary_causal_library: 'none',
          },
        ],
        total: 1,
      };

      const result = KPIListResponseSchema.safeParse(validResponse);
      expect(result.success).toBe(true);
    });
  });

  describe('KPIHealthResponseSchema', () => {
    it('should validate healthy status', () => {
      const healthyResponse = {
        status: 'healthy',
        registry_loaded: true,
        total_kpis: 44,
        cache_enabled: true,
        cache_size: 120,
        database_connected: true,
        workstreams_available: ['ws1_data_quality', 'ws2_triggers'],
      };

      const result = KPIHealthResponseSchema.safeParse(healthyResponse);
      expect(result.success).toBe(true);
    });

    it('should validate degraded status with error', () => {
      const degradedResponse = {
        status: 'degraded',
        registry_loaded: true,
        total_kpis: 44,
        cache_enabled: false,
        cache_size: 0,
        database_connected: true,
        workstreams_available: ['ws1_data_quality'],
        error: 'Redis connection failed',
      };

      const result = KPIHealthResponseSchema.safeParse(degradedResponse);
      expect(result.success).toBe(true);
    });

    it('should reject invalid status', () => {
      const invalidStatus = {
        status: 'unknown', // not in enum
        registry_loaded: true,
        total_kpis: 44,
        cache_enabled: true,
        cache_size: 0,
        database_connected: true,
        workstreams_available: [],
      };

      const result = KPIHealthResponseSchema.safeParse(invalidStatus);
      expect(result.success).toBe(false);
    });
  });
});

// =============================================================================
// CHAT SCHEMA TESTS
// =============================================================================

describe('Chat Schemas', () => {
  describe('ChatResponseSchema', () => {
    it('should validate minimal chat response', () => {
      const minimalResponse = {
        success: true,
        session_id: 'sess-123',
        response: 'Hello, how can I help?',
      };

      const result = ChatResponseSchema.safeParse(minimalResponse);
      expect(result.success).toBe(true);
      if (result.success) {
        // Check defaults are applied
        expect(result.data.orchestrator_used).toBe(false);
        expect(result.data.agents_dispatched).toEqual([]);
      }
    });

    it('should validate full chat response with observability', () => {
      const fullResponse = {
        success: true,
        session_id: 'sess-123',
        response: 'Based on causal analysis...',
        conversation_title: 'Sales Analysis',
        agent_name: 'CausalImpactAgent',
        orchestrator_used: true,
        agents_dispatched: ['causal_impact', 'gap_analyzer'],
        routed_agent: 'causal_impact',
        response_confidence: 0.92,
        execution_time_ms: 1250.5,
        intent: 'causal_analysis',
        intent_confidence: 0.95,
      };

      const result = ChatResponseSchema.safeParse(fullResponse);
      expect(result.success).toBe(true);
    });

    it('should handle error response', () => {
      const errorResponse = {
        success: false,
        session_id: 'sess-123',
        response: '',
        error: 'Model timeout after 30 seconds',
        execution_time_ms: 30000,
      };

      const result = ChatResponseSchema.safeParse(errorResponse);
      expect(result.success).toBe(true);
    });
  });
});

// =============================================================================
// MONITORING SCHEMA TESTS
// =============================================================================

describe('Monitoring Schemas', () => {
  describe('DriftDetectionResponseSchema', () => {
    it('should validate drift detection response', () => {
      const driftResponse = {
        run_id: 'drift-run-001',
        model_id: 'churn-model-v2',
        timestamp: '2024-01-15T10:00:00Z',
        overall_drift_detected: true,
        drift_score: 0.35,
        metrics: [
          {
            feature: 'age',
            drift_score: 0.15,
            threshold: 0.1,
            is_drifted: true,
            drift_type: 'covariate',
          },
          {
            feature: 'income',
            drift_score: 0.05,
            threshold: 0.1,
            is_drifted: false,
          },
        ],
        recommendation: 'Retrain model with recent data',
      };

      const result = DriftDetectionResponseSchema.safeParse(driftResponse);
      expect(result.success).toBe(true);
    });
  });

  describe('AlertSchema', () => {
    it('should validate alert', () => {
      const alert = {
        id: 'alert-001',
        type: 'drift_detected',
        severity: 'high',
        message: 'Model drift detected in production',
        source: 'drift_monitor',
        timestamp: '2024-01-15T10:00:00Z',
        acknowledged: false,
      };

      const result = AlertSchema.safeParse(alert);
      expect(result.success).toBe(true);
    });

    it('should reject invalid severity', () => {
      const invalidAlert = {
        id: 'alert-001',
        type: 'drift_detected',
        severity: 'extreme', // invalid
        message: 'Test',
        source: 'test',
        timestamp: '2024-01-15T10:00:00Z',
        acknowledged: false,
      };

      const result = AlertSchema.safeParse(invalidAlert);
      expect(result.success).toBe(false);
    });
  });
});

// =============================================================================
// GRAPH SCHEMA TESTS
// =============================================================================

describe('Graph Schemas', () => {
  describe('GraphNodeSchema', () => {
    it('should validate graph node', () => {
      const node = {
        id: 'node-001',
        label: 'Treatment A',
        type: 'treatment',
        properties: { dosage: '10mg', frequency: 'daily' },
      };

      const result = GraphNodeSchema.safeParse(node);
      expect(result.success).toBe(true);
    });
  });
});

// =============================================================================
// CAUSAL SCHEMA TESTS
// =============================================================================

describe('Causal Schemas', () => {
  describe('CausalEffectSchema', () => {
    it('should validate causal effect', () => {
      const effect = {
        treatment: 'email_campaign',
        outcome: 'conversion_rate',
        effect: 0.15,
        effect_type: 'ATE',
        confidence_interval: [0.10, 0.20],
        p_value: 0.01,
        sample_size: 5000,
        method: 'doubly_robust',
      };

      const result = CausalEffectSchema.safeParse(effect);
      expect(result.success).toBe(true);
    });
  });
});

// =============================================================================
// VALIDATION UTILITY TESTS
// =============================================================================

describe('Validation Utilities', () => {
  describe('validateApiResponse', () => {
    it('should return validated data on success', () => {
      const validData = {
        id: 'WS1-DQ-001',
        label: 'Test Node',
        type: 'kpi',
      };

      const result = validateApiResponse(GraphNodeSchema, validData, '/test');
      expect(result.id).toBe('WS1-DQ-001');
    });

    it('should throw ApiValidationError on failure', () => {
      const invalidData = { id: 'test' }; // missing required fields

      expect(() =>
        validateApiResponse(GraphNodeSchema, invalidData, '/test/endpoint')
      ).toThrow(ApiValidationError);
    });

    it('should return data without throwing when throwOnError is false', () => {
      const invalidData = { id: 'test' };

      const result = validateApiResponse(
        GraphNodeSchema,
        invalidData,
        '/test',
        { throwOnError: false, logErrors: false }
      );

      // Returns the invalid data as-is
      expect(result.id).toBe('test');
    });
  });

  describe('ApiValidationError', () => {
    it('should have correct properties', () => {
      const issues = [
        { code: 'invalid_type', expected: 'string', received: 'undefined', path: ['label'], message: 'Required' } as const,
      ];

      const error = new ApiValidationError(
        'Validation failed',
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        issues as any,
        '/test/endpoint',
        { id: 'test' }
      );

      expect(error.name).toBe('ApiValidationError');
      expect(error.endpoint).toBe('/test/endpoint');
      expect(error.issues).toHaveLength(1);
      expect(error.formattedMessage).toContain('label');
    });
  });

  describe('schemaRegistry', () => {
    it('should have all expected schemas registered', () => {
      expect(schemaRegistry['kpi.list']).toBeDefined();
      expect(schemaRegistry['kpi.health']).toBeDefined();
      expect(schemaRegistry['chat.response']).toBeDefined();
      expect(schemaRegistry['monitoring.drift']).toBeDefined();
    });

    it('should return schema via getSchema', () => {
      const schema = getSchema('kpi.list');
      expect(schema).toBe(KPIListResponseSchema);
    });
  });
});

// =============================================================================
// WIRE SCHEMA TESTS (C31 — faithful response contracts)
// =============================================================================

describe('Wire Schemas (C31)', () => {
  describe('KPIListResponseWireSchema', () => {
    it('accepts the real /kpis payload (workstream sent as null)', () => {
      const payload = {
        kpis: [
          {
            id: 'WS1-DQ-001',
            name: 'Data Completeness',
            definition: 'Percentage of complete records',
            formula: 'COUNT(complete) / COUNT(*) * 100',
            calculation_type: 'direct',
            workstream: 'ws1_data_quality',
            tables: ['hcp_data'],
            columns: ['*'],
            threshold: { target: 95, warning: 90, critical: 80 },
            unit: '%',
            frequency: 'daily',
            primary_causal_library: 'none',
          },
        ],
        total: 1,
        workstream: null,
        causal_library: null,
      };
      // This is exactly the shape the legacy KPIListResponseSchema REJECTS
      // (workstream: null) — proving the wire schema is the faithful contract.
      const result = KPIListResponseWireSchema.safeParse(payload);
      expect(result.success).toBe(true);
    });

    it('rejects a malformed payload (kpis missing required field)', () => {
      const bad = {
        kpis: [{ id: 'X', name: 'only-id-and-name' }],
        total: 1,
      };
      const result = KPIListResponseWireSchema.safeParse(bad);
      expect(result.success).toBe(false);
    });
  });

  describe('ModelsStatusResponseWireSchema', () => {
    it('accepts the real /models/status payload', () => {
      const payload = {
        total_models: 1,
        healthy_count: 1,
        unhealthy_count: 0,
        models: [
          {
            model_name: 'churn_model',
            status: 'healthy',
            endpoint: 'http://localhost:3000/models/churn',
            last_check: new Date().toISOString(),
          },
        ],
        timestamp: new Date().toISOString(),
      };
      const result = ModelsStatusResponseWireSchema.safeParse(payload);
      expect(result.success).toBe(true);
    });

    it('throws ApiValidationError via validateApiResponse on bad counts', () => {
      const bad = {
        total_models: 'one', // wrong type
        healthy_count: 1,
        unhealthy_count: 0,
        models: [],
        timestamp: new Date().toISOString(),
      };
      expect(() =>
        validateApiResponse(
          ModelsStatusResponseWireSchema,
          bad,
          '/models/status',
          { logErrors: false, throwOnError: true }
        )
      ).toThrow(ApiValidationError);
    });
  });

  describe('AlertListResponseWireSchema', () => {
    it('accepts the real /monitoring/alerts payload', () => {
      const payload = {
        total_count: 1,
        active_count: 1,
        alerts: [
          {
            id: 'alert_001',
            model_version: 'propensity_v2.1.0',
            alert_type: 'drift',
            severity: 'high',
            title: 'High drift detected',
            description: 'Feature shows significant drift',
            status: 'active',
            triggered_at: new Date().toISOString(),
          },
        ],
      };
      const result = AlertListResponseWireSchema.safeParse(payload);
      expect(result.success).toBe(true);
    });

    it('rejects when alerts is not an array', () => {
      const bad = { total_count: 0, active_count: 0, alerts: 'none' };
      const result = AlertListResponseWireSchema.safeParse(bad);
      expect(result.success).toBe(false);
    });
  });

  describe('CausalHealthResponseWireSchema', () => {
    it('accepts a valid causal health payload', () => {
      const payload = {
        status: 'healthy',
        libraries_available: { dowhy: true, econml: true },
        estimators_loaded: 12,
        pipeline_orchestrator_ready: true,
        hierarchical_analyzer_ready: true,
        analysis_count_24h: 5,
      };
      const result = CausalHealthResponseWireSchema.safeParse(payload);
      expect(result.success).toBe(true);
    });

    it('rejects when libraries_available has non-boolean values', () => {
      const bad = {
        status: 'healthy',
        libraries_available: { dowhy: 'yes' },
        estimators_loaded: 1,
        pipeline_orchestrator_ready: true,
        hierarchical_analyzer_ready: true,
        analysis_count_24h: 0,
      };
      const result = CausalHealthResponseWireSchema.safeParse(bad);
      expect(result.success).toBe(false);
    });
  });

  describe('GraphHealthResponseWireSchema', () => {
    it('accepts the real /graph/health payload', () => {
      const payload = {
        status: 'healthy',
        graphiti: 'connected',
        falkordb: 'connected',
        websocket_connections: 3,
        timestamp: new Date().toISOString(),
      };
      const result = GraphHealthResponseWireSchema.safeParse(payload);
      expect(result.success).toBe(true);
    });

    it('rejects an out-of-enum status', () => {
      const bad = {
        status: 'on-fire',
        graphiti: 'connected',
        falkordb: 'connected',
        websocket_connections: 0,
        timestamp: new Date().toISOString(),
      };
      const result = GraphHealthResponseWireSchema.safeParse(bad);
      expect(result.success).toBe(false);
    });
  });
});

// =============================================================================
// COMMON SCHEMA TESTS (finding #5 — unified ApiErrorResponse)
// =============================================================================

describe('ApiErrorResponseSchema (#5 unification)', () => {
  it('accepts an error payload that carries suggested_action', () => {
    const payload = {
      error: 'NotFound',
      message: 'Experiment not found',
      details: { experiment_id: 'exp_x' },
      timestamp: new Date().toISOString(),
      suggested_action: 'Check the experiment ID and retry',
    };
    const result = ApiErrorResponseSchema.safeParse(payload);
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.suggested_action).toBe(
        'Check the experiment ID and retry'
      );
    }
  });

  it('accepts a minimal error payload (suggested_action omitted)', () => {
    const result = ApiErrorResponseSchema.safeParse({
      error: 'ServerError',
      message: 'Boom',
    });
    expect(result.success).toBe(true);
  });
});

// =============================================================================
// DISPUTED-SWEEP WIRE SCHEMA TESTS (#4 — segments/resources/memory/rag/health)
// =============================================================================

describe('Wire Schemas (disputed sweep #4)', () => {
  describe('Segments', () => {
    it('PolicyListResponseWireSchema accepts the real /segments/policies payload', () => {
      const payload = {
        total_count: 1,
        recommendations: [
          {
            segment: 'high_value_north',
            current_treatment_rate: 0.3,
            recommended_treatment_rate: 0.6,
            expected_incremental_outcome: 1200,
            confidence: 0.82,
          },
        ],
        expected_total_lift: 1200,
      };
      expect(PolicyListResponseWireSchema.safeParse(payload).success).toBe(true);
    });

    it('PolicyListResponseWireSchema rejects a recommendation missing required fields', () => {
      const bad = {
        total_count: 1,
        recommendations: [{ segment: 'x' }],
        expected_total_lift: 0,
      };
      expect(PolicyListResponseWireSchema.safeParse(bad).success).toBe(false);
    });

    it('SegmentHealthResponseWireSchema accepts payload with null last_analysis', () => {
      const payload = {
        status: 'healthy',
        agent_available: true,
        econml_available: true,
        causalml_available: true,
        last_analysis: null,
        analyses_24h: 0,
      };
      expect(SegmentHealthResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('SegmentHealthResponseWireSchema rejects non-boolean agent_available', () => {
      const bad = {
        status: 'healthy',
        agent_available: 'yes',
        econml_available: true,
        causalml_available: true,
        analyses_24h: 0,
      };
      expect(SegmentHealthResponseWireSchema.safeParse(bad).success).toBe(false);
    });
  });

  describe('Resources', () => {
    it('ScenarioListResponseWireSchema accepts the real /resources/scenarios payload', () => {
      const payload = {
        total_count: 1,
        scenarios: [
          {
            scenario_name: 'baseline',
            total_allocation: 100000,
            projected_outcome: 250000,
            roi: 2.5,
            constraint_violations: [],
          },
        ],
      };
      expect(ScenarioListResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('ResourceHealthResponseWireSchema accepts payload with null last_optimization', () => {
      const payload = {
        status: 'healthy',
        agent_available: true,
        scipy_available: true,
        last_optimization: null,
        optimizations_24h: 3,
      };
      expect(ResourceHealthResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('ResourceHealthResponseWireSchema rejects missing status', () => {
      const bad = {
        agent_available: true,
        scipy_available: true,
        optimizations_24h: 0,
      };
      expect(ResourceHealthResponseWireSchema.safeParse(bad).success).toBe(
        false
      );
    });
  });

  describe('Memory', () => {
    it('EpisodicMemoryResponseWireSchema accepts a payload with optional nulls', () => {
      const payload = {
        id: 'mem_1',
        content: 'HCP responded positively',
        event_type: 'interaction',
        session_id: null,
        agent_name: 'feedback_learner',
        brand: null,
        region: null,
        created_at: new Date().toISOString(),
        metadata: { source: 'live' },
      };
      expect(EpisodicMemoryResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('EpisodicMemoryListResponseWireSchema accepts an array of memories', () => {
      const payload = [
        {
          id: 'mem_1',
          content: 'x',
          event_type: 'interaction',
          created_at: new Date().toISOString(),
          metadata: {},
        },
      ];
      expect(
        EpisodicMemoryListResponseWireSchema.safeParse(payload).success
      ).toBe(true);
    });

    it('EpisodicMemoryResponseWireSchema rejects when metadata is missing', () => {
      const bad = {
        id: 'mem_1',
        content: 'x',
        event_type: 'interaction',
        created_at: new Date().toISOString(),
      };
      expect(EpisodicMemoryResponseWireSchema.safeParse(bad).success).toBe(
        false
      );
    });

    it('SemanticPathResponseWireSchema accepts a real semantic-paths payload', () => {
      const payload = {
        paths: [{ nodes: ['a', 'b'], confidence: 0.7 }],
        total_paths: 1,
        max_depth_searched: 3,
        query_latency_ms: 12.5,
        timestamp: new Date().toISOString(),
      };
      expect(SemanticPathResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });
  });

  describe('RAG', () => {
    it('ExtractedEntitiesResponseWireSchema accepts the real /entities payload', () => {
      const payload = {
        brands: ['Kisqali'],
        regions: ['west'],
        kpis: ['trx'],
        agents: [],
        journey_stages: [],
        time_references: ['Q3'],
        hcp_segments: [],
      };
      expect(
        ExtractedEntitiesResponseWireSchema.safeParse(payload).success
      ).toBe(true);
    });

    it('CausalSubgraphResponseWireSchema accepts the real subgraph payload', () => {
      const payload = {
        entity: 'kisqali',
        nodes: [
          { id: 'n1', label: 'Kisqali', type: 'brand', properties: {} },
        ],
        edges: [
          {
            source: 'n1',
            target: 'n2',
            relationship: 'affects',
            weight: 1.0,
            properties: {},
          },
        ],
        depth: 2,
        node_count: 1,
        edge_count: 1,
        query_time_ms: 5.5,
      };
      expect(CausalSubgraphResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('CausalPathResponseWireSchema accepts the real causal-path payload', () => {
      const payload = {
        source: 'hcp_engagement',
        target: 'trx',
        paths: [['hcp_engagement', 'awareness', 'trx']],
        shortest_path_length: 2,
        total_paths: 1,
        query_time_ms: 8.2,
      };
      expect(CausalPathResponseWireSchema.safeParse(payload).success).toBe(true);
    });

    it('RAGHealthResponseWireSchema accepts the real /v1/rag/health payload', () => {
      const payload = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        backends: {
          vector: {
            status: 'healthy',
            latency_ms: 12,
            last_check: new Date().toISOString(),
            consecutive_failures: 0,
          },
        },
        monitoring_enabled: true,
      };
      expect(RAGHealthResponseWireSchema.safeParse(payload).success).toBe(true);
    });

    it('RAGHealthResponseWireSchema rejects when monitoring_enabled is missing', () => {
      const bad = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        backends: {},
      };
      expect(RAGHealthResponseWireSchema.safeParse(bad).success).toBe(false);
    });
  });

  describe('Health Score', () => {
    it('HealthScoreResponseWireSchema accepts a full check payload (scope-omitted detail arrays)', () => {
      const payload = {
        check_id: 'hs_1',
        check_scope: 'full',
        overall_health_score: 85.5,
        health_grade: 'B',
        component_health_score: 0.9,
        model_health_score: 0.8,
        pipeline_health_score: 0.85,
        agent_health_score: 0.9,
        component_statuses: null,
        model_metrics: null,
        pipeline_statuses: null,
        agent_statuses: null,
        critical_issues: [],
        warnings: ['Model degraded'],
        recommendations: [],
        health_summary: 'System health is good.',
        check_latency_ms: 1250,
        timestamp: new Date().toISOString(),
      };
      expect(HealthScoreResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('HealthScoreResponseWireSchema accepts null per-dimension scores (backend: Optional, None = unmeasured)', () => {
      // HealthScoreResult (health_score/agent.py) declares every per-dimension
      // score Optional[float] — "None if unmeasured", designed so unmeasured
      // dimensions are never fabricated. The widget renders these as
      // "Not measured in this check"; the wire schema must let them through.
      const payload = {
        check_id: 'hs_2',
        check_scope: 'full',
        overall_health_score: 72.0,
        health_grade: 'C',
        component_health_score: 0.9,
        model_health_score: null,
        pipeline_health_score: null,
        agent_health_score: null,
        component_statuses: null,
        model_metrics: null,
        pipeline_statuses: null,
        agent_statuses: null,
        critical_issues: [],
        warnings: [],
        recommendations: [],
        health_summary: 'Partial check: only components measured.',
        check_latency_ms: 310,
        timestamp: new Date().toISOString(),
      };
      expect(HealthScoreResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('ComponentHealthResponseWireSchema accepts payload with data_provenance', () => {
      const payload = {
        component_health_score: 0.9,
        total_components: 5,
        healthy_count: 5,
        degraded_count: 0,
        unhealthy_count: 0,
        components: [
          {
            component_name: 'database',
            status: 'healthy',
            latency_ms: 12,
            last_check: new Date().toISOString(),
          },
        ],
        check_latency_ms: 100,
        data_provenance: 'measured',
      };
      expect(ComponentHealthResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('HealthScoreModelHealthResponseWireSchema accepts payload with optional model metrics', () => {
      const payload = {
        model_health_score: 0.8,
        total_models: 1,
        healthy_count: 1,
        degraded_count: 0,
        unhealthy_count: 0,
        models: [
          {
            model_id: 'm1',
            model_name: 'churn',
            accuracy: 0.91,
            predictions_last_24h: 100,
            error_rate: 0.01,
            status: 'healthy',
          },
        ],
        check_latency_ms: 200,
      };
      expect(
        HealthScoreModelHealthResponseWireSchema.safeParse(payload).success
      ).toBe(true);
    });

    it('PipelineHealthResponseWireSchema accepts a valid payload', () => {
      const payload = {
        pipeline_health_score: 0.85,
        total_pipelines: 2,
        healthy_count: 2,
        stale_count: 0,
        failed_count: 0,
        pipelines: [
          {
            pipeline_name: 'etl',
            last_run: new Date().toISOString(),
            last_success: new Date().toISOString(),
            rows_processed: 1000,
            freshness_hours: 1.5,
            status: 'healthy',
          },
        ],
        check_latency_ms: 300,
      };
      expect(PipelineHealthResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('AgentHealthResponseWireSchema accepts a valid payload', () => {
      const payload = {
        agent_health_score: 0.95,
        total_agents: 13,
        available_count: 13,
        unavailable_count: 0,
        agents: [
          {
            agent_name: 'causal_impact',
            tier: 2,
            available: true,
            avg_latency_ms: 120,
            success_rate: 0.98,
            invocations_24h: 42,
          },
        ],
        by_tier: { '0': 1, '2': 5 },
        check_latency_ms: 400,
      };
      expect(AgentHealthResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('PipelineHealthResponseWireSchema preserves data_provenance (faithful mirror of backend)', () => {
      const payload = {
        pipeline_health_score: 0.85,
        total_pipelines: 2,
        healthy_count: 2,
        stale_count: 0,
        failed_count: 0,
        pipelines: [
          {
            pipeline_name: 'etl',
            last_run: new Date().toISOString(),
            last_success: new Date().toISOString(),
            rows_processed: 1000,
            freshness_hours: 1.5,
            status: 'healthy',
          },
        ],
        check_latency_ms: 300,
        data_provenance: 'placeholder',
      };
      const parsed = PipelineHealthResponseWireSchema.safeParse(payload);
      expect(parsed.success).toBe(true);
      expect(parsed.success && parsed.data.data_provenance).toBe('placeholder');
    });

    it('AgentHealthResponseWireSchema preserves data_provenance (faithful mirror of backend)', () => {
      const payload = {
        agent_health_score: 0.95,
        total_agents: 13,
        available_count: 13,
        unavailable_count: 0,
        agents: [
          {
            agent_name: 'causal_impact',
            tier: 2,
            available: true,
            avg_latency_ms: 120,
            success_rate: 0.98,
            invocations_24h: 42,
          },
        ],
        by_tier: { '0': 1, '2': 5 },
        check_latency_ms: 400,
        data_provenance: 'placeholder',
      };
      const parsed = AgentHealthResponseWireSchema.safeParse(payload);
      expect(parsed.success).toBe(true);
      expect(parsed.success && parsed.data.data_provenance).toBe('placeholder');
    });

    it('HealthHistoryResponseWireSchema accepts a valid payload', () => {
      const payload = {
        total_checks: 1,
        checks: [
          {
            check_id: 'hs_1',
            timestamp: new Date().toISOString(),
            overall_health_score: 85,
            health_grade: 'B',
            critical_issues_count: 0,
          },
        ],
        avg_health_score: 85,
        trend: 'stable',
      };
      expect(HealthHistoryResponseWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('HealthServiceStatusWireSchema accepts payload with null last_check', () => {
      const payload = {
        status: 'healthy',
        agent_available: true,
        last_check: null,
        checks_24h: 0,
        avg_check_latency_ms: 0,
      };
      expect(HealthServiceStatusWireSchema.safeParse(payload).success).toBe(
        true
      );
    });

    it('HealthServiceStatusWireSchema rejects non-boolean agent_available', () => {
      const bad = {
        status: 'healthy',
        agent_available: 1,
        checks_24h: 0,
        avg_check_latency_ms: 0,
      };
      expect(HealthServiceStatusWireSchema.safeParse(bad).success).toBe(false);
    });
  });
});
