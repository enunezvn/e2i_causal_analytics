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
        total_kpis: 46,
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
        total_kpis: 46,
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
        total_kpis: 46,
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
        { code: 'invalid_type' as const, expected: 'string', received: 'undefined', path: ['label'], message: 'Required' },
      ];

      const error = new ApiValidationError(
        'Validation failed',
        issues,
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
