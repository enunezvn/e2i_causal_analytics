/**
 * Causal API Client Tests
 * =======================
 *
 * Focused tests for the opt-in response validation wired into the causal
 * client (C31). Causal endpoints have no default MSW handlers, so each test
 * registers its own handler via `server.use`.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { getCausalHealth, listEstimators } from './causal';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Causal API Client', () => {
  describe('response validation (C31)', () => {
    it('getCausalHealth passes a valid response through (schema wired)', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            libraries_available: { dowhy: true, econml: true, causalml: false },
            estimators_loaded: 12,
            pipeline_orchestrator_ready: true,
            hierarchical_analyzer_ready: true,
            last_analysis: new Date().toISOString(),
            analysis_count_24h: 7,
            average_latency_ms: 1234,
          })
        )
      );

      const result = await getCausalHealth();
      expect(result.status).toBe('healthy');
      expect(result.estimators_loaded).toBe(12);
    });

    it('getCausalHealth throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            // libraries_available must be Record<string, boolean>
            libraries_available: { dowhy: 'yes' },
            estimators_loaded: 1,
            pipeline_orchestrator_ready: true,
            hierarchical_analyzer_ready: true,
            analysis_count_24h: 0,
          })
        )
      );

      await expect(getCausalHealth()).rejects.toBeInstanceOf(ApiValidationError);
    });

    it('listEstimators passes a valid response through (schema wired)', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/estimators`, () =>
          HttpResponse.json({
            estimators: [
              {
                name: 'causal_forest',
                library: 'econml',
                estimator_type: 'CATE',
                description: 'Causal forest estimator',
                best_for: ['heterogeneous effects'],
                parameters: ['n_estimators'],
                supports_confidence_intervals: true,
                supports_heterogeneous_effects: true,
              },
            ],
            total: 1,
            by_library: { econml: ['causal_forest'] },
          })
        )
      );

      const result = await listEstimators();
      expect(result.total).toBe(1);
      expect(result.estimators[0]?.name).toBe('causal_forest');
    });

    it('listEstimators throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/causal/estimators`, () =>
          HttpResponse.json({
            estimators: [{ name: 'broken' }], // missing required estimator fields
            total: 1,
            by_library: {},
          })
        )
      );

      await expect(listEstimators()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });
});
