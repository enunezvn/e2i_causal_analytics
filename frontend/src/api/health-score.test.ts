/**
 * Health Score API Client — response validation (C31 / disputed sweep)
 * ====================================================================
 *
 * Every GET read in src/api/routes/health_score.py is anchored with a
 * response_model, so each opts into runtime Zod validation. A malformed payload
 * must throw `ApiValidationError`; a faithful one must pass through.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import {
  fullHealthCheck,
  getComponentHealth,
  getModelHealth,
  getPipelineHealth,
  getAgentHealth,
  getHealthHistory,
  getHealthServiceStatus,
} from './health-score';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

const ts = () => new Date().toISOString();

describe('Health Score API Client - response validation (C31)', () => {
  describe('fullHealthCheck', () => {
    it('passes a faithful /health-score/full payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/full`, () =>
          HttpResponse.json({
            check_id: 'hs_1',
            check_scope: 'full',
            overall_health_score: 85,
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
            warnings: [],
            recommendations: [],
            health_summary: 'ok',
            check_latency_ms: 1200,
            timestamp: ts(),
          })
        )
      );

      const result = await fullHealthCheck();
      expect(result.health_grade).toBe('B');
    });

    it('throws ApiValidationError when overall_health_score has the wrong type', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/full`, () =>
          HttpResponse.json({ check_id: 'hs_1', overall_health_score: 'high' })
        )
      );

      await expect(fullHealthCheck()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });

  describe('getComponentHealth', () => {
    it('passes a faithful payload (with data_provenance) through', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/components`, () =>
          HttpResponse.json({
            component_health_score: 0.9,
            total_components: 1,
            healthy_count: 1,
            degraded_count: 0,
            unhealthy_count: 0,
            components: [
              { component_name: 'db', status: 'healthy', last_check: ts() },
            ],
            check_latency_ms: 100,
            data_provenance: 'measured',
          })
        )
      );

      const result = await getComponentHealth();
      expect(result.total_components).toBe(1);
    });

    it('throws ApiValidationError on a malformed payload', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/components`, () =>
          HttpResponse.json({ component_health_score: 0.9, components: 'none' })
        )
      );

      await expect(getComponentHealth()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('getModelHealth', () => {
    it('throws ApiValidationError when models is not an array', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/models`, () =>
          HttpResponse.json({
            model_health_score: 0.8,
            total_models: 1,
            healthy_count: 1,
            degraded_count: 0,
            unhealthy_count: 0,
            models: 'one',
            check_latency_ms: 1,
          })
        )
      );

      await expect(getModelHealth()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });

  describe('getPipelineHealth', () => {
    it('throws ApiValidationError on a malformed payload', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/pipelines`, () =>
          HttpResponse.json({ pipeline_health_score: 'good' })
        )
      );

      await expect(getPipelineHealth()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('getAgentHealth', () => {
    it('passes a faithful payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/agents`, () =>
          HttpResponse.json({
            agent_health_score: 0.95,
            total_agents: 1,
            available_count: 1,
            unavailable_count: 0,
            agents: [
              {
                agent_name: 'a',
                tier: 2,
                available: true,
                avg_latency_ms: 10,
                success_rate: 1,
                invocations_24h: 1,
              },
            ],
            by_tier: { '2': 1 },
            check_latency_ms: 50,
          })
        )
      );

      const result = await getAgentHealth();
      expect(result.total_agents).toBe(1);
    });
  });

  describe('getHealthHistory', () => {
    it('throws ApiValidationError when checks is not an array', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/history`, () =>
          HttpResponse.json({
            total_checks: 1,
            checks: 'nope',
            avg_health_score: 85,
            trend: 'stable',
          })
        )
      );

      await expect(getHealthHistory()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('getHealthServiceStatus', () => {
    it('passes a faithful payload (null last_check) through', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/status`, () =>
          HttpResponse.json({
            status: 'healthy',
            agent_available: true,
            last_check: null,
            checks_24h: 0,
            avg_check_latency_ms: 0,
          })
        )
      );

      const result = await getHealthServiceStatus();
      expect(result.agent_available).toBe(true);
    });

    it('throws ApiValidationError when agent_available has the wrong type', async () => {
      server.use(
        http.get(`${env.apiUrl}/health-score/status`, () =>
          HttpResponse.json({
            status: 'healthy',
            agent_available: 1,
            checks_24h: 0,
            avg_check_latency_ms: 0,
          })
        )
      );

      await expect(getHealthServiceStatus()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });
});
