/**
 * Resources API Client — response validation (C31 / disputed sweep)
 * =================================================================
 *
 * Backend-anchored GET reads (`GET /resources/scenarios` ->
 * response_model=ScenarioListResponse, `GET /resources/health` ->
 * response_model=ResourceHealthResponse) opt into runtime Zod validation. The
 * large `OptimizationResponse` reads are intentionally NOT validated.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { listScenarios, getResourceHealth } from './resources';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Resources API Client - response validation (C31)', () => {
  describe('listScenarios', () => {
    it('passes a faithful /resources/scenarios payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/resources/scenarios`, () =>
          HttpResponse.json({
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
          })
        )
      );

      const result = await listScenarios();
      expect(result.total_count).toBe(1);
      expect(result.scenarios[0].roi).toBe(2.5);
    });

    it('throws ApiValidationError on a malformed payload', async () => {
      server.use(
        http.get(`${env.apiUrl}/resources/scenarios`, () =>
          HttpResponse.json({ total_count: 1, scenarios: [{ roi: 'high' }] })
        )
      );

      await expect(listScenarios()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });

  describe('getResourceHealth', () => {
    it('passes a faithful /resources/health payload (null last_optimization) through', async () => {
      server.use(
        http.get(`${env.apiUrl}/resources/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            agent_available: true,
            scipy_available: true,
            last_optimization: null,
            optimizations_24h: 2,
          })
        )
      );

      const result = await getResourceHealth();
      expect(result.status).toBe('healthy');
    });

    it('throws ApiValidationError when status is missing', async () => {
      server.use(
        http.get(`${env.apiUrl}/resources/health`, () =>
          HttpResponse.json({
            agent_available: true,
            scipy_available: true,
            optimizations_24h: 0,
          })
        )
      );

      await expect(getResourceHealth()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });
});
