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
import {
  listScenarios,
  getResourceHealth,
  runOptimizationAndWait,
} from './resources';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';
import type { RunOptimizationRequest } from '@/types/resources';

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

  describe('runOptimizationAndWait - failed-run error message', () => {
    const req = {
      query: 'Optimize budget allocation to maximize roi',
      resource_type: 'budget',
      allocation_targets: [],
      objective: 'maximize_roi',
      run_scenarios: false,
      scenario_count: 3,
    } as unknown as RunOptimizationRequest;

    it('keeps the real failure cause but drops benign SYNTHETIC DATA provenance', async () => {
      // Initial POST is non-terminal -> the poll loop runs; the polled GET fails
      // carrying BOTH a benign provenance warning and the real solver cause.
      server.use(
        http.post(`${env.apiUrl}/resources/optimize`, () =>
          HttpResponse.json({
            optimization_id: 'opt-fail-1',
            status: 'formulating',
            warnings: [],
          })
        ),
        http.get(`${env.apiUrl}/resources/opt-fail-1`, () =>
          HttpResponse.json({
            optimization_id: 'opt-fail-1',
            status: 'failed',
            warnings: [
              'SYNTHETIC DATA: no real per-entity budget source is wired; dollar values are illustrative.',
              'Solver returned: infeasible',
            ],
          })
        )
      );

      let message = '';
      try {
        await runOptimizationAndWait(req, 1, 2000);
      } catch (e) {
        message = (e as Error).message;
      }

      // The honest failure cause is surfaced...
      expect(message).toContain('Solver returned: infeasible');
      // ...but the benign synthetic-data disclosure is NOT dumped into the red error.
      expect(message).not.toContain('SYNTHETIC DATA');
    });

    it('falls back to a generic message when the only warnings are provenance', async () => {
      server.use(
        http.post(`${env.apiUrl}/resources/optimize`, () =>
          HttpResponse.json({
            optimization_id: 'opt-fail-2',
            status: 'formulating',
            warnings: [],
          })
        ),
        http.get(`${env.apiUrl}/resources/opt-fail-2`, () =>
          HttpResponse.json({
            optimization_id: 'opt-fail-2',
            status: 'failed',
            warnings: [
              'SYNTHETIC DATA: notional budget; dollar values are illustrative.',
            ],
          })
        )
      );

      let message = '';
      try {
        await runOptimizationAndWait(req, 1, 2000);
      } catch (e) {
        message = (e as Error).message;
      }

      expect(message).toContain('Optimization failed');
      expect(message).not.toContain('SYNTHETIC DATA');
    });
  });
});
