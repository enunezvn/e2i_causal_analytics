/**
 * Segments API Client — response validation (C31 / disputed sweep)
 * ================================================================
 *
 * The backend-anchored GET reads (`GET /segments/policies` ->
 * response_model=PolicyListResponse, `GET /segments/health` ->
 * response_model=SegmentHealthResponse) opt into runtime Zod validation. A
 * malformed payload must throw `ApiValidationError`; a faithful payload must
 * pass through. The large `SegmentAnalysisResponse` reads are intentionally
 * NOT validated (heavy/volatile shape) and are not covered here.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { listPolicies, getSegmentHealth } from './segments';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Segments API Client - response validation (C31)', () => {
  describe('listPolicies', () => {
    it('passes a faithful /segments/policies payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/segments/policies`, () =>
          HttpResponse.json({
            total_count: 1,
            recommendations: [
              {
                segment: 'north_high',
                current_treatment_rate: 0.3,
                recommended_treatment_rate: 0.6,
                expected_incremental_outcome: 1000,
                confidence: 0.8,
              },
            ],
            expected_total_lift: 1000,
          })
        )
      );

      const result = await listPolicies();
      expect(result.total_count).toBe(1);
      expect(result.recommendations[0].segment).toBe('north_high');
    });

    it('throws ApiValidationError on a malformed payload', async () => {
      server.use(
        http.get(`${env.apiUrl}/segments/policies`, () =>
          HttpResponse.json({
            total_count: 'one', // wrong type
            recommendations: 'nope',
          })
        )
      );

      await expect(listPolicies()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });

  describe('getSegmentHealth', () => {
    it('passes a faithful /segments/health payload (null last_analysis) through', async () => {
      server.use(
        http.get(`${env.apiUrl}/segments/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            agent_available: true,
            econml_available: true,
            causalml_available: true,
            last_analysis: null,
            analyses_24h: 0,
          })
        )
      );

      const result = await getSegmentHealth();
      expect(result.status).toBe('healthy');
      expect(result.agent_available).toBe(true);
    });

    it('throws ApiValidationError when agent_available has the wrong type', async () => {
      server.use(
        http.get(`${env.apiUrl}/segments/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            agent_available: 'yes', // wrong type
            econml_available: true,
            causalml_available: true,
            analyses_24h: 0,
          })
        )
      );

      await expect(getSegmentHealth()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });
});
