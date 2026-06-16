/**
 * Digital Twin API Client - Contract Tests
 * ========================================
 *
 * Verifies that getSimulationHistory and compareScenarios call the backend
 * endpoints that now exist in src/api/routes/digital_twin.py:
 *   - GET  /digital-twin/simulations/history
 *   - POST /digital-twin/simulations/compare
 *
 * Previously the backend defined NEITHER route: `history` was shadowed by the
 * dynamic GET /simulations/{simulation_id} (UUID('history') -> 500) and
 * `compare` returned 404. These tests lock the frontend->backend contract by
 * asserting the exact paths/payloads the client sends, mocking the transport.
 *
 * @module api/digital-twin.contract.test
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock the transport so we assert on the URL/payload the client constructs,
// independent of any running server.
vi.mock('@/lib/api-client', () => ({
  get: vi.fn().mockResolvedValue({}),
  post: vi.fn().mockResolvedValue({}),
}));

import { get, post } from '@/lib/api-client';
import { getSimulationHistory, compareScenarios } from './digital-twin';
import type { ScenarioComparisonRequest } from '@/types/digital-twin';
import { InterventionType } from '@/types/digital-twin';

describe('digital-twin API contract', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('getSimulationHistory', () => {
    it('calls GET /digital-twin/simulations/history with no params', async () => {
      await getSimulationHistory();
      expect(get).toHaveBeenCalledTimes(1);
      const url = (get as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toBe('/digital-twin/simulations/history');
    });

    it('appends limit/offset query params when provided', async () => {
      await getSimulationHistory({ limit: 10, offset: 5 });
      const url = (get as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(url).toContain('/digital-twin/simulations/history?');
      expect(url).toContain('limit=10');
      expect(url).toContain('offset=5');
    });
  });

  describe('compareScenarios', () => {
    it('calls POST /digital-twin/simulations/compare with the request body', async () => {
      const request: ScenarioComparisonRequest = {
        base_scenario: {
          intervention_type: InterventionType.EMAIL_CAMPAIGN,
          brand: 'Remibrutinib',
          sample_size: 1000,
          duration_days: 90,
        },
        alternative_scenarios: [
          {
            intervention_type: InterventionType.DIGITAL_ENGAGEMENT,
            brand: 'Remibrutinib',
            sample_size: 1000,
            duration_days: 90,
          },
        ],
      };

      await compareScenarios(request);

      expect(post).toHaveBeenCalledTimes(1);
      const [url, body] = (post as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(url).toBe('/digital-twin/simulations/compare');
      expect(body).toEqual(request);
    });
  });
});
