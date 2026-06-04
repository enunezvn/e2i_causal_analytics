/**
 * Graph API Client Tests
 * ======================
 *
 * Focused tests for the opt-in response validation wired into the graph
 * client (C31). The default MSW handlers return contract-valid payloads;
 * malformed-response cases register an override via `server.use`.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { getGraphHealth, getGraphStats } from './graph';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Graph API Client', () => {
  describe('response validation (C31)', () => {
    it('getGraphHealth passes a valid response through (schema wired)', async () => {
      const result = await getGraphHealth();
      expect(result.status).toBeDefined();
      expect(result.graphiti).toBeDefined();
      expect(result.falkordb).toBeDefined();
    });

    it('getGraphHealth throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/graph/health`, () =>
          HttpResponse.json({
            status: 'on-fire', // out of enum
            graphiti: 'connected',
            falkordb: 'connected',
            websocket_connections: 3,
            timestamp: new Date().toISOString(),
          })
        )
      );

      await expect(getGraphHealth()).rejects.toBeInstanceOf(ApiValidationError);
    });

    it('getGraphStats passes a valid response through (schema wired)', async () => {
      const result = await getGraphStats();
      expect(result.total_nodes).toBeDefined();
      expect(result.nodes_by_type).toBeDefined();
    });

    it('getGraphStats throws ApiValidationError on a malformed response', async () => {
      server.use(
        http.get(`${env.apiUrl}/graph/stats`, () =>
          HttpResponse.json({
            total_nodes: 'many', // wrong type
            total_relationships: 0,
            nodes_by_type: {},
            relationships_by_type: {},
            total_episodes: 0,
            total_communities: 0,
            timestamp: new Date().toISOString(),
          })
        )
      );

      await expect(getGraphStats()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });
});
