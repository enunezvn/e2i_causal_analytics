/**
 * RAG API Client — response validation (C31 / disputed sweep)
 * ===========================================================
 *
 * Backend-anchored GET reads opt into runtime Zod validation:
 *   GET /v1/rag/entities        -> response_model=ExtractedEntitiesResponse
 *   GET /v1/rag/graph/{entity}  -> response_model=CausalSubgraphResponse
 *   GET /v1/rag/causal-path     -> response_model=CausalPathResponse
 *   GET /v1/rag/health          -> response_model=HealthResponse
 *
 * GET /v1/rag/stats is intentionally NOT wired (no backend response_model →
 * unmodeled Dict[str, Any]); its request-side contract is covered by rag.test.ts.
 *
 * NOTE: this file uses the REAL api-client (via MSW) so validation actually
 * runs — it is separate from rag.test.ts, which mocks the client to assert the
 * request contract.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import {
  extractEntities,
  getCausalSubgraph,
  getCausalPaths,
  getRAGHealth,
} from './rag';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('RAG API Client - response validation (C31)', () => {
  describe('extractEntities', () => {
    it('passes a faithful /entities payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/entities`, () =>
          HttpResponse.json({
            brands: ['Kisqali'],
            regions: ['west'],
            kpis: ['trx'],
            agents: [],
            journey_stages: [],
            time_references: ['Q3'],
            hcp_segments: [],
          })
        )
      );

      const result = await extractEntities({ query: 'Kisqali in west Q3' });
      expect(result.brands).toEqual(['Kisqali']);
    });

    it('throws ApiValidationError when an entity array is the wrong type', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/entities`, () =>
          HttpResponse.json({ brands: 'Kisqali' }) // should be string[]
        )
      );

      await expect(
        extractEntities({ query: 'x' })
      ).rejects.toBeInstanceOf(ApiValidationError);
    });
  });

  describe('getCausalSubgraph', () => {
    it('passes a faithful subgraph payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/graph/:entity`, () =>
          HttpResponse.json({
            entity: 'kisqali',
            nodes: [{ id: 'n1', label: 'K', type: 'brand', properties: {} }],
            edges: [],
            depth: 2,
            node_count: 1,
            edge_count: 0,
            query_time_ms: 5,
          })
        )
      );

      const result = await getCausalSubgraph('kisqali', 2);
      expect(result.node_count).toBe(1);
    });

    it('throws ApiValidationError on a malformed subgraph payload', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/graph/:entity`, () =>
          HttpResponse.json({ entity: 'kisqali', nodes: 'none' })
        )
      );

      await expect(getCausalSubgraph('kisqali')).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('getCausalPaths', () => {
    it('throws ApiValidationError when paths is not a nested string array', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/causal-path`, () =>
          HttpResponse.json({
            source: 'a',
            target: 'b',
            paths: ['a', 'b'], // should be string[][]
            shortest_path_length: 1,
            total_paths: 1,
            query_time_ms: 1,
          })
        )
      );

      await expect(getCausalPaths('a', 'b')).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('getRAGHealth', () => {
    it('passes a faithful /v1/rag/health payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            backends: {
              vector: {
                status: 'healthy',
                latency_ms: 10,
                last_check: new Date().toISOString(),
                consecutive_failures: 0,
              },
            },
            monitoring_enabled: true,
          })
        )
      );

      const result = await getRAGHealth();
      expect(result.status).toBe('healthy');
    });

    it('throws ApiValidationError when monitoring_enabled is missing', async () => {
      server.use(
        http.get(`${env.apiUrl}/v1/rag/health`, () =>
          HttpResponse.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            backends: {},
          })
        )
      );

      await expect(getRAGHealth()).rejects.toBeInstanceOf(ApiValidationError);
    });
  });
});
