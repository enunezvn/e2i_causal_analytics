/**
 * Memory API Client — response validation (C31 / disputed sweep)
 * ==============================================================
 *
 * Backend-anchored GET reads opt into runtime Zod validation:
 *   GET /memory/episodic       -> response_model=List[EpisodicMemoryResponse]
 *   GET /memory/episodic/{id}  -> response_model=EpisodicMemoryResponse
 *   GET /memory/semantic/paths -> response_model=SemanticPathResponse
 *
 * GET /memory/stats is intentionally NOT wired (no backend response_model →
 * unmodeled Dict[str, Any] passthrough), so it is not exercised here.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import {
  getEpisodicMemories,
  getEpisodicMemory,
  querySemanticPaths,
} from './memory';
import { server } from '@/mocks/server';
import { env } from '@/config/env';
import { ApiValidationError } from '@/lib/api-client';

describe('Memory API Client - response validation (C31)', () => {
  describe('getEpisodicMemories', () => {
    it('passes a faithful list payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/memory/episodic`, () =>
          HttpResponse.json([
            {
              id: 'mem_1',
              content: 'x',
              event_type: 'interaction',
              session_id: null,
              created_at: new Date().toISOString(),
              metadata: {},
            },
          ])
        )
      );

      const result = await getEpisodicMemories();
      expect(Array.isArray(result)).toBe(true);
      expect(result[0].id).toBe('mem_1');
    });

    it('throws ApiValidationError when an item omits required metadata', async () => {
      server.use(
        http.get(`${env.apiUrl}/memory/episodic`, () =>
          HttpResponse.json([
            { id: 'mem_1', content: 'x', event_type: 'interaction', created_at: 'now' },
          ])
        )
      );

      await expect(getEpisodicMemories()).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('getEpisodicMemory', () => {
    it('throws ApiValidationError on a malformed single memory', async () => {
      server.use(
        http.get(`${env.apiUrl}/memory/episodic/:id`, () =>
          HttpResponse.json({ id: 'mem_1' }) // missing content/event_type/created_at/metadata
        )
      );

      await expect(getEpisodicMemory('mem_1')).rejects.toBeInstanceOf(
        ApiValidationError
      );
    });
  });

  describe('querySemanticPaths', () => {
    it('passes a faithful semantic-paths payload through', async () => {
      server.use(
        http.get(`${env.apiUrl}/memory/semantic/paths`, () =>
          HttpResponse.json({
            paths: [{ nodes: ['a', 'b'] }],
            total_paths: 1,
            max_depth_searched: 3,
            query_latency_ms: 12.5,
            timestamp: new Date().toISOString(),
          })
        )
      );

      const result = await querySemanticPaths({ kpi_name: 'TRx' });
      expect(result.total_paths).toBe(1);
    });

    it('throws ApiValidationError when total_paths has the wrong type', async () => {
      server.use(
        http.get(`${env.apiUrl}/memory/semantic/paths`, () =>
          HttpResponse.json({
            paths: [],
            total_paths: 'many', // wrong type
            max_depth_searched: 3,
            query_latency_ms: 1,
            timestamp: new Date().toISOString(),
          })
        )
      );

      await expect(
        querySemanticPaths({ kpi_name: 'TRx' })
      ).rejects.toBeInstanceOf(ApiValidationError);
    });
  });
});
