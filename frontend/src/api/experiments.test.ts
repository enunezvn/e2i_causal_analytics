/**
 * Experiments API Client Tests
 * ============================
 *
 * Focus: the request-side wire contract for `getSegmentResults`.
 *
 * The backend route `GET /experiments/{id}/results/segments` reads
 * `segments: List[str] = Query(...)` (src/api/routes/experiments.py
 * get_segment_results), which only parses the REPEATED-key form
 * (`?segments=region&segments=specialty`). axios v1's default array
 * serialization emits `?segments[]=region&segments[]=specialty`, which FastAPI
 * silently ignores — dropping the requested segments. The client must opt into
 * repeated-key serialization.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import { getSegmentResults } from './experiments';
import { server } from '@/mocks/server';
import { env } from '@/config/env';

describe('Experiments API Client - getSegmentResults segments serialization', () => {
  it('serializes segments as REPEATED keys (segments=region&segments=specialty)', async () => {
    let capturedUrl = '';
    server.use(
      http.get(
        `${env.apiUrl}/experiments/:id/results/segments`,
        ({ request }) => {
          capturedUrl = request.url;
          return HttpResponse.json({
            experiment_id: 'exp_1',
            segments_analyzed: ['region', 'specialty'],
            segment_results: {},
          });
        }
      )
    );

    await getSegmentResults('exp_1', ['region', 'specialty']);

    const query = new URL(capturedUrl).search;
    // FastAPI List[str] requires repeated keys; bracketed `segments[]=` is dropped.
    const params = new URL(capturedUrl).searchParams.getAll('segments');
    expect(params).toEqual(['region', 'specialty']);
    expect(query).not.toContain('segments%5B%5D'); // no bracketed `segments[]=`
    expect(query).not.toContain('segments[]');
  });
});
