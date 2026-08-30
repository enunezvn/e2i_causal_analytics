/**
 * Segments API Client — poll-ceiling expiry keeps the durable analysis_id (#1841)
 * =============================================================================
 *
 * `POST /segments/analyze` (async_mode) returns a PENDING stub for a durable
 * server-side record that keeps computing after the client stops polling. When
 * the page's poll ceiling expired, `runSegmentAnalysisAndWait` threw a plain
 * "timed out" Error that dropped the `analysis_id` — the only affordance left
 * was to click Run again, i.e. a SECOND heavy analysis (live 2026-08-30: the
 * duplicate was rejected "compute capacity saturated" while the original
 * completed unseen).
 *
 * The ceiling expiry must therefore be a typed `SegmentAnalysisTimeoutError`
 * that carries the id, and `waitForSegmentAnalysis(id, …)` must re-attach to
 * that id with GET polling only — never another POST.
 */

import { describe, it, expect } from 'vitest';
import { http, HttpResponse } from 'msw';
import {
  runSegmentAnalysisAndWait,
  waitForSegmentAnalysis,
  SegmentAnalysisTimeoutError,
} from './segments';
import { server } from '@/mocks/server';
import { env } from '@/config/env';

const ANALYSIS_ID = 'seg_1841_still_running';

function record(status: string, warnings: string[] = []) {
  return {
    analysis_id: ANALYSIS_ID,
    status,
    cate_by_segment: {},
    high_responders: [],
    mid_responders: [],
    low_responders: [],
    policy_recommendations: [],
    key_insights: [],
    warnings,
    confidence: 0,
  };
}

/** Counts POST /segments/analyze and GET /segments/:id; GET replies follow `statuses` then repeat the last one. */
function installHandlers(statuses: string[]) {
  const counts = { post: 0, get: 0 };
  server.use(
    http.post(`${env.apiUrl}/segments/analyze`, () => {
      counts.post += 1;
      return HttpResponse.json(record('pending'));
    }),
    http.get(`${env.apiUrl}/segments/${ANALYSIS_ID}`, () => {
      const status = statuses[Math.min(counts.get, statuses.length - 1)];
      counts.get += 1;
      return HttpResponse.json(record(status, status === 'failed' ? ['estimator blew up'] : []));
    })
  );
  return counts;
}

describe('runSegmentAnalysisAndWait — ceiling expiry keeps the analysis_id (#1841)', () => {
  it('rejects with SegmentAnalysisTimeoutError carrying the id after exactly one POST', async () => {
    const counts = installHandlers(['pending']);

    const promise = runSegmentAnalysisAndWait(
      { query: 'HTE of copay_support on persistent_180d', brand: 'Fabhalta' },
      0,
      40
    );

    await expect(promise).rejects.toBeInstanceOf(SegmentAnalysisTimeoutError);
    await expect(promise).rejects.toMatchObject({ analysisId: ANALYSIS_ID, maxWaitMs: 40 });
    expect(counts.post).toBe(1);
    expect(counts.get).toBeGreaterThan(0);
  });
});

describe('waitForSegmentAnalysis — re-attaches to a durable record without a new POST', () => {
  it('polls GET on the same id and resolves the completed record', async () => {
    const counts = installHandlers(['analyzing', 'analyzing', 'completed']);

    const result = await waitForSegmentAnalysis(ANALYSIS_ID, 0, 5_000);

    expect(result.status).toBe('completed');
    expect(result.analysis_id).toBe(ANALYSIS_ID);
    expect(counts.get).toBe(3);
    expect(counts.post).toBe(0);
  });

  it('surfaces a failed record as the normal failure error (not a timeout)', async () => {
    const counts = installHandlers(['analyzing', 'failed']);

    const promise = waitForSegmentAnalysis(ANALYSIS_ID, 0, 5_000);

    await expect(promise).rejects.toThrow('Segment analysis failed: estimator blew up');
    await expect(promise).rejects.not.toBeInstanceOf(SegmentAnalysisTimeoutError);
    expect(counts.post).toBe(0);
  });

  it('expires again with the same id when the record is still not terminal', async () => {
    installHandlers(['analyzing']);

    await expect(waitForSegmentAnalysis(ANALYSIS_ID, 0, 40)).rejects.toMatchObject({
      name: 'SegmentAnalysisTimeoutError',
      analysisId: ANALYSIS_ID,
    });
  });
});
