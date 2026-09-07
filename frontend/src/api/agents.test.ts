/**
 * api/agents — request-shape contract tests.
 *
 * These fetches used to be inline `useQuery` bodies in
 * `pages/AgentOrchestration.tsx` with the query string baked into the endpoint
 * literal (`'/agents/activity?hours=24&limit=50'`). The layered version passes
 * the window as axios `params` instead, so this file pins the WIRE URL — the
 * one thing the move could have changed silently — by observing the real
 * XMLHttpRequest jsdom opens. If a future edit drops `hours`/`limit`, the
 * backend would quietly serve its own defaults and these tests go red.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getAgentStatus,
  getAgentActivity,
  getAgentTierMetrics,
} from './agents';

/** URLs passed to XMLHttpRequest.open, in call order. */
const openedUrls: string[] = [];
const realOpen = XMLHttpRequest.prototype.open;

beforeEach(() => {
  openedUrls.length = 0;
  vi.spyOn(XMLHttpRequest.prototype, 'open').mockImplementation(function (
    this: XMLHttpRequest,
    method: string,
    url: string | URL,
    ...rest: unknown[]
  ) {
    openedUrls.push(String(url));
    // Delegate so axios still gets a well-formed (if unreachable) request and
    // rejects with a network error rather than hanging.
    return (realOpen as unknown as (...a: unknown[]) => void).apply(this, [
      method,
      url,
      ...rest,
    ]);
  } as typeof XMLHttpRequest.prototype.open);
});

afterEach(() => {
  vi.restoreAllMocks();
});

/** Run a fetch we expect to fail (no server); we only care about the URL. */
async function capture(call: () => Promise<unknown>): Promise<string> {
  await expect(call()).rejects.toBeDefined();
  expect(openedUrls.length).toBe(1);
  return openedUrls[0];
}

describe('api/agents request shapes', () => {
  it('getAgentStatus hits /agents/status with no query params', async () => {
    const url = await capture(() => getAgentStatus());
    expect(url).toContain('/agents/status');
    expect(url).not.toContain('?');
  });

  it('getAgentActivity defaults to the page\'s previous hours=24&limit=50', async () => {
    const url = await capture(() => getAgentActivity());
    expect(url).toContain('/agents/activity');
    expect(url).toContain('hours=24');
    expect(url).toContain('limit=50');
  });

  it('getAgentActivity forwards a non-default window', async () => {
    const url = await capture(() => getAgentActivity(6, 10));
    expect(url).toContain('hours=6');
    expect(url).toContain('limit=10');
  });

  it('getAgentTierMetrics defaults to the page\'s previous hours=24', async () => {
    const url = await capture(() => getAgentTierMetrics());
    expect(url).toContain('/analytics/tier-metrics');
    expect(url).toContain('hours=24');
  });
});
