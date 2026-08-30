/**
 * QueryClient + cache-key factory tests
 * =====================================
 *
 * Covers:
 *  - The query-key factories whose result-affecting params were folded in by
 *    the "disputed-findings" cache-key sweep (findings 1-6).
 *  - The tab-visibility handler honesty fix (finding 7): the handler must NOT
 *    fabricate a "cancel in-flight queries on hide" behavior it never had;
 *    background polling is already paused by TanStack's built-in focusManager.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MutationObserver, QueryClient } from '@tanstack/react-query';
import { AxiosError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';

// Control env.isProd / env.isDev deterministically for the visibility tests.
vi.mock('@/config/env', () => ({
  env: { isDev: false, isProd: true, mode: 'test', isTest: true },
}));

import {
  queryKeys,
  queryClient,
  initTabVisibilityListener,
  cleanupTabVisibilityListener,
  isTabCurrentlyVisible,
} from './query-client';

// ===========================================================================
// MUTATION RETRY DEFAULT (#1846)
// ===========================================================================

/**
 * A fresh client carrying exactly the defaults the app ships (`queryClient`
 * is the production singleton; cloning its defaultOptions keeps each test's
 * mutation cache isolated without re-implementing the config here).
 */
function clientWithAppDefaults(): QueryClient {
  return new QueryClient({ defaultOptions: queryClient.getDefaultOptions() });
}

const axiosConfig = { url: '/segments/analyze', method: 'post', headers: {} } as unknown as InternalAxiosRequestConfig;

/** axios 1.x xhr adapter: `request.ontimeout` → code ECONNABORTED, no response. */
function axiosTimeoutError(): AxiosError {
  return new AxiosError('timeout of 30000ms exceeded', AxiosError.ECONNABORTED, axiosConfig);
}

/** axios 1.x xhr adapter: `request.onerror` → code ERR_NETWORK, no response. */
function axiosNetworkError(): AxiosError {
  return new AxiosError('Network Error', AxiosError.ERR_NETWORK, axiosConfig);
}

/** A 5xx that the server produced AFTER it received (and may have acted on) the POST. */
function axiosServerError(status = 500): AxiosError {
  const response = {
    status,
    statusText: 'Internal Server Error',
    data: { error: 'internal', message: 'boom' },
    headers: {},
    config: axiosConfig,
  } as unknown as AxiosResponse;
  return new AxiosError(`Request failed with status code ${status}`, AxiosError.ERR_BAD_RESPONSE, axiosConfig, {}, response);
}

async function runFailingMutation(
  client: QueryClient,
  error: Error,
  overrides: { retry?: number; retryDelay?: number } = {}
): Promise<ReturnType<typeof vi.fn>> {
  const mutationFn = vi.fn().mockRejectedValue(error);
  const observer = new MutationObserver(client, { mutationFn, ...overrides });
  // The promise settles only after react-query has exhausted its retries, so
  // the call count below is the final count, not a snapshot mid-retry.
  await expect(observer.mutate(undefined)).rejects.toBe(error);
  return mutationFn;
}

describe('app-wide mutation retry default (#1846: never replay a mutation blind)', () => {
  it('positive control: a mutation that opts in to retry: 1 IS invoked twice (the harness can observe a replay)', async () => {
    const mutationFn = await runFailingMutation(clientWithAppDefaults(), axiosTimeoutError(), {
      retry: 1,
      retryDelay: 0,
    });
    expect(mutationFn).toHaveBeenCalledTimes(2);
  });

  it('a mutation rejected by an axios client-side timeout (ECONNABORTED, no response) is invoked exactly once', async () => {
    // The server may already have accepted the POST (a job was queued, a row
    // was written); replaying it is a duplicate job — the #1836 mechanism.
    const mutationFn = await runFailingMutation(clientWithAppDefaults(), axiosTimeoutError());
    expect(mutationFn).toHaveBeenCalledTimes(1);
  });

  it('a mutation rejected by a 5xx response is invoked exactly once', async () => {
    // A 5xx proves the server received the request; whatever it persisted
    // before failing (a FAILED batch row, a queued job) would be duplicated.
    const mutationFn = await runFailingMutation(clientWithAppDefaults(), axiosServerError(500));
    expect(mutationFn).toHaveBeenCalledTimes(1);
  });

  it('a mutation rejected by a pure network error (ERR_NETWORK, no response) is ALSO invoked exactly once', async () => {
    // Deliberate: the xhr `onerror` path fires for a connection reset AFTER
    // the request body was sent as well as for a request that never left, so
    // ERR_NETWORK is not proof the server did not act. A caller whose POST is
    // idempotent can opt in with `retry` on its own hook (positive control above).
    const mutationFn = await runFailingMutation(clientWithAppDefaults(), axiosNetworkError());
    expect(mutationFn).toHaveBeenCalledTimes(1);
  });

  it('query (GET) retry behaviour is unchanged: status-aware predicate with exponential backoff', () => {
    const queries = queryClient.getDefaultOptions().queries ?? {};
    const retry = queries.retry;
    const retryDelay = queries.retryDelay;
    expect(typeof retry).toBe('function');
    expect(typeof retryDelay).toBe('function');
    const shouldRetry = retry as (failureCount: number, error: unknown) => boolean;
    const delay = retryDelay as (attemptIndex: number, error: unknown) => number;
    const withStatus = (status: number): Error => Object.assign(new Error(`HTTP ${status}`), { status });
    // 5xx and network-shaped errors retry (up to 3 times); plain 4xx do not;
    // 408 / 429 are the 4xx exceptions.
    expect(shouldRetry(0, withStatus(500))).toBe(true);
    expect(shouldRetry(2, withStatus(500))).toBe(true);
    expect(shouldRetry(3, withStatus(500))).toBe(false);
    expect(shouldRetry(0, withStatus(404))).toBe(false);
    expect(shouldRetry(0, withStatus(408))).toBe(true);
    expect(shouldRetry(0, withStatus(429))).toBe(true);
    expect(shouldRetry(0, axiosTimeoutError())).toBe(true);
    expect(delay(0, null)).toBe(1000);
    expect(delay(1, null)).toBe(2000);
    expect(delay(2, null)).toBe(4000);
  });
});

// ===========================================================================
// QUERY KEY FACTORY TESTS (findings 1 & 5)
// ===========================================================================

describe('queryKeys.digitalTwin.history (finding 1)', () => {
  it('defaults to brand=all, limit=20, offset=0 when no params given', () => {
    expect(queryKeys.digitalTwin.history()).toEqual([
      'e2i',
      'digital-twin',
      'history',
      'all',
      20,
      0,
    ]);
  });

  it('folds brand, limit and offset into the key', () => {
    expect(
      queryKeys.digitalTwin.history({ brand: 'Remibrutinib', limit: 50, offset: 100 })
    ).toEqual(['e2i', 'digital-twin', 'history', 'Remibrutinib', 50, 100]);
  });

  it('different brands produce different keys (no cross-brand cache collision)', () => {
    const all = queryKeys.digitalTwin.history({ limit: 25 });
    const remi = queryKeys.digitalTwin.history({ brand: 'Remibrutinib', limit: 25 });
    const kis = queryKeys.digitalTwin.history({ brand: 'Kisqali', limit: 25 });
    expect(all).not.toEqual(remi);
    expect(remi).not.toEqual(kis);
  });

  it('different limit/offset produce different keys', () => {
    const a = queryKeys.digitalTwin.history({ limit: 10, offset: 0 });
    const b = queryKeys.digitalTwin.history({ limit: 10, offset: 20 });
    const c = queryKeys.digitalTwin.history({ limit: 30, offset: 0 });
    expect(a).not.toEqual(b);
    expect(a).not.toEqual(c);
  });
});

describe('queryKeys.experiments.segmentResults key shape (finding 5)', () => {
  it('embeds the segment string verbatim (order normalisation is done by the hook)', () => {
    expect(queryKeys.experiments.segmentResults('exp-1', 'region,specialty')).toEqual([
      'e2i',
      'experiments',
      'results',
      'exp-1',
      'segment',
      'region,specialty',
    ]);
  });
});

// ===========================================================================
// TAB VISIBILITY HANDLER (finding 7)
// ===========================================================================

describe('tab visibility handler (finding 7: honest, no fabricated cancel)', () => {
  function setHidden(hidden: boolean): void {
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => (hidden ? 'hidden' : 'visible'),
    });
    Object.defineProperty(document, 'hidden', {
      configurable: true,
      get: () => hidden,
    });
  }

  beforeEach(() => {
    setHidden(false);
    initTabVisibilityListener();
  });

  afterEach(() => {
    cleanupTabVisibilityListener();
    vi.restoreAllMocks();
  });

  it('does NOT call cancelQueries when the tab becomes hidden (no fabricated behavior)', () => {
    const cancelSpy = vi.spyOn(queryClient, 'cancelQueries');

    setHidden(true);
    document.dispatchEvent(new Event('visibilitychange'));

    expect(cancelSpy).not.toHaveBeenCalled();
    expect(isTabCurrentlyVisible()).toBe(false);
  });

  it('tracks visibility state across hide/show transitions', () => {
    setHidden(true);
    document.dispatchEvent(new Event('visibilitychange'));
    expect(isTabCurrentlyVisible()).toBe(false);

    setHidden(false);
    document.dispatchEvent(new Event('visibilitychange'));
    expect(isTabCurrentlyVisible()).toBe(true);
  });

  it('on return-to-visible in production, nudges stale queries to refetch (existing real behavior preserved)', () => {
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries').mockResolvedValue(undefined);

    // hide then show
    setHidden(true);
    document.dispatchEvent(new Event('visibilitychange'));
    setHidden(false);
    document.dispatchEvent(new Event('visibilitychange'));

    expect(invalidateSpy).toHaveBeenCalledTimes(1);
    // It is a predicate-based stale refetch, not a blanket invalidation.
    const arg = invalidateSpy.mock.calls[0][0] as { predicate?: unknown } | undefined;
    expect(typeof arg?.predicate).toBe('function');
  });
});
