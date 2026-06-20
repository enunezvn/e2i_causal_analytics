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
