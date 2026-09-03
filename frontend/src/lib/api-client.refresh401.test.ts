/**
 * API Client 401 refresh-and-replay tests
 * =======================================
 *
 * Red-first (2026-09-03). On tab wake after a long idle, react-query's focus
 * refetch fired with the auth store's STALE access token before supabase-js's
 * own refresh ran: four endpoints 401'd in one burst ("Invalid or expired
 * token"), the 4xx was never retried, and /segments/datasets sat in error
 * ("Couldn't load the full analysis options…") until the next focus/mount.
 *
 * The error interceptor must resolve a fresh session ONCE for the burst,
 * store it (so the request interceptor attaches it from then on), and replay
 * each rejected request one time with the new bearer token.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from 'axios';
import type { Session, User } from '@supabase/supabase-js';

const { mockAxiosInstance, mockGetSession, mockRefreshSession, mockIsSupabaseConfigured } =
  vi.hoisted(() => ({
    mockAxiosInstance: {
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      patch: vi.fn(),
      delete: vi.fn(),
      request: vi.fn(),
      interceptors: {
        request: { use: vi.fn() },
        response: { use: vi.fn() },
      },
    },
    mockGetSession: vi.fn(),
    mockRefreshSession: vi.fn(),
    mockIsSupabaseConfigured: vi.fn(),
  }));

vi.mock('axios', async () => {
  const actual = await vi.importActual<typeof import('axios')>('axios');
  return {
    ...actual,
    default: {
      ...(actual as { default: object }).default,
      create: vi.fn(() => mockAxiosInstance),
    },
    create: vi.fn(() => mockAxiosInstance),
  };
});

vi.mock('@/config/env', () => ({
  env: {
    apiUrl: 'http://localhost:8000',
    isDev: false,
  },
  buildApiUrl: (path: string) => `http://localhost:8000${path}`,
}));

vi.mock('@/lib/supabase', () => ({
  isSupabaseConfigured: () => mockIsSupabaseConfigured(),
  supabase: {
    auth: {
      getSession: (...args: unknown[]) => mockGetSession(...args),
      refreshSession: (...args: unknown[]) => mockRefreshSession(...args),
    },
  },
}));

// Importing registers the interceptors on the mocked axios instance
import { ApiError } from './api-client';
import { useAuthStore } from '@/stores/auth-store';

type ErrorInterceptor = (error: AxiosError) => Promise<unknown>;

function getErrorInterceptor(): ErrorInterceptor {
  const calls = mockAxiosInstance.interceptors.response.use.mock.calls;
  expect(calls.length).toBeGreaterThan(0);
  return calls[0][1] as ErrorInterceptor;
}

const user = { id: 'u1', email: 'u@example.com' } as User;
const staleSession = {
  access_token: 'stale-token',
  refresh_token: 'stale-refresh',
  user,
} as Session;
const freshSession = {
  access_token: 'fresh-token',
  refresh_token: 'fresh-refresh',
  user,
} as Session;

function makeError(
  status: number,
  token: string | null,
  extra: Partial<InternalAxiosRequestConfig> = {}
): AxiosError {
  const headers = new AxiosHeaders();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const config = {
    headers,
    method: 'get',
    url: '/segments/datasets',
    params: { brand: 'Remibrutinib' },
    ...extra,
  } as InternalAxiosRequestConfig;
  return new AxiosError(
    `Request failed with status code ${status}`,
    status === 401 ? 'ERR_BAD_REQUEST' : 'ERR_BAD_RESPONSE',
    config,
    {},
    {
      status,
      statusText: status === 401 ? 'Unauthorized' : 'Server Error',
      data: { error: 'unauthorized', message: 'Invalid or expired token' },
      headers: {},
      config,
    }
  );
}

function bearerOf(config: InternalAxiosRequestConfig): string | undefined {
  return AxiosHeaders.from(config.headers).get('Authorization') as string | undefined;
}

beforeEach(() => {
  mockAxiosInstance.request.mockReset();
  mockGetSession.mockReset();
  mockRefreshSession.mockReset();
  mockIsSupabaseConfigured.mockReset();
  mockIsSupabaseConfigured.mockReturnValue(true);
  useAuthStore.setState({ session: staleSession, user });
});

describe('errorInterceptor: 401 refresh-and-replay', () => {
  it('replays a rejected request once with the refreshed token and stores the session', async () => {
    mockGetSession.mockResolvedValue({ data: { session: freshSession }, error: null });
    mockAxiosInstance.request.mockResolvedValue({ status: 200, data: { ok: true } });

    const result = await getErrorInterceptor()(makeError(401, 'stale-token'));

    expect(result).toEqual({ status: 200, data: { ok: true } });
    expect(mockAxiosInstance.request).toHaveBeenCalledTimes(1);
    const replayed = mockAxiosInstance.request.mock.calls[0][0] as InternalAxiosRequestConfig;
    expect(bearerOf(replayed)).toBe('Bearer fresh-token');
    expect(replayed.url).toBe('/segments/datasets');
    expect(replayed.params).toEqual({ brand: 'Remibrutinib' });
    // The store now carries the fresh session, so the request interceptor
    // attaches it to every later request.
    expect(useAuthStore.getState().session?.access_token).toBe('fresh-token');
    expect(mockRefreshSession).not.toHaveBeenCalled();
  });

  it('forces one token rotation when getSession still returns the rejected token', async () => {
    // Not expired by the local clock (clock skew / server-side revocation of
    // the access token) — supabase-js will not refresh on its own.
    mockGetSession.mockResolvedValue({ data: { session: staleSession }, error: null });
    mockRefreshSession.mockResolvedValue({ data: { session: freshSession, user }, error: null });
    mockAxiosInstance.request.mockResolvedValue({ status: 200, data: [] });

    await getErrorInterceptor()(makeError(401, 'stale-token'));

    expect(mockRefreshSession).toHaveBeenCalledTimes(1);
    const replayed = mockAxiosInstance.request.mock.calls[0][0] as InternalAxiosRequestConfig;
    expect(bearerOf(replayed)).toBe('Bearer fresh-token');
    expect(useAuthStore.getState().session?.access_token).toBe('fresh-token');
  });

  it('rejects with the original 401 ApiError when no fresher session exists', async () => {
    mockGetSession.mockResolvedValue({ data: { session: staleSession }, error: null });
    mockRefreshSession.mockResolvedValue({
      data: { session: null, user: null },
      error: { name: 'AuthApiError', message: 'Invalid Refresh Token' },
    });

    await expect(getErrorInterceptor()(makeError(401, 'stale-token'))).rejects.toMatchObject({
      name: 'ApiError',
      status: 401,
    });
    expect(mockAxiosInstance.request).not.toHaveBeenCalled();
    // The store is left alone — AuthProvider owns sign-out on a dead session.
    expect(useAuthStore.getState().session?.access_token).toBe('stale-token');
  });

  it('never replays a request that has already been replayed', async () => {
    mockGetSession.mockResolvedValue({ data: { session: freshSession }, error: null });

    const second = makeError(401, 'fresh-token', { e2iAuthReplayed: true } as Partial<InternalAxiosRequestConfig>);
    await expect(getErrorInterceptor()(second)).rejects.toBeInstanceOf(ApiError);

    expect(mockGetSession).not.toHaveBeenCalled();
    expect(mockAxiosInstance.request).not.toHaveBeenCalled();
  });

  it('leaves 401s alone when Supabase is not configured', async () => {
    mockIsSupabaseConfigured.mockReturnValue(false);

    await expect(getErrorInterceptor()(makeError(401, null))).rejects.toBeInstanceOf(ApiError);

    expect(mockGetSession).not.toHaveBeenCalled();
    expect(mockAxiosInstance.request).not.toHaveBeenCalled();
  });

  it('resolves the session once for a burst of concurrent 401s', async () => {
    let release: (value: { data: { session: Session }; error: null }) => void = () => {};
    mockGetSession.mockImplementation(
      () =>
        new Promise((resolve) => {
          release = resolve;
        })
    );
    mockAxiosInstance.request.mockImplementation(async (config: InternalAxiosRequestConfig) => ({
      status: 200,
      data: { url: config.url },
    }));
    const interceptor = getErrorInterceptor();

    const burst = ['/segments/policies', '/agents/status', '/causal/variables', '/segments/datasets'].map(
      (url) => interceptor(makeError(401, 'stale-token', { url }))
    );
    // Let every interceptor reach the shared resolution before releasing it.
    await new Promise((r) => setTimeout(r, 0));
    release({ data: { session: freshSession }, error: null });
    const results = (await Promise.all(burst)) as Array<{ data: { url: string } }>;

    expect(mockGetSession).toHaveBeenCalledTimes(1);
    expect(mockAxiosInstance.request).toHaveBeenCalledTimes(4);
    expect(results.map((r) => r.data.url).sort()).toEqual(
      ['/agents/status', '/causal/variables', '/segments/datasets', '/segments/policies'].sort()
    );
    for (const call of mockAxiosInstance.request.mock.calls) {
      expect(bearerOf(call[0] as InternalAxiosRequestConfig)).toBe('Bearer fresh-token');
    }
  });

  it('does not let a 401 carrying a newer token join an older token\'s refresh', async () => {
    // Codex iter-1 MED: a burst can carry two different rejected tokens (a
    // request built before a rotation and one built after it). The second must
    // not adopt the first burst's result if that result IS its own rejected
    // token — that would spend its only replay on a token the API just refused.
    let releaseA: (value: { data: { session: Session }; error: null }) => void = () => {};
    mockGetSession
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            releaseA = resolve;
          })
      )
      .mockResolvedValueOnce({ data: { session: freshSession }, error: null });
    const rotatedSession = { ...freshSession, access_token: 'rotated-token' } as Session;
    mockRefreshSession.mockResolvedValue({ data: { session: rotatedSession, user }, error: null });
    mockAxiosInstance.request.mockImplementation(async (config: InternalAxiosRequestConfig) => ({
      status: 200,
      data: { url: config.url },
    }));
    const interceptor = getErrorInterceptor();

    const forA = interceptor(makeError(401, 'stale-token', { url: '/a' }));
    const forB = interceptor(makeError(401, 'fresh-token', { url: '/b' }));
    await new Promise((r) => setTimeout(r, 0));
    releaseA({ data: { session: freshSession }, error: null });
    await Promise.all([forA, forB]);

    const replays = mockAxiosInstance.request.mock.calls.map(
      (c) => c[0] as InternalAxiosRequestConfig
    );
    const replayA = replays.find((c) => c.url === '/a');
    const replayB = replays.find((c) => c.url === '/b');
    expect(replayA && bearerOf(replayA)).toBe('Bearer fresh-token');
    // B was rejected WITH fresh-token, so its replay must carry something newer.
    expect(replayB && bearerOf(replayB)).toBe('Bearer rotated-token');
    expect(mockRefreshSession).toHaveBeenCalledTimes(1);
  });

  it('does not touch non-401 errors', async () => {
    await expect(getErrorInterceptor()(makeError(500, 'stale-token'))).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
    });
    expect(mockGetSession).not.toHaveBeenCalled();
    expect(mockAxiosInstance.request).not.toHaveBeenCalled();
  });
});
