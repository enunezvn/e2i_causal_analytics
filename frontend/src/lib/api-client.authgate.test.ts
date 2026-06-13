/**
 * API Client Auth-Gate Tests
 * ==========================
 *
 * Red-first tests for fail-closed token attachment (codex iter1 MED).
 * The request interceptor read the persisted store session directly, so a
 * stale session rehydrated from localStorage (written by an older CONFIGURED
 * build) would still be attached as a bearer token in an UNCONFIGURED build
 * during the window before AuthProvider's effect clears auth. The
 * interceptor must not attach Authorization unless Supabase is configured.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { InternalAxiosRequestConfig } from 'axios';

const { mockAxiosInstance } = vi.hoisted(() => ({
  mockAxiosInstance: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  },
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

const mockIsSupabaseConfigured = vi.fn();
vi.mock('@/lib/supabase', () => ({
  isSupabaseConfigured: () => mockIsSupabaseConfigured(),
}));

// Importing registers the interceptors on the mocked axios instance
import './api-client';
import { useAuthStore } from '@/stores/auth-store';
import type { Session, User } from '@supabase/supabase-js';

type RequestInterceptor = (
  config: InternalAxiosRequestConfig
) => InternalAxiosRequestConfig;

function getRequestInterceptor(): RequestInterceptor {
  const calls = mockAxiosInstance.interceptors.request.use.mock.calls;
  expect(calls.length).toBeGreaterThan(0);
  return calls[0][0] as RequestInterceptor;
}

function makeConfig(): InternalAxiosRequestConfig {
  return { headers: {}, method: 'get', url: '/kpis' } as InternalAxiosRequestConfig;
}

const user = { id: 'u1', email: 'u@example.com' } as User;
const staleSession = { access_token: 'stale-persisted-token', user } as Session;

beforeEach(() => {
  mockIsSupabaseConfigured.mockReset();
  useAuthStore.setState({ session: null, user: null });
});

describe('request interceptor token gating', () => {
  it('does NOT attach Authorization when Supabase is unconfigured, even with a persisted session', () => {
    mockIsSupabaseConfigured.mockReturnValue(false);
    useAuthStore.setState({ session: staleSession, user });

    const result = getRequestInterceptor()(makeConfig());

    expect(result.headers['Authorization']).toBeUndefined();
  });

  it('attaches Authorization when configured and a session exists', () => {
    mockIsSupabaseConfigured.mockReturnValue(true);
    useAuthStore.setState({ session: staleSession, user });

    const result = getRequestInterceptor()(makeConfig());

    expect(result.headers['Authorization']).toBe('Bearer stale-persisted-token');
  });

  it('attaches no Authorization when configured but no session exists', () => {
    mockIsSupabaseConfigured.mockReturnValue(true);

    const result = getRequestInterceptor()(makeConfig());

    expect(result.headers['Authorization']).toBeUndefined();
  });
});
