/**
 * useAuth Tests
 * =============
 *
 * Red-first tests for the auth fail-closed derivation. The pre-fix hook
 * hard-coded isAuthenticated=true whenever Supabase was unconfigured
 * (commit 57e151fd, a demo-era convenience) - a latent production auth
 * bypass: any build with missing VITE_SUPABASE_* silently granted access to
 * every protected route. The hook must fail CLOSED instead and expose the
 * configuration state so the UI can show a visible error.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import * as React from 'react';

const mockIsSupabaseConfigured = vi.fn();

vi.mock('@/lib/supabase', () => ({
  isSupabaseConfigured: () => mockIsSupabaseConfigured(),
  // AuthProvider only touches the client when configured; tests below run
  // unconfigured or with store-injected sessions, so a stub suffices.
  supabase: {
    auth: {
      getSession: vi.fn().mockResolvedValue({ data: { session: null }, error: null }),
      onAuthStateChange: vi
        .fn()
        .mockReturnValue({ data: { subscription: { unsubscribe: vi.fn() } } }),
    },
  },
}));

import { AuthProvider } from '@/providers/AuthProvider';
import { useAuth } from './use-auth';
import { useAuthStore } from '@/stores/auth-store';
import type { Session, User } from '@supabase/supabase-js';

function wrapper({ children }: { children: React.ReactNode }) {
  return <AuthProvider>{children}</AuthProvider>;
}

beforeEach(() => {
  mockIsSupabaseConfigured.mockReset();
  useAuthStore.setState({
    user: null,
    session: null,
    isLoading: false,
    isInitialized: false,
    error: null,
    redirectTo: null,
  });
});

describe('useAuth fail-closed derivation', () => {
  it('reports unauthenticated when Supabase is not configured (no silent bypass)', () => {
    mockIsSupabaseConfigured.mockReturnValue(false);

    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isAuthConfigured).toBe(false);
  });

  it('reports unauthenticated when configured but no session exists', () => {
    mockIsSupabaseConfigured.mockReturnValue(true);

    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.isAuthConfigured).toBe(true);
  });

  it('reports authenticated only with a real session and user', () => {
    mockIsSupabaseConfigured.mockReturnValue(true);
    const user = { id: 'u1', email: 'u@example.com' } as User;
    const session = { access_token: 'tok', user } as Session;
    useAuthStore.setState({ user, session, isInitialized: true });

    const { result } = renderHook(() => useAuth(), { wrapper });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.isAuthConfigured).toBe(true);
  });
});
