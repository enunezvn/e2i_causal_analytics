/**
 * Auth Store Selector Tests
 * =========================
 *
 * Red-first test for fail-closed useAccessToken (codex iter1 MED): the
 * selector returned the persisted session token unconditionally, so shell
 * integrations mounted outside ProtectedRoute (CopilotKitWrapper passes it
 * as Authorization) could send a stale rehydrated token in an UNCONFIGURED
 * build. The selector must return null unless Supabase is configured.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';

const mockIsSupabaseConfigured = vi.fn();
vi.mock('@/lib/supabase', () => ({
  isSupabaseConfigured: () => mockIsSupabaseConfigured(),
}));

import { useAuthStore, useAccessToken } from './auth-store';
import type { Session, User } from '@supabase/supabase-js';

const user = { id: 'u1', email: 'u@example.com' } as User;
const session = { access_token: 'persisted-token', user } as Session;

beforeEach(() => {
  mockIsSupabaseConfigured.mockReset();
  useAuthStore.setState({ session: null, user: null });
});

describe('useAccessToken fail-closed', () => {
  it('returns null when Supabase is unconfigured, even with a persisted session', () => {
    mockIsSupabaseConfigured.mockReturnValue(false);
    useAuthStore.setState({ session, user });

    const { result } = renderHook(() => useAccessToken());

    expect(result.current).toBeNull();
  });

  it('returns the token when configured and a session exists', () => {
    mockIsSupabaseConfigured.mockReturnValue(true);
    useAuthStore.setState({ session, user });

    const { result } = renderHook(() => useAccessToken());

    expect(result.current).toBe('persisted-token');
  });

  it('returns null when configured but no session exists', () => {
    mockIsSupabaseConfigured.mockReturnValue(true);

    const { result } = renderHook(() => useAccessToken());

    expect(result.current).toBeNull();
  });
});
