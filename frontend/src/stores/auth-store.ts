/**
 * Auth Store
 * ==========
 *
 * Zustand store for managing authentication state.
 * Handles user session, loading states, and auth actions.
 *
 * Usage:
 *   import { useAuthStore } from '@/stores/auth-store'
 *   const { user, isAuthenticated, login, logout } = useAuthStore()
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { Session, User } from '@supabase/supabase-js';

/**
 * Auth error type for consistent error handling
 */
export interface AuthError {
  code: string;
  message: string;
}

/**
 * Auth store state interface
 */
export interface AuthState {
  /** Current authenticated user */
  user: User | null;
  /** Current session (contains access token) */
  session: Session | null;
  /** Loading state during auth operations */
  isLoading: boolean;
  /** Initial auth check completed */
  isInitialized: boolean;
  /** Last auth error */
  error: AuthError | null;
  /** Intended redirect after login */
  redirectTo: string | null;
}

/**
 * Auth store actions interface
 */
export interface AuthActions {
  /** Set the current user */
  setUser: (user: User | null) => void;
  /** Set the current session */
  setSession: (session: Session | null) => void;
  /** Set both user and session at once */
  setAuth: (user: User | null, session: Session | null) => void;
  /** Set loading state */
  setLoading: (isLoading: boolean) => void;
  /** Mark auth as initialized (initial check complete) */
  setInitialized: (isInitialized: boolean) => void;
  /** Set auth error */
  setError: (error: AuthError | null) => void;
  /** Set redirect destination */
  setRedirectTo: (path: string | null) => void;
  /** Clear all auth state (logout) */
  clearAuth: () => void;
  /** Reset error state */
  clearError: () => void;
}

/**
 * Combined auth store type
 */
export type AuthStore = AuthState & AuthActions;

/**
 * Initial state for the auth store
 */
const initialState: AuthState = {
  user: null,
  session: null,
  isLoading: true, // Start loading until initialized
  isInitialized: false,
  error: null,
  redirectTo: null,
};

/**
 * Auth Store
 *
 * Manages authentication state across the application.
 * Session is persisted to localStorage for refresh.
 */
export const useAuthStore = create<AuthStore>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        ...initialState,

        // User actions
        setUser: (user) => set({ user, error: null }),

        // Session actions
        setSession: (session) =>
          set({
            session,
            user: session?.user ?? null,
            error: null,
          }),

        // Combined auth action
        setAuth: (user, session) =>
          set({
            user,
            session,
            error: null,
            isLoading: false,
          }),

        // Loading state
        setLoading: (isLoading) => set({ isLoading }),

        // Initialization state
        setInitialized: (isInitialized) =>
          set({ isInitialized, isLoading: false }),

        // Error handling
        setError: (error) => set({ error, isLoading: false }),
        clearError: () => set({ error: null }),

        // Redirect handling
        setRedirectTo: (redirectTo) => set({ redirectTo }),

        // Clear auth (logout)
        clearAuth: () =>
          set({
            user: null,
            session: null,
            error: null,
            isLoading: false,
            redirectTo: null,
          }),
      }),
      {
        name: 'e2i-auth-store',
        // Only persist session for token refresh on page reload
        // User will be re-fetched from session
        partialize: (state) => ({
          session: state.session,
          redirectTo: state.redirectTo,
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

/**
 * Selector: Check if user is authenticated
 */
export const useIsAuthenticated = () =>
  useAuthStore((state) => Boolean(state.session?.access_token && state.user));

/**
 * Selector: Check if user is admin
 */
export const useIsAdmin = () =>
  useAuthStore((state) => {
    const { user } = state;
    if (!user) return false;

    // Check for admin role in app_metadata or user_metadata
    const appMetadata = user.app_metadata ?? {};
    const role = user.role ?? appMetadata.role ?? '';

    return (
      role === 'admin' ||
      appMetadata.is_admin === true ||
      appMetadata.role === 'admin'
    );
  });

/**
 * Selector: Get current user info
 */
export const useCurrentUser = () =>
  useAuthStore((state) => ({
    user: state.user,
    email: state.user?.email,
    name: state.user?.user_metadata?.name ?? state.user?.email?.split('@')[0],
    avatarUrl: state.user?.user_metadata?.avatar_url,
  }));

/**
 * Selector: Get auth loading state
 */
export const useAuthLoading = () =>
  useAuthStore((state) => ({
    isLoading: state.isLoading,
    isInitialized: state.isInitialized,
  }));

/**
 * Selector: Get auth error
 */
export const useAuthError = () =>
  useAuthStore((state) => ({
    error: state.error,
    clearError: state.clearError,
  }));

/**
 * Selector: Get access token for API requests
 */
export const useAccessToken = () =>
  useAuthStore((state) => state.session?.access_token ?? null);

export default useAuthStore;
