/**
 * useAuth Hook
 * ============
 *
 * Unified hook for accessing authentication state and actions.
 * Combines the Zustand auth store selectors with AuthContext actions.
 *
 * This provides a single, convenient interface for components to:
 * - Check authentication status
 * - Access current user information
 * - Perform auth actions (login, logout, signup)
 *
 * Usage:
 *   import { useAuth } from '@/hooks/use-auth'
 *
 *   function MyComponent() {
 *     const {
 *       user,
 *       isAuthenticated,
 *       isLoading,
 *       login,
 *       logout,
 *     } = useAuth()
 *
 *     if (isLoading) return <Spinner />
 *     if (!isAuthenticated) return <LoginButton onClick={() => login(...)} />
 *     return <UserProfile user={user} />
 *   }
 *
 * @module hooks/use-auth
 */

import { useAuthStore } from '@/stores/auth-store';
import { useAuthContext } from '@/providers/AuthProvider';
import { isSupabaseConfigured } from '@/lib/supabase';
import type { User, Session } from '@supabase/supabase-js';

// =============================================================================
// TYPES
// =============================================================================

export interface UseAuthReturn {
  // State
  /** Current authenticated user */
  user: User | null;
  /** Current session (contains access token) */
  session: Session | null;
  /** Whether auth is initializing */
  isLoading: boolean;
  /** Whether initial auth check completed */
  isInitialized: boolean;
  /** Whether user is authenticated */
  isAuthenticated: boolean;
  /**
   * Whether Supabase auth is configured (VITE_SUPABASE_URL +
   * VITE_SUPABASE_ANON_KEY present at build time). When false, auth FAILS
   * CLOSED: isAuthenticated is always false and the UI must surface a
   * configuration error (see ProtectedRoute) instead of granting access.
   */
  isAuthConfigured: boolean;
  /** Whether user has admin role */
  isAdmin: boolean;
  /** Last auth error */
  error: { code: string; message: string } | null;
  /** Access token for API requests */
  accessToken: string | null;
  /** User display info */
  userInfo: {
    email: string | null;
    name: string | null;
    avatarUrl: string | null;
  };

  // Actions
  /** Sign in with email and password */
  login: (credentials: { email: string; password: string }) => Promise<void>;
  /** Sign out the current user */
  logout: () => Promise<void>;
  /** Sign up a new user */
  signup: (credentials: {
    email: string;
    password: string;
    name?: string;
  }) => Promise<void>;
  /** Send password reset email */
  resetPassword: (email: string) => Promise<void>;
  /** Update user password */
  updatePassword: (newPassword: string) => Promise<void>;
  /** Clear auth error */
  clearError: () => void;
  /** Set redirect destination for post-login */
  setRedirectTo: (path: string | null) => void;
  /** Get redirect destination */
  redirectTo: string | null;
}

// =============================================================================
// HOOK
// =============================================================================

/**
 * Unified authentication hook
 *
 * Provides all authentication state and actions in a single hook.
 * Must be used within AuthProvider.
 *
 * @returns Authentication state and actions
 */
export function useAuth(): UseAuthReturn {
  // Get auth actions from context
  const { login, logout, signup, resetPassword, updatePassword } = useAuthContext();

  // Get state from Zustand store
  const user = useAuthStore((state) => state.user);
  const session = useAuthStore((state) => state.session);
  const isLoading = useAuthStore((state) => state.isLoading);
  const isInitialized = useAuthStore((state) => state.isInitialized);
  const error = useAuthStore((state) => state.error);
  const redirectTo = useAuthStore((state) => state.redirectTo);
  const clearError = useAuthStore((state) => state.clearError);
  const setRedirectTo = useAuthStore((state) => state.setRedirectTo);

  // Derived state
  // SECURITY - FAIL CLOSED: when Supabase is not configured the user is NEVER
  // authenticated. The previous behavior (57e151fd) hard-coded
  // isAuthenticated=true here so the demo app could run without Supabase;
  // once real auth shipped, that became a latent production auth bypass (any
  // build missing VITE_SUPABASE_* silently granted access to every protected
  // route). ProtectedRoute uses isAuthConfigured to show a visible
  // configuration-error state instead.
  const supabaseConfigured = isSupabaseConfigured();
  const isAuthenticated = supabaseConfigured && Boolean(session?.access_token && user);

  const isAdmin = (() => {
    if (!user) return false;
    const appMetadata = user.app_metadata ?? {};
    const role = user.role ?? appMetadata.role ?? '';
    return (
      role === 'admin' ||
      appMetadata.is_admin === true ||
      appMetadata.role === 'admin'
    );
  })();

  // Token exposure mirrors isAuthenticated: fail closed when unconfigured.
  const accessToken = supabaseConfigured ? (session?.access_token ?? null) : null;

  const userInfo = {
    email: user?.email ?? null,
    name: user?.user_metadata?.name ?? user?.email?.split('@')[0] ?? null,
    avatarUrl: user?.user_metadata?.avatar_url ?? null,
  };

  return {
    // State
    user,
    session,
    isLoading,
    isInitialized,
    isAuthenticated,
    isAuthConfigured: supabaseConfigured,
    isAdmin,
    error,
    accessToken,
    userInfo,
    redirectTo,

    // Actions
    login,
    logout,
    signup,
    resetPassword,
    updatePassword,
    clearError,
    setRedirectTo,
  };
}

export default useAuth;
