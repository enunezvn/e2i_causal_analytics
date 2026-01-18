/**
 * Auth Provider
 * =============
 *
 * React context provider for authentication state using Supabase Auth.
 * Listens to auth state changes and syncs with the Zustand auth store.
 *
 * Features:
 * - Auto-initializes session from localStorage
 * - Listens to auth state changes (login, logout, token refresh)
 * - Provides login/logout/signup functions
 * - Integrates with React Query for cache invalidation on logout
 *
 * Usage:
 *   import { AuthProvider, useAuthContext } from '@/providers/AuthProvider'
 *
 *   // In main.tsx (wrap inside QueryClientProvider)
 *   <AuthProvider>
 *     <App />
 *   </AuthProvider>
 *
 *   // In components
 *   const { login, logout, signup } = useAuthContext()
 *
 * @module providers/AuthProvider
 */

import * as React from 'react';
import { supabase, isSupabaseConfigured } from '@/lib/supabase';
import { useAuthStore } from '@/stores/auth-store';
import { queryClient } from '@/lib/query-client';
import type { AuthError as SupabaseAuthError } from '@supabase/supabase-js';

// =============================================================================
// TYPES
// =============================================================================

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface SignupCredentials {
  email: string;
  password: string;
  name?: string;
}

export interface AuthContextValue {
  /** Sign in with email and password */
  login: (credentials: LoginCredentials) => Promise<void>;
  /** Sign out the current user */
  logout: () => Promise<void>;
  /** Sign up a new user */
  signup: (credentials: SignupCredentials) => Promise<void>;
  /** Send password reset email */
  resetPassword: (email: string) => Promise<void>;
  /** Update user password */
  updatePassword: (newPassword: string) => Promise<void>;
}

export interface AuthProviderProps {
  children: React.ReactNode;
}

// =============================================================================
// CONTEXT
// =============================================================================

const AuthContext = React.createContext<AuthContextValue | null>(null);

/**
 * Hook to access auth actions (login, logout, signup)
 * Use this alongside the useAuthStore selectors for state
 *
 * @throws Error if used outside AuthProvider
 */
export function useAuthContext(): AuthContextValue {
  const context = React.useContext(AuthContext);
  if (!context) {
    throw new Error('useAuthContext must be used within AuthProvider');
  }
  return context;
}

// =============================================================================
// ERROR HELPERS
// =============================================================================

/**
 * Convert Supabase auth error to our AuthError type
 */
function mapAuthError(error: SupabaseAuthError): { code: string; message: string } {
  // Map common Supabase error codes to user-friendly messages
  const errorMap: Record<string, string> = {
    invalid_credentials: 'Invalid email or password',
    email_not_confirmed: 'Please verify your email address',
    user_already_exists: 'An account with this email already exists',
    weak_password: 'Password is too weak. Use at least 6 characters.',
    invalid_email: 'Please enter a valid email address',
    signup_disabled: 'Sign up is currently disabled',
  };

  const code = error.code ?? 'unknown_error';
  const message = errorMap[code] ?? error.message ?? 'An unexpected error occurred';

  return { code, message };
}

// =============================================================================
// PROVIDER
// =============================================================================

/**
 * AuthProvider
 *
 * Provides authentication context and manages Supabase auth state.
 * Must be wrapped inside QueryClientProvider.
 */
export function AuthProvider({ children }: AuthProviderProps) {
  const { setAuth, setSession, setLoading, setInitialized, setError, clearAuth } =
    useAuthStore();

  // Check if Supabase is configured
  const supabaseConfigured = isSupabaseConfigured();

  // -------------------------------------------------------------------------
  // Initialize auth state from Supabase on mount
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    let mounted = true;

    // If Supabase is not configured, skip auth initialization
    if (!supabaseConfigured) {
      console.warn(
        '[Auth] Supabase is not configured. Authentication is disabled. ' +
          'Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable auth.'
      );
      clearAuth();
      setInitialized(true);
      return;
    }

    async function initializeAuth() {
      try {
        // Get initial session
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession();

        if (!mounted) return;

        if (error) {
          console.error('[Auth] Failed to get session:', error.message);
          setError(mapAuthError(error));
          setInitialized(true);
          return;
        }

        // Set auth state from session
        if (session) {
          setAuth(session.user, session);
        } else {
          clearAuth();
        }
        setInitialized(true);
      } catch (err) {
        console.error('[Auth] Initialization error:', err);
        if (mounted) {
          setError({
            code: 'init_error',
            message: 'Failed to initialize authentication',
          });
          setInitialized(true);
        }
      }
    }

    initializeAuth();

    // Listen for auth state changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange(async (event, session) => {
      if (!mounted) return;

      console.debug('[Auth] State change:', event);

      switch (event) {
        case 'SIGNED_IN':
        case 'TOKEN_REFRESHED':
          if (session) {
            setSession(session);
          }
          break;

        case 'SIGNED_OUT':
          clearAuth();
          // Clear React Query cache on logout
          queryClient.clear();
          break;

        case 'USER_UPDATED':
          if (session) {
            setSession(session);
          }
          break;

        case 'PASSWORD_RECOVERY':
          // User clicked password reset link
          // The session will be set automatically
          break;

        default:
          // Handle other events if needed
          break;
      }
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, [supabaseConfigured, setAuth, setSession, setLoading, setInitialized, setError, clearAuth]);

  // -------------------------------------------------------------------------
  // Auth Actions
  // -------------------------------------------------------------------------

  /**
   * Sign in with email and password
   */
  const login = React.useCallback(
    async ({ email, password }: LoginCredentials): Promise<void> => {
      if (!supabaseConfigured) {
        throw new Error('Authentication is not configured');
      }
      setLoading(true);
      setError(null);

      try {
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) {
          setError(mapAuthError(error));
          throw new Error(error.message);
        }

        // Auth state change listener will handle setting the session
        console.debug('[Auth] Login successful:', data.user?.email);
      } catch (err) {
        setLoading(false);
        throw err;
      }
    },
    [supabaseConfigured, setLoading, setError]
  );

  /**
   * Sign out the current user
   */
  const logout = React.useCallback(async (): Promise<void> => {
    if (!supabaseConfigured) {
      // If Supabase is not configured, just clear local auth state
      clearAuth();
      return;
    }
    setLoading(true);

    try {
      const { error } = await supabase.auth.signOut();

      if (error) {
        console.error('[Auth] Logout error:', error.message);
        setError(mapAuthError(error));
        throw new Error(error.message);
      }

      // Auth state change listener will handle clearing state
      console.debug('[Auth] Logout successful');
    } catch (err) {
      setLoading(false);
      throw err;
    }
  }, [supabaseConfigured, setLoading, setError, clearAuth]);

  /**
   * Sign up a new user
   */
  const signup = React.useCallback(
    async ({ email, password, name }: SignupCredentials): Promise<void> => {
      if (!supabaseConfigured) {
        throw new Error('Authentication is not configured');
      }
      setLoading(true);
      setError(null);

      try {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: {
              name: name ?? email.split('@')[0],
            },
          },
        });

        if (error) {
          setError(mapAuthError(error));
          throw new Error(error.message);
        }

        // Check if email confirmation is required
        if (data.user && !data.session) {
          // User needs to confirm email
          console.debug('[Auth] Signup successful, email confirmation required');
        } else {
          console.debug('[Auth] Signup successful:', data.user?.email);
        }
      } catch (err) {
        setLoading(false);
        throw err;
      }
    },
    [supabaseConfigured, setLoading, setError]
  );

  /**
   * Send password reset email
   */
  const resetPassword = React.useCallback(
    async (email: string): Promise<void> => {
      if (!supabaseConfigured) {
        throw new Error('Authentication is not configured');
      }
      setLoading(true);
      setError(null);

      try {
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/reset-password`,
        });

        if (error) {
          setError(mapAuthError(error));
          throw new Error(error.message);
        }

        console.debug('[Auth] Password reset email sent to:', email);
      } finally {
        setLoading(false);
      }
    },
    [supabaseConfigured, setLoading, setError]
  );

  /**
   * Update user password (after password reset)
   */
  const updatePassword = React.useCallback(
    async (newPassword: string): Promise<void> => {
      if (!supabaseConfigured) {
        throw new Error('Authentication is not configured');
      }
      setLoading(true);
      setError(null);

      try {
        const { error } = await supabase.auth.updateUser({
          password: newPassword,
        });

        if (error) {
          setError(mapAuthError(error));
          throw new Error(error.message);
        }

        console.debug('[Auth] Password updated successfully');
      } finally {
        setLoading(false);
      }
    },
    [supabaseConfigured, setLoading, setError]
  );

  // -------------------------------------------------------------------------
  // Context Value
  // -------------------------------------------------------------------------

  const contextValue = React.useMemo<AuthContextValue>(
    () => ({
      login,
      logout,
      signup,
      resetPassword,
      updatePassword,
    }),
    [login, logout, signup, resetPassword, updatePassword]
  );

  return <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>;
}

export default AuthProvider;
