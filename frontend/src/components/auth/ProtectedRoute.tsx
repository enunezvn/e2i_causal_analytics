/**
 * Protected Route Component
 * =========================
 *
 * Route guard that protects routes requiring authentication.
 * Redirects unauthenticated users to the login page.
 *
 * Features:
 * - Checks authentication status
 * - Shows loading spinner during auth initialization
 * - Preserves intended destination for post-login redirect
 * - Optional admin-only protection
 *
 * Usage:
 *   import { ProtectedRoute } from '@/components/auth'
 *
 *   // In router:
 *   <Route path="/dashboard" element={
 *     <ProtectedRoute>
 *       <Dashboard />
 *     </ProtectedRoute>
 *   } />
 *
 *   // Admin-only route:
 *   <Route path="/admin" element={
 *     <ProtectedRoute requireAdmin>
 *       <AdminPanel />
 *     </ProtectedRoute>
 *   } />
 *
 * @module components/auth/ProtectedRoute
 */

import * as React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/hooks/use-auth';

// =============================================================================
// TYPES
// =============================================================================

export interface ProtectedRouteProps {
  children: React.ReactNode;
  /** Require admin role */
  requireAdmin?: boolean;
  /** Custom redirect path (default: /login) */
  redirectTo?: string;
  /** Custom loading component */
  loadingFallback?: React.ReactNode;
}

// =============================================================================
// LOADING SPINNER
// =============================================================================

function DefaultLoadingFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)]">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[var(--color-primary)] mx-auto" />
        <p className="mt-4 text-[var(--color-muted-foreground)]">Loading...</p>
      </div>
    </div>
  );
}

// =============================================================================
// CONFIGURATION ERROR STATE
// =============================================================================

/**
 * Visible fail-closed state shown when Supabase auth is not configured.
 *
 * Rendering this (rather than redirecting to /login, which cannot work
 * without Supabase either) makes a misconfigured build IMPOSSIBLE to miss
 * while still never granting access (see use-auth.ts fail-closed derivation).
 */
function AuthConfigurationError() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-background)] p-4">
      <div
        role="alert"
        className="max-w-lg rounded-lg border border-[var(--color-destructive)]/40 bg-[var(--color-destructive)]/10 p-6 text-center"
      >
        <h1 className="text-xl font-semibold text-[var(--color-destructive)]">
          Authentication is not configured
        </h1>
        <p className="mt-3 text-sm text-[var(--color-foreground)]">
          This deployment is missing its Supabase configuration, so sign-in is
          unavailable and access to the application is disabled.
        </p>
        <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
          Set <code className="font-mono">VITE_SUPABASE_URL</code> and{' '}
          <code className="font-mono">VITE_SUPABASE_ANON_KEY</code> at build
          time (see <code className="font-mono">frontend/.env.production</code>),
          then rebuild the frontend.
        </p>
      </div>
    </div>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

/**
 * ProtectedRoute
 *
 * Guards routes that require authentication.
 * Redirects to login page if not authenticated.
 */
export function ProtectedRoute({
  children,
  requireAdmin = false,
  redirectTo = '/login',
  loadingFallback,
}: ProtectedRouteProps) {
  const location = useLocation();
  const { isAuthenticated, isAuthConfigured, isAdmin, isInitialized, setRedirectTo } =
    useAuth();

  // FAIL CLOSED: without Supabase configuration there is no way to
  // authenticate anyone - show a visible configuration error instead of
  // either granting access (the old bypass) or redirecting to a login page
  // that cannot work. Checked before isInitialized: the configuration state
  // is static for the lifetime of the bundle.
  if (!isAuthConfigured) {
    return <AuthConfigurationError />;
  }

  // Show loading state while initializing
  if (!isInitialized) {
    return <>{loadingFallback ?? <DefaultLoadingFallback />}</>;
  }

  // Redirect if not authenticated
  if (!isAuthenticated) {
    // Save the intended destination for post-login redirect
    setRedirectTo(location.pathname);

    return (
      <Navigate
        to={redirectTo}
        state={{ from: location.pathname }}
        replace
      />
    );
  }

  // Redirect if admin required but user is not admin
  if (requireAdmin && !isAdmin) {
    return (
      <Navigate
        to="/"
        replace
      />
    );
  }

  // Render protected content
  return <>{children}</>;
}

export default ProtectedRoute;
