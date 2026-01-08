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
  const { isAuthenticated, isAdmin, isInitialized, setRedirectTo } = useAuth();

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
