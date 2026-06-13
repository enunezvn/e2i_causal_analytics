/**
 * ProtectedRoute Tests
 * ====================
 *
 * Red-first tests for auth fail-closed behavior: when Supabase is NOT
 * configured (VITE_SUPABASE_* missing at build time), protected routes must
 * NOT render their children. They must show a visible configuration-error
 * state instead of silently authenticating (the pre-fix latent bypass:
 * use-auth.ts returned isAuthenticated=true when unconfigured).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';

// Mock the unified auth hook - ProtectedRoute derives everything from it
const mockUseAuth = vi.fn();
vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => mockUseAuth(),
}));

import { ProtectedRoute } from './ProtectedRoute';

function renderProtected() {
  return render(
    <MemoryRouter initialEntries={['/protected']}>
      <Routes>
        <Route
          path="/protected"
          element={
            <ProtectedRoute>
              <div data-testid="protected-content">secret dashboard</div>
            </ProtectedRoute>
          }
        />
        <Route path="/login" element={<div data-testid="login-page">login</div>} />
      </Routes>
    </MemoryRouter>
  );
}

const baseAuth = {
  isInitialized: true,
  isAdmin: false,
  setRedirectTo: vi.fn(),
};

beforeEach(() => {
  mockUseAuth.mockReset();
});

describe('ProtectedRoute fail-closed', () => {
  it('shows a configuration error and hides content when Supabase is not configured', () => {
    // Simulates the pre-fix bypass state: hook reports authenticated although
    // auth is unconfigured. ProtectedRoute must fail CLOSED on isAuthConfigured.
    mockUseAuth.mockReturnValue({
      ...baseAuth,
      isAuthConfigured: false,
      isAuthenticated: true,
    });

    renderProtected();

    expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent(/authentication is not configured/i);
    expect(alert).toHaveTextContent(/VITE_SUPABASE_URL/);
    expect(alert).toHaveTextContent(/VITE_SUPABASE_ANON_KEY/);
  });

  it('does not redirect to login when unconfigured (login cannot work either)', () => {
    mockUseAuth.mockReturnValue({
      ...baseAuth,
      isAuthConfigured: false,
      isAuthenticated: false,
    });

    renderProtected();

    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('redirects unauthenticated users to login when configured', () => {
    mockUseAuth.mockReturnValue({
      ...baseAuth,
      isAuthConfigured: true,
      isAuthenticated: false,
    });

    renderProtected();

    expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
    expect(screen.getByTestId('login-page')).toBeInTheDocument();
  });

  it('renders children for authenticated users when configured', () => {
    mockUseAuth.mockReturnValue({
      ...baseAuth,
      isAuthConfigured: true,
      isAuthenticated: true,
    });

    renderProtected();

    expect(screen.getByTestId('protected-content')).toBeInTheDocument();
  });
});
