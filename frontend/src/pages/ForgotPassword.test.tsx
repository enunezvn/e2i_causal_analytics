/**
 * ForgotPassword Page Tests
 * =========================
 *
 * Red-first tests for the password recovery request page. Wires the existing
 * AuthProvider resetPassword action (supabase.auth.resetPasswordForEmail) to
 * a real page so the login page's "Forgot password?" link stops 404ing.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockResetPassword = vi.fn();
const mockClearError = vi.fn();
const mockUseAuth = vi.fn();

vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => mockUseAuth(),
}));

import ForgotPassword from './ForgotPassword';

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/forgot-password']}>
      <ForgotPassword />
    </MemoryRouter>
  );
}

beforeEach(() => {
  mockResetPassword.mockReset().mockResolvedValue(undefined);
  mockUseAuth.mockReset().mockReturnValue({
    resetPassword: mockResetPassword,
    isLoading: false,
    error: null,
    clearError: mockClearError,
  });
});

describe('ForgotPassword', () => {
  it('renders an email form', () => {
    renderPage();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send reset link/i })).toBeInTheDocument();
  });

  it('validates the email before submitting', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/email/i), 'not-an-email');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    expect(await screen.findByText(/valid email/i)).toBeInTheDocument();
    expect(mockResetPassword).not.toHaveBeenCalled();
  });

  it('submits the email through resetPassword and shows a success state', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/email/i), 'hcp@example.com');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(mockResetPassword).toHaveBeenCalledWith('hcp@example.com');
    });
    expect(await screen.findByText(/check your email/i)).toBeInTheDocument();
  });

  it('surfaces auth errors from the store', () => {
    mockUseAuth.mockReturnValue({
      resetPassword: mockResetPassword,
      isLoading: false,
      error: { code: 'unexpected_failure', message: 'Error sending recovery email' },
      clearError: mockClearError,
    });
    renderPage();

    expect(screen.getByText(/error sending recovery email/i)).toBeInTheDocument();
  });

  it('links back to the login page', () => {
    renderPage();
    expect(screen.getByRole('link', { name: /back to sign in/i })).toHaveAttribute(
      'href',
      '/login'
    );
  });
});
