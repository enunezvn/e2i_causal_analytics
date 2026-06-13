/**
 * ResetPassword Page Tests
 * ========================
 *
 * Red-first tests for the password update page - the redirect target of
 * AuthProvider.resetPassword ({origin}/reset-password). Supabase's
 * detectSessionInUrl establishes the recovery session from the email link;
 * this page then calls the existing updatePassword action
 * (supabase.auth.updateUser).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockUpdatePassword = vi.fn();
const mockClearError = vi.fn();
const mockUseAuth = vi.fn();

vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => mockUseAuth(),
}));

import ResetPassword from './ResetPassword';

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/reset-password']}>
      <ResetPassword />
    </MemoryRouter>
  );
}

beforeEach(() => {
  mockUpdatePassword.mockReset().mockResolvedValue(undefined);
  mockUseAuth.mockReset().mockReturnValue({
    updatePassword: mockUpdatePassword,
    isLoading: false,
    error: null,
    clearError: mockClearError,
  });
});

describe('ResetPassword', () => {
  it('renders new password and confirmation fields', () => {
    renderPage();
    expect(screen.getByLabelText(/^new password$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /update password/i })).toBeInTheDocument();
  });

  it('rejects mismatched passwords without calling updatePassword', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/^new password$/i), 'Str0ng!Passw0rd');
    await user.type(screen.getByLabelText(/confirm password/i), 'Different!Passw0rd');
    await user.click(screen.getByRole('button', { name: /update password/i }));

    expect(await screen.findByText(/do not match/i)).toBeInTheDocument();
    expect(mockUpdatePassword).not.toHaveBeenCalled();
  });

  it('rejects short passwords', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/^new password$/i), 'abc');
    await user.type(screen.getByLabelText(/confirm password/i), 'abc');
    await user.click(screen.getByRole('button', { name: /update password/i }));

    expect(await screen.findByText(/at least 8 characters/i)).toBeInTheDocument();
    expect(mockUpdatePassword).not.toHaveBeenCalled();
  });

  it('updates the password and shows a success state with a login link', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/^new password$/i), 'Str0ng!Passw0rd');
    await user.type(screen.getByLabelText(/confirm password/i), 'Str0ng!Passw0rd');
    await user.click(screen.getByRole('button', { name: /update password/i }));

    await waitFor(() => {
      expect(mockUpdatePassword).toHaveBeenCalledWith('Str0ng!Passw0rd');
    });
    expect(await screen.findByText(/password updated/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /sign in/i })).toHaveAttribute('href', '/login');
  });

  it('surfaces auth errors (e.g. expired recovery link)', () => {
    mockUseAuth.mockReturnValue({
      updatePassword: mockUpdatePassword,
      isLoading: false,
      error: { code: 'unknown_error', message: 'Auth session missing!' },
      clearError: mockClearError,
    });
    renderPage();

    expect(screen.getByText(/auth session missing/i)).toBeInTheDocument();
    // Recovery escape hatch back to requesting a new link
    expect(screen.getByRole('link', { name: /request a new link/i })).toHaveAttribute(
      'href',
      '/forgot-password'
    );
  });
});
