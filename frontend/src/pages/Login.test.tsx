/**
 * Login Page Tests
 * ================
 *
 * Form UX hardening (#25):
 *   - The <form> must carry `noValidate` so react-hook-form / zod owns
 *     validation and native browser bubbles are suppressed (otherwise the
 *     browser's own "Please fill out this field" popups race with — and mask —
 *     the zod messages, and the assertions below would be flaky).
 *   - RHF/zod validation must still fire and block submit on invalid input.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockLogin = vi.fn();
const mockClearError = vi.fn();
const mockSetRedirectTo = vi.fn();
const mockUseAuth = vi.fn();

vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => mockUseAuth(),
}));

import { Login } from './Login';

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/login']}>
      <Login />
    </MemoryRouter>
  );
}

beforeEach(() => {
  mockLogin.mockReset().mockResolvedValue(undefined);
  mockClearError.mockReset();
  mockSetRedirectTo.mockReset();
  mockUseAuth.mockReset().mockReturnValue({
    login: mockLogin,
    isLoading: false,
    error: null,
    clearError: mockClearError,
    isAuthenticated: false,
    redirectTo: null,
    setRedirectTo: mockSetRedirectTo,
  });
});

describe('Login form', () => {
  it('sets noValidate on the form so RHF/zod owns validation', () => {
    const { container } = renderPage();
    const form = container.querySelector('form');
    expect(form).not.toBeNull();
    // noValidate reflects as the boolean DOM property on HTMLFormElement.
    expect((form as HTMLFormElement).noValidate).toBe(true);
  });

  it('still runs zod validation and blocks submit when email is invalid', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/email/i), 'not-an-email');
    await user.type(screen.getByLabelText(/password/i), 'secret123');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    // The zod message must surface (not the native browser bubble), and the
    // auth action must NOT have been called with the invalid input.
    expect(await screen.findByText(/valid email address/i)).toBeInTheDocument();
    expect(mockLogin).not.toHaveBeenCalled();
  });

  it('submits through login() when the form is valid', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText(/email/i), 'hcp@example.com');
    await user.type(screen.getByLabelText(/password/i), 'secret123');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith({
        email: 'hcp@example.com',
        password: 'secret123',
      });
    });
  });
});
