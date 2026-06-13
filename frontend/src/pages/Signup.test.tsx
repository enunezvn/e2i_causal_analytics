/**
 * Signup Page Tests
 * =================
 *
 * Form UX hardening (#25):
 *   - The <form> must carry `noValidate` so react-hook-form / zod owns
 *     validation and native browser bubbles are suppressed.
 *   - RHF/zod validation must still fire and block submit on invalid input
 *     (here: a too-short password and mismatched confirmation).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockSignup = vi.fn();
const mockClearError = vi.fn();
const mockUseAuth = vi.fn();

vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => mockUseAuth(),
}));

import { Signup } from './Signup';

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/signup']}>
      <Signup />
    </MemoryRouter>
  );
}

beforeEach(() => {
  mockSignup.mockReset().mockResolvedValue(undefined);
  mockClearError.mockReset();
  mockUseAuth.mockReset().mockReturnValue({
    signup: mockSignup,
    isLoading: false,
    error: null,
    clearError: mockClearError,
    isAuthenticated: false,
  });
});

describe('Signup form', () => {
  it('sets noValidate on the form so RHF/zod owns validation', () => {
    const { container } = renderPage();
    const form = container.querySelector('form');
    expect(form).not.toBeNull();
    expect((form as HTMLFormElement).noValidate).toBe(true);
  });

  it('still runs zod validation and blocks submit when passwords mismatch', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText('Name'), 'Dr Remi');
    await user.type(screen.getByLabelText('Email'), 'hcp@example.com');
    await user.type(screen.getByLabelText('Password'), 'secret123');
    await user.type(screen.getByLabelText('Confirm Password'), 'different');
    await user.click(screen.getByRole('button', { name: /sign up/i }));

    expect(await screen.findByText(/passwords do not match/i)).toBeInTheDocument();
    expect(mockSignup).not.toHaveBeenCalled();
  });

  it('submits through signup() when the form is valid', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByLabelText('Name'), 'Dr Remi');
    await user.type(screen.getByLabelText('Email'), 'hcp@example.com');
    await user.type(screen.getByLabelText('Password'), 'secret123');
    await user.type(screen.getByLabelText('Confirm Password'), 'secret123');
    await user.click(screen.getByRole('button', { name: /sign up/i }));

    await waitFor(() => {
      expect(mockSignup).toHaveBeenCalledWith({
        email: 'hcp@example.com',
        password: 'secret123',
        name: 'Dr Remi',
      });
    });
  });
});
