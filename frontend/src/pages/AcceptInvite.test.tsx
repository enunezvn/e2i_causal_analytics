/**
 * AcceptInvite Page Tests
 * =======================
 * The page verifies an invite token_hash via supabase.auth.verifyOtp, then
 * lets the invitee set a password (updateUser) and enter the app.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';

const mockVerifyOtp = vi.fn();
const mockUpdateUser = vi.fn();

vi.mock('@/lib/supabase', () => ({
  supabase: {
    auth: {
      verifyOtp: (...args: unknown[]) => mockVerifyOtp(...args),
      updateUser: (...args: unknown[]) => mockUpdateUser(...args),
    },
  },
  isSupabaseConfigured: () => true,
}));

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const mod = await importOriginal<typeof import('react-router-dom')>();
  return { ...mod, useNavigate: () => mockNavigate };
});

import AcceptInvite from './AcceptInvite';

function renderAt(url: string) {
  return render(
    <MemoryRouter initialEntries={[url]}>
      <AcceptInvite />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('AcceptInvite', () => {
  it('shows an error when the link has no token_hash', () => {
    renderAt('/accept-invite');
    expect(screen.getByText(/invalid invite link/i)).toBeInTheDocument();
    expect(mockVerifyOtp).not.toHaveBeenCalled();
  });

  it('verifies the token and shows the set-password form', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: { access_token: 't' }, user: { email: 'new@x.com' } },
      error: null,
    });
    renderAt('/accept-invite?token_hash=abc123');
    await waitFor(() =>
      expect(mockVerifyOtp).toHaveBeenCalledWith({ type: 'invite', token_hash: 'abc123' })
    );
    expect(await screen.findByLabelText(/new password/i)).toBeInTheDocument();
  });

  it('falls back to recovery-type verification when invite type fails', async () => {
    mockVerifyOtp
      .mockResolvedValueOnce({
        data: { session: null, user: null },
        error: { message: 'Email link is invalid or has expired' },
      })
      .mockResolvedValueOnce({
        data: { session: { access_token: 't' }, user: { email: 'back@x.com' } },
        error: null,
      });
    renderAt('/accept-invite?token_hash=rec456');
    await waitFor(() =>
      expect(mockVerifyOtp).toHaveBeenCalledWith({ type: 'recovery', token_hash: 'rec456' })
    );
    expect(await screen.findByLabelText(/new password/i)).toBeInTheDocument();
  });

  it('shows expired-link error when verification fails', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: null, user: null },
      error: { message: 'Email link is invalid or has expired' },
    });
    renderAt('/accept-invite?token_hash=stale');
    expect(await screen.findByText(/invalid or has expired/i)).toBeInTheDocument();
  });

  it('sets the password and navigates into the app', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: { access_token: 't' }, user: { email: 'new@x.com' } },
      error: null,
    });
    mockUpdateUser.mockResolvedValue({ data: { user: {} }, error: null });
    renderAt('/accept-invite?token_hash=abc123');
    const pw = await screen.findByLabelText(/new password/i);
    const confirm = screen.getByLabelText(/confirm password/i);
    await userEvent.type(pw, 'Str0ng!Passw0rd');
    await userEvent.type(confirm, 'Str0ng!Passw0rd');
    await userEvent.click(screen.getByRole('button', { name: /set password/i }));
    await waitFor(() =>
      expect(mockUpdateUser).toHaveBeenCalledWith({ password: 'Str0ng!Passw0rd' })
    );
    expect(mockNavigate).toHaveBeenCalledWith('/', { replace: true });
  });

  it('rejects mismatched passwords without calling updateUser', async () => {
    mockVerifyOtp.mockResolvedValue({
      data: { session: { access_token: 't' }, user: { email: 'new@x.com' } },
      error: null,
    });
    renderAt('/accept-invite?token_hash=abc123');
    await userEvent.type(await screen.findByLabelText(/new password/i), 'Str0ng!Passw0rd');
    await userEvent.type(screen.getByLabelText(/confirm password/i), 'different');
    await userEvent.click(screen.getByRole('button', { name: /set password/i }));
    expect(await screen.findByText(/passwords do not match/i)).toBeInTheDocument();
    expect(mockUpdateUser).not.toHaveBeenCalled();
  });
});
