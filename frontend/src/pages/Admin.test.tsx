/**
 * Admin Page Tests — Users tab behaviors.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import Admin from './Admin';
import type { AdminUser } from '@/types/admin';

vi.mock('@/hooks/api/use-admin', () => ({
  useAdminUsers: vi.fn(),
  useUserActivity: vi.fn(),
  usePlatformActivity: vi.fn(),
  useAuditFeed: vi.fn(),
  useInviteUser: vi.fn(),
  useReinviteUser: vi.fn(),
  useRecoveryLink: vi.fn(),
  useUpdateUser: vi.fn(),
  useDisableUser: vi.fn(),
  useEnableUser: vi.fn(),
  useDeleteUser: vi.fn(),
}));
vi.mock('@/hooks/use-auth', () => ({
  useAuth: () => ({ isAdmin: true, user: { id: 'me-id', email: 'me@x.com' } }),
}));

import * as adminHooks from '@/hooks/api/use-admin';

const USERS: AdminUser[] = [
  {
    id: 'me-id',
    email: 'me@x.com',
    full_name: 'Me Admin',
    role: 'admin',
    brands: ['all'],
    status: 'active',
    created_at: '2026-02-01T00:00:00Z',
    last_sign_in_at: '2026-07-10T00:00:00Z',
    total_conversations: 5,
    total_messages: 40,
    last_active_at: '2026-07-10T00:00:00Z',
  },
  {
    id: 'u2',
    email: 'viewer@x.com',
    full_name: null,
    role: 'viewer',
    brands: ['Kisqali'],
    status: 'invited',
    created_at: '2026-07-01T00:00:00Z',
    last_sign_in_at: null,
    total_conversations: 0,
    total_messages: 0,
    last_active_at: null,
  },
];

const idleMutation = () => ({
  mutate: vi.fn(),
  mutateAsync: vi.fn(),
  isPending: false,
  data: undefined,
  error: null,
  reset: vi.fn(),
});

beforeEach(() => {
  vi.clearAllMocks();
  (adminHooks.useAdminUsers as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { users: USERS },
    isLoading: false,
    isError: false,
  });
  (adminHooks.usePlatformActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { days: [] },
    isLoading: false,
  });
  (adminHooks.useAuditFeed as ReturnType<typeof vi.fn>).mockReturnValue({
    data: { events: [] },
    isLoading: false,
  });
  (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  for (const name of [
    'useInviteUser',
    'useReinviteUser',
    'useRecoveryLink',
    'useUpdateUser',
    'useDisableUser',
    'useEnableUser',
    'useDeleteUser',
  ] as const) {
    (adminHooks[name] as ReturnType<typeof vi.fn>).mockReturnValue(idleMutation());
  }
});

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <Admin />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Admin page — Users tab', () => {
  it('renders the users table with roles and statuses', () => {
    renderPage();
    expect(screen.getByRole('heading', { name: /administration/i })).toBeInTheDocument();
    const table = screen.getByRole('table');
    expect(within(table).getByText('me@x.com')).toBeInTheDocument();
    expect(within(table).getByText('viewer@x.com')).toBeInTheDocument();
    expect(within(table).getByText(/invited/i)).toBeInTheDocument();
  });

  it('opens the invite dialog and shows the one-time link on success', async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      user_id: 'u3',
      email: 'new@x.com',
      invite_link: 'https://eznomics.site/accept-invite?token_hash=xyz',
      link_type: 'invite',
    });
    (adminHooks.useInviteUser as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idleMutation(),
      mutateAsync,
    });
    renderPage();
    await userEvent.click(screen.getByRole('button', { name: /invite user/i }));
    await userEvent.type(screen.getByLabelText(/email/i), 'new@x.com');
    await userEvent.click(screen.getByRole('button', { name: /^send invite$/i }));
    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    expect(
      await screen.findByText(/eznomics\.site\/accept-invite\?token_hash=xyz/)
    ).toBeInTheDocument();
    expect(screen.getByText(/shown once/i)).toBeInTheDocument();
  });

  it('delete requires typing the email to confirm', async () => {
    const mutateAsync = vi.fn().mockResolvedValue({ deleted: true });
    (adminHooks.useDeleteUser as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idleMutation(),
      mutateAsync,
    });
    renderPage();
    const row = screen.getByText('viewer@x.com').closest('tr')!;
    await userEvent.click(within(row).getByRole('button', { name: /delete/i }));
    const confirmBtn = screen.getByRole('button', { name: /delete permanently/i });
    expect(confirmBtn).toBeDisabled();
    await userEvent.type(screen.getByLabelText(/type the email/i), 'viewer@x.com');
    expect(confirmBtn).toBeEnabled();
    await userEvent.click(confirmBtn);
    await waitFor(() => expect(mutateAsync).toHaveBeenCalledWith('u2'));
  });

  it('does not offer delete/disable on your own row', () => {
    renderPage();
    const myRow = screen.getByText('me@x.com').closest('tr')!;
    expect(within(myRow).queryByRole('button', { name: /delete/i })).toBeNull();
    expect(within(myRow).queryByRole('button', { name: /disable/i })).toBeNull();
  });

  it('edit dialog updates role via the mutation', async () => {
    const mutateAsync = vi.fn().mockResolvedValue({ user_id: 'u2', role: 'analyst', brands: ['Kisqali'] });
    (adminHooks.useUpdateUser as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idleMutation(),
      mutateAsync,
    });
    renderPage();
    const row = screen.getByText('viewer@x.com').closest('tr')!;
    await userEvent.click(within(row).getByRole('button', { name: /edit/i }));
    await userEvent.selectOptions(screen.getByLabelText(/role/i), 'analyst');
    await userEvent.click(screen.getByRole('button', { name: /save changes/i }));
    await waitFor(() =>
      expect(mutateAsync).toHaveBeenCalledWith({
        userId: 'u2',
        body: expect.objectContaining({ role: 'analyst' }),
      })
    );
  });

  it('reinvite action surfaces the fresh link', async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      user_id: 'u2',
      email: 'viewer@x.com',
      invite_link: 'https://eznomics.site/accept-invite?token_hash=fresh',
      link_type: 'invite',
    });
    (adminHooks.useReinviteUser as ReturnType<typeof vi.fn>).mockReturnValue({
      ...idleMutation(),
      mutateAsync,
    });
    renderPage();
    const row = screen.getByText('viewer@x.com').closest('tr')!;
    await userEvent.click(within(row).getByRole('button', { name: /reinvite/i }));
    expect(
      await screen.findByText(/eznomics\.site\/accept-invite\?token_hash=fresh/)
    ).toBeInTheDocument();
  });
});
