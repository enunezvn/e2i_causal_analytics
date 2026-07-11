/**
 * ActivityTab Tests — platform chart, per-user drill-down, audit feed.
 * recharts renders SVG in jsdom; assertions target section headings, stats,
 * and hook wiring (charts themselves are visual — verified live at the end).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { AdminUser } from '@/types/admin';

vi.mock('@/hooks/api/use-admin', () => ({
  usePlatformActivity: vi.fn(),
  useUserActivity: vi.fn(),
  useAuditFeed: vi.fn(),
}));

import * as adminHooks from '@/hooks/api/use-admin';
import { ActivityTab } from './ActivityTab';

const USERS = [
  { id: 'u1', email: 'a@x.com' } as AdminUser,
  { id: 'u2', email: 'b@x.com' } as AdminUser,
];

beforeEach(() => {
  vi.clearAllMocks();
  (adminHooks.usePlatformActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: {
      days: [
        { day: '2026-07-01', logins: 4, active_users: 3 },
        { day: '2026-07-02', logins: 6, active_users: 5 },
      ],
    },
    isLoading: false,
  });
  (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
    data: undefined,
    isLoading: false,
  });
  (adminHooks.useAuditFeed as ReturnType<typeof vi.fn>).mockReturnValue({
    data: {
      events: [
        {
          event_id: 'e1',
          event_type: 'admin.user.modified',
          severity: 'warning',
          timestamp: '2026-07-11T10:00:00Z',
          message: 'Invited new@x.com as viewer',
          user_email: 'me@x.com',
          resource_id: 'u9',
          metadata: {},
        },
      ],
    },
    isLoading: false,
  });
});

describe('ActivityTab', () => {
  it('renders platform activity section with data', () => {
    render(<ActivityTab users={USERS} />);
    expect(screen.getByRole('heading', { name: /platform activity/i })).toBeInTheDocument();
  });

  it('prompts to select a user before drilling down', () => {
    render(<ActivityTab users={USERS} />);
    expect(screen.getByText(/select a user to see their activity/i)).toBeInTheDocument();
  });

  it('drills into a selected user', async () => {
    (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        user_id: 'u2',
        email: 'b@x.com',
        auth_events: [{ day: '2026-07-01', event_type: 'login', event_count: 2 }],
        api_activity: [
          {
            endpoint_group: 'causal',
            http_method: 'GET',
            bucket_minute: '2026-07-01T10:00:00Z',
            request_count: 7,
          },
        ],
        recent_events: [{ occurred_at: '2026-07-01T10:00:00Z', action: 'login' }],
        chat: { total_conversations: 1, total_messages: 9, last_active_at: null },
      },
      isLoading: false,
    });
    render(<ActivityTab users={USERS} />);
    await userEvent.selectOptions(screen.getByLabelText(/select user/i), 'u2');
    expect(adminHooks.useUserActivity).toHaveBeenLastCalledWith('u2', expect.any(Number));
    expect(screen.getByRole('heading', { name: /user activity/i })).toBeInTheDocument();
    expect(screen.getByText('9')).toBeInTheDocument(); // chat messages stat
    expect(screen.getByText(/chat messages/i)).toBeInTheDocument();
  });

  it('renders the admin audit feed', () => {
    render(<ActivityTab users={USERS} />);
    expect(screen.getByRole('heading', { name: /admin audit/i })).toBeInTheDocument();
    expect(screen.getByText(/invited new@x\.com as viewer/i)).toBeInTheDocument();
  });

  it('shows the accrual note when a user has no API activity yet', async () => {
    (adminHooks.useUserActivity as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        user_id: 'u1',
        email: 'a@x.com',
        auth_events: [],
        api_activity: [],
        recent_events: [],
        chat: { total_conversations: 0, total_messages: 0, last_active_at: null },
      },
      isLoading: false,
    });
    render(<ActivityTab users={USERS} />);
    await userEvent.selectOptions(screen.getByLabelText(/select user/i), 'u1');
    expect(screen.getByText(/no api activity recorded yet/i)).toBeInTheDocument();
  });
});
