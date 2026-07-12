/**
 * Admin Page (/admin — admin-only via ProtectedRoute requireAdmin)
 * ================================================================
 *
 * Users tab: invite (copyable link), edit role & brand access, disable/enable,
 * guarded delete, reinvite/recovery links.
 * Activity tab: platform + per-user activity over time, admin audit feed.
 */

import { useState } from 'react';
import { useAuth } from '@/hooks/use-auth';
import { useAdminUsers } from '@/hooks/api/use-admin';
import { UsersTable } from '@/components/admin/UsersTable';
import { InviteUserDialog } from '@/components/admin/InviteUserDialog';
import { ActivityTab } from '@/components/admin/ActivityTab';
import { RoleLegend } from '@/components/admin/RoleLegend';
import { ObservabilityTab } from '@/components/admin/ObservabilityTab';

type Tab = 'users' | 'activity' | 'observability';

export default function Admin() {
  const [tab, setTab] = useState<Tab>('users');
  const [inviteOpen, setInviteOpen] = useState(false);
  const { user } = useAuth();
  const { data, isLoading, isError } = useAdminUsers();

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-[var(--color-foreground)]">
            Administration
          </h1>
          <p className="text-sm text-[var(--color-muted-foreground)]">
            Invite users, manage roles and brand access, review activity and LLM usage.
          </p>
        </div>
        {tab === 'users' && (
          <button
            type="button"
            onClick={() => setInviteOpen(true)}
            className="rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white"
          >
            Invite user
          </button>
        )}
      </div>

      <div role="tablist" className="flex gap-2 border-b border-[var(--color-border)]">
        {(['users', 'activity', 'observability'] as const).map((t) => (
          <button
            key={t}
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize ${
              tab === t
                ? 'border-b-2 border-[var(--color-primary)] text-[var(--color-foreground)]'
                : 'text-[var(--color-muted-foreground)]'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'users' && (
        <>
          <RoleLegend />
          <UsersTable
            users={data?.users ?? []}
            currentUserId={user?.id ?? ''}
            isLoading={isLoading}
            isError={isError}
          />
        </>
      )}
      {tab === 'activity' && <ActivityTab users={data?.users ?? []} />}
      {tab === 'observability' && <ObservabilityTab />}

      <InviteUserDialog open={inviteOpen} onClose={() => setInviteOpen(false)} />
    </div>
  );
}
