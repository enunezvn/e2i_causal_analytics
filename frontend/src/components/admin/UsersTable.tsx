/**
 * UsersTable — merged auth+profile user list with per-row admin actions.
 * Self-row hides delete/disable (footgun removal; the backend guards are the
 * real enforcement). Reinvite shows for invited users, recovery for active.
 */

import { useState } from 'react';
import {
  useDisableUser,
  useEnableUser,
  useRecoveryLink,
  useReinviteUser,
} from '@/hooks/api/use-admin';
import type { AdminUser, LinkResponse } from '@/types/admin';
import { ConfirmDeleteDialog } from './ConfirmDeleteDialog';
import { EditUserDialog } from './EditUserDialog';
import { LinkDialog } from './LinkDialog';

interface UsersTableProps {
  users: AdminUser[];
  currentUserId: string;
  isLoading: boolean;
  isError: boolean;
}

const STATUS_STYLES: Record<AdminUser['status'], string> = {
  active: 'bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-300',
  invited: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
  disabled: 'bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300',
};

const ROLE_STYLES: Record<string, string> = {
  admin: 'bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300',
  operator: 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300',
  analyst: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-950 dark:text-cyan-300',
  viewer: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300',
};

function formatDate(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleDateString();
}

export function UsersTable({ users, currentUserId, isLoading, isError }: UsersTableProps) {
  const reinvite = useReinviteUser();
  const recovery = useRecoveryLink();
  const disable = useDisableUser();
  const enable = useEnableUser();

  const [editUser, setEditUser] = useState<AdminUser | null>(null);
  const [deleteUser, setDeleteUser] = useState<AdminUser | null>(null);
  const [link, setLink] = useState<LinkResponse | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function run<T>(fn: () => Promise<T>, onDone?: (r: T) => void) {
    setActionError(null);
    try {
      const r = await fn();
      onDone?.(r);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Action failed');
    }
  }

  if (isLoading) {
    return <p className="text-sm text-[var(--color-muted-foreground)]">Loading users…</p>;
  }
  if (isError) {
    return (
      <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
        Failed to load users. Refresh to retry.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
      {actionError && (
        <div role="alert" className="border-b border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
          {actionError}
        </div>
      )}
      <table className="w-full text-left text-sm">
        <thead className="border-b border-[var(--color-border)] bg-[var(--color-muted)] text-xs uppercase text-[var(--color-muted-foreground)]">
          <tr>
            <th className="px-4 py-3">User</th>
            <th className="px-4 py-3">Role</th>
            <th className="px-4 py-3">Brands</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Last sign-in</th>
            <th className="px-4 py-3">Chat msgs</th>
            <th className="px-4 py-3">Created</th>
            <th className="px-4 py-3">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--color-border)]">
          {users.map((user) => {
            const isSelf = user.id === currentUserId;
            return (
              <tr key={user.id}>
                <td className="px-4 py-3">
                  <div className="font-medium text-[var(--color-foreground)]">{user.email}</div>
                  {user.full_name && (
                    <div className="text-xs text-[var(--color-muted-foreground)]">
                      {user.full_name}
                      {isSelf ? ' (you)' : ''}
                    </div>
                  )}
                  {!user.full_name && isSelf && (
                    <div className="text-xs text-[var(--color-muted-foreground)]">(you)</div>
                  )}
                </td>
                <td className="px-4 py-3">
                  <span className={`rounded-full px-2 py-1 text-xs font-medium ${ROLE_STYLES[user.role] ?? ROLE_STYLES.viewer}`}>
                    {user.role}
                  </span>
                </td>
                <td className="px-4 py-3 text-[var(--color-muted-foreground)]">
                  {user.brands.length ? user.brands.join(', ') : '—'}
                </td>
                <td className="px-4 py-3">
                  <span className={`rounded-full px-2 py-1 text-xs font-medium ${STATUS_STYLES[user.status]}`}>
                    {user.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-[var(--color-muted-foreground)]">
                  {formatDate(user.last_sign_in_at)}
                </td>
                <td className="px-4 py-3 text-[var(--color-muted-foreground)]">
                  {user.total_messages}
                </td>
                <td className="px-4 py-3 text-[var(--color-muted-foreground)]">
                  {formatDate(user.created_at)}
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => setEditUser(user)}
                      className="text-xs font-medium text-[var(--color-primary)] hover:underline"
                    >
                      Edit
                    </button>
                    {user.status === 'invited' && (
                      <button
                        type="button"
                        onClick={() => run(() => reinvite.mutateAsync(user.id), setLink)}
                        className="text-xs font-medium text-[var(--color-primary)] hover:underline"
                      >
                        Reinvite
                      </button>
                    )}
                    {user.status === 'active' && (
                      <button
                        type="button"
                        onClick={() => run(() => recovery.mutateAsync(user.id), setLink)}
                        className="text-xs font-medium text-[var(--color-primary)] hover:underline"
                      >
                        Recovery link
                      </button>
                    )}
                    {!isSelf && user.status !== 'disabled' && (
                      <button
                        type="button"
                        onClick={() => run(() => disable.mutateAsync(user.id))}
                        className="text-xs font-medium text-amber-600 hover:underline"
                      >
                        Disable
                      </button>
                    )}
                    {user.status === 'disabled' && (
                      <button
                        type="button"
                        onClick={() => run(() => enable.mutateAsync(user.id))}
                        className="text-xs font-medium text-green-600 hover:underline"
                      >
                        Enable
                      </button>
                    )}
                    {!isSelf && (
                      <button
                        type="button"
                        onClick={() => setDeleteUser(user)}
                        className="text-xs font-medium text-red-600 hover:underline"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <EditUserDialog
        key={editUser?.id ?? 'no-edit'}
        user={editUser}
        onClose={() => setEditUser(null)}
      />
      <ConfirmDeleteDialog
        key={deleteUser?.id ?? 'no-delete'}
        user={deleteUser}
        onClose={() => setDeleteUser(null)}
      />
      <LinkDialog link={link} onClose={() => setLink(null)} />
    </div>
  );
}
