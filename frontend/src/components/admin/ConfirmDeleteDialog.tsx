/**
 * ConfirmDeleteDialog — hard delete requires typing the exact email.
 * The backend additionally enforces no-self-delete and last-admin guards.
 */

import { useState } from 'react';
import { useDeleteUser } from '@/hooks/api/use-admin';
import type { AdminUser } from '@/types/admin';

interface ConfirmDeleteDialogProps {
  user: AdminUser | null;
  onClose: () => void;
}

export function ConfirmDeleteDialog({ user, onClose }: ConfirmDeleteDialogProps) {
  const deleteUser = useDeleteUser();
  const [typed, setTyped] = useState('');
  const [error, setError] = useState<string | null>(null);

  if (!user) return null;

  async function handleDelete() {
    setError(null);
    try {
      await deleteUser.mutateAsync(user!.id);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed');
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-6 shadow-lg">
        <h2 className="text-lg font-semibold text-red-600">Delete {user.email}?</h2>
        <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
          This permanently removes the account and its profile. Activity history is
          retained for audit. This cannot be undone.
        </p>
        <div className="mt-4">
          <label
            htmlFor="confirm-delete-email"
            className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
          >
            Type the email to confirm
          </label>
          <input
            id="confirm-delete-email"
            type="text"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            autoComplete="off"
            className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
          />
        </div>
        {error && (
          <div
            role="alert"
            className="mt-3 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
          >
            {error}
          </div>
        )}
        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-foreground)]"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={typed !== user.email || deleteUser.isPending}
            onClick={handleDelete}
            className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {deleteUser.isPending ? 'Deleting…' : 'Delete permanently'}
          </button>
        </div>
      </div>
    </div>
  );
}
