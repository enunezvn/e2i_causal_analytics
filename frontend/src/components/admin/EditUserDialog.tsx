/**
 * EditUserDialog — role select + brand checkboxes + display name.
 * Backend guards (last-admin demote, invalid combos) surface as errors.
 */

import { useState } from 'react';
import { useUpdateUser } from '@/hooks/api/use-admin';
import { ADMIN_BRANDS, ADMIN_ROLES, ROLE_DESCRIPTIONS } from '@/types/admin';
import type { AdminRole, AdminUser } from '@/types/admin';

interface EditUserDialogProps {
  user: AdminUser | null;
  onClose: () => void;
}

export function EditUserDialog({ user, onClose }: EditUserDialogProps) {
  const update = useUpdateUser();
  const [role, setRole] = useState<AdminRole>(user?.role ?? 'viewer');
  const [brands, setBrands] = useState<string[]>(user?.brands?.length ? user.brands : ['all']);
  const [fullName, setFullName] = useState(user?.full_name ?? '');
  const [error, setError] = useState<string | null>(null);

  if (!user) return null;

  function toggleBrand(brand: string) {
    setBrands((prev) => {
      if (brand === 'all') return ['all'];
      const without = prev.filter((b) => b !== 'all' && b !== brand);
      return prev.includes(brand) ? (without.length ? without : ['all']) : [...without, brand];
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await update.mutateAsync({
        userId: user!.id,
        body: {
          role,
          brands,
          ...(fullName !== (user!.full_name ?? '') ? { full_name: fullName } : {}),
        },
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Update failed');
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-6 shadow-lg">
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">
          Edit {user.email}
        </h2>
        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div>
            <label
              htmlFor="edit-role"
              className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
            >
              Role
            </label>
            <select
              id="edit-role"
              value={role}
              onChange={(e) => setRole(e.target.value as AdminRole)}
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
            >
              {ADMIN_ROLES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
            <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
              {ROLE_DESCRIPTIONS[role]}
            </p>
          </div>
          <fieldset>
            <legend className="mb-1 block text-sm font-medium text-[var(--color-foreground)]">
              Brand access
            </legend>
            <div className="flex flex-wrap gap-3">
              {ADMIN_BRANDS.map((brand) => (
                <label key={brand} className="flex items-center gap-1 text-sm text-[var(--color-foreground)]">
                  <input
                    type="checkbox"
                    checked={brands.includes(brand)}
                    onChange={() => toggleBrand(brand)}
                  />
                  {brand}
                </label>
              ))}
            </div>
          </fieldset>
          <div>
            <label
              htmlFor="edit-name"
              className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
            >
              Display name
            </label>
            <input
              id="edit-name"
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
            />
          </div>
          {error && (
            <div
              role="alert"
              className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
            >
              {error}
            </div>
          )}
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-foreground)]"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={update.isPending}
              className="rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {update.isPending ? 'Saving…' : 'Save changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
