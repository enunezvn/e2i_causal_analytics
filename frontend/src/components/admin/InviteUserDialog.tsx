/**
 * InviteUserDialog — email + name + role + brands form. On success shows the
 * one-time copyable invite link (via LinkDialog states inline).
 */

import { useState } from 'react';
import { useInviteUser } from '@/hooks/api/use-admin';
import { ADMIN_BRANDS, ADMIN_ROLES } from '@/types/admin';
import type { AdminRole, LinkResponse } from '@/types/admin';
import { LinkDialog } from './LinkDialog';

interface InviteUserDialogProps {
  open: boolean;
  onClose: () => void;
}

export function InviteUserDialog({ open, onClose }: InviteUserDialogProps) {
  const invite = useInviteUser();
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [role, setRole] = useState<AdminRole>('viewer');
  const [brands, setBrands] = useState<string[]>(['all']);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LinkResponse | null>(null);

  if (!open) return null;

  if (result) {
    return (
      <LinkDialog
        link={result}
        onClose={() => {
          setResult(null);
          setEmail('');
          setFullName('');
          setRole('viewer');
          setBrands(['all']);
          onClose();
        }}
      />
    );
  }

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
      const res = await invite.mutateAsync({
        email,
        role,
        brands,
        ...(fullName ? { full_name: fullName } : {}),
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invite failed');
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="w-full max-w-md rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-6 shadow-lg">
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">Invite user</h2>
        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <div>
            <label
              htmlFor="invite-email"
              className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
            >
              Email
            </label>
            <input
              id="invite-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
            />
          </div>
          <div>
            <label
              htmlFor="invite-name"
              className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
            >
              Full name (optional)
            </label>
            <input
              id="invite-name"
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
            />
          </div>
          <div>
            <label
              htmlFor="invite-role"
              className="mb-1 block text-sm font-medium text-[var(--color-foreground)]"
            >
              Role
            </label>
            <select
              id="invite-role"
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
              disabled={invite.isPending}
              className="rounded-md bg-[var(--color-primary)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              {invite.isPending ? 'Sending…' : 'Send invite'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
