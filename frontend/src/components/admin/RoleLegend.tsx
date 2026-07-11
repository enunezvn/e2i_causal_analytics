/**
 * RoleLegend — collapsible explanation of the four user types on the Users
 * tab. Descriptions mirror the backend RBAC hierarchy (types/admin.ts).
 */

import { ADMIN_ROLES, ROLE_DESCRIPTIONS } from '@/types/admin';

const ROLE_BADGES: Record<string, string> = {
  admin: 'bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300',
  operator: 'bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300',
  analyst: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-950 dark:text-cyan-300',
  viewer: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300',
};

export function RoleLegend() {
  return (
    <details className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] px-4 py-3">
      <summary className="cursor-pointer text-sm font-medium text-[var(--color-foreground)]">
        What do the user types mean?
      </summary>
      <div className="mt-3 space-y-2">
        <p className="text-xs text-[var(--color-muted-foreground)]">
          Roles are hierarchical — each role includes everything the roles below it can do.
          Brand access is set separately and limits which brands&apos; data a user sees.
        </p>
        <dl className="space-y-2">
          {[...ADMIN_ROLES].reverse().map((role) => (
            <div key={role} className="flex items-baseline gap-3">
              <dt className="shrink-0">
                <span
                  className={`rounded-full px-2 py-1 text-xs font-medium capitalize ${ROLE_BADGES[role]}`}
                >
                  {role}
                </span>
              </dt>
              <dd className="text-sm text-[var(--color-foreground)]">
                {ROLE_DESCRIPTIONS[role]}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </details>
  );
}
