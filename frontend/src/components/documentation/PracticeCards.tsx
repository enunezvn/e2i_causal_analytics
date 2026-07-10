/**
 * PracticeCards — paired Do/Don't cards grounded in real product behavior
 * (refutation gates, Informational KPIs, honest nulls, what-if ranges,
 * per-brand scope), filterable by audience role.
 */
import { useState } from 'react';
import { Check, X } from 'lucide-react';
import { PRACTICES } from './content';
import type { PracticeRole } from './content';

type Filter = 'all' | PracticeRole;

const FILTERS: Array<{ id: Filter; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'exec', label: 'Exec' },
  { id: 'analyst', label: 'Analyst' },
];

export function PracticeCards() {
  const [filter, setFilter] = useState<Filter>('all');
  const visible = PRACTICES.filter((p) => filter === 'all' || p.roles.includes(filter));

  return (
    <div>
      <div className="mb-3 flex items-center gap-1.5" role="group" aria-label="Filter practices by role">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setFilter(f.id)}
            aria-pressed={filter === f.id}
            className={`rounded-full border px-3 py-1 text-xs font-medium transition-colors motion-reduce:transition-none ${
              filter === f.id
                ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                : 'border-[var(--color-border)] text-[var(--color-muted-foreground)] hover:text-[var(--color-foreground)]'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>
      <ul className="grid gap-3 md:grid-cols-2">
        {visible.map((p) => (
          <li key={p.id} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
            <div className="flex items-start gap-2">
              <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" aria-hidden="true" />
              <p className="text-sm leading-6 text-[var(--color-foreground)]">{p.doText}</p>
            </div>
            <div className="mt-2 flex items-start gap-2">
              <X className="mt-0.5 h-4 w-4 shrink-0 text-red-600 dark:text-red-400" aria-hidden="true" />
              <p className="text-sm leading-6 text-[var(--color-muted-foreground)]">{p.dontText}</p>
            </div>
            <p className="mt-2 border-t border-[var(--color-border)] pt-2 text-xs leading-5 text-[var(--color-muted-foreground)]">
              {p.why}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
