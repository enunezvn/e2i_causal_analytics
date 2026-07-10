/**
 * CausalScopeMap — the three linked causal levels E2I operates on.
 * Node labels are drawn verbatim from the causal registry's _NODE_LABELS
 * (src/insights/causal_context.py): every node shown is a modeled node.
 */
import { useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { SCOPE_LEVELS } from './content';
import type { CausalLevel } from './content';

const LEVEL_STYLES: Record<CausalLevel, { active: string; dot: string }> = {
  hcp: { active: 'border-blue-500/60 bg-blue-500/5', dot: 'bg-blue-500' },
  patient: { active: 'border-emerald-500/60 bg-emerald-500/5', dot: 'bg-emerald-500' },
  market: { active: 'border-purple-500/60 bg-purple-500/5', dot: 'bg-purple-500' },
};

export function CausalScopeMap() {
  const [selected, setSelected] = useState<CausalLevel>('hcp');
  const active = SCOPE_LEVELS.find((l) => l.id === selected) ?? SCOPE_LEVELS[0];

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="flex flex-col items-stretch">
        {SCOPE_LEVELS.map((level, i) => (
          <div key={level.id} className="flex flex-col items-center">
            {i > 0 && (
              <ArrowDown className="my-1 h-4 w-4 text-[var(--color-muted-foreground)]" aria-hidden="true" />
            )}
            <button
              type="button"
              onClick={() => setSelected(level.id)}
              aria-pressed={selected === level.id}
              className={`w-full rounded-lg border px-4 py-3 text-left transition-colors motion-reduce:transition-none ${
                selected === level.id
                  ? LEVEL_STYLES[level.id].active
                  : 'border-[var(--color-border)] bg-[var(--color-card)] hover:border-[var(--color-muted-foreground)]/40'
              }`}
            >
              <span className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${LEVEL_STYLES[level.id].dot}`} aria-hidden="true" />
                <span className="text-sm font-medium text-[var(--color-foreground)]">{level.title}</span>
              </span>
            </button>
          </div>
        ))}
      </div>
      <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
        <p className="text-sm leading-6 text-[var(--color-foreground)]">{active.summary}</p>
        <p className="mt-3 text-xs font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]">
          Modeled nodes at this level
        </p>
        <ul className="mt-2 flex flex-wrap gap-1.5">
          {active.nodes.map((node) => (
            <li
              key={node}
              className="rounded-full border border-[var(--color-border)] px-2.5 py-0.5 text-xs text-[var(--color-muted-foreground)]"
            >
              {node}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
