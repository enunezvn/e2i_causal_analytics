/**
 * AgentTierStack — the 6-tier / 21-agent architecture. The corrected
 * successor to the deleted kpi/AgenticMethodology.tsx; roster source:
 * src/agents/factory.py AGENT_REGISTRY_CONFIG (content.test.ts enforces 21/6).
 */
import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { AGENT_TIERS } from './content';

export function AgentTierStack() {
  const [openTier, setOpenTier] = useState<number | null>(null);

  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold text-[var(--color-foreground)]">
        The agent system: 21 agents in 6 tiers
      </h3>
      <div className="space-y-2">
        {AGENT_TIERS.map((tier) => {
          const open = openTier === tier.tier;
          return (
            <div key={tier.tier} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
              <button
                type="button"
                onClick={() => setOpenTier(open ? null : tier.tier)}
                aria-expanded={open}
                className="flex w-full items-center justify-between px-4 py-2.5 text-left"
              >
                <span className="flex items-baseline gap-2">
                  <span className="text-xs font-mono text-[var(--color-muted-foreground)]">T{tier.tier}</span>
                  <span className="text-sm font-medium text-[var(--color-foreground)]">{tier.name}</span>
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    ({tier.agents.length} agent{tier.agents.length > 1 ? 's' : ''})
                  </span>
                </span>
                <ChevronDown
                  className={`h-4 w-4 text-[var(--color-muted-foreground)] transition-transform motion-reduce:transition-none ${open ? 'rotate-180' : ''}`}
                  aria-hidden="true"
                />
              </button>
              {open && (
                <div className="border-t border-[var(--color-border)] px-4 py-3">
                  <p className="mb-2 text-xs text-[var(--color-muted-foreground)]">{tier.blurb}</p>
                  <ul className="grid gap-1.5 sm:grid-cols-2">
                    {tier.agents.map((agent) => (
                      <li key={agent.id} className="text-xs leading-5">
                        <span className="font-mono text-[var(--color-foreground)]">{agent.id}</span>
                        <span className="text-[var(--color-muted-foreground)]"> — {agent.role}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
