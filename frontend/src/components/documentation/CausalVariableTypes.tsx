/**
 * CausalVariableTypes — color-coded explainer of the four variable roles
 * (treatment, mediator, outcome, confounder). Colors and shapes come from
 * CAUSAL_VARIABLE_TYPES, the same key CausalImpactDag paints its nodes with,
 * so this legend IS the DAG key.
 */
import { CAUSAL_EFFECTS_INTRO, CAUSAL_VARIABLES_LEAD, CAUSAL_VARIABLE_TYPES } from './content';

function Glyph({ color, shape }: { color: string; shape: 'circle' | 'diamond' }) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" className="shrink-0">
      {shape === 'diamond' ? (
        <polygon
          points="12,2 22,12 12,22 2,12"
          fill={color}
          fillOpacity="0.15"
          stroke={color}
          strokeWidth="2.5"
        />
      ) : (
        <circle cx="12" cy="12" r="9" fill={color} fillOpacity="0.15" stroke={color} strokeWidth="2.5" />
      )}
    </svg>
  );
}

export function CausalVariableTypes() {
  return (
    <section
      aria-labelledby="causal-variable-types-heading"
      className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4"
    >
      <h3
        id="causal-variable-types-heading"
        className="text-sm font-semibold text-[var(--color-foreground)]"
      >
        Four types of causal variables
      </h3>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--color-foreground)]">
        {CAUSAL_EFFECTS_INTRO} {CAUSAL_VARIABLES_LEAD}
      </p>
      <dl className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {CAUSAL_VARIABLE_TYPES.map((v) => (
          <div
            key={v.type}
            className="rounded-md border border-[var(--color-border)] border-l-4 bg-[var(--color-muted)]/40 px-3 py-2"
            style={{ borderLeftColor: v.color }}
          >
            <dt className="flex items-center gap-2 text-sm font-semibold" style={{ color: v.color }}>
              <Glyph color={v.color} shape={v.shape} />
              {v.label}
            </dt>
            <dd className="mt-1 text-xs leading-5 text-[var(--color-muted-foreground)]">
              {v.definition}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
