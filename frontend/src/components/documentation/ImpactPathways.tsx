/**
 * ImpactPathways — Expected Impact, honestly framed. Mechanism-focused cards
 * with NO fabricated ROI digits (enforced by content.test.ts); each links to
 * the live page where users see their own numbers.
 */
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { IMPACT_PATHWAYS } from './content';

export function ImpactPathways() {
  return (
    <section aria-label="Expected impact pathways">
      <p className="mb-3 max-w-3xl text-sm leading-6 text-[var(--color-foreground)]">
        E2I does not promise a number — fabricated ROI figures are exactly what this platform is
        built to eliminate. It promises mechanisms, each measurable on its own live page with your
        data:
      </p>
      <ul className="grid gap-3 sm:grid-cols-2">
        {IMPACT_PATHWAYS.map((p) => (
          <li key={p.title} className="flex flex-col rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">{p.title}</h3>
            <p className="mt-1 flex-1 text-xs leading-5 text-[var(--color-muted-foreground)]">{p.mechanism}</p>
            <Link
              to={p.href}
              className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-[var(--color-primary)] hover:underline"
            >
              {p.linkLabel}
              <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
