/**
 * ClinicalGrounding — "Business insights, grounded in clinical reality."
 * DESCRIBES the platform's external clinical knowledge integrations (UMLS,
 * OpenFDA, ClinicalTrials.gov, PubMed, ChEMBL) as static constants — this
 * page never calls those APIs itself.
 */
import { BookMarked, ShieldCheck } from 'lucide-react';
import { CLINICAL_SOURCES } from './content';

export function ClinicalGrounding() {
  const prominent = CLINICAL_SOURCES.filter((s) => s.prominent);
  const rest = CLINICAL_SOURCES.filter((s) => !s.prominent);

  return (
    <section aria-label="Business insights, grounded in clinical reality">
      <h3 className="mb-2 text-sm font-semibold text-[var(--color-foreground)]">
        Business insights, grounded in clinical reality
      </h3>
      <p className="mb-3 max-w-3xl text-xs leading-5 text-[var(--color-muted-foreground)]">
        Commercial signals only mean something inside their clinical context. E2I links entities
        through medical terminology and gates insight language against official drug labeling,
        drawing on five authoritative external sources.
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        {prominent.map((source) => (
          <div
            key={source.name}
            className="rounded-lg border border-[var(--color-primary)]/40 bg-[var(--color-primary)]/5 p-4"
          >
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-[var(--color-primary)]" aria-hidden="true" />
              <span className="text-sm font-semibold text-[var(--color-foreground)]">{source.name}</span>
            </div>
            <p className="mt-1.5 text-xs leading-5 text-[var(--color-muted-foreground)]">{source.role}</p>
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        {rest.map((source) => (
          <div key={source.name} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-3">
            <div className="flex items-center gap-2">
              <BookMarked className="h-3.5 w-3.5 text-[var(--color-muted-foreground)]" aria-hidden="true" />
              <span className="text-xs font-semibold text-[var(--color-foreground)]">{source.name}</span>
            </div>
            <p className="mt-1 text-xs leading-5 text-[var(--color-muted-foreground)]">{source.role}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
