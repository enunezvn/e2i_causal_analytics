/**
 * Documentation Page — "Understanding E2I"
 * ========================================
 *
 * Scroll narrative in four sections: Purpose, Methodology, Best Practices,
 * Expected Impact. Spec: docs/superpowers/specs/2026-07-10-documentation-page-design.md
 *
 * Honesty constraints (platform-wide):
 * - ONE network call (useKPIList) for the live KPI-count chip; on error the
 *   chip silently disappears — no error UI on a docs page.
 * - No fabricated performance/ROI digits anywhere; illustrative content is
 *   visually labeled "illustrative".
 */
import { useKPIList } from '@/hooks/api/use-kpi';
import {
  SectionNav,
  CausalScopeMap,
  CorrelationCausationToggle,
  CapabilityIndex,
  CausalPipeline,
  AgentTierStack,
  ClinicalGrounding,
  PracticeCards,
  ImpactPathways,
  DOC_SECTIONS,
  STAT_CHIPS,
} from '@/components/documentation';

function Section({ id, title, children }: { id: string; title: string; children?: React.ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-heading`} className="scroll-mt-28 space-y-6 pb-12">
      <h2 id={`${id}-heading`} className="text-xl font-semibold text-[var(--color-foreground)]">
        {title}
      </h2>
      {children}
    </section>
  );
}

function StatChipView({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] px-4 py-3 text-center">
      <div className="text-2xl font-bold text-[var(--color-foreground)]">{value}</div>
      <div className="text-xs text-[var(--color-muted-foreground)]">{label}</div>
    </div>
  );
}

export function Documentation() {
  // Live chip: renders only on success with a positive count; degrades to
  // absence (never a spinner, error banner, or placeholder number).
  const { data: kpiData } = useKPIList(undefined, { retry: false });
  const kpiTotal = kpiData?.total;
  const showLiveChip = typeof kpiTotal === 'number' && kpiTotal > 0;

  return (
    <div className="space-y-6 px-1">
      <div>
        <h1 className="text-2xl font-bold text-[var(--color-foreground)]">Understanding E2I</h1>
        <p className="text-[var(--color-muted-foreground)]">
          What this platform is for, how its causal methodology works, how to use it well, and what
          impact to expect.
        </p>
      </div>

      <SectionNav sections={DOC_SECTIONS} />

      <Section id="purpose" title="Purpose — why E2I exists">
        <p className="max-w-3xl text-sm leading-6 text-[var(--color-foreground)]">
          Commercial pharma teams are surrounded by correlations: calls correlate with
          prescriptions, programs correlate with adoption. Correlation is cheap — and often wrong
          about what to do next. E2I applies formal causal inference, checked by adversarial
          refutation tests and grounded in clinical context, to answer the question that matters:{' '}
          <em>what actually causes the outcomes we care about?</em> It operates at three linked
          levels — HCP prescribing behavior, patient journey outcomes, and market &amp; brand
          performance.
        </p>
        <div
          className={`grid grid-cols-2 gap-3 sm:grid-cols-4 md:max-w-3xl ${
            showLiveChip
              ? 'md:[grid-template-columns:repeat(5,minmax(0,1fr))]'
              : 'md:[grid-template-columns:repeat(4,minmax(0,1fr))]'
          }`}
        >
          {STAT_CHIPS.map((chip) => (
            <StatChipView key={chip.label} value={chip.value} label={chip.label} />
          ))}
          {showLiveChip && <StatChipView value={String(kpiTotal)} label="governed KPIs" />}
        </div>
        <CausalScopeMap />
        <CorrelationCausationToggle />
        <CapabilityIndex />
      </Section>

      <Section id="methodology" title="Methodology — how it works">
        <CausalPipeline />
        <AgentTierStack />
        <ClinicalGrounding />
      </Section>

      <Section id="practices" title="Best Practices — using E2I well">
        <PracticeCards />
      </Section>

      <Section id="impact" title="Expected Impact — what good looks like">
        <ImpactPathways />
      </Section>
    </div>
  );
}

export default Documentation;
