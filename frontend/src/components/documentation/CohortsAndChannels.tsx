/**
 * CohortsAndChannels — Purpose-section explainer for the two nouns the stat
 * chips count: the four predictive cohorts and the eight intervention channels.
 *
 * Content is data-driven from content.ts (PREDICTIVE_COHORTS,
 * INTERVENTION_CHANNELS), which are pinned to the backend enums by
 * tests/unit/test_docs/test_documentation_cohorts_channels_ssot.py — so this
 * component never states a cohort or channel the platform does not evaluate.
 */
import {
  INTERVENTION_CHANNELS,
  INTERVENTION_CHANNEL_INTRO,
  PREDICTIVE_COHORTS,
  PREDICTIVE_COHORT_INTRO,
  type InterventionChannelKind,
} from './content';

const CARD =
  'rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4 space-y-3';
const PILL =
  'rounded-full border border-[var(--color-border)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--color-muted-foreground)]';

const CHANNEL_GROUPS: { kind: InterventionChannelKind; title: string; note: string }[] = [
  { kind: 'hcp', title: 'HCP-level channels', note: 'each varies one exposure a prescriber receives' },
  {
    kind: 'program',
    title: 'Program-level levers',
    note: 'modeled as HCP-level proxies of a program the HCP is exposed to',
  },
];

function CohortsCard() {
  return (
    <section aria-labelledby="predictive-cohorts-heading" className={CARD}>
      <h3 id="predictive-cohorts-heading" className="text-sm font-semibold text-[var(--color-foreground)]">
        Four predictive cohorts
      </h3>
      <p className="text-sm leading-6 text-[var(--color-foreground)]">{PREDICTIVE_COHORT_INTRO}</p>
      <ol className="space-y-2">
        {PREDICTIVE_COHORTS.map((c) => (
          <li
            key={c.id}
            data-cohort={c.id}
            className="rounded-md border border-[var(--color-border)] px-3 py-2 text-sm"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-medium text-[var(--color-foreground)]">{c.name}</span>
              <span className={PILL}>{c.entity}</span>
            </div>
            <p className="mt-1 text-[var(--color-muted-foreground)]">
              Probability of <span className="text-[var(--color-foreground)]">{c.outcome}</span>
            </p>
            <code className="block text-[11px] text-[var(--color-muted-foreground)]">{c.labelColumn}</code>
          </li>
        ))}
      </ol>
    </section>
  );
}

function ChannelsCard() {
  return (
    <section aria-labelledby="intervention-channels-heading" className={CARD}>
      <h3
        id="intervention-channels-heading"
        className="text-sm font-semibold text-[var(--color-foreground)]"
      >
        Eight intervention channels
      </h3>
      <p className="text-sm leading-6 text-[var(--color-foreground)]">{INTERVENTION_CHANNEL_INTRO}</p>
      {CHANNEL_GROUPS.map((g) => {
        const items = INTERVENTION_CHANNELS.filter((c) => c.kind === g.kind);
        return (
          <div key={g.kind} className="space-y-1.5">
            <h4 className="flex flex-wrap items-baseline gap-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
              {g.title} ({items.length})
              <span className="text-[11px] font-normal normal-case tracking-normal">— {g.note}</span>
            </h4>
            <ul className="grid gap-1.5 sm:grid-cols-2">
              {items.map((c) => (
                <li
                  key={c.id}
                  data-channel={c.id}
                  data-kind={c.kind}
                  className="rounded-md border border-[var(--color-border)] px-3 py-2 text-sm"
                >
                  <span className="font-medium text-[var(--color-foreground)]">{c.name}</span>
                  <p className="text-xs text-[var(--color-muted-foreground)]">lever: {c.lever}</p>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </section>
  );
}

export function CohortsAndChannels() {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <CohortsCard />
      <ChannelsCard />
    </div>
  );
}
