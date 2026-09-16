/**
 * ToolComposerSection — how the tool composer actually behaved in the window.
 * GET /api/admin/observability/tool-composer (spec 2026-09-11 §8).
 *
 * Honesty rules, the same ones the backend enforces:
 * - the verdict word leads every tool row; the counts are there to explain it, not to be read
 *   past it;
 * - a tool without enough health runs says "Too few runs to judge (n=k)" rather than showing a
 *   rate computed from nothing;
 * - measured latency renders "—" until there are 20 successful runs; the registry's declared
 *   number is shown separately and labelled, never substituted into the measured column;
 * - a failed composition names the step classes that caused it, not just that it failed.
 */
import { useToolComposerObservability } from '@/hooks/api/use-admin';
import type { ToolComposerRecentFailure, ToolComposerToolRow, ToolVerdict } from '@/types/admin';

const TH = 'px-3 py-2 text-left text-xs font-medium text-[var(--color-muted-foreground)]';
const TD = 'px-3 py-2 text-sm text-[var(--color-foreground)]';

const fmtInt = (n: number) => n.toLocaleString();
const fmtMs = (n: number | null | undefined) => (n == null ? '—' : `${Math.round(n)} ms`);

// The secondary line under a tool's refused count: null when there is nothing to add.
//
// A post-ml/043 backend always sends these four keys, so a tool with no coded refusal sends an
// explicit `null` rather than omitting the key — this MUST stay a falsy check (not `!==
// undefined`), or an explicit null renders "most common: null".
function refusalSummary(tool: ToolComposerToolRow): string | null {
  if (!tool.most_common_refusal_reason) {
    return null;
  }
  const text = tool.most_common_refusal_sentence ?? tool.most_common_refusal_reason;
  const k = tool.n_most_common_refusal_reason;
  const n = tool.n_refused_coded;
  // Both counts travel with the code so it is never read as representative of every refusal.
  // Either being null means UNKNOWN (pre-ml/043), not 0 — the whole parenthetical, uncoded
  // remainder included, is omitted rather than showing "0 of n" or a count computed from an
  // unknown. A real 0 (e.g. k=0) is a number, not null, and DOES render.
  if (k == null || n == null) {
    return `most common: ${text}`;
  }
  const coded = `${fmtInt(k)} of ${fmtInt(n)} coded`;
  // A minority code must not read as representative of every refusal: name the uncoded
  // remainder too, unless there is none. Both counts share the same filter (ml/043), so this
  // should never go negative — guard it anyway and treat a negative remainder like zero.
  const uncoded = tool.n_refused - n;
  const suffix = uncoded > 0 ? `${coded}; ${fmtInt(uncoded)} uncoded` : coded;
  return `most common: ${text} (${suffix})`;
}

const VERDICT_LABEL: Record<ToolVerdict, string> = {
  caveat: 'Caveat',
  reliable: 'Reliable',
  inconclusive: 'Inconclusive',
  too_few_runs: 'Too few runs',
  no_runs: 'No runs',
};

const VERDICT_CLASS: Record<ToolVerdict, string> = {
  caveat: 'text-amber-700 dark:text-amber-400',
  reliable: 'text-emerald-700 dark:text-emerald-400',
  inconclusive: 'text-[var(--color-foreground)]',
  too_few_runs: 'text-[var(--color-muted-foreground)]',
  no_runs: 'text-[var(--color-muted-foreground)]',
};

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] p-4">
      <p className="text-xs text-[var(--color-muted-foreground)]">{label}</p>
      <p className="mt-1 text-xl font-semibold text-[var(--color-foreground)]">{value}</p>
    </div>
  );
}

// The numeric details recorded with a refusal, as "(key=value, …)": null when there are none.
// Two refusals under one code share a sentence, so these are what tell them apart (#2050). Keys
// are sorted so the same details always read the same way. Values render exactly and locale-free
// (never fmtInt): under de-DE a grouped 12345 reads "12.345", a decimal, and en-US's "," group
// separator collides with the ", " pair join.
function detailsSuffix(details: Record<string, number | boolean> | undefined): string | null {
  const keys = Object.keys(details ?? {}).sort();
  if (!details || keys.length === 0) {
    return null;
  }
  const parts = keys.map((key) => `${key}=${String(details[key])}`);
  return `(${parts.join(', ')})`;
}

function StepClasses({ failure }: { failure: ToolComposerRecentFailure }) {
  if (failure.step_classes.length === 0) {
    return <span className="text-[var(--color-muted-foreground)]">no step recorded</span>;
  }
  return (
    <>
      {failure.step_classes.map((step, index) => {
        // Prefer the catalogue sentence; a non-null code with a null sentence is a code this
        // build does not know, so show the bare code rather than inventing a sentence for it.
        const reason = step.reason ?? step.reason_code;
        const details = detailsSuffix(step.reason_details);
        return (
          <span key={`${failure.composition_id}-${step.step_number ?? index}`}>
            {/* "; " not ", ": several catalogue sentences contain commas, so a comma join would
                make a step boundary indistinguishable from punctuation inside a sentence. */}
            {index > 0 && '; '}
            {step.tool_name ?? 'unknown'}: {(step.outcome_class ?? 'unknown').replace(/_/g, ' ')}
            {reason ? ` — ${reason}` : ''}
            {details ? ` ${details}` : ''}
          </span>
        );
      })}
    </>
  );
}

export function ToolComposerSection({ days }: { days: number }) {
  const { data, isLoading, isError } = useToolComposerObservability(days);

  if (isLoading) {
    return (
      <p className="p-6 text-sm text-[var(--color-muted-foreground)]">
        Loading tool composer activity…
      </p>
    );
  }
  if (isError || !data) {
    return (
      <p className="p-6 text-sm text-[var(--color-muted-foreground)]">
        Failed to load tool composer activity.
      </p>
    );
  }

  const { compositions, tools, recent_failures: recentFailures } = data;

  return (
    <section className="space-y-4" data-testid="tool-composer-section">
      <div>
        <h3 className="text-sm font-medium text-[var(--color-foreground)]">Tool composer</h3>
        <p className="text-sm text-[var(--color-muted-foreground)]">
          Compositions recorded in this window, each tool&apos;s reliability verdict, and what
          recently went wrong.
          {!data.include_synthetic && ' Synthetic-substrate runs are excluded.'}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <StatCard label="Compositions" value={fmtInt(compositions.total)} />
        <StatCard label="Success" value={fmtInt(compositions.success)} />
        <StatCard label="Partial" value={fmtInt(compositions.partial)} />
        <StatCard label="Failed" value={fmtInt(compositions.failed)} />
        <StatCard label="Cancelled" value={fmtInt(compositions.cancelled)} />
        {/* Unfinished is what is left once the four outcomes are counted: without it the cards
            do not add up to the total, and a composition still in flight is invisible. */}
        <StatCard label="Unfinished" value={fmtInt(compositions.unfinished)} />
        <StatCard label="Abandoned" value={fmtInt(compositions.abandoned)} />
      </div>

      <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
        <table className="w-full">
          <thead className="bg-[var(--color-muted)]">
            <tr>
              <th className={TH}>Verdict</th>
              <th className={TH}>Tool</th>
              <th className={TH}>Invoked</th>
              <th className={TH}>Succeeded</th>
              <th className={TH}>Refused</th>
              <th className={TH}>Health failures</th>
              <th className={TH}>Measured p50 / p95</th>
              <th className={TH}>Declared</th>
            </tr>
          </thead>
          <tbody>
            {tools.map((tool) => {
              const refusalNote = refusalSummary(tool);
              return (
                <tr key={tool.tool_name} className="border-b border-[var(--color-border)]">
                  <td className={`${TD} font-medium ${VERDICT_CLASS[tool.verdict]}`}>
                    {tool.verdict === 'too_few_runs'
                      ? `Too few runs to judge (n=${tool.n_health})`
                      : VERDICT_LABEL[tool.verdict]}
                  </td>
                  <td className={TD}>{tool.tool_name}</td>
                  <td className={TD}>{fmtInt(tool.n_invoked)}</td>
                  <td className={TD}>{fmtInt(tool.n_succeeded)}</td>
                  <td className={TD}>
                    {fmtInt(tool.n_refused)}
                    {refusalNote && (
                      <div className="text-xs text-[var(--color-muted-foreground)]">
                        {refusalNote}
                      </div>
                    )}
                  </td>
                  <td className={TD}>
                    {fmtInt(tool.n_health_failures)}
                    {tool.most_common_health_error ? ` (${tool.most_common_health_error})` : ''}
                  </td>
                  <td className={TD}>
                    {tool.p50_latency_ms == null && tool.p95_latency_ms == null
                      ? '—'
                      : `${fmtMs(tool.p50_latency_ms)} / ${fmtMs(tool.p95_latency_ms)}`}
                  </td>
                  <td className={`${TD} text-[var(--color-muted-foreground)]`}>
                    {fmtMs(tool.declared_latency_ms)} declared
                  </td>
                </tr>
              );
            })}
            {tools.length === 0 && (
              <tr>
                <td
                  colSpan={8}
                  className={`${TD} text-center text-[var(--color-muted-foreground)]`}
                >
                  No tool runs recorded in this window.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div>
        <h4 className="mb-2 text-sm font-medium text-[var(--color-foreground)]">
          Recent failures
        </h4>
        {recentFailures.length === 0 ? (
          <p className="rounded-lg border border-dashed border-[var(--color-border)] p-4 text-center text-sm text-[var(--color-muted-foreground)]">
            No failed compositions in this window.
          </p>
        ) : (
          <ul className="space-y-2">
            {recentFailures.map((failure) => (
              <li
                key={failure.composition_id}
                data-testid={`recent-failure-${failure.composition_id}`}
                className="rounded-lg border border-[var(--color-border)] p-3 text-sm"
              >
                <p className="text-[var(--color-foreground)]">
                  <span className="font-medium">{failure.outcome ?? failure.status}</span>
                  {failure.failed_phase ? ` in ${failure.failed_phase}` : ''}
                  {failure.error_type ? ` (${failure.error_type})` : ''}
                </p>
                {/* break-words: a long detail key has no break opportunity and would overflow. */}
                <p className="break-words text-[var(--color-muted-foreground)]">
                  <StepClasses failure={failure} />
                </p>
                <p
                  data-testid="query-preview"
                  className="mt-1 truncate text-xs text-[var(--color-muted-foreground)]"
                >
                  {failure.query_preview}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
