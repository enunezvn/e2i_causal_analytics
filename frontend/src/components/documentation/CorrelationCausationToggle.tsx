/**
 * CorrelationCausationToggle — the platform's core pitch, made visible.
 * View A: a convincing spurious scatter ("calls correlate with TRx").
 * View B: the confounder (specialty) revealed as a small DAG.
 * All coordinates are hand-authored and labeled "illustrative example" —
 * they are NOT real metrics (platform honesty discipline).
 */
import { useState } from 'react';

// Hand-authored scatter: two specialty clusters that create an overall upward
// trend even though within each cluster the relationship is flat.
const SPECIALISTS: Array<[number, number]> = [
  [120, 40], [135, 44], [150, 38], [165, 45], [180, 41], [195, 46],
];
const GENERALISTS: Array<[number, number]> = [
  [30, 95], [45, 99], [60, 93], [75, 100], [90, 96], [105, 101],
];

function Dot({ x, y, cls }: { x: number; y: number; cls: string }) {
  return <circle cx={x} cy={y} r={4} className={cls} />;
}

export function CorrelationCausationToggle() {
  const [revealed, setRevealed] = useState(false);

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-[var(--color-foreground)]">
          {revealed ? 'Causation: specialty drives both' : 'Correlation: calls correlate with TRx'}
        </h3>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-amber-500/50 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-600 dark:text-amber-400">
            Illustrative example
          </span>
          <button
            type="button"
            onClick={() => setRevealed((v) => !v)}
            className="rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs font-medium text-[var(--color-foreground)] transition-colors hover:bg-[var(--color-muted)] motion-reduce:transition-none"
          >
            {revealed ? 'Back to the raw scatter' : 'Reveal the confounder'}
          </button>
        </div>
      </div>

      {!revealed ? (
        <div>
          <svg viewBox="0 0 240 130" role="img" aria-label="Illustrative scatter plot: HCP calls versus TRx, trending upward" className="w-full max-w-xl">
            <line x1="20" y1="115" x2="230" y2="115" className="stroke-[var(--color-border)]" strokeWidth="1" />
            <line x1="20" y1="115" x2="20" y2="10" className="stroke-[var(--color-border)]" strokeWidth="1" />
            <text x="125" y="128" textAnchor="middle" className="fill-[var(--color-muted-foreground)] text-[8px]">HCP calls →</text>
            <text x="10" y="60" textAnchor="middle" transform="rotate(-90 10 60)" className="fill-[var(--color-muted-foreground)] text-[8px]">TRx →</text>
            {/* One visual population — the trend looks real */}
            {[...GENERALISTS, ...SPECIALISTS].map(([x, y]) => (
              <Dot key={`${x}-${y}`} x={x} y={y} cls="fill-[var(--color-primary)] opacity-70" />
            ))}
            <line x1="30" y1="100" x2="200" y2="35" className="stroke-[var(--color-primary)]" strokeWidth="1.5" strokeDasharray="4 3" />
          </svg>
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            More calls, more prescriptions — so call everyone more? Not so fast.
          </p>
        </div>
      ) : (
        <div>
          <svg viewBox="0 0 240 130" role="img" aria-label="Illustrative DAG: physician specialty causes both call targeting and TRx" className="w-full max-w-xl">
            <defs>
              <marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" className="fill-[var(--color-muted-foreground)]" />
              </marker>
            </defs>
            <rect x="85" y="10" width="70" height="24" rx="6" className="fill-amber-500/15 stroke-amber-500" strokeWidth="1.5" />
            <text x="120" y="26" textAnchor="middle" className="fill-[var(--color-foreground)] text-[9px] font-medium">Specialty</text>
            <rect x="15" y="90" width="70" height="24" rx="6" className="fill-[var(--color-muted)] stroke-[var(--color-border)]" strokeWidth="1.5" />
            <text x="50" y="106" textAnchor="middle" className="fill-[var(--color-foreground)] text-[9px] font-medium">HCP calls</text>
            <rect x="155" y="90" width="70" height="24" rx="6" className="fill-[var(--color-muted)] stroke-[var(--color-border)]" strokeWidth="1.5" />
            <text x="190" y="106" textAnchor="middle" className="fill-[var(--color-foreground)] text-[9px] font-medium">TRx</text>
            <line x1="100" y1="36" x2="58" y2="88" className="stroke-[var(--color-muted-foreground)]" strokeWidth="1.5" markerEnd="url(#dag-arrow)" />
            <line x1="140" y1="36" x2="182" y2="88" className="stroke-[var(--color-muted-foreground)]" strokeWidth="1.5" markerEnd="url(#dag-arrow)" />
            <line x1="87" y1="102" x2="153" y2="102" className="stroke-[var(--color-border)]" strokeWidth="1.5" strokeDasharray="4 3" markerEnd="url(#dag-arrow)" />
            <text x="120" y="97" textAnchor="middle" className="fill-[var(--color-muted-foreground)] text-[8px]">much weaker, adjusted</text>
          </svg>
          <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
            Specialty drives both: specialists get more calls AND their patients need this therapy
            more. Adjust for specialty and the calls→TRx effect shrinks dramatically. E2I finds the
            adjustment automatically — and refutation-tests what remains.
          </p>
        </div>
      )}
    </div>
  );
}
