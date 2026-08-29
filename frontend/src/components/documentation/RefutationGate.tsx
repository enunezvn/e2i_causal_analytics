/**
 * RefutationGate — the quality gate on causal estimates: the five refutation
 * tests, each with an illustration that can be flipped between "estimate
 * survives" and "estimate fails", and the proceed / review / block bands the
 * results feed. Content (defaults, pass rules, bands) lives in content.ts and
 * mirrors src/causal_engine/refutation_runner.py.
 *
 * The illustrations are schematic — no measured values — and the figure is
 * badged "Illustrative example" per the documentation spec.
 */
import { useId, useState } from 'react';
import { GATE_BANDS, REFUTATION_INTRO, REFUTATION_TESTS } from './content';
import type { GateDecision, RefutationTestDef, RefutationTestId } from './content';

type Outcome = 'pass' | 'fail';

const TREATMENT = '#0ea5e9';
const OUTCOME = '#22c55e';
const CONFOUNDER = '#f59e0b';
const FAIL = '#ef4444';
const PASS = '#22c55e';

const GATE_COLOR: Record<GateDecision, string> = {
  proceed: PASS,
  review: CONFOUNDER,
  block: FAIL,
};

/** Small-text class for SVG captions — text wears the text token, never a series color. */
const TXT = 'fill-[var(--color-foreground)]';
const MUTED = 'fill-[var(--color-muted-foreground)]';
const AXIS = 'stroke-[var(--color-muted-foreground)]';
const ANIM = 'transition-all duration-300 motion-reduce:transition-none';

function Verdict({ x, y, outcome, pass, fail }: { x: number; y: number; outcome: Outcome; pass: string; fail: string }) {
  const ok = outcome === 'pass';
  return (
    <text x={x} y={y} fontSize="10" fontWeight="600" textAnchor="middle" fill={ok ? PASS : FAIL}>
      {ok ? `✓ ${pass}` : `✗ ${fail}`}
    </text>
  );
}

/* ---------------------------------------------------------------- Placebo */
function PlaceboIllustration({ outcome }: { outcome: Outcome }) {
  const ok = outcome === 'pass';
  const base = 92;
  const placeboH = ok ? 4 : 44;
  return (
    <>
      <line x1="20" y1={base} x2="220" y2={base} className={AXIS} strokeWidth="1" />
      {/* real treatment */}
      <rect x="46" y={base - 56} width="34" height="56" rx="3" fill={TREATMENT} />
      <text x="63" y={base - 62} fontSize="9" textAnchor="middle" className={MUTED}>effect</text>
      <text x="63" y={base + 14} fontSize="10" textAnchor="middle" className={TXT}>real treatment</text>
      {/* placebo treatment */}
      <rect
        x="156"
        y={base - placeboH}
        width="34"
        height={placeboH}
        rx="3"
        fill={ok ? TREATMENT : FAIL}
        fillOpacity={ok ? 0.5 : 1}
        className={ANIM}
      />
      <text x="173" y={base + 14} fontSize="10" textAnchor="middle" className={TXT}>shuffled (noise)</text>
      <text x="118" y="40" fontSize="14" textAnchor="middle" className={MUTED}>→</text>
      <Verdict x="173" y={base - placeboH - 6} outcome={outcome} pass="≈ 0" fail="still an effect" />
    </>
  );
}

/* ---------------------------------------------------- Random common cause */
function RandomCommonCauseIllustration({ outcome }: { outcome: Outcome }) {
  const ok = outcome === 'pass';
  const base = 104;
  const afterH = ok ? 30 : 12;
  return (
    <>
      {/* U → T, U → Y (dashed: it carries no information) */}
      <circle cx="120" cy="22" r="12" fill={CONFOUNDER} fillOpacity="0.15" stroke={CONFOUNDER} strokeWidth="2" strokeDasharray="4 3" />
      <text x="120" y="26" fontSize="10" textAnchor="middle" className={TXT}>U</text>
      <text x="120" y="8" fontSize="9" textAnchor="middle" className={MUTED}>random confounder</text>
      <line x1="109" y1="30" x2="52" y2="54" stroke={CONFOUNDER} strokeWidth="1.5" strokeDasharray="4 3" />
      <line x1="131" y1="30" x2="188" y2="54" stroke={CONFOUNDER} strokeWidth="1.5" strokeDasharray="4 3" />
      {/* T → Y */}
      <circle cx="40" cy="62" r="13" fill={TREATMENT} fillOpacity="0.15" stroke={TREATMENT} strokeWidth="2" />
      <text x="40" y="66" fontSize="10" textAnchor="middle" className={TXT}>T</text>
      <circle cx="200" cy="62" r="13" fill={OUTCOME} fillOpacity="0.15" stroke={OUTCOME} strokeWidth="2" />
      <text x="200" y="66" fontSize="10" textAnchor="middle" className={TXT}>Y</text>
      <line x1="54" y1="62" x2="183" y2="62" className="stroke-[var(--color-foreground)]" strokeWidth="1.5" />
      <path d="M183,57 L192,62 L183,67 z" className="fill-[var(--color-foreground)]" />
      {/* before / after bars */}
      <line x1="84" y1={base} x2="156" y2={base} className={AXIS} strokeWidth="1" />
      <rect x="92" y={base - 30} width="22" height="30" rx="2" fill={TREATMENT} />
      <rect x="126" y={base - afterH} width="22" height={afterH} rx="2" fill={ok ? TREATMENT : FAIL} fillOpacity={ok ? 0.5 : 1} className={ANIM} />
      <text x="103" y={base + 12} fontSize="9" textAnchor="middle" className={MUTED}>before</text>
      <text x="137" y={base + 12} fontSize="9" textAnchor="middle" className={MUTED}>after</text>
      <Verdict x="200" y="100" outcome={outcome} pass="Δ < 20 %" fail="Δ > 30 %" />
    </>
  );
}

/* ----------------------------------------------------------- Data subset */
// Fixed 20 % of the 40 cells left out (deterministic, so the picture is stable).
const HOLDOUT = new Set([3, 7, 12, 18, 21, 26, 30, 35]);
const SUBSET_DOTS_PASS = [-0.55, -0.2, 0.05, 0.3, 0.6];
const SUBSET_DOTS_FAIL = [-1.7, -0.35, 0.2, 1.5, 1.9];

function DataSubsetIllustration({ outcome }: { outcome: Outcome }) {
  const ok = outcome === 'pass';
  const dots = ok ? SUBSET_DOTS_PASS : SUBSET_DOTS_FAIL;
  const cx0 = 160, half = 34; // CI band spans cx0 ± half (126..194); outliers stay right of the grid
  return (
    <>
      {Array.from({ length: 40 }, (_, i) => {
        const col = i % 8, row = Math.floor(i / 8);
        const out = HOLDOUT.has(i);
        return (
          <circle
            key={i}
            cx={14 + col * 9}
            cy={20 + row * 11}
            r="3.2"
            fill={out ? 'none' : TREATMENT}
            stroke={TREATMENT}
            strokeWidth={out ? 1 : 0}
            strokeOpacity="0.6"
          />
        );
      })}
      <text x="46" y="80" fontSize="9" textAnchor="middle" className={MUTED}>80 % of rows, ×5</text>
      <text x="46" y="100" fontSize="9" textAnchor="middle" className={MUTED}>{ok ? 'coverage ≥ 80 %' : 'coverage < 70 %'}</text>
      {/* CI band with the original effect tick */}
      <text x={cx0} y="20" fontSize="9" textAnchor="middle" className={MUTED}>original effect · 95 % CI</text>
      <rect x={cx0 - half} y="30" width={half * 2} height="26" rx="4" fill={OUTCOME} fillOpacity="0.15" />
      <line x1={cx0} y1="26" x2={cx0} y2="60" className={AXIS} strokeWidth="1" strokeDasharray="2 2" />
      {dots.map((d, i) => {
        const x = cx0 + d * half;
        const inside = Math.abs(d) <= 1;
        return <circle key={i} cx={x} cy={43} r="4" fill={inside ? OUTCOME : FAIL} className={ANIM} />;
      })}
      <text x={cx0} y="74" fontSize="9" textAnchor="middle" className={MUTED}>subset estimates</text>
      <Verdict x={cx0} y="100" outcome={outcome} pass="5 of 5 inside" fail="3 of 5 outside" />
    </>
  );
}

/* ------------------------------------------------------------- Bootstrap */
const HIST_PASS = [3, 8, 18, 32, 46, 54, 46, 32, 18, 8, 3];
const HIST_FAIL = [20, 26, 31, 35, 38, 40, 38, 35, 31, 26, 20];

function BootstrapIllustration({ outcome }: { outcome: Outcome }) {
  const ok = outcome === 'pass';
  const bars = ok ? HIST_PASS : HIST_FAIL;
  const base = 76, x0 = 42, w = 12, gap = 2;
  const ciHalf = ok ? 2 : 5; // bars either side of the centre inside the CI
  const cxCentre = x0 + 5 * (w + gap) + w / 2;
  const ciL = cxCentre - ciHalf * (w + gap) - w / 2;
  const ciR = cxCentre + ciHalf * (w + gap) + w / 2;
  return (
    <>
      <line x1="30" y1={base} x2="210" y2={base} className={AXIS} strokeWidth="1" />
      {bars.map((h, i) => (
        <rect
          key={i}
          x={x0 + i * (w + gap)}
          y={base - h}
          width={w}
          height={h}
          rx="2"
          fill={ok ? TREATMENT : FAIL}
          fillOpacity={0.9}
          className={ANIM}
        />
      ))}
      {/* CI bracket */}
      <line x1={ciL} y1={base + 10} x2={ciR} y2={base + 10} className="stroke-[var(--color-foreground)]" strokeWidth="1.5" />
      <line x1={ciL} y1={base + 6} x2={ciL} y2={base + 14} className="stroke-[var(--color-foreground)]" strokeWidth="1.5" />
      <line x1={ciR} y1={base + 6} x2={ciR} y2={base + 14} className="stroke-[var(--color-foreground)]" strokeWidth="1.5" />
      <text x={cxCentre} y={base + 22} fontSize="9" textAnchor="middle" className={MUTED}>bootstrap CI (50×)</text>
      <text x="120" y="12" fontSize="9" textAnchor="middle" className={MUTED}>distribution of resampled effects</text>
      <Verdict x="120" y="114" outcome={outcome} pass="CI ≤ 1.5× original" fail="CI > 1.75× original" />
    </>
  );
}

/* --------------------------------------------------------------- E-value */
function EValueIllustration({ outcome }: { outcome: Outcome }) {
  const ok = outcome === 'pass';
  const x0 = 24, x1 = 216, y = 56; // scale 1.0 → 4.0
  const px = (e: number) => x0 + ((e - 1) / 3) * (x1 - x0);
  const marker = ok ? 2.8 : 1.3;
  return (
    <>
      <text x="120" y="14" fontSize="9" textAnchor="middle" className={MUTED}>how strong a hidden confounder must be →</text>
      <rect x={px(1)} y={y - 9} width={px(2) - px(1)} height="18" rx="3" fill={FAIL} fillOpacity="0.15" />
      <rect x={px(2)} y={y - 9} width={px(4) - px(2)} height="18" rx="3" fill={PASS} fillOpacity="0.15" />
      <line x1={x0} y1={y} x2={x1} y2={y} className={AXIS} strokeWidth="1" />
      {[1, 2, 3, 4].map((e) => (
        <g key={e}>
          <line x1={px(e)} y1={y - 12} x2={px(e)} y2={y + 12} className={AXIS} strokeWidth={e === 2 ? 2 : 1} />
          <text x={px(e)} y={y + 24} fontSize="9" textAnchor="middle" className={MUTED}>{e.toFixed(1)}</text>
        </g>
      ))}
      <text x={px(1.5)} y={y - 16} fontSize="9" textAnchor="middle" className={MUTED}>explained away easily</text>
      <text x={px(3)} y={y - 16} fontSize="9" textAnchor="middle" className={MUTED}>robust</text>
      <text x={px(2)} y={y + 34} fontSize="9" fontWeight="600" textAnchor="middle" className={TXT}>threshold 2.0</text>
      <path
        d={`M${px(marker)},${y - 2} l-6,-10 l12,0 z`}
        fill={ok ? PASS : FAIL}
        className={ANIM}
        style={{ transitionProperty: 'd, fill' }}
      />
      <Verdict x={Math.min(176, Math.max(64, px(marker)))} y="110" outcome={outcome} pass="above the 2.0 threshold" fail="below 1.5 — fragile" />
    </>
  );
}

const ILLUSTRATIONS: Record<RefutationTestId, (p: { outcome: Outcome }) => JSX.Element> = {
  placebo_treatment: PlaceboIllustration,
  random_common_cause: RandomCommonCauseIllustration,
  data_subset: DataSubsetIllustration,
  bootstrap: BootstrapIllustration,
  sensitivity_e_value: EValueIllustration,
};

const ILLUSTRATION_ALT: Record<RefutationTestId, Record<Outcome, string>> = {
  placebo_treatment: {
    pass: 'Bar chart: the real treatment shows an effect; the shuffled treatment shows almost none.',
    fail: 'Bar chart: the shuffled treatment still shows a large effect — the test fails.',
  },
  random_common_cause: {
    pass: 'Small graph: a random confounder U is added to T and Y; the effect before and after is the same.',
    fail: 'Small graph: adding a random confounder U changes the effect markedly — the test fails.',
  },
  data_subset: {
    pass: 'Five subset estimates all fall inside the original confidence interval.',
    fail: 'Three of five subset estimates fall outside the original confidence interval — the test fails.',
  },
  bootstrap: {
    pass: 'A narrow histogram of resampled effects with a bootstrap interval close to the original width.',
    fail: 'A flat, wide histogram of resampled effects whose interval is far wider than the original — the test fails.',
  },
  sensitivity_e_value: {
    pass: 'A scale from 1 to 4 with the E-value marker above the 2.0 threshold, in the robust zone.',
    fail: 'A scale from 1 to 4 with the E-value marker below 1.5 — a weak confounder could explain the effect.',
  },
};

function TestCard({ test, outcome }: { test: RefutationTestDef; outcome: Outcome }) {
  const Illustration = ILLUSTRATIONS[test.id];
  const headingId = useId();
  return (
    <li
      aria-labelledby={headingId}
      data-test-id={test.id}
      data-critical={test.critical ? 'true' : 'false'}
      data-outcome={outcome}
      className="flex flex-col rounded-md border border-[var(--color-border)] bg-[var(--color-muted)]/40"
    >
      <div className="flex items-start justify-between gap-2 px-3 pt-3">
        <h4 id={headingId} className="text-sm font-semibold text-[var(--color-foreground)]">
          {test.name}
        </h4>
        <span
          className={`shrink-0 rounded-full border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
            test.critical
              ? 'border-red-500/50 bg-red-500/10 text-red-600 dark:text-red-400'
              : 'border-[var(--color-border)] text-[var(--color-muted-foreground)]'
          }`}
          title={
            test.critical
              ? 'A failure on this test blocks the estimate on its own.'
              : 'Feeds the confidence score; a failure lowers it but does not block on its own.'
          }
        >
          {test.critical ? 'critical' : 'stability'}
        </span>
      </div>
      <svg
        viewBox="0 0 240 120"
        role="img"
        aria-label={ILLUSTRATION_ALT[test.id][outcome]}
        className="mt-2 h-auto w-full"
      >
        <Illustration outcome={outcome} />
      </svg>
      <div className="space-y-1 px-3 pb-3 text-xs leading-5">
        <p className="text-[var(--color-foreground)]">
          {test.action} — <strong>{test.mustHold}</strong>
        </p>
        <p className="text-[var(--color-muted-foreground)]">
          <span className="font-medium text-[var(--color-foreground)]">Default:</span> {test.defaults}
        </p>
        <p className="text-[var(--color-muted-foreground)]">
          <span className="font-medium text-[var(--color-foreground)]">Passes when:</span>{' '}
          <span>{test.passRule}</span>
        </p>
        {outcome === 'fail' && (
          <p className="border-l-2 pl-2 text-[var(--color-muted-foreground)]" style={{ borderLeftColor: FAIL }}>
            {test.failSign}
          </p>
        )}
      </div>
    </li>
  );
}

function OutcomeButton({
  label,
  value,
  current,
  onSelect,
}: {
  label: string;
  value: Outcome;
  current: Outcome;
  onSelect: (o: Outcome) => void;
}) {
  const pressed = current === value;
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={() => onSelect(value)}
      className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors motion-reduce:transition-none ${
        pressed
          ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
          : 'border-[var(--color-border)] bg-[var(--color-card)] text-[var(--color-foreground)] hover:border-[var(--color-primary)]/50'
      }`}
    >
      {label}
    </button>
  );
}

export function RefutationGate() {
  const [outcome, setOutcome] = useState<Outcome>('pass');
  // Every test failing at once is a BLOCK (three of them are critical); every
  // test passing clears the confidence bar → PROCEED.
  const activeGate: GateDecision = outcome === 'pass' ? 'proceed' : 'block';

  return (
    <section
      aria-labelledby="refutation-tests-heading"
      className="rounded-lg border border-[var(--color-border)] bg-[var(--color-card)] p-4"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 id="refutation-tests-heading" className="text-sm font-semibold text-[var(--color-foreground)]">
            Five refutation tests
          </h3>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--color-foreground)]">{REFUTATION_INTRO}</p>
        </div>
        <span className="rounded-full border border-amber-500/50 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-600 dark:text-amber-400">
          Illustrative example
        </span>
      </div>

      <div
        role="group"
        aria-label="Show the estimate surviving or failing the tests"
        className="mt-3 flex flex-wrap items-center gap-2"
      >
        <span className="text-xs text-[var(--color-muted-foreground)]">Show:</span>
        <OutcomeButton label="Estimate survives" value="pass" current={outcome} onSelect={setOutcome} />
        <OutcomeButton label="Estimate fails" value="fail" current={outcome} onSelect={setOutcome} />
      </div>

      <ol className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {REFUTATION_TESTS.map((t) => (
          <TestCard key={t.id} test={t} outcome={outcome} />
        ))}
      </ol>

      <h4 className="mt-5 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
        The gate the results feed
      </h4>
      <dl className="mt-2 grid gap-3 sm:grid-cols-3">
        {GATE_BANDS.map((band) => {
          const active = band.decision === activeGate;
          return (
            <div
              key={band.decision}
              data-gate={band.decision}
              data-gate-active={active ? 'true' : 'false'}
              className={`rounded-md border border-l-4 px-3 py-2 transition-colors motion-reduce:transition-none ${
                active ? 'bg-[var(--color-muted)]/60' : 'border-[var(--color-border)] opacity-70'
              }`}
              style={{ borderLeftColor: GATE_COLOR[band.decision], borderColor: active ? GATE_COLOR[band.decision] : undefined }}
            >
              <dt className="flex items-center gap-2 text-sm font-semibold" style={{ color: GATE_COLOR[band.decision] }}>
                {band.label}
                {active && (
                  <span className="rounded-full border border-current px-1.5 text-[10px] font-medium uppercase tracking-wide">
                    this example
                  </span>
                )}
              </dt>
              <dd className="mt-1 text-xs leading-5 text-[var(--color-foreground)]">{band.rule}</dd>
              <dd className="text-xs leading-5 text-[var(--color-muted-foreground)]">{band.consequence}</dd>
            </div>
          );
        })}
      </dl>
      <p className="mt-3 text-xs leading-5 text-[var(--color-muted-foreground)]">
        The toggle shows every test passing or every test failing at once. In a real run each test
        reports its own status (pass, warning, fail) and the gate combines them: a critical failure
        blocks immediately; otherwise the weighted confidence score decides.
      </p>
    </section>
  );
}
