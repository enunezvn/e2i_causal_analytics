# Confusion-matrix + ROC Interpretation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dynamic, model-specific interpretation (derived metrics + plain-language verdict + AUC quality band, framed in the model's cohort domain) under the confusion matrix and ROC curve on the Model Performance page.

**Architecture:** Pure frontend. One new module `frontend/src/lib/model-performance/interpret.ts` exposes three pure functions (`describeModel`, `interpretConfusion`, `interpretRoc`) that derive everything from the existing API responses + the model name. The page's `ConfusionMatrixView` / `RocCurveView` call them and render always-visible blocks. No backend/API/schema change.

**Tech Stack:** TypeScript, React, Vitest + Testing Library. Run FE tests from `frontend/`.

**Spec:** `docs/superpowers/specs/2026-06-20-confusion-roc-interpretation-design.md`

> **Note on test command (known FE vitest OOM):** run targeted with fork pool, e.g.
> `cd frontend && npx vitest run <file> --no-file-parallelism --pool=forks`. Append
> `--reporter=basic` if output is noisy.

---

## File Structure

- **Create** `frontend/src/lib/model-performance/interpret.ts` — the three pure helpers + their types. One responsibility: turn confusion/ROC API data + a model name into human interpretation.
- **Create** `frontend/src/lib/model-performance/interpret.test.ts` — unit tests for all three.
- **Modify** `frontend/src/pages/ModelPerformance.tsx` — pass `modelName` into the two views; render the interpretation blocks.
- **Modify** `frontend/src/pages/ModelPerformance.test.tsx` — assert interpretation text renders for an available fixture.

---

### Task 1: `describeModel` — cohort → real-world meaning

**Files:**
- Create: `frontend/src/lib/model-performance/interpret.ts`
- Test: `frontend/src/lib/model-performance/interpret.test.ts`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/lib/model-performance/interpret.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { describeModel } from './interpret';

describe('describeModel', () => {
  it('maps initiation cohort to patient/initiated treatment', () => {
    const m = describeModel('initiation_remibrutinib_goldstd_lr_v1');
    expect(m).toEqual({
      subject: 'patient',
      subjectPlural: 'patients',
      positiveEvent: 'initiated treatment',
      known: true,
    });
  });

  it('maps hcp_adoption to HCP/adopted the brand (not a patient cohort)', () => {
    const m = describeModel('hcp_adoption_kisqali_goldstd_lr_v1');
    expect(m.subject).toBe('HCP');
    expect(m.subjectPlural).toBe('HCPs');
    expect(m.positiveEvent).toBe('adopted the brand');
    expect(m.known).toBe(true);
  });

  it('maps persistence and discontinuation cohorts', () => {
    expect(describeModel('persistence_fabhalta_goldstd_lr_v1').positiveEvent).toBe(
      'persisted ≥180 days'
    );
    expect(describeModel('discontinuation_remibrutinib_goldstd_lr_v1').positiveEvent).toBe(
      'discontinued within 180 days'
    );
  });

  it('matches legacy names (csu_initiation, pnh_persistence)', () => {
    expect(describeModel('csu_initiation_goldstd_lr_v1').positiveEvent).toBe('initiated treatment');
    expect(describeModel('pnh_persistence_goldstd_lr_v1').positiveEvent).toBe('persisted ≥180 days');
  });

  it('falls back to generic for unknown names without throwing', () => {
    const m = describeModel('some_unknown_model');
    expect(m.known).toBe(false);
    expect(m.subjectPlural).toBe('cases');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/model-performance/interpret.test.ts --no-file-parallelism --pool=forks`
Expected: FAIL — cannot resolve `./interpret` (module not found).

- [ ] **Step 3: Write minimal implementation**

Create `frontend/src/lib/model-performance/interpret.ts`:

```ts
/**
 * Model-performance interpretation helpers.
 *
 * Pure functions that turn the confusion-matrix / ROC API responses plus the
 * model name into human-readable, model-specific interpretation. No network,
 * no fabrication: a metric with a zero denominator reads "n/a" rather than a
 * fake 0%/100%.
 *
 * @module lib/model-performance/interpret
 */
import type { ConfusionMatrixResponse } from '@/types/monitoring';

export interface ModelMeaning {
  /** Singular subject noun, e.g. "patient" | "HCP". */
  subject: string;
  /** Plural subject noun, e.g. "patients" | "HCPs". */
  subjectPlural: string;
  /** Positive-class verb phrase, e.g. "initiated treatment". */
  positiveEvent: string;
  /** False when the cohort could not be identified (generic fallback used). */
  known: boolean;
}

/**
 * Derive the real-world meaning of the positive class from a gold-standard
 * model name. Case-insensitive substring match; hcp_adoption is checked before
 * the patient cohorts. Unknown names get a generic, never-throwing fallback.
 */
export function describeModel(modelName: string): ModelMeaning {
  const n = (modelName || '').toLowerCase();
  if (n.includes('hcp_adoption') || n.includes('hcp')) {
    return { subject: 'HCP', subjectPlural: 'HCPs', positiveEvent: 'adopted the brand', known: true };
  }
  if (n.includes('initiation')) {
    return { subject: 'patient', subjectPlural: 'patients', positiveEvent: 'initiated treatment', known: true };
  }
  if (n.includes('persistence') || n.includes('persistent')) {
    return { subject: 'patient', subjectPlural: 'patients', positiveEvent: 'persisted ≥180 days', known: true };
  }
  if (n.includes('discontinuation') || n.includes('discontinued')) {
    return {
      subject: 'patient',
      subjectPlural: 'patients',
      positiveEvent: 'discontinued within 180 days',
      known: true,
    };
  }
  return { subject: 'case', subjectPlural: 'cases', positiveEvent: 'were in the positive class', known: false };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/model-performance/interpret.test.ts --no-file-parallelism --pool=forks`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/model-performance/interpret.ts frontend/src/lib/model-performance/interpret.test.ts
git commit -m "feat(model-perf): describeModel cohort->meaning helper (TDD)"
```

---

### Task 2: `interpretConfusion` — metrics + domain-framed verdict

**Files:**
- Modify: `frontend/src/lib/model-performance/interpret.ts`
- Test: `frontend/src/lib/model-performance/interpret.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/lib/model-performance/interpret.test.ts`:

```ts
import { interpretConfusion } from './interpret';
import type { ConfusionMatrixResponse } from '@/types/monitoring';

function cm(partial: Partial<ConfusionMatrixResponse>): ConfusionMatrixResponse {
  return {
    model_id: 'm',
    available: true,
    tn: 0,
    fp: 0,
    fn: 0,
    tp: 0,
    threshold: 0.5,
    sample_size: null,
    measured_at: null,
    ...partial,
  } as ConfusionMatrixResponse;
}

describe('interpretConfusion', () => {
  const meaning = describeModel('initiation_remibrutinib_goldstd_lr_v1');

  it('computes metrics from real initiation_remibrutinib counts', () => {
    const r = interpretConfusion(cm({ tn: 2946, fp: 346, fn: 1277, tp: 506 }), meaning);
    expect(r.precision.value).toBeCloseTo(0.5939, 3);
    expect(r.recall.value).toBeCloseTo(0.2838, 3);
    expect(r.specificity.value).toBeCloseTo(0.8949, 3);
    expect(r.accuracy.value).toBeCloseTo(0.6802, 3);
    expect(r.f1.value).toBeCloseTo(0.3841, 3);
    expect(r.precision.pct).toBe('59%');
    expect(r.recall.pct).toBe('28%');
  });

  it('selects the conservative archetype and includes real counts + domain event', () => {
    const r = interpretConfusion(cm({ tn: 2946, fp: 346, fn: 1277, tp: 506 }), meaning);
    expect(r.verdict).toContain('conservative');
    expect(r.verdict).toContain('initiated treatment');
    expect(r.verdict).toContain('506');
    expect(r.verdict).toContain('1,783'); // tp + fn
  });

  it('selects the aggressive archetype on high-recall/low-precision', () => {
    // tp=90 fn=10 -> recall .9 ; fp=200 -> precision 90/290 ~ .31 ; tn=100 -> spec .33
    const r = interpretConfusion(cm({ tn: 100, fp: 200, fn: 10, tp: 90 }), meaning);
    expect(r.verdict).toContain('aggressive');
  });

  it('selects the balanced archetype', () => {
    // tp=70 fn=30 -> recall .7 ; fp=30 -> precision .7 ; tn=70 -> spec .7
    const r = interpretConfusion(cm({ tn: 70, fp: 30, fn: 30, tp: 70 }), meaning);
    expect(r.verdict).toContain('balanced');
  });

  it('reads n/a (never a fake 0/100%) when a denominator is zero', () => {
    const r = interpretConfusion(cm({ tn: 50, fp: 0, fn: 0, tp: 0 }), meaning);
    expect(r.precision.value).toBeNull(); // tp+fp == 0
    expect(r.precision.pct).toBe('n/a');
    expect(r.recall.value).toBeNull(); // tp+fn == 0
  });

  it('returns the undetermined verdict when recall and precision are both n/a', () => {
    const r = interpretConfusion(cm({ tn: 50, fp: 0, fn: 0, tp: 0 }), meaning);
    expect(r.verdict.toLowerCase()).toContain('not enough');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/model-performance/interpret.test.ts --no-file-parallelism --pool=forks`
Expected: FAIL — `interpretConfusion` is not exported.

- [ ] **Step 3: Write minimal implementation**

Append to `frontend/src/lib/model-performance/interpret.ts`:

```ts
export interface Metric {
  /** Fraction in [0,1], or null when undefined (zero denominator). */
  value: number | null;
  /** Display string: "59%" or "n/a". */
  pct: string;
}

function metric(numerator: number, denominator: number): Metric {
  if (denominator <= 0) return { value: null, pct: 'n/a' };
  const v = numerator / denominator;
  return { value: v, pct: `${Math.round(v * 100)}%` };
}

export interface ConfusionInterpretation {
  precision: Metric;
  recall: Metric;
  specificity: Metric;
  accuracy: Metric;
  f1: Metric;
  verdict: string;
}

function positivesPhrase(meaning: ModelMeaning): string {
  return meaning.known
    ? `${meaning.subjectPlural} who actually ${meaning.positiveEvent}`
    : 'positive cases';
}

function buildVerdict(
  tp: number,
  fn: number,
  precision: Metric,
  recall: Metric,
  specificity: Metric,
  meaning: ModelMeaning
): string {
  const R = recall.value;
  const P = precision.value;
  const S = specificity.value;

  if (R === null && P === null) {
    return 'Not enough holdout outcomes to characterize this model’s behavior.';
  }

  const actualPos = tp + fn;
  const caught =
    `catches ${tp.toLocaleString()} of ${actualPos.toLocaleString()} ` +
    `${positivesPhrase(meaning)} (recall ${recall.pct})`;
  const right = P !== null ? `, and is right ${precision.pct} of the time when it predicts so (precision)` : '';
  const spec = S !== null ? `, with specificity ${specificity.pct}` : '';

  let archetype: string;
  if (R !== null && S !== null && R < 0.5 && S >= 0.7) {
    archetype = ' — conservative: it under-calls and misses most true cases.';
  } else if (R !== null && P !== null && R >= 0.7 && P < 0.5) {
    archetype = ' — aggressive: it over-calls, trading false alarms for coverage.';
  } else if (R !== null && P !== null && R >= 0.6 && P >= 0.6) {
    archetype = ' — a balanced classifier.';
  } else {
    archetype = ' — limited discrimination at this threshold; read predictions with caution.';
  }

  return `This model ${caught}${right}${spec}${archetype}`;
}

/**
 * Derive precision/recall/specificity/accuracy/F1 (each "n/a" when its
 * denominator is zero) plus a rule-based, domain-framed verdict from a binary
 * confusion matrix.
 */
export function interpretConfusion(
  data: ConfusionMatrixResponse,
  meaning: ModelMeaning
): ConfusionInterpretation {
  const { tn, fp, fn, tp } = data;
  const n = tn + fp + fn + tp;
  const precision = metric(tp, tp + fp);
  const recall = metric(tp, tp + fn);
  const specificity = metric(tn, tn + fp);
  const accuracy = metric(tp + tn, n);

  let f1: Metric = { value: null, pct: 'n/a' };
  if (precision.value !== null && recall.value !== null && precision.value + recall.value > 0) {
    const v = (2 * precision.value * recall.value) / (precision.value + recall.value);
    f1 = { value: v, pct: `${Math.round(v * 100)}%` };
  }

  const verdict = buildVerdict(tp, fn, precision, recall, specificity, meaning);
  return { precision, recall, specificity, accuracy, f1, verdict };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/model-performance/interpret.test.ts --no-file-parallelism --pool=forks`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/model-performance/interpret.ts frontend/src/lib/model-performance/interpret.test.ts
git commit -m "feat(model-perf): interpretConfusion metrics + domain verdict (TDD)"
```

---

### Task 3: `interpretRoc` — AUC band + ranking sentence

**Files:**
- Modify: `frontend/src/lib/model-performance/interpret.ts`
- Test: `frontend/src/lib/model-performance/interpret.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/lib/model-performance/interpret.test.ts`:

```ts
import { aucBand, interpretRoc } from './interpret';

describe('aucBand', () => {
  it('maps every band boundary', () => {
    expect(aucBand(0.59)).toBe('near-random');
    expect(aucBand(0.6)).toBe('weak');
    expect(aucBand(0.69)).toBe('weak');
    expect(aucBand(0.7)).toBe('acceptable');
    expect(aucBand(0.79)).toBe('acceptable');
    expect(aucBand(0.8)).toBe('good');
    expect(aucBand(0.89)).toBe('good');
    expect(aucBand(0.9)).toBe('excellent');
  });
});

describe('interpretRoc', () => {
  const meaning = describeModel('initiation_remibrutinib_goldstd_lr_v1');

  it('formats AUC, band, ranking % and domain framing', () => {
    const r = interpretRoc(0.671, meaning);
    expect(r.band).toBe('weak');
    expect(r.text).toContain('AUC 0.671 (weak)');
    expect(r.text).toContain('67%');
    expect(r.text).toContain('patient who initiated treatment');
    expect(r.text).toContain('coin-flip');
  });

  it('uses each comparison branch', () => {
    expect(interpretRoc(0.55, meaning).text).toContain('barely above');
    expect(interpretRoc(0.65, meaning).text).toContain('modestly better');
    expect(interpretRoc(0.78, meaning).text).toContain('clearly better than chance');
    expect(interpretRoc(0.9, meaning).text).toContain('strong separation');
  });

  it('uses generic framing when the cohort is unknown', () => {
    const r = interpretRoc(0.72, describeModel('mystery_model'));
    expect(r.text).toContain('a random positive case above a random negative case');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/lib/model-performance/interpret.test.ts --no-file-parallelism --pool=forks`
Expected: FAIL — `aucBand` / `interpretRoc` not exported.

- [ ] **Step 3: Write minimal implementation**

Append to `frontend/src/lib/model-performance/interpret.ts`:

```ts
export interface RocInterpretation {
  /** Quality band label. */
  band: string;
  /** Full plain-language sentence. */
  text: string;
}

/** Non-inflated AUC quality band (0.5 = chance). */
export function aucBand(auc: number): string {
  if (auc < 0.6) return 'near-random';
  if (auc < 0.7) return 'weak';
  if (auc < 0.8) return 'acceptable';
  if (auc < 0.9) return 'good';
  return 'excellent';
}

/**
 * Interpret an ROC AUC: quality band + the ranking-probability sentence vs the
 * 0.5 chance baseline, framed in the model's cohort domain.
 */
export function interpretRoc(auc: number, meaning: ModelMeaning): RocInterpretation {
  const band = aucBand(auc);
  const pct = Math.round(auc * 100);
  const rank = meaning.known
    ? `a random ${meaning.subject} who ${meaning.positiveEvent} above a random one who did not`
    : 'a random positive case above a random negative case';

  let compare: string;
  if (auc <= 0.55) compare = 'barely above the 0.50 coin-flip baseline';
  else if (auc <= 0.7) compare = 'modestly better than the 0.50 coin-flip baseline';
  else if (auc <= 0.85) compare = 'clearly better than chance (0.50)';
  else compare = 'strong separation, well above chance (0.50)';

  const text = `AUC ${auc.toFixed(3)} (${band}). The model ranks ${rank} ${pct}% of the time — ${compare}.`;
  return { band, text };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/lib/model-performance/interpret.test.ts --no-file-parallelism --pool=forks`
Expected: PASS (Tasks 1–3, all green).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/model-performance/interpret.ts frontend/src/lib/model-performance/interpret.test.ts
git commit -m "feat(model-perf): interpretRoc AUC band + ranking sentence (TDD)"
```

---

### Task 4: Render interpretation in the page

**Files:**
- Modify: `frontend/src/pages/ModelPerformance.tsx` (`ConfusionMatrixView` ~L128, `RocCurveView` ~L160, the two render sites ~L742 and the ROC tab)
- Test: `frontend/src/pages/ModelPerformance.test.tsx`

- [ ] **Step 1: Write the failing test**

Find the existing confusion/ROC render test in `frontend/src/pages/ModelPerformance.test.tsx` (search for `available` fixtures / `Confusion`/`ROC`). Add a focused test. If the file mocks the monitoring hooks, mirror that mock; the snippet below assumes the hooks `useConfusionMatrix` / `useRocCurve` are mocked to return available data (match the file's existing mock style — read it first and reuse its helpers).

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
// ... reuse the file's existing imports, providers, and hook mocks ...

it('renders the confusion-matrix verdict and ROC AUC band for an available model', async () => {
  // Arrange: mock the monitoring hooks so the selected model is an initiation
  // cohort with available confusion + ROC data. Reuse the file's existing
  // mock setup for useModels/useConfusionMatrix/useRocCurve; set:
  //   model_name: 'initiation_remibrutinib_goldstd_lr_v1'
  //   confusion available: { tn: 2946, fp: 346, fn: 1277, tp: 506, threshold: 0.5, sample_size: 5075 }
  //   roc available: { auc: 0.671, points: [{ fpr: 0, tpr: 0, threshold: 1 }, { fpr: 1, tpr: 1, threshold: 0 }], sample_size: 5075 }

  render(<ModelPerformanceUnderTest />); // however the file renders the page

  // Confusion tab content
  expect(await screen.findByText(/conservative/i)).toBeInTheDocument();
  expect(screen.getByText(/initiated treatment/i)).toBeInTheDocument();
  // ROC tab content
  expect(screen.getByText(/AUC 0\.671 \(weak\)/i)).toBeInTheDocument();
});
```

> If the page renders confusion + ROC only inside their tabs and the tabs are not
> both mounted at once, split into two tests (one per tab) and activate the tab
> with `await userEvent.click(screen.getByRole('tab', { name: /confusion matrix/i }))`
> before asserting. Reuse the file's existing tab-activation pattern if present.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/pages/ModelPerformance.test.tsx --no-file-parallelism --pool=forks`
Expected: FAIL — verdict / "AUC 0.671 (weak)" text not in the DOM (not rendered yet).

- [ ] **Step 3: Write minimal implementation**

In `frontend/src/pages/ModelPerformance.tsx`:

(a) Add the import near the other `@/` imports:

```tsx
import {
  describeModel,
  interpretConfusion,
  interpretRoc,
} from '@/lib/model-performance/interpret';
```

(b) Change `ConfusionMatrixView` to accept `modelName` and render metrics + verdict. Replace the existing function body's return with the augmented version:

```tsx
function ConfusionMatrixView({
  data,
  modelName,
}: {
  data: ConfusionMatrixResponse;
  modelName: string;
}) {
  const cells = [
    { label: 'True Negative', value: data.tn, good: true },
    { label: 'False Positive', value: data.fp, good: false },
    { label: 'False Negative', value: data.fn, good: false },
    { label: 'True Positive', value: data.tp, good: true },
  ];
  const meaning = describeModel(modelName);
  const interp = interpretConfusion(data, meaning);
  const metrics = [
    { label: 'Precision', m: interp.precision },
    { label: 'Recall', m: interp.recall },
    { label: 'Specificity', m: interp.specificity },
    { label: 'Accuracy', m: interp.accuracy },
    { label: 'F1', m: interp.f1 },
  ];
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 max-w-md">
        {cells.map((c) => (
          <div
            key={c.label}
            className={`rounded-md border p-4 text-center ${
              c.good ? 'bg-emerald-50 dark:bg-emerald-900/20' : 'bg-rose-50 dark:bg-rose-900/20'
            }`}
          >
            <div className="text-2xl font-bold">{c.value.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground mt-1">{c.label}</div>
          </div>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        Holdout @ threshold {data.threshold.toFixed(2)}
        {data.sample_size ? ` · n=${data.sample_size.toLocaleString()}` : ''} · rows = actual,
        columns = predicted
      </p>
      <div className="flex flex-wrap gap-4 text-sm">
        {metrics.map(({ label, m }) => (
          <div key={label} className="flex flex-col">
            <span className="text-muted-foreground text-xs">{label}</span>
            <span className="font-medium">{m.pct}</span>
          </div>
        ))}
      </div>
      <div className="rounded-md bg-muted p-3 text-sm">{interp.verdict}</div>
    </div>
  );
}
```

(c) Change `RocCurveView` to accept `modelName` and render the interpretation. Add the props and, after the existing `AUC =` caption `<p>`, insert the block:

```tsx
function RocCurveView({ data, modelName }: { data: RocCurveResponse; modelName: string }) {
  const roc = interpretRoc(data.auc, describeModel(modelName));
  return (
    <div className="space-y-2">
      {/* ...existing ResponsiveContainer/LineChart unchanged... */}
      <p className="text-xs text-muted-foreground">
        AUC = {data.auc.toFixed(3)}
        {data.sample_size ? ` · n=${data.sample_size.toLocaleString()}` : ''} · holdout
      </p>
      <div className="rounded-md bg-muted p-3 text-sm">
        <span className="text-xs uppercase tracking-wide text-muted-foreground mr-2">
          {roc.band}
        </span>
        {roc.text}
      </div>
    </div>
  );
}
```

(d) Pass `modelName` at the two render sites (confusion tab ~L742 and the ROC tab). `effectiveModelId` is the selected model_name:

```tsx
// confusion tab
<ConfusionMatrixView data={confusionQuery.data} modelName={effectiveModelId} />
// roc tab
<RocCurveView data={rocQuery.data} modelName={effectiveModelId} />
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/pages/ModelPerformance.test.tsx --no-file-parallelism --pool=forks`
Expected: PASS (the new render test + the file's existing tests).

- [ ] **Step 5: Typecheck + lint + commit**

```bash
cd frontend && npx tsc -b && npm run lint
git add frontend/src/pages/ModelPerformance.tsx frontend/src/pages/ModelPerformance.test.tsx
git commit -m "feat(model-perf): render confusion/ROC interpretation in the page (TDD)"
```

Expected: `tsc -b` clean (the page `build` job is stricter than `tsc --noEmit`).

---

## Self-Review

**1. Spec coverage:**
- describeModel (cohort map incl. legacy + generic) → Task 1. ✓
- interpretConfusion (5 metrics, n/a guards, 4 verdict archetypes, domain framing) → Task 2. ✓
- interpretRoc (5 bands, 4 compare branches, generic fallback) → Task 3. ✓
- Rendering under both views, always-visible, empty states untouched → Task 4. ✓
- TDD unit tests + page render assertion → Tasks 1–4. ✓
- No backend change → confirmed (only the 4 files in File Structure). ✓

**2. Placeholder scan:** No TBD/TODO. Task 4 Step 1 intentionally defers to the test file's existing hook-mock style (it must be read first); the assertions and data are concrete. All helper code blocks are complete.

**3. Type consistency:** `ModelMeaning`, `Metric`, `ConfusionInterpretation`, `RocInterpretation` defined in Task 1–3 and consumed in Task 4. Function names (`describeModel`, `interpretConfusion`, `interpretRoc`, `aucBand`) consistent across tasks and the page import. `ConfusionMatrixResponse` / `RocCurveResponse` imported from `@/types/monitoring` (verified fields: `tn/fp/fn/tp/threshold/sample_size`, `auc/points/sample_size`). `effectiveModelId` is the model_name string (verified in the page).
