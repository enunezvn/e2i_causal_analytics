# Confusion-matrix + ROC interpretation on Model Performance

**Date:** 2026-06-20
**Status:** Approved (brainstorming) — pending spec review
**Scope:** Frontend only. No backend / API / schema / migration change.

## Problem

The Model Performance page (`frontend/src/pages/ModelPerformance.tsx`) now renders a
confusion matrix (`ConfusionMatrixView`) and ROC curve (`RocCurveView`) for the 12
gold-standard models (data populated 2026-06-20). But each shows only the raw artifact
plus a terse caption:

- Confusion: four count cells (TN/FP/FN/TP) + `Holdout @ threshold 0.50 · n=5,075`.
- ROC: the curve + chance diagonal + `AUC = 0.671 · n=5,075 · holdout`.

An analyst sees the artifact but not **what it says about the model's performance**.
There is no precision/recall/specificity, no AUC quality judgement, and no plain-language
verdict. This spec adds a **dynamic, model-specific interpretation with cohort-domain
framing** under each viz.

## Goals

1. Under the confusion matrix: derived metrics (precision, recall/sensitivity,
   specificity, accuracy, F1) **and** a plain-language verdict computed from *this*
   model's counts, framed in the model's cohort domain.
2. Under the ROC curve: an AUC quality band + a plain-language sentence (ranking
   interpretation vs the 0.5 chance baseline + curve-shape note), domain-framed.
3. All values derived from data already in the API responses + the model name. Honest
   on degenerate inputs (never fabricate a metric when its denominator is 0).

## Non-goals (YAGNI)

- No backend/API/schema/migration change (everything derives client-side).
- No brand-level prose (kept to subject + event; brand omitted from the sentence).
- No threshold slider / multi-threshold exploration.
- No export/chat wiring.
- No LLM-generated narrative (non-deterministic; risks hallucinated stats).

## Approach

Approach **A — pure frontend helpers** (chosen over backend-returns-prose and
LLM-narrative). Two pure, unit-tested helper modules feed small presentational blocks
appended to the existing `ConfusionMatrixView` / `RocCurveView`. The page already holds
the selected model (`selectedModel.model_name`) and the API responses
(`ConfusionMatrixResponse` = `tn/fp/fn/tp/threshold/sample_size`; `RocCurveResponse` =
`auc/points/sample_size`), so no new data is required.

## Components

All new pure logic lives in **`frontend/src/lib/model-performance/interpret.ts`** (new
directory), each function independently testable.

### 1. `describeModel(modelName: string): ModelMeaning`

Parses the cohort prefix of the model name → the real-world meaning of the positive class.

```
interface ModelMeaning {
  subject: string;        // "patient" | "HCP"
  subjectPlural: string;  // "patients" | "HCPs"
  positiveEvent: string;  // verb phrase, e.g. "initiated treatment"
  known: boolean;         // false → generic fallback used
}
```

| cohort token in model name | subject | positiveEvent |
|---|---|---|
| `initiation` | patient | initiated treatment |
| `persistence` | patient | persisted ≥180 days |
| `discontinuation` | patient | discontinued within 180 days |
| `hcp_adoption` (or `hcp`) | HCP | adopted the brand |
| none matched | patient | _(generic: positiveEvent = "were in the positive class", `known=false`)_ |

Matching is case-insensitive substring on the model name, checked in the order above
(`hcp_adoption` before the bare patient cohorts). Covers current names
(`initiation_remibrutinib_goldstd_lr_v1`, `hcp_adoption_kisqali_goldstd_lr_v1`) and
legacy (`csu_initiation_goldstd_lr_v1`, `pnh_persistence_goldstd_lr_v1`). Unknown →
generic, never throws.

### 2. `interpretConfusion(data: ConfusionMatrixResponse, meaning: ModelMeaning): ConfusionInterpretation`

```
interface Metric { value: number | null; pct: string; }  // null + "n/a" when undefined
interface ConfusionInterpretation {
  precision: Metric;    // tp/(tp+fp)         — null if tp+fp == 0
  recall: Metric;       // tp/(tp+fn)         — null if tp+fn == 0  (sensitivity)
  specificity: Metric;  // tn/(tn+fp)         — null if tn+fp == 0
  accuracy: Metric;     // (tp+tn)/n          — null if n == 0
  f1: Metric;           // 2PR/(P+R)          — null if P or R null, or P+R == 0
  verdict: string;      // plain-language, domain-framed, uses real counts
}
```

**Verdict archetype** (rule-based; evaluate top-down on recall R, precision P,
specificity S; all as fractions):

| condition | archetype | sentence skeleton |
|---|---|---|
| R null **and** P null | `undetermined` | "Not enough holdout outcomes to characterize behavior." |
| R < 0.5 **and** S ≥ 0.7 | `conservative` | "…catches {tp} of {tp+fn} {subjectPlural} who actually {positiveEvent} (recall {R%}), is right {P%} of the time when it predicts so (precision), and rarely false-alarms (specificity {S%}) — conservative: it under-calls and misses most true cases." |
| R ≥ 0.7 **and** P < 0.5 | `aggressive` | "…catches most {subjectPlural} who actually {positiveEvent} (recall {R%}) but is right only {P%} of the time when it predicts so (precision) — aggressive: it over-calls, trading false alarms for coverage." |
| R ≥ 0.6 **and** P ≥ 0.6 | `balanced` | "…balances catching true cases (recall {R%}) with being right when it predicts (precision {P%}) — a balanced classifier." |
| otherwise | `weak` | "…recall {R%} / precision {P%} at this threshold — limited discrimination; predictions should be read with caution." |

Each non-undetermined sentence is prefixed `"This model "`. `n/a` metrics are omitted
gracefully from the sentence (skeleton degrades — e.g., if precision is n/a the "is right
X% of the time" clause is dropped). All percentages are rendered as whole numbers
(`Math.round(value * 100)`).

### 3. `interpretRoc(auc: number, meaning: ModelMeaning): RocInterpretation`

```
interface RocInterpretation { band: string; text: string; }
```

**AUC bands** (approved):

| AUC | band |
|---|---|
| `< 0.6` | near-random |
| `0.6 – < 0.7` | weak |
| `0.7 – < 0.8` | acceptable |
| `0.8 – < 0.9` | good |
| `≥ 0.9` | excellent |

**Text:** `"AUC {auc.toFixed(3)} ({band}). The model ranks a random {subject} who
{positiveEvent} above a random one who did not {round(auc*100)}% of the time — {compare}."`
where `compare` (as-built — boundaries aligned to the `aucBand` edges so the band label and
the comparison phrasing never contradict; see "As-built refinements" below):
- `auc < 0.6` → "barely above the 0.50 coin-flip baseline"   (near-random band)
- `0.6 ≤ auc < 0.7` → "modestly better than the 0.50 coin-flip baseline"   (weak band)
- `0.7 ≤ auc ≤ 0.85` → "clearly better than chance (0.50)"   (acceptable + low-good)
- `auc > 0.85` → "strong separation, well above chance (0.50)"

Generic fallback (`meaning.known === false`) substitutes "a random positive case above a
random negative case".

## Rendering

Extend the existing views in `ModelPerformance.tsx` (signatures gain `modelName: string`,
passed from `selectedModel.model_name`):

- `ConfusionMatrixView`: after the count grid + existing caption, render
  - a metrics row (Precision · Recall · Specificity · Accuracy · F1, each value or "n/a"),
  - an always-visible muted card (`bg-muted` / `text-sm`) containing `verdict`.
- `RocCurveView`: after the chart + existing `AUC =` caption, render an always-visible
  muted card containing `interpretRoc(...).text`. As-built: no separate band-label chip —
  the band is already embedded in the sentence ("AUC 0.671 (weak). …"), so a separate
  uppercase label only duplicated it; the card matches the confusion verdict card.

Always visible (the explicit goal is that the explanation is present, not hidden behind a
toggle). The empty/`available=false` states are unchanged (interpretation only renders
when `available` data exists).

## As-built refinements (2026-06-20, from per-task code review)

Two presentation refinements emerged during implementation review; both keep the design
intent and improve it, and are recorded here so this spec stays a truthful reference:

1. **ROC `compare` thresholds aligned to the `aucBand` edges.** The original draft cut at
   `≤ 0.55` / `≤ 0.70`, which could pair the "near-random" band with "modestly better"
   (at 0.58) and the "acceptable" band with "modestly better" (at exactly 0.70) — a
   self-contradictory readout. Boundaries now sit at the band edges (`< 0.6` / `< 0.7`), so
   the band label and the comparison phrasing are always consistent.
2. **ROC band rendered inside the sentence, not as a separate chip.** `interpretRoc(...).text`
   already begins "AUC 0.671 (weak). …", so a separate uppercase band label was redundant
   and was dropped.

## Testing (TDD, vitest)

New `frontend/src/lib/model-performance/interpret.test.ts`:

- `describeModel`: each cohort token → correct subject/event; `hcp_adoption` not
  mis-matched as a patient cohort; legacy names; unknown → generic `known=false`.
- `interpretConfusion`: metric math on a known matrix (e.g. real `initiation_remibrutinib`
  tn=2946/fp=346/fn=1277/tp=506 → P≈0.594, R≈0.284, S≈0.895, acc≈0.680, F1≈0.385);
  degenerate denominators → `value: null`, `pct: "n/a"`; each archetype selected on a
  representative matrix; verdict contains the real counts + the domain event.
- `interpretRoc`: every band boundary (0.59, 0.6, 0.69, 0.7, 0.79, 0.8, 0.89, 0.9);
  each `compare` branch; generic fallback wording.

Plus a render assertion in `ModelPerformance.test.tsx` that, given an `available`
confusion/ROC fixture, the verdict/band text appears in the DOM.

## Files

- **new** `frontend/src/lib/model-performance/interpret.ts`
- **new** `frontend/src/lib/model-performance/interpret.test.ts`
- **edit** `frontend/src/pages/ModelPerformance.tsx` (wire `modelName`; render the two blocks)
- **edit** `frontend/src/pages/ModelPerformance.test.tsx` (render assertion)

## Honesty constraints (anti-mocking)

- No hardcoded metric values — everything computed from the response.
- A metric with a zero denominator reads **"n/a"**, never `0%`/`100%`.
- AUC bands are non-inflated (0.671 → "weak", not "good").
- Interpretation renders only when the API reports `available` data; the honest
  empty state is untouched.
