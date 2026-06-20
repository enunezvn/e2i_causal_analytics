# Feature Importance — Honest Labels + Strategic Insight (Design)

**Date:** 2026-06-20 · **Scope:** Option 1 (presentation + strategic insight). Option 2 (model
enrichment) is CONTRAINDICATED by measured evidence — see "Why 3 covariates".
**Page:** https://eznomics.site/feature-importance · **Example cohort:** initiation / Remibrutinib
**Workflow:** worktree `wt_feature_importance_insight` (branch `feat/feature-importance-strategic-insight`,
base `main`) + TDD red-first + codex review. Batched with #1063/#1059/#1060; **merge/deploy/live-run HELD.**

## User review (4 concerns, verbatim intent)
1. "hard to believe this cohort has only n=25 patients"
2. "why 9 features … is the user presented with Feature Rankings = 3? … oversimplification"
3. "beeswarm displays vertical dot groupings that seem periodic — normal?"
4. "no strategic insight interpretation to feature importance"

## Investigation findings (ALL code-grounded, not theory)
- **n=25 is a SHAP SAMPLE CAP, not the cohort size.** `COHORT_SAMPLE_SIZE=25`
  (`FeatureImportance.tsx:84`) → `sample_size` (the response's honest n_succeeded,
  `explain.py:330`). `_sample_entity_ids(model_type, limit=25)` (`explain.py:2002`) pulls a
  deterministic 25-ID prefix from `patient_journeys`. The cohort is **8,420 patients**
  (experiment doc: train 2103 / val 750 / test 492 / holdout 5075). Cap exists to keep the
  *cold* SHAP compute < the 30s client timeout; result is cached (durable cache works post-#1043,
  the `data_split="unassigned"` fix at `explain.py:2124`).
- **9 encoded vs 3 raw covariates — same info, two granularities.** `explain.py:80`: initiation =
  "3 RAW leakage-safe covariates → 9 ENCODED". `groupByCovariate` (`shap-covariates.ts`) folds the
  9 encoded columns (region one-hots ≈5 + numeric `X`+`X__isna` twins) back to 3 parents. Summary
  card shows encoded count (9, `FeatureImportance.tsx:801`), ranking badge shows grouped count
  (3, `:857`). User seeing BOTH 9 and 3 ⇒ **#1043 is LIVE** (pre-#1043 would show 9-and-9).
- **Beeswarm periodic vertical bands = EXPECTED.** LinearExplainer on a logistic model → a
  binary/one-hot feature takes only 2 SHAP values (x∈{0,1}) → dots align in 2 vertical bands; only
  continuous `disease_severity` spreads. Caption already says this (`FeatureImportance.tsx:945`) but
  it is buried in a chart `CardDescription`.
- **No strategic insight (genuine gap).** Only a generic per-feature blurb shown on click
  (`FeatureImportance.tsx:1122`). No cohort-level "what the drivers mean / what to do".

## Why 3 covariates (the user's precondition — empirically locked, NOT a collapse)
`feature_builder.py:23-34` + `docs/.../experiments/2026-06-14-initiation-features.md`: on REAL
synthetic Remibrutinib-initiation rows, 10 leakage-safe candidates were narrowed by MEASURED
holdout AUC (LogisticRegression, class_weight balanced):

| tier | columns | holdout AUC |
|------|---------|-------------|
| **A** | base 3 (`disease_severity, academic_hcp, geographic_region`) | **0.6709** |
| B | A + 4 PJ extras (age_at_diagnosis, engagement_score, insurance_type, urticaria_severity_uas7) | 0.6694 |
| C | B + 3 patient-keyed (comorbidity_count, prior_treatment_count, insurance_tier) | 0.6659 |

More features measurably *lowered* generalization → LOCK Tier A. Cross-split stability
val 0.685 / test 0.643 / holdout 0.671. LR coefficients: **academic_hcp 0.417, disease_severity 0.296,
region one-hots ≈ 0**. AUC≈0.67 (not ≈1.0) ⇒ no hidden leakage. `LEAKAGE_DENYLIST`
(`feature_builder.py:68-80`) bars post-decision columns. ⇒ "3" is honest AND optimal here;
enrichment is a settled, rejected experiment.

## Deliverables

### D1 — Honest labels (FeatureImportance.tsx)
- Summary "n = {sample_size} patients" → "n = {sample_size} sampled {patients|HCPs}" + tooltip
  ("mean |SHAP| over a {sample_size}-{grain} sample of the cohort"). NEVER imply cohort size.
- Reconcile the "{features.length} features" chip with the "Feature Rankings {covariateGroups.length}"
  badge so they don't read as contradictory: show **"{nCovariates} covariates · {nEncoded} encoded"**.
  Drive both from the same data (covariateGroups.length + global.features.length). No fabricated numbers.

### D2 — Beeswarm note prominence
- Promote the existing vertical-band explanation from the chart `CardDescription` into a small,
  always-visible callout near the beeswarm (Info icon), so the "is this normal?" question is answered
  on sight. Wording unchanged in substance (expected under a linear model; binary feats → bands).

### D3 — Strategic Interpretation panel (NEW, rule-based, NO LLM)
New pure module **`frontend/src/lib/feature-importance/interpret.ts`** (mirrors PR #1061
`model-performance/interpret.ts`: pure, deterministic, `"n/a"` on missing data, never fabricated).

`interpretGlobalImportance(groups: CovariateGroup<FeatureContribution>[], opts) → InsightReport` where
opts = { modelType, brand, sampleSize, grain }. Statements (all derived from live data):
- **Dominant driver + share**: top covariate by summed importance, and its % of total Σ|mean SHAP|.
- **Direction**: dominant driver raises vs lowers the predicted outcome (sign of net signed effect).
- **Concentration**: HHI-style — "concentrated in one driver" vs "spread across N drivers"
  (e.g. top-driver share ≥ 0.6 → concentrated).
- **Minimal-contribution flag**: covariates with ~0 mean|SHAP| (e.g. region) called out as
  "negligible contribution" — directly answers the segment-page-style "is region redundant" worry,
  but HONESTLY (region genuinely ≈0 here).
- **Honest caveats**: importance is computed over a {sampleSize}-entity sample; SHAP shows
  associations under the model, not causal effects (link users to the causal pages for that).
- **Model-design note (clearly-attributed static provenance, NOT a live metric):** "This deployed
  model uses {n} empirically-locked leakage-safe covariates; a 2026-06-14 holdout-AUC experiment
  found richer feature sets did not improve generalization." (Documentation fact; not presented as a
  live number, so it can't drift.)
- Render as an always-visible card in BOTH cohort and individual modes (individual = per-entity
  driver narrative from `top_features`), inside the existing `hasData` guard so honest-empty when no data.

> NOTE on AUC: the `/explain/global` response carries `base_value` but NOT AUC, so the panel does
> NOT display a live AUC (model quality lives on /model-performance). The "weak (≈0.67)" framing stays
> as attributed provenance only — never hardcoded as if live. This avoids the model-perf
> measurement-artifact trap (mixing sources).

## Tests (TDD red-first)
- `frontend/src/lib/feature-importance/interpret.test.ts`: dominant-driver+share, direction sign,
  concentration threshold, minimal-contribution flag, `"n/a"`/empty-safe, single-covariate edge.
- `frontend/src/pages/FeatureImportance.test.tsx` (extend): label reconciliation
  ("covariates · encoded"), "sampled" wording, strategic panel renders with drivers, honest-empty.
- Vitest: `--no-file-parallelism --pool=forks` + `NODE_OPTIONS=--max-old-space-size=2048` (OOM history);
  Radix Tabs need `userEvent` not `fireEvent`. Faithful gate: `tsc -b --noEmit`.

## Cheapest-disproof checks still queued (need a working tool)
1. Confirm #1043 is in live `main` (git) — strongly inferred from user seeing "Rankings 3".
2. Confirm live `/explain/global?model_type=initiation&brand=Remibrutinib` returns
   features.length=9, covariate-groups=3, sample_size=25 (faithful: browser MCP or authed curl).

## Gates / held
TDD green + `tsc -b` + ruff/vitest clean + codex review → PR. **HELD for user auth:** merge, deploy,
faithful live run. Batches with #1063 (segment-analysis), #1059 (#973 provenance), #1060 (label-gater).
