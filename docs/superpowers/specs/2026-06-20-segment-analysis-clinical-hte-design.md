# Segment Analysis — Clinical HTE, Agent-Driven (Design)

**Date:** 2026-06-20
**Branch:** `feat/segment-analysis-clinical-hte` (worktree `wt_segment_hte`, base `main`)
**Page:** https://eznomics.site/segment-analysis (Heterogeneous Optimizer, Tier 2)

## Problem (user review, 2026-06-20)

The page is "an afterthought": meager config (single frozen treatment+outcome), CATE only by
region, tautological feature importance, only high/low responders (no mid), no drill-down,
policies/insights are "just reporting", uplift "not implemented". It is meant to be
**agent-driven** with **ALL HTE effects surfaced for exploration**.

## Root-cause findings (code + cheapest-disproof, not theory)

1. **Substrate lock.** FE hardcodes `treatment=engagement_score`, `outcome=conversion_rate`,
   `segment_vars=['region']`, `effect_modifiers=['region']` (SegmentAnalysis.tsx:330,385–386),
   pinned to `business_metrics/per_hcp_rollup` — the one substrate with a single planted causal
   pair and one segmentable column. Any other column → `42703`. This single decision causes
   complaints 1–3.
2. **Feature importance is keyed by `effect_modifiers`** (cate_estimator). With one modifier
   (region), importance over a single feature is tautological *by construction*.
3. **Multi-dim CATE already works**: `cate_estimator` loops `for segment_var in segment_vars`
   and emits `cate_by_segment[var]` per dimension. The page only ever passes `['region']`.
4. **No mid bucket**: `segment_analyzer` emits `high` (|CATE|≥1.5|ATE|) and `low` (|CATE|≤0.5|ATE|)
   only; the middle band is never emitted, though `responder_type` Literal includes `"average"`.
5. **`strategic_interpretation` dropped at the route**: `profile_generator` computes a 3-tier
   business narrative; `_execute_segment_analysis` (segments.py:1212–1240) never maps it (nor
   hierarchical fields) into `SegmentAnalysisResponse`.
6. **Uplift node is not wired into the graph.** `graph.py:128–134` wires only
   estimate_cate→analyze_segments→hierarchical_analysis→learn_policy→generate_profiles.
   `uplift_analyzer.py` is fully coded but never imported/added → `overall_auuc` never set →
   `_convert_uplift_metrics` returns None → Uplift tab structurally empty on ANY substrate.

### Cheapest-disproof (load-bearing assumption SURVIVED, for free)

Assumption: `patient_journeys` has a treatment→outcome pair whose effect **varies across clinical
segments** (recoverable heterogeneity). Disproof = read the DGP:
- `patient_generator.py:118` `binary_outcome_with_cate(treatment_arm, confounders, segment, latent_cate_map,…)` → per-patient `tau_i`.
- `:111` `segment = assign_segment(confounders["disease_severity"])`; `:358–366` `cate = where(severity>7 → 0.50 strong, …)` — effect varies by disease-severity segment (high/med/low).
- `:112` `brand_scaled_cate(brand_enum)` — and by brand. `:256` `df.attrs["cate_by_segment"]=cate_map`.
Conclusion: heterogeneity is planted **by construction**; the planted high/med/low severity
segments map onto the high/mid/low responder buckets. Sibling causal-analysis page already
confirmed CausalForestDML recovers this effect on the same frame (+8.5pp, p≈0). The full
end-to-end agent run is **memory-gated** on the dev box → verified in CI + a gated live run.

## Design

### Data substrate & contract (reuse the proven gold-standard config from causal.py)

- Source: `patient_journeys` (gold-standard, `include_synthetic=true` — deployment includes synthetic).
- **Treatment** (selectable, curated): `treatment_arm` (default), `treatment_initiated`.
- **Outcome** (selectable, curated): `persistent_180d` (default), `discontinued_180d`, `treatment_initiated`.
- **Server-side load + `tier0_data` passthrough (NOT connector-fetch).** The route loads the
  gold-standard frame server-side with `include_synthetic=True` (patient_journeys rows are
  `is_synthetic=true`; the connector defaults `include_synthetic=False` → silent empty — codex
  HIGH-2), applies the brand filter, BANDS continuous columns (codex HIGH-4), and passes the
  prepared frame as `tier0_data`. CATE, hierarchical, AND uplift all consume `tier0_data`
  (priority-1) so there is ONE fetch, one banded frame (resolves double-fetch, MED-1). Fail
  CLOSED with a specific "synthetic substrate returned no rows" error if the frame is empty —
  never show empty/fabricated results.
- **Effect modifiers (X → heterogeneity model + feature importance)** = the segmentable clinical
  set: `disease_severity, age_at_diagnosis, academic_hcp, ecog_performance_status, egfr,
  proteinuria_g_day, ldh_ratio, urticaria_severity_uas7` (+ `geographic_region` one-hot). 100%
  -populated-with-variance columns verified live in #1027.
- **Confounders (W → pure controls, NOT in X)** = `engagement_score`. NO variable appears in both
  X and W (codex MED-2). De-confounding for the X covariates happens via the DML nuisance models
  (fit on X); keeping pure controls in W (not X) avoids the ~5.7x CI inflation the page already
  documented for continuous-covariate-in-X.
- **Segment dimensions (post-hoc CATE breakdown)** = banded categoricals that EXIST in the
  prepared frame: `disease_severity_band` (low/med/high, the planted DGP segments),
  `geographic_region`, `ecog_performance_status`, `academic_hcp`, `age_band`. Agent emits
  `cate_by_segment` per dimension; FE ranks by heterogeneity.
- **Brand**: cohort FILTER (data-driven dropdown like `/causal/brands`), applied server-side
  during the load — not a covariate.
- **Curated allowlist enforced SERVER-SIDE (codex HIGH-3).** `treatment`/`outcome` are selectable
  only from the curated valid set (treatment_arm/treatment_initiated × persistent_180d/
  discontinued_180d/treatment_initiated); `segment_vars`/`effect_modifiers`/`confounders` are
  fixed to the clinical allowlist. The route rejects any out-of-allowlist column with HTTP 400
  BEFORE graph invocation — this is what makes "agent-driven + selectable" safe without
  re-introducing the `42703` failure the page was locked down to avoid. Reuse causal.py's
  `_CAUSAL_DATASET_SPECS["patient_journeys"]` as the SSOT for the valid set.

### Backend changes

1. **uplift_analyzer.py** (codex HIGH-1): add `data_connector=None` to `__init__` and set
   `self.data_connector`; add `tier0_data` priority-1 read in `_get_data` mirroring
   cate/hierarchical so it consumes the prepared frame; make `execute` fully non-fatal (catch the
   `RuntimeError`/any exception → append warning, return state without `status="failed"`), so a
   uplift failure can NEVER abort the graph.
2. **graph.py**: import + wire `uplift_analyzer` with the shared `data_connector`, after
   `hierarchical_analysis`, via a plain `add_edge` (NOT a status-conditional edge) so it is
   structurally non-fatal. Gate via `enable_uplift: bool = True` mirroring `enable_hierarchical`.
   Order: …analyze_segments → hierarchical_analysis → uplift_analysis → learn_policy →
   generate_profiles.
3. **segment_analyzer.py**: emit `mid_responders` (band `0.5|ATE| < |CATE| < 1.5|ATE|`,
   `responder_type="average"`); include mid count in `segment_comparison` (codex LOW-1); keep
   behaviour identical when none qualify.
4. **state.py**: add `mid_responders: Optional[List[SegmentProfile]]`.
5. **policy_learner.py / profile_generator.py**: include mid count in `optimal_allocation_summary`
   / `executive_summary` so narratives don't imply exactly two buckets (codex LOW-1).
6. **segments.py**:
   - **Server-side gold-standard loader** (codex HIGH-2/HIGH-4/MED-1): reuse causal.py's
     gold-standard load path (`MLDataLoader` / the `/estimation-data` loader) with
     `include_synthetic=True` + brand filter; band `disease_severity→_band`, `age_at_diagnosis→
     age_band`; pass the prepared frame as `tier0_data` in `initial_state`. Fail-closed with a
     specific error when empty.
   - **Curated allowlist guard** (codex HIGH-3): validate treatment/outcome/segment/effect/
     confounder against the patient_journeys allowlist (SSOT = causal.py `_CAUSAL_DATASET_SPECS`)
     and reject out-of-set columns with HTTP 400 BEFORE graph invocation.
   - `SegmentAnalysisResponse`: add `strategic_interpretation`, `mid_responders`,
     `segment_heterogeneity` (I²), `n_segments_analyzed`, `segmentation_method_used`,
     `overall_hierarchical_ate`, `hierarchical_segment_results`, `segment_comparison`,
     `uplift_by_segment`. Map all from `result` (final graph state — the route reads
     `graph.ainvoke` result directly, NOT `agent._build_output`, codex LOW-2).
   - `RunSegmentAnalysisRequest`: add `brand` (Optional) + curated `treatment_var`/`outcome_var`.
   - `GET /segments/datasets` → curated treatment/outcome options + brands (data-driven FE config).
7. **Label-gater composition (graceful)**: response off-label fields stay optional; when the
   gater (PR #1060) merges, off-label responder/policy segments are flagged. No hard coupling.

### Frontend rebuild (SegmentAnalysis.tsx + segments.ts)

- **Config**: brand dropdown (data-driven) + curated treatment/outcome dropdowns (default
  treatment_arm→persistent_180d). Copy: "Agent estimates CATE across all clinical segments."
- **CATE tab**: one chart per segment dimension, ordered by heterogeneity; granular
  multi-covariate feature importance with a real caption.
- **Responders tab**: High / **Mid** / Low (3 columns); each card click → drill-down panel
  (defining features, CATE + CI, sample, recommendation, label verdict when present).
- **Policies tab**: strategic framing (who/why/expected lift) + `optimal_allocation_summary`.
- **Uplift tab**: real AUUC/Qini/targeting + per-segment uplift; honest empty-state if a run
  genuinely lacks it (no fabricated bars).
- **Insights tab**: surface `strategic_interpretation` (the 3-tier narrative) + executive
  summary + key insights + I² heterogeneity + impact framing.

## Test plan (TDD red-first, real data, no behaviour mocking)

- graph: assert `uplift_analysis` node present and runs; `overall_auuc` populated on a fixture
  frame with binary treatment+outcome and within-segment variance.
- segment_analyzer: mid bucket emitted for a CATE in the mid band; high/low unchanged; empty when
  none qualify.
- route: response carries strategic_interpretation + hierarchical + mid_responders +
  uplift_by_segment (converter guard, mirrors gater converter test).
- request: brand threads into initial_state; default-off byte-identical for legacy callers.
- FE: vitest for mid column + drill-down + honest uplift empty-state; `tsc -b` clean.
- Gated faithful: full agent run on live `patient_journeys` (memory-gated) — confirm multi-dim
  CATE, non-trivial feature importance, populated uplift, mid bucket.

## Gated (held for user authorization)

1. Merge + deploy.
2. Live end-to-end faithful run (real agent + live patient_journeys + brand filter).
3. Label-gater composition lights up after PR #1060 merges.

## Codex review — adopted revisions (pre-build, REVISE-BEFORE-BUILD)

All six findings adopted into the design above:

- **HIGH-1 — uplift wiring would abort the graph.** `UpliftAnalyzerNode.__init__` (uplift_analyzer.py:63)
  takes no `data_connector`; `_get_data` raises `RuntimeError` when mock disabled (`:231`), re-raised
  by `execute` (`:180`). → Add `data_connector` arg + `tier0_data` read + fully non-fatal `execute`;
  wire via plain `add_edge`.
- **HIGH-2 — silent empty frame.** patient_journeys is `is_synthetic=true`; connector defaults
  `include_synthetic=False` (supabase_connector.py:96); cate calls it without override
  (cate_estimator.py:536). → Server-side load with `include_synthetic=True` + `tier0_data` passthrough;
  fail-closed when empty.
- **HIGH-3 — 42703 injection.** segments.py forwards arbitrary column strings (`:1174`) with no
  allowlist (causal.py:1991 has one; segments does not). → Server-side curated allowlist guard, 400 on
  violation.
- **HIGH-4 — continuous segment columns.** cate_estimator iterates raw `df[segment_var].unique()`
  (`:707`), `<10` rows skipped → one segment per float / all-skipped. → Band server-side before
  `tier0_data`; segment only on banded categoricals that exist.
- **MED-1 — double fetch.** Shared connector avoids dual clients, not dual reads (cate `:536`,
  uplift `:217`). → Single server-side load consumed by all three nodes via `tier0_data`.
- **MED-2 — X/W overlap unspecified.** → X = segmentable clinical set, W = pure controls
  (`engagement_score`); no variable in both.
- **LOW-1 — mid bucket downstream.** Converter already accepts `average` (segments.py:1303) but
  `segment_comparison`/policy/profile imply two buckets. → Thread mid count through all summaries.
- **LOW-2 — map from final state.** Route reads `graph.ainvoke` result directly (`:1206`), not
  `_build_output`. → Mapping targets `result`; `_build_output` expanded only if orchestrator needs it.

## Addendum 2026-09-03 (wave 53) — question slots never enter X; clinical axes run as 0/1

Since #1321 a brand's own clinical column is BOTH an effect modifier in the set above AND
that brand's treatment axis (Remibrutinib `urticaria_severity_uas7`, Kisqali `disease_stage`,
Fabhalta `complement_inhibitor_status`). Live `seg_05f29d1b3295` (uas7 → persistent_180d)
put the treatment inside X: the median-split T was a deterministic function of an X column,
CausalForestDML's propensity model was perfect (AUC 1.000, zero residual) and the forest
returned ATE −0.514 on a 0/1 outcome (per-segment CATE −1.6..+0.4) against a planted +0.150.

Contract changes (all red-first tested):

- **X = brand-scoped modifier set MINUS the question's treatment and outcome**
  (`_segment_effect_modifiers(brand, treatment_var=, outcome_var=)`), matching the causal
  page's submit-path dedup. The nodes enforce the same invariant themselves
  (`heterogeneous_optimizer/design.sanitize_effect_modifiers`), so no caller can reintroduce it.
- **A #1321 axis in a QUESTION slot is loaded as its 0/1 contrast** via
  `_CAUSAL_NUMERIC_DERIVATIONS` — the contrast the option label states ("UAS7 ≥ 28") and the
  causal_paths edge was validated on; the two TEXT axes previously reached cate_estimator as
  strings and failed closed. The same column as an effect modifier stays raw.
- **All three nodes binarize a continuous treatment by ONE rule**
  (`design.binarize_treatment`, strictly above the median). The uplift node previously handed
  CausalML the raw score (27 groups, control "16.0"), so the cross-library validator compared
  two estimands (9% "agreement").

Faithful in-process run on the real Remibrutinib cohort after the fix: ATE +0.153, CATE
+0.08..+0.26 across 12 segments, cross-library agreement 0.776 (sign 1.00) → PASSED.
