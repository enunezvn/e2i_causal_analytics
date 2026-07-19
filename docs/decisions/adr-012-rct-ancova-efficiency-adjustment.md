# ADR-012: RCT questions use baseline-ANCOVA efficiency adjustment, not confounder machinery

**Date**: 2026-07-13 | **Status**: Accepted | **Implemented by**: PRs #1217 (migration 106), #1219; parent arc #1188–#1191

## Context

The causal engine treats real-world-evidence (RWE) questions and randomized (design-declared) treatments differently since the #1188–#1191 fix set: randomized treatments must not run confounder-adjustment machinery built for observational data. But the plain RCT contrast leaves precision on the table — baseline covariates that are *prognostic* for the outcome can shrink the confidence interval without changing the estimand (classic ANCOVA). Two defects surfaced on the way:

- The DML wrappers' `effect_inference(X).conf_int_mean()` silently fell back to a heterogeneity-spread formula ~50× too narrow; fixed platform-wide with `ate_inference(X)` (observational CIs widened — intended).
- The synthetic DGP's baselines had zero prognostic signal (within-arm R² ≈ 0.0004), making "CI narrower" unsatisfiable until the DGP was made prognostic.

## Decision

1. **Opt-in efficiency adjustment**: the RCT path exposes `baseline_candidates` (`GET /api/causal/variables`) and an `adjust_baselines` flag. Covariate-capable estimators receive baselines as efficiency controls (X=W); the result is labeled `adjustment_type: "efficiency"` — explicitly **not** `"confounding"`.
2. **The OLS anchor stays unadjusted** (zero-width covariate frame) so there is always a naive contrast to compare against.
3. **Refutation** threads baselines into the DoWhy reconstruction only for covariate-selected winners; an OLS-selected result stays unadjusted end to end.
4. **E-value sensitivity gate is skipped for design-declared randomized treatments** (PR #1219): unmeasured-confounding sensitivity analysis is a category error under randomization. The skip is per-treatment and fail-closed — only treatments the design declares randomized qualify.

## Consequences

- (+) Measured CI narrowing of 7–10% vs the naive contrast at production n, with the ATE tracking planted truth through refutation.
- (+) Honest interval semantics platform-wide (the 50×-too-narrow fallback is gone everywhere, not just for RCTs).
- (−) Adjusted and naive ATEs legitimately differ by the chance-imbalance correction (β·Δx̄) — tests must assert against planted truth, not the noisy naive contrast.
- (−) The efficiency/confounding distinction is carried in metadata; downstream consumers must not conflate the two.
