# #2031 cheapest disproof — is the production LinearDML seed-sensitive beyond its CI, and does a larger RF leaf fix it? (2026-09-12)

Scripts: `sweep_leaf_seed.py` (planted-truth DGP, seed 21, Remibrutinib, heterogeneous; production-mirrored fit:
RF(n_estimators=50, min_samples_leaf=LEAF, min_impurity_decrease=1e-7, random_state=SEED), LinearDML(discrete_treatment,
random_state=SEED), X = W = covariates, CI = `ate_inference(X).conf_int_mean()`), `sweep_live_frame.py` (the LIVE
Remibrutinib discovery frames through the route's own loader, n = 5000, SSOT adjustment sets, read-only). Outputs:
`summary_n1500.md`, `summary_n5000.md`, `summary_live.md`, `rows_*.jsonl`. ~11 min total, ≤ 1 GB.

## Premise tested
"Production nuisance config (RF leaf 5) makes small-ATE estimates seed-sensitive beyond their own CI; a larger leaf recovers
the planted ATEs with correct coverage." (Issue #2031, from the Lane F1 probe: seeds 42/7/123 gave 0.034 / 0.072 / 0.086
on `treatment_arm → persistent_180d` vs CI [−0.003, 0.074].)

## Result 1 — the premise is FALSE in general, TRUE for exactly one live pair
Live frames, 11 pairs × 6 seeds (`summary_live.md`):

| leaf | median seed-SD / SE | max seed-SD / SE (pair) | pairs where seed-42's CI excludes another seed's ATE |
|---|---|---|---|
| 5 | 0.46 | 0.89 (treatment_arm → persistent_180d) | 1 / 11 |
| 50 | 0.19 | 0.38 | 0 / 11 |
| 100 | 0.16 | 0.27 | 0 / 11 |

On 10 / 11 live pairs the seed spread is ≤ 0.6 SE at leaf 5 and every seed's estimate lies inside seed-42's CI. The one
exception is the pair the finding came from: `treatment_arm → persistent_180d`, seed SD 0.018 vs SE 0.020, seed 42 =
0.034 while the 6-seed mean is 0.066 (1.8 SD below). At leaf 50 the same pair reads 0.065 [0.027, 0.104] (seed SD 0.006):
**the live "null finding" on that pair (sensitivity WARNING, rcc WARNING 1.3 SE, REVIEW band) is a leaf-5 / seed-42 artefact**
— every rcc / bootstrap refit that landed +0.02–0.026 above it (Lane E cert, F1 table) was pointing at the leaf-50 value.

## Result 2 — a larger leaf removes seed variance and changes nothing else
- Live: leaf 50 shrinks seed SD 3–5× on every pair; mean ATE moves ≤ 0.014 on 10 pairs (and +0.031 on the artefact pair);
  SE unchanged (0.0149).
- DGP n = 1500 (`summary_n1500.md`): coverage 1.00 at every leaf; seed-SD / SE 0.52 → 0.34 → 0.20 → 0.12 (leaf 5/20/50/100);
  mean |bias| 0.014 → 0.011.
- DGP n = 5000 (`summary_n5000.md`): coverage 0.80 / 0.82 / 0.82 (leaf 5/50/100), seed-SD / SE 0.66 → 0.20 → 0.17,
  mean |bias| 0.019 → 0.017 → 0.016. Coverage does NOT improve with leaf because the misses are BIAS on two copay pairs
  (see Result 3), not variance.

## Result 3 — a DGP truth-attribute defect (separate finding)
`src/ml/synthetic/generators/patient_generator.py:687-705`: `copay_support → adherent_180d`, `copay_support → low_gap_180d`
and `psp_enrolled → adherent_180d` store `np.mean(list(rd_by_segment.values()))` — the UNWEIGHTED mean over 3 severity
segments — while every other arm/outcome weights by the per-row `segment`. With segment shares 0.54 / 0.31 / 0.16 the
unweighted truth overstates the population ATE by ≈ 0.02 (n = 5000: 0.120 vs 0.101, 0.115 vs 0.096, 0.099 vs 0.080).
Against the weighted truth, the leaf-50 DGP estimates are within 1 SE on psp and 1.5–1.8 SE under on the two copay pairs
(OLS with the registry covariates lands at 0.089 / 0.080 there too, so the residual is not the nuisance model). At n = 1500
the SE (0.027) hid this. No existing calibration test compares to these truths numerically (they check sign / bands), so
nothing broke; but a planted-truth COVERAGE gate — which #2031 asks for — cannot be stated until the attr is the
population-weighted mean.

## What this means for #2031
- The estimator is NOT broadly mis-calibrated; the CI is honest on 10 / 11 live pairs and on the DGP at both scales.
- min_samples_leaf 5 is the source of a rare but real low-tail draw that flips a CI-vs-zero verdict on one live pair.
  Leaf 50 removes it with no measurable cost (bias, SE, coverage unchanged; fit time unchanged at n ≤ 5000).
- Recommendation: raise `min_samples_leaf` 5 → 50 in `LinearDMLWrapper` (`energy_score/estimator_selector.py:649-666`)
  AND in `_reconstruction_nuisance_init_params` (`nodes/refutation.py:306`, must match or the reconstruction tolerance guard
  compares two different estimators), pinned by (a) a test that the two settings are the same object/value, (b) a DGP
  calibration pin: seed-SD / SE ≤ 0.5 on every planted pair at n = 1500 (today 0.20 at leaf 50), coverage ≥ 0.9 against the
  POPULATION-WEIGHTED truth (needs Result 3 fixed first), (c) the live re-estimation table above (already measured: every
  served Remibrutinib ATE moves ≤ 0.014 except the artefact pair, which moves 0.034 → 0.065 and leaves the null band).
- Fix Result 3 in the same lane (weighted mean, red-first test on the generator), or as a prerequisite issue.
