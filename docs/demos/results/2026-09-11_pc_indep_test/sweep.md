# PC independence-test sweep on the structural-recovery DGP (#2009)

Frames: `_make_frame(n, seed)` from `tests/unit/test_causal_engine/test_discovery/test_structural_recovery.py` (binary treatment/outcome, continuous covariates), sweep n in {500, 2000} x seeds 1-10, guided prod shape (anchored=[], declared=ALL, B=20, latent FCI diagnostic at the guided default), driven through the real `GraphBuilderNode` via the benchmark's `_build_dag`; the SHIPPED DAG is scored with `_structural_metrics`. chisq/gsq frames have the three continuous covariates quantile-binned to 10 levels in the harness; 0/1 columns untouched. kci: unmodified frame, n=500 points only, gated on the probe below.

Seam: `DiscoveryRunner.register_algorithm(PC, ForcedPC)`; `ForcedPC` is `PCAlgorithm` with only `_select_independence_test` overridden (plus timing and a cooperative per-point deadline). Main discover() and all 20 bootstrap resamples run on that instance. `pc_tests_used` is asserted per point.

Run: {'started_utc': '2026-09-11T08:48:50Z', 'budget_min': 40.0, 'point_timeout_s': 1800.0, 'pins': {'causal-learn': '0.1.4.3', 'numpy': '2.3.5', 'pandas': '2.3.3', 'networkx': '3.6.1'}, 'kci_probe': {'seconds': 82.66, 'converged': True, 'n_edges': 8, 'indep_test': 'kci', 'error': None, 'within_limit': True}, 'kci_not_attempted': [{'n': 500, 'seed': 1, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 2, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 3, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 4, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 5, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 6, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 7, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 8, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 9, 'expected_s': 1909, 'window_s': 1800}, {'n': 500, 'seed': 10, 'expected_s': 1909, 'window_s': 1800}], 'elapsed_min': 2.9}

## Per-test summary

| test | runs done / attempted | ACCEPT | AUGMENT | REVIEW | REJECT | exact | reversed | invented CC | omitted conf (adj set) | SHD>1 on ACCEPT | mean / max SHD | mean P | mean R | mean F1 | mean / max wall s | mean / max PC-only s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fisherz | 20 / 20 | 19 | 1 | 0 | 0 | 8 | 0 | 1 | 0 | 5 | 0.95 / 4.0 | 0.949 | 0.929 | 0.933 | 1.151 / 6.15 | 0.801 / 1.13 |
| chisq | 20 / 20 | 19 | 1 | 0 | 0 | 6 | 0 | 1 | 0 | 11 | 1.9 / 4.0 | 0.976 | 0.764 | 0.832 | 3.663 / 7.03 | 3.579 / 6.9 |
| gsq | 20 / 20 | 18 | 2 | 0 | 0 | 7 | 0 | 2 | 0 | 11 | 1.65 / 5.0 | 0.948 | 0.843 | 0.872 | 3.948 / 5.7 | 3.874 / 5.62 |

`omitted conf (adj set)` is the benchmark invariant (true confounders in the shipped adjustment set; in the prod shape the guarantee channel makes it hold by construction). Structural omissions (a true confounder not carrying BOTH conf->T and conf->Y in the shipped DAG): fisherz 8/20, chisq 12/20, gsq 10/20.

## By n

| test | n | done | ACCEPT+AUGMENT | withheld | mean SHD | mean recall | mean F1 | mean wall s |
|---|---|---|---|---|---|---|---|---|
| fisherz | 500 | 10 | 10 | 0 | 1.6 | 0.857 | 0.885 | 1.337 |
| fisherz | 2000 | 10 | 10 | 0 | 0.3 | 1.0 | 0.98 | 0.966 |
| chisq | 500 | 10 | 10 | 0 | 3.3 | 0.586 | 0.703 | 2.72 |
| chisq | 2000 | 10 | 10 | 0 | 0.5 | 0.943 | 0.961 | 4.605 |
| gsq | 500 | 10 | 10 | 0 | 2.9 | 0.729 | 0.776 | 3.289 |
| gsq | 2000 | 10 | 10 | 0 | 0.4 | 0.957 | 0.969 | 4.606 |

## Per-point

| test | n | seed | gate | basis | SHD | P | R | F1 | spurious | missing | reversed | invented CC | wall s | PC s | PC calls | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fisherz | 500 | 1 | accept | bootstrap_stability | 2 | 0.86 | 0.86 | 0.86 | region_south->academic_hcp | academic_hcp->treatment_arm | - | - | 6.15 | 0.76 | 21 | INVARIANT VIOLATION |
| fisherz | 500 | 2 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->treatment_arm | - | - | 0.6 | 0.5 | 21 |  |
| fisherz | 500 | 3 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; region_south->treatment_arm | - | - | 0.6 | 0.53 | 21 | INVARIANT VIOLATION |
| fisherz | 500 | 4 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->persistent_180d | - | - | 0.61 | 0.52 | 21 |  |
| fisherz | 500 | 5 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->treatment_arm; region_south->treatment_arm | - | - | 0.78 | 0.68 | 21 | INVARIANT VIOLATION |
| fisherz | 500 | 6 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 0.84 | 0.76 | 21 |  |
| fisherz | 500 | 7 | accept | bootstrap_stability | 2 | 0.86 | 0.86 | 0.86 | noise_cov->persistent_180d | academic_hcp->persistent_180d | - | - | 1.11 | 1.02 | 21 | INVARIANT VIOLATION |
| fisherz | 500 | 8 | augment | bootstrap_stability | 4 | 0.64 | 1.00 | 0.78 | noise_cov->persistent_180d; noise_cov->treatment_arm; prognostic_only->treatment_arm; region_south->persistent_180d | - | - | noise_cov, prognostic_only, region_south | 0.94 | 0.84 | 21 | INVARIANT VIOLATION |
| fisherz | 500 | 9 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->persistent_180d | - | - | 0.94 | 0.83 | 21 |  |
| fisherz | 500 | 10 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->treatment_arm | - | - | 0.8 | 0.74 | 21 |  |
| fisherz | 2000 | 1 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 0.93 | 0.85 | 21 |  |
| fisherz | 2000 | 2 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 1.01 | 0.94 | 21 |  |
| fisherz | 2000 | 3 | accept | bootstrap_stability | 1 | 0.88 | 1.00 | 0.93 | noise_cov->persistent_180d | - | - | - | 1.25 | 1.13 | 21 |  |
| fisherz | 2000 | 4 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 1.12 | 1.01 | 21 |  |
| fisherz | 2000 | 5 | accept | bootstrap_stability | 1 | 0.88 | 1.00 | 0.93 | prognostic_only->disease_severity | - | - | - | 0.95 | 0.87 | 21 |  |
| fisherz | 2000 | 6 | accept | bootstrap_stability | 1 | 0.88 | 1.00 | 0.93 | noise_cov->prognostic_only | - | - | - | 0.94 | 0.86 | 21 |  |
| fisherz | 2000 | 7 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 0.86 | 0.79 | 21 |  |
| fisherz | 2000 | 8 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 0.89 | 0.81 | 21 |  |
| fisherz | 2000 | 9 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 0.88 | 0.8 | 21 |  |
| fisherz | 2000 | 10 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 0.83 | 0.77 | 21 |  |
| chisq | 500 | 1 | augment | bootstrap_stability | 4 | 0.64 | 1.00 | 0.78 | noise_cov->persistent_180d; noise_cov->treatment_arm; prognostic_only->treatment_arm; region_south->persistent_180d | - | - | noise_cov, prognostic_only, region_south | 3.4 | 3.33 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 2 | accept | bootstrap_stability | 4 | 1.00 | 0.43 | 0.60 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.39 | 2.32 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 3 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; region_south->treatment_arm | - | - | 2.5 | 2.45 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 4 | accept | bootstrap_stability | 3 | 1.00 | 0.57 | 0.73 | - | academic_hcp->persistent_180d; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.63 | 2.56 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 5 | accept | bootstrap_stability | 3 | 1.00 | 0.57 | 0.73 | - | academic_hcp->treatment_arm; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 3.45 | 3.39 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 6 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm | - | - | 3.04 | 2.97 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 7 | accept | bootstrap_stability | 4 | 1.00 | 0.43 | 0.60 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.35 | 2.29 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 8 | accept | bootstrap_stability | 3 | 1.00 | 0.57 | 0.73 | - | academic_hcp->persistent_180d; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.28 | 2.2 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 9 | accept | bootstrap_stability | 4 | 1.00 | 0.43 | 0.60 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.87 | 2.81 | 21 | INVARIANT VIOLATION |
| chisq | 500 | 10 | accept | bootstrap_stability | 4 | 1.00 | 0.43 | 0.60 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.29 | 2.22 | 21 | INVARIANT VIOLATION |
| chisq | 2000 | 1 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 2.69 | 2.63 | 21 |  |
| chisq | 2000 | 2 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 3.64 | 3.55 | 21 |  |
| chisq | 2000 | 3 | accept | bootstrap_stability | 1 | 0.88 | 1.00 | 0.93 | noise_cov->region_south | - | - | - | 3.91 | 3.82 | 21 |  |
| chisq | 2000 | 4 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 4.67 | 4.58 | 21 |  |
| chisq | 2000 | 5 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm | - | - | 3.63 | 3.54 | 21 | INVARIANT VIOLATION |
| chisq | 2000 | 6 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->persistent_180d | - | - | 7.03 | 6.9 | 21 |  |
| chisq | 2000 | 7 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 5.69 | 5.58 | 21 |  |
| chisq | 2000 | 8 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 5.11 | 5.0 | 21 |  |
| chisq | 2000 | 9 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 5.03 | 4.86 | 21 |  |
| chisq | 2000 | 10 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->treatment_arm | - | - | 4.65 | 4.57 | 21 |  |
| gsq | 500 | 1 | accept | bootstrap_stability | 2 | 0.86 | 0.86 | 0.86 | region_south->academic_hcp | academic_hcp->treatment_arm | - | - | 3.8 | 3.72 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 2 | augment | bootstrap_stability | 4 | 0.64 | 1.00 | 0.78 | noise_cov->persistent_180d; noise_cov->treatment_arm; prognostic_only->treatment_arm; region_south->persistent_180d | - | - | noise_cov, prognostic_only, region_south | 2.89 | 2.83 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 3 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; region_south->treatment_arm | - | - | 3.37 | 3.32 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 4 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; prognostic_only->persistent_180d | - | - | 2.94 | 2.88 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 5 | accept | bootstrap_stability | 3 | 1.00 | 0.57 | 0.73 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; region_south->treatment_arm | - | - | 4.57 | 4.5 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 6 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm | - | - | 3.41 | 3.34 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 7 | accept | bootstrap_stability | 3 | 1.00 | 0.57 | 0.73 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; region_south->treatment_arm | - | - | 2.77 | 2.7 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 8 | augment | bootstrap_stability | 5 | 0.58 | 1.00 | 0.74 | noise_cov->persistent_180d; noise_cov->treatment_arm; prognostic_only->treatment_arm; region_south->persistent_180d; region_south->prognostic_only | - | - | noise_cov, prognostic_only, region_south | 2.98 | 2.88 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 9 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; region_south->treatment_arm | - | - | 3.52 | 3.47 | 21 | INVARIANT VIOLATION |
| gsq | 500 | 10 | accept | bootstrap_stability | 4 | 1.00 | 0.43 | 0.60 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm; prognostic_only->persistent_180d; region_south->treatment_arm | - | - | 2.64 | 2.57 | 21 | INVARIANT VIOLATION |
| gsq | 2000 | 1 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 4.15 | 4.06 | 21 |  |
| gsq | 2000 | 2 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 4.94 | 4.85 | 21 |  |
| gsq | 2000 | 3 | accept | bootstrap_stability | 1 | 0.88 | 1.00 | 0.93 | noise_cov->region_south | - | - | - | 3.99 | 3.9 | 21 |  |
| gsq | 2000 | 4 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 5.07 | 4.99 | 21 |  |
| gsq | 2000 | 5 | accept | bootstrap_stability | 2 | 1.00 | 0.71 | 0.83 | - | academic_hcp->persistent_180d; academic_hcp->treatment_arm | - | - | 3.8 | 3.73 | 21 | INVARIANT VIOLATION |
| gsq | 2000 | 6 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 5.7 | 5.62 | 21 |  |
| gsq | 2000 | 7 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 5.04 | 4.97 | 21 |  |
| gsq | 2000 | 8 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 4.46 | 4.39 | 21 |  |
| gsq | 2000 | 9 | accept | bootstrap_stability | 0 | 1.00 | 1.00 | 1.00 | - | - | - | - | 4.36 | 4.27 | 21 |  |
| gsq | 2000 | 10 | accept | bootstrap_stability | 1 | 1.00 | 0.86 | 0.92 | - | academic_hcp->treatment_arm | - | - | 4.55 | 4.49 | 21 |  |

## Reading against the decision rule

**chisq**: SHD 1.9 vs 0.95, recall 0.764 vs 0.929, invariant violations 0/1/0/11 vs 0/1/0/5 (reversed/invented/omitted/SHD>1-on-accept), mean wall 3.663 s vs 1.151 s (2x bound 2.3 s), 20 runs vs 20 -> does NOT pass the rule.

**gsq**: SHD 1.65 vs 0.95, recall 0.843 vs 0.929, invariant violations 0/2/0/11 vs 0/1/0/5 (reversed/invented/omitted/SHD>1-on-accept), mean wall 3.948 s vs 1.151 s (2x bound 2.3 s), 20 runs vs 20 -> does NOT pass the rule.

No alternative passes the rule: pin today's fisherz selection (guard + characterization test), do not change the selector.

**Where the loss sits.** Paired per (n, seed) on SHD of the shipped DAG — chisq: fisherz better 10, tie 9, chisq better 1; gsq: fisherz better 10, tie 9, gsq better 1. The alternatives lose almost entirely at n=500 (mean recall fisherz 0.857, chisq 0.586, gsq 0.729): a 10-level quantile bin turns every conditional test on a binned covariate into a sparse contingency table (10 x 10 x 2 ... cells over 500 rows), so chisq/gsq lose power and PC drops true edges (the missing edges in the per-point table are conf->T / conf->Y and the instrument edge, exactly the recall loss). At n=2000 the three tests are within one edge of each other. PC-only time is ~4.5x fisherz for chisq/gsq (the binned frame is a heavier contingency path than a partial correlation) — over the 2x bound on its own.

**kci.** The single-frame probe (one forced-kci `discover()` on n=500 seed 1, guided prod config, no bootstrap) converged in 82.66 s with 8 edges — under the 120 s gate, so kci was scheduled. But a prod-shape point is 1 + B=20 `discover()` calls, i.e. ~1736 s (~29 min) per point, ~1508x fisherz's mean wall-clock: it cannot pass the 2x rule regardless of what it recovers. The harness's kci pre-check refused to START a point the per-point window could not finish (expected ~1909 s > the 1800 s per-point cap; the 40-min budget could have fitted exactly one point), so kci has 0 completed points — recorded as `kci_not_attempted` in run_meta.json. Timing alone settles kci's place under the rule; its recovery on this DGP is unmeasured.

**Reading the `INVARIANT VIOLATION` notes.** The per-point flag is raised when a row fails ANY of the four invariants; on fisherz every flagged row is the `SHD > 1 on ACCEPT/AUGMENT` one (n=500 seeds 1, 3, 5, 7 at SHD 2, seed 8 AUGMENT at SHD 4), never a reversed edge and never an omitted true confounder. The shipped benchmark asserts `SHD <= 1 on ACCEPT` (`test_structural_error_stays_within_one_edge`) under the HONEST-PRIORS shape (anchored = the true confounders); under the PRODUCTION shape measured here (anchored=[], declared=ALL — docstring item 2) the pinned band is F1 0.78-1.00 at n=500 with no SHD assertion, and this sweep reproduces it (fisherz n=500 F1 0.83-1.00, n=2000 mean F1 0.98). The `invented CC` column is likewise a fallback artefact, not a test effect: every counted row is an AUGMENT (fisherz n=500 seed 8; chisq seed 1; gsq seeds 2 and 8) where the shipped DAG is the curated all-covariate manual assertion plus corroborated edges, which places every DECLARED covariate as a common cause by construction. The invariant the benchmark actually asserts for the adjustment set — true confounders present, `omitted conf (adj set)` — is 0/20 under all three tests. The structural-omission count (a true confounder missing conf->T or conf->Y in the DAG itself, fisherz 8/20) is informative only: the adjustment guarantee channel conditions on the declared covariates regardless, which is why the benchmark records it as a recall loss and not a correctness one.

**Verdict.** fisherz is the best of the three measured tests on every recovery number (SHD, recall, F1, exact recoveries) and the cheapest by >3x; chisq and gsq lose recall at n=500 and exceed the 2x wall-clock bound, kci exceeds it by ~2 orders of magnitude. No invariant that fisherz holds is held better by an alternative. Under the decision rule the selector stays on fisherz for mixed binary/continuous frames; the follow-up is the guard + a characterization test that names this measurement, not a selector change. This agrees with the 2026-09-02 all-binary measurement (docstring item 5: chisq F1 0.943 vs fisherz 0.953) and extends it to the live mixed shape, where the gap is larger (0.832 vs 0.933).
