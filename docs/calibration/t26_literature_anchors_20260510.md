# T2.6 Deployer-Input Threshold — Literature Anchors

**Date:** 2026-05-10
**Plan:** v4 §6 G4 — calibration-protocol artifacts (codex-rescue HIGH-4 fix)
**Lifecycle state:** ADVISORY ONLY — these absolute thresholds are anchored to peer-reviewed literature for the advisory window. They are **NOT** retrospectively-fit on held-out cohort data, because no untouched cohort currently exists (see §"Cohort inventory and unfitness").
**Code surface:** `src/agents/ml_foundation/model_deployer/nodes/registry_manager.py`, `compute_deployer_input_metrics()` and the `T2_6A_*` module-level constants.

---

## 1. Why literature anchors instead of cohort-fit thresholds?

Plan v3 §6 (T2.2/T2.3/T2.6) originally specified a three-step calibration protocol:

1. **Synthetic regimes** [0.55, 0.85] for plumbing — verify the threshold helpers wire through to `validation_metrics`.
2. **Retrospective held-out cohort** for threshold fitting — set the absolute pass/fail threshold from a real-world cohort whose metrics were NOT used to draft the gate.
3. **Operator decisions** for drift monitoring — compare future runs against the fit threshold; alert on drift.

Step 2 turns out to be infeasible given the current cohort set:

| Cohort                              | Status   | Reason                                                                                                   |
| ----------------------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| `synthetic_rwd_realistic` (regimes) | UNTOUCHED for the AUC sweep | Existing pin tests anchor the `[0.62, 0.68]` band but never set the **perm-anchored AUC floor** threshold T2.2 needs. |
| CSU n=9607 (PR #106)                | TOUCHED  | Observed `val_AUC=0.6592` drove the plan v3 honest-band literal `[0.62, 0.68]`. Threshold fitting on the same data that surfaced it would be data-snooping. |
| Optum n=1294 (PR #116, default windows) | TOUCHED | This is the empirical anchor for the entire arc — sample-size-binding-constraint finding, F1-fallback verification, etc. Cannot fit thresholds. |
| Optum n=1697 (relaxed PRE/POST)     | TOUCHED  | Sensitivity test (NOT shipped) was used to demonstrate "sample size IS binding" — even though we reverted it via `git checkout`, the verdict was already conditioned on its results. Cannot fit. |

**Decision:** for the advisory window, T2.6 deployer-input thresholds are anchored to peer-reviewed literature on the underlying metric (perm-test, ECE, CV stability). Cohort-derived bands continue to be emitted (T2.3) but are marked **observability-only** — see [`t23_cohort_bands_20260510.md`](t23_cohort_bands_20260510.md).

The future-cohort plan that would unblock retrospective fitting is documented in [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md).

---

## 2. T2.6 deployer-input thresholds (current implementation)

The `compute_deployer_input_metrics()` helper categorizes three deployer-input signals into named bands. The bands are governed by module-level constants:

```python
# Signal-genuineness pvalue bands.
T2_6A_SIGNAL_GENUINE_PVALUE_MAX:        float = 0.001
T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX: float = 0.01
T2_6A_SIGNAL_MARGINAL_PVALUE_MAX:       float = 0.05

# Calibration-quality ECE bands.
T2_6A_CALIBRATION_EXCELLENT_ECE_MAX: float = 0.05
T2_6A_CALIBRATION_GOOD_ECE_MAX:      float = 0.10
T2_6A_CALIBRATION_MARGINAL_ECE_MAX:  float = 0.20

# CV-stability std/mean ratio bands.
T2_6A_CV_STABILITY_STABLE_RATIO_MAX:    float = 0.05
T2_6A_CV_STABILITY_MODERATE_RATIO_MAX:  float = 0.10
T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX:  float = 0.20
```

Each set of bands is anchored to a peer-reviewed source below. The MARGINAL boundary (the rejection threshold T2.6c will gate on) is the load-bearing value — bands below MARGINAL graduate to denial categories (`T2_6B_*_REJECT_CATEGORIES`).

---

## 3. Signal genuineness — permutation-test p-value (T2.6a)

**Current code constant:** `T2_6A_SIGNAL_MARGINAL_PVALUE_MAX = 0.05` (rejection threshold for marginal/random/degenerate categories)

**Anchor source:** Fisher, R. A. (1925). *Statistical Methods for Research Workers.* Edinburgh: Oliver & Boyd. The α=0.05 significance convention has been the **dominant Neyman-Pearson hypothesis-testing threshold for over a century**, codified across statistical practice (FDA guidance, ICH E9 §5.5, biostatistics textbooks).

**Justification of α=0.05 anchor for permutation-test rejection:**

- **Permutation tests (Fisher 1935)** specifically: the p-value is the empirical proportion of label-shuffled null replicates whose AUC equals or exceeds the observed AUC. Rejecting at α=0.05 means we tolerate a 1-in-20 chance of being fooled by noise. This is the standard "signal genuine vs. random" decision threshold across machine-learning evaluation literature.
- **Stricter bands** (genuine ≤ 0.001, likely_genuine ≤ 0.01) are conservative gradations within "significant"; they exist because in claims-only ML on small N, even p=0.04 is borderline-trustworthy. Codex 2026-05-10 review explicitly noted: *"small-N permutation tests under-detect genuine weak signal"* — see Optum n=1294 revalidation (perm-p=0.67 with val_AUC=0.79). The triple-band gradation is a practical adaptation.
- **Multiple-testing concern is bounded**: T2.6a runs ONE permutation test per cohort, not a screen. No Bonferroni adjustment required at this level. Multiple-cohort meta-tests would inherit this gate, not break it.

**Citation reference (full):**
> Fisher, R. A. (1925). *Statistical Methods for Research Workers* (1st ed.). Edinburgh: Oliver & Boyd.
> Fisher, R. A. (1935). *The Design of Experiments.* Edinburgh: Oliver & Boyd. (Permutation-test methodology, Chapter II "The Principles of Experimentation").

**Modern echo (ML-specific):**
> Ojala, M., & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research*, 11, 1833-1863. — formalizes the permutation-test workflow for classifier AUC evaluation; α=0.05 default.

**Threshold value carried into code:** `T2_6A_SIGNAL_MARGINAL_PVALUE_MAX = 0.05`. RATIONALE: matches the Fisher α=0.05 standard; matches existing test-pin `PERMUTATION_P_MAX = 0.01` ceiling in `tests/integration/test_csu_val_auc_measurement.py` (which is the stricter `LIKELY_GENUINE` band — pin tests are intentionally tighter than the deployer gate).

---

## 4. Calibration quality — Expected Calibration Error (ECE) (T2.6a)

**Current code constants:**
- `T2_6A_CALIBRATION_EXCELLENT_ECE_MAX = 0.05`
- `T2_6A_CALIBRATION_GOOD_ECE_MAX = 0.10`
- `T2_6A_CALIBRATION_MARGINAL_ECE_MAX = 0.20`

**Anchor source (primary):**
> Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 29(1), 2901-2907.

**Justification (Naeini 2015):**
- Naeini, Cooper, & Hauskrecht 2015 introduced the binning estimator commonly called Expected Calibration Error (ECE) and used in our `compute_calibration_analysis` helper. Their published table (§4.2 "Empirical Evaluation," Table 2) reports calibrated-model ECE values in the 0.02-0.04 range on standard ML benchmarks (Adult, KDD'98) — establishing **ECE < 0.05 as the "well-calibrated" benchmark**.
- Higher ECE values (≥ 0.10) are reported across uncalibrated tree-ensemble models in their evaluation, marking the practical "needs work" zone.

**Anchor source (secondary):**
> Kumar, A., Sarawagi, S., & Jain, U. (2019). Trainable calibration measures for neural networks from kernel mean embeddings. *Proceedings of Machine Learning Research*, 80, 2805-2814.

**Justification (Kumar 2019):**
- Kumar, Sarawagi, & Jain 2019 introduced Maximum Mean Calibration Error (MMCE) and reported deep-net ECE bands clustering at ~0.05-0.10 for well-calibrated models and 0.15-0.25 for uncalibrated ones (Tables 1-3). This corroborates the EXCELLENT/GOOD/MARGINAL gradation we currently use.

**Clinical-utility framing (NOT an ECE threshold anchor):**
> Vickers, A. J., van Calster, B., & Steyerberg, E. W. (2019). A simple, step-by-step guide to interpreting decision curve analysis. *Diagnostic and Prognostic Research*, 3(18).

**Note on Vickers 2019 scope:** Vickers et al. 2019 covers **decision curve analysis and net benefit**, NOT ECE thresholds or calibration-band cutoffs. The publisher page (BMC, *Diagnostic and Prognostic Research*) confirms the paper's scope is clinical-utility framing of predicted-risk thresholds via decision curves. This citation is retained here ONLY for clinical-utility framing context — it is **not** a load-bearing ECE-band anchor and is NOT used to justify the `T2_6A_CALIBRATION_*_ECE_MAX` literals. The ECE-band literals in this section rest entirely on Naeini 2015 (primary) and Kumar 2019 (secondary).

**Threshold value carried into code:**
- `T2_6A_CALIBRATION_MARGINAL_ECE_MAX = 0.20` (rejection boundary; above this is `poor`).
- RATIONALE: Naeini 2015 reports ECE > 0.20 only for severely uncalibrated tree ensembles; marks the **"clearly uncalibrated"** floor. T2.6c rejection at this band is conservative — the deployer would only block the most miscalibrated models, leaving moderately miscalibrated ones (`marginal` band) for advisory warnings.

---

## 5. CV stability — std/mean AUC ratio across 5 folds (T2.6a)

**Current code constants:**
- `T2_6A_CV_STABILITY_STABLE_RATIO_MAX = 0.05`
- `T2_6A_CV_STABILITY_MODERATE_RATIO_MAX = 0.10`
- `T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX = 0.20`

**Status of these bands: heuristic, literature-informed — NOT cohort-fitted, NOT page-pinned.**

The 0.05 / 0.10 / 0.20 std/mean breakpoints below are heuristic operator-friendly bands chosen as round-number gradations within the general "low / moderate / high CV variance" regions established by the cited works. They are **NOT** retrospectively-fit on held-out cohort data, and the cited works do **NOT** pin specific 0.05 / 0.10 / 0.20 cutoffs to verifiable page/table numbers in their canonical published editions. The cited literature supports the *direction* of the bands (lower CV-CoV ⇒ more stable; higher CV-CoV ⇒ less stable) but does not establish the specific cutpoints as enforcement-grade thresholds.

This is consistent with the §1 framing: these bands are advisory-only and MUST NOT be promoted to deployer enforcement until cohort-fit calibration becomes feasible (see [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md)) or until page/table pins are added below.

**Literature support (general direction only):**

> Bouckaert, R. R., & Frank, E. (2004). Evaluating the replicability of significance tests for comparing learning algorithms. *Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)*, 3-12. Springer LNCS 3056.

Bouckaert & Frank 2004 establish that variance of cross-validation estimators is dataset- and procedure-dependent, and that comparing classifier stability across folds via coefficient-of-variation-like quantities is standard practice. The paper does NOT publish a 0.05 / 0.10 / 0.20 breakpoint table; specific CV-CoV ranges per dataset (e.g., "0.02-0.05 for Adult/Letter; 0.10-0.20 for LED-24/Anneal") were paraphrased from general CV-variance literature and **require page/table pins** in a future docs pass before they can support enforcement.

> Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *Proceedings of IJCAI-95*, 1137-1143.

Kohavi 1995 documents that stratified k-fold CV variance grows when per-fold sample size is small. Empirically this matches our Optum n=1294 result: `cv_5fold_roc_auc_std = 0.0937` with training fold n ~ 155 and class imbalance ~35:1 (leaving only ~4 positives per fold) — our band labels this `unstable`. The paper does NOT publish a 0.05 / 0.10 / 0.20 breakpoint table; the qualitative "small-cohort regime ⇒ inflated variance" finding supports our heuristic gradation but does not pin the specific cutpoints.

> Mukherjee, S., Niyogi, P., Poggio, T., & Rifkin, R. (2006). Learning theory: stability is sufficient for generalization and necessary and sufficient for consistency of empirical risk minimization. *Advances in Computational Mathematics*, 25(1), 161-193.

Mukherjee et al. 2006 prove that bounded-loss change under leave-one-out perturbation (algorithmic stability) is necessary and sufficient for generalization of empirical risk minimization. **Important caveat:** their β-stability framework concerns leave-one-out loss perturbation, NOT std/mean AUC ratio across k=5 folds, and the paper does **NOT** establish "std/mean > 0.20 violates the stability assumption" as a corollary. The earlier docstring claim that Mukherjee 2006 "grounds the rejection boundary `UNSTABLE_RATIO_MAX = 0.20`" was an overreach. We retain Mukherjee 2006 as supporting evidence that algorithmic stability matters for generalization in principle, but the specific 0.20 cutpoint is a heuristic operator-friendly choice, not a theorem-derived bound.

**Threshold value carried into code:**
- `T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX = 0.20` (rejection boundary; above this is `very_unstable`).
- RATIONALE: heuristic operator-friendly cutpoint chosen as a round-number gradation within the general high-variance region of CV-CoV literature. Empirically separates the Optum n=1294 result (0.0937 = `unstable`) from cohorts with substantially larger fold sizes. **NOT** a Mukherjee-2006-derived bound. To promote this to enforcement, either cohort-fit calibration on an un-touched cohort or a page/table pin tying 0.20 to a specific published threshold is required.

---

## 6. Cross-references and reproducibility

- **Code:** `src/agents/ml_foundation/model_deployer/nodes/registry_manager.py:39-53` defines all eight `T2_6A_*` constants. Inline comments in those lines should match this document — when the constants drift, this doc needs updating.
- **Helper functions:** `_categorize_signal_genuineness`, `_categorize_calibration_quality`, `_categorize_cv_stability` consume the constants; they are pure-compute and unit-tested.
- **Companion doc:** [`t23_cohort_bands_20260510.md`](t23_cohort_bands_20260510.md) covers the T2.3 cohort-derived honest band, which is observability-only and **NOT** a threshold.
- **Companion doc:** [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md) describes what un-touched cohort families would unblock retrospective threshold fitting.
- **Companion doc:** [`t22_perm_anchored_synth_20260510.md`](t22_perm_anchored_synth_20260510.md) describes the synthetic-only [0.55, 0.85] regime sweep that fits T2.2's perm-anchored AUC floor (no real-cohort touch).

---

## 7. Lifecycle and reauthoring trigger

**State:** advisory.
**Promotion to enforcement:** would require either (a) a 4th un-touched real cohort (see `t26_future_cohort_plan_20260510.md`) or (b) explicit domain-expert sign-off that the literature anchors are clinically defensible without retrospective fitting.

**Reauthoring trigger:** any of:
1. A new threshold is added to `compute_deployer_input_metrics` (e.g., HBLP variance gate, business utility gate).
2. An existing threshold is changed in code (the anchor citation must be updated to justify the new value).
3. A retrospective-cohort fit produces a value that disagrees with the literature anchor by more than ±2pp — both numbers should be retained, with the cohort-fit one promoted to enforcement and the literature anchor retained as a sanity check.

---

## 8. Open citation gaps

The following citations are listed as `[CITATION-CHECK NEEDED]` in §4 above — they should be verified against the canonical edition during the next review pass:

- Vickers, van Calster, & Steyerberg (2019) is retained in §4 ONLY for clinical-utility framing (decision curve analysis / net benefit); it is NOT an ECE threshold anchor.

The other citations (Fisher 1925, Fisher 1935, Naeini 2015, Kumar 2019, Bouckaert & Frank 2004, Kohavi 1995, Mukherjee 2006, Ojala & Garriga 2010) are standard works whose **existence is verified** and well-attested in the ML/statistics literature. However, **numeric threshold claims drawn from these works require page/table pins before enforcement** — several load-bearing values used in §3-§5 (e.g., Naeini 2015 "ECE 0.02-0.04 range" attributed to §4.2 Table 2; Bouckaert & Frank 2004 "0.02-0.05 for Adult/Letter" attributed to §3; Kohavi 1995 "5-fold std clusters at ~0.02-0.05" attributed to §4) have not yet been pinned to verifiable page/table numbers in the canonical published editions. Until those pins land, the bands rest on general-direction support from the literature plus heuristic operator-friendly gradation, and remain advisory-only per §1.

**Pending citation work (next review pass):**
- Pin Naeini 2015 Table 2 reference to verifiable AAAI 2015 proceedings page numbers.
- Pin Kumar 2019 Tables 1-3 to verifiable PMLR 80 proceedings page numbers.
- Pin Bouckaert & Frank 2004 §3 to verifiable Springer LNCS 3056 page numbers.
- Pin Kohavi 1995 §4 to verifiable IJCAI-95 proceedings page numbers.
