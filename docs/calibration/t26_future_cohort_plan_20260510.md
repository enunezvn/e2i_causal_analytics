# T2.6 Future-Cohort Plan — Unblocking Retrospective Threshold Fitting

**Date:** 2026-05-10
**Plan:** v4 §6 G4 — calibration-protocol artifacts (codex-rescue HIGH-4 fix)
**Companion docs:**
- [`t26_literature_anchors_20260510.md`](t26_literature_anchors_20260510.md) — current advisory thresholds
- [`t23_cohort_bands_20260510.md`](t23_cohort_bands_20260510.md) — observability-only honest band
- [`t22_perm_anchored_synth_20260510.md`](t22_perm_anchored_synth_20260510.md) — synthetic-only T2.2 sweep

---

## 1. Why we cannot retrospectively fit T2.6 thresholds today

Plan v3 §6 specified **retrospective held-out cohorts** as the threshold-fitting step. The cohort inventory below documents why no current cohort qualifies:

| Cohort                                  | Touched-by-arc?                                                                        | Eligible for T2.6 fitting? |
| --------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------- |
| `synthetic_rwd_realistic` regimes       | UNTOUCHED for the 0.55-0.85 sweep (existing pin tests anchor band only)                 | Eligible for T2.2 only — see `t22_perm_anchored_synth_20260510.md` |
| CSU n=9607 (PR #106)                    | TOUCHED — `val_AUC=0.6592` directly drove the v3 honest-band literal `[0.62, 0.68]`     | NO                         |
| Optum n=1294 default windows (PR #116)  | TOUCHED — empirical anchor for the entire arc; F1-fallback lift evidence; sample-size finding | NO                         |
| Optum n=1697 relaxed PRE/POST           | TOUCHED — sensitivity test that was already used to argue "sample size IS binding"; reverted via `git checkout` but verdict was conditioned on its outputs | NO                         |

**Data-snooping bar:** any cohort whose metric **values** were already used to argue for a threshold's correctness is conditioned on that data. Re-using it to fit the threshold is mathematically the same as fitting on the training set and reporting test accuracy on the same set — the bias is real even if the workflow is informal.

---

## 2. What un-touched cohort families would unblock fitting?

The minimum viable retrospective-fit dataset is **at least one independent cohort family** that shares the framework's gate semantics but has not informed any prior threshold decisions. Two paths are realistic:

### Path A — Acquire a 4th disease cohort (NOT atopic dermatitis, NOT CSU, NOT psoriasis)

**Examples currently in scope for the broader Novartis Causal Analytics pipeline:**
- Severe asthma (Optum claims; codex CSU-benchmark research already cited a baseline AUC of 0.66 for severe asthma — but this is a literature anchor, not a procurement plan).
- Inflammatory bowel disease (Optum or Truven); claims-only initiation prediction is a known parallel modeling target.
- Multiple sclerosis (Optum; high-cost biologic initiation similar to CSU).
- Rheumatoid arthritis (Optum or CSU competitor).

**Procurement criteria (must satisfy ALL):**
1. **Independent enrollment year**: cohort enrollment overlap with current Optum/CSU sources is ≤ 10% (avoids correlated patients at the row level).
2. **Independent feature lineage**: at least 80% of the manifest contracts must be reusable from existing CSU/Optum manifests (`src/data/feature_manifests/csu.json`, `src/data/feature_manifests/optum.json`); NEW disease-specific contracts are added without modifying existing ones.
3. **Independent claims source OR independent claims-vintage**: Optum 2019-2021 ≠ Optum 2023-2025; if same source, vintage gap ≥ 24 months.
4. **Sample size floor**: n_train_positives ≥ 100 (CSU has 98; Optum n=1294 has 22 — the latter is what made T2.6 fitting nonviable for Optum).
5. **Class imbalance ≤ 50:1**: avoids the extreme-imbalance regime where AUC and ECE are dominated by class-prior noise.
6. **Pre-publication freeze on threshold values**: the 4th-cohort metrics MUST NOT be discussed in any plan document, PR, or commit message until threshold fitting is complete. Mentioning the metrics taints the cohort.

**Procurement timeline:** 6-12 weeks if Optum data-rights cover the new disease (data already procured; new SQL slice). 12-26 weeks if a new vendor agreement is needed (Truven, IBM Watson Health, Veeva).

**Decision deferred:** the disease selection should be driven by Novartis pipeline value, NOT framework needs. Engineering team flags the framework readiness; product team picks the disease.

---

### Path B — Pull a 2nd Optum sub-population with non-overlapping inclusion criteria

**Concrete proposal:**
- Optum cohort definition currently uses chronic-spontaneous-urticaria initiation codes (the CSU surface) PLUS feasibility-required enrollment windows.
- An independent sub-population: pull a different chronic-disease initiation cohort from the same Optum source but with **NO patient overlap** with the current n=1294 — e.g., 2nd-line oncology initiation or chronic-pain biologic initiation, drawn from the same vintage but disjoint diagnosis codes and disjoint patient_ids.

**Procurement criteria (must satisfy ALL):**
1. `patient_id` set has zero intersection with the current n=1294 Optum cohort.
2. Diagnosis code set has zero intersection with CSU's diagnosis-code set (no shared ICD-10 codes that drive eligibility).
3. Feasibility windows (PRE/POST around index date) are derived from the new cohort independently — no copy-paste of CSU/Optum-current parameters.
4. Sample size floor: n_train_positives ≥ 100 (same as Path A).
5. Pre-publication freeze: same as Path A criterion #6.

**Procurement timeline:** 2-6 weeks if data-rights cover the codeset (likely Yes for Optum standard SDK access). Conversion can reuse `scripts/convert_optum_rwd.py` skeleton.

**Decision deferred:** the codeset selection should match a Novartis pipeline target, NOT a framework-internal pick.

---

### Path C (NOT preferred) — Use a synthetic regime as if it were retrospective

**Why this fails:** synthetic regimes do not have the noise structure of real claims data. The leakage signature, class-imbalance distribution, missingness pattern, and feature-correlation structure are all generative-process artifacts. Threshold values fit on synthetic regimes will systematically misfire on real data.

**Acceptable use of synthetics:** plumbing verification only (Plan v3 §6 step 1) — already satisfied by the synthetic_rwd_realistic regime test pins.

---

## 3. Promotion conditions: from advisory to enforced thresholds

The deployer's `T2_6A_*` constants are currently **advisory-anchor literals** (per `t26_literature_anchors_20260510.md`). Promotion to enforcement requires:

### Necessary conditions (BOTH must hold)

1. **At least ONE un-touched cohort** has been added to the framework via Path A or Path B above. The cohort has a complete pipeline run with `validation_metrics.permutation_pvalue`, `validation_metrics.cv_5fold_roc_auc_{mean,std}`, and `metrics_result.calibration_error` populated.
2. **Domain-expert sign-off** (medical leader OR causal analytics governance) on the deployer-input thresholds via PR review. The sign-off must reference the literature anchors AND the cohort-fit results AND a comparison.

### Sufficient conditions (any ONE will trigger advisory→enforcement promotion)

- **Cohort-fit values agree with literature anchors within ±2pp** for ECE and ±0.02 for std/mean ratio: promote literature anchors verbatim, document the cohort-fit corroboration in the deployer constants comment block.
- **Cohort-fit values disagree with literature anchors by > ±2pp**: replace the literature anchor with the cohort-fit value, retain literature anchor in the citation block as a "comparison reference."
- **Cohort-fit values cannot be computed** (e.g., new cohort halts before reaching evaluator) → KEEP literature anchors, extend the advisory window by another quarter, document the halt cause in the deployer constants comment block.

### Anti-promotion conditions (any ONE blocks promotion)

- The new cohort has been used in any plan/PR/commit discussion before threshold fitting was completed — TAINTED, treat as if no cohort was added.
- Sample size below 100 train positives — STAY ADVISORY; the variance dominates the mean and threshold fits will overfit.
- Class imbalance > 50:1 — STAY ADVISORY; ECE estimates are dominated by class-prior noise.

---

## 4. Per-threshold promotion order

Three deployer-input metrics have separate fit pathways:

### Signal genuineness (perm-test p-value)

- **Lowest dependency on cohort heterogeneity** — Fisher α=0.05 is well-established across all biomedical statistics.
- **Promotion plan:** can advance to enforcement first, before ECE or stability. Advisory data shows perm-test rejects on data-limited cohorts (Optum n=1294 perm-p=0.67 → MARGINAL band). The advisory band already matches the cohort-fit pattern; promotion is cosmetic.
- **Suggested literal-fit value:** retain `T2_6A_SIGNAL_MARGINAL_PVALUE_MAX = 0.05`. Stricter `T2_6A_SIGNAL_LIKELY_GENUINE_PVALUE_MAX = 0.01` may rise to 0.02 if ML stability wants more bin granularity.

### Calibration quality (ECE)

- **Medium dependency on cohort heterogeneity** — claims-only ML on small N has different ECE structure than imaging or genomics ML. The 4th cohort fit may surface a domain-specific shift.
- **Promotion plan:** advance only AFTER 4th-cohort fit. Stay advisory until then.
- **Suggested cohort-fit guard:** if 4th cohort produces ECE < 0.05 with `genuine` permutation gate AND `stable` CV stability, promote `T2_6A_CALIBRATION_GOOD_ECE_MAX` to 0.05 (tightening). Otherwise retain current 0.10.

### CV stability (std/mean)

- **High dependency on cohort heterogeneity** — sample size and class imbalance both shift the std/mean baseline. Optum n=1294 already exhibits std/mean=0.138 (`unstable` band) vs. a likely CSU std/mean estimate of 0.05-0.08 (untouched calculation; would require recomputing).
- **Promotion plan:** advance only AFTER 4th cohort + at least one shift in sample-size regime AND class-imbalance regime.
- **Suggested cohort-fit guard:** keep at advisory. If we ever ship a sample-size-corrected std/mean threshold (e.g., `T2_6A_CV_STABILITY_UNSTABLE_RATIO_MAX = 0.20 / sqrt(n_train_pos / 100)`), the static literal goes away entirely.

---

## 5. Drift-monitoring use case (the only allowed use of cohort-derived bands today)

While retrospective threshold fitting is blocked, the cohort-derived honest band (T2.3) and the deployer-input metrics (T2.6a) are **emitted as observability signals**. Operators can use them for drift detection:

- Compare the per-run `validation_metrics.honest_band_lo/hi` to the historical median over 10+ prior runs on the same cohort. Drift > 5pp on either bound is a flag for cohort or pipeline change.
- Compare per-run `cv_5fold_roc_auc_std` to its historical median. Drift > 50% (e.g., 0.04 → 0.08) is a flag for label-quality or feature-leakage shift.
- Compare per-run `permutation_pvalue` to its historical median. A transition from `genuine` (≤ 0.001) to `marginal` (~0.05) on the same cohort is a flag for distribution drift OR the new feature manifest accidentally including post-anchor leakage.

These are observability uses **only** — do not use the cohort bands as deployment thresholds until the promotion conditions in §3 are met.

---

## 6. Lifecycle markers in code

**Constant:** `T23_BAND_LIFECYCLE_STATE = "advisory"` near `_emit_cohort_derived_honest_band` (introduced this commit). This is the in-code analog of this document's lifecycle state. When the cohort-fit unblock is complete and the band graduates to enforcement, this constant should change to `"enforced"` AND this document should be archived to `docs/calibration/archive/` with a "graduated YYYY-MM-DD" suffix.

**Decision-record trigger:** if any contributor proposes changing a `T2_6A_*` constant in `registry_manager.py` while this advisory state is active, the PR description must reference this doc and explain whether the change is (a) a literature-anchor refinement (UPDATE this doc), (b) a synthetic-fit improvement (UPDATE `t22_perm_anchored_synth_20260510.md`), or (c) a retrospective cohort fit (REQUIRES the §3 promotion-condition checklist to be filed).
