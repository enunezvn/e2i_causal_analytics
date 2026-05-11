# Gate G2 — Tier 1B Pre-Spec Memo Amendment (2026-05-11)

**Status:** Post-experiment correction memo. NOT a new pre-spec; G2 closed `pre_spec_design=FAILED` per
`disease_agnostic_quality_uplift_v4.md` §2 amendment (2026-05-11). This memo documents the SE
calibration error in `tier1b_b2_prespec_20260510.md` §3 + the v5 Workstream A1 empirical confirmation
that the harness was producing correct AUC under both the pre-A1 Welch surrogate and the
production-parity probe.

**Predecessor:** `docs/specs/tier1b_b2_prespec_20260510.md` (the locked pre-spec; FAILED on Optum n=1294).

**Plan reference:** `disease_agnostic_quality_uplift_v5.md` §2 A1 (acceptance: "Update the pre-spec memo's
§3 SE calibration to reflect the actual cohort prevalence (per-positive-class SE, NOT per-patient SE).").

---

## 1. Defect — pre-spec §3 SE math

The original §3 paragraph stated:

> "per-seed AUC standard error on n_test ≈ 200 is ~0.03, so the 5-seed-mean SE is ~0.013,
> which is comfortably below 0.03."

This calibration assumed AUC SE scales with **total patient count** at n_test ≈ 200. The Hanley-McNeil
AUC SE formula scales with **per-class count**, primarily the **positive-class count**.

For Optum `optum_initiation_default` (cohort spec at pre-spec §2.1) under the harness's 60/20/20
split, the **test-fold** positive count is what governs AUC SE (NOT the train-fold count):

- n_total = 1294 patients
- A1 re-run observed `n_train_pos = 22` per seed at n_train ≈ 776 (60% of 1294) → cohort
  prevalence (treatment_initiated=1) ≈ 22/776 ≈ **2.8%**, not 8.4%.
- n_test ≈ 259 patients (20% of 1294) → if stratify preserves prevalence,
  **n_pos_test ≈ 7-8 per seed**, NOT 22.

The Hanley-McNeil SE for AUC ≈ 0.66 on (n_pos=7, n_neg=252) is approximately:

```
Q1 = AUC / (2 - AUC)              ≈ 0.493
Q2 = 2 * AUC^2 / (1 + AUC)         ≈ 0.525
var(AUC) = (AUC*(1-AUC) + (n_pos-1)*(Q1-AUC^2) + (n_neg-1)*(Q2-AUC^2)) / (n_pos * n_neg)
       ≈ (0.2244 + 6*0.0574 + 251*0.0894) / (7 * 252)
       ≈ 23.01 / 1764
       ≈ 0.01304
SE(AUC) = √0.01304 ≈ 0.1142
```

5-seed-mean SE: `0.1142 / √5 ≈ 0.0511`.

**Corrected calibration**: per-seed AUC SE on Optum default cohort ≈ **0.114** (not 0.03);
5-seed-mean SE ≈ **0.051** (not 0.013). This is **3.9× the locked memo's claim** — matching the
team report's framing (3.7-3.9×). Combined with the expected ΔAUC ≈ 0.0 under HBLP's structural
inertness on Optum's clean Layer-1-filtered surface, the T1 threshold ΔAUC ≥ 0.03 had
power ≈ 0.21 (per v4 G2 closure root-cause analysis).

**Sanity check against the A1 re-run**: the per-seed test AUCs in 2026-05-11 were
0.5683, 0.5788, 0.6599, 0.6993, 0.8044 with mean 0.6621 and sample-std 0.090. The observed
between-seed std (0.090) is comparable to the Hanley-McNeil per-seed SE (0.114) given the small
sample (5 seeds), reinforcing that the SE estimate is in the right neighborhood — and confirming
the locked memo's 0.030 was off by ~3-4×.

**Earlier draft note** — an earlier version of this amendment derived `n_pos=22, n_neg=237`
treating the train-positive count as the test-positive count (corresponding to per-seed SE ≈ 0.066,
5-seed SE ≈ 0.030 → 2.3× rather than 3.9×). That derivation conflated train-positives with
test-positives. The correct calibration is the (n_pos=7, n_neg=252) form above, which matches the
team report's 3.7-3.9× framing.

## 2. v5 A1 empirical confirmation

Per `docs/results/g2_harness_a1_parity_run_optum_20260511.md`:

- Replacing the harness Welch surrogate with the production permutation-null probe
  (`compute_adversarial_score`) produced **identical AUC** (0.6621 mean) and **identical retained
  feature set** (`features_diverged=[]` across all 5 seeds).
- The HBLP relaxation band is empty on Optum n=22 train-positives: effective high threshold is
  7.5σ–11.3σ; only `initiated_biologic_180d` has z above either band; all other features have
  z < 5σ. Both arms drop the same feature.

**Implication**: G2's `pre_spec_design = FAILED` closure is empirically watertight. The HBLP arm
cannot diverge from baseline on this cohort regardless of probe choice. v4 G2 closure transitions
from `empirical_result = PENDING_HARNESS_PARITY` to `empirical_result = CONFIRMED_NULL_AT_HARNESS_PARITY`.

## 3. What changes — and what does NOT

**Does NOT change**:

- The locked pre-spec at `tier1b_b2_prespec_20260510.md` is preserved verbatim for audit purposes.
- The G2 ENFORCED-mode pre-spec threshold values (ΔAUC ≥ 0.03, etc.) remain the locked record.
- The G2 lifecycle state remains ADVISORY; no promotion to ENFORCED is sought (the gate is
  closed `pre_spec_design = FAILED`, not retried).

**Does change**:

- §3 SE calibration is documented here as defective (this memo).
- v4 plan §2/§3/§7/§8 amendments (2026-05-11) record the closure status.
- v5 plan §2 A1 records the acceptance criterion that this memo satisfies.
- Future positive-evidence quality claims on Optum require either:
  - (a) cohort expansion to n_pos ≥ 150 (v4 backlog #33), where SE drops to ~0.025;
  - (b) Candidate B external-reviewer signoff (v4 backlog #32) for relaxed window n=1697;
  - OR (c) a non-Optum cohort (CSU, future Optum sub-cohorts).

## 4. Procedural note

This is an amendment memo, not a fresh pre-spec. The locked pre-spec at `tier1b_b2_prespec_20260510.md`
is NOT being edited (per §7 of the locked memo, threshold-deviation requires a new pre-spec with the
correct protocol). The corrected SE math is documented HERE so future plans / reviewers can:

- Read the locked memo as the audit-trail record of what was actually pre-specified;
- Read THIS memo as the post-experiment defect record + corrected calibration;
- Avoid replicating the per-patient-SE → per-positive-class-SE confusion in future pre-spec design.

## 5. Cross-references

- Locked pre-spec: `docs/specs/tier1b_b2_prespec_20260510.md`
- v4 G2 closure status: `.claude/plans/disease_agnostic_quality_uplift_v4.md` §2 (amended 2026-05-11)
- v5 A1 acceptance criterion: `.claude/plans/disease_agnostic_quality_uplift_v5.md` §2 A1
- A1 empirical evidence: `docs/results/g2_harness_a1_parity_run_optum_20260511.md`
- v4 G2 team synthesis: `~/.claude/projects/-home-enunez-Projects-e2i-causal-analytics/memory/v4_g2_team_synthesis_20260511.md`
- v4 empirical anchor (Optum n=1294 0.66): `docs/results/optum_initiation_revalidation_20260510.md`
- Hanley-McNeil SE formula: standard reference (Hanley & McNeil 1982; Radiology 143(1):29-36).
