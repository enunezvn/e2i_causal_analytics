# G2 Harness Layer 3 Production-Parity Re-run — Optum n=1294 — 2026-05-11

**v5 Workstream A1 acceptance evidence** per `disease_agnostic_quality_uplift_v5.md` §2 A1.

## Setup

- Harness commit (parent of A1 fix): `7682f235` (v4 closure HEAD on `main`).
- Probe swap: `_compute_marginal_z_scores` now delegates to
  `src.data.adversarial_leakage.compute_adversarial_score` (production
  Layer 3 permutation-null probe), replacing the pre-A1 Welch's-t surrogate.
- Cohort: `optum_initiation_default` (n=1294, target=`treatment_initiated`).
- Seeds: G2_SEEDS = (42, 43, 44, 45, 46).
- Permutations: 200 (matches production `DEFAULT_PERMUTATIONS`).
- Adversarial seed: 7 (matches production default).
- Wall time: 3m53s on 5-seed × 81-feature × 200-permutation run.

## Result

| Metric | Pre-A1 (Welch) | A1 (production parity) | Threshold | Pass |
|---|---|---|---|---|
| baseline_auc_mean | 0.6621 | **0.6621** | — | — |
| hblp_auc_mean | 0.6621 | **0.6621** | — | — |
| ΔAUC (T1) | 0.0000 | **0.0000** | ≥ 0.03 | FAIL |
| baseline_ece_mean | 0.3965 | **0.3965** | — | — |
| hblp_ece_mean | 0.3965 | **0.3965** | — | — |
| ECE ratio (T2) | 1.0000 | **1.0000** | ≤ 0.5 | FAIL |
| CV stability ratio (T3) | 1.0000 | **1.0000** | ≤ 0.7 | FAIL |
| `g2_passes_pre_spec` | False | **False** | — | — |

### Per-seed retention (all 5 seeds identical)

- `n_train_pos = 22` per seed.
- Both arms retain 81/82 features.
- Both arms drop `initiated_biologic_180d` (the high-z leak).
- `features_diverged = []` on every seed.

## Interpretation

**Failure mode (b) per v5 §2 A1**: the production permutation-null probe flags
**identical** feature set as the pre-A1 Welch surrogate. The harness's pre-A1
AUC of 0.6621 was already correct on this cohort because the only feature
above either threshold is `initiated_biologic_180d`, which has effective AUC
≈ 1.0 vs target — both Welch z (since the feature has positive-class variance
once tied with target) AND production z flag it. Welch's blind spot on binary
perfect-correlation features did NOT change the empirical result here.

**HBLP relaxation band is empty** on Optum n=22 train positives by structural
property (per v4 G2 closure):

- Variance inflation = √(50/22) ≈ 1.51.
- HBLP-effective threshold (`declared_safe=False`): 5σ × 1.51 = 7.5σ.
- HBLP-effective threshold (`declared_safe=True`): 5σ × 1.51 × 1.5 = 11.3σ.
- The Layer-3 leak has z far above both → both arms drop.
- All other features have z below 5σ → both arms retain.
- No feature lands in the [5σ, 7.5σ] or [5σ, 11.3σ] band on this cohort.

## Closure transition

**v4 G2 status under A1**:

- `pre_spec_design = FAILED` (unchanged; SE calibration defect in pre-spec memo).
- `empirical_result = PENDING_HARNESS_PARITY` (v4 closure status) →
  **`empirical_result = CONFIRMED_NULL_AT_HARNESS_PARITY`** (after this run).
- `quality_uplift_claim = FALSIFIED_AT_CURRENT_N` (unchanged; reinforced).

The G2 closure as `pre_spec_design=FAILED` is now empirically watertight: even
with a production-parity Layer 3 probe, the HBLP arm cannot diverge from
baseline on Optum n=1294. Future positive-evidence quality claims at this
cohort size are blocked by HBLP's structural inertness, not by harness defects.

## Next steps

- v5 Workstream B (non-HBLP quality levers) remains the actionable quality
  path: calibration (B1), survival modeling (B2), feature engineering on the
  clean pre-anchor surface (B3).
- v5 Workstream C (CSU as production-grade deployment target via N1
  regulatory-eligibility) remains the load-bearing v5 deliverable.
- v4 backlog #33 (Optum cohort expansion to n_pos ≥ 150) and #32 (Candidate B
  external-reviewer onboarding) still gate any future Optum-side positive
  evidence — A1 confirms neither is unblocked by harness work alone.

## Artifacts

- Manifest JSON: not committed (large, deterministic per probe params);
  reproducible via:

      .venv/bin/python -m scripts.run_tier1b_b2_experiment \
          --cohort-label optum_initiation_default \
          --manifest-out optum_default_a1.json \
          --no-fail-on-empty-declared-safe

- Cross-reference: `docs/results/optum_initiation_revalidation_20260510.md`
  (v4 empirical anchor; same AUC=0.662 ± 0.10 across 5 seeds at default window).
