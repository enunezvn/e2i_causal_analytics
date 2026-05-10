# Gate G5 — T2.4 Coefficient-Sensitivity Pre-Spec Memo (2026-05-10)

**Status:** Pre-Specification (committed BEFORE running the experiment).

**Plan reference:** `.claude/plans/disease_agnostic_quality_uplift_v4.md` §2 Gate G5.

**Why this memo exists:** v3 §6 T2.4 acceptance criterion required "coefficient
sensitivity tests pass on Optum and CSU." PR #125 shipped only the missingness
profile half of T2.4 (`compute_imputation_audit`) with zero callers, leaving the
coefficient-sensitivity half unaddressed. v4-draft initially listed thresholds as
`(TBD per cohort)`; that is post-hoc threshold choice and conflicts with v3 §8's
explicit prohibition of threshold-shopping. This memo locks the thresholds BEFORE
any cohort-level run is executed, so passing the test is not a tautology.

---

## 1. Pre-specified thresholds (LOCKED — do not edit without protocol below)

The coefficient-sensitivity test fits a baseline model on the cohort's training
split, then re-fits the same model with each numeric feature imputed under the
strategy recommended by `compute_imputation_audit` (T2.4 missingness audit).
The two coefficient vectors are compared per-feature. A "significant" feature
is one whose baseline coefficient absolute value exceeds 1 standard deviation
of the baseline coefficient distribution (`|effect_size_baseline| > 1σ`).

The following three thresholds are pre-specified and load-bearing:

| # | Scope | Threshold | Description |
|---|-------|-----------|-------------|
| **T1** | Per-feature (significant only) | `flips_per_feature ≤ 1` | Among features with `|effect_size_baseline| > 1σ`, at most 1 of the imputation re-fits may flip the coefficient sign. Effectively zero with a single comparison run; future multi-strategy sweeps would tolerate one outlier. |
| **T2** | Per-feature (significant only) | `std(effect_size) / \|mean(effect_size)\| ≤ 0.5` | Coefficient effect-size variance across baseline + imputed runs must not exceed 50% of the absolute mean. A feature whose magnitude is unstable under imputation is not a stable predictor regardless of sign. |
| **T3** | Per-cohort aggregate | `fraction_significant_flipped ≤ 0.10` | At most 10% of the cohort's "significant" features may flip sign at all (counting any flip, not bounded by T1). This is the cohort-level acceptance gate. |

### Threshold rationale

- **T1 (≤ 1)** mirrors the conventional "single outlier" tolerance in
  bootstrap-style sensitivity analyses (Sterne 2009 BMJ; Donders 2006 review on
  multiple imputation). The current G5 implementation runs ONE imputed fit per
  feature, so T1 reduces to "no flip" in practice; the `≤ 1` framing is
  forward-compatible with multi-strategy sweeps.
- **T2 (CV ≤ 0.5)** corresponds to the conventional "moderate stability"
  coefficient-of-variation cutoff used in clinical-prediction sensitivity work
  (Steyerberg, *Clinical Prediction Models*, ch. 5). Below 0.5 is "stable
  enough"; above 0.5 indicates the imputation choice is materially affecting
  the coefficient's magnitude.
- **T3 (≤ 10%)** is the standard "false-discovery" cap applied to coefficient
  sign-flip frequency in MICE multiple-imputation sensitivity literature
  (van Buuren, *Flexible Imputation of Missing Data*, 2nd ed. §5.3). At
  fewer than 10% flips, the conclusion that "imputation does not materially
  change the model's directional structure" is defensible.

### What "significant" means here

A baseline coefficient is "significant" iff its absolute value exceeds the
baseline coefficient distribution's standard deviation:

```python
sigma = float(np.std(np.abs(coef_baseline_vector)))
significant_features = [f for f in features if abs(coef_baseline[f]) > sigma]
```

The 1σ definition is intentionally conservative (it surfaces top-20-30% of
features by magnitude) and avoids any p-value-based filter that would itself
require post-hoc decisions. The constant `1.0` is locked.

---

## 2. Data-hash + commit-graph protocol

This pre-spec memo IS the load-bearing artifact for G5's threshold-shopping
defense, mirroring the protocol committed for Gate G2 (`tier1b_b2_prespec_20260510.md`):

1. **Pre-spec parent constraint**: the experiment commit (the commit that
   introduces `tests/integration/test_t24_coefficient_sensitivity_20260510.py`)
   MUST be a CHILD of the commit that introduces THIS memo. CI verification is
   deferred to G2's commit-graph workflow which generalizes the check across
   all `docs/specs/*_prespec_*.md` files.
2. **Dataset content hashes** are pinned at the bottom of this memo for the
   cohort parquet/json files referenced by the integration test. CI will
   compute fresh hashes and compare against this memo at experiment time;
   mismatch fails the run loudly.
3. **Allowed updates** to this memo: only via a new `g5_*_prespec_<date>.md`
   memo at a fresh date. Editing thresholds in this memo in-place is
   forbidden and will be caught by the commit-graph audit (the experiment
   commit's parent must reference *this* memo by its committed SHA, not a
   later edit).

### Dataset content hashes (PINNED at the time of this memo)

The integration test references the following cohort artifacts. Their
sha256 hashes are PINNED below (M1 closure). The CI helper at
``scripts/verify_g5_prespec_hashes.py`` re-computes the live hashes
of these artifacts and compares them to the values committed here;
mismatch is a hard failure (the cohort drifted between memo-lock and
experiment-run, violating the threshold-shopping defense).

If a cohort artifact is ABSENT (e.g., not yet on disk in a fresh
checkout), the integration test treats the absence per the M2 protocol
(CI=true → fail; local → skip with clear pointer).

```yaml
# Pinned sha256 values.
# Computed via:
#   sha256sum data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet
#   sha256sum data/rwd/csu/e2i_ml_v3_patient_journeys.json
#   etc.
#
# IMPORTANT: when generating fresh cohorts via the converter, the new
# hashes invalidate this memo. Per Section 4 a NEW
# g5_*_prespec_<date>.md memo is required at a fresh date BEFORE the
# integration test runs against the regenerated cohort.

g5_dataset_hashes:
  # data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet
  optum_initiation_patient_journeys_parquet:
    path: "data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet"
    sha256: "7e334ca26e64a7e42d317876c9eee58189aa756afd46c70ab8464474e1cb68cd"
  # data/rwd/csu/e2i_ml_v3_patient_journeys.json
  csu_patient_journeys_json:
    path: "data/rwd/csu/e2i_ml_v3_patient_journeys.json"
    sha256: "13652dac7d6da887d3e7084d622bc52eb743e02e815ca6e50955f9020a40a952"
  # data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet
  optum_initiation_treatment_events_parquet:
    path: "data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet"
    sha256: "2449d7fb460c71e506e526e699f84d76fee46e8d5cc82060fef3033e3bc9cf67"
  # data/rwd/csu/e2i_ml_v3_treatment_events.json
  csu_treatment_events_json:
    path: "data/rwd/csu/e2i_ml_v3_treatment_events.json"
    sha256: "36eb50a75eac2dffe5927892ce48319ded7002651fd765621605faad2e46c5ee"
```

#### Hash pinning provenance

The four cohort artifacts referenced above are present on disk in
this worktree as of 2026-05-10 (G5 iter-3 fix commit). The sha256
hashes were computed via ``sha256sum`` on the exact file bytes and
pinned into this memo in the same commit that closes M1.

Subsequent CI runs invoke ``scripts/verify_g5_prespec_hashes.py``
(no flag) which re-computes live sha256 values for these artifacts
and fails on mismatch. The script's ``--update`` flag is reserved
for the operator-only path of pinning hashes when a NEW cohort lands
on disk via a fresh ``g5_*_prespec_<date>.md`` memo (not in-place
editing of this memo).

The data-hash protocol's load-bearing property is that the committed
hash AT THE PRE-SPEC SHA equals the hash AT THE EXPERIMENT SHA — i.e.,
the data did not change between locking the spec and running the test.
The verify script is the CI-side enforcement of that property.

---

## 3. Acceptance test outline

The test harness lives at:

- `src/agents/ml_foundation/data_preparer/nodes/coefficient_sensitivity.py`
  (helper: `compute_coefficient_sensitivity`)
- `tests/integration/test_t24_coefficient_sensitivity_20260510.py`
  (Optum + CSU integration; skip-if-data-missing)
- `tests/unit/test_agents/test_ml_foundation/test_data_preparer/test_coefficient_sensitivity.py`
  (synthetic-fixture unit tests for the helper)

The integration test:

1. Loads the cohort's `patient_journeys` artifact via the same path the
   tier0 runner uses (`_load_from_files` semantics).
2. Materializes a feature matrix (numeric columns only — categoricals are
   excluded from the coefficient-sensitivity audit because their imputed
   coefficients are not directly comparable across strategies).
3. Calls `compute_imputation_audit` to derive per-feature `recommendations`.
4. Calls `compute_coefficient_sensitivity(X, y, recommendations)` to fit
   baseline + imputed models and produce the per-feature comparison.
5. Asserts T1 + T2 + T3 against the helper's output.

---

## 4. Threshold-deviation procedure

If a cohort fails any of T1/T2/T3, the response is **NOT** to relax the
threshold. Per v3 §8 ("threshold shopping"), retroactive threshold adjustment
to make a failing cohort pass requires:

1. Written justification identifying the data property that motivates the
   change.
2. A new validation cohort whose threshold-fitting role was NOT exercised by
   the failing cohort.
3. A NEW `g5_*_prespec_<date>.md` memo at a fresh date locking the new
   threshold BEFORE the new cohort's run.

The default response to a failure is: "the imputation strategy recommendation
is too aggressive for this cohort", and the `compute_imputation_audit`
recommendation thresholds are revisited (not the sensitivity thresholds).

---

## 5. Pre-spec checksum

This memo's load-bearing content is sections 1 + 2 + 4. Sections 3 and 5 are
descriptive only.

The thresholds T1/T2/T3 are the load-bearing pre-specified values. They are
encoded as constants in
`src/agents/ml_foundation/data_preparer/nodes/coefficient_sensitivity.py`:

```python
G5_FLIPS_PER_FEATURE_MAX: int = 1            # T1
G5_EFFECT_SIZE_CV_MAX: float = 0.5            # T2
G5_FRACTION_SIGNIFICANT_FLIPPED_MAX: float = 0.10  # T3
G5_SIGNIFICANCE_SIGMA_MULTIPLE: float = 1.0   # 1σ definition of "significant"
```

The integration test imports these constants directly. Any edit to the
constants requires a new memo per Section 4.
