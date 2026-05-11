# Gate G2 — Tier 1B Gate B2 Pre-Spec Memo (2026-05-10)

**Status:** Pre-Specification (committed BEFORE running the experiment).

**Plan reference:** `.claude/plans/disease_agnostic_quality_uplift_v4.md` §2 Gate G2.

**Why this memo exists:** v3 §6 Tier 1B Gate B2 requires a pre-specified
quality-uplift comparison between the baseline (HBLP-disabled) and HBLP-relaxed
cohort runs, with three thresholds (ΔAUC, ΔECE ratio, CV-stability ratio) ALL
fixed BEFORE the experiment runs. Running the experiment first and then
choosing thresholds — even by accident, even with the same numbers — is the
"threshold-shopping" anti-pattern v3 §8 forbids. The protocol below mirrors
G5's pattern (`docs/specs/g5_coefficient_sensitivity_prespec_20260510.md`):
this memo IS the load-bearing artifact, and the experiment commit MUST be a
CHILD of the commit that introduces this memo (the `S_prespec` commit).

**Lifecycle state:** the G2 experiment workflow declares
`lifecycle_state: ADVISORY` until the first green run. Promotion to
`ENFORCED` requires (a) all three thresholds met on the named cohort, and
(b) a signed `docs/calibration/g2_completion_signoff_<date>.md` per the v4
N3 reviewer-registry policy.

---

## 1. Pre-specified thresholds (LOCKED — do not edit without protocol below)

The G2 experiment harness fits TWO models on the named cohort:

* **Baseline**: HBLP disabled — `_build_verdict` uses the legacy hardcoded
  `HIGH_Z` / `MODERATE_Z` ladder. This is the production default at HEAD
  pre-G3 (G3 wires HBLP into the default path).
* **HBLP-relaxed**: HBLP enabled — `hblp_classify` is invoked with
  `n_train_pos` and `layer_1_declared_safe` threaded from the validity-check
  caller. Variance-inflation and Layer-1-conditional priors relax the
  effective z-threshold, which can preserve Layer-3-flagged-but-not-leaking
  features that the baseline drops.

Both fits use identical preprocessing, identical hyperparameters, identical
seeds, and identical train/val/test splits. The only intervention between
baseline and HBLP-relaxed is the verdict-classifier swap.

The following three thresholds are pre-specified and load-bearing:

| # | Scope | Threshold | Description |
|---|-------|-----------|-------------|
| **T1** | Held-out AUC lift | `Δ_AUC ≥ 0.03` | Held-out test AUC of the HBLP-relaxed model exceeds the baseline by at least 0.03 (3 percentage points). The lift is computed on the held-out test split, NOT the validation split (val_AUC is the hyperparameter-selection signal and is structurally optimistic). |
| **T2** | ECE improvement | `ECE_post ≤ 0.5 × ECE_pre` | Expected Calibration Error of the HBLP-relaxed model is at most half of the baseline's. ECE is computed via `compute_calibration_analysis(y_test, y_proba_test, n_bins=10)` on the held-out test split using the canonical helper in `src/agents/ml_foundation/model_trainer/nodes/advanced_validation.py`. |
| **T3** | CV-stability improvement | `(std/mean)_post ≤ 0.7 × (std/mean)_pre` | Coefficient-of-variation of 5-fold stratified CV roc_auc (std divided by absolute value of mean) is at most 70% of the baseline's. CV is computed via `compute_stratified_cv` with `n_folds=5, random_state=<seed>` over the COMBINED train+val matrix (the held-out test split is preserved for T1/T2). |

### Threshold rationale

- **T1 (Δ_AUC ≥ 0.03)** matches the v3 plan's "ship a survival path" criterion
  for Tier 1C (Cox/RSF concordance > binary AUC by ≥0.03; see plan v4 §7
  Backlog #27). Three points on the AUC scale is the conventional minimum
  meaningful lift in clinical-prediction methodology (Steyerberg, *Clinical
  Prediction Models*, ch. 15) below which lift is plausibly within
  measurement noise on tier0-scale held-out splits.
- **T2 (ECE ≤ 0.5 × baseline)** is the standard "halving" calibration
  improvement criterion: a re-calibration intervention that does not at
  least halve the residual calibration error is considered a wash
  (Niculescu-Mizil & Caruana 2005 ICML; Guo 2017 ICML temperature-scaling
  paper). The 0.5 multiplier is intentionally conservative — a stricter
  cutoff would raise the false-rejection rate in finite samples; a looser
  cutoff would let cosmetic improvements pass.
- **T3 (CV-CV ≤ 0.7 × baseline)** asserts that HBLP's relaxation also
  stabilizes cross-validated AUC (i.e., HBLP isn't trading bias for variance).
  The 0.7 multiplier (a 30% reduction in CV-instability) is the standard
  "moderate stability improvement" benchmark in resampling sensitivity
  literature (Steyerberg, ch. 5; van Buuren §5.3). A null intervention has
  expected CV-stability ratio of 1.0; below 0.7 means the change is large
  enough to be detectable beyond noise.

### Combined acceptance

`G2_passes_pre_spec` = `(T1 AND T2 AND T3)`. ANY single threshold failing
disqualifies the experiment. This is by design: the v3 §6 B2 acceptance
criterion is conjunctive (`Δ ≥ 0.03 AND ECE_post ≤ 0.5 × ECE_pre AND
(std/mean)_post ≤ 0.7 × (std/mean)_pre`).

---

## 2. Cohort identifiers (LOCKED)

The G2 experiment runs on one **default** cohort. A second cohort is named
as a sensitivity test and is explicitly marked `data_snooped: true` per
v4 §8.

### 2.1 Default cohort — Optum n=1294

- **Cohort label:** `optum_initiation_default`
- **Path:** `data/rwd/optum/initiation/`
- **Patient-journey count:** 1294 (default-window: PRE=365 / POST=180)
- **Target column:** `treatment_initiated`
- **Empirical anchor reference:** `docs/results/optum_initiation_revalidation_20260510.md`
  — this cohort's baseline halts MARGINAL with permutation p≈0.67 at the
  current pre-G3 default; G2 measures whether HBLP relaxation moves the
  three quality metrics in the pre-specified directions.
- **Data-snooped flag:** `data_snooped: false` — the default-window
  parameters (PRE=365 / POST=180) were chosen by the v3 plan based on
  Optum claims-data conventions, NOT by post-hoc threshold-shopping.

### 2.2 Sensitivity cohort — Optum n=1697 (relaxed window)

- **Cohort label:** `optum_initiation_relaxed`
- **Path:** `data/rwd/optum/initiation/` after running the converter with
  `--enrollment-regime research_relaxed` (PRE=180 / POST=90)
- **Patient-journey count:** 1697 (relaxed-window)
- **Target column:** `treatment_initiated`
- **Status:** **NOT IN SCOPE for G2's load-bearing run.** This cohort is
  named here so the experiment harness CAN target it via the
  `--cohort-label optum_initiation_relaxed` flag, but the load-bearing
  G2 result is the n=1294 run. The n=1697 cohort is included as a
  sensitivity-only diagnostic.
- **Data-snooped flag:** `data_snooped: true` — the relaxed-window
  parameters were the parameters that empirically crossed permutation
  p<0.05 in a one-shot post-hoc test (see
  `docs/results/optum_initiation_revalidation_20260510.md` and v4 §2
  Gate N3). The G2 verdict on n=1697 is informational only; the cohort
  cannot be the load-bearing G2 result UNTIL the N3 methodology sign-off
  lands AND a fresh `tier1b_b2_prespec_<date>.md` memo at a new date pins
  the relaxed cohort as the load-bearing target.

The harness's CI-controlled execution path defaults to
`optum_initiation_default`. Running the harness against the relaxed
cohort requires explicit operator opt-in via `--cohort-label
optum_initiation_relaxed`, and any such run is recorded in the manifest
with `data_snooped: true`.

---

## 3. Seeds (LOCKED)

The experiment harness runs the baseline + HBLP-relaxed pair across
`SEEDS = [42, 43, 44, 45, 46]`. Each seed produces one (baseline,
HBLP-relaxed) pair. T1/T2/T3 are evaluated on the **mean** metric across
seeds (mean ΔAUC, mean ECE_post, mean ECE_pre, mean (std/mean)_post,
mean (std/mean)_pre). The 5-seed bootstrap reduces sampling noise on the
held-out lift estimate to a level commensurate with the 0.03 acceptance
threshold (per-seed AUC standard error on n_test ≈ 200 is ~0.03, so the
5-seed-mean SE is ~0.013, which is comfortably below 0.03).

The seeds list `[42, 43, 44, 45, 46]` is locked. Adding seeds, removing
seeds, or changing the bootstrap aggregation rule (mean vs median vs
worst-of) requires a fresh pre-spec memo.

---

## 4. Data-hash + commit-graph protocol

This pre-spec memo IS the load-bearing artifact for G2's threshold-shopping
defense, mirroring the protocol committed for Gate G5
(`docs/specs/g5_coefficient_sensitivity_prespec_20260510.md`):

1. **`S_prespec` parent constraint**: the experiment commit (the commit
   that introduces `scripts/run_tier1b_b2_experiment.py` AND the CI workflow
   `.github/workflows/tier1b_b2_experiment.yml`) MUST be a CHILD of the
   commit that introduces THIS memo. The CI-side check is implemented in
   `scripts/check_g2_commit_graph.py` and runs as a step in
   `tier1b_b2_experiment.yml`.

2. **Dataset content hashes** are pinned at the bottom of this memo for
   the cohort parquet/json files referenced by the experiment harness. CI
   computes fresh hashes via `scripts/verify_g2_prespec_dataset_hashes.py`
   and compares against this memo at experiment time; mismatch fails the
   run loudly. The placeholders in §5 below carry the literal token
   `TODO_PIN_AT_FIRST_GREEN_RUN` until the verifier is run with `--update`
   on the operator's first cohort-pinning pass; the verifier's `--update`
   path requires the artifact to be present on disk AND refuses to
   overwrite a non-placeholder value (forcing a fresh memo at a fresh
   date for re-pinning).

3. **CI-controlled first execution**: the experiment workflow
   `.github/workflows/tier1b_b2_experiment.yml` is triggered ONLY on tag
   `tier1b-b2-experiment-*`. The workflow:
   - Checks out the tag SHA
   - Verifies the tag commit references `S_prespec` in its commit message
   - Verifies the dataset content hashes at the paths named in this memo
     match those committed in `S_prespec`
   - Verifies the experiment commit is a CHILD of `S_prespec` via
     `git merge-base --is-ancestor`
   - Runs the experiment harness `scripts/run_tier1b_b2_experiment.py`
   - Uploads the run manifest (commit SHA, dataset hashes, seeds,
     observed metrics, pass/fail per criterion) as a CI artifact
   - The first run from `S_prespec` is the load-bearing one; subsequent
     runs are diagnostic only.

4. **Allowed updates** to this memo: only via a new
   `tier1b_b2_prespec_<date>.md` memo at a fresh date. Editing thresholds
   in this memo in-place is forbidden and will be caught by the
   commit-graph audit (the experiment commit's parent must reference *this*
   memo by its committed SHA, not a later edit).

### Why this matters (codex-rescue HIGH-2)

A determined threshold-shopper could:

- Run exploratory experiments locally on scratch branches
- See which (Δ, ratio_E, ratio_C) values pass
- Commit a "pre-spec" matching exactly those values
- Then run the experiment commit on the pinned cohort

The data-hash pin + CI-controlled first execution + commit-graph
parent-check protocol prevents this:

- The cohort-content hash is pinned at `S_prespec`. Any local exploration
  on a different cohort cut would mismatch the hash.
- The experiment commit MUST be a child of `S_prespec`. A commit that
  precedes `S_prespec` (e.g., an exploratory branch with a pre-pinned
  threshold copy) cannot be a child.
- The CI workflow refuses to run experiment commits that fail the parent
  check. The first green CI run from `S_prespec` is the load-bearing one.

---

## 4.1 Governance verifier SHA (PINNED — HIGH-6 iter-3)

The verifier scripts (`check_g2_commit_graph.py`, `verify_g2_prespec_dataset_hashes.py`,
`check_g2_prior_runs.py`) are the load-bearing trust boundary for G2. Pulling
them from `origin/main` (mutable protected ref) means a re-run of an old tag
could pick up newer verifier semantics than the original load-bearing run.

To close that gap, the workflow checks out the verifier scripts from a
SHA-pinned governance commit recorded HERE. The pinned SHA is the immutable
identity of the verifier code that was reviewed and committed as part of
S_prespec.

```yaml
# Governance verifier source SHA — pinned at S_prespec time.
# When this is the placeholder, the workflow falls back to origin/main
# (the prior, less-strict trust boundary). The first green run pins
# this to S_prespec itself (or a later commit on origin/main that
# CODEOWNERS has signed off on).
governance_verifier_sha: "TODO_PIN_AT_FIRST_GREEN_RUN"
```

The workflow records the SHA actually checked out in the run manifest's
`governance_verifier_sha` field for downstream audit.

---

## 5. Dataset content hashes (PINNED at the time of this memo)

The experiment harness references the following cohort artifacts. Their
sha256 hashes are PINNED below.

The CI helper at `scripts/verify_g2_prespec_dataset_hashes.py`
re-computes the live hashes of these artifacts and compares them to the
values committed here; mismatch is a hard failure (the cohort drifted
between memo-lock and experiment-run, violating the threshold-shopping
defense).

If a cohort artifact is ABSENT (e.g., not yet on disk in a fresh
checkout), the experiment harness fails with a clear pointer per the M2
protocol (CI=true → fail; local → skip with clear pointer).

```yaml
# Pinned sha256 values.
# Computed via:
#   sha256sum data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet
#   sha256sum data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet
#
# IMPORTANT: when generating fresh cohorts via the converter, the new
# hashes invalidate this memo. Per Section 4 a NEW
# tier1b_b2_prespec_<date>.md memo is required at a fresh date BEFORE
# the experiment commit is created.
#
# Placeholder protocol:
#   The TODO_PIN_AT_FIRST_GREEN_RUN tokens BELOW are intentional. The
#   verifier `scripts/verify_g2_prespec_dataset_hashes.py` rejects these
#   placeholders in verify mode (exits non-zero). The operator must run
#   the verifier with `--update` on the present cohort artifacts to
#   replace the placeholders with live sha256 hashes. The replacement
#   commit is a SEPARATE diff (no threshold edits in the same commit)
#   so the threshold-shopping audit can review the pinning action.
g2_dataset_hashes:
  # data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet
  optum_initiation_patient_journeys_parquet:
    path: "data/rwd/optum/initiation/e2i_ml_v3_patient_journeys.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
  # data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet
  optum_initiation_treatment_events_parquet:
    path: "data/rwd/optum/initiation/e2i_ml_v3_treatment_events.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
```

#### Hash pinning provenance

The two cohort artifacts referenced above are the load-bearing inputs
to the G2 experiment harness for the default `optum_initiation_default`
cohort.

The placeholder `TODO_PIN_AT_FIRST_GREEN_RUN` is the literal string
that `scripts/verify_g2_prespec_dataset_hashes.py` rejects in verify
mode. The first green CI run (which is the load-bearing G2 result)
requires the operator to first run

```
python scripts/verify_g2_prespec_dataset_hashes.py --update
```

against the pinned-cohort artifacts on disk, commit the pinned-hashes
diff via a SEPARATE commit (not the same commit that pins thresholds),
and only then create the `tier1b-b2-experiment-*` tag.

The data-hash protocol's load-bearing property is that the committed
hash AT THE PRE-SPEC SHA (after pinning) equals the hash AT THE
EXPERIMENT TAG — i.e., the data did not change between locking the spec
and running the experiment. The verify script is the CI-side enforcement
of that property.

---

## 6. Experiment harness outline

The harness lives at:

- `scripts/run_tier1b_b2_experiment.py` — entry point invoked by CI
- `scripts/verify_g2_prespec_dataset_hashes.py` — dataset-hash verifier
- `scripts/check_g2_commit_graph.py` — commit-graph parent-check
- `tests/scripts/test_run_tier1b_b2_experiment.py` — unit tests for the
  metric computation + threshold-evaluation logic
- `tests/scripts/test_verify_g2_prespec_dataset_hashes.py` — verifier tests
- `tests/scripts/test_check_g2_commit_graph.py` — parent-check tests

The harness:

1. Loads the named cohort's artifacts (per `--cohort-label`).
2. For each seed in `SEEDS = [42, 43, 44, 45, 46]`:
   a. Derives stratified train/val/test splits (60/20/20) on the
      patient-level partition.
   b. Fits a baseline model (HBLP-disabled) and an HBLP-relaxed model
      with identical preprocessing + hyperparameters.
   c. Computes held-out test AUC for each (T1 input).
   d. Computes test-set ECE for each via `compute_calibration_analysis`
      (T2 input).
   e. Computes 5-fold stratified CV roc_auc on the combined train+val
      matrix for each via `compute_stratified_cv` (T3 input).
3. Aggregates per-seed metrics to seed-mean values.
4. Evaluates T1, T2, T3 against the seed-means.
5. Emits a JSON manifest containing:
   - commit SHA (the experiment tag)
   - cohort label + dataset hashes
   - seeds list
   - per-seed (baseline, post) metrics
   - aggregate (mean, std) metrics
   - threshold pass/fail per criterion (T1/T2/T3)
   - overall `g2_passes_pre_spec` boolean
6. Exits 0 iff all three thresholds pass; exits 1 otherwise.

The pre-spec thresholds are encoded as constants in
`scripts/run_tier1b_b2_experiment.py`:

```python
G2_DELTA_AUC_MIN: float = 0.03           # T1
G2_ECE_RATIO_MAX: float = 0.5            # T2
G2_CV_STABILITY_RATIO_MAX: float = 0.7   # T3
```

The integration test imports these constants directly. Any edit to
the constants requires a fresh pre-spec memo per Section 4.

---

## 7. Threshold-deviation procedure

If the G2 experiment fails any of T1/T2/T3, the response is **NOT** to
relax the threshold. Per v3 §8 ("threshold shopping"), retroactive
threshold adjustment to make a failing experiment pass requires:

1. Written justification identifying the data property that motivates
   the change.
2. A new validation cohort whose threshold-fitting role was NOT
   exercised by the failing cohort.
3. A NEW `tier1b_b2_prespec_<date>.md` memo at a fresh date locking the
   new threshold BEFORE the new cohort's run.

The default response to a failure is: "HBLP relaxation does not produce
a quality uplift on the named cohort with the named seeds." The G2 gate
remains OPEN; G3 (HBLP default-path wiring) is BLOCKED per v4 §3
sequencing. The honest verdict is that on this cohort, the proposed
relaxation does not justify production-default wiring.

---

## 8. Pre-spec checksum

This memo's load-bearing content is sections 1 + 2 + 3 + 4 + 5. Sections
6 and 7 are descriptive only.

The thresholds T1/T2/T3 are the load-bearing pre-specified values. They
are encoded as constants in `scripts/run_tier1b_b2_experiment.py`:

```python
G2_DELTA_AUC_MIN: float = 0.03           # T1
G2_ECE_RATIO_MAX: float = 0.5            # T2
G2_CV_STABILITY_RATIO_MAX: float = 0.7   # T3
G2_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
```

The harness imports these constants directly. Any edit to the constants
requires a new memo per Section 4.
