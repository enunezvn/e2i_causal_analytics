# T2.2 Permutation-Anchored AUC Floor — Synthetic-Only Calibration

> **DERIVED FROM SYNTHETIC ONLY, NOT FROM OPTUM/CSU.** This calibration is the only T2-series threshold for which the v3 plan's "synthetic for plumbing → retrospective held-out for fitting" protocol can run as designed, because the `synthetic_rwd_realistic` regimes were NOT used to argue for any threshold value during the prior arc. The synthetic AUC sweep is therefore eligible for threshold-fitting role; CSU and Optum are NOT (see [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md) §1).

**Date:** 2026-05-10
**Plan:** v4 §6 G4 — calibration-protocol artifacts (codex-rescue HIGH-4 fix)
**Code surface:**
- `src/agents/ml_foundation/model_trainer/nodes/evaluator.py:154` `_emit_permutation_anchored_auc_advisory`
- `src/agents/ml_foundation/model_trainer/nodes/evaluator.py:71` `T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT = 0.05`
- Synthetic regime: `src/data/synthetic_rwd/synthetic_rwd_realistic.py` (signal scaling parameter sweepable)

---

## 1. What T2.2 measures and why a sweep is needed

The T2.2 advisory criterion gates on the margin
```
auc_above_permutation_null = test_auc - permutation_null_p99
```
A run violates the advisory when this margin is below the buffer (default `T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT = 0.05`).

**The buffer literal needs calibration evidence**: the 0.05 default is a domain-typical "above-noise" margin (5pp lift over the upper tail of label-shuffle noise). To know that 0.05 is the correct floor — not 0.03, not 0.08 — we need to:

1. Run the framework on a regime where the **target/empirical AUC is controlled and measured** (i.e., the synthetic generator's signal scale parameterizes the realized AUC, and we measure that realized AUC end-to-end). Note: we do NOT analytically compute the Bayes-optimal AUC of the regime, so the realized AUC is an empirical observation under controlled signal scale, not a closed-form ground truth.
2. Compute the empirical perm-null distribution at each controlled-AUC point.
3. Find the largest buffer value that still passes the advisory at every point in the legitimate-AUC range.

The synthetic regimes are suitable for this: their signal magnitude is parameterized, the noise structure is reproducible, the realized AUC is measured per-cell, and the AUC range is well-bounded. **No real-cohort touch is required.**

---

## 2. Sweep methodology (what will be run)

### 2.1 Sweep grid

A 5-seed × 7-AUC-point × N-replicates synthetic regime sweep:

- **Seeds**: 5 fixed seeds `{0, 1, 2, 3, 4}` (matches existing test-pin convention; deterministic and fast).
- **Target AUC points**: 7 values across the `synthetic_rwd_realistic` calibration range `[0.55, 0.85]`:
  ```
  [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
  ```
  Mapping target AUC to the regime's signal scale uses the existing tuning surface in `synthetic_rwd_realistic.py` — set the `signal_to_noise_scale` parameter and run the cohort generator + trainer end-to-end. Implementation note: the [0.62, 0.68] band literal in the synthetic regime test pins is a CALIBRATION TARGET, not a fitness criterion; the sweep will need to allow the regime's signal scale to vary outside that pin band, then restore the pinned scale post-sweep.
- **Number of replicates per cell**: 1 (seed-determined).
- **Total cells**: 5 × 7 = 35.

### 2.2 Per-cell computation

For each (seed, target_auc) cell:

1. Generate a synthetic cohort using `synthetic_rwd_realistic.py` with the seed and signal scale that targets that AUC. Use the standard cohort size used by tests (n_train ≈ 1000, n_val ≈ 200, n_test ≈ 200).
2. Run the full model-trainer pipeline: preprocessor → trainer → evaluator. The evaluator emits both `test_metrics.roc_auc` and `permutation_test.permutation_null_p99` (PR #117).
3. Extract:
   - `test_auc` from `test_metrics.roc_auc` — the realized AUC (may differ from target by ±0.02 due to seed noise).
   - `perm_null_p99` from `permutation_test.permutation_null_p99` — the upper tail of the null distribution at this signal level.
   - `margin_p99 = test_auc - perm_null_p99`.
4. Record the (target_auc, realized_auc, perm_null_p99, margin_p99) tuple.

### 2.3 Threshold-fit logic

After all 35 cells run:

1. **Confirm the sweep is well-conditioned**: realized_auc should track target_auc within ±0.02 across all seeds. If a target_auc point's realized values cluster at a different value (e.g., target=0.55 but realized=0.62 for all seeds), the regime's scale parameter is non-monotonic at that point and that point should be discarded.
2. **Find the buffer**: at each target_auc point, compute the **5th-percentile margin** across seeds (i.e., the worst-seed margin). The buffer must pass at every target point — otherwise legitimate signal at the lower end of the AUC range would trigger advisory false alarms.
3. **Suggested threshold**: `buffer = floor(min over target_auc points of P5_margin) - safety_margin`, rounded down to a clean fraction (e.g., 0.04 → 0.03; 0.06 → 0.05).
4. **Test under α=0.05 sanity check**: confirm that at the buffer's chosen value, the advisory does NOT fire on the synthetic_rwd_realistic regime's pinned `[0.62, 0.68]` cell — that pin must remain green.

---

## 3. Recommended threshold: TBD

> **COMPUTE PENDING.** The full 35-cell sweep is estimated at 30-45 min wall-time on a single-process pipeline runner. It is deferred from this G4 documentation gate to a separate compute-runtime step. This section will be filled in once the sweep completes.

| Target AUC | Realized AUC (mean ± std over 5 seeds) | Perm null p99 (mean ± std) | Margin p99 (mean ± std) | Margin p99 (P5) |
| ---------- | -------------------------------------- | -------------------------- | ----------------------- | --------------- |
| 0.55       | TBD                                    | TBD                        | TBD                     | TBD             |
| 0.60       | TBD                                    | TBD                        | TBD                     | TBD             |
| 0.65       | TBD                                    | TBD                        | TBD                     | TBD             |
| 0.70       | TBD                                    | TBD                        | TBD                     | TBD             |
| 0.75       | TBD                                    | TBD                        | TBD                     | TBD             |
| 0.80       | TBD                                    | TBD                        | TBD                     | TBD             |
| 0.85       | TBD                                    | TBD                        | TBD                     | TBD             |

**Recommended `T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT`:** TBD (after sweep). Provisional retention of `0.05` until sweep completes.

---

## 4. Compute-runner harness (proposed)

When the sweep is run, suggested invocation pattern:

```bash
# Pseudo-code; the real script does not exist yet.
for seed in 0 1 2 3 4; do
  for auc in 0.55 0.60 0.65 0.70 0.75 0.80 0.85; do
    PYTHONPATH=src python scripts/calibration/run_t22_synth_sweep.py \
      --seed "$seed" \
      --target-auc "$auc" \
      --output-jsonl "calibration_runs/t22_synth_seed${seed}_auc${auc}.jsonl"
  done
done
# Aggregate results
python scripts/calibration/aggregate_t22_sweep.py \
  --input-glob "calibration_runs/t22_synth_*.jsonl" \
  --output-md "docs/calibration/t22_perm_anchored_synth_20260510_results.md"
```

The harness scripts (`run_t22_synth_sweep.py`, `aggregate_t22_sweep.py`) do not yet exist. They will:

1. **`run_t22_synth_sweep.py`** — invoke the trainer pipeline on a single (seed, auc) cell. Honor existing test config (`scenario_a_balanced` regime; n=1000 train; permutation count = 200 per Tier 1B step 1). Emit a JSONL row with `{seed, target_auc, realized_auc, perm_null_p99, margin_p99}`.
2. **`aggregate_t22_sweep.py`** — read all JSONL rows, compute per-target-AUC P5 margin, fit the buffer per the §2.3 logic, write the recommended threshold + the table in §3 to a markdown report.

---

## 5. Acceptance criteria for the sweep

The sweep counts as **completed and threshold-fit valid** when ALL of the following hold:

1. All 35 cells produced a non-error pipeline run with `test_auc`, `perm_null_p99` populated.
2. Per-target-AUC realized vs. target tracks within ±0.02 mean across seeds. Cells outside this band are flagged for re-run (regime scale parameter may need re-tuning) and excluded from the buffer fit until the band is restored.
3. The recommended buffer value passes the synthetic_rwd_realistic regime's pinned `[0.62, 0.68]` cell — i.e., when applied to the existing pin test, the advisory does NOT fire.
4. The recommended buffer value rejects (advisory fires) on a deliberately-broken cell where `signal_to_noise_scale = 0` (pure-noise cohort) — this is a regression check that the buffer is large enough to discriminate noise from signal.

---

## 6. Why this counts as eligible threshold-fitting (vs. real-cohort fitting)

**The synthetic regime is generative.** No real-cohort metric was used to argue for the buffer's correctness: the buffer is set against the empirical perm-null distribution of a parameterized regime. The synthetic regime's calibration to claims-only literature anchor values (`[0.62, 0.68]`) was done **once** in PR #84 and pinned via test; the sweep below TUNES the buffer literal but does NOT re-tune the regime.

**Counter-argument considered:** "The synthetic regime is itself anchored to CSU/Optum-published values — the band literal was tuned to match published literature anchors that may be the same anchors CSU/Optum metric values inhabit." Response: the synthetic regime's calibration target was the **literature anchor** (`[0.62, 0.68]` derived from psoriasis 0.67, AD 0.63, severe asthma 0.66 published claims-only research), not the CSU n=9607 observed `0.6592` or the Optum n=1294 observed `cv_mean=0.6795`. The literature anchor and the observed cohort metrics happen to coincide; the synthetic regime was scaled against the former, NOT the latter.

**Counter-argument considered:** "The synthetic regime may be too easy / too hard relative to real cohorts." Response: this is the right concern. The sweep mitigates by spanning [0.55, 0.85] — a wider range than any single cohort produces. If real-cohort buffer needs differ from synthetic-buffer needs, the difference will manifest as a buffer too small / too large when the eventually-onboarded 4th cohort runs. At that point, the advisory mode will surface the issue and the buffer can be re-tuned. THIS DOC CARRIES THE COMMITMENT to re-tune when the 4th cohort lands.

---

## 7. Lifecycle and re-tuning trigger

**State:** advisory.
**Promotion to enforcement:** when ALL of:
1. The 35-cell sweep above is complete and the recommended buffer is in code.
2. At least one un-touched real-cohort run lands without violating the advisory at the buffer chosen — confirms the synthetic-fit value generalizes.
3. Domain expert sign-off on the buffer value (similar to T2.6 promotion gate).

**Re-tuning trigger** (advisory or enforced):
- A new `synthetic_rwd_realistic` regime variant is added (e.g., heavy-tailed noise or non-stationary class imbalance) — re-run the 35-cell sweep on the new regime.
- A real-cohort run violates advisory at the chosen buffer + on a cohort with otherwise good signal (perm-p ≤ 0.01, ECE ≤ 0.05, std/mean ≤ 0.05) — likely the buffer is too tight; re-run sweep with a tighter target-AUC granularity (15 points instead of 7).
- The constants `T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT` is changed in code without first updating this doc — block PR.

---

## 8. Cross-references

- **Code:** `_emit_permutation_anchored_auc_advisory` (`evaluator.py:154`); buffer constant (`evaluator.py:71`).
- **Companion docs:**
  - [`t23_cohort_bands_20260510.md`](t23_cohort_bands_20260510.md) — T2.3 cohort-derived honest band, observability-only.
  - [`t26_literature_anchors_20260510.md`](t26_literature_anchors_20260510.md) — peer-reviewed deployer-input thresholds.
  - [`t26_future_cohort_plan_20260510.md`](t26_future_cohort_plan_20260510.md) — promotion roadmap.
- **Test pin:** `tests/integration/test_t22_perm_anchored_auc_advisory.py` (PR #119) tests the advisory wiring on synthetic_rwd_realistic; will need a sweep-result-aware update post-fit.
