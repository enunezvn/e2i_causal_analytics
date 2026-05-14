# Synthetic v3 design — `rwd_realistic` regime

Canonical design reference for the `rwd_realistic` synthetic data regime that
ships in [`src/repositories/synthetic_rwd_realistic.py`](../src/repositories/synthetic_rwd_realistic.py).

Closes Phase S.3 of `.claude/plans/adaptive_temporal_validity_redesign.md` (line 266).

> **Premise correction (issue #200, verified 2026-05-14 against source):** the
> plan and the issue body both speak of "5 leakage variants"; the regime
> actually exposes **6 active variants** plus the `"none"` no-op
> (post-`borderline_genuine` addition). `signal_scale` was introduced by
> **PR #153** (commit `99f77fc6`, backlog #135 — NOT PR #148, which shipped
> v5 B2 Cox + RSF survival modeling). Both corrections were applied to this
> document rather than to the plan/issue body so the source remains the
> ground truth.

---

## 1. Why this regime exists

The legacy `clean` and `default` regimes in
[`src/repositories/sample_data.py`](../src/repositories/sample_data.py)
produce a val_AUC ≈ 0.87 on tier-0 (see
[`tests/synthetic/test_synthetic_regimes.py:401`](../tests/synthetic/test_synthetic_regimes.py)).
That is unrealistically high for a claims-only specialty-pharma model.
Published claims-only initiation models for CSU, AD and severe asthma
converge in `val_AUC ∈ [0.61, 0.67]`, and the information-theoretic
ceiling for the 6-feature / 2.4 %-prevalence regime sits at `[0.62, 0.68]`
([regime source lines 9-13](../src/repositories/synthetic_rwd_realistic.py)).

Training the leakage-defense pipeline on the easy regime hides RWD-shape
leaks (vendor-encoded post-hoc fields, partial-panel masking). The
`rwd_realistic` regime is the production-shape generator used by the
adaptive-temporal-validity test suite.

Origin: codex research output 2026-05-07, option (c) hybrid — keep the
existing regimes for plumbing tests, add `rwd_realistic` for production-
shape testing
([regime source lines 42-43](../src/repositories/synthetic_rwd_realistic.py)).

---

## 2. Regime invariants

All defaults are pinned in `RwdRealisticConfig`
([regime source lines 114-134](../src/repositories/synthetic_rwd_realistic.py)).

| Invariant                       | Default | Source line                                                                       |
|---------------------------------|---------|-----------------------------------------------------------------------------------|
| `n_patients`                    | 7000    | [`synthetic_rwd_realistic.py:118`](../src/repositories/synthetic_rwd_realistic.py) |
| `prevalence`                    | 0.024   | [`synthetic_rwd_realistic.py:119`](../src/repositories/synthetic_rwd_realistic.py) |
| `panel_fragmentation_rate`     | 0.50    | [`synthetic_rwd_realistic.py:120`](../src/repositories/synthetic_rwd_realistic.py) |
| `missing_demo_rate`             | 0.05    | [`synthetic_rwd_realistic.py:121`](../src/repositories/synthetic_rwd_realistic.py) |
| `signal_scale`                  | 1.0     | [`synthetic_rwd_realistic.py:131`](../src/repositories/synthetic_rwd_realistic.py) |
| `leakage_pattern`               | `"none"`| [`synthetic_rwd_realistic.py:122`](../src/repositories/synthetic_rwd_realistic.py) |
| `seed`                          | 42      | [`synthetic_rwd_realistic.py:132`](../src/repositories/synthetic_rwd_realistic.py) |
| Pinned val_AUC honest band      | `[0.62, 0.68]` | [`synthetic_rwd_realistic.py:263`](../src/repositories/synthetic_rwd_realistic.py) |

### 2.1 Prevalence 0.024

Default `prevalence=0.024` matches the CSU 2.4 % claims-data anchor; it
also approximates AD 4.1 % and severe-asthma 3.8 % (regime source
[line 19](../src/repositories/synthetic_rwd_realistic.py)). The realised
prevalence stays within ±1 pp of the configured target — pinned by
[`tests/unit/test_data/test_synthetic_rwd_realistic.py:41-54` (`test_prevalence_matches_target`)](../tests/unit/test_data/test_synthetic_rwd_realistic.py).

### 2.2 Panel fragmentation

`panel_fragmentation_rate=0.50` means ~50 % of patients have < 12 months
of observation, plus ~5 % have demographics-only with no clinical claims
([regime source lines 23-25](../src/repositories/synthetic_rwd_realistic.py)).
Pinned by
[`tests/unit/test_data/test_synthetic_rwd_realistic.py:56-67` (`test_panel_fragmentation_rate`)](../tests/unit/test_data/test_synthetic_rwd_realistic.py).
Fragmented patients also get index 1-6 months post-enrollment
(non-fragmented: 6-18 months);
[`_generate_eligibility` at lines 217-225](../src/repositories/synthetic_rwd_realistic.py)
threads this into the `eligeff → index_date → eligend` window calculation
that the `post_hoc_termination` leak exploits.

### 2.3 The pinned val_AUC honest band `[0.62, 0.68]`

The 4 demographic coefficients in `_generate_target` (0.25 / 0.45 / 0.20 /
0.15, [regime source lines 282-289](../src/repositories/synthetic_rwd_realistic.py))
are tuned so that a vanilla XGBoost trained on the leakage-clean feature
set lands in `val_AUC ∈ [0.62, 0.68]`. This matches the published
claims-only ceiling and is *the* discriminating contract that flags
unrealistic generators.

> **T2.3 lifecycle note:** the hardcoded `[0.62, 0.68]` literal predates
> the per-cohort honest-band derivation now in
> [`evaluator.py:117-120` (`T2_3_HONEST_BAND_*_DEFAULT`)](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py).
> Going forward, the honest band is derived per cohort from
> `baseline_test_auc`, `permutation_null_p99` and `permutation_auc_std`
> via the four lift / ceiling constants
> ([`evaluator.py:103-119`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)).
> The hardcoded `[0.62, 0.68]` is preserved on the
> `synthetic_rwd_realistic` regime as a calibration anchor (see
> [`evaluator.py:153-157`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)).
> The lifecycle is currently **advisory-observability-only** — band
> violations emit a flag, they do NOT block the deployer.

### 2.4 Demographic-only feature surface

Six demographics + four eligibility fields, no labs, no clinical severity,
no prior-medication history
([regime source lines 20-22](../src/repositories/synthetic_rwd_realistic.py)).
This matches CSU vendor-data limitations and is the reason the achievable
AUC ceiling is the modest `[0.62, 0.68]` band rather than 0.87+.

### 2.5 Access-driven missingness

Missing-data masking is conditioned on insurance product (medicaid_managed,
exchange, other) — not MCAR
([`_apply_missing_data` lines 373-383](../src/repositories/synthetic_rwd_realistic.py)).
This matches ConcertAI claims-data missingness patterns.

---

## 3. Leakage variants (6 active + 1 no-op)

The regime injects exactly one leakage column per call. Each variant is
designed to be detectable by a specific layer of the four-layer adaptive
defense
(`.claude/plans/adaptive_temporal_validity_redesign.md`).

| Variant                     | Type                | Column name suffix     | Source line                                                                       |
|-----------------------------|---------------------|------------------------|-----------------------------------------------------------------------------------|
| `none`                      | no-op (default)     | (no column added)      | [`synthetic_rwd_realistic.py:308-309`](../src/repositories/synthetic_rwd_realistic.py) |
| `post_index_aggregation`    | leak                | `_LEAK`                | [`synthetic_rwd_realistic.py:314-319`](../src/repositories/synthetic_rwd_realistic.py) |
| `post_hoc_termination`      | leak                | `_LEAK`                | [`synthetic_rwd_realistic.py:321-328`](../src/repositories/synthetic_rwd_realistic.py) |
| `treatment_leaked_code`     | leak                | `_LEAK`                | [`synthetic_rwd_realistic.py:330-334`](../src/repositories/synthetic_rwd_realistic.py) |
| `spurious_correlation`      | leak                | `_LEAK`                | [`synthetic_rwd_realistic.py:336-340`](../src/repositories/synthetic_rwd_realistic.py) |
| `pure_noise`                | CONTROL (must not flag) | `_CONTROL`           | [`synthetic_rwd_realistic.py:342-344`](../src/repositories/synthetic_rwd_realistic.py) |
| `borderline_genuine`        | v5 Gate C2 sanity-check | manifest-anchored name | [`synthetic_rwd_realistic.py:346-363`](../src/repositories/synthetic_rwd_realistic.py) |

The `Literal` type that pins the variant set is at
[`synthetic_rwd_realistic.py:68-76`](../src/repositories/synthetic_rwd_realistic.py)
— any new variant must extend it.

### 3.1 `post_index_aggregation`

A feature that counts events strictly post-index. By construction the
column is `target * rng.integers(1, 10, n)` so untreated patients land at
0 deterministically
([line 319](../src/repositories/synthetic_rwd_realistic.py)).
Should be caught by Layer 1 (temporal-validity declaration) when the
manifest's `knowable_at` window is enforced.

### 3.2 `post_hoc_termination`

A vendor-encoded `months_remaining_eligibility` feature where `eligend`
reflects the actual post-hoc termination. Untreated patients get
`12 + N(6, 3)`, treated patients get `3 + N(2, 1)`
([lines 325-328](../src/repositories/synthetic_rwd_realistic.py)).
Should be caught by Layer 3 (statistical drift / single-feature AUC
inspection) and by Layer 1 when `eligend` is correctly declared
`knowable_at=post-event`.

### 3.3 `treatment_leaked_code`

A boolean `has_z79_long_term_drug_LEAK` flag. ICD-Z79.899 ("encounter for
long-term drug therapy") is assigned post-treatment; the leak rate is
`0.85 * strength` for treated vs 0.05 for untreated
([line 332](../src/repositories/synthetic_rwd_realistic.py)).
Should be caught by Layer 1 (Z79 is in the deny-list) and by Layer 2
(adversarial-leakage z-score).

### 3.4 `spurious_correlation`

A "high single-feature AUC but no causal path" leak — a Gaussian whose
mean depends on the target (treated: N(2, 0.5), untreated: N(0, 0.5))
([lines 338-340](../src/repositories/synthetic_rwd_realistic.py)).
Should be caught by Layer 3 (statistical inspection) since the per-feature
discrimination is visible without any temporal signal.

### 3.5 `pure_noise` — CONTROL

A pure-noise `random_noise_CONTROL` column
([line 344](../src/repositories/synthetic_rwd_realistic.py)). Each layer
**must NOT** flag this; doing so is a false-positive regression.

### 3.6 `borderline_genuine` — v5 Gate C2 engineering sanity-check

A class-conditional Gaussian tuned so the permutation-null z lands in the
HBLP variance-relaxation band `[5σ, 7.5σ]` at `n_patients=20000`,
`prevalence=0.024`, `seed=42`
([regime source lines 99-111, 347-363](../src/repositories/synthetic_rwd_realistic.py)).

The injected feature is declared `knowable_at=index_date` in the synthetic
feature manifest
([`src/data/manifests/synthetic_feature_manifest.py:43`](../src/data/manifests/synthetic_feature_manifest.py)),
so the pipeline sees it as Layer 1 declared-safe.

Contract: legacy 5σ → DROP, HBLP `5σ × 1.5 = 7.5σ` declared-safe →
RETAIN. The integration test
[`tests/integration/test_synthetic_borderline_genuine_hblp_contrast.py`](../tests/integration/test_synthetic_borderline_genuine_hblp_contrast.py)
pins this contrast.

**This is a v5 Gate C2 engineering CI sanity-check, NOT RWD positive
evidence** — the synthetic generator can produce any AUC by construction;
the test pins that the pipeline routing (legacy vs HBLP) decides correctly
at the boundary
([regime source lines 94-98](../src/repositories/synthetic_rwd_realistic.py)).

---

## 4. The `signal_scale` knob

`RwdRealisticConfig.signal_scale: float = 1.0`
([source line 131](../src/repositories/synthetic_rwd_realistic.py)) is a
multiplier on the **4 demographic coefficients** in `_generate_target`
([lines 282-289](../src/repositories/synthetic_rwd_realistic.py)).

- `signal_scale = 1.0` (default) reproduces the pinned `[0.62, 0.68]`
  val_AUC band — the published claims-only ceiling.
- `signal_scale = 0` produces a pure-noise cohort (single-feature
  AUC ≈ 0.50). The base-rate and noise terms are NOT scaled, so the
  prevalence offset and noise floor survive.
- `signal_scale > 1` produces higher AUCs (the T2.2 calibration sweep
  spans target AUCs `[0.55, 0.85]` via this knob).

**Origin:** PR #153 (commit `99f77fc6`, `feat(backlog-135): T2.2
perm-anchored AUC buffer calibration sweep`). The issue #200 body credits
PR #148; PR #148 is actually "v5 B2: Cox + RSF survival modeling" — the
attribution was incorrect and has been corrected in this document.

**Why it exists:** the T2.2 perm-anchored AUC buffer calibration
(backlog #135) needs to generate cohorts at known target AUCs to compute
the buffer that separates "deployable" from "permutation-null noise."
With `signal_scale` fixed at 1.0 there was no way to sweep across target
AUCs; backlog #135 needed `[0.55, 0.85]`.

---

## 5. Cross-references

### 5.1 Executable specification

The executable spec for the regime is
[`tests/unit/test_data/test_synthetic_rwd_realistic.py`](../tests/unit/test_data/test_synthetic_rwd_realistic.py)
(20 tests, ~334 LOC). It pins:

- Cohort shape + required columns
  ([`test_basic_generation_shape`](../tests/unit/test_data/test_synthetic_rwd_realistic.py))
- Prevalence within ±1 pp of target
  ([`test_prevalence_matches_target`](../tests/unit/test_data/test_synthetic_rwd_realistic.py))
- Panel fragmentation rate
  ([`test_panel_fragmentation_rate`](../tests/unit/test_data/test_synthetic_rwd_realistic.py))
- Each leakage variant produces the expected column shape +
  adversarial-leakage z-score
  ([file lines 99-330](../tests/unit/test_data/test_synthetic_rwd_realistic.py))

End-to-end pipeline contracts on the `rwd_realistic` regime live in:

- [`tests/integration/test_layer_5_pipeline_integration.py`](../tests/integration/test_layer_5_pipeline_integration.py)
  — Layer 5 catches `post_index_aggregation` on a real `rwd_realistic`
  cohort.
- [`tests/integration/test_synthetic_borderline_genuine_hblp_contrast.py`](../tests/integration/test_synthetic_borderline_genuine_hblp_contrast.py)
  — `borderline_genuine` engineering CI contrast.

> The legacy [`tests/synthetic/test_synthetic_regimes.py`](../tests/synthetic/test_synthetic_regimes.py)
> covers the `default / adverse / clean` regimes shipped by
> `SampleDataGenerator.ml_patients()`, not `rwd_realistic` itself. A
> header pointer on that file links readers to this design doc; the
> file:line table above is the executable spec for the `rwd_realistic`
> regime proper. The cross-references are intentionally non-circular —
> the test files point at the doc; the doc points at the test files.

### 5.2 Calibration harness

[`scripts/calibration/run_t22_synth_sweep.py`](../scripts/calibration/run_t22_synth_sweep.py)
is the single-cell sweep runner for the T2.2 perm-anchored AUC buffer
calibration. It:

1. Maps `target_auc → signal_scale` via the empirical table at
   [`run_t22_synth_sweep.py:58-66` (`TARGET_AUC_TO_SIGNAL_SCALE`)](../scripts/calibration/run_t22_synth_sweep.py).
2. Generates a cohort at `n_patients=1400`, `prevalence=0.10`,
   `missing_demo_rate=0.0`
   ([lines 72-74](../scripts/calibration/run_t22_synth_sweep.py)).
3. Trains a logistic regression on the same 4 demographic features that
   `_generate_target` uses
   ([`_extract_target_features` lines 79-95](../scripts/calibration/run_t22_synth_sweep.py)
   — must stay in lockstep with the regime's coefficient set).
4. Calls `compute_permutation_test` (200 permutations) and emits one
   JSONL row per cell
   ([lines 124-149](../scripts/calibration/run_t22_synth_sweep.py)).

The downstream aggregator
[`scripts/calibration/aggregate_t22_sweep.py`](../scripts/calibration/aggregate_t22_sweep.py)
applies the §2.3 threshold-fit logic and emits a calibrated buffer (last
result: `0.05` provisional → `0.04` calibrated, pinned at
[`evaluator.py:93` (`T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT`)](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)).

### 5.3 Other consumers

- [`src/data/manifests/synthetic_feature_manifest.py`](../src/data/manifests/synthetic_feature_manifest.py)
  — declares the `borderline_genuine_feature` as `knowable_at=index_date`
  so Layer 1 sees it as declared-safe.
- [`src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:185`](../src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py)
  — Layer 1 logic that interacts with the manifest declaration.
- [`src/agents/ml_foundation/model_trainer/nodes/evaluator.py`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)
  — T2.2 / T2.3 honest-band derivation; references the `[0.62, 0.68]`
  anchor.

---

## 6. Invariants that MUST stay in lockstep

A future PR touching this regime is at risk of silently breaking the
calibration. The reviewer must verify:

1. **The 4 demographic coefficients** in `_generate_target` (line 282-289)
   AND **`_extract_target_features` in `run_t22_synth_sweep.py`**
   (lines 89-94) — these MUST be the same feature set with the same
   normalisation. Adding a 5th coefficient to one without the other
   silently breaks the T2.2 sweep.
2. **The `[0.62, 0.68]` honest-band band** is currently hardcoded as a
   calibration anchor in the regime docstring (line 263) AND repeated in
   `evaluator.py:156`. Both must move together if the band is widened.
3. **The `BORDERLINE_GENUINE_*` constants** (regime lines 107-111) AND
   the manifest declaration (`synthetic_feature_manifest.py:43`) AND the
   HBLP threshold (5σ × 1.5) must agree, otherwise the v5 Gate C2 sanity
   test passes / fails for the wrong reason.
4. **The `LeakagePattern` Literal type** (line 68-76) is the canonical
   list of variants. The branch dispatch in `_inject_leakage` (lines
   308-363) must cover exactly the same set.

---

## 7. Out of scope for this doc

- Real-Optum calibration (separate document tree under `docs/results/`).
- T2.3 cohort-derived honest band derivation logic (see
  `docs/calibration/t23_cohort_bands_20260510.md` per
  [`evaluator.py:158-160`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)).
- T2.2 calibration result interpretation (see
  `docs/calibration/t22_perm_anchored_synth_20260510_results.md` per
  [`evaluator.py:88`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)).

---

*Document last verified against source: 2026-05-14.*
*Issue: [#200](https://github.com/enunezvn/e2i_causal_analytics/issues/200).*
