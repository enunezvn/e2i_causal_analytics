# Synthetic v3 design — `rwd_realistic` regime

Canonical design reference for the `rwd_realistic` synthetic data regime that
ships in [`src/repositories/synthetic_rwd_realistic.py`](../src/repositories/synthetic_rwd_realistic.py).

Closes Phase S.3 of `.claude/plans/adaptive_temporal_validity_redesign.md`
(line 266; host-side plan file — `.claude/` is git-ignored so the path is
relative to the project root, not the repo working tree).

> **Premise corrections (issue #200, verified 2026-05-14 against source):**
> the issue body says "5 leakage variants"; the regime's `LeakagePattern`
> Literal has **7 values**: 4 leak branches (`post_index_aggregation`,
> `post_hoc_termination`, `treatment_leaked_code`, `spurious_correlation`),
> 1 pure-noise `CONTROL` (`pure_noise`), 1 declared-safe Gate C2
> sanity-check (`borderline_genuine`), and 1 `"none"` no-op. The issue
> body also credits `signal_scale` to PR #148; the actual PR is **#153**
> (commit `99f77fc6`, backlog #135). PR #148 shipped v5 B2 Cox + RSF
> survival modeling. Corrections applied to this doc, not the issue body,
> so source remains ground truth.

---

## 1. Why this regime exists

The legacy `clean` / `default` regimes in
[`sample_data.py`](../src/repositories/sample_data.py) produce val_AUC ≈
0.87 on tier-0
([`test_synthetic_regimes.py:413`](../tests/synthetic/test_synthetic_regimes.py)
holds the `val_auc=0.8746` literal) — unrealistic for claims-only
specialty-pharma. Published initiation models converge at `val_AUC ∈
[0.61, 0.67]`; the information-theoretic ceiling for the 6-feature /
2.4 %-prevalence regime sits at `[0.62, 0.68]`
([regime source lines 11-14](../src/repositories/synthetic_rwd_realistic.py)).
Training the leakage-defense pipeline on the easy regime hides RWD-shape
leaks (vendor-encoded post-hoc fields, partial-panel masking); the
`rwd_realistic` regime is the production-shape generator used by the
adaptive-temporal-validity test suite. Origin: codex research 2026-05-07
option (c) hybrid
([regime source lines 50-51](../src/repositories/synthetic_rwd_realistic.py)).

---

## 2. Regime invariants

All defaults are pinned in `RwdRealisticConfig`
([regime source lines 128-148](../src/repositories/synthetic_rwd_realistic.py)).

| Invariant                       | Default | Source line                                                                       |
|---------------------------------|---------|-----------------------------------------------------------------------------------|
| `n_patients`                    | 7000    | [`synthetic_rwd_realistic.py:131`](../src/repositories/synthetic_rwd_realistic.py) |
| `prevalence`                    | 0.024   | [`synthetic_rwd_realistic.py:132`](../src/repositories/synthetic_rwd_realistic.py) |
| `panel_fragmentation_rate`     | 0.50    | [`synthetic_rwd_realistic.py:133`](../src/repositories/synthetic_rwd_realistic.py) |
| `missing_demo_rate`             | 0.05    | [`synthetic_rwd_realistic.py:134`](../src/repositories/synthetic_rwd_realistic.py) |
| `leakage_pattern`               | `"none"`| [`synthetic_rwd_realistic.py:135`](../src/repositories/synthetic_rwd_realistic.py) |
| `leakage_strength`              | 1.0     | [`synthetic_rwd_realistic.py:136`](../src/repositories/synthetic_rwd_realistic.py) |
| `signal_scale`                  | 1.0     | [`synthetic_rwd_realistic.py:144`](../src/repositories/synthetic_rwd_realistic.py) |
| `seed`                          | 42      | [`synthetic_rwd_realistic.py:145`](../src/repositories/synthetic_rwd_realistic.py) |
| `start_date` / `end_date`       | 2022-01-01 / 2024-12-31 | [`synthetic_rwd_realistic.py:146-147`](../src/repositories/synthetic_rwd_realistic.py) |
| Pinned val_AUC honest band      | `[0.62, 0.68]` | [`synthetic_rwd_realistic.py:276`](../src/repositories/synthetic_rwd_realistic.py) |
| Output cols `is_fragmented` / `observation_months` | — | [`synthetic_rwd_realistic.py:264-265`](../src/repositories/synthetic_rwd_realistic.py) |

### 2.1 Prevalence 0.024

Default `prevalence=0.024` matches the CSU 2.4 % claims-data anchor; it
also approximates AD 4.1 % and severe-asthma 3.8 % (regime source
[line 21](../src/repositories/synthetic_rwd_realistic.py)). Realised
prevalence stays within **1.5 pp** of target (assertion `abs(realized -
target) < 0.015`) — pinned by
[`test_prevalence_matches_target` (test_synthetic_rwd_realistic.py:50-63)](../tests/unit/test_data/test_synthetic_rwd_realistic.py).

### 2.2 Panel fragmentation

`panel_fragmentation_rate=0.50` ⇒ ~50 % of patients have < 12 mo
observation, plus ~5 % demographics-only
([regime source lines 26-28](../src/repositories/synthetic_rwd_realistic.py)).
Pinned by
[`test_panel_fragmentation_rate`](../tests/unit/test_data/test_synthetic_rwd_realistic.py).
Fragmented patients get index 1-6 mo post-enrollment (non-fragmented:
6-18 mo) via
[`_generate_eligibility:230-236`](../src/repositories/synthetic_rwd_realistic.py)
— this threads into the `eligeff → index_date → eligend` window that the
`post_hoc_termination` leak exploits.

### 2.3 The pinned val_AUC honest band `[0.62, 0.68]`

The 4 demographic coefficients in `_generate_target` (0.25 / 0.45 / 0.20 /
0.15, [regime source lines 298-301](../src/repositories/synthetic_rwd_realistic.py))
are tuned so that a vanilla XGBoost trained on the leakage-clean feature
set lands in `val_AUC ∈ [0.62, 0.68]`. This matches the published
claims-only ceiling and is *the* discriminating contract that flags
unrealistic generators.

> **T2.3 lifecycle note:** the hardcoded `[0.62, 0.68]` literal predates
> the per-cohort honest-band derivation in
> [`evaluator.py:117-120` (`T2_3_HONEST_BAND_*_DEFAULT`)](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py);
> the literal is preserved as a `synthetic_rwd_realistic` calibration
> anchor at
> [`evaluator.py:153-157`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py).
> Currently **advisory-observability-only** — band violations flag, do
> not block the deployer.

### 2.4 Demographic-only feature surface

Six demographics + four eligibility fields, no labs, no clinical severity,
no prior-medication history
([regime source lines 23-25](../src/repositories/synthetic_rwd_realistic.py)).
This matches CSU vendor-data limitations and is the reason the achievable
AUC ceiling is the modest `[0.62, 0.68]` band rather than 0.87+.

### 2.5 Access-driven missingness

Missing-data masking is conditioned on insurance product (medicaid_managed,
exchange, other) — not MCAR
([`_apply_missing_data` lines 394-401](../src/repositories/synthetic_rwd_realistic.py)).
This matches ConcertAI claims-data missingness patterns.

---

## 3. Leakage variants

The regime injects exactly one leakage column per call. The
`LeakagePattern` Literal at
[`synthetic_rwd_realistic.py:76-84`](../src/repositories/synthetic_rwd_realistic.py)
is canonical: 4 leak branches + 1 CONTROL + 1 declared-safe sanity-check
+ 1 no-op = 7 values. Each is designed to exercise a specific layer of
the 4-layer adaptive defense (host-side plan
`.claude/plans/adaptive_temporal_validity_redesign.md`; `.claude/` is
git-ignored).

| Variant                     | Type / `_LEAK` vs `_CONTROL`            | Expected defense layer | Mechanic                                                                 | Source                                                                          |
|-----------------------------|-----------------------------------------|------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `none`                      | no-op (default; no column added)        | n/a                    | early return                                                              | [`:321`](../src/repositories/synthetic_rwd_realistic.py)                       |
| `post_index_aggregation`    | leak (`_LEAK`)                          | Layer 1 (temporal)     | `target * rng.integers(1, 10, n)` — 0 for untreated by construction       | [`:327-332`](../src/repositories/synthetic_rwd_realistic.py)                   |
| `post_hoc_termination`      | leak (`_LEAK`)                          | Layer 1 + Layer 3      | `eligend` reflects actual termination; untreated `12+N(6,3)` vs treated `3+N(2,1)` | [`:334-341`](../src/repositories/synthetic_rwd_realistic.py)         |
| `treatment_leaked_code`     | leak (`_LEAK`)                          | Layer 1 + Layer 2      | Z79.899 ("long-term drug therapy") assigned post-treatment; rate `0.85·strength` vs 0.05 | [`:343-347`](../src/repositories/synthetic_rwd_realistic.py)  |
| `spurious_correlation`      | leak (`_LEAK`)                          | Layer 3                | target-conditional Gaussian; treated N(2, 0.5), untreated N(0, 0.5)       | [`:349-353`](../src/repositories/synthetic_rwd_realistic.py)                   |
| `pure_noise`                | CONTROL (`_CONTROL`; **must not flag**) | regression sentinel    | pure Gaussian noise; flagging is a false-positive regression              | [`:355-357`](../src/repositories/synthetic_rwd_realistic.py)                   |
| `borderline_genuine`        | declared-safe sanity-check (manifest-anchored name) | v5 Gate C2 routing | see §3.1 below                                                            | [`:359-385`](../src/repositories/synthetic_rwd_realistic.py)                   |

### 3.1 `borderline_genuine` — v5 Gate C2 routing sanity-check

A class-conditional Gaussian tuned so the permutation-null z lands in the
HBLP variance-relaxation band `[5σ, 7.5σ]` and `|delta_AUC| ≈ 0.05` at
`n_patients=20000`, `prevalence=0.024`, `seed=42` (calibration constants
at
[`:120-124`](../src/repositories/synthetic_rwd_realistic.py)). The
injected feature is declared `knowable_at=index_date` in the synthetic
manifest
([`synthetic_feature_manifest.py:52`](../src/data/manifests/synthetic_feature_manifest.py)),
so the pipeline sees it as Layer 1 declared-safe.

**Contract (post-issue-#194):** the Layer 5 joint check
`severity ∈ {moderate, high} ⇔ (z > k) AND (|delta_AUC| > epsilon=0.10)`
applies to BOTH arms. Since `|delta_AUC| ≈ 0.05 < 0.10` floor, BOTH the
legacy 5σ arm AND the HBLP `5σ × 1.5 = 7.5σ` declared-safe arm RETAIN
the feature — the joint check correctly classifies it as a benign weak
signal, not a leak. HBLP's variance-inflation prior remains active and
is verified separately by `test_v5_c2_hblp_relaxation_actually_fired`.
Pre-issue-#194 the contract was "legacy DROPS, HBLP RETAINS" via the z
threshold alone; the executable spec is now
[`test_synthetic_borderline_genuine_hblp_contrast.py`](../tests/integration/test_synthetic_borderline_genuine_hblp_contrast.py)
(see `test_v5_c2_legacy_drops_hblp_retains_borderline_genuine` line 158
— the function name preserves the historical phrasing; the retain
assertions at lines 200-205 and the relaxation-fired test at line 289
pin the post-#194 behavior).

**This is a v5 Gate C2 engineering CI sanity-check, NOT RWD positive
evidence** — the generator can produce any AUC by construction; the test
pins routing-at-boundary correctness only
([regime source lines 107-110](../src/repositories/synthetic_rwd_realistic.py)).

---

## 4. The `signal_scale` knob

`RwdRealisticConfig.signal_scale: float = 1.0`
([line 144](../src/repositories/synthetic_rwd_realistic.py)) multiplies
the 4 demographic coefficients in `_generate_target`
([lines 298-301](../src/repositories/synthetic_rwd_realistic.py)). Base
rate + noise are NOT scaled. `scale = 1.0` reproduces `[0.62, 0.68]`;
`scale = 0` produces single-feature AUC ≈ 0.50; `scale > 1` raises AUC.
The T2.2 calibration sweep spans target AUCs `[0.55, 0.85]` through this
knob.

**Origin:** PR #153 commit `99f77fc6` (`feat(backlog-135): T2.2
perm-anchored AUC buffer calibration sweep`). The T2.2 calibration
(backlog #135) needs cohorts at known target AUCs to compute the
deployable-vs-noise buffer; pre-PR-#153 the regime had no AUC knob.

---

## 5. Cross-references

### 5.1 Executable specification

The unit spec is
[`test_synthetic_rwd_realistic.py`](../tests/unit/test_data/test_synthetic_rwd_realistic.py)
(14 tests, 343 LOC after pass-1 MED-3 header pointer): pins cohort
shape, prevalence (1.5 pp tolerance), panel fragmentation rate, and
per-variant column + adversarial-leakage z-score (leakage tests at
file lines 101-343). End-to-end pipeline contracts live in
[`test_layer_5_pipeline_integration.py`](../tests/integration/test_layer_5_pipeline_integration.py)
(Layer 5 catches `post_index_aggregation`) and
[`test_synthetic_borderline_genuine_hblp_contrast.py`](../tests/integration/test_synthetic_borderline_genuine_hblp_contrast.py)
(`borderline_genuine` CI contrast).

The legacy
[`test_synthetic_regimes.py`](../tests/synthetic/test_synthetic_regimes.py)
covers `default / adverse / clean` (`SampleDataGenerator.ml_patients()`),
NOT `rwd_realistic`. Both the legacy file and the rwd_realistic spec
file carry header pointers back to this doc — cross-references are
non-circular and symmetric.

### 5.2 Calibration harness

The T2.2 perm-anchored AUC buffer is calibrated via
[`scripts/calibration/run_t22_synth_sweep.py`](../scripts/calibration/run_t22_synth_sweep.py)
(`target_auc → signal_scale` map at
[`:58-66`](../scripts/calibration/run_t22_synth_sweep.py), cohort
constants at `:71-73`, feature extraction at `:79-96` — must stay
lockstep with `_generate_target`). The aggregator
[`aggregate_t22_sweep.py`](../scripts/calibration/aggregate_t22_sweep.py)
emits the calibrated buffer pinned at
[`evaluator.py:93`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)
(`T2_2_PERMUTATION_ANCHORED_AUC_BUFFER_DEFAULT`; last result 0.05
provisional → 0.04 calibrated).

### 5.3 Other consumers

- [`src/data/manifests/synthetic_feature_manifest.py`](../src/data/manifests/synthetic_feature_manifest.py)
  — declares the `borderline_genuine_feature` as `knowable_at=index_date`
  so Layer 1 sees it as declared-safe.
- [`src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py:273-284`](../src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py)
  — Layer-1-conditional HBLP inflation: declared-safe features
  (`knowable_at <= index_date`) get the 1.5× prior multiplier on the
  Layer 3 z-threshold (encodes the structural prior that manifest-cleared
  features need stronger statistical evidence to be reclassified as
  leaks). Helper: `T2_1B_HBLP_DECLARED_SAFE_PRIOR_MULTIPLIER`.
- [`src/agents/ml_foundation/model_trainer/nodes/evaluator.py`](../src/agents/ml_foundation/model_trainer/nodes/evaluator.py)
  — T2.2 / T2.3 honest-band derivation; references the `[0.62, 0.68]`
  anchor.

---

## 6. Invariants that MUST stay in lockstep

A future PR touching this regime is at risk of silently breaking the
calibration. The reviewer must verify:

1. **The 4 demographic coefficients** in `_generate_target` (lines 298-301)
   AND **`_extract_target_features` in `run_t22_synth_sweep.py`**
   (lines 89-95) — these MUST be the same feature set with the same
   normalisation. Adding a 5th coefficient to one without the other
   silently breaks the T2.2 sweep.
2. **The `[0.62, 0.68]` honest-band band** is currently hardcoded as a
   calibration anchor in the regime docstring (line 276) AND repeated in
   `evaluator.py:156`. Both must move together if the band is widened.
3. **The `BORDERLINE_GENUINE_*` constants** (regime lines 120-124) AND
   the manifest declaration (`synthetic_feature_manifest.py:52`) AND the
   HBLP threshold (5σ × 1.5) must agree, otherwise the v5 Gate C2 sanity
   test passes / fails for the wrong reason.
4. **The `LeakagePattern` Literal type** (lines 76-84) is the canonical
   list of variants. The branch dispatch in `_inject_leakage` (lines
   321-385) must cover exactly the same set.

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
