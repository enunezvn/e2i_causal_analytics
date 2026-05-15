# T1.2 Closure Memo — Phase 3.4 model_trainer ablation hook

**Path chosen**: SHIP
**Branch**: `phase-3-4-model-trainer-ablation`
**Commits**:
- `394913b9` initial wiring (3 files; 6 integration tests)
- `71a9fa81` self-review fixes (tie-break + duplicate-column guards + 2 new tests)
- `5a307bdd` + `2f396ee5` closure memo
- `81f6b23a` codex MED-1: rename per_feature.delta_auc → ablation_delta_auc
- `8b058ac1` codex MED-2: apply issue #194 delta_AUC joint check to perm pass (cohort recalibration)
- `d9dd5640` codex MED-3: align decided_by to Phase 3.3 convention ("adversarial" / "adversarial_ablation")
- `7635805e` codex LOW-1+LOW-2: n_permutations<1 guard + decided_by attribution branch pin
- `664a7e71` codex LOW-3+NIT: typed state fields + malformed noqa cleanup

**Status**: Implementation + 9 integration tests + adversarial self-review + codex pass-1 fixes COMPLETE. Awaiting codex pass-2.
**PR URL**: https://github.com/enunezvn/e2i_causal_analytics/pull/229 (supervisor merges per agent constraints).

## Decision Rationale: SHIP

The acceptance gate (.claude/plans/adaptive_temporal_validity_redesign.md line 401) required:
EITHER ship a wiring that catches a leak class Phase 3.3 cannot see, OR close NULL.

**Leak class caught**: **Categorical per-category leak through OneHotEncoder.**

- Phase 3.3 numeric Layer-3 pass SKIPS non-numeric columns by design (`_select_features` at `adaptive_validity_check.py:2530` filters `not is_numeric_dtype(df[c])`).
- The legacy `check_categorical_class_separation` (`leakage_detector.py:1074`) uses Cramér's V with thresholds 0.5/0.7 on the WHOLE column. A categorical column with 11 categories where only one rare category (~12-15% prevalent) is target-leaky has whole-column Cramér's V below 0.5 (other categories dilute the signal).
- After `preprocessor.fit_transform` runs `OneHotEncoder(verbose_feature_names_out=False)`, the leaky category becomes its own binary indicator `region_<rare_category>` with strong label-shuffle z-score (e.g., z=36σ at n=2000) — well above the 5σ threshold.

**Investigation insight that motivated architecture**: My initial single-pass ablation design failed: `compute_feature_ablation`'s column-shuffle null **collapses** on rare binary indicators (shuffling preserves the marginal distribution, so the joint model trained on shuffled column has nearly the same AUC drop as the dropped-column model → `null_mean ≈ actual_delta` → z → 0). This is the same column-shuffle null weakness Phase 3.3 addresses with its strong-effect escape (|delta_AUC| > 0.30), but for rare per-category leaks the absolute delta is bounded above by ~`leak_p × p_lift` ≈ 0.05-0.10, below the escape band.

Solution: BOTH passes (label-shuffle perm + column-shuffle ablation) with MAX-rule combination, mirroring the FULL Phase 3.3 architecture (perm + ablation) but on encoded features. Label-shuffle null IS sensitive to per-category leak (label shuffle destroys target-conditional structure that makes the indicator predictive).

## Files Created/Modified

- **NEW** `src/agents/ml_foundation/model_trainer/nodes/model_eval_ablation.py` (~640 LOC after self-review fixes)
  - `classify_model_eval_ablation_severity` — byte-identical to Phase 3.3 `_classify_ablation_severity`
  - `_classify_permutation_severity` — simple z-band ladder (pre-issue-#194 form; documented limitation)
  - `_max_rule_severity` — MAX-rule combiner; perm wins ties (matches Phase 3.3 `_combine_ablation_with_permutation:2320`)
  - `_run_permutation_pass` — per-encoded-feature `compute_adversarial_score` with constant-column degradation
  - `run_model_eval_ablation` — entry point; runs perm + ablation passes, combines via MAX-rule, attaches `decided_by` tag
  - `_skipped_result` — schema-uniform "ran=False" payload builder
  - `_build_dataframe_with_names` — encoded-name attachment with duplicate-name + shape guards

- **MODIFIED** `src/agents/ml_foundation/model_trainer/nodes/evaluator.py` (added ~165 LOC after the permutation_test block)
  - Reads 7 state keys (`model_trainer_layer3_ablation_enabled` master gate + 6 tuning knobs)
  - Runs ablation on TRAIN split (rationale: `compute_feature_ablation` internally retrains; needs sufficient data)
  - Emits to `metrics_result["model_eval_ablation"]` + 3 promoted keys on `validation_metrics`
  - Advisory mode: does NOT mutate `success_criteria_met` (mirrors §4 T2.2/T2.3 lifecycle pattern)
  - Default OFF (`model_trainer_layer3_ablation_enabled=False`)

- **NEW** `tests/integration/test_model_trainer_layer3_ablation.py` (6 test cases, ~570 LOC)
  - `test_phase33_misses_per_category_categorical_leak` (acceptance pin 1+2: Phase 3.3 cannot see it + Cramér's V < 0.5)
  - `test_phase34_model_eval_ablation_catches_per_category_categorical_leak` (LOAD-BEARING: decided_by="adversarial_permutation")
  - `test_phase34_flag_off_is_inert` (schema pin: default-OFF contract)
  - `test_phase34_severity_classifier_mirrors_phase33` (composition pin: 8 cases byte-identical)
  - `test_phase34_max_rule_tie_break_mirrors_phase33` (9 cases: tie-break + strict-win semantics)
  - `test_phase34_pure_noise_does_not_flag_any_ohe_indicator` (false-positive pin: 11 OHE indicators on noise → zero flags)

## Codex/Self-Review Iteration History

**Pass-0 (self-review, commit `71a9fa81`)**: Codex-rescue subagent unavailable from subagent context; performed thorough self-review covering 8 specific concerns.

- **MEDIUM-1**: `decided_by` tie-break MISMATCH vs Phase 3.3. Initial code used `a_rank >= p_rank` for ablation credit; Phase 3.3 at `adaptive_validity_check.py:2320` uses `<=` so perm wins ties. Fixed to use strict `a_rank > p_rank`. Without fix, audit convention would silently invert between Phase 3.3 (perm tag on ties) and Phase 3.4 (ablation tag on ties).

- **MEDIUM-2**: Duplicate-column-name guard missing. `X.drop(columns=[name])` drops ALL columns with that name (not just one), silently breaking the per-feature drop loop. Added guards in `_build_dataframe_with_names` for both DataFrame input and numpy+names input.

**Pass-1 (codex adversarial review, SHIP-WITH-FIXES; commits `81f6b23a` → `664a7e71`)**: Supervisor dispatched codex pass-1; 3 MEDIUM + 3 LOW + 1 NIT findings. All addressed per-fix per-commit:

- **MEDIUM-1** (codex): `model_eval_ablation.py:669` emitted `delta_auc` field instead of Phase 3.3 audit-trail parity field `ablation_delta_auc` (canonical name at `adaptive_validity_check.py:2313`). Fixed: per_feature row now emits `ablation_delta_auc` matching Phase 3.3.

- **MEDIUM-2** (codex): perm pass was z-only (pre-issue-#194 ladder); at n≥10k label-shuffle null variance shrinks per CLT producing false-positive z >> 5σ on weak predictors. Fixed: `_classify_permutation_severity` now takes `delta_auc` + `delta_auc_floor`, applies issue #194 joint check, mirrors Phase 3.3's `hblp_classify` semantics. Cohort recalibrated (leak_p 0.08 → 0.12) to satisfy both joint-check AND Cramér's V invariants with margin. 7-case test pin.

- **MEDIUM-3** (codex): `decided_by` strings invented `"adversarial_permutation"` not in Phase 3.3's convention (`"adversarial"` for perm-only at `:1315`, `:1371`, `:1422`; only overwritten to `"adversarial_ablation"` on escalation at `:2930-2931`). Fixed: Phase 3.4 now uses byte-identical Phase 3.3 strings; `_DECIDED_BY_TO_LAYER` maps both to layer "3".

- **LOW-1**: `n_permutations < 1` guard missing — silent empty null distributions would degrade every feature's severity to info. Added `ValueError` guards at evaluator state-read boundary for both `model_trainer_ablation_n_permutations` and `model_trainer_ablation_permutation_n_permutations`.

- **LOW-2**: `decided_by` attribution branch was not exercised through assembled per_feature rows in a focused test. Added `test_phase34_decided_by_attribution_per_feature_row` pinning all 4 branches.

- **LOW-3**: Ad-hoc extra state keys not declared in typed `ModelTrainerState` schema. Added 10 typed Optional fields covering the master gate + 7 tuning knobs + factory hook + output payload.

- **NIT**: Invalid `# noqa: random_state=42 — ...` directive at `evaluator.py:523` (no comma-separated code list). Converted to plain comment.

**Concerns reviewed in pass-0 and found ACCEPTABLE**:
1. Phase 3.3 misses pin: structurally sound (relies on `_select_features` numeric filter + Cramér's V whole-column check, both stable invariants).
2. Encoded names: correct level of granularity for per-category leak class (leak lives IN `region_leak_region`, not raw column). Known limitation that legitimate strong predictors will also flag via perm pass — same as Phase 3.3's behavior on numeric features; downstream Layer 4 makes the causal call.
3. State flag namespace `model_trainer_*` distinct from Phase 3.3's `adaptive_*`; different agents/states.
4. Severity classifier byte-identical to Phase 3.3 (8-case composition pin).
5. Default OFF / advisory mode / schema uniformity invariants all preserved.
6. (Fixed) Tie-break aligned with Phase 3.3.
7. Performance: 250s worst-case at max_features=100; acceptable given opt-in advisory mode.
8. ~~Known limitation (issue #194 mirror): simple z-band has 5σ FPR blowup at n≥10k. Documented; promote to joint check if lifecycle transitions ADVISORY → ENFORCED.~~ **FIXED in codex pass-1 MED-2** (commit `8b058ac1`). The perm classifier now applies the issue #194 joint (z, |delta_AUC|) check from the start — no deferred-promotion item remains.

## Verification (post codex pass-1)

```
pytest tests/integration/test_model_trainer_layer3_ablation.py -v
================== 9 passed, 75 warnings in 153.29s (0:02:33) ==================

pytest tests/integration/test_adaptive_validity_check_ablation_layer3.py
tests/unit/test_data/test_adversarial_leakage.py -v
================== 11 passed, 75 warnings in 84.40s (0:01:24) ==================
(Phase 3.3 + adversarial_leakage unit tests unchanged.)

ruff check src/agents/ml_foundation/model_trainer/nodes/ src/agents/ml_foundation/model_trainer/state.py tests/integration/test_model_trainer_layer3_ablation.py
All checks passed!

mypy --config-file pyproject.toml src/agents/ml_foundation/model_trainer/nodes/model_eval_ablation.py src/agents/ml_foundation/model_trainer/state.py
Found 21 errors in 15 files (checked 2 source files)
(All 21 pre-existing in other files; zero new errors in new modules.)
```

Total integration tests after codex pass-1: **9 cases** (added 3 over the original 6).

## Open Follow-ups for Supervisor

1. **Codex review missed**: Codex-rescue subagent wasn't reachable from this subagent context. Supervisor or future cycle should request explicit codex pass to verify the 2 MEDIUM findings I caught are exhaustive. Likely codex would surface 1-3 additional MED/LOW issues (typical pass-1 outcome). Adversarial-review pattern from Phase 3.3 (issue #196) recorded 4-6 passes to PASS; one self-review pass is insufficient signal of completeness.

2. **Promotion to ENFORCED**: Currently advisory-only (matches §4 T2.2/T2.3 lifecycle). Promotion to gating requires (a) signed doc at `docs/calibration/{slug}_lifecycle_change_*.md` per Gate N2 acceptance, (b) cohort-derived FPR/TPR calibration sweep at production scale.

3. **issue #194 promotion**: If Phase 3.4 transitions to ENFORCED, the `_classify_permutation_severity` simple z-band should be upgraded to the joint (z, |delta_AUC|) check from issue #194 — currently has known 5σ FPR blowup at n≥10k. Documentation note added in module docstring.

4. **Audit-trail integration**: The `decided_by` field on per-feature rows uses `"adversarial_ablation"` / `"adversarial_permutation"` tags that match Phase 3.3 conventions but are NOT yet plumbed into a global verdict object equivalent to Phase 3.3's `_compose_legacy_verdict`. Downstream audit consumers reading `model_eval_ablation.per_feature[i].decided_by` will get the tag, but there's no top-level model-trainer "verdict" payload. Consider if downstream needs this for parity.

5. **CI runtime budget**: Per-encoded-feature perm pass × 200 perms + ablation pass × 30 perms can take 75-250s at max_features=100. CI doesn't currently run model_trainer Phase 3.4 — only the unit-test composition pin. Confirm CI configuration if Phase 3.4 should run in CI flag-ON mode for any synthetic regime.

## Architectural Insights Worth Recording

1. **Column-shuffle null collapses on rare binary indicators**: The exact same issue #196 strong-effect-escape rationale extends to model-eval space. `compute_feature_ablation`'s null is structurally weak for ANY binary indicator where shuffling preserves marginal distribution. Need label-shuffle perm as complementary defender. (Already implicit in Phase 3.3 architecture; reified here for OHE leak class.)

2. **Categorical leak hand-off between Layer 1 (Cramér's V whole-column) and Layer 3 (numeric only) creates structural gap**: Per-category rare leak with whole-column V < 0.5 falls through both. Model-trainer Phase 3.4 hook plugs this gap by running on encoded columns. This is a NET-NEW capability not duplicating Phase 3.3.

3. **Severity classifier reuse**: The `_classify_ablation_severity` algorithm (two-tier rule, strong-effect escape, signed delta) is genuinely cohort-agnostic and trivially reusable. Locating it in `data_preparer/.../adaptive_validity_check.py` (vs a shared module) requires Phase 3.4 to maintain a byte-identical re-implementation; future refactor could extract to `src/data/adversarial_leakage.py` alongside the perm/ablation primitives.

4. **Tie-break convention is load-bearing for audit consistency**: Perm wins ties (matches Phase 3.3); flipping to ablation-wins-ties would silently invert audit attribution. This is exactly the kind of regression an adversarial review must catch — and I did catch it on the self-review pass.

5. **Pipeline ordering analysis (data_preparer → model_trainer)**: The transforms between Phase 3.3 firing and Phase 3.4 firing are: SimpleImputer (can't introduce leak), StandardScaler (monotonic, AUC-invariant), OneHotEncoder (CAN introduce per-category exposure), apply_resampling (train-only, doesn't affect test-set evaluation). Only OHE has the structural ability to expose a leak class Phase 3.3 misses. This narrows the SHIP-vs-NULL decision space to "test categorical leakage" — done.
