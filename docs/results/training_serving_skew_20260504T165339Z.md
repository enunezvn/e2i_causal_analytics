# Training-serving skew — R6 invariant evidence

**Date:** 2026-05-04
**Branch:** `feat/phase6-training-serving-skew`
**Base:** `0309020` (post PR #40 / phase 4b CSU masking)
**Origin:** `tier0_evaluation_gap_report_20260504.md` shard #9 + `prod_readiness_backlog.md` §1
**Rubric line:** R6 of `mlops_data_pipeline_engineering_distilled.md` — *"serving features identical to training features."*

This document captures the evidence for adding a CI-runnable regression
guard on the training-serving feature-schema invariant. The Block-2 Feast
parity tests (PR #36, 9/9 FVs under `FEAST_INTEGRATION=1`) verify VALUE
equality between offline and online stores; this shard verifies
FEATURE-SET equality between the trained model and the BentoML serving
wrapper.

## Pipeline reality (verified)

The canonical ml-foundation pipeline order, per
`scripts/run_tier0_test.py:2537-3306`:

```
data_preparer → model_trainer (step 5) →
feature_analyzer (step 6) → model_deployer (step 7)
```

`feature_analyzer` runs **after** `model_trainer`, not before. Its
pruning logic in `feature_selector.py:107-162` produces
`selected_features` / `selected_features_all` for SHAP and importance
ranking — but those state keys are **NOT consumed** by the trainer
(which reads `state["train_data"]["X"]` directly at
`preprocessor.py:264-269`) nor by the deployer (input dict at
`run_tier0_test.py:3341` omits the key).

Net effect: `feature_analyzer.selected_features` is **advisory only**
in the current pipeline. The model is trained, registered, and served
on the pre-pruning column set.

## What this shard delivers

A new CI-runnable integration test at
`tests/integration/test_training_serving_skew.py` (default Integration
Tests lane, not `FEAST_INTEGRATION=1`-gated) that asserts three
invariants on the model + preprocessor contract that the BentoML
serving wrapper depends on:

1. **Serving request schema = preprocessor fit columns.** The columns
   `bentoml_service.py:497-510` requests at inference time
   (`numeric_features + categorical_features`) match exactly what the
   preprocessor was fit on.
2. **Preprocessor output schema = model input schema.** The
   `preprocessor.feature_names_out_` (post-encoding) equals
   `model.feature_names_in_`.
3. **End-to-end serving call succeeds.** A DataFrame containing only
   the serving-request columns transforms cleanly through the
   preprocessor and predicts without column-mismatch errors.

The test serves as a **regression guard**: today's wiring (per the
audit above) is skew-free because pruning is advisory, but a future
change wiring `selected_features` into the trainer or deployer
without updating the other side will break invariant 1 or 2 and the
test will catch it.

## Discriminating coverage

Per `feedback_pr_merge_workflow.md` §7, every passing test is checked
against vacuous-pass risk:

- `test_fixture_exercises_both_numeric_and_categorical_branches`:
  asserts ≥4 numeric + ≥1 categorical features were auto-detected,
  catching the case where the fixture degrades to a single-branch
  exercise.
- `test_categorical_dtype_branch_exercised`: confirms both the
  string-dtype and `pd.CategoricalDtype` branches at
  `preprocessor.py:172-178` fire on the test fixture.

The fixture deliberately mixes 4 numeric features (gaussian + uniform
distributions) with 2 low-cardinality categoricals (string + Categorical
dtype), producing a non-trivial post-encoding schema with ≥2 one-hot
expansion columns.

## Out of scope

- **High-cardinality categorical handling.** `_detect_feature_types`
  at `preprocessor.py:175-180` skips columns with > 50 unique values,
  but the `ColumnTransformer` at `preprocessor.py:93-97` uses
  `remainder="passthrough"`, which routes those columns to the output
  unchanged — that fails downstream scaling on string content. The
  fixture intentionally omits such columns; the underlying issue is
  pre-existing and not addressed here.
- **Active wiring of feature_analyzer pruning to the trainer.** That
  would be a separate shard; this test guards the invariant
  *regardless* of whether such wiring lands.
- **Source-derived timestamp provenance.** R4 production-hardening
  concern unchanged from `tier0_evaluation_vs_distilled_mlops.md`
  R4-semantic gap — orthogonal to feature-schema skew.

## Acceptance evidence

### CI gates (all green)

```
$ .venv/bin/python -m ruff check tests/integration/test_training_serving_skew.py
All checks passed!

$ .venv/bin/python -m ruff format --check tests/integration/test_training_serving_skew.py
1 file already formatted

$ .venv/bin/python -m mypy --config-file pyproject.toml tests/integration/test_training_serving_skew.py
   # 14 pre-existing baseline errors in OTHER modules; 0 errors in test file

$ .venv/bin/python -m pytest tests/integration/test_training_serving_skew.py -v
   tests/integration/test_training_serving_skew.py::test_fixture_exercises_both_numeric_and_categorical_branches PASSED
   tests/integration/test_training_serving_skew.py::test_categorical_dtype_branch_exercised PASSED
   tests/integration/test_training_serving_skew.py::test_serving_feature_names_match_preprocessor_fit_columns PASSED
   tests/integration/test_training_serving_skew.py::test_preprocessor_output_schema_matches_model_input_schema PASSED
   tests/integration/test_training_serving_skew.py::test_end_to_end_serving_input_predicts_without_column_mismatch PASSED
   tests/integration/test_training_serving_skew.py::test_advisory_pruning_path_does_not_silently_alter_serving_contract PASSED
   ====== 6 passed in 52.74s ======
```

### Failure modes the test catches (regression-guard semantics)

If a future PR introduces any of the following, the test fails:

| Failure mode | Test that catches it |
|---|---|
| Preprocessor refit on a different column set than serving requests | `test_serving_feature_names_match_preprocessor_fit_columns` |
| Model retrained on schema diverging from preprocessor output | `test_preprocessor_output_schema_matches_model_input_schema` |
| Serving column ORDER drifts from training order | `test_end_to_end_serving_input_predicts_without_column_mismatch` |
| `feature_analyzer.selected_features` wired into trainer without deployer update | `test_serving_feature_names_match_preprocessor_fit_columns` (preprocessor would be fit on different columns than serving requests) |

## R6 re-grade

Per `.claude/plans/tier0_evaluation_vs_distilled_mlops.md` §2 R6, the
prior grade was:

```
R6 Pre-deploy gates — PASS | D ✅ | E ✅ | Ex ⚠️ (no planted-hazard confirmation that gate fires)
```

The Ex ⚠️ on R6 in the source plan was about planted-hazard
confirmation on the QC gate, not feature-schema skew. This shard
addresses an adjacent R6 sub-concern (training-serving feature-set
equality) that the gap report flagged as IMPORTANT but the source plan
hadn't itemised. After this shard:

```
R6 Pre-deploy gates — PASS | D ✅ | E ✅ (feature-schema skew now CI-guarded) | Ex ⚠️ (planted-hazard suite gating QC pending)
```

The feature-schema invariant is now part of the default Integration
Tests CI lane and will fire on any drift introduced by future
refactors.

## Files touched

```
tests/integration/test_training_serving_skew.py    (new, 6 tests)
docs/results/training_serving_skew_20260504T165339Z.md (this file)
.gitignore                                          (allowlist exception)
```

No source code changes. The test exercises existing wiring and
documents the invariant as a regression guard.
