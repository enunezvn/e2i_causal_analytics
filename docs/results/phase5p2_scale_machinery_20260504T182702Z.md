# Phase 5.2 dry-run scale-machinery validation

**Date:** 2026-05-04
**Branch:** `feat/phase5p2-scale-machinery`
**Base:** `a2bd6b1` (post PR #41 / phase 6 R6 regression guard)
**Origin:** discussion follow-up to `prod_readiness_backlog.md` §2 — synthetic-data validation of pipeline machinery at scale boundaries.

This document captures the **synthetic-data scale-machinery dry run** complementing the `feat/phase5p2-initiation-revalidation` shard (real-data n=972 Optum re-validation). It does NOT close Phase 5.2 — that requires real Optum cohorts at n ≥ 200 for all three of {initiation, discontinuation, persistence}, and the latter two are still gated at n=47.

## Scope

- **In scope.** CI-runnable invariant suite at boundary points the real Optum cohorts can't reach (`tests/integration/test_phase5p2_scale_machinery.py`, 10 tests). Pins Step-5 split-validation floor, synthetic-target signal integrity, and runtime envelope on `ml_patients` generator.
- **Out of scope.** Full tier-0 grid runs at n × prevalence (codex consult 2026-05-04 estimated ~25–75 min for 4–6 full runs); not added in this PR. Documented as recurring local-only validation per codex verdict H.

## Why this is NOT Phase 5.2

Phase 5.2 (`tier0_evaluation_vs_distilled_mlops.md:691-703`) requires:
1. n ≥ 200 per cohort with explicit positive-class power analysis
2. All R1–R7 grades ≥ B at **Optum scale**

Synthetic data can answer the **scale-machinery readiness** question (does the pipeline tolerate n=200 to n=5000 across imbalance regimes?) but NOT the **Optum-specific R-grade** question (does the leakage-safe Optum converter produce ≥ B grades against real data?). The companion `feat/phase5p2-initiation-revalidation` PR closes Phase 5.2 partially for the initiation cohort only.

## What this shard delivers

`tests/integration/test_phase5p2_scale_machinery.py` — 10 tests in 4 sections:

### A. Discriminating-coverage guards (vacuous-pass protection per `feedback_pr_merge_workflow.md` §7)

| Test | Pins |
|---|---|
| `test_grid_is_non_empty` | SCALE_GRID has ≥3 combos and WEAK_SIGNAL_GRID has ≥1 |
| `test_sample_data_generator_imports_cleanly` | SampleDataGenerator exposes `ml_patients()` |

### B. Synthetic data generation invariants

| Test | Combos | Pins |
|---|---|---|
| `test_ml_patients_produces_valid_frame_at_scale` | `(200, 0.10)`, `(200, 0.25)`, `(1500, 0.10)` | row count, target presence, both classes represented, realised prevalence within tolerance |
| `test_ml_patients_at_weak_signal_boundary` | `(200, 0.02)` | generator does not crash at the must-skip boundary |

### C. Step-5 split-validation floor (the empirical 2026-04-24 boundary)

| Test | Pins |
|---|---|
| `test_step5_split_floor_passes_at_n200_prev10` | At n=200/prev=10%, the 60/20/15/5 split (120/40/30/10) clears the documented `min_samples_per_split=10` floor at `run_tier0_test.py:5573-5581` |
| `test_step5_split_floor_below_threshold_at_n47` | At n=47 (Optum discontinuation/persistence), train=28/val=9/test=7/holdout=3 violates the floor in ≥2 splits — explaining the empirical Step 5 failure documented in `optum_tier0_cohort_run_20260424_011323.md` |

### D. Permutation invariant (verdict-gate sanity)

| Test | Pins |
|---|---|
| `test_permutation_shuffled_target_breaks_signal_at_scale` | At n=1500/prev=30%, the canonical signal feature `days_on_therapy` correlates with the target ≥3× more strongly than with shuffled labels — confirming the verdict-gate's permutation-test invariant has substrate |

### E. Scale-runtime sanity

| Test | Pins |
|---|---|
| `test_n5000_generation_completes_within_runtime_envelope` | `ml_patients(n=5000)` finishes in <30s — guards against generator performance regressions that would compound at full-tier-0 scale |

## Combo grid rationale (per codex consult 2026-05-04, agent `a847abbf19d4da2e9`)

| Combo | Purpose |
|---|---|
| n=200, prev=10% | Step-5 split-validation floor — minimum reliable boundary |
| n=200, prev=25% | Verdict-gate stability at tiny n |
| n=1500, prev=10% | OOM-boundary regime (matches `repeated_k10_test_oom_followup.md` n=1500 OOM size) |
| n=200, prev=2% | Documented "must-skip" weak-signal boundary — generator-only assertion |

Skipped from CI (local-only follow-up):
- n=5000 / prev=10% (full tier-0 stress) — codex risk #2 says `xdist_group` doesn't fix the eager-import OOM root cause; deferred to manual local validation
- n=1500 / prev=2% (severe imbalance + scale) — overlaps with n=200/prev=10% machinery coverage

## Companion data point — real Optum n=972 (from `feat/phase5p2-initiation-revalidation`)

The companion shard's runtime: tier-0 against real Optum initiation (n=972, 2.88% prev) finished in 159.8s. Per-step:

| Step | Status | Duration |
|---|---|---|
| 1 SCOPE DEFINER | ✅ | ~0s |
| 2 DATA PREPARER | ✅ | ~5s |
| 3 COHORT CONSTRUCTOR | ✅ | ~2s |
| 4 MODEL SELECTOR | ✅ | ~10s |
| 5 MODEL TRAINER | ⚠️ | 37.1s |
| 5b ALGORITHM COMPARISON | ✅ | (4 algos) |
| 6 FEATURE ANALYZER | ⚠️ | 0.4s |
| 7 MODEL DEPLOYER | ❌ correct refusal | ~0s |
| 8 OBSERVABILITY CONNECTOR | ✅ | — |

Step 5 is the dominant runtime component (~23% of total). At n=200, this would be ~7-10s; at n=5000, ~2-5 min. Codex's `~25-75 min for 4-6 runs` budget assumes default HPO trials = 10; with `--hpo-trials 1`, full grid would shrink to ~10-25 min.

## Recommended follow-up (recurring, not one-off)

Per codex verdict H — re-run the full synthetic grid every ~5 PRs that touch:
- `src/agents/ml_foundation/data_preparer/` (split logic)
- `src/agents/ml_foundation/model_trainer/nodes/preprocessor.py` or `model_trainer_node.py`
- `src/agents/ml_foundation/feature_analyzer/nodes/feature_selector.py`
- `src/repositories/sample_data.py` (`ml_patients` interface or signal features)
- `scripts/run_tier0_test.py` verdict-gate logic

Manual local recipe:

```bash
# Per-combo command pattern (run from repo root)
.venv/bin/python scripts/run_tier0_test.py \
  --regime adverse \
  --imbalanced 0.10 \
  --hpo-trials 1 \
  --no-bentoml --no-save --disable-mlflow
```

Capture: pass/fail per step, peak RSS, per-combo Step-5 duration. Compare against baseline grid (this doc) and flag regressions.

## Acceptance evidence

```
$ ruff check tests/integration/test_phase5p2_scale_machinery.py
All checks passed!

$ ruff format --check tests/integration/test_phase5p2_scale_machinery.py
1 file already formatted

$ mypy --config-file pyproject.toml tests/integration/test_phase5p2_scale_machinery.py
   # no errors in test file (14 pre-existing baseline errors in OTHER modules — unchanged)

$ pytest tests/integration/test_phase5p2_scale_machinery.py -v
   10 passed in ~55s
```

## Files touched

```
tests/integration/test_phase5p2_scale_machinery.py        (NEW, 10 tests)
docs/results/phase5p2_scale_machinery_20260504T182702Z.md (this file)
.gitignore                                                  (allowlist exception)
```

No source code changes. This is a behaviour-pinning test + evidence shard.
