# Phase 1 W4 Days 2-5 — Multi-Disease Validation Outcome

- **Date**: 2026-05-03
- **Branch**: `feat/adaptive-v3-phase1` (50 ahead `main`, NOT pushed)
- **Source-of-truth plan**: `.claude/plans/synthetic_data_generator_v2/09-acceptance-and-risks.md` Section F
- **Driver**: `.claude/plans/synthetic_data_generator_v2/ralph_loop_w4_d25_brief.md`

## Summary

Synthetic generator v2 successfully fed Phase 1 W4 days 2-5 multi-disease validation end-to-end. All 3 scenarios (A/B/C) materialize at target prevalence within ±0.02, and the multi-seed AUC band acceptance gate (9/10 / 8/10 / 10/10 per shard) holds at full n_total=6000 sample size. The orchestrator emitted v2-sourced JSON+MD artifacts; per-scenario diagnostic JSONs carry `schema_version="phase1_diagnostic.v2"`.

The unchecked closing-checklist item in shard 09 §F is now ticked.

## Acceptance evidence

### Multi-disease orchestrator (`scripts/run_phase1_multi_disease.py --no-rwd-validation`)

- **Runtime**: 2.6s (well under 900s budget per shard 09 §C.3)
- **Exit code**: 0
- **Artifacts**:
  - `docs/results/phase1_multi_disease_20260503.json`
  - `docs/results/phase1_multi_disease_20260503.md`
- **schema_version**: `phase1_multi_disease.v1`
- **Scenarios**: A, B, C all reported success

| short | franchise | n_total | target_prev | realized_prev | Δ | audit_fingerprint (head) |
|---|---|---|---|---|---|---|
| A | Kisqali (ribociclib) | 6000 | 0.200 | 0.196 | -0.004 | fd46a94b |
| B | Fabhalta (iptacopan) | 6000 | 0.050 | 0.050 | +0.000 | 1573b5a5 |
| C | Remibrutinib (Rhapsido) | 6000 | 0.400 | 0.403 | +0.003 | 0e447fb1 |

All Δ within ±0.02 tolerance per shard 09 §A.1.

### Individual diagnostics (`scripts/run_phase1_diagnostic.py --scenario {A,B,C} --n-total 6000`)

Each emitted `docs/results/phase1_diagnostic_<short>_20260503.json` with `schema_version="phase1_diagnostic.v2"`. Both algorithms (LogisticRegression + LightGBM) returned `status="ok"`. Single-seed (seed=42) results:

| Scenario | LR AUROC | LR in band? | LightGBM AUROC | LightGBM in band? |
|---|---|---|---|---|
| A | 0.8181 | ✓ | 0.7961 | ✗ (just below 0.78) |
| B | 0.6970 | ✗ (single-seed miss; expected under 8/10 acceptance) | 0.6864 | ✗ |
| C | 0.8319 | ✓ | 0.7895 | ✗ |

Single-seed misses are **statistically expected** because the band acceptance is 9/10 (A), 8/10 (B), or 10/10 (C) over 10 seeds — see next section. LightGBM is not part of the band acceptance contract; the calibration locks LR-only per the v2 test suite design.

### 10-seed band sweep (LR-only, `scripts/_w4_d25_band_sweep.py` — temporary)

Full per-seed AUC distributions captured in `docs/results/phase1_w4_days25_band_sweep_20260503.json`.

| Scenario | Band | Median | Min | Max | in_band | Acceptance | PASS? |
|---|---|---|---|---|---|---|---|
| A | [0.78, 0.83] | 0.7932 | 0.7665 | 0.8201 | 9/10 | ≥ 9/10 | ✓ |
| B | [0.72, 0.78] | 0.7380 | 0.6651 | 0.7773 | 8/10 | ≥ 8/10 (relaxed per shard 04 §B.4.1) | ✓ |
| C | [0.82, 0.88] | 0.8362 | 0.8234 | 0.8437 | 10/10 | ≥ 9/10 | ✓ |

Matches the calibrated values from the v2 implementation cycle exactly:
- A slope_multiplier=0.67 → 9/10 in band ✓
- B slope_multiplier=0.70 → 8/10 in band (R-1 risk relaxation realized) ✓
- C slope_multiplier=1.25 → 10/10 in band ✓

### Cross-confirmation: pytest band regression tests

`pytest tests/ml/synthetic_v2/test_scenario_{a,b,c}.py::TestScenario{A,B,C}AUCBandRegression -m slow` — **3 passed, 22.58s**:
- `test_scenario_a.py::TestScenarioAAUCBandRegression::test_lr_auc_band_9_of_10_seeds` — PASS
- `test_scenario_b.py::TestScenarioBAUCBandRegression::test_lr_auc_band_8_of_10_seeds` — PASS
- `test_scenario_c.py::TestScenarioCAUCBandRegression::test_lr_auc_band_9_of_10_seeds` — PASS

The pytest band tests use the same SEEDS=range(10) and LR-only acceptance contract as the manual sweep, providing independent verification.

## Deviations from expected

None. All acceptance criteria met:

- [x] All 3 scenarios materialize at target prevalence ±0.02
- [x] Per-scenario AUC band acceptance gate holds (9/10 / 8/10 / 10/10)
- [x] Multi-disease runner produces JSON + MD artifacts in <5s
- [x] Diagnostic runner produces v2-schema JSON for each scenario
- [x] Audit fingerprints stable (deterministic byte-identity)

LightGBM single-seed AUC values are not part of the v2 acceptance contract; they are reported for diagnostic visibility only. They tend to track 0.02-0.04 below LR for these scenarios at n=6000, consistent with the small-n + low-feature-cardinality regime where LR's linear fit does not lose to gradient boosting.

## Artifacts list (this work cycle)

| Path | Type | Committable? |
|---|---|---|
| `docs/results/phase1_multi_disease_20260503.json` | runner output | yes (overwrites prior smoke) |
| `docs/results/phase1_multi_disease_20260503.md` | runner output | yes |
| `docs/results/phase1_diagnostic_A_20260503.json` | runner output | yes (overwrites prior smoke) |
| `docs/results/phase1_diagnostic_B_20260503.json` | runner output | yes (NEW — was absent before n=6000 run) |
| `docs/results/phase1_diagnostic_C_20260503.json` | runner output | yes (NEW) |
| `docs/results/phase1_w4_days25_band_sweep_20260503.json` | sweep evidence | yes |
| `docs/results/phase1_w4_days25_outcome_20260503.md` | this doc | yes |
| `scripts/_w4_d25_band_sweep.py` | one-off sweep | DELETE before commit |
| `.claude/plans/synthetic_data_generator_v2/ralph_loop_w4_d25_brief.md` | ralph-loop driver | NO — user-only `.claude/` |

## Conclusion

**Phase 1 W4 days 2-5 multi-disease validation: COMPLETE.**

Closing-checklist item in shard 09 §F (line 195) flips from `[ ]` to `[x]`:

> [x] Phase 1 W4 days 2-5 multi-disease runs successfully consume v2 outputs end-to-end with the new --scenario A|B|C paths producing v2-sourced JSON artifacts.

The synthetic_data_generator_v2 plan is now **fully complete**. Next-step options for the broader work stream:

1. **(a) Push branch + open PR** — `feat/adaptive-v3-phase1` is 50 ahead `main`; bundles 35 v3-phase1 commits + 14 v2 implementation commits + 2 v2 IMPORTANT-closure commits + 1 W4 d25 outcome commit.
2. **(b) Real RWD CSU loader implementation** — deferred per shard 07 §C.4.1; the synthesized fixture suffices for v2 acceptance, real-RWD loading awaits W4 days 4-5 needing real-cohort comparison.

Recommend option (a). Option (b) is non-blocking for v2 acceptance.

## Reproduction commands

```bash
.venv/bin/python scripts/run_phase1_multi_disease.py --no-rwd-validation
.venv/bin/python scripts/run_phase1_diagnostic.py --scenario A --n-total 6000
.venv/bin/python scripts/run_phase1_diagnostic.py --scenario B --n-total 6000
.venv/bin/python scripts/run_phase1_diagnostic.py --scenario C --n-total 6000
.venv/bin/python -m pytest tests/ml/synthetic_v2/test_scenario_a.py::TestScenarioAAUCBandRegression \
    tests/ml/synthetic_v2/test_scenario_b.py::TestScenarioBAUCBandRegression \
    tests/ml/synthetic_v2/test_scenario_c.py::TestScenarioCAUCBandRegression -m slow -v
```
