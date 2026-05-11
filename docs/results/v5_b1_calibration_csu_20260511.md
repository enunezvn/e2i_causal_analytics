# v5 B1 Calibration Validation on Real CSU — 2026-05-11

**v5 Workstream B1 acceptance evidence** per `disease_agnostic_quality_uplift_v5.md` §2 B1.

## Setup

- Data: `data/rwd/csu/e2i_ml_v3_patient_journeys.json` (9607 CSU patient journeys).
- Filtered to records with non-null `discontinuation_flag` target → 1743 rows; 1618 positives (~93% prevalence; CSU's binary-target prevalence in this subset).
- Features: `journey_duration_days`, `data_quality_score`, `journey_stage`, `journey_status`, `brand`, `age_group`, `gender`, `insurance_type` (one-hot encoded; 28 features after expansion).
- Split: 60/20/20 stratified at seed=42. Train=1045 (970 pos), Val=349 (324 pos), Test=349 (324 pos).
- Model: LogisticRegression(max_iter=500, class_weight="balanced", random_state=42).
- Calibration fit on val split; ECE evaluated on val + held-out test split.

## Result

| Method | Resolved | Val ECE before → after | Val drop | Test ECE before → after | Test drop |
|---|---|---|---|---|---|
| `auto` (v5 B1 default) | `isotonic` | 0.4093 → 0.0000 | +100.0% | 0.4153 → 0.0373 | **+91.0%** |
| `isotonic` (legacy) | `isotonic` | 0.4093 → 0.0000 | +100.0% | 0.4153 → 0.0373 | +91.0% |
| `sigmoid` (Platt) | `sigmoid` | 0.4093 → 0.0078 | +98.1% | 0.4153 → 0.0029 | +99.3% |

**Auto-policy resolution**: val n_pos = 324 > 100 (B1_AUTO_POLICY_N_POS_CROSSOVER), so the policy correctly chose `isotonic`. Test ECE drop = **+91.0%**, far above the v5 §2 B1 observability assertion of "≥30% on val set after calibration."

## Honest framing

- Val ECE drop is partly self-fit (calibration fit on val → near-zero ECE on val by construction). The **test ECE drop** is the production-relevant number.
- Both methods (isotonic + sigmoid) far exceed the 30% acceptance threshold on this CSU subset. The plan's default policy (isotonic at n_pos > 100) is correct in direction but sigmoid happens to win on this specific train/val/test split (99.3% vs 91.0% test drop). This is a single-dataset observation; policy default is anchored on the established literature (Niculescu-Mizil & Caruana 2005; Duan et al. 2020), not on this one split.
- The pre-calibration ECE of 0.41 is large because the LR was trained with `class_weight="balanced"` which biases probabilities away from the true ~93% prevalence (the model outputs probabilities near 0.5 while observed accuracy is ~0.93). Post-hoc calibration corrects this by remapping the rank-preserved scores.

## Conclusion

v5 §2 B1 acceptance criterion ("ECE drops by ≥30% on val set after calibration") is **MET** on real CSU. The auto-policy correctly routes to isotonic at n_pos = 324. Test-set ECE drop of 91.0% confirms post-hoc calibration is a load-bearing lever for the v5 C1 deployment manifest emission.

## Artifacts

- Validation script (not committed; reproducible): see `docs/specs/v5_b1_calibration_spec_20260511.md` Appendix A.

## Cross-references

- v5 plan §2 B1: `.claude/plans/disease_agnostic_quality_uplift_v5.md`
- Calibration policy + auto-resolution: `src/agents/ml_foundation/model_trainer/nodes/advanced_validation.py:apply_post_hoc_calibration`
- Calibration regression tests: `tests/unit/test_agents/test_ml_foundation/test_model_trainer/test_apply_post_hoc_calibration.py::TestV5GateB1AutoPolicy`
- Niculescu-Mizil & Caruana 2005 "Predicting Good Probabilities With Supervised Learning" (n≥1000 isotonic guidance).
- Duan et al. 2020 "Predicting Probabilities of Diabetes Complications With Calibrated Machine-Learning Models" (n_pos ≈ 100 practical crossover).
