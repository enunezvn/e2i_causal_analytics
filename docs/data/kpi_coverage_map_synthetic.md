# Synthetic KPI Coverage Map (Shard 09)

**Goal:** map every one of the 44 calculable KPIs in `config/kpi_definitions.yaml` to the
synthetic substrate that makes it return **non-NULL**, and prove it on the faithful
docker Supabase. **Result: 44/44 MAPPED — ZERO N/A, ZERO EMPTY.** (WS1-MP-008 was decommissioned in #1068 — needs protected-group fairness_metrics the substrate does not populate. WS1-DQ-008 "Label Quality (IAA)" was decommissioned in T8 by product decision — a working metric, κ≈0.76, removed from the live set; `v_kpi_label_quality` + `ml_annotations` retained in the DB.)

Reproduce:

```bash
# load the synthetic substrate (rolling-window anchored):
PYTHONPATH=$(pwd) LOKY_MAX_CPU_COUNT=1 \
  dotenv -f /path/to/.env run -- python scripts/load_synthetic_data.py --small --anchor-to-now
# probe all 44 KPIs against the faithful DB:
E2I_DB_INTEGRATION=1 python scripts/check_kpi_coverage.py
# -> TOTAL 44  MAPPED 44  EMPTY 0  N/A 0
```

## How a KPI is proven

The `kpi_query` RPC reads from `kpi_query_registry`. The DEFAULT query for each KPI
filters `is_synthetic = false` (Shard 07 default-exclude); each has an
`*_include_synthetic` twin that reads the synthetic rows. The 7 view-backed KPIs read
a `v_kpi_*` view that excludes synthetic (migration 067) and fall back to an
`*_include_synthetic` / direct-table query that includes it. WS1-MP-002/003/004/005/006
have **no registry entry** — the model-performance agent reads `ml_predictions`
columns directly; those are proven by a direct COUNT of the populated synthetic column
(the same column the agent reads). The probe map is `scripts/check_kpi_coverage.py`.

Every probe value below is the **measured** output of `kpi_query(<id>_include_synthetic, ...)`
on the faithful docker DB after a `--small --anchor-to-now` load.

## Coverage table (44 KPIs)

> WS1-DQ-008 (Label Quality / IAA) was decommissioned in T8 (product decision) and is
> omitted from this calculable-coverage table — mirroring WS1-MP-008 (#1068).

| KPI id | name | substrate (table.column → resolver) | verdict | measured |
|---|---|---|---|---|
| WS1-DQ-001 | Source Coverage - Patients | `patient_journeys.patient_id` + `reference_universe.target_count` → `data_quality_source_coverage_patients` | MAPPED | covered=1740 total=130408 |
| WS1-DQ-002 | Source Coverage - HCPs | `hcp_profiles.coverage_status` + `reference_universe` → `data_quality_source_coverage_hcps` | MAPPED | covered=1046 total=21240 |
| WS1-DQ-003 | Cross-source Match Rate | `data_source_tracking.match_rate_*` → `v_kpi_cross_source_match` (Task 5) | MAPPED | match_rate=0.904 |
| WS1-DQ-004 | Stacking Lift | `data_source_tracking.stacking_lift_percentage` → `v_kpi_stacking_lift` (Task 5) | MAPPED | lift_score=15.66 |
| WS1-DQ-005 | Completeness Pass Rate | `patient_journeys` (patient_id/brand/event_date non-null) → `data_quality_completeness_pass_rate` | MAPPED | pass_rate=0.999 |
| WS1-DQ-006 | Geographic Consistency | `patient_journeys.geographic_region` + `reference_universe` (universe_type=patient) → `data_quality_geographic_consistency` | MAPPED | max_gap=0.120 |
| WS1-DQ-007 | Data Lag (Median) | `patient_journeys.data_lag_hours` → `v_kpi_data_lag` — **stamped by Task 5b** | MAPPED | median_lag_days=7.96 |
| WS1-DQ-009 | Time-to-Release (TTR) | `etl_pipeline_metrics.time_to_release_hours` → `v_kpi_time_to_release` (Task 5) | MAPPED | avg_ttr_hours=58.59 |
| WS1-MP-001 | ROC-AUC | `ml_predictions.model_auc` → `model_performance_roc_auc` — **stamped by `stamp_model_metrics`** | MAPPED | roc_auc=0.774 |
| WS1-MP-002 | PR-AUC | `ml_predictions.model_pr_auc` (agent-read, direct) — **stamped** | MAPPED | 3738 synthetic rows populated |
| WS1-MP-003 | F1 Score | `ml_predictions.model_precision/recall` (agent-read, direct) — **stamped** | MAPPED | 3738 synthetic rows populated |
| WS1-MP-004 | Recall@Top-K | `ml_predictions.rank_metrics` (agent-read, direct) — **stamped** | MAPPED | 3738 synthetic rows populated |
| WS1-MP-005 | Brier Score | `ml_predictions.brier_score` (agent-read, direct) — **stamped** | MAPPED | 3738 synthetic rows populated |
| WS1-MP-006 | Calibration Slope | `ml_predictions.calibration_score` (agent-read, direct) — **stamped** | MAPPED | 3738 synthetic rows populated |
| WS1-MP-007 | SHAP Coverage | `ml_predictions.shap_values` → `model_performance_shap_coverage` — **stamped** | MAPPED | coverage=1.0 |
| WS1-MP-009 | Feature Drift (PSI) | `ml_drift_history.test_statistic` (test_type=psi) → `model_performance_feature_drift` | MAPPED | avg_psi=0.094 |
| WS2-TR-001 | Trigger Precision | `triggers.outcome_tracked/outcome_value` → `trigger_performance_precision` (Shard 05 arms) | MAPPED | precision=0.391 |
| WS2-TR-002 | Trigger Recall | `triggers`+`treatment_events` → `trigger_performance_recall` | MAPPED | recall=0.474 |
| WS2-TR-003 | Action Rate Uplift | `triggers.action_taken/control_group_flag` → `trigger_performance_action_rate_uplift` (Shard 05) | MAPPED | uplift=0.257 |
| WS2-TR-004 | Acceptance Rate | `triggers.acceptance_status` → `trigger_performance_acceptance_rate` (#1124: migration 092 denominator = delivered) | MAPPED | rate≈0.50 of delivered |
| WS2-TR-005 | False Alert Rate | `triggers.false_positive_flag` → `trigger_performance_false_alert_rate` (#1118: DGP marks ~60% of tracked-but-unproductive triggers) | MAPPED | rate≈0.13–0.15 (WARNING — coherent with TR-001 precision) |
| WS2-TR-006 | Override Rate | `triggers.acceptance_status='overridden'` → `trigger_performance_override_rate` (#1119: DGP emits `overridden` at P=0.14 of delivered; migration 090 denominator = delivered) | MAPPED | rate≈0.14 of delivered |
| WS2-TR-007 | Lead Time | `triggers.lead_time_days` → `trigger_performance_lead_time` | MAPPED | median=16 |
| WS2-TR-008 | Change-Fail Rate (CFR) | `triggers.previous_trigger_id/change_failed` → `trigger_performance_cfr` — **stamped by `stamp_change_tracking`** | MAPPED | cfr=0.222 |
| WS3-BI-001 | Monthly Active Users | `user_sessions.user_id/session_start` → `v_kpi_active_users` / mau fallback (Task 5) | MAPPED | mau=30 |
| WS3-BI-002 | Weekly Active Users | `user_sessions` → `v_kpi_active_users` / wau fallback (Task 5) | MAPPED | wau=30 |
| WS3-BI-003 | Patient Touch Rate | `triggers`+`patient_journeys` → `business_impact_patient_touch_rate` (Shard 05/06) | MAPPED | touch_rate=0.916 |
| WS3-BI-004 | HCP Coverage | `hcp_profiles.coverage_status` → `business_impact_hcp_coverage` | MAPPED | coverage=52.3 |
| WS3-BI-005 | Total Prescriptions (TRx) | `treatment_events.event_type='prescription'` → `business_impact_trx` (Shard 05) | MAPPED | trx=1238 |
| WS3-BI-006 | New Prescriptions (NRx) | `treatment_events.event_type/sequence_number=1` → `business_impact_nrx` — **`sequence_number` stamped by Task 5b helper** | MAPPED | nrx=301 |
| WS3-BI-007 | New-to-Brand Rx (NBRx) | `treatment_events.event_type/brand` (first per patient) → `business_impact_nbrx` | MAPPED | nbrx=301 |
| WS3-BI-008 | TRx Share | `treatment_events.brand` → `business_impact_trx_share` | MAPPED | share=0.310 |
| WS3-BI-009 | Conversion Rate | `triggers`+`treatment_events` → `business_impact_conversion_rate` (Shard 05) | MAPPED | conversion=0.615 |
| WS3-BI-010 | Return on Investment | `business_metrics.roi` → `business_impact_roi_business_metrics` (Shard 02) | MAPPED | avg_roi=1.888 |
| BR-001 | Remi - AH Uncontrolled % | `patient_journeys.lab_values` → `brand_specific_remi_ah_uncontrolled` (Shard 04/06) | MAPPED | rate=0.763 |
| BR-002 | Remi - Intent-to-Prescribe Δ | `hcp_intent_surveys.intent_to_prescribe_change` → `v_kpi_intent_to_prescribe` / fallback (Task 5) | MAPPED | intent_delta=1.006 |
| BR-003 | Fabhalta - % PNH Tested | `treatment_events.event_type` → `brand_specific_fabhalta_pnh_tested` | MAPPED | tested_rate=0.335 |
| BR-004 | Kisqali - Dx Adoption | `patient_journeys`+`treatment_events` → `brand_specific_kisqali_dx_adoption` | MAPPED | median_days=13 |
| BR-005 | Kisqali - Oncologist Reach | `hcp_profiles.specialty`+`triggers.hcp_id` → `brand_specific_kisqali_oncologist_reach` | MAPPED | reach=0.307 |
| CM-001 | Average Treatment Effect (ATE) | `ml_predictions.treatment_effect_estimate` → `causal_metrics_ate` (Shard 03) | MAPPED | ate=0.171 n=2208 |
| CM-002 | Conditional ATE (CATE) | `ml_predictions.heterogeneous_effect/segment_assignment` → `causal_metrics_cate` (Shard 03) | MAPPED | cate=0.285 |
| CM-003 | Causal Impact | `causal_paths.causal_effect_size/confidence_level` → `causal_metrics_causal_impact` — **inserted by Task 5c** | MAPPED | effect=0.304 n_paths=25 |
| CM-004 | Counterfactual Outcome | `ml_predictions.counterfactual_outcome` → `causal_metrics_counterfactual` — **stamped by `stamp_model_metrics`** | MAPPED | mean_cf=0.138 n=925 |
| CM-005 | Mediation Effect | `causal_paths.mediators_identified/indirect_effect` → `causal_metrics_mediation` — **inserted by Task 5c** | MAPPED | prop_mediated=0.235 n_paths=75 |

## BLOCKED-BY-finding agents (substrate present, code defect gates the read)

These are **NOT** substrate gaps — the synthetic rows are loaded and tagged. The
agent's own code defect (out of Shard 09 scope, tracked by the audit P0–P3 plan)
gates the read. Verified the substrate is present below.

| Agent | Finding | Substrate present (faithful COUNT) |
|---|---|---|
| observability_connector | F3 — `client=`→`supabase_client=` kwarg bug | `ml_observability_spans` synthetic slice loaded (60 rows) + 5319 real recent |
| experiment_monitor | F7 — sync `await get_supabase_client()` | `ml_experiments WHERE status='running'` = 36 synthetic (+621 real) |
| health_score | F1 — fabricates grade-A on missing backend | reads ml_experiments/ml_deployments — synthetic present |
| gap_analyzer | F2 — launders KeyError→HTTP200 | `business_metrics`/`triggers` substrate present |
| feature_analyzer | F8 — SHAP | `ml_predictions.shap_values` synthetic populated (3738) |

## §agent-smoke — 17 non-named agents

A smoke PASS = the agent reads real synthetic rows without crashing. Where a known
audit code-defect gates the read, the row IS present and tagged (`SUBSTRATE-OK`); the
defect is `BLOCKED-BY-Fn` (not a Shard-09 failure).

| Agent | Substrate table | Synthetic rows | Verdict |
|---|---|---|---|
| experiment_monitor | ml_experiments (running) + ab_experiment_assignments | 36 exps / 21600 assignments | SUBSTRATE-OK (read BLOCKED-BY-F7) |
| experiment_designer | ml_experiments | 36 | SUBSTRATE-OK |
| scope_definer | ml_experiments | 36 | SUBSTRATE-OK |
| model_selector | ml_training_runs / ml_model_registry | 72 / 72 | SUBSTRATE-OK |
| model_trainer | ml_training_runs | 72 | SUBSTRATE-OK |
| model_deployer | ml_deployments (active) | 36 | SUBSTRATE-OK (read BLOCKED-BY-F4) |
| observability_connector | ml_observability_spans | 60 synthetic (+5319 real) | SUBSTRATE-OK (read BLOCKED-BY-F3) |
| drift_monitor | ml_drift_history / feature_values | present | SUBSTRATE-OK |
| feedback_learner | learning_signals (is_training_example) | 30 | SUBSTRATE-OK (was F15-starved) |
| health_score | ml_experiments / ml_deployments | present | SUBSTRATE-OK (read BLOCKED-BY-F1) |
| feature_analyzer | ml_predictions.shap_values | 3738 | SUBSTRATE-OK (read BLOCKED-BY-F8) |
| gap_analyzer | business_metrics / triggers | present | SUBSTRATE-OK (read BLOCKED-BY-F2) |
| data_preparer | patient_journeys | 2500 | SUBSTRATE-OK |
| causal_impact | causal_paths | 25 | SUBSTRATE-OK |
| resource_optimizer | cohort frames (Shard 06) | present | SUBSTRATE-OK |
| prediction_synthesizer | ml_predictions | 3738 | SUBSTRATE-OK |
| heterogeneous_optimizer | ml_predictions.heterogeneous_effect | 3738 | SUBSTRATE-OK |

## Notes / honest caveats

- **WS2-TR-005 / WS2-TR-006 return 0**, not NULL. 0 is a legitimate value (the
  synthetic triggers carry no false-positive or overridden rows). Non-NULL = MAPPED.
- **WS1-MP-002/003/004/005/006/008** have no `kpi_query` registry entry; they are
  proven by a direct COUNT of the populated synthetic `ml_predictions` column (the
  same column the model-performance agent reads). The registry-backed twins
  (roc_auc=MP-001, shap=MP-007) confirm the RPC path returns the values too.
- **feature_store reseed** (`feature_values.is_synthetic`): migration 069 ADDS the
  `is_synthetic` column to `feature_groups`/`features`/`feature_values` (the loader
  already registered it but it was never on the DB). After that, the reseed still fails
  on TWO pre-existing Shard-02/04 bugs unrelated to Shard 09: (a) `feature_groups_name_key`
  unique violation — the FeatureStoreSeeder emits fixed names (`hcp_demographics`) that
  already exist while the loader UPSERTs on `id` (uuid); (b) `feature_values`
  `valid_event_timestamp` CHECK rejects the FeatureValueGenerator's future-dated
  anchored timestamps. **No KPI in the 44 depends on `feature_values`** (drift_monitor's
  WS1-MP-009 reads `ml_drift_history`, which loads fine), so this does not affect the
  44/44 coverage result. The is_synthetic gap is fixed; the seeder/CHECK bugs are
  deferred to their owning shards.
- The substrate completions added beyond the plan's Task list (model-quality metrics,
  `sequence_number`, change-tracking) are post-hoc column stamps onto **existing,
  nullable** columns — the same pattern as `data_lag_hours` (Task 5b). They were
  required because the plan's coverage table assumed upstream shards populate those
  columns, but they were empty on synthetic rows. Migration 069 adds `is_synthetic` to
  the 8 Shard-09 tables that Shard 01's migration 063 did not cover.
