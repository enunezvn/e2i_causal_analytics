# Feature-role panel — optum_mart · treatment_dupixent → persistent_at_180d_g28

built_at: 2026-09-22T14:11:06.671270+00:00  ·  n_rows: 15209  ·  covariates: 64
activation_profile: `{"adaptive_layer4_enabled": true, "adaptive_structural_decider_enabled": true, "kg_mode": "shadow"}`
Layer 4 LM: **fake**. Fake LM (DummyLM): zero paid calls; measures which features fire.

## Which layers fired

| Layer | Activity |
|---|---|
| Layer 1 contracts | consulted 64, contracted 64, declared-safe 64, post-index 0 |
| Layer 2 KG (shadow) | cache bound True, with cached edges 24, signalled 2 `{"leak_drug_treats_disease": 2, "no_signal": 62}` |
| Layer 3 adversarial | scored 57, pre-joint severities `{"high": 9, "info": 38, "moderate": 10}`, FDR-confident 0, declared-safe immunity applied 9, fdr `{"active": true, "confident_features": [], "enabled": true, "n_confident": 0, "n_permutations": 569, "q": 0.1, "reason": "active"}` |
| Layer 4 LLM | enabled True, classifier loaded True, fired 19, roles `{"confounder": 19}` |
| Ensemble | decided_by `{"None": 7, "abstain": 19, "adversarial": 36, "kg": 2}`, abstain rate 0.406, leak verdicts 0 `{}` |

promotion_eligibility: `{"cross_source_disagreement_rate": 0.0, "kg_decided_count": 2, "n_features": 57, "n_patients": 15209, "non_abstain_pct": 0.6666666666666666, "passes": false, "patient_count_pass": true}` (KG promotion is an owner decision, spec §7)

## Per feature

| Feature | L1 (temporal status) | L2 signal | L3 z / pre-joint sev | L4 role | decided_by | final_role | conf | excluded (why) |
|---|---|---|---|---|---|---|---|---|
| `age_at_index` | pre_index (pre_index) | no_signal | 10.02 / high | confounder | abstain | — | 0.00 |  |
| `gdr_cd` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `payer_category` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `payer_product` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `payer_bus` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `health_exchange_flag` | pre_index (pre_index) | no_signal | -0.31 / info | — | adversarial | — | — |  |
| `lis_dual_flag` | pre_index (pre_index) | no_signal | 14.37 / high | confounder | abstain | — | 0.00 |  |
| `geographic_region` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `enrollment_duration_days` | pre_index (pre_index) | no_signal | 3.96 / info | — | adversarial | — | — |  |
| `charlson_score` | pre_index (pre_index) | no_signal | 7.55 / high | confounder | abstain | — | 0.00 |  |
| `charlson_risk_band` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `elixhauser_van_walraven_score` | pre_index (pre_index) | no_signal | 5.56 / moderate | confounder | abstain | — | 0.00 |  |
| `elixhauser_risk_band` | pre_index (pre_index) | no_signal | — / — | — | — | — | — |  |
| `comorbidity_diag_distinct_count` | pre_index (pre_index) | no_signal | 11.19 / high | confounder | abstain | — | 0.00 |  |
| `comorbidity_diag_claim_count` | pre_index (pre_index) | no_signal | 12.37 / high | confounder | abstain | — | 0.00 |  |
| `high_comorbidity_burden_flag` | pre_index (pre_index) | no_signal | 7.84 / high | confounder | abstain | — | 0.00 |  |
| `cci_mi` | pre_index (pre_index) | no_signal | 0.77 / info | — | adversarial | — | — |  |
| `cci_chf` | pre_index (pre_index) | no_signal | 8.21 / high | confounder | abstain | — | 0.00 |  |
| `cci_pvd` | pre_index (pre_index) | no_signal | 6.92 / moderate | confounder | abstain | — | 0.00 |  |
| `cci_cerebrovascular` | pre_index (pre_index) | no_signal | 5.14 / moderate | confounder | abstain | — | 0.00 |  |
| `cci_dementia` | pre_index (pre_index) | no_signal | 2.34 / info | — | adversarial | — | — |  |
| `cci_chronic_pulmonary` | pre_index (pre_index) | leak_drug_treats_disease | 2.20 / info | — | kg | — | 0.70 |  |
| `cci_rheumatic` | pre_index (pre_index) | no_signal | 1.29 / info | — | adversarial | — | — |  |
| `cci_peptic_ulcer` | pre_index (pre_index) | no_signal | -0.56 / info | — | adversarial | — | — |  |
| `cci_mild_liver` | pre_index (pre_index) | no_signal | 0.24 / info | — | adversarial | — | — |  |
| `cci_diabetes_no_complication` | pre_index (pre_index) | no_signal | 0.17 / info | — | adversarial | — | — |  |
| `cci_diabetes_complication` | pre_index (pre_index) | no_signal | 6.70 / moderate | confounder | abstain | — | 0.00 |  |
| `cci_paraplegia` | pre_index (pre_index) | no_signal | 0.29 / info | — | adversarial | — | — |  |
| `cci_renal` | pre_index (pre_index) | no_signal | 4.22 / info | — | adversarial | — | — |  |
| `cci_malignancy` | pre_index (pre_index) | no_signal | 1.16 / info | — | adversarial | — | — |  |
| `cci_severe_liver` | pre_index (pre_index) | no_signal | 0.73 / info | — | adversarial | — | — |  |
| `cci_metastatic_cancer` | pre_index (pre_index) | no_signal | 0.32 / info | — | adversarial | — | — |  |
| `cci_hiv` | pre_index (pre_index) | no_signal | 0.92 / info | — | adversarial | — | — |  |
| `elx_chf` | pre_index (pre_index) | no_signal | 8.21 / high | confounder | abstain | — | 0.00 |  |
| `elx_cardiac_arrhythmia` | pre_index (pre_index) | no_signal | 7.70 / high | confounder | abstain | — | 0.00 |  |
| `elx_valvular_disease` | pre_index (pre_index) | no_signal | 6.14 / moderate | confounder | abstain | — | 0.00 |  |
| `elx_pulmonary_circulation` | pre_index (pre_index) | no_signal | 4.45 / info | — | adversarial | — | — |  |
| `elx_pvd` | pre_index (pre_index) | no_signal | 6.92 / moderate | confounder | abstain | — | 0.00 |  |
| `elx_hypertension_uncomplicated` | pre_index (pre_index) | no_signal | 5.65 / moderate | confounder | abstain | — | 0.00 |  |
| `elx_hypertension_complicated` | pre_index (pre_index) | no_signal | 4.21 / info | — | adversarial | — | — |  |
| `elx_paralysis` | pre_index (pre_index) | no_signal | 0.29 / info | — | adversarial | — | — |  |
| `elx_other_neurological` | pre_index (pre_index) | no_signal | 2.33 / info | — | adversarial | — | — |  |
| `elx_chronic_pulmonary` | pre_index (pre_index) | leak_drug_treats_disease | 2.20 / info | — | kg | — | 0.70 |  |
| `elx_diabetes_uncomplicated` | pre_index (pre_index) | no_signal | 0.17 / info | — | adversarial | — | — |  |
| `elx_diabetes_complicated` | pre_index (pre_index) | no_signal | 6.70 / moderate | confounder | abstain | — | 0.00 |  |
| `elx_hypothyroidism` | pre_index (pre_index) | no_signal | 0.47 / info | — | adversarial | — | — |  |
| `elx_renal_failure` | pre_index (pre_index) | no_signal | 4.19 / info | — | adversarial | — | — |  |
| `elx_liver_disease` | pre_index (pre_index) | no_signal | -0.24 / info | — | adversarial | — | — |  |
| `elx_peptic_ulcer` | pre_index (pre_index) | no_signal | -0.56 / info | — | adversarial | — | — |  |
| `elx_aids_hiv` | pre_index (pre_index) | no_signal | 0.92 / info | — | adversarial | — | — |  |
| `elx_lymphoma` | pre_index (pre_index) | no_signal | -0.47 / info | — | adversarial | — | — |  |
| `elx_metastatic_cancer` | pre_index (pre_index) | no_signal | 0.32 / info | — | adversarial | — | — |  |
| `elx_solid_tumor_no_metastasis` | pre_index (pre_index) | no_signal | 1.10 / info | — | adversarial | — | — |  |
| `elx_rheumatoid_collagen` | pre_index (pre_index) | no_signal | 1.29 / info | — | adversarial | — | — |  |
| `elx_coagulopathy` | pre_index (pre_index) | no_signal | -0.80 / info | — | adversarial | — | — |  |
| `elx_obesity` | pre_index (pre_index) | no_signal | 0.52 / info | — | adversarial | — | — |  |
| `elx_weight_loss` | pre_index (pre_index) | no_signal | 2.35 / info | — | adversarial | — | — |  |
| `elx_fluid_electrolyte` | pre_index (pre_index) | no_signal | 5.92 / moderate | confounder | abstain | — | 0.00 |  |
| `elx_blood_loss_anemia` | pre_index (pre_index) | no_signal | 1.70 / info | — | adversarial | — | — |  |
| `elx_deficiency_anemia` | pre_index (pre_index) | no_signal | 3.60 / info | — | adversarial | — | — |  |
| `elx_alcohol_abuse` | pre_index (pre_index) | no_signal | 3.98 / info | — | adversarial | — | — |  |
| `elx_drug_abuse` | pre_index (pre_index) | no_signal | 1.24 / info | — | adversarial | — | — |  |
| `elx_psychoses` | pre_index (pre_index) | no_signal | 3.51 / info | — | adversarial | — | — |  |
| `elx_depression` | pre_index (pre_index) | no_signal | 5.00 / moderate | confounder | abstain | — | 0.00 |  |

Layer 4 fired on 19 feature(s): `age_at_index`, `lis_dual_flag`, `charlson_score`, `elixhauser_van_walraven_score`, `comorbidity_diag_distinct_count`, `comorbidity_diag_claim_count`, `high_comorbidity_burden_flag`, `cci_chf`, `cci_pvd`, `cci_cerebrovascular`, `cci_diabetes_complication`, `elx_chf`, `elx_cardiac_arrhythmia`, `elx_valvular_disease`, `elx_pvd`, `elx_hypertension_uncomplicated`, `elx_diabetes_complicated`, `elx_fluid_electrolyte`, `elx_depression`

Excluded from the adjustment set — proven post-index leakage (0): none

Excluded per spec 3(b), pending temporal review (0; Layer-3 high on a column with no manifest contract — predictiveness of Y, NOT proven timing): none
