# Cohort DAG review: optum_mart — treatment_dupixent → persistent_at_180d_g28

- treatment `treatment_dupixent` = remibrutinib vs competitor biologic (CSU escalation therapy; rehearsed as Dupixent vs Xolair)
- outcome `persistent_at_180d_g28` = persistent_at_180d_g28
- author LM: `fake`; prompt hash `1971d0d38fa1c895360eb48b3f5ab6bb53d8dff95fb9fea9e57eaa3b5c4bac7c`; guide hash `04009c46e34c2c457755f8ca322b25510ee8646c5caf32093e14b8ec7223889a`
- tree: commit edb96fbd24ddb63ca6b2e3bb93fa619c86df9af1 (dirty src/scripts/tests: False)
- provenance: `machine` on every fragment (audit-only until this review is approved)
- features authored: 64; latents: none
- is a DAG: True; admissible observed adjustment set: True
- minimal adjustment set: ['age_at_index', 'cci_cerebrovascular', 'cci_chf', 'cci_chronic_pulmonary', 'cci_dementia', 'cci_diabetes_complication', 'cci_diabetes_no_complication', 'cci_hiv', 'cci_malignancy', 'cci_metastatic_cancer', 'cci_mi', 'cci_mild_liver', 'cci_paraplegia', 'cci_peptic_ulcer', 'cci_pvd', 'cci_renal', 'cci_rheumatic', 'cci_severe_liver', 'charlson_risk_band', 'charlson_score', 'comorbidity_diag_claim_count', 'comorbidity_diag_distinct_count', 'elixhauser_risk_band', 'elixhauser_van_walraven_score', 'elx_aids_hiv', 'elx_alcohol_abuse', 'elx_blood_loss_anemia', 'elx_cardiac_arrhythmia', 'elx_chf', 'elx_chronic_pulmonary', 'elx_coagulopathy', 'elx_deficiency_anemia', 'elx_depression', 'elx_diabetes_complicated', 'elx_diabetes_uncomplicated', 'elx_drug_abuse', 'elx_fluid_electrolyte', 'elx_hypertension_complicated', 'elx_hypertension_uncomplicated', 'elx_hypothyroidism', 'elx_liver_disease', 'elx_lymphoma', 'elx_metastatic_cancer', 'elx_obesity', 'elx_other_neurological', 'elx_paralysis', 'elx_peptic_ulcer', 'elx_psychoses', 'elx_pulmonary_circulation', 'elx_pvd', 'elx_renal_failure', 'elx_rheumatoid_collagen', 'elx_solid_tumor_no_metastasis', 'elx_valvular_disease', 'elx_weight_loss', 'enrollment_duration_days', 'gdr_cd', 'geographic_region', 'health_exchange_flag', 'high_comorbidity_burden_flag', 'lis_dual_flag', 'payer_bus', 'payer_category', 'payer_product']
- full admissible set: ['age_at_index', 'cci_cerebrovascular', 'cci_chf', 'cci_chronic_pulmonary', 'cci_dementia', 'cci_diabetes_complication', 'cci_diabetes_no_complication', 'cci_hiv', 'cci_malignancy', 'cci_metastatic_cancer', 'cci_mi', 'cci_mild_liver', 'cci_paraplegia', 'cci_peptic_ulcer', 'cci_pvd', 'cci_renal', 'cci_rheumatic', 'cci_severe_liver', 'charlson_risk_band', 'charlson_score', 'comorbidity_diag_claim_count', 'comorbidity_diag_distinct_count', 'elixhauser_risk_band', 'elixhauser_van_walraven_score', 'elx_aids_hiv', 'elx_alcohol_abuse', 'elx_blood_loss_anemia', 'elx_cardiac_arrhythmia', 'elx_chf', 'elx_chronic_pulmonary', 'elx_coagulopathy', 'elx_deficiency_anemia', 'elx_depression', 'elx_diabetes_complicated', 'elx_diabetes_uncomplicated', 'elx_drug_abuse', 'elx_fluid_electrolyte', 'elx_hypertension_complicated', 'elx_hypertension_uncomplicated', 'elx_hypothyroidism', 'elx_liver_disease', 'elx_lymphoma', 'elx_metastatic_cancer', 'elx_obesity', 'elx_other_neurological', 'elx_paralysis', 'elx_peptic_ulcer', 'elx_psychoses', 'elx_pulmonary_circulation', 'elx_pvd', 'elx_renal_failure', 'elx_rheumatoid_collagen', 'elx_solid_tumor_no_metastasis', 'elx_valvular_disease', 'elx_weight_loss', 'enrollment_duration_days', 'gdr_cd', 'geographic_region', 'health_exchange_flag', 'high_comorbidity_burden_flag', 'lis_dual_flag', 'payer_bus', 'payer_category', 'payer_product']

## Reviewer checklist

- [ ] escalation_decision_point: Remibrutinib enters at the same decision point as the biologics (second line after H1-antihistamine failure), so the confounders of WHICH escalation therapy a patient receives are the same set; Lane A's rehearsal therefore uses this DAG with treatment_dupixent (Dupixent vs Xolair) on the same confounders.

## Features

| feature | fragment role | cohort role | drift | ambiguous | review | panel role | leak | grades |
|---|---|---|---|---|---|---|---|---|
| age_at_index | confounder | confounder |  |  |  | None |  | age_at_index->T:unsupported, age_at_index->Y:unsupported, T->Y:estimand |
| cci_cerebrovascular | confounder | confounder |  |  |  | None |  | cci_cerebrovascular->T:unsupported, cci_cerebrovascular->Y:unsupported, T->Y:estimand |
| cci_chf | confounder | confounder |  |  |  | None |  | cci_chf->T:unsupported, cci_chf->Y:unsupported, T->Y:estimand |
| cci_chronic_pulmonary | confounder | confounder |  |  |  | None |  | cci_chronic_pulmonary->T:unsupported, cci_chronic_pulmonary->Y:unsupported, T->Y:estimand |
| cci_dementia | confounder | confounder |  |  |  | None |  | cci_dementia->T:unsupported, cci_dementia->Y:unsupported, T->Y:estimand |
| cci_diabetes_complication | confounder | confounder |  |  |  | None |  | cci_diabetes_complication->T:unsupported, cci_diabetes_complication->Y:unsupported, T->Y:estimand |
| cci_diabetes_no_complication | confounder | confounder |  |  |  | None |  | cci_diabetes_no_complication->T:unsupported, cci_diabetes_no_complication->Y:unsupported, T->Y:estimand |
| cci_hiv | confounder | confounder |  |  |  | None |  | cci_hiv->T:unsupported, cci_hiv->Y:unsupported, T->Y:estimand |
| cci_malignancy | confounder | confounder |  |  |  | None |  | cci_malignancy->T:unsupported, cci_malignancy->Y:unsupported, T->Y:estimand |
| cci_metastatic_cancer | confounder | confounder |  |  |  | None |  | cci_metastatic_cancer->T:unsupported, cci_metastatic_cancer->Y:unsupported, T->Y:estimand |
| cci_mi | confounder | confounder |  |  |  | None |  | cci_mi->T:unsupported, cci_mi->Y:unsupported, T->Y:estimand |
| cci_mild_liver | confounder | confounder |  |  |  | None |  | cci_mild_liver->T:unsupported, cci_mild_liver->Y:unsupported, T->Y:estimand |
| cci_paraplegia | confounder | confounder |  |  |  | None |  | cci_paraplegia->T:unsupported, cci_paraplegia->Y:unsupported, T->Y:estimand |
| cci_peptic_ulcer | confounder | confounder |  |  |  | None |  | cci_peptic_ulcer->T:unsupported, cci_peptic_ulcer->Y:unsupported, T->Y:estimand |
| cci_pvd | confounder | confounder |  |  |  | None |  | cci_pvd->T:unsupported, cci_pvd->Y:unsupported, T->Y:estimand |
| cci_renal | confounder | confounder |  |  |  | None |  | cci_renal->T:unsupported, cci_renal->Y:unsupported, T->Y:estimand |
| cci_rheumatic | confounder | confounder |  |  |  | None |  | cci_rheumatic->T:unsupported, cci_rheumatic->Y:unsupported, T->Y:estimand |
| cci_severe_liver | confounder | confounder |  |  |  | None |  | cci_severe_liver->T:unsupported, cci_severe_liver->Y:unsupported, T->Y:estimand |
| charlson_risk_band | confounder | confounder |  |  |  | None |  | charlson_risk_band->T:unsupported, charlson_risk_band->Y:unsupported, T->Y:estimand |
| charlson_score | confounder | confounder |  |  |  | None |  | charlson_score->T:unsupported, charlson_score->Y:unsupported, T->Y:estimand |
| comorbidity_diag_claim_count | confounder | confounder |  |  |  | None |  | comorbidity_diag_claim_count->T:unsupported, comorbidity_diag_claim_count->Y:unsupported, T->Y:estimand |
| comorbidity_diag_distinct_count | confounder | confounder |  |  |  | None |  | comorbidity_diag_distinct_count->T:unsupported, comorbidity_diag_distinct_count->Y:unsupported, T->Y:estimand |
| elixhauser_risk_band | confounder | confounder |  |  |  | None |  | elixhauser_risk_band->T:unsupported, elixhauser_risk_band->Y:unsupported, T->Y:estimand |
| elixhauser_van_walraven_score | confounder | confounder |  |  |  | None |  | elixhauser_van_walraven_score->T:unsupported, elixhauser_van_walraven_score->Y:unsupported, T->Y:estimand |
| elx_aids_hiv | confounder | confounder |  |  |  | None |  | elx_aids_hiv->T:unsupported, elx_aids_hiv->Y:unsupported, T->Y:estimand |
| elx_alcohol_abuse | confounder | confounder |  |  |  | None |  | elx_alcohol_abuse->T:unsupported, elx_alcohol_abuse->Y:unsupported, T->Y:estimand |
| elx_blood_loss_anemia | confounder | confounder |  |  |  | None |  | elx_blood_loss_anemia->T:unsupported, elx_blood_loss_anemia->Y:unsupported, T->Y:estimand |
| elx_cardiac_arrhythmia | confounder | confounder |  |  |  | None |  | elx_cardiac_arrhythmia->T:unsupported, elx_cardiac_arrhythmia->Y:unsupported, T->Y:estimand |
| elx_chf | confounder | confounder |  |  |  | None |  | elx_chf->T:unsupported, elx_chf->Y:unsupported, T->Y:estimand |
| elx_chronic_pulmonary | confounder | confounder |  |  |  | None |  | elx_chronic_pulmonary->T:unsupported, elx_chronic_pulmonary->Y:unsupported, T->Y:estimand |
| elx_coagulopathy | confounder | confounder |  |  |  | None |  | elx_coagulopathy->T:unsupported, elx_coagulopathy->Y:unsupported, T->Y:estimand |
| elx_deficiency_anemia | confounder | confounder |  |  |  | None |  | elx_deficiency_anemia->T:unsupported, elx_deficiency_anemia->Y:unsupported, T->Y:estimand |
| elx_depression | confounder | confounder |  |  |  | None |  | elx_depression->T:unsupported, elx_depression->Y:unsupported, T->Y:estimand |
| elx_diabetes_complicated | confounder | confounder |  |  |  | None |  | elx_diabetes_complicated->T:unsupported, elx_diabetes_complicated->Y:unsupported, T->Y:estimand |
| elx_diabetes_uncomplicated | confounder | confounder |  |  |  | None |  | elx_diabetes_uncomplicated->T:unsupported, elx_diabetes_uncomplicated->Y:unsupported, T->Y:estimand |
| elx_drug_abuse | confounder | confounder |  |  |  | None |  | elx_drug_abuse->T:unsupported, elx_drug_abuse->Y:unsupported, T->Y:estimand |
| elx_fluid_electrolyte | confounder | confounder |  |  |  | None |  | elx_fluid_electrolyte->T:unsupported, elx_fluid_electrolyte->Y:unsupported, T->Y:estimand |
| elx_hypertension_complicated | confounder | confounder |  |  |  | None |  | elx_hypertension_complicated->T:unsupported, elx_hypertension_complicated->Y:unsupported, T->Y:estimand |
| elx_hypertension_uncomplicated | confounder | confounder |  |  |  | None |  | elx_hypertension_uncomplicated->T:unsupported, elx_hypertension_uncomplicated->Y:unsupported, T->Y:estimand |
| elx_hypothyroidism | confounder | confounder |  |  |  | None |  | elx_hypothyroidism->T:unsupported, elx_hypothyroidism->Y:unsupported, T->Y:estimand |
| elx_liver_disease | confounder | confounder |  |  |  | None |  | elx_liver_disease->T:unsupported, elx_liver_disease->Y:unsupported, T->Y:estimand |
| elx_lymphoma | confounder | confounder |  |  |  | None |  | elx_lymphoma->T:unsupported, elx_lymphoma->Y:unsupported, T->Y:estimand |
| elx_metastatic_cancer | confounder | confounder |  |  |  | None |  | elx_metastatic_cancer->T:unsupported, elx_metastatic_cancer->Y:unsupported, T->Y:estimand |
| elx_obesity | confounder | confounder |  |  |  | None |  | elx_obesity->T:unsupported, elx_obesity->Y:unsupported, T->Y:estimand |
| elx_other_neurological | confounder | confounder |  |  |  | None |  | elx_other_neurological->T:unsupported, elx_other_neurological->Y:unsupported, T->Y:estimand |
| elx_paralysis | confounder | confounder |  |  |  | None |  | elx_paralysis->T:unsupported, elx_paralysis->Y:unsupported, T->Y:estimand |
| elx_peptic_ulcer | confounder | confounder |  |  |  | None |  | elx_peptic_ulcer->T:unsupported, elx_peptic_ulcer->Y:unsupported, T->Y:estimand |
| elx_psychoses | confounder | confounder |  |  |  | None |  | elx_psychoses->T:unsupported, elx_psychoses->Y:unsupported, T->Y:estimand |
| elx_pulmonary_circulation | confounder | confounder |  |  |  | None |  | elx_pulmonary_circulation->T:unsupported, elx_pulmonary_circulation->Y:unsupported, T->Y:estimand |
| elx_pvd | confounder | confounder |  |  |  | None |  | elx_pvd->T:unsupported, elx_pvd->Y:unsupported, T->Y:estimand |
| elx_renal_failure | confounder | confounder |  |  |  | None |  | elx_renal_failure->T:unsupported, elx_renal_failure->Y:unsupported, T->Y:estimand |
| elx_rheumatoid_collagen | confounder | confounder |  |  |  | None |  | elx_rheumatoid_collagen->T:unsupported, elx_rheumatoid_collagen->Y:unsupported, T->Y:estimand |
| elx_solid_tumor_no_metastasis | confounder | confounder |  |  |  | None |  | elx_solid_tumor_no_metastasis->T:unsupported, elx_solid_tumor_no_metastasis->Y:unsupported, T->Y:estimand |
| elx_valvular_disease | confounder | confounder |  |  |  | None |  | elx_valvular_disease->T:unsupported, elx_valvular_disease->Y:unsupported, T->Y:estimand |
| elx_weight_loss | confounder | confounder |  |  |  | None |  | elx_weight_loss->T:unsupported, elx_weight_loss->Y:unsupported, T->Y:estimand |
| enrollment_duration_days | confounder | confounder |  |  |  | None |  | enrollment_duration_days->T:unsupported, enrollment_duration_days->Y:unsupported, T->Y:estimand |
| gdr_cd | confounder | confounder |  |  |  | None |  | gdr_cd->T:unsupported, gdr_cd->Y:unsupported, T->Y:estimand |
| geographic_region | confounder | confounder |  |  |  | None |  | geographic_region->T:unsupported, geographic_region->Y:unsupported, T->Y:estimand |
| health_exchange_flag | confounder | confounder |  |  |  | None |  | health_exchange_flag->T:unsupported, health_exchange_flag->Y:unsupported, T->Y:estimand |
| high_comorbidity_burden_flag | confounder | confounder |  |  |  | None |  | high_comorbidity_burden_flag->T:unsupported, high_comorbidity_burden_flag->Y:unsupported, T->Y:estimand |
| lis_dual_flag | confounder | confounder |  |  |  | None |  | lis_dual_flag->T:unsupported, lis_dual_flag->Y:unsupported, T->Y:estimand |
| payer_bus | confounder | confounder |  |  |  | None |  | payer_bus->T:unsupported, payer_bus->Y:unsupported, T->Y:estimand |
| payer_category | confounder | confounder |  |  |  | None |  | payer_category->T:unsupported, payer_category->Y:unsupported, T->Y:estimand |
| payer_product | confounder | confounder |  |  |  | None |  | payer_product->T:unsupported, payer_product->Y:unsupported, T->Y:estimand |

## Review items (0)

- none

## Edge rationale (non-estimand edges)

- age_at_index → T (unsupported): — — no citation
- age_at_index → Y (unsupported): — — no citation
- gdr_cd → T (unsupported): — — no citation
- gdr_cd → Y (unsupported): — — no citation
- payer_category → T (unsupported): — — no citation
- payer_category → Y (unsupported): — — no citation
- payer_product → T (unsupported): — — no citation
- payer_product → Y (unsupported): — — no citation
- payer_bus → T (unsupported): — — no citation
- payer_bus → Y (unsupported): — — no citation
- health_exchange_flag → T (unsupported): — — no citation
- health_exchange_flag → Y (unsupported): — — no citation
- lis_dual_flag → T (unsupported): — — no citation
- lis_dual_flag → Y (unsupported): — — no citation
- geographic_region → T (unsupported): — — no citation
- geographic_region → Y (unsupported): — — no citation
- enrollment_duration_days → T (unsupported): — — no citation
- enrollment_duration_days → Y (unsupported): — — no citation
- charlson_score → T (unsupported): — — no citation
- charlson_score → Y (unsupported): — — no citation
- charlson_risk_band → T (unsupported): — — no citation
- charlson_risk_band → Y (unsupported): — — no citation
- elixhauser_van_walraven_score → T (unsupported): — — no citation
- elixhauser_van_walraven_score → Y (unsupported): — — no citation
- elixhauser_risk_band → T (unsupported): — — no citation
- elixhauser_risk_band → Y (unsupported): — — no citation
- comorbidity_diag_distinct_count → T (unsupported): — — no citation
- comorbidity_diag_distinct_count → Y (unsupported): — — no citation
- comorbidity_diag_claim_count → T (unsupported): — — no citation
- comorbidity_diag_claim_count → Y (unsupported): — — no citation
- high_comorbidity_burden_flag → T (unsupported): — — no citation
- high_comorbidity_burden_flag → Y (unsupported): — — no citation
- cci_mi → T (unsupported): — — no citation
- cci_mi → Y (unsupported): — — no citation
- cci_chf → T (unsupported): — — no citation
- cci_chf → Y (unsupported): — — no citation
- cci_pvd → T (unsupported): — — no citation
- cci_pvd → Y (unsupported): — — no citation
- cci_cerebrovascular → T (unsupported): — — no citation
- cci_cerebrovascular → Y (unsupported): — — no citation
- cci_dementia → T (unsupported): — — no citation
- cci_dementia → Y (unsupported): — — no citation
- cci_chronic_pulmonary → T (unsupported): — — no citation
- cci_chronic_pulmonary → Y (unsupported): — — no citation
- cci_rheumatic → T (unsupported): — — no citation
- cci_rheumatic → Y (unsupported): — — no citation
- cci_peptic_ulcer → T (unsupported): — — no citation
- cci_peptic_ulcer → Y (unsupported): — — no citation
- cci_mild_liver → T (unsupported): — — no citation
- cci_mild_liver → Y (unsupported): — — no citation
- cci_diabetes_no_complication → T (unsupported): — — no citation
- cci_diabetes_no_complication → Y (unsupported): — — no citation
- cci_diabetes_complication → T (unsupported): — — no citation
- cci_diabetes_complication → Y (unsupported): — — no citation
- cci_paraplegia → T (unsupported): — — no citation
- cci_paraplegia → Y (unsupported): — — no citation
- cci_renal → T (unsupported): — — no citation
- cci_renal → Y (unsupported): — — no citation
- cci_malignancy → T (unsupported): — — no citation
- cci_malignancy → Y (unsupported): — — no citation
- cci_severe_liver → T (unsupported): — — no citation
- cci_severe_liver → Y (unsupported): — — no citation
- cci_metastatic_cancer → T (unsupported): — — no citation
- cci_metastatic_cancer → Y (unsupported): — — no citation
- cci_hiv → T (unsupported): — — no citation
- cci_hiv → Y (unsupported): — — no citation
- elx_chf → T (unsupported): — — no citation
- elx_chf → Y (unsupported): — — no citation
- elx_cardiac_arrhythmia → T (unsupported): — — no citation
- elx_cardiac_arrhythmia → Y (unsupported): — — no citation
- elx_valvular_disease → T (unsupported): — — no citation
- elx_valvular_disease → Y (unsupported): — — no citation
- elx_pulmonary_circulation → T (unsupported): — — no citation
- elx_pulmonary_circulation → Y (unsupported): — — no citation
- elx_pvd → T (unsupported): — — no citation
- elx_pvd → Y (unsupported): — — no citation
- elx_hypertension_uncomplicated → T (unsupported): — — no citation
- elx_hypertension_uncomplicated → Y (unsupported): — — no citation
- elx_hypertension_complicated → T (unsupported): — — no citation
- elx_hypertension_complicated → Y (unsupported): — — no citation
- elx_paralysis → T (unsupported): — — no citation
- elx_paralysis → Y (unsupported): — — no citation
- elx_other_neurological → T (unsupported): — — no citation
- elx_other_neurological → Y (unsupported): — — no citation
- elx_chronic_pulmonary → T (unsupported): — — no citation
- elx_chronic_pulmonary → Y (unsupported): — — no citation
- elx_diabetes_uncomplicated → T (unsupported): — — no citation
- elx_diabetes_uncomplicated → Y (unsupported): — — no citation
- elx_diabetes_complicated → T (unsupported): — — no citation
- elx_diabetes_complicated → Y (unsupported): — — no citation
- elx_hypothyroidism → T (unsupported): — — no citation
- elx_hypothyroidism → Y (unsupported): — — no citation
- elx_renal_failure → T (unsupported): — — no citation
- elx_renal_failure → Y (unsupported): — — no citation
- elx_liver_disease → T (unsupported): — — no citation
- elx_liver_disease → Y (unsupported): — — no citation
- elx_peptic_ulcer → T (unsupported): — — no citation
- elx_peptic_ulcer → Y (unsupported): — — no citation
- elx_aids_hiv → T (unsupported): — — no citation
- elx_aids_hiv → Y (unsupported): — — no citation
- elx_lymphoma → T (unsupported): — — no citation
- elx_lymphoma → Y (unsupported): — — no citation
- elx_metastatic_cancer → T (unsupported): — — no citation
- elx_metastatic_cancer → Y (unsupported): — — no citation
- elx_solid_tumor_no_metastasis → T (unsupported): — — no citation
- elx_solid_tumor_no_metastasis → Y (unsupported): — — no citation
- elx_rheumatoid_collagen → T (unsupported): — — no citation
- elx_rheumatoid_collagen → Y (unsupported): — — no citation
- elx_coagulopathy → T (unsupported): — — no citation
- elx_coagulopathy → Y (unsupported): — — no citation
- elx_obesity → T (unsupported): — — no citation
- elx_obesity → Y (unsupported): — — no citation
- elx_weight_loss → T (unsupported): — — no citation
- elx_weight_loss → Y (unsupported): — — no citation
- elx_fluid_electrolyte → T (unsupported): — — no citation
- elx_fluid_electrolyte → Y (unsupported): — — no citation
- elx_blood_loss_anemia → T (unsupported): — — no citation
- elx_blood_loss_anemia → Y (unsupported): — — no citation
- elx_deficiency_anemia → T (unsupported): — — no citation
- elx_deficiency_anemia → Y (unsupported): — — no citation
- elx_alcohol_abuse → T (unsupported): — — no citation
- elx_alcohol_abuse → Y (unsupported): — — no citation
- elx_drug_abuse → T (unsupported): — — no citation
- elx_drug_abuse → Y (unsupported): — — no citation
- elx_psychoses → T (unsupported): — — no citation
- elx_psychoses → Y (unsupported): — — no citation
- elx_depression → T (unsupported): — — no citation
- elx_depression → Y (unsupported): — — no citation
