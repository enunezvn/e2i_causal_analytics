# Cohort DAG review: optum — biologic_initiation → initiated_biologic_180d

- treatment `biologic_initiation` = biologic_initiation
- outcome `initiated_biologic_180d` = initiated_biologic_180d
- author LM: `fake`; prompt hash `1971d0d38fa1c895360eb48b3f5ab6bb53d8dff95fb9fea9e57eaa3b5c4bac7c`; guide hash `04009c46e34c2c457755f8ca322b25510ee8646c5caf32093e14b8ec7223889a`
- tree: commit 25b04ac334c1701c594765b109ff3743fc252e3e (dirty src/scripts/tests: False)
- provenance: `machine` on every fragment (audit-only until this review is approved)
- features authored: 110; latents: none
- is a DAG: True; admissible observed adjustment set: True
- minimal adjustment set: ['age_at_index', 'age_group', 'allergic_rhinitis_claim_count', 'ana_abnormal_flag', 'ana_result_last', 'ana_tested', 'angioedema_claim_count', 'anxiety_claim_count', 'asthma_claim_count', 'atopic_dermatitis_claim_count', 'atopy_score', 'cbc_abnormal_flag', 'cbc_result_last', 'cbc_tested', 'charlson_score', 'comorbidity_load_total', 'crp_abnormal_flag', 'crp_result_last', 'crp_tested', 'csu_chronicity', 'csu_dx_intensity', 'depression_claim_count', 'dx_angioedema_count', 'dx_l50_1_count', 'dx_l50_8_count', 'dx_l50_9_count', 'dx_total_csu', 'ed_visits_total', 'ed_visits_urticaria_angio', 'elixhauser_score', 'eosinophil_abnormal_flag', 'eosinophil_result_last', 'eosinophil_tested', 'free_t4_abnormal_flag', 'free_t4_result_last', 'free_t4_tested', 'gender', 'geographic_region', 'h1_1g_days_since_last_fill', 'h1_1g_days_supply_total', 'h1_1g_ever_filled', 'h1_1g_fill_count', 'h1_2g_days_since_last_fill', 'h1_2g_days_supply_total', 'h1_2g_ever_filled', 'h1_2g_fill_count', 'h2_days_since_last_fill', 'h2_days_supply_total', 'h2_ever_filled', 'h2_fill_count', 'has_allergic_rhinitis', 'has_angioedema', 'has_anxiety', 'has_asthma', 'has_atopic_dermatitis', 'has_depression', 'has_nsaid_hypersensitivity', 'has_thyroid_autoimmune', 'hospitalizations_total', 'ige_total_abnormal_flag', 'ige_total_result_last', 'ige_total_tested', 'immunosupp_days_since_last_fill', 'immunosupp_days_supply_total', 'immunosupp_ever_filled', 'immunosupp_fill_count', 'index_date', 'insurance_product', 'lab_workup_completeness', 'lookback_start_date', 'ltra_days_since_last_fill', 'ltra_days_supply_total', 'ltra_ever_filled', 'ltra_fill_count', 'mental_health_flag', 'months_since_first_dx', 'nsaid_hypersensitivity_claim_count', 'office_visits_allergist', 'office_visits_dermatology', 'office_visits_pcp', 'office_visits_total', 'payer_category', 'plan_type', 'polypharmacy_breadth', 'primary_diagnosis_code', 'primary_specialist_type', 'saw_allergist_flag', 'saw_dermatologist_flag', 'specialist_concentration', 'specialist_visit_interaction', 'sys_steroid_days_since_last_fill', 'sys_steroid_days_supply_total', 'sys_steroid_ever_filled', 'sys_steroid_fill_count', 'thyroid_autoimmune_claim_count', 'top_steroid_days_since_last_fill', 'top_steroid_days_supply_total', 'top_steroid_ever_filled', 'top_steroid_fill_count', 'tpo_ab_abnormal_flag', 'tpo_ab_result_last', 'tpo_ab_tested', 'tsh_abnormal_flag', 'tsh_result_last', 'tsh_tested', 'unique_providers', 'urban_rural_code', 'zip3', 'zip5', 'zip_code']
- full admissible set: ['age_at_index', 'age_group', 'allergic_rhinitis_claim_count', 'ana_abnormal_flag', 'ana_result_last', 'ana_tested', 'angioedema_claim_count', 'anxiety_claim_count', 'asthma_claim_count', 'atopic_dermatitis_claim_count', 'atopy_score', 'cbc_abnormal_flag', 'cbc_result_last', 'cbc_tested', 'charlson_score', 'comorbidity_load_total', 'crp_abnormal_flag', 'crp_result_last', 'crp_tested', 'csu_chronicity', 'csu_dx_intensity', 'depression_claim_count', 'dx_angioedema_count', 'dx_l50_1_count', 'dx_l50_8_count', 'dx_l50_9_count', 'dx_total_csu', 'ed_visits_total', 'ed_visits_urticaria_angio', 'elixhauser_score', 'eosinophil_abnormal_flag', 'eosinophil_result_last', 'eosinophil_tested', 'free_t4_abnormal_flag', 'free_t4_result_last', 'free_t4_tested', 'gender', 'geographic_region', 'h1_1g_days_since_last_fill', 'h1_1g_days_supply_total', 'h1_1g_ever_filled', 'h1_1g_fill_count', 'h1_2g_days_since_last_fill', 'h1_2g_days_supply_total', 'h1_2g_ever_filled', 'h1_2g_fill_count', 'h2_days_since_last_fill', 'h2_days_supply_total', 'h2_ever_filled', 'h2_fill_count', 'has_allergic_rhinitis', 'has_angioedema', 'has_anxiety', 'has_asthma', 'has_atopic_dermatitis', 'has_depression', 'has_nsaid_hypersensitivity', 'has_thyroid_autoimmune', 'hospitalizations_total', 'ige_total_abnormal_flag', 'ige_total_result_last', 'ige_total_tested', 'immunosupp_days_since_last_fill', 'immunosupp_days_supply_total', 'immunosupp_ever_filled', 'immunosupp_fill_count', 'index_date', 'insurance_product', 'lab_workup_completeness', 'lookback_start_date', 'ltra_days_since_last_fill', 'ltra_days_supply_total', 'ltra_ever_filled', 'ltra_fill_count', 'mental_health_flag', 'months_since_first_dx', 'nsaid_hypersensitivity_claim_count', 'office_visits_allergist', 'office_visits_dermatology', 'office_visits_pcp', 'office_visits_total', 'payer_category', 'plan_type', 'polypharmacy_breadth', 'primary_diagnosis_code', 'primary_specialist_type', 'saw_allergist_flag', 'saw_dermatologist_flag', 'specialist_concentration', 'specialist_visit_interaction', 'sys_steroid_days_since_last_fill', 'sys_steroid_days_supply_total', 'sys_steroid_ever_filled', 'sys_steroid_fill_count', 'thyroid_autoimmune_claim_count', 'top_steroid_days_since_last_fill', 'top_steroid_days_supply_total', 'top_steroid_ever_filled', 'top_steroid_fill_count', 'tpo_ab_abnormal_flag', 'tpo_ab_result_last', 'tpo_ab_tested', 'tsh_abnormal_flag', 'tsh_result_last', 'tsh_tested', 'unique_providers', 'urban_rural_code', 'zip3', 'zip5', 'zip_code']

## Features

| feature | fragment role | cohort role | drift | ambiguous | review | panel role | leak | grades |
|---|---|---|---|---|---|---|---|---|
| age_at_index | confounder | confounder |  |  |  | None |  | age_at_index->T:unsupported, age_at_index->Y:unsupported, T->Y:estimand |
| age_group | confounder | confounder |  |  |  | None |  | age_group->T:unsupported, age_group->Y:unsupported, T->Y:estimand |
| allergic_rhinitis_claim_count | confounder | confounder |  |  |  | None |  | allergic_rhinitis_claim_count->T:unsupported, allergic_rhinitis_claim_count->Y:unsupported, T->Y:estimand |
| ana_abnormal_flag | confounder | confounder |  |  |  | None |  | ana_abnormal_flag->T:unsupported, ana_abnormal_flag->Y:unsupported, T->Y:estimand |
| ana_result_last | confounder | confounder |  |  |  | None |  | ana_result_last->T:unsupported, ana_result_last->Y:unsupported, T->Y:estimand |
| ana_tested | confounder | confounder |  |  |  | None |  | ana_tested->T:unsupported, ana_tested->Y:unsupported, T->Y:estimand |
| angioedema_claim_count | confounder | confounder |  |  |  | None |  | angioedema_claim_count->T:unsupported, angioedema_claim_count->Y:unsupported, T->Y:estimand |
| anxiety_claim_count | confounder | confounder |  |  |  | None |  | anxiety_claim_count->T:unsupported, anxiety_claim_count->Y:unsupported, T->Y:estimand |
| asthma_claim_count | confounder | confounder |  |  |  | None |  | asthma_claim_count->T:unsupported, asthma_claim_count->Y:unsupported, T->Y:estimand |
| atopic_dermatitis_claim_count | confounder | confounder |  |  |  | None |  | atopic_dermatitis_claim_count->T:unsupported, atopic_dermatitis_claim_count->Y:unsupported, T->Y:estimand |
| atopy_score | confounder | confounder |  |  |  | None |  | atopy_score->T:unsupported, atopy_score->Y:unsupported, T->Y:estimand |
| cbc_abnormal_flag | confounder | confounder |  |  |  | None |  | cbc_abnormal_flag->T:unsupported, cbc_abnormal_flag->Y:unsupported, T->Y:estimand |
| cbc_result_last | confounder | confounder |  |  |  | None |  | cbc_result_last->T:unsupported, cbc_result_last->Y:unsupported, T->Y:estimand |
| cbc_tested | confounder | confounder |  |  |  | None |  | cbc_tested->T:unsupported, cbc_tested->Y:unsupported, T->Y:estimand |
| charlson_score | confounder | confounder |  |  |  | None |  | charlson_score->T:unsupported, charlson_score->Y:unsupported, T->Y:estimand |
| comorbidity_load_total | confounder | confounder |  |  |  | None |  | comorbidity_load_total->T:unsupported, comorbidity_load_total->Y:unsupported, T->Y:estimand |
| crp_abnormal_flag | confounder | confounder |  |  |  | None |  | crp_abnormal_flag->T:unsupported, crp_abnormal_flag->Y:unsupported, T->Y:estimand |
| crp_result_last | confounder | confounder |  |  |  | None |  | crp_result_last->T:unsupported, crp_result_last->Y:unsupported, T->Y:estimand |
| crp_tested | confounder | confounder |  |  |  | None |  | crp_tested->T:unsupported, crp_tested->Y:unsupported, T->Y:estimand |
| csu_chronicity | confounder | confounder |  |  |  | None |  | csu_chronicity->T:unsupported, csu_chronicity->Y:unsupported, T->Y:estimand |
| csu_dx_intensity | confounder | confounder |  |  |  | None |  | csu_dx_intensity->T:unsupported, csu_dx_intensity->Y:unsupported, T->Y:estimand |
| depression_claim_count | confounder | confounder |  |  |  | None |  | depression_claim_count->T:unsupported, depression_claim_count->Y:unsupported, T->Y:estimand |
| dx_angioedema_count | confounder | confounder |  |  |  | None |  | dx_angioedema_count->T:unsupported, dx_angioedema_count->Y:unsupported, T->Y:estimand |
| dx_l50_1_count | confounder | confounder |  |  |  | None |  | dx_l50_1_count->T:unsupported, dx_l50_1_count->Y:unsupported, T->Y:estimand |
| dx_l50_8_count | confounder | confounder |  |  |  | None |  | dx_l50_8_count->T:unsupported, dx_l50_8_count->Y:unsupported, T->Y:estimand |
| dx_l50_9_count | confounder | confounder |  |  |  | None |  | dx_l50_9_count->T:unsupported, dx_l50_9_count->Y:unsupported, T->Y:estimand |
| dx_total_csu | confounder | confounder |  |  |  | None |  | dx_total_csu->T:unsupported, dx_total_csu->Y:unsupported, T->Y:estimand |
| ed_visits_total | confounder | confounder |  |  |  | None |  | ed_visits_total->T:unsupported, ed_visits_total->Y:unsupported, T->Y:estimand |
| ed_visits_urticaria_angio | confounder | confounder |  |  |  | None |  | ed_visits_urticaria_angio->T:unsupported, ed_visits_urticaria_angio->Y:unsupported, T->Y:estimand |
| elixhauser_score | confounder | confounder |  |  |  | None |  | elixhauser_score->T:unsupported, elixhauser_score->Y:unsupported, T->Y:estimand |
| eosinophil_abnormal_flag | confounder | confounder |  |  |  | None |  | eosinophil_abnormal_flag->T:unsupported, eosinophil_abnormal_flag->Y:unsupported, T->Y:estimand |
| eosinophil_result_last | confounder | confounder |  |  |  | None |  | eosinophil_result_last->T:unsupported, eosinophil_result_last->Y:unsupported, T->Y:estimand |
| eosinophil_tested | confounder | confounder |  |  |  | None |  | eosinophil_tested->T:unsupported, eosinophil_tested->Y:unsupported, T->Y:estimand |
| free_t4_abnormal_flag | confounder | confounder |  |  |  | None |  | free_t4_abnormal_flag->T:unsupported, free_t4_abnormal_flag->Y:unsupported, T->Y:estimand |
| free_t4_result_last | confounder | confounder |  |  |  | None |  | free_t4_result_last->T:unsupported, free_t4_result_last->Y:unsupported, T->Y:estimand |
| free_t4_tested | confounder | confounder |  |  |  | None |  | free_t4_tested->T:unsupported, free_t4_tested->Y:unsupported, T->Y:estimand |
| gender | confounder | confounder |  |  |  | None |  | gender->T:unsupported, gender->Y:unsupported, T->Y:estimand |
| geographic_region | confounder | confounder |  |  |  | None |  | geographic_region->T:unsupported, geographic_region->Y:unsupported, T->Y:estimand |
| h1_1g_days_since_last_fill | confounder | confounder |  |  |  | None |  | h1_1g_days_since_last_fill->T:unsupported, h1_1g_days_since_last_fill->Y:unsupported, T->Y:estimand |
| h1_1g_days_supply_total | confounder | confounder |  |  |  | None |  | h1_1g_days_supply_total->T:unsupported, h1_1g_days_supply_total->Y:unsupported, T->Y:estimand |
| h1_1g_ever_filled | confounder | confounder |  |  |  | None |  | h1_1g_ever_filled->T:unsupported, h1_1g_ever_filled->Y:unsupported, T->Y:estimand |
| h1_1g_fill_count | confounder | confounder |  |  |  | None |  | h1_1g_fill_count->T:unsupported, h1_1g_fill_count->Y:unsupported, T->Y:estimand |
| h1_2g_days_since_last_fill | confounder | confounder |  |  |  | None |  | h1_2g_days_since_last_fill->T:unsupported, h1_2g_days_since_last_fill->Y:unsupported, T->Y:estimand |
| h1_2g_days_supply_total | confounder | confounder |  |  |  | None |  | h1_2g_days_supply_total->T:unsupported, h1_2g_days_supply_total->Y:unsupported, T->Y:estimand |
| h1_2g_ever_filled | confounder | confounder |  |  |  | None |  | h1_2g_ever_filled->T:unsupported, h1_2g_ever_filled->Y:unsupported, T->Y:estimand |
| h1_2g_fill_count | confounder | confounder |  |  |  | None |  | h1_2g_fill_count->T:unsupported, h1_2g_fill_count->Y:unsupported, T->Y:estimand |
| h2_days_since_last_fill | confounder | confounder |  |  |  | None |  | h2_days_since_last_fill->T:unsupported, h2_days_since_last_fill->Y:unsupported, T->Y:estimand |
| h2_days_supply_total | confounder | confounder |  |  |  | None |  | h2_days_supply_total->T:unsupported, h2_days_supply_total->Y:unsupported, T->Y:estimand |
| h2_ever_filled | confounder | confounder |  |  |  | None |  | h2_ever_filled->T:unsupported, h2_ever_filled->Y:unsupported, T->Y:estimand |
| h2_fill_count | confounder | confounder |  |  |  | None |  | h2_fill_count->T:unsupported, h2_fill_count->Y:unsupported, T->Y:estimand |
| has_allergic_rhinitis | confounder | confounder |  |  |  | None |  | has_allergic_rhinitis->T:unsupported, has_allergic_rhinitis->Y:unsupported, T->Y:estimand |
| has_angioedema | confounder | confounder |  |  |  | None |  | has_angioedema->T:unsupported, has_angioedema->Y:unsupported, T->Y:estimand |
| has_anxiety | confounder | confounder |  |  |  | None |  | has_anxiety->T:unsupported, has_anxiety->Y:unsupported, T->Y:estimand |
| has_asthma | confounder | confounder |  |  |  | None |  | has_asthma->T:unsupported, has_asthma->Y:unsupported, T->Y:estimand |
| has_atopic_dermatitis | confounder | confounder |  |  |  | None |  | has_atopic_dermatitis->T:unsupported, has_atopic_dermatitis->Y:unsupported, T->Y:estimand |
| has_depression | confounder | confounder |  |  |  | None |  | has_depression->T:unsupported, has_depression->Y:unsupported, T->Y:estimand |
| has_nsaid_hypersensitivity | confounder | confounder |  |  |  | None |  | has_nsaid_hypersensitivity->T:unsupported, has_nsaid_hypersensitivity->Y:unsupported, T->Y:estimand |
| has_thyroid_autoimmune | confounder | confounder |  |  |  | None |  | has_thyroid_autoimmune->T:unsupported, has_thyroid_autoimmune->Y:unsupported, T->Y:estimand |
| hospitalizations_total | confounder | confounder |  |  |  | None |  | hospitalizations_total->T:unsupported, hospitalizations_total->Y:unsupported, T->Y:estimand |
| ige_total_abnormal_flag | confounder | confounder |  |  |  | None |  | ige_total_abnormal_flag->T:unsupported, ige_total_abnormal_flag->Y:unsupported, T->Y:estimand |
| ige_total_result_last | confounder | confounder |  |  |  | None |  | ige_total_result_last->T:unsupported, ige_total_result_last->Y:unsupported, T->Y:estimand |
| ige_total_tested | confounder | confounder |  |  |  | None |  | ige_total_tested->T:unsupported, ige_total_tested->Y:unsupported, T->Y:estimand |
| immunosupp_days_since_last_fill | confounder | confounder |  |  |  | None |  | immunosupp_days_since_last_fill->T:unsupported, immunosupp_days_since_last_fill->Y:unsupported, T->Y:estimand |
| immunosupp_days_supply_total | confounder | confounder |  |  |  | None |  | immunosupp_days_supply_total->T:unsupported, immunosupp_days_supply_total->Y:unsupported, T->Y:estimand |
| immunosupp_ever_filled | confounder | confounder |  |  |  | None |  | immunosupp_ever_filled->T:unsupported, immunosupp_ever_filled->Y:unsupported, T->Y:estimand |
| immunosupp_fill_count | confounder | confounder |  |  |  | None |  | immunosupp_fill_count->T:unsupported, immunosupp_fill_count->Y:unsupported, T->Y:estimand |
| index_date | confounder | confounder |  |  |  | None |  | index_date->T:unsupported, index_date->Y:unsupported, T->Y:estimand |
| insurance_product | confounder | confounder |  |  |  | None |  | insurance_product->T:unsupported, insurance_product->Y:unsupported, T->Y:estimand |
| lab_workup_completeness | confounder | confounder |  |  |  | None |  | lab_workup_completeness->T:unsupported, lab_workup_completeness->Y:unsupported, T->Y:estimand |
| lookback_start_date | confounder | confounder |  |  |  | None |  | lookback_start_date->T:unsupported, lookback_start_date->Y:unsupported, T->Y:estimand |
| ltra_days_since_last_fill | confounder | confounder |  |  |  | None |  | ltra_days_since_last_fill->T:unsupported, ltra_days_since_last_fill->Y:unsupported, T->Y:estimand |
| ltra_days_supply_total | confounder | confounder |  |  |  | None |  | ltra_days_supply_total->T:unsupported, ltra_days_supply_total->Y:unsupported, T->Y:estimand |
| ltra_ever_filled | confounder | confounder |  |  |  | None |  | ltra_ever_filled->T:unsupported, ltra_ever_filled->Y:unsupported, T->Y:estimand |
| ltra_fill_count | confounder | confounder |  |  |  | None |  | ltra_fill_count->T:unsupported, ltra_fill_count->Y:unsupported, T->Y:estimand |
| mental_health_flag | confounder | confounder |  |  |  | None |  | mental_health_flag->T:unsupported, mental_health_flag->Y:unsupported, T->Y:estimand |
| months_since_first_dx | confounder | confounder |  |  |  | None |  | months_since_first_dx->T:unsupported, months_since_first_dx->Y:unsupported, T->Y:estimand |
| nsaid_hypersensitivity_claim_count | confounder | confounder |  |  |  | None |  | nsaid_hypersensitivity_claim_count->T:unsupported, nsaid_hypersensitivity_claim_count->Y:unsupported, T->Y:estimand |
| office_visits_allergist | confounder | confounder |  |  |  | None |  | office_visits_allergist->T:unsupported, office_visits_allergist->Y:unsupported, T->Y:estimand |
| office_visits_dermatology | confounder | confounder |  |  |  | None |  | office_visits_dermatology->T:unsupported, office_visits_dermatology->Y:unsupported, T->Y:estimand |
| office_visits_pcp | confounder | confounder |  |  |  | None |  | office_visits_pcp->T:unsupported, office_visits_pcp->Y:unsupported, T->Y:estimand |
| office_visits_total | confounder | confounder |  |  |  | None |  | office_visits_total->T:unsupported, office_visits_total->Y:unsupported, T->Y:estimand |
| payer_category | confounder | confounder |  |  |  | None |  | payer_category->T:unsupported, payer_category->Y:unsupported, T->Y:estimand |
| plan_type | confounder | confounder |  |  |  | None |  | plan_type->T:unsupported, plan_type->Y:unsupported, T->Y:estimand |
| polypharmacy_breadth | confounder | confounder |  |  |  | None |  | polypharmacy_breadth->T:unsupported, polypharmacy_breadth->Y:unsupported, T->Y:estimand |
| primary_diagnosis_code | confounder | confounder |  |  |  | None |  | primary_diagnosis_code->T:unsupported, primary_diagnosis_code->Y:unsupported, T->Y:estimand |
| primary_specialist_type | confounder | confounder |  |  |  | None |  | primary_specialist_type->T:unsupported, primary_specialist_type->Y:unsupported, T->Y:estimand |
| saw_allergist_flag | confounder | confounder |  |  |  | None |  | saw_allergist_flag->T:unsupported, saw_allergist_flag->Y:unsupported, T->Y:estimand |
| saw_dermatologist_flag | confounder | confounder |  |  |  | None |  | saw_dermatologist_flag->T:unsupported, saw_dermatologist_flag->Y:unsupported, T->Y:estimand |
| specialist_concentration | confounder | confounder |  |  |  | None |  | specialist_concentration->T:unsupported, specialist_concentration->Y:unsupported, T->Y:estimand |
| specialist_visit_interaction | confounder | confounder |  |  |  | None |  | specialist_visit_interaction->T:unsupported, specialist_visit_interaction->Y:unsupported, T->Y:estimand |
| sys_steroid_days_since_last_fill | confounder | confounder |  |  |  | None |  | sys_steroid_days_since_last_fill->T:unsupported, sys_steroid_days_since_last_fill->Y:unsupported, T->Y:estimand |
| sys_steroid_days_supply_total | confounder | confounder |  |  |  | None |  | sys_steroid_days_supply_total->T:unsupported, sys_steroid_days_supply_total->Y:unsupported, T->Y:estimand |
| sys_steroid_ever_filled | confounder | confounder |  |  |  | None |  | sys_steroid_ever_filled->T:unsupported, sys_steroid_ever_filled->Y:unsupported, T->Y:estimand |
| sys_steroid_fill_count | confounder | confounder |  |  |  | None |  | sys_steroid_fill_count->T:unsupported, sys_steroid_fill_count->Y:unsupported, T->Y:estimand |
| thyroid_autoimmune_claim_count | confounder | confounder |  |  |  | None |  | thyroid_autoimmune_claim_count->T:unsupported, thyroid_autoimmune_claim_count->Y:unsupported, T->Y:estimand |
| top_steroid_days_since_last_fill | confounder | confounder |  |  |  | None |  | top_steroid_days_since_last_fill->T:unsupported, top_steroid_days_since_last_fill->Y:unsupported, T->Y:estimand |
| top_steroid_days_supply_total | confounder | confounder |  |  |  | None |  | top_steroid_days_supply_total->T:unsupported, top_steroid_days_supply_total->Y:unsupported, T->Y:estimand |
| top_steroid_ever_filled | confounder | confounder |  |  |  | None |  | top_steroid_ever_filled->T:unsupported, top_steroid_ever_filled->Y:unsupported, T->Y:estimand |
| top_steroid_fill_count | confounder | confounder |  |  |  | None |  | top_steroid_fill_count->T:unsupported, top_steroid_fill_count->Y:unsupported, T->Y:estimand |
| tpo_ab_abnormal_flag | confounder | confounder |  |  |  | None |  | tpo_ab_abnormal_flag->T:unsupported, tpo_ab_abnormal_flag->Y:unsupported, T->Y:estimand |
| tpo_ab_result_last | confounder | confounder |  |  |  | None |  | tpo_ab_result_last->T:unsupported, tpo_ab_result_last->Y:unsupported, T->Y:estimand |
| tpo_ab_tested | confounder | confounder |  |  |  | None |  | tpo_ab_tested->T:unsupported, tpo_ab_tested->Y:unsupported, T->Y:estimand |
| tsh_abnormal_flag | confounder | confounder |  |  |  | None |  | tsh_abnormal_flag->T:unsupported, tsh_abnormal_flag->Y:unsupported, T->Y:estimand |
| tsh_result_last | confounder | confounder |  |  |  | None |  | tsh_result_last->T:unsupported, tsh_result_last->Y:unsupported, T->Y:estimand |
| tsh_tested | confounder | confounder |  |  |  | None |  | tsh_tested->T:unsupported, tsh_tested->Y:unsupported, T->Y:estimand |
| unique_providers | confounder | confounder |  |  |  | None |  | unique_providers->T:unsupported, unique_providers->Y:unsupported, T->Y:estimand |
| urban_rural_code | confounder | confounder |  |  |  | None |  | urban_rural_code->T:unsupported, urban_rural_code->Y:unsupported, T->Y:estimand |
| zip3 | confounder | confounder |  |  |  | None |  | zip3->T:unsupported, zip3->Y:unsupported, T->Y:estimand |
| zip5 | confounder | confounder |  |  |  | None |  | zip5->T:unsupported, zip5->Y:unsupported, T->Y:estimand |
| zip_code | confounder | confounder |  |  |  | None |  | zip_code->T:unsupported, zip_code->Y:unsupported, T->Y:estimand |

## Review items (0)

- none

## Edge rationale (non-estimand edges)

- age_at_index → T (unsupported): — — no citation
- age_at_index → Y (unsupported): — — no citation
- age_group → T (unsupported): — — no citation
- age_group → Y (unsupported): — — no citation
- gender → T (unsupported): — — no citation
- gender → Y (unsupported): — — no citation
- zip5 → T (unsupported): — — no citation
- zip5 → Y (unsupported): — — no citation
- zip3 → T (unsupported): — — no citation
- zip3 → Y (unsupported): — — no citation
- zip_code → T (unsupported): — — no citation
- zip_code → Y (unsupported): — — no citation
- geographic_region → T (unsupported): — — no citation
- geographic_region → Y (unsupported): — — no citation
- insurance_product → T (unsupported): — — no citation
- insurance_product → Y (unsupported): — — no citation
- plan_type → T (unsupported): — — no citation
- plan_type → Y (unsupported): — — no citation
- payer_category → T (unsupported): — — no citation
- payer_category → Y (unsupported): — — no citation
- urban_rural_code → T (unsupported): — — no citation
- urban_rural_code → Y (unsupported): — — no citation
- primary_diagnosis_code → T (unsupported): — — no citation
- primary_diagnosis_code → Y (unsupported): — — no citation
- dx_l50_1_count → T (unsupported): — — no citation
- dx_l50_1_count → Y (unsupported): — — no citation
- dx_l50_8_count → T (unsupported): — — no citation
- dx_l50_8_count → Y (unsupported): — — no citation
- dx_l50_9_count → T (unsupported): — — no citation
- dx_l50_9_count → Y (unsupported): — — no citation
- dx_total_csu → T (unsupported): — — no citation
- dx_total_csu → Y (unsupported): — — no citation
- dx_angioedema_count → T (unsupported): — — no citation
- dx_angioedema_count → Y (unsupported): — — no citation
- months_since_first_dx → T (unsupported): — — no citation
- months_since_first_dx → Y (unsupported): — — no citation
- csu_chronicity → T (unsupported): — — no citation
- csu_chronicity → Y (unsupported): — — no citation
- has_atopic_dermatitis → T (unsupported): — — no citation
- has_atopic_dermatitis → Y (unsupported): — — no citation
- atopic_dermatitis_claim_count → T (unsupported): — — no citation
- atopic_dermatitis_claim_count → Y (unsupported): — — no citation
- has_asthma → T (unsupported): — — no citation
- has_asthma → Y (unsupported): — — no citation
- asthma_claim_count → T (unsupported): — — no citation
- asthma_claim_count → Y (unsupported): — — no citation
- has_allergic_rhinitis → T (unsupported): — — no citation
- has_allergic_rhinitis → Y (unsupported): — — no citation
- allergic_rhinitis_claim_count → T (unsupported): — — no citation
- allergic_rhinitis_claim_count → Y (unsupported): — — no citation
- has_anxiety → T (unsupported): — — no citation
- has_anxiety → Y (unsupported): — — no citation
- anxiety_claim_count → T (unsupported): — — no citation
- anxiety_claim_count → Y (unsupported): — — no citation
- has_depression → T (unsupported): — — no citation
- has_depression → Y (unsupported): — — no citation
- depression_claim_count → T (unsupported): — — no citation
- depression_claim_count → Y (unsupported): — — no citation
- has_thyroid_autoimmune → T (unsupported): — — no citation
- has_thyroid_autoimmune → Y (unsupported): — — no citation
- thyroid_autoimmune_claim_count → T (unsupported): — — no citation
- thyroid_autoimmune_claim_count → Y (unsupported): — — no citation
- has_nsaid_hypersensitivity → T (unsupported): — — no citation
- has_nsaid_hypersensitivity → Y (unsupported): — — no citation
- nsaid_hypersensitivity_claim_count → T (unsupported): — — no citation
- nsaid_hypersensitivity_claim_count → Y (unsupported): — — no citation
- has_angioedema → T (unsupported): — — no citation
- has_angioedema → Y (unsupported): — — no citation
- angioedema_claim_count → T (unsupported): — — no citation
- angioedema_claim_count → Y (unsupported): — — no citation
- atopy_score → T (unsupported): — — no citation
- atopy_score → Y (unsupported): — — no citation
- mental_health_flag → T (unsupported): — — no citation
- mental_health_flag → Y (unsupported): — — no citation
- elixhauser_score → T (unsupported): — — no citation
- elixhauser_score → Y (unsupported): — — no citation
- charlson_score → T (unsupported): — — no citation
- charlson_score → Y (unsupported): — — no citation
- office_visits_total → T (unsupported): — — no citation
- office_visits_total → Y (unsupported): — — no citation
- office_visits_allergist → T (unsupported): — — no citation
- office_visits_allergist → Y (unsupported): — — no citation
- office_visits_dermatology → T (unsupported): — — no citation
- office_visits_dermatology → Y (unsupported): — — no citation
- office_visits_pcp → T (unsupported): — — no citation
- office_visits_pcp → Y (unsupported): — — no citation
- ed_visits_total → T (unsupported): — — no citation
- ed_visits_total → Y (unsupported): — — no citation
- ed_visits_urticaria_angio → T (unsupported): — — no citation
- ed_visits_urticaria_angio → Y (unsupported): — — no citation
- hospitalizations_total → T (unsupported): — — no citation
- hospitalizations_total → Y (unsupported): — — no citation
- unique_providers → T (unsupported): — — no citation
- unique_providers → Y (unsupported): — — no citation
- h1_1g_ever_filled → T (unsupported): — — no citation
- h1_1g_ever_filled → Y (unsupported): — — no citation
- h1_1g_fill_count → T (unsupported): — — no citation
- h1_1g_fill_count → Y (unsupported): — — no citation
- h1_1g_days_supply_total → T (unsupported): — — no citation
- h1_1g_days_supply_total → Y (unsupported): — — no citation
- h1_1g_days_since_last_fill → T (unsupported): — — no citation
- h1_1g_days_since_last_fill → Y (unsupported): — — no citation
- h1_2g_ever_filled → T (unsupported): — — no citation
- h1_2g_ever_filled → Y (unsupported): — — no citation
- h1_2g_fill_count → T (unsupported): — — no citation
- h1_2g_fill_count → Y (unsupported): — — no citation
- h1_2g_days_supply_total → T (unsupported): — — no citation
- h1_2g_days_supply_total → Y (unsupported): — — no citation
- h1_2g_days_since_last_fill → T (unsupported): — — no citation
- h1_2g_days_since_last_fill → Y (unsupported): — — no citation
- h2_ever_filled → T (unsupported): — — no citation
- h2_ever_filled → Y (unsupported): — — no citation
- h2_fill_count → T (unsupported): — — no citation
- h2_fill_count → Y (unsupported): — — no citation
- h2_days_supply_total → T (unsupported): — — no citation
- h2_days_supply_total → Y (unsupported): — — no citation
- h2_days_since_last_fill → T (unsupported): — — no citation
- h2_days_since_last_fill → Y (unsupported): — — no citation
- ltra_ever_filled → T (unsupported): — — no citation
- ltra_ever_filled → Y (unsupported): — — no citation
- ltra_fill_count → T (unsupported): — — no citation
- ltra_fill_count → Y (unsupported): — — no citation
- ltra_days_supply_total → T (unsupported): — — no citation
- ltra_days_supply_total → Y (unsupported): — — no citation
- ltra_days_since_last_fill → T (unsupported): — — no citation
- ltra_days_since_last_fill → Y (unsupported): — — no citation
- sys_steroid_ever_filled → T (unsupported): — — no citation
- sys_steroid_ever_filled → Y (unsupported): — — no citation
- sys_steroid_fill_count → T (unsupported): — — no citation
- sys_steroid_fill_count → Y (unsupported): — — no citation
- sys_steroid_days_supply_total → T (unsupported): — — no citation
- sys_steroid_days_supply_total → Y (unsupported): — — no citation
- sys_steroid_days_since_last_fill → T (unsupported): — — no citation
- sys_steroid_days_since_last_fill → Y (unsupported): — — no citation
- top_steroid_ever_filled → T (unsupported): — — no citation
- top_steroid_ever_filled → Y (unsupported): — — no citation
- top_steroid_fill_count → T (unsupported): — — no citation
- top_steroid_fill_count → Y (unsupported): — — no citation
- top_steroid_days_supply_total → T (unsupported): — — no citation
- top_steroid_days_supply_total → Y (unsupported): — — no citation
- top_steroid_days_since_last_fill → T (unsupported): — — no citation
- top_steroid_days_since_last_fill → Y (unsupported): — — no citation
- immunosupp_ever_filled → T (unsupported): — — no citation
- immunosupp_ever_filled → Y (unsupported): — — no citation
- immunosupp_fill_count → T (unsupported): — — no citation
- immunosupp_fill_count → Y (unsupported): — — no citation
- immunosupp_days_supply_total → T (unsupported): — — no citation
- immunosupp_days_supply_total → Y (unsupported): — — no citation
- immunosupp_days_since_last_fill → T (unsupported): — — no citation
- immunosupp_days_since_last_fill → Y (unsupported): — — no citation
- ige_total_tested → T (unsupported): — — no citation
- ige_total_tested → Y (unsupported): — — no citation
- ige_total_result_last → T (unsupported): — — no citation
- ige_total_result_last → Y (unsupported): — — no citation
- ige_total_abnormal_flag → T (unsupported): — — no citation
- ige_total_abnormal_flag → Y (unsupported): — — no citation
- eosinophil_tested → T (unsupported): — — no citation
- eosinophil_tested → Y (unsupported): — — no citation
- eosinophil_result_last → T (unsupported): — — no citation
- eosinophil_result_last → Y (unsupported): — — no citation
- eosinophil_abnormal_flag → T (unsupported): — — no citation
- eosinophil_abnormal_flag → Y (unsupported): — — no citation
- crp_tested → T (unsupported): — — no citation
- crp_tested → Y (unsupported): — — no citation
- crp_result_last → T (unsupported): — — no citation
- crp_result_last → Y (unsupported): — — no citation
- crp_abnormal_flag → T (unsupported): — — no citation
- crp_abnormal_flag → Y (unsupported): — — no citation
- tpo_ab_tested → T (unsupported): — — no citation
- tpo_ab_tested → Y (unsupported): — — no citation
- tpo_ab_result_last → T (unsupported): — — no citation
- tpo_ab_result_last → Y (unsupported): — — no citation
- tpo_ab_abnormal_flag → T (unsupported): — — no citation
- tpo_ab_abnormal_flag → Y (unsupported): — — no citation
- free_t4_tested → T (unsupported): — — no citation
- free_t4_tested → Y (unsupported): — — no citation
- free_t4_result_last → T (unsupported): — — no citation
- free_t4_result_last → Y (unsupported): — — no citation
- free_t4_abnormal_flag → T (unsupported): — — no citation
- free_t4_abnormal_flag → Y (unsupported): — — no citation
- tsh_tested → T (unsupported): — — no citation
- tsh_tested → Y (unsupported): — — no citation
- tsh_result_last → T (unsupported): — — no citation
- tsh_result_last → Y (unsupported): — — no citation
- tsh_abnormal_flag → T (unsupported): — — no citation
- tsh_abnormal_flag → Y (unsupported): — — no citation
- ana_tested → T (unsupported): — — no citation
- ana_tested → Y (unsupported): — — no citation
- ana_result_last → T (unsupported): — — no citation
- ana_result_last → Y (unsupported): — — no citation
- ana_abnormal_flag → T (unsupported): — — no citation
- ana_abnormal_flag → Y (unsupported): — — no citation
- cbc_tested → T (unsupported): — — no citation
- cbc_tested → Y (unsupported): — — no citation
- cbc_result_last → T (unsupported): — — no citation
- cbc_result_last → Y (unsupported): — — no citation
- cbc_abnormal_flag → T (unsupported): — — no citation
- cbc_abnormal_flag → Y (unsupported): — — no citation
- specialist_concentration → T (unsupported): — — no citation
- specialist_concentration → Y (unsupported): — — no citation
- primary_specialist_type → T (unsupported): — — no citation
- primary_specialist_type → Y (unsupported): — — no citation
- saw_allergist_flag → T (unsupported): — — no citation
- saw_allergist_flag → Y (unsupported): — — no citation
- saw_dermatologist_flag → T (unsupported): — — no citation
- saw_dermatologist_flag → Y (unsupported): — — no citation
- index_date → T (unsupported): — — no citation
- index_date → Y (unsupported): — — no citation
- lookback_start_date → T (unsupported): — — no citation
- lookback_start_date → Y (unsupported): — — no citation
- comorbidity_load_total → T (unsupported): — — no citation
- comorbidity_load_total → Y (unsupported): — — no citation
- csu_dx_intensity → T (unsupported): — — no citation
- csu_dx_intensity → Y (unsupported): — — no citation
- polypharmacy_breadth → T (unsupported): — — no citation
- polypharmacy_breadth → Y (unsupported): — — no citation
- lab_workup_completeness → T (unsupported): — — no citation
- lab_workup_completeness → Y (unsupported): — — no citation
- specialist_visit_interaction → T (unsupported): — — no citation
- specialist_visit_interaction → Y (unsupported): — — no citation

## Diff against the manifest's machine attestations

- compared 110/110; edge-exact agreement 93; role agreement 93; disagreements 17 (no threshold)
- **zip5**: authored confounder vs manifest instrument; authored-only ['zip5->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:59): - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **zip3**: authored confounder vs manifest instrument; authored-only ['zip3->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:59): - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **zip_code**: authored confounder vs manifest instrument; authored-only ['zip_code->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:59): - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **geographic_region**: authored confounder vs manifest instrument; authored-only ['geographic_region->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:59): - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **insurance_product**: authored confounder vs manifest instrument; authored-only ['insurance_product->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:60): - Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **plan_type**: authored confounder vs manifest instrument; authored-only ['plan_type->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:60): - Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **payer_category**: authored confounder vs manifest instrument; authored-only ['payer_category->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:60): - Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **urban_rural_code**: authored confounder vs manifest instrument; authored-only ['urban_rural_code->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:59): - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **office_visits_allergist**: authored confounder vs manifest instrument; authored-only ['office_visits_allergist->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **office_visits_dermatology**: authored confounder vs manifest instrument; authored-only ['office_visits_dermatology->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **specialist_concentration**: authored confounder vs manifest instrument; authored-only ['specialist_concentration->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **primary_specialist_type**: authored confounder vs manifest instrument; authored-only ['primary_specialist_type->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **saw_allergist_flag**: authored confounder vs manifest instrument; authored-only ['saw_allergist_flag->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **saw_dermatologist_flag**: authored confounder vs manifest instrument; authored-only ['saw_dermatologist_flag->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **index_date**: authored confounder vs manifest instrument; authored-only ['index_date->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:62): - Calendar anchors — `index_date`, `lookback_start_date`: temporal adoption of biologics (PMID 32382379; PMID 40004611).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **lookback_start_date**: authored confounder vs manifest instrument; authored-only ['lookback_start_date->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:62): - Calendar anchors — `index_date`, `lookback_start_date`: temporal adoption of biologics (PMID 32382379; PMID 40004611).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
- **specialist_visit_interaction**: authored confounder vs manifest instrument; authored-only ['specialist_visit_interaction->initiated_biologic_180d']; manifest-only []
  - manifest (docs/layer4/optum_initiation_attestation_research.md:61): - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934).
  - author: fake LM (dry run): stand-in confounder fragment, not an authored claim
