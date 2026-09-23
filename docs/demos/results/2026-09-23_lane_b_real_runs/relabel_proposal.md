# Relabel PROPOSAL — the 110 `optum` machine attestations vs the real structural author (run b)

**Status: PROPOSAL. Nothing in `src/data/manifests/optum_feature_manifest.py` was edited, and nothing is recommended for relabelling on this table alone.** The relabel is applied only after the owner reads this diff AND approves the pending run (b) review (spec §3 Lane B item 1; PR #2230 owner decision 3).

**What `machine_reviewed` means (the contract, `src/data/feature_contract.py` `ATTESTATION_DECIDING_PROVENANCES`, `may_decide()`):** *machine-authored, approved by a human in the expert-review queue (review id on the record)* — a DECIDING provenance: it acts in the structural decider and may seed a structural prior, exactly like `human`. A machine-vs-machine agreement is therefore NOT sufficient for it; the human sign-off it denotes is the owner's approval of the pending `initial_dag` review for this cohort, **review `2fa5a4a5-4035-4511-9fa1-1aedf1bd6645`** (`author_real_review/optum_biologic_initiation_initiated_biologic_180d/review.md`, the same 110 fragments as this table).

- source: `docs/demos/results/2026-09-23_lane_b_real_runs/author_real_review/optum_biologic_initiation_initiated_biologic_180d/manifest_diff.json` + `attestations.json`; author model `openai/gpt-5.6-terra`; tree `{"commit": "c67f2e5928fad81af3c8dec3ddcfe62174baef00", "dirty_src_scripts_tests": false}`
- compared: 110 attested features; the real author reproduces the manifest's edge set AND role on **93** (candidates for `machine_reviewed` IF the owner approves review 2fa5a4a5 after reading their rows); disagrees on **17** (stay `machine` whatever the review says — the review would approve the AUTHOR's fragment, not the manifest's)
- edge-exact agreement 93/110, role agreement 93/110
- manifest-side research grounding (a bullet of `docs/layer4/optum_initiation_attestation_research.md` that names the feature): present on 22/93 agreement rows and 17/17 disagreement rows; on the other 71 agreement rows the manifest side is ONLY the `_optum_attestation` edge pattern in code — those rows have one rationale (the author's), not two, and are marked as such
- authored edge grades over all non-estimand edges: {'unsupported': 218} — every authored edge is `unsupported` by the citation grader (abstracts resolved but no entity match); the author's rationale is prose, not verified citation
- disagreement role pairs (manifest → authored): {('instrument', 'confounder'): 15, ('instrument', 'descendant'): 2}
- authored records flagged ambiguous by the author itself: 73 (its stated reason: T and Y are overlapping initiation constructs); review_required: 0

Rule applied: a row is a `machine_reviewed` CANDIDATE only where the author's fragment reproduces the manifest's edge set exactly AND derives the same role. The candidate becomes `machine_reviewed` (with `review_id = 2fa5a4a5…` on the record) only if the owner approves that review; until then every row stays `machine` (audit-only). A disagreement is NOT a verdict against either side; it is the list the owner reads first.

## Disagreements (owner reads these first)

| feature | manifest edges → role | authored edges → role | edge/role agree | authored rationale | manifest rationale (research doc) | current → recommended |
|---|---|---|---|---|---|---|
| `zip5` | `biologic_initiation->initiated_biologic_180d`, `zip5->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `zip5->biologic_initiation`, `zip5->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | zip5->T [unsupported 10.1016/j.ypmed.2004.11.005✗]: Residential ZIP code captures geographic access conditions, including spatial availability of relevant clinicians and travel/access barriers, which can affect whether a patient initiates a biologic at the index date. ‖ zip5->Y [unsupported 10.1016/j.ypmed.2004.11.005… | L59: - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access). | `machine` → `machine` |
| `zip3` | `biologic_initiation->initiated_biologic_180d`, `zip3->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `zip3->biologic_initiation`, `zip3->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | zip3->T [unsupported 10.1186/1476-072X-3-3✗,10.1097/00005650-199501000-00004✗]: Residential geographic area influences health-care access, including availability and accessibility of clinicians and services that can evaluate patients and initiate specialty/biologic treatment. ZIP3 is a measured area-level location mar… | L59: - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access). | `machine` → `machine` |
| `zip_code` | `biologic_initiation->initiated_biologic_180d`, `zip_code->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `zip_code->biologic_initiation`, `zip_code->initiated_biologic_180d` → **confounder** | no / no | zip_code->T [unsupported PMID:22250236✓,DOI:10.1001/archdermatol.2011.318✗]: Residential ZIP code captures geographic access to specialty care, including local specialist supply and travel burden. Geographic access affects whether a patient can receive an index biologic-initiation decision. ‖ zip_code->Y [unsupported … | L59: - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access). | `machine` → `machine` |
| `geographic_region` | `biologic_initiation->initiated_biologic_180d`, `geographic_region->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `geographic_region->biologic_initiation`, `geographic_region->initiated_biologic_180d` → **confounder** | no / no | geographic_region->T [unsupported 10.1056/NEJMsa022921✗]: Patient geographic region influences biologic initiation through geographic variation in clinician supply and specialty access, local prescribing practice, and payer or formulary policies. These regional health-system differences can change the likelihood of re… | L59: - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access). | `machine` → `machine` |
| `insurance_product` | `biologic_initiation->initiated_biologic_180d`, `insurance_product->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `insurance_product->biologic_initiation`, `insurance_product->initiated_biologic_180d` → **confounder** | no / no | insurance_product->T [unsupported 10.1001/jamanetworkopen.2018.1691✗]: Insurance product causally influences biologic initiation through formulary design, specialty-drug coverage, prior authorization, network restrictions, and patient cost sharing. ‖ insurance_product->Y [unsupported 10.1001/jamanetworkopen.2018.1691✗… | L60: - Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities). | `machine` → `machine` |
| `plan_type` | `biologic_initiation->initiated_biologic_180d`, `plan_type->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `plan_type->biologic_initiation`, `plan_type->initiated_biologic_180d` → **confounder** | no / no | plan_type->T [unsupported 10.1001/jama.291.19.2344✗]: Insurance plan type determines benefit design and access restrictions (including cost sharing and coverage rules), which can causally affect whether a patient initiates a biologic at the index date. ‖ plan_type->Y [unsupported 10.1001/jama.291.19.2344✗]: The same p… | L60: - Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities). | `machine` → `machine` |
| `payer_category` | `biologic_initiation->initiated_biologic_180d`, `payer_category->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `payer_category->biologic_initiation`, `payer_category->initiated_biologic_180d` → **confounder** | no / no | payer_category->T [unsupported 10.1001/jama.291.19.2344✗]: Payer category determines drug-benefit design, formulary coverage, utilization management, and out-of-pocket cost. These access constraints can causally affect whether a patient initiates a biologic. ‖ payer_category->Y [unsupported 10.1001/jama.291.19.2344✗]:… | L60: - Coverage/payer — `insurance_product`, `plan_type`, `payer_category`: prior-auth / step-therapy / formulary gate initiation (Aetna CPB 0670; Cigna & UHC Xolair PA policies; PMID 40004611 cost/access disparities). | `machine` → `machine` |
| `urban_rural_code` | `biologic_initiation->initiated_biologic_180d`, `urban_rural_code->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `urban_rural_code->biologic_initiation`, `urban_rural_code->initiated_biologic_180d` → **confounder** | no / no | urban_rural_code->T [unsupported PMID:28252652✓]: Urban versus rural residence affects geographic access to health care and specialty services, which can affect the opportunity and timing of biologic initiation at the index date. ‖ urban_rural_code->Y [unsupported PMID:28252652✓]: The same rural-urban differences in a… | L59: - Geography/residence — `zip5`, `zip3`, `zip_code`, `geographic_region`, `urban_rural_code`: specialist proximity & regional adoption shift omalizumab use (PMID 36481046 — use varied by patient ZIP, higher nearer the allergist practice; PMID 40169378 — BRIT registry, deprivation/distance affect access). | `machine` → `machine` |
| `office_visits_allergist` | `biologic_initiation->initiated_biologic_180d`, `office_visits_allergist->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `office_visits_allergist->biologic_initiation`, `office_visits_allergist->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | office_visits_allergist->T [unsupported DOI:10.1111/all.15090✓]: Pre-index allergist encounters provide direct opportunities for specialist evaluation and treatment escalation, including prescribing or arranging biologic therapy. International chronic urticaria guidance describes escalation to biologic treatment as a … | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |
| `office_visits_dermatology` | `biologic_initiation->initiated_biologic_180d`, `office_visits_dermatology->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `office_visits_dermatology->biologic_initiation`, `office_visits_dermatology->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | office_visits_dermatology->T [unsupported DOI:10.1111/all.15090✓]: Pre-index dermatology visits represent opportunities for specialist assessment of persistent or uncontrolled disease and for implementation of guideline-recommended escalation, including biologic treatment. ‖ office_visits_dermatology->Y [unsupported D… | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |
| `specialist_concentration` | `biologic_initiation->initiated_biologic_180d`, `specialist_concentration->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `specialist_concentration->biologic_initiation`, `specialist_concentration->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | specialist_concentration->T [unsupported PMID:33957209✓]: A higher concentration of care from relevant specialists represents greater specialist involvement, which can causally affect whether a patient initiates a biologic through specialist assessment, referral, prescribing, and access to advanced therapy. ‖ speciali… | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |
| `primary_specialist_type` | `biologic_initiation->initiated_biologic_180d`, `primary_specialist_type->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `primary_specialist_type->biologic_initiation`, `primary_specialist_type->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | primary_specialist_type->T [unsupported]: The type of specialist managing the patient can directly influence whether biologic therapy is initiated at the index date through specialty-specific prescribing practices, treatment expertise, referral patterns, and access/authorization workflows. ‖ primary_specialist_type->Y… | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |
| `saw_allergist_flag` | `biologic_initiation->initiated_biologic_180d`, `saw_allergist_flag->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `saw_allergist_flag->biologic_initiation`, `saw_allergist_flag->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | saw_allergist_flag->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Allergist involvement can causally affect biologic initiation: allergy specialists assess chronic urticaria severity and inadequate response to second-generation H1-antihistamines and implement guideline-recommended escalation to biologic thera… | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |
| `saw_dermatologist_flag` | `biologic_initiation->initiated_biologic_180d`, `saw_dermatologist_flag->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `saw_dermatologist_flag->biologic_initiation`, `saw_dermatologist_flag->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | saw_dermatologist_flag->T [unsupported 10.1016/j.jaad.2018.11.057✗,10.1111/all.15214✓]: Dermatologist involvement can causally influence biologic initiation through specialist assessment of inflammatory skin disease, selection of systemic therapy, prescribing authority, and management of biologic eligibility and monit… | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |
| `index_date` | `biologic_initiation->initiated_biologic_180d`, `index_date->biologic_initiation` → **instrument** | `biologic_initiation->index_date`, `biologic_initiation->initiated_biologic_180d` → **descendant** (ambiguous) | no / no | T->index_date [unsupported]: The feature is a derived index/anchor timestamp. With biologic initiation specified as the treatment and index_date specified as the prediction anchor, the initiation event operationally defines the cohort index date; this is an administrative derivation rather than a patient-level prognos… | L62: - Calendar anchors — `index_date`, `lookback_start_date`: temporal adoption of biologics (PMID 32382379; PMID 40004611). | `machine` → `machine` |
| `lookback_start_date` | `biologic_initiation->initiated_biologic_180d`, `lookback_start_date->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `biologic_initiation->lookback_start_date` → **descendant** (ambiguous) | no / no | T->lookback_start_date [unsupported]: The feature is derived solely from index_date, and the index/anchor is the biologic-initiation event. The lookback-start timestamp is therefore an operational field constructed once the initiation-indexed record is created; it is not a patient-level cause of treatment or outcome. | L62: - Calendar anchors — `index_date`, `lookback_start_date`: temporal adoption of biologics (PMID 32382379; PMID 40004611). | `machine` → `machine` |
| `specialist_visit_interaction` | `biologic_initiation->initiated_biologic_180d`, `specialist_visit_interaction->biologic_initiation` → **instrument** | `biologic_initiation->initiated_biologic_180d`, `specialist_visit_interaction->biologic_initiation`, `specialist_visit_interaction->initiated_biologic_180d` → **confounder** (ambiguous) | no / no | specialist_visit_interaction->T [unsupported 10.2307/2137284✗]: Greater involvement with relevant specialists can causally increase the opportunity for assessment, referral, recommendation, and prescribing of a biologic, thereby affecting biologic initiation. ‖ specialist_visit_interaction->Y [unsupported 10.2307/2137… | L61: - Specialist access — `office_visits_allergist`, `office_visits_dermatology`, `specialist_concentration`, `primary_specialist_type`, `saw_allergist_flag`, `saw_dermatologist_flag`, `specialist_visit_interaction`: allergist/derm contact is the biologic-prescribing channel (PMID 36481046; PMID 33528934). | `machine` → `machine` |

## Agreements (candidates for `machine_reviewed` — only upon the owner's approval of review 2fa5a4a5)

| feature | manifest edges → role | authored edges → role | edge/role agree | authored rationale | manifest rationale (research doc) | current → recommended |
|---|---|---|---|---|---|---|
| `age_at_index` | `age_at_index->biologic_initiation`, `age_at_index->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `age_at_index->biologic_initiation`, `age_at_index->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | age_at_index->T [unsupported PMID:24472253✓]: Age is a pre-treatment determinant of prescribing decisions because it is associated with comorbidity burden, treatment safety considerations, and patient/clinician preferences that affect whether advanced therapies such as biologics are initiated. ‖ age_at_index->Y [unsup… | L65: - Demographics — `age_at_index`, `age_group`, `gender`, `primary_diagnosis_code` (PMID 34622498; PMID 32382379). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `age_group` | `age_group->biologic_initiation`, `age_group->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `age_group->biologic_initiation`, `age_group->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | age_group->T [unsupported]: Age at enrollment/index precedes treatment and affects biologic-initiation decisions through clinical eligibility, age-related comorbidity and safety assessment, and clinician prescribing behavior. ‖ age_group->Y [unsupported]: Age also affects the likelihood and timing of biologic initiati… | L65: - Demographics — `age_at_index`, `age_group`, `gender`, `primary_diagnosis_code` (PMID 34622498; PMID 32382379). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `gender` | `biologic_initiation->initiated_biologic_180d`, `gender->biologic_initiation`, `gender->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `gender->biologic_initiation`, `gender->initiated_biologic_180d` → **confounder** | yes / yes | gender->T [unsupported DOI:10.1016/S0140-6736(19)30250-4✗]: Gender-related differences in disease presentation, care seeking, access, clinician decision-making, and treatment utilization can affect whether a patient initiates a biologic at the index date. ‖ gender->Y [unsupported DOI:10.1016/S0140-6736(19)30250-4✗]: T… | L65: - Demographics — `age_at_index`, `age_group`, `gender`, `primary_diagnosis_code` (PMID 34622498; PMID 32382379). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `primary_diagnosis_code` | `biologic_initiation->initiated_biologic_180d`, `primary_diagnosis_code->biologic_initiation`, `primary_diagnosis_code->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `primary_diagnosis_code->biologic_initiation`, `primary_diagnosis_code->initiated_biologic_180d` → **confounder** | yes / yes | primary_diagnosis_code->T [unsupported]: The enrollment primary diagnosis encodes the condition/indication under management. Biologic prescribing is indication-specific, so the diagnosis influences whether a patient is initiated on a biologic at the index date. ‖ primary_diagnosis_code->Y [unsupported]: The underlying… | L65: - Demographics — `age_at_index`, `age_group`, `gender`, `primary_diagnosis_code` (PMID 34622498; PMID 32382379). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `dx_l50_1_count` | `biologic_initiation->initiated_biologic_180d`, `dx_l50_1_count->biologic_initiation`, `dx_l50_1_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `dx_l50_1_count->biologic_initiation`, `dx_l50_1_count->initiated_biologic_180d` → **confounder** | yes / yes | dx_l50_1_count->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: A higher recent count of idiopathic-urticaria diagnoses denotes recurrent/active chronic spontaneous urticaria care. Disease activity and inadequate control are clinical reasons to escalate to biologic therapy, including omalizumab, and thus can in… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `dx_l50_1_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `dx_l50_8_count` | `biologic_initiation->initiated_biologic_180d`, `dx_l50_8_count->biologic_initiation`, `dx_l50_8_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `dx_l50_8_count->biologic_initiation`, `dx_l50_8_count->initiated_biologic_180d` → **confounder** | yes / yes | dx_l50_8_count->T [unsupported 10.1111/all.15090✓]: A higher pre-index frequency of urticaria diagnoses represents recurrent/persistent urticaria requiring care. Persistent symptoms despite standard management are a reason to escalate chronic urticaria treatment to biologic therapy. ‖ dx_l50_8_count->Y [unsupported 10… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `dx_l50_8_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `dx_l50_9_count` | `biologic_initiation->initiated_biologic_180d`, `dx_l50_9_count->biologic_initiation`, `dx_l50_9_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `dx_l50_9_count->biologic_initiation`, `dx_l50_9_count->initiated_biologic_180d` → **confounder** | yes / yes | dx_l50_9_count->T [unsupported PMID: 34536239✓]: A higher recent count of urticaria-coded encounters represents recurrent or insufficiently controlled urticaria requiring ongoing clinical attention. In chronic urticaria, inadequate disease control is a clinical reason to escalate treatment to biologic therapy. ‖ dx_l5… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `dx_l50_9_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `dx_total_csu` | `biologic_initiation->initiated_biologic_180d`, `dx_total_csu->biologic_initiation`, `dx_total_csu->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `dx_total_csu->biologic_initiation`, `dx_total_csu->initiated_biologic_180d` → **confounder** | yes / yes | dx_total_csu->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: A documented history of chronic spontaneous urticaria establishes the clinical indication for escalation to biologic therapy when symptoms are inadequately controlled with antihistamines; therefore recognized CSU causally informs the biologic-initiat… | L67: - CSU dx burden — `dx_l50_1/8/9_count`, `dx_total_csu`, `dx_angioedema_count`, `csu_dx_intensity` (PMID 39325444; PMID 34984792). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `dx_angioedema_count` | `biologic_initiation->initiated_biologic_180d`, `dx_angioedema_count->biologic_initiation`, `dx_angioedema_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `dx_angioedema_count->biologic_initiation`, `dx_angioedema_count->initiated_biologic_180d` → **confounder** | yes / yes | dx_angioedema_count->T [unsupported DOI:10.1111/all.15090✓]: A recent history of angioedema reflects clinically significant chronic urticaria activity/phenotype and is relevant to escalation of therapy, including biologic treatment, when disease is inadequately controlled with standard therapy. ‖ dx_angioedema_count->… | L67: - CSU dx burden — `dx_l50_1/8/9_count`, `dx_total_csu`, `dx_angioedema_count`, `csu_dx_intensity` (PMID 39325444; PMID 34984792). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `months_since_first_dx` | `biologic_initiation->initiated_biologic_180d`, `months_since_first_dx->biologic_initiation`, `months_since_first_dx->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `months_since_first_dx->biologic_initiation`, `months_since_first_dx->initiated_biologic_180d` → **confounder** | yes / yes | months_since_first_dx->T [unsupported PMID: 34509334✓]: Pre-index time since diagnosis represents disease duration. In chronic inflammatory disease management, longer-standing disease provides opportunity to document persistence, recurrence, inadequate response to prior therapies, and progression through treatment ste… | L69: - Disease duration/chronicity — `months_since_first_dx`, `csu_chronicity` (authored to the true clinical meaning; see caveat). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `csu_chronicity` | `biologic_initiation->initiated_biologic_180d`, `csu_chronicity->biologic_initiation`, `csu_chronicity->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `csu_chronicity->biologic_initiation`, `csu_chronicity->initiated_biologic_180d` → **confounder** | yes / yes | csu_chronicity->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Chronic spontaneous urticaria that persists despite standard management is a clinical reason to escalate therapy, including to biologic treatment. International CSU guidance recommends stepwise escalation for patients with ongoing disease activity/… | L69: - Disease duration/chronicity — `months_since_first_dx`, `csu_chronicity` (authored to the true clinical meaning; see caveat). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_atopic_dermatitis` | `biologic_initiation->initiated_biologic_180d`, `has_atopic_dermatitis->biologic_initiation`, `has_atopic_dermatitis->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_atopic_dermatitis->biologic_initiation`, `has_atopic_dermatitis->initiated_biologic_180d` → **confounder** | yes / yes | has_atopic_dermatitis->T [unsupported PMID:27690741✓,DOI:10.1056/NEJMoa1610020✗]: Atopic dermatitis is itself an established indication for biologic treatment; therefore, a documented diagnosis can causally affect clinician treatment selection and biologic initiation. ‖ has_atopic_dermatitis->Y [unsupported PMID:27690… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_atopic_dermatitis`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `atopic_dermatitis_claim_count` | `atopic_dermatitis_claim_count->biologic_initiation`, `atopic_dermatitis_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `atopic_dermatitis_claim_count->biologic_initiation`, `atopic_dermatitis_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | atopic_dermatitis_claim_count->T [unsupported 10.1016/j.jaad.2023.01.029✗]: A greater pre-index burden of documented atopic dermatitis reflects clinically recognized disease for which systemic biologic treatment is considered, particularly when disease is moderate-to-severe or inadequately controlled with topical ther… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `atopic_dermatitis_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_asthma` | `biologic_initiation->initiated_biologic_180d`, `has_asthma->biologic_initiation`, `has_asthma->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_asthma->biologic_initiation`, `has_asthma->initiated_biologic_180d` → **confounder** | yes / yes | has_asthma->T [unsupported DOI:10.1183/13993003.01658-2019✗]: Asthma, particularly severe or uncontrolled asthma, is an indication for biologic therapy; thus an asthma diagnosis can causally affect a clinician's decision to initiate a biologic. ‖ has_asthma->Y [unsupported DOI:10.1183/13993003.01658-2019✗]: Because th… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_asthma`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `asthma_claim_count` | `asthma_claim_count->biologic_initiation`, `asthma_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `asthma_claim_count->biologic_initiation`, `asthma_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | asthma_claim_count->T [unsupported PMID:34658302✓]: A greater pre-index burden of asthma-related claims represents clinically recognized asthma activity and/or severity. Severe asthma is an established setting in which biologic therapies are considered and initiated, so this burden can affect index biologic-initiation… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `asthma_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_allergic_rhinitis` | `biologic_initiation->initiated_biologic_180d`, `has_allergic_rhinitis->biologic_initiation`, `has_allergic_rhinitis->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_allergic_rhinitis->biologic_initiation`, `has_allergic_rhinitis->initiated_biologic_180d` → **confounder** | yes / yes | has_allergic_rhinitis->T [unsupported DOI:10.1111/all.14010✓]: Allergic rhinitis is an atopic comorbidity relevant to allergic-airway disease assessment and treatment selection. Its documented presence can increase clinical impetus to initiate a biologic at the index encounter. ‖ has_allergic_rhinitis->Y [unsupported … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_allergic_rhinitis`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `allergic_rhinitis_claim_count` | `allergic_rhinitis_claim_count->biologic_initiation`, `allergic_rhinitis_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `allergic_rhinitis_claim_count->biologic_initiation`, `allergic_rhinitis_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | allergic_rhinitis_claim_count->T [unsupported 10.1016/j.jaci.2020.07.007✗]: A pre-index history of recurrent allergic rhinitis documents allergic upper-airway disease and can inform clinician assessment of allergic/type-2 disease burden, specialty management, and selection of biologic therapy. ‖ allergic_rhinitis_clai… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `allergic_rhinitis_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_anxiety` | `biologic_initiation->initiated_biologic_180d`, `has_anxiety->biologic_initiation`, `has_anxiety->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_anxiety->biologic_initiation`, `has_anxiety->initiated_biologic_180d` → **confounder** | yes / yes | has_anxiety->T [unsupported PMID: 23044370✓]: Baseline anxiety can affect care seeking, patient preferences and acceptance of treatment, and clinician treatment decision-making; consequently it can affect whether biologic treatment is initiated. ‖ has_anxiety->Y [unsupported PMID: 23044370✓]: Because anxiety affects e… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_anxiety`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `anxiety_claim_count` | `anxiety_claim_count->biologic_initiation`, `anxiety_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `anxiety_claim_count->biologic_initiation`, `anxiety_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | anxiety_claim_count->T [unsupported PMID:10335673✓]: Pre-index anxiety burden can influence treatment decisions through patient treatment preferences, capacity to engage in care, and clinician assessment of treatment suitability. The anxiety-claim count operationalizes documented anxiety burden available at the treatm… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `anxiety_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_depression` | `biologic_initiation->initiated_biologic_180d`, `has_depression->biologic_initiation`, `has_depression->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_depression->biologic_initiation`, `has_depression->initiated_biologic_180d` → **confounder** | yes / yes | has_depression->T [unsupported PMID: 19454074✓]: Pre-existing depression can causally influence treatment-selection and prescribing decisions through its effects on anticipated adherence, care engagement, and the practical management of chronic disease therapy. Depression is consistently associated with medication non… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_depression`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `depression_claim_count` | `biologic_initiation->initiated_biologic_180d`, `depression_claim_count->biologic_initiation`, `depression_claim_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `depression_claim_count->biologic_initiation`, `depression_claim_count->initiated_biologic_180d` → **confounder** | yes / yes | depression_claim_count->T [unsupported PMID:17134623✓]: A higher pre-index burden of depression can alter engagement with healthcare, patient acceptance of treatment, and clinician treatment decision-making, thereby affecting initiation of a biologic. ‖ depression_claim_count->Y [unsupported PMID:17134623✓]: Depressio… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `depression_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_thyroid_autoimmune` | `biologic_initiation->initiated_biologic_180d`, `has_thyroid_autoimmune->biologic_initiation`, `has_thyroid_autoimmune->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_thyroid_autoimmune->biologic_initiation`, `has_thyroid_autoimmune->initiated_biologic_180d` → **confounder** | yes / yes | has_thyroid_autoimmune->T [unsupported 10.1371/journal.pone.0126829✗,10.1111/all.15090✓]: Autoimmune thyroid disease identifies an autoimmune comorbidity phenotype associated with chronic spontaneous urticaria. In clinical practice, comorbidity and evidence of difficult-to-control disease contribute to assessment and … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_thyroid_autoimmune`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `thyroid_autoimmune_claim_count` | `biologic_initiation->initiated_biologic_180d`, `thyroid_autoimmune_claim_count->biologic_initiation`, `thyroid_autoimmune_claim_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `thyroid_autoimmune_claim_count->biologic_initiation`, `thyroid_autoimmune_claim_count->initiated_biologic_180d` → **confounder** | yes / yes | thyroid_autoimmune_claim_count->T [unsupported PMID:28407273✓,DOI:10.1111/all.15090✓]: Pre-index documented autoimmune thyroid disease burden is clinically available when treatment escalation is considered. Autoimmune thyroid disease is a recognized comorbidity and autoimmune phenotype marker in chronic spontaneous ur… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `thyroid_autoimmune_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_nsaid_hypersensitivity` | `biologic_initiation->initiated_biologic_180d`, `has_nsaid_hypersensitivity->biologic_initiation`, `has_nsaid_hypersensitivity->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_nsaid_hypersensitivity->biologic_initiation`, `has_nsaid_hypersensitivity->initiated_biologic_180d` → **confounder** | yes / yes | has_nsaid_hypersensitivity->T [unsupported PMID:34536239✓,PMID:17210064✓]: NSAID-exacerbated cutaneous disease is a recognized chronic spontaneous urticaria phenotype in which NSAID exposure can worsen urticaria. This additional trigger and more difficult symptom control can contribute to escalation decisions, includi… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_nsaid_hypersensitivity`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `nsaid_hypersensitivity_claim_count` | `biologic_initiation->initiated_biologic_180d`, `nsaid_hypersensitivity_claim_count->biologic_initiation`, `nsaid_hypersensitivity_claim_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `nsaid_hypersensitivity_claim_count->biologic_initiation`, `nsaid_hypersensitivity_claim_count->initiated_biologic_180d` → **confounder** | yes / yes | nsaid_hypersensitivity_claim_count->T [unsupported PMID: 26967389✓,DOI: 10.1111/all.14009✓]: Pre-index documented NSAID hypersensitivity identifies a clinically relevant chronic-urticaria phenotype. NSAID-exacerbated cutaneous disease occurs in patients with chronic spontaneous urticaria; ongoing uncontrolled disease … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `nsaid_hypersensitivity_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `has_angioedema` | `biologic_initiation->initiated_biologic_180d`, `has_angioedema->biologic_initiation`, `has_angioedema->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `has_angioedema->biologic_initiation`, `has_angioedema->initiated_biologic_180d` → **confounder** | yes / yes | has_angioedema->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: At index, angioedema is a clinically important manifestation of chronic urticaria burden. Greater burden and uncontrolled symptoms inform escalation to guideline-directed therapies, including biologic treatment; thus documented angioedema can causa… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `has_angioedema`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `angioedema_claim_count` | `angioedema_claim_count->biologic_initiation`, `angioedema_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `angioedema_claim_count->biologic_initiation`, `angioedema_claim_count->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | angioedema_claim_count->T [unsupported DOI:10.1111/all.15090✓]: A higher pre-index burden of documented angioedema represents clinically important, insufficiently controlled urticaria/angioedema activity and can prompt treatment escalation to a biologic. ‖ angioedema_claim_count->Y [unsupported DOI:10.1111/all.15090✓]… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `angioedema_claim_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `atopy_score` | `atopy_score->biologic_initiation`, `atopy_score->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `atopy_score->biologic_initiation`, `atopy_score->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | atopy_score->T [unsupported 10.1056/NEJMoa1610020✗,10.1056/NEJMoa1804092✗]: Baseline atopic disease burden (atopic dermatitis, asthma, and allergic rhinitis) identifies an allergic/type-2 phenotype for which biologic therapies may be clinically indicated; documented atopic comorbidity can therefore affect the decision… | L68: - Comorbidities — atopic dermatitis / asthma / allergic rhinitis / thyroid-autoimmune / angioedema / nsaid-hypersensitivity / anxiety / depression families (`has_*` + `*_claim_count`), `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score`, `comorbidity_load_total` (PMID 42050840; PMID 40220911… | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `mental_health_flag` | `biologic_initiation->initiated_biologic_180d`, `mental_health_flag->biologic_initiation`, `mental_health_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `mental_health_flag->biologic_initiation`, `mental_health_flag->initiated_biologic_180d` → **confounder** | yes / yes | mental_health_flag->T [unsupported PMID:15008660✓]: Documented anxiety or depression can influence treatment selection and initiation through patient preferences, treatment decision-making, access to care, and clinician assessment of treatment feasibility. Therefore baseline mental-health status can causally affect in… | L68: - Comorbidities — atopic dermatitis / asthma / allergic rhinitis / thyroid-autoimmune / angioedema / nsaid-hypersensitivity / anxiety / depression families (`has_*` + `*_claim_count`), `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score`, `comorbidity_load_total` (PMID 42050840; PMID 40220911… | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `elixhauser_score` | `biologic_initiation->initiated_biologic_180d`, `elixhauser_score->biologic_initiation`, `elixhauser_score->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `elixhauser_score->biologic_initiation`, `elixhauser_score->initiated_biologic_180d` → **confounder** | yes / yes | elixhauser_score->T [unsupported PMID:9431328✓,DOI:10.1016/S0895-4356(98)00100-7✗]: The Elixhauser score summarizes baseline comorbid disease burden. Comorbidity burden is available to clinicians at treatment selection and can causally influence biologic initiation through treatment risk, contraindications, competing … | L68: - Comorbidities — atopic dermatitis / asthma / allergic rhinitis / thyroid-autoimmune / angioedema / nsaid-hypersensitivity / anxiety / depression families (`has_*` + `*_claim_count`), `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score`, `comorbidity_load_total` (PMID 42050840; PMID 40220911… | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `charlson_score` | `biologic_initiation->initiated_biologic_180d`, `charlson_score->biologic_initiation`, `charlson_score->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `charlson_score->biologic_initiation`, `charlson_score->initiated_biologic_180d` → **confounder** | yes / yes | charlson_score->T [unsupported 10.1016/0021-9681(87)90171-8✗]: Charlson score summarizes clinically important baseline comorbidity. Such comorbidity is available to clinicians at treatment selection and can alter biologic-initiation decisions through safety considerations, competing illness, and treatment eligibility.… | L68: - Comorbidities — atopic dermatitis / asthma / allergic rhinitis / thyroid-autoimmune / angioedema / nsaid-hypersensitivity / anxiety / depression families (`has_*` + `*_claim_count`), `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score`, `comorbidity_load_total` (PMID 42050840; PMID 40220911… | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `office_visits_total` | `biologic_initiation->initiated_biologic_180d`, `office_visits_total->biologic_initiation`, `office_visits_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `office_visits_total->biologic_initiation`, `office_visits_total->initiated_biologic_180d` → **confounder** | yes / yes | office_visits_total->T [unsupported 10.2307/2137284✗]: A greater number of pre-index office encounters provides opportunities for clinician assessment, referral, discussion, and prescribing of a biologic; health-service use is determined in part by access to and contact with the health-care system. ‖ office_visits_tot… | L66: - Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `office_visits_pcp` | `biologic_initiation->initiated_biologic_180d`, `office_visits_pcp->biologic_initiation`, `office_visits_pcp->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `office_visits_pcp->biologic_initiation`, `office_visits_pcp->initiated_biologic_180d` → **confounder** | yes / yes | office_visits_pcp->T [unsupported]: Pre-index primary-care encounters provide opportunities for clinical assessment, documentation, referral, and escalation to biologic therapy; consequently, greater PCP utilization can causally increase the probability of biologic initiation. ‖ office_visits_pcp->Y [unsupported]: Bec… | L66: - Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ed_visits_total` | `biologic_initiation->initiated_biologic_180d`, `ed_visits_total->biologic_initiation`, `ed_visits_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ed_visits_total->biologic_initiation`, `ed_visits_total->initiated_biologic_180d` → **confounder** | yes / yes | ed_visits_total->T [unsupported PMID:23664752✓]: Pre-index ED utilization represents acute illness burden and unscheduled-care need. Such utilization supplies clinical evidence prompting treatment escalation, including initiation of advanced/biologic therapy. ‖ ed_visits_total->Y [unsupported PMID:23664752✓]: Prior ED… | L66: - Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ed_visits_urticaria_angio` | `biologic_initiation->initiated_biologic_180d`, `ed_visits_urticaria_angio->biologic_initiation`, `ed_visits_urticaria_angio->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ed_visits_urticaria_angio->biologic_initiation`, `ed_visits_urticaria_angio->initiated_biologic_180d` → **confounder** | yes / yes | ed_visits_urticaria_angio->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Recent emergency-department encounters for urticaria/angioedema represent uncontrolled, high-burden disease requiring acute care. Disease control, activity, and impact are central determinants of escalation to guideline-directed therapie… | L66: - Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `hospitalizations_total` | `biologic_initiation->initiated_biologic_180d`, `hospitalizations_total->biologic_initiation`, `hospitalizations_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `hospitalizations_total->biologic_initiation`, `hospitalizations_total->initiated_biologic_180d` → **confounder** | yes / yes | hospitalizations_total->T [unsupported DOI:10.1001/jama.2011.1515✗]: Hospitalizations before the index date can alter clinician treatment decisions by documenting acute disease burden, creating specialist follow-up, and prompting therapeutic escalation, including biologic initiation. ‖ hospitalizations_total->Y [unsup… | L66: - Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `unique_providers` | `biologic_initiation->initiated_biologic_180d`, `unique_providers->biologic_initiation`, `unique_providers->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `unique_providers->biologic_initiation`, `unique_providers->initiated_biologic_180d` → **confounder** | yes / yes | unique_providers->T [unsupported PMID: 29772242✓]: A greater number of pre-index providers creates more clinical encounters and referral pathways through which patients can be evaluated by clinicians able to prescribe or initiate a biologic; care fragmentation and transitions across clinicians are recognized determina… | L66: - Utilization severity proxies — `office_visits_total`, `office_visits_pcp`, `ed_visits_total`, `ed_visits_urticaria_angio`, `hospitalizations_total`, `unique_providers` (PMID 29429043; PMID 34622498). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_1g_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `h1_1g_ever_filled->biologic_initiation`, `h1_1g_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_1g_ever_filled->biologic_initiation`, `h1_1g_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | h1_1g_ever_filled->T [unsupported PMID:34536239✓]: A prior first-generation H1-antihistamine fill represents antihistamine treatment history and can affect clinician escalation to biologic therapy when symptom control is inadequate or prior conventional management has been attempted. International chronic urticaria gu… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_1g_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_1g_fill_count` | `biologic_initiation->initiated_biologic_180d`, `h1_1g_fill_count->biologic_initiation`, `h1_1g_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_1g_fill_count->biologic_initiation`, `h1_1g_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | h1_1g_fill_count->T [unsupported DOI:10.1111/all.15090✓,PMID:34536239✓]: Pre-index H1-antihistamine fill burden reflects prior antihistamine treatment and clinical need. Chronic urticaria management uses second-generation H1-antihistamines first and recommends escalation to omalizumab/other advanced therapy when sympt… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_1g_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_1g_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `h1_1g_days_supply_total->biologic_initiation`, `h1_1g_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_1g_days_supply_total->biologic_initiation`, `h1_1g_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | h1_1g_days_supply_total->T [unsupported PMID:29336054✓,DOI:10.1111/all.13397✓]: Pre-index H1-antihistamine treatment history is part of the clinical treatment-escalation pathway: chronic urticaria uncontrolled on standard- or increased-dose second-generation H1-antihistamines is an indication to escalate to biologic t… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_1g_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_1g_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `h1_1g_days_since_last_fill->biologic_initiation`, `h1_1g_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_1g_days_since_last_fill->biologic_initiation`, `h1_1g_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | h1_1g_days_since_last_fill->T [unsupported PMID:29336008✗,PMID:34536239✓]: Recency of H1-antihistamine dispensing represents current/recent antihistamine treatment history available at the prescribing encounter. In chronic urticaria, escalation to biologic therapy is guided by inadequate control despite second-generat… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_1g_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_2g_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `h1_2g_ever_filled->biologic_initiation`, `h1_2g_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_2g_ever_filled->biologic_initiation`, `h1_2g_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | h1_2g_ever_filled->T [unsupported PMID: 34536239✓]: A prior fill of a second-generation H1 antihistamine represents antecedent first-line pharmacologic management. In chronic spontaneous urticaria, inadequate control despite second-generation H1-antihistamine treatment is a clinical basis for escalation to biologic tr… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_2g_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_2g_fill_count` | `biologic_initiation->initiated_biologic_180d`, `h1_2g_fill_count->biologic_initiation`, `h1_2g_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_2g_fill_count->biologic_initiation`, `h1_2g_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | h1_2g_fill_count->T [unsupported PMID: 34536239✓]: A greater pre-index burden of second-generation H1-antihistamine dispensing represents more intensive recent guideline-directed urticaria management and evidence of treatment escalation/insufficient control, which informs the decision to initiate biologic therapy. ‖ h… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_2g_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_2g_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `h1_2g_days_supply_total->biologic_initiation`, `h1_2g_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_2g_days_supply_total->biologic_initiation`, `h1_2g_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | h1_2g_days_supply_total->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Second-generation H1-antihistamines are first-line therapy for chronic urticaria, and escalation to omalizumab/other advanced treatment is recommended when symptoms remain inadequately controlled despite antihistamine treatment. Thus, grea… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_2g_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h1_2g_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `h1_2g_days_since_last_fill->biologic_initiation`, `h1_2g_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h1_2g_days_since_last_fill->biologic_initiation`, `h1_2g_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | h1_2g_days_since_last_fill->T [unsupported PMID: 34536239✓,DOI: 10.1111/all.15090✓]: Recency of second-generation H1-antihistamine dispensing represents recent standard-of-care antihistamine treatment history. Persistent symptoms or inadequate control despite such treatment is a clinical reason to escalate chronic urt… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h1_2g_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h2_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `h2_ever_filled->biologic_initiation`, `h2_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h2_ever_filled->biologic_initiation`, `h2_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | h2_ever_filled->T [unsupported DOI:10.1111/all.13397✓]: A recent pre-index H2-receptor antagonist fill records prior antihistamine-based management. In chronic urticaria, pharmacologic therapy is escalated in a stepwise manner for inadequately controlled disease; prior adjunctive antihistamine treatment is therefore c… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h2_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h2_fill_count` | `biologic_initiation->initiated_biologic_180d`, `h2_fill_count->biologic_initiation`, `h2_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h2_fill_count->biologic_initiation`, `h2_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | h2_fill_count->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: A greater pre-index history of H2-receptor-antagonist dispensing can represent prior antihistamine treatment attempts and incomplete symptom control. Prior treatment history and inadequate control inform escalation decisions, including whether to in… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h2_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h2_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `h2_days_supply_total->biologic_initiation`, `h2_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h2_days_supply_total->biologic_initiation`, `h2_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | h2_days_supply_total->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Pre-index antihistamine/H2-antagonist treatment history is available to the treating clinician and can influence escalation to a biologic when disease remains inadequately controlled on prior therapies. International urticaria guidance frames… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h2_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `h2_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `h2_days_since_last_fill->biologic_initiation`, `h2_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `h2_days_since_last_fill->biologic_initiation`, `h2_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | h2_days_since_last_fill->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Recency of a baseline H2-receptor-antagonist fill encodes the patient's antecedent urticaria treatment trajectory. Treatment control and prior/ongoing pharmacotherapy are assessed when escalating management to biologic therapy; thus this p… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `h2_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ltra_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `ltra_ever_filled->biologic_initiation`, `ltra_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ltra_ever_filled->biologic_initiation`, `ltra_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | ltra_ever_filled->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: A prior LTRA fill represents prior pharmacologic management of urticaria/allergic disease. Treatment history, including inadequate control on prior therapies, informs escalation decisions and can increase the likelihood of biologic initiation. ‖ … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ltra_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ltra_fill_count` | `biologic_initiation->initiated_biologic_180d`, `ltra_fill_count->biologic_initiation`, `ltra_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ltra_fill_count->biologic_initiation`, `ltra_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | ltra_fill_count->T [unsupported PMID:29336054✓,DOI:10.1111/all.13397✓]: Repeated pre-index LTRA fills encode prior controller therapy and an escalation/refractory-treatment history. Such prior treatment history is clinically used in deciding whether to escalate to advanced therapy, including biologic treatment. ‖ ltra… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ltra_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ltra_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `ltra_days_supply_total->biologic_initiation`, `ltra_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ltra_days_supply_total->biologic_initiation`, `ltra_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | ltra_days_supply_total->T [unsupported PMID: 34655584✓]: Pre-index LTRA dispensing represents prior controller-treatment exposure and treatment history. In asthma care, biologic therapy is considered for severe/uncontrolled disease despite optimized controller therapy; prior controller use and failure therefore inform… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ltra_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ltra_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `ltra_days_since_last_fill->biologic_initiation`, `ltra_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ltra_days_since_last_fill->biologic_initiation`, `ltra_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | ltra_days_since_last_fill->T [unsupported 10.1183/13993003.00583-2020✗]: Time since the last leukotriene receptor antagonist dispensing captures recent controller-treatment use and treatment escalation history. Prior and current controller therapy is part of the clinical assessment used to determine whether asthma man… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ltra_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `sys_steroid_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_ever_filled->biologic_initiation`, `sys_steroid_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_ever_filled->biologic_initiation`, `sys_steroid_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | sys_steroid_ever_filled->T [unsupported PMID:24337046✓]: Pre-index systemic corticosteroid treatment history is evidence of prior treatment escalation and is used in clinical decisions about escalation to biologic therapy; prior systemic corticosteroid exposure/requirement is a recognized marker of severe or difficult… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `sys_steroid_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `sys_steroid_fill_count` | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_fill_count->biologic_initiation`, `sys_steroid_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_fill_count->biologic_initiation`, `sys_steroid_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | sys_steroid_fill_count->T [unsupported DOI:10.1183/13993003.00635-2020✗]: A history of repeated systemic corticosteroid dispensing documents prior exacerbation management and treatment escalation. In severe inflammatory airway disease, recurrent systemic-corticosteroid use is a criterion supporting escalation to biolo… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `sys_steroid_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `sys_steroid_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_days_supply_total->biologic_initiation`, `sys_steroid_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_days_supply_total->biologic_initiation`, `sys_steroid_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | sys_steroid_days_supply_total->T [unsupported DOI:10.1164/rccm.202003-0866ST✗]: Pre-index systemic corticosteroid exposure is a treatment-history signal of poor control or recurrent exacerbation and is used in escalation decisions. Severe-disease guidance identifies recurrent systemic corticosteroid need/exacerbation … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `sys_steroid_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `sys_steroid_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_days_since_last_fill->biologic_initiation`, `sys_steroid_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `sys_steroid_days_since_last_fill->biologic_initiation`, `sys_steroid_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | sys_steroid_days_since_last_fill->T [unsupported PMID:31558662✓,DOI:10.1183/13993003.00658-2019✗]: Pre-index recency of systemic corticosteroid treatment captures recent need for rescue therapy and is used in clinical assessment of uncontrolled/severe inflammatory disease when considering escalation to steroid-sparing… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `sys_steroid_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `top_steroid_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `top_steroid_ever_filled->biologic_initiation`, `top_steroid_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `top_steroid_ever_filled->biologic_initiation`, `top_steroid_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | top_steroid_ever_filled->T [unsupported PMID: 36641009✓,PMID: 37943240✓]: A documented prior topical-corticosteroid fill is part of the observed treatment history used in clinical escalation decisions. In atopic dermatitis care, topical corticosteroids are recommended topical therapy, while biologic/systemic therapy i… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `top_steroid_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `top_steroid_fill_count` | `biologic_initiation->initiated_biologic_180d`, `top_steroid_fill_count->biologic_initiation`, `top_steroid_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `top_steroid_fill_count->biologic_initiation`, `top_steroid_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | top_steroid_fill_count->T [unsupported PMID: 33314141✓,DOI: 10.1111/bjd.20697✗]: A greater pre-index burden of topical corticosteroid treatment indicates prior treatment exposure and insufficient disease control, factors used in clinical escalation decisions toward systemic or biologic therapy when topical treatment i… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `top_steroid_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `top_steroid_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `top_steroid_days_supply_total->biologic_initiation`, `top_steroid_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `top_steroid_days_supply_total->biologic_initiation`, `top_steroid_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | top_steroid_days_supply_total->T [unsupported PMID: 28754115✓,DOI: 10.1016/j.jaad.2017.06.007✗]: Pre-index topical corticosteroid treatment burden is part of the patient's observed prior-treatment history. Repeated or substantial topical-treatment use reflects insufficient control with topical management and can promp… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `top_steroid_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `top_steroid_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `top_steroid_days_since_last_fill->biologic_initiation`, `top_steroid_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `top_steroid_days_since_last_fill->biologic_initiation`, `top_steroid_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | top_steroid_days_since_last_fill->T [unsupported PMID: 24813298✓,PMID: 24813302✓]: Recency of topical corticosteroid dispensing represents recent topical-treatment use and is relevant to therapeutic escalation. In inflammatory dermatoses such as atopic dermatitis, persistent or inadequately controlled disease despite … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `top_steroid_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `immunosupp_ever_filled` | `biologic_initiation->initiated_biologic_180d`, `immunosupp_ever_filled->biologic_initiation`, `immunosupp_ever_filled->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `immunosupp_ever_filled->biologic_initiation`, `immunosupp_ever_filled->initiated_biologic_180d` → **confounder** | yes / yes | immunosupp_ever_filled->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: A recorded history of immunosuppressive therapy is clinically used as evidence of prior treatment exposure and potential refractory disease, and can therefore affect escalation to, or selection of, biologic treatment at the index date. Trea… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `immunosupp_ever_filled`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `immunosupp_fill_count` | `biologic_initiation->initiated_biologic_180d`, `immunosupp_fill_count->biologic_initiation`, `immunosupp_fill_count->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `immunosupp_fill_count->biologic_initiation`, `immunosupp_fill_count->initiated_biologic_180d` → **confounder** | yes / yes | immunosupp_fill_count->T [unsupported]: A pre-index count of immunosuppressant dispensings encodes prior treatment exposure and treatment trajectory, including prior therapy use or failure and potential step-therapy requirements; these can influence the clinical decision to initiate a biologic at the index date. ‖ imm… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `immunosupp_fill_count`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `immunosupp_days_supply_total` | `biologic_initiation->initiated_biologic_180d`, `immunosupp_days_supply_total->biologic_initiation`, `immunosupp_days_supply_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `immunosupp_days_supply_total->biologic_initiation`, `immunosupp_days_supply_total->initiated_biologic_180d` → **confounder** | yes / yes | immunosupp_days_supply_total->T [unsupported]: Pre-index cumulative immunosuppressant supply can directly influence biologic initiation through prior-treatment failure or intolerance, completion of step-therapy requirements, and authorization or clinician escalation decisions. ‖ immunosupp_days_supply_total->Y [unsupp… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `immunosupp_days_supply_total`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `immunosupp_days_since_last_fill` | `biologic_initiation->initiated_biologic_180d`, `immunosupp_days_since_last_fill->biologic_initiation`, `immunosupp_days_since_last_fill->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `immunosupp_days_since_last_fill->biologic_initiation`, `immunosupp_days_since_last_fill->initiated_biologic_180d` → **confounder** | yes / yes | immunosupp_days_since_last_fill->T [unsupported DOI:10.1016/j.jaad.2018.11.057✗]: Recency of prior immunosuppressant dispensing represents active/recent systemic-treatment exposure and treatment history. Prior systemic treatment response, failure, intolerance, and sequencing inform the clinical decision to escalate or… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `immunosupp_days_since_last_fill`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ige_total_tested` | `biologic_initiation->initiated_biologic_180d`, `ige_total_tested->biologic_initiation`, `ige_total_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ige_total_tested->biologic_initiation`, `ige_total_tested->initiated_biologic_180d` → **confounder** | yes / yes | ige_total_tested->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Receipt of total serum IgE testing represents an active allergy/urticaria diagnostic and phenotyping work-up. Such work-up can provide information used by clinicians in escalation and biologic-treatment decisions, including decision-making at the… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ige_total_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ige_total_result_last` | `biologic_initiation->initiated_biologic_180d`, `ige_total_result_last->biologic_initiation`, `ige_total_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ige_total_result_last->biologic_initiation`, `ige_total_result_last->initiated_biologic_180d` → **confounder** | yes / yes | ige_total_result_last->T [unsupported PMID:36116373✓]: A baseline total serum IgE result can affect biologic-treatment selection because it identifies an allergic phenotype and is used in clinical decision-making for anti-IgE therapy, including omalizumab eligibility/dosing in allergic asthma. ‖ ige_total_result_last-… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ige_total_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ige_total_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `ige_total_abnormal_flag->biologic_initiation`, `ige_total_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `ige_total_abnormal_flag->biologic_initiation`, `ige_total_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | ige_total_abnormal_flag->T [unsupported 10.1056/NEJMoa071499✗]: An abnormal total serum IgE result documents an allergic/type-2 phenotype and can directly inform clinician selection and initiation of biologic therapy, especially anti-IgE treatment, for eligible allergic disease. ‖ ige_total_abnormal_flag->Y [unsupport… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ige_total_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `eosinophil_tested` | `biologic_initiation->initiated_biologic_180d`, `eosinophil_tested->biologic_initiation`, `eosinophil_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `eosinophil_tested->biologic_initiation`, `eosinophil_tested->initiated_biologic_180d` → **confounder** | yes / yes | eosinophil_tested->T [unsupported DOI:10.1164/rccm.202003-0596ST✗]: Obtaining a blood eosinophil assessment is part of inflammatory phenotype evaluation used to select and initiate eosinophil-targeted or other phenotype-directed biologic therapy; thus, availability of this work-up can affect the initiation decision. ‖… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `eosinophil_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `eosinophil_result_last` | `biologic_initiation->initiated_biologic_180d`, `eosinophil_result_last->biologic_initiation`, `eosinophil_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `eosinophil_result_last->biologic_initiation`, `eosinophil_result_last->initiated_biologic_180d` → **confounder** | yes / yes | eosinophil_result_last->T [unsupported PMID:31558655✓]: An elevated blood eosinophil result identifies a type-2/eosinophilic inflammatory phenotype and is used in selecting patients for eosinophil-targeted or other biologic therapy; it can therefore affect the decision to initiate a biologic. ‖ eosinophil_result_last-… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `eosinophil_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `eosinophil_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `eosinophil_abnormal_flag->biologic_initiation`, `eosinophil_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `eosinophil_abnormal_flag->biologic_initiation`, `eosinophil_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | eosinophil_abnormal_flag->T [unsupported PMID:31558662✓,DOI:10.1183/13993003.00588-2019✓]: A pre-index abnormal eosinophil result is a clinical marker of eosinophilic/type-2 inflammation. Blood eosinophil levels are used to phenotype patients and guide selection or eligibility for eosinophil-targeting biologic therapi… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `eosinophil_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `crp_tested` | `biologic_initiation->initiated_biologic_180d`, `crp_tested->biologic_initiation`, `crp_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `crp_tested->biologic_initiation`, `crp_tested->initiated_biologic_180d` → **confounder** | yes / yes | crp_tested->T [unsupported PMID:10928403✓,DOI:10.1172/JCI11104✗]: Ordering CRP is part of an inflammatory-condition diagnostic and disease-activity workup. The resulting workup information can affect a clinician's decision to initiate biologic therapy at the index encounter. ‖ crp_tested->Y [unsupported PMID:10928403✓… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `crp_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `crp_result_last` | `biologic_initiation->initiated_biologic_180d`, `crp_result_last->biologic_initiation`, `crp_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `crp_result_last->biologic_initiation`, `crp_result_last->initiated_biologic_180d` → **confounder** | yes / yes | crp_result_last->T [unsupported PMID:30851428✓,PMID:32097259✓]: CRP is an objective acute-phase marker of inflammation. In inflammatory diseases, evidence of active inflammation, including elevated CRP, is used in clinical assessment and therapeutic escalation decisions, including selection/initiation of advanced biol… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `crp_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `crp_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `crp_abnormal_flag->biologic_initiation`, `crp_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `crp_abnormal_flag->biologic_initiation`, `crp_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | crp_abnormal_flag->T [unsupported PMID: 17945490✓,DOI: 10.1016/j.jaci.2011.05.013✗]: An abnormal CRP is a clinically available marker of systemic inflammation and disease activity. Such evidence of inflammatory burden can influence clinicians' decisions to escalate to biologic therapy. ‖ crp_abnormal_flag->Y [unsuppor… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `crp_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `tpo_ab_tested` | `biologic_initiation->initiated_biologic_180d`, `tpo_ab_tested->biologic_initiation`, `tpo_ab_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `tpo_ab_tested->biologic_initiation`, `tpo_ab_tested->initiated_biologic_180d` → **confounder** | yes / yes | tpo_ab_tested->T [unsupported PMID:34536239✓]: Thyroid autoimmunity assessment, including anti-thyroid peroxidase antibodies, is part of the diagnostic work-up for chronic spontaneous urticaria in specialist care. Completing this pre-index diagnostic work-up can affect the clinician's treatment-escalation decision, in… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `tpo_ab_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `tpo_ab_result_last` | `biologic_initiation->initiated_biologic_180d`, `tpo_ab_result_last->biologic_initiation`, `tpo_ab_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `tpo_ab_result_last->biologic_initiation`, `tpo_ab_result_last->initiated_biologic_180d` → **confounder** | yes / yes | tpo_ab_result_last->T [unsupported DOI:10.1111/all.15090✓]: A positive or elevated thyroid peroxidase antibody result identifies thyroid autoimmunity and is a clinically available autoimmune phenotype at baseline. Anti-TPO testing is included in specialist evaluation/phenotyping of chronic spontaneous urticaria, a con… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `tpo_ab_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `tpo_ab_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `tpo_ab_abnormal_flag->biologic_initiation`, `tpo_ab_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `tpo_ab_abnormal_flag->biologic_initiation`, `tpo_ab_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | tpo_ab_abnormal_flag->T [unsupported PMID:19874331✓,PMID:30738175✓]: An abnormal thyroid peroxidase antibody result identifies thyroid autoimmunity, which is enriched in chronic spontaneous urticaria and can inform clinician assessment of an autoimmune or difficult-to-control phenotype when selecting/escalating treatm… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `tpo_ab_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `free_t4_tested` | `biologic_initiation->initiated_biologic_180d`, `free_t4_tested->biologic_initiation`, `free_t4_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `free_t4_tested->biologic_initiation`, `free_t4_tested->initiated_biologic_180d` → **confounder** | yes / yes | free_t4_tested->T [unsupported PMID: 29336056✓]: A pre-index free-thyroxine test represents thyroid-function workup and information available to the treating clinician. Thyroid evaluation is part of the diagnostic assessment of chronic urticaria when clinically indicated; recognition of thyroid comorbidity can affect … | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `free_t4_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `free_t4_result_last` | `biologic_initiation->initiated_biologic_180d`, `free_t4_result_last->biologic_initiation`, `free_t4_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `free_t4_result_last->biologic_initiation`, `free_t4_result_last->initiated_biologic_180d` → **confounder** | yes / yes | free_t4_result_last->T [unsupported 10.1111/all.15090✓]: A free-T4 result available at the treatment decision can identify thyroid dysfunction and can consequently affect clinical evaluation, competing-condition management, and the timing or selection of biologic treatment. ‖ free_t4_result_last->Y [unsupported 10.111… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `free_t4_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `free_t4_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `free_t4_abnormal_flag->biologic_initiation`, `free_t4_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `free_t4_abnormal_flag->biologic_initiation`, `free_t4_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | free_t4_abnormal_flag->T [unsupported DOI:10.1111/all.15090✓]: An abnormal free-T4 result denotes clinically relevant thyroid dysfunction. Thyroid disease/thyroid autoimmunity is a recognized comorbidity and endotypic feature of chronic spontaneous urticaria that can influence clinical evaluation, perceived disease co… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `free_t4_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `tsh_tested` | `biologic_initiation->initiated_biologic_180d`, `tsh_tested->biologic_initiation`, `tsh_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `tsh_tested->biologic_initiation`, `tsh_tested->initiated_biologic_180d` → **confounder** | yes / yes | tsh_tested->T [unsupported PMID: 34536239✓]: A thyroid-function test at or before the index date denotes an active clinician work-up. Diagnostic evaluation and identification of comorbidity can directly inform treatment planning and the decision to initiate a biologic; the test is temporally prior to the indexed initi… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `tsh_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `tsh_result_last` | `biologic_initiation->initiated_biologic_180d`, `tsh_result_last->biologic_initiation`, `tsh_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `tsh_result_last->biologic_initiation`, `tsh_result_last->initiated_biologic_180d` → **confounder** | yes / yes | tsh_result_last->T [unsupported PMID:25266247✓]: A pre-index TSH result represents thyroid functional status. Clinically significant thyroid dysfunction is a comorbidity considered during clinical assessment and medication selection, including whether and when to initiate systemic therapies. ‖ tsh_result_last->Y [unsu… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `tsh_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `tsh_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `tsh_abnormal_flag->biologic_initiation`, `tsh_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `tsh_abnormal_flag->biologic_initiation`, `tsh_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | tsh_abnormal_flag->T [unsupported 10.1136/ard-2022-223356✗,10.1089/thy.2015.0020✓]: An abnormal TSH identifies thyroid dysfunction, a clinically relevant comorbidity that can prompt additional evaluation and affect individualized treatment selection and timing, including decisions to initiate advanced/biologic therapy… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `tsh_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ana_tested` | `ana_tested->biologic_initiation`, `ana_tested->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `ana_tested->biologic_initiation`, `ana_tested->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | ana_tested->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: Ordering ANA testing is part of evaluation for suspected systemic autoimmune disease or an alternative autoimmune diagnosis. This diagnostic workup can directly inform the clinician's contemporaneous decision whether to initiate biologic therapy. ‖ ana… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ana_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ana_result_last` | `ana_result_last->biologic_initiation`, `ana_result_last->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `ana_result_last->biologic_initiation`, `ana_result_last->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | ana_result_last->T [unsupported PMID:31385443✓,DOI:10.1136/annrheumdis-2019-215089✗]: An ANA result available at the treatment decision can support recognition of a systemic autoimmune disease phenotype. Such phenotypes may alter specialist referral and selection/initiation of immunomodulatory treatment, including bio… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ana_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `ana_abnormal_flag` | `ana_abnormal_flag->biologic_initiation`, `ana_abnormal_flag->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | `ana_abnormal_flag->biologic_initiation`, `ana_abnormal_flag->initiated_biologic_180d`, `biologic_initiation->initiated_biologic_180d` → **confounder** | yes / yes | ana_abnormal_flag->T [unsupported PMID:34536239✓]: An abnormal antinuclear-antibody result is clinically used as evidence prompting assessment for systemic autoimmune/connective-tissue disease. Such diagnostic information can alter clinician treatment selection, including whether and when to initiate a biologic. ‖ ana… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `ana_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `cbc_tested` | `biologic_initiation->initiated_biologic_180d`, `cbc_tested->biologic_initiation`, `cbc_tested->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `cbc_tested->biologic_initiation`, `cbc_tested->initiated_biologic_180d` → **confounder** | yes / yes | cbc_tested->T [unsupported PMID:20725599✓]: An index CBC is part of baseline safety and clinical work-up before biologic therapy; availability of this assessment can affect whether initiation proceeds or is deferred because of hematologic abnormalities or treatment-readiness findings. ‖ cbc_tested->Y [unsupported PMID… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `cbc_tested`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `cbc_result_last` | `biologic_initiation->initiated_biologic_180d`, `cbc_result_last->biologic_initiation`, `cbc_result_last->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `cbc_result_last->biologic_initiation`, `cbc_result_last->initiated_biologic_180d` → **confounder** | yes / yes | cbc_result_last->T [unsupported PMID:34101376✓]: A baseline complete blood count result can influence immediate biologic-treatment decisions because clinically important hematologic abnormalities may alter perceived treatment safety, eligibility, monitoring requirements, or the choice to defer therapy. ‖ cbc_result_la… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `cbc_result_last`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `cbc_abnormal_flag` | `biologic_initiation->initiated_biologic_180d`, `cbc_abnormal_flag->biologic_initiation`, `cbc_abnormal_flag->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `cbc_abnormal_flag->biologic_initiation`, `cbc_abnormal_flag->initiated_biologic_180d` → **confounder** | yes / yes | cbc_abnormal_flag->T [unsupported PMID:27031035✓,PMID:25077606✓]: A pre-index abnormal CBC can indicate cytopenia, infection/inflammation, or hematologic comorbidity. Such safety-relevant clinical findings can affect a clinician's decision to start a biologic. ‖ cbc_abnormal_flag->Y [unsupported PMID:27031035✓,PMID:25… | NO research bullet names this feature (no bullet in docs/layer4/optum_initiation_attestation_research.md names `cbc_abnormal_flag`) — the manifest side is only the `_optum_attestation` edge pattern | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `comorbidity_load_total` | `biologic_initiation->initiated_biologic_180d`, `comorbidity_load_total->biologic_initiation`, `comorbidity_load_total->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `comorbidity_load_total->biologic_initiation`, `comorbidity_load_total->initiated_biologic_180d` → **confounder** | yes / yes | comorbidity_load_total->T [unsupported PMID:34536239✓,DOI:10.1111/all.15090✓]: The aggregate includes clinically relevant chronic urticaria comorbidities and angioedema. Comorbidity burden and disease manifestations inform assessment of disease burden, risk, and escalation to guideline-directed biologic therapy. ‖ com… | L68: - Comorbidities — atopic dermatitis / asthma / allergic rhinitis / thyroid-autoimmune / angioedema / nsaid-hypersensitivity / anxiety / depression families (`has_*` + `*_claim_count`), `atopy_score`, `mental_health_flag`, `elixhauser_score`, `charlson_score`, `comorbidity_load_total` (PMID 42050840; PMID 40220911… | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `csu_dx_intensity` | `biologic_initiation->initiated_biologic_180d`, `csu_dx_intensity->biologic_initiation`, `csu_dx_intensity->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `csu_dx_intensity->biologic_initiation`, `csu_dx_intensity->initiated_biologic_180d` → **confounder** | yes / yes | csu_dx_intensity->T [unsupported PMID:34536239✓]: This pre-index diagnostic-intensity/chronicity measure summarizes documented persistent CSU burden. Persistent or inadequately controlled CSU drives treatment escalation, including use of biologic therapy, under international CSU treatment guidance. ‖ csu_dx_intensity-… | L67: - CSU dx burden — `dx_l50_1/8/9_count`, `dx_total_csu`, `dx_angioedema_count`, `csu_dx_intensity` (PMID 39325444; PMID 34984792). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `polypharmacy_breadth` | `biologic_initiation->initiated_biologic_180d`, `polypharmacy_breadth->biologic_initiation`, `polypharmacy_breadth->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `polypharmacy_breadth->biologic_initiation`, `polypharmacy_breadth->initiated_biologic_180d` → **confounder** | yes / yes | polypharmacy_breadth->T [unsupported DOI:10.1111/all.15090✓]: Breadth of prior antihistamine, corticosteroid, leukotriene-antagonist, and immunosuppressant use represents prior inadequate control and therapeutic escalation. In chronic urticaria, escalation to biologic treatment is recommended after inadequate response… | L70: - Prior pharmacotherapy (the EAACI/GA²LEN CSU ladder: 2nd-gen H1 → up-dose → +H2/+LTRA → omalizumab → cyclosporine) — `h1_1g`, `h1_2g`, `h2`, `ltra`, `sys_steroid`, `top_steroid`, `immunosupp` families (×4 cols), `polypharmacy_breadth` (all.15090 EAACI guideline; PMC6735630 steroid-as-severity). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |
| `lab_workup_completeness` | `biologic_initiation->initiated_biologic_180d`, `lab_workup_completeness->biologic_initiation`, `lab_workup_completeness->initiated_biologic_180d` → **confounder** | `biologic_initiation->initiated_biologic_180d`, `lab_workup_completeness->biologic_initiation`, `lab_workup_completeness->initiated_biologic_180d` → **confounder** | yes / yes | lab_workup_completeness->T [unsupported PMID:34536239✓]: Completion of an appropriate diagnostic evaluation is part of clinical assessment before treatment escalation in chronic urticaria; documenting the diagnostic workup can facilitate a clinician's decision to initiate advanced/biologic therapy. ‖ lab_workup_comple… | L71: - Baseline labs / endotype markers — `ige_total`, `eosinophil`, `crp`, `tpo_ab`, `free_t4`, `tsh`, `ana`, `cbc` families (×3 cols), `lab_workup_completeness` (WAO 1939-4551(24)00036-X IgE/eos predictors; falgy.2025.1706705 autoimmune endotype; PMID 37634502 CRP). | `machine` → `machine_reviewed IF review 2fa5a4a5 approved` |

## Machine-readable

```json
{
 "semantics": "machine_reviewed = human-approved in the review queue and DECIDING (feature_contract.ATTESTATION_DECIDING_PROVENANCES); apply only with review_id 2fa5a4a5-4035-4511-9fa1-1aedf1bd6645 approved",
 "rows": {
  "age_at_index": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "age_group": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "gender": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "zip5": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "zip3": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "zip_code": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "geographic_region": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "insurance_product": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "plan_type": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "payer_category": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "urban_rural_code": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "primary_diagnosis_code": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "dx_l50_1_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "dx_l50_8_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "dx_l50_9_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "dx_total_csu": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "dx_angioedema_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "months_since_first_dx": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "csu_chronicity": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "has_atopic_dermatitis": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "atopic_dermatitis_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_asthma": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "asthma_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_allergic_rhinitis": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "allergic_rhinitis_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_anxiety": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "anxiety_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_depression": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "depression_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_thyroid_autoimmune": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "thyroid_autoimmune_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_nsaid_hypersensitivity": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "nsaid_hypersensitivity_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "has_angioedema": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "angioedema_claim_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "atopy_score": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "mental_health_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "elixhauser_score": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "charlson_score": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "office_visits_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "office_visits_allergist": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "office_visits_dermatology": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "office_visits_pcp": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "ed_visits_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "ed_visits_urticaria_angio": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "hospitalizations_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "unique_providers": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "h1_1g_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_1g_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_1g_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_1g_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_2g_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_2g_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_2g_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h1_2g_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h2_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h2_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h2_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "h2_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ltra_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ltra_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ltra_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ltra_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "sys_steroid_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "sys_steroid_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "sys_steroid_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "sys_steroid_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "top_steroid_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "top_steroid_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "top_steroid_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "top_steroid_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "immunosupp_ever_filled": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "immunosupp_fill_count": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "immunosupp_days_supply_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "immunosupp_days_since_last_fill": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ige_total_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ige_total_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ige_total_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "eosinophil_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "eosinophil_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "eosinophil_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "crp_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "crp_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "crp_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "tpo_ab_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "tpo_ab_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "tpo_ab_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "free_t4_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "free_t4_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "free_t4_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "tsh_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "tsh_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "tsh_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ana_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ana_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "ana_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "cbc_tested": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "cbc_result_last": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "cbc_abnormal_flag": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": false
  },
  "specialist_concentration": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "primary_specialist_type": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "saw_allergist_flag": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "saw_dermatologist_flag": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "index_date": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "lookback_start_date": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  },
  "comorbidity_load_total": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "csu_dx_intensity": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "polypharmacy_breadth": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "lab_workup_completeness": {
   "current": "machine",
   "candidate_if_review_approved": "machine_reviewed",
   "agree": true,
   "manifest_research_grounded": true
  },
  "specialist_visit_interaction": {
   "current": "machine",
   "candidate_if_review_approved": null,
   "agree": false,
   "manifest_research_grounded": true
  }
 }
}
```
