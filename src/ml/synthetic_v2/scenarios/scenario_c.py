"""Scenario C — CSU 12-wk UAS7=0 response on remibrutinib (Rhapsido franchise).

Per shard 05 (.claude/plans/synthetic_data_generator_v2/05-scenario-c-treatment-csu-response.md):

- 60 features across 7 clinical clusters + 1 noise cluster.
- target_prevalence = 0.40 (REMIX-1/2 12-wk UAS7=0 non-response complement).
- target_auc_band = (0.82, 0.88) (matches Maurer 2023 + REMIX biomarker analyses).
- primary_tau = 0.30 (predicted-failure ≥ 30% triggers escalation).
- Use case: treatment_decision; clinical_threshold_range τ ∈ [0.20, 0.50].
- **MAIN RWD COHORT**: only Phase 1 scenario with concurrent-validation
  hook to RWD CSU cohort (per shard 05 §G; loader lands in commit 13).

Slope calibration: SLOPE_MULTIPLIER locked once at implementation time so
median LR test AUC over 10 seeds lands at ~0.85 (band midpoint) with
9/10 seeds in [0.82, 0.88].
"""

from __future__ import annotations

import numpy as np

from src.ml.synthetic_v2.dgp import sample_one_feature
from src.ml.synthetic_v2.manifest import FeatureManifest
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.scenarios._base import ScenarioBuilder

SLOPE_MULTIPLIER: float = 1.25  # calibrated 2026-05-03 via §B.4 bisection (post full-cohort standardization fix): 10/10 seeds in [0.82, 0.88] AUC band, median 0.836.
# Bisection log:
#   slope=1.10 -> 4/10 in band, median 0.816
#   slope=1.15 -> 9/10 in band, median 0.827
#   slope=1.20 -> 8/10 in band, median 0.833
#   slope=1.25 -> 10/10 in band, median 0.836  *** LOCKED ***
#   slope=1.30 -> 10/10 in band, median 0.848
# Higher slope than A/B because Scenario C has the strongest manifest signal
# (60 features incl. UAS7 cluster + biomarkers) needs amplification to land
# at the [0.82, 0.88] band rather than overshooting (LR will saturate at
# ~0.99 AUC if slope is doubled). Action item: shard 05 §B.4 prose should
# be updated to reflect that initial estimate ~0.50 was off post-fix.

SCENARIO_C_MANIFEST: tuple[FeatureManifest, ...] = (
    # Cluster 1: Demographics + CSU history (8) [indices 0-7]
    FeatureManifest(
        name="age_at_csu_diagnosis_years",
        distribution="normal",
        distribution_params={"loc": 42.0, "scale": 14.0},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Older age at CSU onset modestly correlates with longer disease duration + reduced spontaneous remission probability; weakly elevates non-response to BTK monotherapy (Maurer 2017 Allergy review).",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="sex_female",
        distribution="bernoulli",
        distribution_params={"p": 0.66},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="CSU is female-predominant (~2:1); female sex modestly associated with type IIb autoimmunity prevalence + harder-to-treat disease (Schoepke 2019 EAACI/GA2LEN task force).",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="bmi_kg_m2",
        distribution="normal",
        distribution_params={"loc": 28.0, "scale": 6.5},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Higher BMI weakly elevates CSU severity + reduces omalizumab response rates (Magen 2017); BTK-targeted therapy effect of BMI is hypothesized similar but not yet established in REMIX subgroup analyses.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="csu_disease_duration_years",
        distribution="normal",
        distribution_params={"loc": 4.5, "scale": 4.0},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Longer disease duration (>5y) correlates with refractoriness; spontaneous remission window has passed and chronic mast-cell hyperresponsiveness is established (Sussman 2014).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="prior_csu_remission_relapse_history",
        distribution="bernoulli",
        distribution_params={"p": 0.30},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Pattern of remission-then-relapse marks chronic refractory disease; predicts non-response to monotherapy biologics.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="ethnicity_white",
        distribution="bernoulli",
        distribution_params={"p": 0.65},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Ancestry stratification in CSU response is weakly studied; included for cohort-balance audit + RWD concurrent-validation comparison. Direction effectively zero in REMIX trials (Metz 2024).",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="employment_status_active",
        distribution="bernoulli",
        distribution_params={"p": 0.62},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Active employment correlates with healthcare-engagement + adherence proxy; modestly protective via adherence channel.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="age_at_assessment_years",
        distribution="normal",
        distribution_params={"loc": 47.0, "scale": 14.5},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Age at H1-failure assessment timepoint; closely correlated with age_at_csu_diagnosis_years + csu_disease_duration_years (mathematically derived in real cohorts).",
        citation_strength="weak",
    ),
    # Cluster 2: Urticaria activity (UAS7/UCT/AAS7/QoL) (10) [indices 8-17]
    FeatureManifest(
        name="baseline_uas7_total",
        distribution="normal",
        distribution_params={"loc": 28.0, "scale": 8.0},
        coefficient=+0.45,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="UAS7 (Urticaria Activity Score over 7 days, 0-42) at baseline is the dominant predictor of failure to achieve UAS7=0; high baseline activity (>28) reduces complete-response probability by half (Mlynek 2008; REMIX-1/2 baseline-stratified analyses).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="baseline_uas7_pruritus_component",
        distribution="normal",
        distribution_params={"loc": 14.5, "scale": 4.5},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="UAS7 pruritus subscore (0-21) drives QoL impact and predicts non-response; itch-dominant phenotype harder to control (Hawro 2018).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="baseline_uas7_hives_component",
        distribution="normal",
        distribution_params={"loc": 13.5, "scale": 4.5},
        coefficient=+0.18,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="UAS7 hive subscore (0-21); correlated with pruritus subscore but separate measurement. Higher hive burden predicts non-response.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="baseline_uct_score",
        distribution="normal",
        distribution_params={"loc": 4.5, "scale": 2.5},
        coefficient=-0.30,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="UCT (Urticaria Control Test, 0-16, higher=better control) at baseline; UCT <12 defines uncontrolled disease (Weller 2014). Higher baseline UCT (more controlled) predicts achieving UAS7=0 (negative coefficient on failure outcome).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="baseline_aas7_score",
        distribution="normal",
        distribution_params={"loc": 22.0, "scale": 18.0},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="AAS7 (Angioedema Activity Score over 7 days, 0-105) measures angioedema burden; conditional on angioedema-present subset (~40% of CSU). Higher AAS7 marks severe-end disease + lower complete-response probability (Weller 2013).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="angioedema_present_at_baseline",
        distribution="bernoulli",
        distribution_params={"p": 0.42},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Concomitant angioedema (~40% of CSU patients) marks severe disease + reduces UAS7=0 achievability; REMIX-1/2 angioedema-positive subgroup had lower complete-response rate.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="baseline_dlqi_score",
        distribution="normal",
        distribution_params={"loc": 14.0, "scale": 6.5},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Dermatology Life Quality Index (0-30, higher=worse QoL); DLQI >10 marks severe QoL impact and predicts treatment-resistant CSU.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="baseline_cu_q2ol_score",
        distribution="normal",
        distribution_params={"loc": 50.0, "scale": 22.0},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="CU-Q2oL (Chronic Urticaria Quality of Life Questionnaire, 23 items x 0-5, higher=worse) is CSU-specific QoL instrument; partially redundant with DLQI.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="sleep_disturbance_baseline_severity",
        distribution="normal",
        distribution_params={"loc": 2.4, "scale": 1.2},
        coefficient=+0.12,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Sleep disturbance score (0-5 ordinal) reflects nocturnal pruritus burden; Maurer 2017 EAACI sleep-disturbance is a key QoL component.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="work_productivity_loss_pct_baseline",
        distribution="normal",
        distribution_params={"loc": 32.0, "scale": 22.0},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="WPAI:CSU work-productivity-loss percentage at baseline; higher loss correlates with severity but adds modest signal beyond UAS7 + DLQI.",
        citation_strength="moderate",
    ),
    # Cluster 3: Immunology biomarkers (8) [indices 18-25]
    FeatureManifest(
        name="total_serum_ige_iu_ml",
        distribution="normal",
        distribution_params={"loc": 220.0, "scale": 280.0},
        coefficient=-0.20,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="High baseline total IgE (>100 IU/mL) marks type I (IgE-mediated) autoimmunity; predicts BETTER response to anti-IgE (omalizumab) AND to BTK inhibition because BTK is downstream of FcepsilonRI cross-linking. Modestly negative coefficient on failure outcome (Maurer 2018; Metz 2024 REMIX biomarker analysis).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="anti_thyroid_peroxidase_anti_tpo_positive",
        distribution="bernoulli",
        distribution_params={"p": 0.30},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Anti-TPO positivity is a type IIb (autoantibody-mediated) autoimmunity marker; predicts REDUCED omalizumab response. For BTK inhibition the signal is weaker because BTK acts downstream of both FcepsilonRI cross-linking pathways. Modest positive coefficient (Maurer 2023).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="anti_ige_or_anti_fceri_autoantibody_positive",
        distribution="bernoulli",
        distribution_params={"p": 0.20},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Anti-IgE or anti-FcepsilonRI autoantibodies define type IIb autoimmunity; correlates with anti-TPO+ subset; predicts reduced omalizumab response, modestly reduced BTK response.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="crp_mg_l",
        distribution="normal",
        distribution_params={"loc": 4.5, "scale": 5.0},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Elevated baseline CRP marks systemic inflammation + correlates with type IIb autoimmunity; higher CRP predicts non-response to multiple biologics (Asero 2020).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="d_dimer_ng_ml",
        distribution="normal",
        distribution_params={"loc": 950.0, "scale": 700.0},
        coefficient=+0.30,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Elevated d-dimer marks coagulation cascade activation in CSU + correlates with severe disease and omalizumab non-response (Asero 2017); increasingly used as a biomarker for refractory CSU.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="eosinophil_count_cells_ul",
        distribution="normal",
        distribution_params={"loc": 250.0, "scale": 180.0},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Higher peripheral eosinophil count weakly associated with type I inflammation + better anti-IgE response; weak signal for BTK inhibition.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="basophil_count_cells_ul",
        distribution="normal",
        distribution_params={"loc": 18.0, "scale": 18.0},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Basopenia (low circulating basophils) is a CSU finding (basophils marginated to lesional skin); marks active disease + predicts non-response. Coefficient on raw count is NEGATIVE on the failure outcome (lower basophils -> higher failure -> beta < 0 in a logistic model where outcome=1 is failure). Codex I-3 (2026-05-03) flipped sign + monotone direction from prior +0.10/+1 typo.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="complement_c3_c4_normal",
        distribution="bernoulli",
        distribution_params={"p": 0.92},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Normal C3/C4 rules out urticarial vasculitis (which mimics CSU but does not respond to BTK inhibition); included as inclusion-criterion marker. Most patients are C3/C4-normal by design.",
        citation_strength="moderate",
    ),
    # Cluster 4: Comorbidities (8) [indices 26-33]
    FeatureManifest(
        name="allergic_rhinitis_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.35},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Atopic comorbidity marks type-I-skewed immunology + modestly better anti-IgE/BTK response.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="asthma_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Asthma comorbidity is part of atopic triad; type I biased immunology, mildly better response.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="atopic_dermatitis_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.15},
        coefficient=-0.05,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="AD comorbidity completes atopic triad; type I skew.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="autoimmune_thyroiditis_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Hashimoto thyroiditis correlates with anti-TPO+ subset (type IIb); modestly elevates failure.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="type1_diabetes_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.04},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="T1DM as autoimmune comorbidity marks broader autoimmune diathesis; rare so weak signal.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="systemic_lupus_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.03},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="SLE as autoimmune comorbidity; rare. Could complicate diagnosis (urticarial vasculitis differential).",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="psoriasis_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.06},
        coefficient=+0.03,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Psoriasis as autoimmune comorbidity; weak overlap signal.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="anxiety_or_depression_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.40},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Psychological comorbidity correlates with refractory CSU + worse adherence + amplified symptom perception (Maurer 2017 EAACI).",
        citation_strength="moderate",
    ),
    # Cluster 5: Prior treatment history (10) [indices 34-43]
    FeatureManifest(
        name="prior_h1_antihistamine_standard_dose",
        distribution="bernoulli",
        distribution_params={"p": 0.999},
        coefficient=0.0,
        monotone_direction=0,
        is_noise=True,
        clinical_justification="Inclusion-criterion feature (all patients failed H1 standard dose by definition); zero signal but preserved for cohort-definition audit. Marked is_noise=True because coefficient=0.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="prior_h1_antihistamine_4x_dose_failed",
        distribution="bernoulli",
        distribution_params={"p": 0.999},
        coefficient=0.0,
        monotone_direction=0,
        is_noise=True,
        clinical_justification="REMIX-1/2 inclusion criterion: failure of H1 4x standard dose. All patients in cohort by design; included for audit. is_noise=True since coefficient=0.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="months_on_h1_antihistamines_before_remib",
        distribution="normal",
        distribution_params={"loc": 16.0, "scale": 14.0},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Longer H1 exposure before BTK initiation marks chronic refractory disease + reduced spontaneous remission probability.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="prior_omalizumab_use",
        distribution="bernoulli",
        distribution_params={"p": 0.30},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Prior anti-IgE exposure marks more refractory disease; coefficient sign on failure mildly positive but interaction with response category complicates (responders may have rotated for other reasons).",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="prior_omalizumab_response_complete",
        distribution="bernoulli",
        distribution_params={"p": 0.10},
        coefficient=-0.20,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Patients who achieved complete response on omalizumab (then rotated for other reasons - discontinuation, access, etc.) carry favorable type-I biology profile + are more likely to respond to BTK.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="prior_cyclosporine_use",
        distribution="bernoulli",
        distribution_params={"p": 0.15},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Cyclosporine reserved for refractory CSU; prior exposure marks treatment-resistant subset.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="prior_cyclosporine_response_partial_or_complete",
        distribution="bernoulli",
        distribution_params={"p": 0.08},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Response to cyclosporine (even partial) marks T-cell-mediated autoimmune component; mildly favorable for BTK inhibition.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="prior_montelukast_use",
        distribution="bernoulli",
        distribution_params={"p": 0.40},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="LTRA add-on; commonly tried before biologics escalation. Weak refractoriness marker.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="prior_corticosteroid_burst_frequency_per_year",
        distribution="normal",
        distribution_params={"loc": 2.5, "scale": 2.5},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Frequent oral steroid bursts (>2/year) marks severe uncontrolled disease + treatment-resistant phenotype.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="prior_systemic_immunomodulator_failed_count",
        distribution="normal",
        distribution_params={"loc": 1.2, "scale": 1.0},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Number of prior systemic agents failed (omalizumab, cyclosporine, dapsone, dupilumab, etc.); higher count marks heavily-pretreated refractory cohort with reduced BTK response.",
        citation_strength="strong",
    ),
    # Cluster 6: Disease characteristics + triggers (8) [indices 44-51]
    FeatureManifest(
        name="trigger_pattern_idiopathic",
        distribution="bernoulli",
        distribution_params={"p": 0.65},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="No identifiable trigger (true 'spontaneous' CSU, the majority); modestly harder to control than identifiable-trigger subsets.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="trigger_pattern_pressure",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Concomitant delayed pressure urticaria; chronic inducible-urticaria overlap predicts non-response to monotherapy.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="trigger_pattern_dermatographism",
        distribution="bernoulli",
        distribution_params={"p": 0.22},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Symptomatic dermatographism overlap; weak negative response signal.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="autologous_serum_skin_test_positive",
        distribution="bernoulli",
        distribution_params={"p": 0.32},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="ASST+ is a screening marker for type IIb autoimmunity (basophil-activating autoantibodies in serum); modestly elevates failure on anti-IgE; effect on BTK weaker.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="csu_severity_score_clinician_assessed",
        distribution="normal",
        distribution_params={"loc": 3.2, "scale": 1.0},
        coefficient=+0.30,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Integrative 5-point clinician severity assessment (1=mild ... 5=very severe); captures gestalt judgment beyond instrument scores.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="number_of_flares_past_3_months",
        distribution="normal",
        distribution_params={"loc": 12.0, "scale": 8.0},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="High flare frequency marks active uncontrolled disease + worse short-horizon response.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="concurrent_chronic_inducible_urticaria",
        distribution="bernoulli",
        distribution_params={"p": 0.25},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Cold/cholinergic/aquagenic CIndU overlap with CSU; partial overlap with trigger_pattern_*; complicates response assessment.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="mast_cell_activation_syndrome_concurrent",
        distribution="bernoulli",
        distribution_params={"p": 0.04},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="MCAS comorbidity is rare but predicts non-response to anti-mediator therapy; included for completeness.",
        citation_strength="weak",
    ),
    # Cluster 7: Adherence + access (4) [indices 52-55]
    FeatureManifest(
        name="medication_adherence_score_baseline",
        distribution="normal",
        distribution_params={"loc": 0.78, "scale": 0.18},
        coefficient=-0.25,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Baseline adherence (proportion-of-days-covered last 6mo) is the single largest modifiable predictor of treatment response across chronic conditions; PDC <0.80 strongly elevates failure (Hershman 2010 BC analogue).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="specialty_dermatology_or_allergy_access",
        distribution="bernoulli",
        distribution_params={"p": 0.60},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Specialty care access (vs primary care only) correlates with timely escalation + better adherence + monitoring; modestly protective against failure.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="insurance_specialty_drug_coverage",
        distribution="bernoulli",
        distribution_params={"p": 0.78},
        coefficient=-0.08,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Coverage for specialty biologic (predictor of access continuity); modest protective channel via adherence + dose continuity.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="distance_to_specialty_care_miles",
        distribution="normal",
        distribution_params={"loc": 22.0, "scale": 22.0},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Geographic access barrier; longer distance weakly elevates failure via missed visits + delayed escalation.",
        citation_strength="weak",
    ),
    # Cluster 8: Noise (4) [indices 56-59]
    FeatureManifest(
        name="noise_admin_1",
        distribution="normal",
        distribution_params={"loc": 0.0, "scale": 1.0},
        coefficient=0.0,
        monotone_direction=0,
        is_noise=True,
        clinical_justification="Pure noise feature (admin code variation); zero coefficient by construction.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="noise_admin_2",
        distribution="normal",
        distribution_params={"loc": 0.0, "scale": 1.0},
        coefficient=0.0,
        monotone_direction=0,
        is_noise=True,
        clinical_justification="Pure noise feature.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="noise_admin_3",
        distribution="normal",
        distribution_params={"loc": 0.0, "scale": 1.0},
        coefficient=0.0,
        monotone_direction=0,
        is_noise=True,
        clinical_justification="Pure noise feature.",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="noise_admin_4",
        distribution="normal",
        distribution_params={"loc": 0.0, "scale": 1.0},
        coefficient=0.0,
        monotone_direction=0,
        is_noise=True,
        clinical_justification="Pure noise feature.",
        citation_strength="weak",
    ),
)


# Per dgp.py overlap rejection: each column may belong to at most one block.
# Shard 05 §B.3 lists 16 blocks but several cross-cluster blocks overlap with
# within-cluster blocks. Following the codex I-1 rule (within-cluster
# clinically-mandated correlations take precedence), the implementation
# drops the following overlapping blocks documented in the shard:
#   - ([8, 11], -0.7)  uas7 <-> uct      [overlaps with [8, 9, 10] subscores]
#   - ([8, 21, 22], 0.4) uas7<->crp<->d_dimer [overlaps with [8, 9, 10]]
#   - ([19, 29], 0.5) anti_tpo<->thyroiditis  [overlaps with [19, 20]]
#   - ([46, 50], 0.3) dermatographism<->cind   [overlaps with [45, 46, 47]]
#   - ([53, 55], -0.4) specialty<->distance    [overlaps with [53, 54]]
# Action item: shard 05 §B.3 prose update needed to either split the conflicting
# blocks into non-overlapping unions at uniform r OR explicitly mark the
# overlap-dropped blocks as "phase 2" deferrals.
SCENARIO_C_CORRELATION_BLOCKS: list[tuple[list[int], float]] = [
    # Cluster 1: age cluster — diagnosis age + duration -> assessment age
    ([0, 3, 7], 0.7),
    # Cluster 2: UAS7 cluster — total <-> pruritus + hives subscores (sum constraint)
    ([8, 9, 10], 0.85),
    # Cluster 2: angioedema burden — AAS7 conditional on angioedema-present
    ([12, 13], 0.6),
    # Cluster 2: QoL cluster — DLQI <-> CU-Q2oL <-> sleep <-> work productivity
    ([14, 15, 16, 17], 0.7),
    # Cluster 3: type IIb autoimmunity cluster — anti-TPO <-> anti-IgE/anti-FcepsilonRI
    ([19, 20], 0.6),
    # Cluster 3: type I cluster — IgE <-> eosinophil count
    ([18, 23], 0.4),
    # Cluster 4: atopic triad
    ([26, 27, 28], 0.5),
    # Cluster 5: refractoriness ladder
    ([36, 37, 39, 43], 0.5),
    # Cluster 5/6: prior steroid bursts <-> severity score
    ([42, 48], 0.4),
    # Cluster 6: trigger overlap — pressure <-> dermatographism <-> ASST+
    ([45, 46, 47], 0.4),
    # Cluster 7: access cluster — specialty access <-> insurance
    ([53, 54], 0.5),
]


class ScenarioCBuilder(ScenarioBuilder):
    @property
    def name(self) -> ScenarioName:
        return ScenarioName.C_TREATMENT_CSU_RESPONSE

    @property
    def target_prevalence(self) -> float:
        return 0.40

    @property
    def target_auc_band(self) -> tuple[float, float]:
        return (0.82, 0.88)

    @property
    def n_features(self) -> int:
        return 60

    @property
    def correlation_strength(self) -> float:
        return 0.50

    @property
    def slope_multiplier(self) -> float:
        return SLOPE_MULTIPLIER

    @property
    def feature_manifest(self) -> tuple[FeatureManifest, ...]:
        return SCENARIO_C_MANIFEST

    @property
    def default_n_total(self) -> int:
        return 6000

    @property
    def correlation_blocks(self) -> list[tuple[list[int], float]]:
        return SCENARIO_C_CORRELATION_BLOCKS

    def sample_features(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cols = [
            sample_one_feature(rng, n, m.distribution, m.distribution_params)
            for m in self.feature_manifest
        ]
        return np.column_stack(cols)


SCENARIO_REGISTRY[ScenarioName.C_TREATMENT_CSU_RESPONSE] = ScenarioCBuilder
