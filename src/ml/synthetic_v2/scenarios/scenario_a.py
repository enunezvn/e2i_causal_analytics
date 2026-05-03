"""Scenario A — HR+/HER2- early breast cancer 5-yr iDFS (Kisqali franchise).

Per shard 03 (.claude/plans/synthetic_data_generator_v2/03-scenario-a-diagnostic-bc-idfs.md):

- 40 features across 6 clinical clusters + 1 noise cluster.
- target_prevalence = 0.20 (NATALEE 5y iDFS placebo-arm event rate).
- target_auc_band = (0.78, 0.83) (matches MINDACT/TAILORx/Sammut 2022).
- primary_tau = 0.20 (NATALEE high-risk eligibility threshold).
- Use case: diagnostic; clinical_threshold_range τ ∈ [0.05, 0.30].

Slope multiplier calibration: ``SLOPE_MULTIPLIER`` is calibrated once at
implementation time so median LR test AUC over 10 seeds lands at ~0.805
(band midpoint) with at least 9/10 seeds in the band. The locked value
+ calibration date are recorded in the constant's inline comment below.
"""

from __future__ import annotations

import numpy as np

from src.ml.synthetic_v2.dgp import sample_one_feature
from src.ml.synthetic_v2.manifest import FeatureManifest
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.scenarios._base import ScenarioBuilder

SLOPE_MULTIPLIER: float = 0.67  # calibrated 2026-05-03 via §B.4 bisection (post full-cohort standardization fix): 9/10 seeds in [0.78, 0.83] AUC band, median 0.793.
# Bisection log (after api.py full-cohort standardization fix):
#   slope=0.65 -> 5/10 in band, median 0.784
#   slope=0.66 -> 7/10 in band, median 0.794
#   slope=0.67 -> 9/10 in band, median 0.793  *** LOCKED ***
#   slope=0.68 -> 7/10 in band, median 0.817
# Action item: shard 03 §B.4 prose should be updated to reflect that
# manifest coefficients work on standardized (unit-variance) features per
# the api.py step-3 standardization fix; initial estimate ~0.55 is now ~0.67.

SCENARIO_A_MANIFEST: tuple[FeatureManifest, ...] = (
    # Cluster 1: Demographics + reproductive history (5)
    FeatureManifest(
        name="age_at_diagnosis_years",
        distribution="normal",
        distribution_params={"loc": 58.0, "scale": 12.0},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Older age at HR+/HER2- early BC diagnosis correlates with modestly higher recurrence-or-death event in 5y window (cumulative competing-risk death contribution); EBCTCG 2005 meta-analysis.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="menopause_status_postmenopausal",
        distribution="bernoulli",
        distribution_params={"p": 0.65},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Postmenopausal status shifts endocrine-therapy choice (AI vs tamoxifen) and modifies recurrence kinetics; net direction modestly positive on 5y iDFS event vs premenopausal-with-OFS arm (BIG 1-98; SOFT/TEXT).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="bmi_kg_m2",
        distribution="normal",
        distribution_params={"loc": 28.5, "scale": 6.0},
        coefficient=+0.18,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Obesity (BMI >=30) raises endogenous estrogen via aromatase, increasing HR+ recurrence; Protani 2010 meta-analysis (HR ~1.33 for obese vs normal-weight).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="family_hx_first_degree_breast_cancer",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="First-degree FHx raises baseline recurrence/contralateral primary risk; Brewer 2017 BMJ pooled cohort.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="ecog_performance_status_geq_1",
        distribution="bernoulli",
        distribution_params={"p": 0.22},
        coefficient=+0.08,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="ECOG >=1 at diagnosis correlates with frailty + competing-risk death contribution to iDFS event; modest in early BC where most are ECOG 0-1.",
        citation_strength="moderate",
    ),
    # Cluster 2: Tumor characteristics (8)
    FeatureManifest(
        name="tumor_size_mm",
        distribution="normal",
        distribution_params={"loc": 22.0, "scale": 12.0},
        coefficient=+0.40,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="T-stage proxy; tumor size >=20mm strongly predicts recurrence (NATALEE Stage II-III high-risk eligibility hinges on T2+ AND nodal positivity OR T3+).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="tumor_grade_3",
        distribution="bernoulli",
        distribution_params={"p": 0.30},
        coefficient=+0.45,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Nottingham grade 3 (poorly differentiated) is a NATALEE high-risk inclusion modifier; strongest single histopathologic predictor of early recurrence.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="nodal_status_positive",
        distribution="bernoulli",
        distribution_params={"p": 0.55},
        coefficient=+0.50,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="N+ status is the single highest-leverage feature in HR+/HER2- early BC recurrence; monarchE eligibility requires N+ disease (>=4 nodes OR 1-3 nodes with high-risk features).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="n_positive_nodes",
        distribution="normal",
        distribution_params={"loc": 1.8, "scale": 2.5},
        coefficient=+0.35,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Continuous nodal-burden marker beyond binary N+; >=4 positive nodes (N2+) is a separate stratification cut in monarchE / NATALEE.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="lvi_present",
        distribution="bernoulli",
        distribution_params={"p": 0.32},
        coefficient=+0.25,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Lymphovascular invasion is an independent recurrence predictor; Rakha 2012 meta-analysis (HR ~1.6 for distant recurrence).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="multifocal_disease",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Multifocal/multicentric disease modestly elevates recurrence; effect attenuates after adjustment for size + nodal.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="histology_lobular",
        distribution="bernoulli",
        distribution_params={"p": 0.15},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Invasive lobular carcinoma has a delayed-recurrence kinetic distinct from ductal; net 5y iDFS event rate slightly elevated due to higher contralateral primary risk; Pestalozzi 2008 EBCTCG.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="ki67_percent",
        distribution="normal",
        distribution_params={"loc": 22.0, "scale": 14.0},
        coefficient=+0.40,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Ki-67 proliferation index >20% defines luminal-B-like biology with elevated early-recurrence risk; NATALEE and monarchE both stratify on Ki-67 as a high-risk modifier.",
        citation_strength="strong",
    ),
    # Cluster 3: Biomarkers (HR / HER2 / genomic) (7)
    FeatureManifest(
        name="er_h_score",
        distribution="normal",
        distribution_params={"loc": 240.0, "scale": 50.0},
        coefficient=-0.20,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Higher ER H-score (intensity x % positive cells, 0-300) marks robust hormone-receptor dependence and better endocrine-therapy response; lower iDFS event probability (EBCTCG 2011 endocrine-therapy meta-analysis).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="pr_h_score",
        distribution="normal",
        distribution_params={"loc": 180.0, "scale": 80.0},
        coefficient=-0.15,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="PR positivity within HR+ disease confers additional prognostic favor; PR-low/negative HR+ (~25% of HR+) has worse iDFS than PR-high.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="her2_low_status",
        distribution="bernoulli",
        distribution_params={"p": 0.55},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="HER2-low (IHC 1+ or 2+/ISH-) is an emerging biomarker subset with marginally distinct biology from HER2-zero per DESTINY-Breast04 (Modi 2022); for the early-BC iDFS endpoint the prognostic effect is modest.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="oncotype_rs",
        distribution="normal",
        distribution_params={"loc": 22.0, "scale": 12.0},
        coefficient=+0.45,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Oncotype DX recurrence score is the dominant genomic predictor; TAILORx (Sparano 2018) validated RS as a recurrence-risk continuum, with RS >25 driving the strongest distant-recurrence signal.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="mammaprint_high_risk",
        distribution="bernoulli",
        distribution_params={"p": 0.35},
        coefficient=+0.30,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="MammaPrint 70-gene high-risk classification predicts distant recurrence at 5y; MINDACT (Cardoso 2016) showed clinical-high/genomic-high cohort has the worst 5y DRFS.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="genomic_instability_index",
        distribution="normal",
        distribution_params={"loc": 0.0, "scale": 1.0},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Z-scored composite of homologous-recombination-deficiency / chromosomal-instability signatures (e.g., HRD score, Tutt 2018); modestly elevated for harder-to-treat HR+ subset.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="tils_percent",
        distribution="normal",
        distribution_params={"loc": 8.0, "scale": 6.0},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Stromal tumor-infiltrating lymphocytes have a weak protective signal in HR+/HER2- (much stronger in TNBC and HER2+); included for completeness - Loi 2019 Lancet Oncol.",
        citation_strength="weak",
    ),
    # Cluster 4: Treatment + adherence (8)
    FeatureManifest(
        name="received_adjuvant_chemotherapy",
        distribution="bernoulli",
        distribution_params={"p": 0.55},
        coefficient=-0.15,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Adjuvant chemo reduces 5y recurrence in higher-risk HR+/HER2- (EBCTCG 2012); coefficient reflects within-cohort treatment-effect signal after adjustment for selection.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="chemotherapy_anthracycline_taxane_regimen",
        distribution="bernoulli",
        distribution_params={"p": 0.40},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="AC-T or TC-style regimen (vs. less intense) modestly reduces residual recurrence among chemo recipients; signal is modest because of confounding-by-indication.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="received_endocrine_therapy",
        distribution="bernoulli",
        distribution_params={"p": 0.95},
        coefficient=-0.30,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Endocrine therapy (5-10y AI or tamoxifen +/- OFS) is the backbone for HR+ early BC; non-receipt strongly elevates recurrence risk (EBCTCG 2011 meta-analysis: ~50% recurrence reduction over 15y).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="endocrine_therapy_ai_vs_tamoxifen",
        distribution="bernoulli",
        distribution_params={"p": 0.70},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="AI superior to tamoxifen in postmenopausal HR+ early BC for 5y DRFS (BIG 1-98); modest absolute advantage at 5y (~2-3% DFS gap).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="received_ovarian_function_suppression",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=-0.10,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="OFS + AI in premenopausal high-risk HR+ early BC reduces recurrence (SOFT/TEXT 2014); subset effect, conditional on premenopausal eligibility.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="received_adjuvant_radiation",
        distribution="bernoulli",
        distribution_params={"p": 0.72},
        coefficient=-0.08,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Adjuvant RT post-BCS or post-mastectomy with N+ reduces loco-regional recurrence component of iDFS (EBCTCG 2014).",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="received_cdk46_inhibitor_adjuvant",
        distribution="bernoulli",
        distribution_params={"p": 0.20},
        coefficient=-0.20,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Adjuvant CDK4/6 inhibitor (ribociclib in NATALEE eligible; abemaciclib in monarchE eligible) reduces 5y iDFS event in high-risk HR+/HER2- early BC; KISQALI franchise anchor. NATALEE 3y iDFS HR ~0.75; monarchE 4y iDFS HR ~0.66.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="endocrine_adherence_score_year1",
        distribution="normal",
        distribution_params={"loc": 0.78, "scale": 0.18},
        coefficient=-0.20,
        monotone_direction=-1,
        is_noise=False,
        clinical_justification="Endocrine-therapy adherence (proportion-of-days-covered) in year 1 strongly predicts 5y recurrence; non-adherence (PDC <0.80) carries HR ~1.5 for recurrence (Hershman 2010).",
        citation_strength="strong",
    ),
    # Cluster 5: Comorbidity + lifestyle (4)
    FeatureManifest(
        name="diabetes_t2",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=+0.12,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="T2DM modestly elevates BC recurrence and competing-risk death; mechanism via insulin/IGF-1 axis + adherence interactions (Lipscombe 2012).",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="cardiovascular_disease_pmh",
        distribution="bernoulli",
        distribution_params={"p": 0.22},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="PMH of CVD elevates competing-risk death contribution to 5y iDFS; also modulates AI vs tamoxifen choice (AI VTE risk).",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="osteoporosis_at_diagnosis",
        distribution="bernoulli",
        distribution_params={"p": 0.15},
        coefficient=+0.05,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Osteoporosis at diagnosis modulates AI tolerance + bone-modifying therapy use; weak direct iDFS signal but interaction with adjuvant zoledronate (AZURE).",
        citation_strength="weak",
    ),
    FeatureManifest(
        name="current_smoker",
        distribution="bernoulli",
        distribution_params={"p": 0.14},
        coefficient=+0.10,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Active smoking at BC diagnosis elevates recurrence + non-BC mortality; Pierce 2014 EBCTCG-adjacent meta-analysis.",
        citation_strength="moderate",
    ),
    # Cluster 6: Surgery + initial risk-stratification (4)
    FeatureManifest(
        name="surgical_margin_re_excision_required",
        distribution="bernoulli",
        distribution_params={"p": 0.12},
        coefficient=+0.15,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Positive/close initial margin requiring re-excision marks more aggressive disease + retained microscopic disease; modestly elevates loco-regional component of iDFS event.",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="days_from_surgery_to_first_systemic",
        distribution="normal",
        distribution_params={"loc": 45.0, "scale": 18.0},
        coefficient=+0.12,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Delay >60 days from surgery to first systemic therapy associated with worse 5y outcomes in HR+/HER2- (Gagliato 2014).",
        citation_strength="moderate",
    ),
    FeatureManifest(
        name="initial_clinical_stage_iii",
        distribution="bernoulli",
        distribution_params={"p": 0.20},
        coefficient=+0.30,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Stage III (vs IIA/IIB) is the strongest single AJCC-stage cutpoint for 5y recurrence; NATALEE Stage III subset has the largest absolute iDFS benefit from ribociclib.",
        citation_strength="strong",
    ),
    FeatureManifest(
        name="neoadjuvant_chemotherapy_received_no_pcr",
        distribution="bernoulli",
        distribution_params={"p": 0.18},
        coefficient=+0.20,
        monotone_direction=+1,
        is_noise=False,
        clinical_justification="Receipt of neoadjuvant chemo *without* pCR (RCB >=1) marks chemo-resistant biology + elevated residual recurrence risk in HR+/HER2- (where pCR is rare ~10%); composite encoding.",
        citation_strength="strong",
    ),
    # Cluster 7: Noise (4)
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


SCENARIO_A_CORRELATION_BLOCKS: list[tuple[list[int], float]] = [
    # Cluster 1: age <-> menopause (postmenopausal frequency rises with age)
    ([0, 1], 0.5),
    # Cluster 2: tumor characteristics — size <-> grade <-> nodal status <-> Ki67
    ([5, 6, 7, 8, 12], 0.5),
    # Cluster 2 sub: LVI <-> multifocal (modest)
    ([9, 10], 0.3),
    # Cluster 3: ER <-> PR (HR co-expression)
    ([13, 14], 0.6),
    # Cluster 3: Oncotype RS <-> MammaPrint (genomic agreement)
    ([16, 17], 0.5),
    # Cluster 4: chemo ladder — adjuvant chemo <-> AC-T regimen
    ([20, 21], 0.7),
    # Cluster 4: endocrine ladder — endocrine <-> AI/tamoxifen choice <-> OFS
    ([22, 23, 24], 0.4),
    # Cluster 6: stage III <-> neoadjuvant-no-pCR (high-risk biology)
    ([34, 35], 0.4),
    # Cluster 5/6: comorbidity light-correlation
    ([28, 29], 0.3),
]
# Note: shard 03 §B.3 also lists ([12, 16], 0.4) ki67 <-> oncotype_rs but that
# overlaps with the index-12 entry in the larger Cluster 2 block ([5,6,7,8,12])
# and the index-16 entry in the genomic block ([16, 17]). The base DGP rejects
# overlapping blocks (commit 03), so this cross-block correlation is dropped at
# implementation time. Retaining it would require a single combined block
# [5,6,7,8,12,16,17] with a uniform r — distorting the within-cluster r=0.5 and
# r=0.5 specifications. Action item: shard 03 §B.3 prose update needed.


class ScenarioABuilder(ScenarioBuilder):
    @property
    def name(self) -> ScenarioName:
        return ScenarioName.A_DIAGNOSTIC_BC_IDFS

    @property
    def target_prevalence(self) -> float:
        return 0.20

    @property
    def target_auc_band(self) -> tuple[float, float]:
        return (0.78, 0.83)

    @property
    def n_features(self) -> int:
        return 40

    @property
    def correlation_strength(self) -> float:
        return 0.30  # average across blocks

    @property
    def slope_multiplier(self) -> float:
        return SLOPE_MULTIPLIER

    @property
    def feature_manifest(self) -> tuple[FeatureManifest, ...]:
        return SCENARIO_A_MANIFEST

    @property
    def default_n_total(self) -> int:
        return 6000

    @property
    def correlation_blocks(self) -> list[tuple[list[int], float]]:
        return SCENARIO_A_CORRELATION_BLOCKS

    def sample_features(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cols = [
            sample_one_feature(rng, n, m.distribution, m.distribution_params)
            for m in self.feature_manifest
        ]
        return np.column_stack(cols)


# Auto-register at import time so api.generate_scenario can dispatch.
SCENARIO_REGISTRY[ScenarioName.A_DIAGNOSTIC_BC_IDFS] = ScenarioABuilder
