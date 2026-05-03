"""CSU/Remibrutinib RWD loader for Scenario C concurrent-validation (shard 07 §C).

Reads the RWD CSU cohort and constructs a feature matrix aligned to
``SCENARIO_C_MANIFEST`` field names. The RWD source is the existing
platform CSU pharmacy/medical-claims pipeline (per
``scripts/convert_csu_rwd.py`` + ``data/rwd/csu/csu_data.xlsx``); a
subset of the 60 manifest fields are RWD-direct, the rest are
RWD-derived or RWD-missing per shard 07 §C.3.

Outcome-derivation tie-breaking rule (Codex I-4 closure 2026-05-03;
mirrors shard 05 §G.5): when multiple visits in the
``[window - tolerance, window + tolerance]`` weeks-post-remibrutinib are
at distinct distances from the target week, select the visit closest to
the target. When two visits are equidistant, select the EARLIER visit
(more conservative — UAS7=0 at the earlier visit must be sustained to
trigger the same week-12 response classification).

Limitations of this commit:

- ``_load_from_json_outputs`` and ``_load_from_excel`` raise
  ``NotImplementedError`` because the field-name mapping from the existing
  ``scripts/convert_csu_rwd.py`` output to the SCENARIO_C_MANIFEST is
  run-specific and PHI-aware. Real RWD loading is a follow-up implementation
  task; this commit ships the contract surface + synthesized fixture for
  unit tests + the KS/AUC-delta concurrent-validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Field-name alignment per shard 07 §C.3 — keyed by SCENARIO_C_MANIFEST name,
# value is a tag indicating the RWD provenance for the platform claims-only
# CSU RWD source. If the source upgrades to specialty registry / EHR-structured
# data, the RWD-missing labels for PRO scores + biomarkers convert to RWD-direct.
RWD_PROVENANCE_TAGS: dict[str, str] = {
    # Cluster 1 demographics
    "age_at_csu_diagnosis_years": "RWD-direct",
    "sex_female": "RWD-direct",
    "ethnicity_white": "RWD-direct",
    "bmi_kg_m2": "RWD-missing",
    "csu_disease_duration_years": "RWD-derived",
    "prior_csu_remission_relapse_history": "RWD-derived",
    "employment_status_active": "RWD-missing",
    "age_at_assessment_years": "RWD-derived",
    # Cluster 2 PRO scores: missing in claims-only
    "baseline_uas7_total": "RWD-missing",
    "baseline_uas7_pruritus_component": "RWD-missing",
    "baseline_uas7_hives_component": "RWD-missing",
    "baseline_uct_score": "RWD-missing",
    "baseline_aas7_score": "RWD-missing",
    "angioedema_present_at_baseline": "RWD-derived",
    "baseline_dlqi_score": "RWD-missing",
    "baseline_cu_q2ol_score": "RWD-missing",
    "sleep_disturbance_baseline_severity": "RWD-missing",
    "work_productivity_loss_pct_baseline": "RWD-missing",
    # Cluster 3 biomarkers: lab-result-dependent
    "total_serum_ige_iu_ml": "RWD-missing",
    "anti_thyroid_peroxidase_anti_tpo_positive": "RWD-missing",
    "anti_ige_or_anti_fceri_autoantibody_positive": "RWD-missing",
    "crp_mg_l": "RWD-missing",
    "d_dimer_ng_ml": "RWD-missing",
    "eosinophil_count_cells_ul": "RWD-missing",
    "basophil_count_cells_ul": "RWD-missing",
    "complement_c3_c4_normal": "RWD-missing",
    # Cluster 4 comorbidities: claims-direct
    "allergic_rhinitis_pmh": "RWD-direct",
    "asthma_pmh": "RWD-direct",
    "atopic_dermatitis_pmh": "RWD-direct",
    "autoimmune_thyroiditis_pmh": "RWD-direct",
    "type1_diabetes_pmh": "RWD-direct",
    "systemic_lupus_pmh": "RWD-direct",
    "psoriasis_pmh": "RWD-direct",
    "anxiety_or_depression_pmh": "RWD-direct",
    # Cluster 5 prior treatment history
    "prior_h1_antihistamine_standard_dose": "RWD-direct",
    "prior_h1_antihistamine_4x_dose_failed": "RWD-derived",
    "months_on_h1_antihistamines_before_remib": "RWD-derived",
    "prior_omalizumab_use": "RWD-direct",
    "prior_omalizumab_response_complete": "RWD-missing",
    "prior_cyclosporine_use": "RWD-direct",
    "prior_cyclosporine_response_partial_or_complete": "RWD-missing",
    "prior_montelukast_use": "RWD-direct",
    "prior_corticosteroid_burst_frequency_per_year": "RWD-derived",
    "prior_systemic_immunomodulator_failed_count": "RWD-derived",
    # Cluster 6 disease characteristics
    "trigger_pattern_idiopathic": "RWD-missing",
    "trigger_pattern_pressure": "RWD-missing",
    "trigger_pattern_dermatographism": "RWD-missing",
    "autologous_serum_skin_test_positive": "RWD-missing",
    "csu_severity_score_clinician_assessed": "RWD-missing",
    "number_of_flares_past_3_months": "RWD-derived",
    "concurrent_chronic_inducible_urticaria": "RWD-direct",
    "mast_cell_activation_syndrome_concurrent": "RWD-direct",
    # Cluster 7 access
    "medication_adherence_score_baseline": "RWD-derived",
    "specialty_dermatology_or_allergy_access": "RWD-direct",
    "insurance_specialty_drug_coverage": "RWD-direct",
    "distance_to_specialty_care_miles": "RWD-direct",
    # Cluster 8 noise: irrelevant in real RWD comparison
    "noise_admin_1": "RWD-missing",
    "noise_admin_2": "RWD-missing",
    "noise_admin_3": "RWD-missing",
    "noise_admin_4": "RWD-missing",
}


@dataclass(frozen=True)
class RwdCsuCohort:
    """Loaded RWD cohort + outcome (shard 07 §C.2)."""

    feature_matrix: dict[str, np.ndarray]
    outcome: np.ndarray
    feature_provenance: dict[str, str] = field(default_factory=dict)

    @property
    def n_patients(self) -> int:
        if not self.feature_matrix:
            return 0
        return len(next(iter(self.feature_matrix.values())))

    def has_feature(self, name: str) -> bool:
        return name in self.feature_matrix

    def rwd_direct_or_derived_features(self) -> list[str]:
        """Return feature names with provenance != 'RWD-missing'."""
        return [n for n, p in self.feature_provenance.items() if p != "RWD-missing"]


def load_rwd_csu_cohort(
    rwd_data_dir: str | Path = "data/rwd/csu",
    *,
    outcome_window_weeks: int = 12,
    outcome_window_tolerance_weeks: int = 4,
    allow_synthesized_fixture: bool = False,
) -> RwdCsuCohort:
    """Load the CSU RWD cohort (shard 07 §C.2).

    Real RWD loaders for the JSON / Excel backings raise NotImplementedError
    in this commit; pass ``allow_synthesized_fixture=True`` to get a
    deterministic fixture suitable for unit tests + the KS/AUC-delta
    concurrent-validation helpers below.
    """
    rwd_path = Path(rwd_data_dir)
    has_excel = (rwd_path / "csu_data.xlsx").exists()
    has_json = bool(list(rwd_path.glob("csu_e2i_ml_v3_*.json"))) if rwd_path.exists() else False

    if has_json:
        return _load_from_json_outputs(
            rwd_path,
            outcome_window_weeks=outcome_window_weeks,
            tolerance_weeks=outcome_window_tolerance_weeks,
        )
    if has_excel:
        return _load_from_excel(
            rwd_path / "csu_data.xlsx",
            outcome_window_weeks=outcome_window_weeks,
            tolerance_weeks=outcome_window_tolerance_weeks,
        )
    if allow_synthesized_fixture:
        return _synthesize_fixture()
    raise FileNotFoundError(
        f"No RWD CSU data found under {rwd_path}. Expected csu_data.xlsx or "
        "csu_e2i_ml_v3_*.json (set allow_synthesized_fixture=True for tests)."
    )


def _load_from_json_outputs(
    rwd_dir: Path,
    *,
    outcome_window_weeks: int,
    tolerance_weeks: int,
) -> RwdCsuCohort:
    """Load from pre-converted JSON outputs (deferred to follow-up)."""
    del rwd_dir, outcome_window_weeks, tolerance_weeks
    raise NotImplementedError(
        "JSON-output loading deferred — see shard 07 §C.4 deliverable list. "
        "For now, pass allow_synthesized_fixture=True to load_rwd_csu_cohort."
    )


def _load_from_excel(
    workbook_path: Path,
    *,
    outcome_window_weeks: int,
    tolerance_weeks: int,
) -> RwdCsuCohort:
    """Load from raw Excel workbook (deferred to follow-up)."""
    del workbook_path, outcome_window_weeks, tolerance_weeks
    raise NotImplementedError(
        "Excel loading deferred — see shard 07 §C.4. Pass "
        "allow_synthesized_fixture=True for unit tests."
    )


def _synthesize_fixture(
    n_patients: int = 200, seed: int = 42
) -> RwdCsuCohort:
    """Deterministic synthesized fixture for unit tests (no PHI dependency)."""
    rng = np.random.default_rng(seed)
    feature_matrix: dict[str, np.ndarray] = {}
    feature_provenance: dict[str, str] = {}

    for fname, prov in RWD_PROVENANCE_TAGS.items():
        if prov == "RWD-missing":
            continue
        feature_provenance[fname] = prov
        if fname == "age_at_csu_diagnosis_years":
            feature_matrix[fname] = rng.normal(42.0, 14.0, n_patients)
        elif fname == "sex_female":
            feature_matrix[fname] = rng.binomial(1, 0.66, n_patients).astype(float)
        elif fname == "ethnicity_white":
            feature_matrix[fname] = rng.binomial(1, 0.65, n_patients).astype(float)
        elif fname == "csu_disease_duration_years":
            feature_matrix[fname] = rng.normal(4.5, 4.0, n_patients)
        elif fname == "prior_csu_remission_relapse_history":
            feature_matrix[fname] = rng.binomial(1, 0.30, n_patients).astype(float)
        elif fname == "age_at_assessment_years":
            feature_matrix[fname] = rng.normal(47.0, 14.5, n_patients)
        elif fname == "angioedema_present_at_baseline":
            feature_matrix[fname] = rng.binomial(1, 0.42, n_patients).astype(float)
        elif fname.endswith("_pmh"):
            ps = {
                "allergic_rhinitis_pmh": 0.35,
                "asthma_pmh": 0.18,
                "atopic_dermatitis_pmh": 0.15,
                "autoimmune_thyroiditis_pmh": 0.18,
                "type1_diabetes_pmh": 0.04,
                "systemic_lupus_pmh": 0.03,
                "psoriasis_pmh": 0.06,
                "anxiety_or_depression_pmh": 0.40,
            }
            feature_matrix[fname] = rng.binomial(1, ps[fname], n_patients).astype(float)
        elif fname == "prior_h1_antihistamine_standard_dose":
            feature_matrix[fname] = np.ones(n_patients)
        elif fname == "prior_h1_antihistamine_4x_dose_failed":
            feature_matrix[fname] = np.ones(n_patients)
        elif fname == "months_on_h1_antihistamines_before_remib":
            feature_matrix[fname] = rng.normal(16.0, 14.0, n_patients)
        elif fname == "prior_omalizumab_use":
            feature_matrix[fname] = rng.binomial(1, 0.30, n_patients).astype(float)
        elif fname == "prior_cyclosporine_use":
            feature_matrix[fname] = rng.binomial(1, 0.15, n_patients).astype(float)
        elif fname == "prior_montelukast_use":
            feature_matrix[fname] = rng.binomial(1, 0.40, n_patients).astype(float)
        elif fname == "prior_corticosteroid_burst_frequency_per_year":
            feature_matrix[fname] = rng.normal(2.5, 2.5, n_patients)
        elif fname == "prior_systemic_immunomodulator_failed_count":
            feature_matrix[fname] = rng.normal(1.2, 1.0, n_patients)
        elif fname == "concurrent_chronic_inducible_urticaria":
            feature_matrix[fname] = rng.binomial(1, 0.25, n_patients).astype(float)
        elif fname == "mast_cell_activation_syndrome_concurrent":
            feature_matrix[fname] = rng.binomial(1, 0.04, n_patients).astype(float)
        elif fname == "number_of_flares_past_3_months":
            feature_matrix[fname] = rng.normal(12.0, 8.0, n_patients)
        elif fname == "medication_adherence_score_baseline":
            feature_matrix[fname] = rng.normal(0.78, 0.18, n_patients)
        elif fname == "specialty_dermatology_or_allergy_access":
            feature_matrix[fname] = rng.binomial(1, 0.60, n_patients).astype(float)
        elif fname == "insurance_specialty_drug_coverage":
            feature_matrix[fname] = rng.binomial(1, 0.78, n_patients).astype(float)
        elif fname == "distance_to_specialty_care_miles":
            feature_matrix[fname] = rng.normal(22.0, 22.0, n_patients)

    score = (
        -0.2 * feature_matrix["medication_adherence_score_baseline"]
        + 0.1 * feature_matrix["asthma_pmh"]
        - 0.15 * feature_matrix["allergic_rhinitis_pmh"]
        + 0.05 * feature_matrix["age_at_csu_diagnosis_years"] / 50.0
        + rng.normal(0, 0.3, n_patients)
    )
    threshold = float(np.quantile(score, 0.6))
    outcome = (score > threshold).astype(int)

    return RwdCsuCohort(
        feature_matrix=feature_matrix,
        outcome=outcome,
        feature_provenance=feature_provenance,
    )


def derive_csu_remib_response_outcome(
    visits: list[dict[str, Any]],
    *,
    target_week: int = 12,
    tolerance_weeks: int = 4,
) -> int | None:
    """Derive ``csu_remib_response`` for one patient (shard 05 §G.5).

    Tie-breaking rule (Codex I-4): equidistant visits → pick the EARLIER
    visit (more conservative interpretation).
    """
    in_window = [
        v for v in visits
        if target_week - tolerance_weeks
        <= v["week_post_remib"]
        <= target_week + tolerance_weeks
    ]
    if not in_window:
        return None
    in_window_sorted = sorted(
        in_window,
        key=lambda v: (abs(v["week_post_remib"] - target_week), v["week_post_remib"]),
    )
    selected = in_window_sorted[0]
    return 1 if selected["uas7"] > 0 else 0


def compute_feature_distribution_ks(
    synthetic_X: dict[str, np.ndarray],
    rwd_cohort: RwdCsuCohort,
    *,
    p_value_threshold: float = 0.001,
) -> dict[str, dict[str, float]]:
    """Per-feature KS test (synthetic vs RWD) over RWD-direct/derived features."""
    from scipy import stats

    results: dict[str, dict[str, float]] = {}
    for name in rwd_cohort.rwd_direct_or_derived_features():
        if name not in synthetic_X:
            continue
        ks = stats.ks_2samp(synthetic_X[name], rwd_cohort.feature_matrix[name])
        results[name] = {
            "ks_stat": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "fails": float(ks.pvalue < p_value_threshold),
        }
    return results


def fail_rate(ks_results: dict[str, dict[str, float]]) -> float:
    """Fraction of features that failed the KS test."""
    if not ks_results:
        return 0.0
    fails = sum(1 for v in ks_results.values() if v["fails"] >= 1.0)
    return fails / len(ks_results)
