"""Synthetic-CSU tier0 feature manifest — declared-safe BY CONSTRUCTION.

Catalogs the feature columns ``scripts/export_synthetic_tier0.py`` emits into
the tier0 contract dirs (``data/rwd/synthetic_<SUFFIX>/tier0/<cohort>/``) for
the synthetic causal-validation dataset (docs/data/SYNTHETIC-CAUSAL-DATA-GUIDE.md).

Every declared feature is generated BEFORE the outcome draw, so pre-index
declarations are correct by construction:

- Patient grain: confounders and covariates are drawn first
  (``patient_generator.py:262-287``), the treatment arm is assigned from them
  (``dgp/treatment_arm.py:27-47``), and only then is the outcome generated
  (``binary_outcome_with_cate``). ``segment_assignment`` is a deterministic
  function of pre-index ``disease_severity``.
- HCP grain: network centrality is exogenous lognormal topology generated
  before the arm and the adoption draw (``hcp_adoption_artifact.py:103-124``;
  leak-safety documented at lines 22-25).

Without these declarations the statistical/LLM leakage layers false-positively
drop the DESIGNED causal drivers (observed 2026-06-10: ``disease_severity``
dropped on initiation; ``influence_network_size`` + ``peer_influence_score``
dropped on hcp_adoption, collapsing AUC 0.78 -> 0.51 because the remaining
columns are noise w.r.t. the synthetic DGP).

The DGP answer-key columns (``propensity_score``, ``treatment_effect_estimate``)
and the outcome-derived ``days_to_treatment`` are FORBIDDEN as features — the
exporter already excludes them from the tier0 frames; the forbidden list is
defense-in-depth should they ever reach a frame.

Like the ``synthetic`` fixture manifest, this is correctness BY CONSTRUCTION,
not RWD positive evidence. A real cohort registers against the CSU or Optum
manifests instead.
"""

from __future__ import annotations

from src.data.feature_contract import FeatureContract, KnowableAt

_ENROLLMENT = KnowableAt(reference="enrollment")
_INDEX = KnowableAt(reference="index_date")
_POST_INDEX = KnowableAt(reference="post_index")


def _pre(name: str, source: str, *inputs: str, at: KnowableAt = _INDEX) -> FeatureContract:
    return FeatureContract(
        name=name,
        knowable_at=at,
        source=source,
        derivation_inputs=tuple(inputs),
    )


_PATIENT_FEATURES = [
    # Designed confounders (enter both the propensity and the outcome —
    # patient_generator.py:262-287, treatment_arm.py:27-47)
    _pre("disease_severity", "synthetic_dgp", "rng_normal_5_2"),
    _pre("academic_hcp", "synthetic_dgp", "rng_bernoulli_0_30", at=_ENROLLMENT),
    # Demographics / access attributes (static draws)
    _pre("age_at_diagnosis", "synthetic_dgp", at=_ENROLLMENT),
    _pre("insurance_type", "synthetic_dgp", at=_ENROLLMENT),
    # T9 prognostic persistence drivers — pre-index static draws (independent of
    # treatment_arm), wired into the discontinuation/persistence outcome equation
    # (cohort_outcomes.py). Declared pre-index so manifest-filtered pipelines keep them.
    _pre("comorbidity_burden", "synthetic_dgp", at=_ENROLLMENT),
    _pre("prior_therapy_lines", "synthetic_dgp", at=_ENROLLMENT),
    _pre("geographic_region", "synthetic_dgp", at=_ENROLLMENT),
    _pre("primary_diagnosis_code", "synthetic_dgp", at=_ENROLLMENT),
    # Emitted covariate, explicitly non-causal in the live DGP
    _pre("engagement_score", "synthetic_dgp"),
    # Deterministic f(disease_severity) — severity tier at index
    _pre("segment_assignment", "derived", "disease_severity"),
    # Promotional exposure: assigned from confounders BEFORE the outcome draw;
    # the designed treatment whose effect the pipeline recovers
    _pre("treatment_arm", "synthetic_dgp", "disease_severity", "academic_hcp"),
    # CSU indication panel (brand-eligibility columns, generated pre-outcome)
    _pre("urticaria_severity_uas7", "synthetic_dgp", "disease_severity"),
    _pre("prior_antihistamine_therapy", "synthetic_dgp", at=_ENROLLMENT),
]

_HCP_FEATURES = [
    # Exogenous lognormal topology generated before arm and adoption
    # (hcp_adoption_artifact.py:103-124)
    _pre("influence_network_size", "synthetic_topology", "rng_lognormal_3_1_1"),
    _pre("peer_influence_score", "synthetic_topology", at=_ENROLLMENT),
    _pre("total_patient_volume", "synthetic_profile", at=_ENROLLMENT),
    _pre("years_experience", "synthetic_profile", at=_ENROLLMENT),
    _pre("specialty", "synthetic_profile", at=_ENROLLMENT),
    _pre("practice_type", "synthetic_profile", at=_ENROLLMENT),
]

_ANSWER_KEY = [
    # The DGP answer key — stamped ground truth, never a feature
    FeatureContract(
        name="propensity_score",
        knowable_at=_POST_INDEX,
        source="synthetic_dgp_ground_truth",
        derivation_inputs=("disease_severity", "academic_hcp"),
    ),
    FeatureContract(
        name="treatment_effect_estimate",
        knowable_at=_POST_INDEX,
        source="synthetic_dgp_ground_truth",
        derivation_inputs=("segment_assignment", "brand"),
    ),
    # Populated only for initiators -> outcome-derived missingness
    FeatureContract(
        name="days_to_treatment",
        knowable_at=_POST_INDEX,
        source="derived",
        derivation_inputs=("treatment_initiated",),
    ),
]

SYNTHETIC_CSU_FEATURES: dict[str, FeatureContract] = {
    c.name: c for c in [*_PATIENT_FEATURES, *_HCP_FEATURES, *_ANSWER_KEY]
}

# Defense-in-depth for _select_features: the exporter already strips these from
# the tier0 frames; listing them here keeps them out of the feature surface
# should a future frame carry them.
SYNTHETIC_CSU_FORBIDDEN_AS_FEATURES: list[str] = [
    "propensity_score",
    "treatment_effect_estimate",
    "days_to_treatment",
]


def synthetic_csu_contract_for(name: str) -> FeatureContract | None:
    """Return the FeatureContract for ``name``, or None (falls through to
    statistical governance, matching the other manifest sources)."""
    return SYNTHETIC_CSU_FEATURES.get(name)


__all__ = [
    "SYNTHETIC_CSU_FEATURES",
    "SYNTHETIC_CSU_FORBIDDEN_AS_FEATURES",
    "synthetic_csu_contract_for",
]
