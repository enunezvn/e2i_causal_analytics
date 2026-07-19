"""Canonical model-name → Feast feature-refs registry.

This module is the single source of truth for the mapping between a
deployed model name and the list of Feast feature references it expects
at inference / explanation time. It deduplicates the previously
hand-rolled tables in:

  - ``src/api/routes/predictions.py`` (single-prediction route)
  - ``src/api/routes/explain.py`` (SHAP explanation route)
  - ``scripts/bentoml/e2i_serving_service.py`` (BentoML container —
    keeps a LOCAL copy because BentoML cannot import ``src.*``; the
    local copy is kept in sync via the
    ``tests/unit/test_serving/test_model_feature_refs_match.py``
    parity test.)

When you add a new model:
  1. Add the model name → feature_refs entry to ``MODEL_FEATURE_REFS``
     below.
  2. Update the BentoML local copy if applicable.
  3. The parity test will fail-loud if the two ever diverge.

Architectural constraint: this file MUST NOT import anything from
``src.*`` other than stdlib / typing — the BentoML build context bundles
this module as part of the container image generation flow, and pulling
in heavy dependencies would balloon the image.
"""

from __future__ import annotations

from typing import Dict, List

# Canonical registry. Lower-case model_name keys, in priority order:
#   propensity, risk_stratification, churn_prediction, next_best_action.
MODEL_FEATURE_REFS: Dict[str, List[str]] = {
    "propensity": [
        "patient_engagement_features:days_since_last_hcp_visit",
        "patient_engagement_features:total_hcp_interactions_90d",
        "patient_engagement_features:therapy_adherence_score",
    ],
    "risk_stratification": [
        "patient_risk_features:comorbidity_count",
        "patient_risk_features:lab_value_trend",
        "patient_risk_features:prior_brand_experience",
    ],
    "churn_prediction": [
        "patient_churn_features:days_since_last_visit",
        "patient_churn_features:engagement_trend",
        "patient_churn_features:satisfaction_score",
    ],
    "next_best_action": [
        "patient_nba_features:channel_preference",
        "patient_nba_features:response_history",
        "patient_nba_features:timing_preference",
    ],
    # Gold-standard cohort families (#39 + T9/T11 enrichment): the real
    # ``*_goldstd_lr_v1`` models consume the 7 RAW leakage-safe ``_BASE7``
    # covariates served by the ``goldstd_cohort_features`` Feast view
    # (feature_repo/features/goldstd_cohort_features.py). The served model's
    # bundled FeatureBuilder one-hot/median-encodes these into the 19 numeric
    # features SHAP runs over. Fetching only the base 3 here would hand the
    # 7-covariate bundle an incomplete vector (the #576 null-trap → 503), and
    # the live Feature-Importance page would show 3 covariates instead of 7.
    "initiation": [
        "goldstd_cohort_features:disease_severity",
        "goldstd_cohort_features:academic_hcp",
        "goldstd_cohort_features:geographic_region",
        "goldstd_cohort_features:insurance_type",
        "goldstd_cohort_features:age_at_diagnosis",
        "goldstd_cohort_features:comorbidity_burden",
        "goldstd_cohort_features:prior_therapy_lines",
    ],
    # COMM-ARMS Phase 1/2: persistence/discontinuation additionally consume the
    # commercial arms copay_support (Phase 1) + psp_enrolled (Phase 2) — 9 refs, not 7.
    # Both enter the DISCONTINUATION logit, so they are real outcome signal the model
    # can legitimately observe (assigned pre-index, not leakage columns). These two
    # lists must stay in lockstep with _PATIENT_COVARIATES in cohort_spec.py — spec
    # says 9, refs must fetch 9, or the bundle gets an incomplete vector (#576
    # null-trap → 503).
    "persistence": [
        "goldstd_cohort_features:disease_severity",
        "goldstd_cohort_features:academic_hcp",
        "goldstd_cohort_features:geographic_region",
        "goldstd_cohort_features:insurance_type",
        "goldstd_cohort_features:age_at_diagnosis",
        "goldstd_cohort_features:comorbidity_burden",
        "goldstd_cohort_features:prior_therapy_lines",
        "goldstd_cohort_features:copay_support",
        "goldstd_cohort_features:psp_enrolled",
    ],
    "discontinuation": [
        "goldstd_cohort_features:disease_severity",
        "goldstd_cohort_features:academic_hcp",
        "goldstd_cohort_features:geographic_region",
        "goldstd_cohort_features:insurance_type",
        "goldstd_cohort_features:age_at_diagnosis",
        "goldstd_cohort_features:comorbidity_burden",
        "goldstd_cohort_features:prior_therapy_lines",
        "goldstd_cohort_features:copay_support",
        "goldstd_cohort_features:psp_enrolled",
    ],
    # HCP-grain gold-standard adoption cohort (#39 multi-model): the per-brand
    # ``hcp_adoption_{brand}_goldstd_lr_v1`` models consume the 5 RAW leakage-safe
    # HCP covariates served by the ``goldstd_hcp_cohort_features`` view
    # (entity: hcp / hcp_id). The served model's FeatureBuilder one-hot/median-
    # encodes them into the 19 numeric features SHAP runs over. ``specialty`` +
    # ``geographic_region`` are categorical (one-hot); the other 3 are numeric.
    "hcp_adoption": [
        "goldstd_hcp_cohort_features:peer_influence_score",
        "goldstd_hcp_cohort_features:influence_network_size",
        "goldstd_hcp_cohort_features:years_experience",
        "goldstd_hcp_cohort_features:specialty",
        "goldstd_hcp_cohort_features:geographic_region",
    ],
}


def feature_refs_for_model(model_name: str) -> List[str]:
    """Resolve the Feast feature-refs to fetch for ``model_name``.

    Falls back to the ``propensity`` refs for unknown model names so the
    Feast code path always exercises real feature names rather than
    silently passing an empty list (which Feast rejects at the wire
    level with an ungraceful 4xx).
    """
    return MODEL_FEATURE_REFS.get(model_name) or MODEL_FEATURE_REFS["propensity"]
