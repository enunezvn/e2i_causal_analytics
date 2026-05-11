"""Feature engineering node — v5 Gate B3.

Adds deterministic engineered features (interactions, ratios, composites) to
the pre-anchor feature surface. Every engineered feature is computed from
columns that the cohort's manifest already declares as ``knowable_at <=
index_date``, so the resulting feature inherits pre-anchor status by
construction. The cohort manifest (csu_feature_manifest.py /
optum_feature_manifest.py) declares each engineered feature explicitly so
Layer 1 audit traces the derivation chain and Layer 3 (adversarial probe)
runs on the materialized column during the downstream
``adaptive_validity_check`` node.

Pre-spec: ``docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md``.

Acceptance gate (per v5 plan §B3):
- >=3 engineered features per cohort that pass Layer 1 manifest declaration
  + Layer 3 adversarial probe (z < 5sigma on real cohort data).
- Net val_AUC effect >= +0.02 OR documented null result.

Disease-agnostic by construction: this module dispatches on
``scope_spec.feature_manifest_source`` ("csu" / "optum" / unknown -> no-op).
The transforms themselves are universal (interactions, ratios, log-scales);
the dispatch table maps each cohort's available input columns to the family
of engineered features that uses them.

Gating: this node is OPT-IN via ``state["enable_feature_engineering"] = True``.
The default is False so existing production runs are unaffected. The v5 Gate
B3 measurement methodology compares val_AUC with the flag True vs. False on
the same held-out split.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..state import DataPreparerState

# Type alias for the per-cohort engineering dispatch table.
_EngineerFn = Callable[[pd.DataFrame], Tuple[pd.DataFrame, List[str]]]

logger = logging.getLogger(__name__)


# Minimum denominator for ratio transforms — avoids divide-by-zero and
# silently-infinite intermediate values. Operationally: an
# `eligibility_duration_days` of 0 means the patient was never enrolled,
# which is a data-quality issue elsewhere; the ratio just returns 0/1 = 0.
_RATIO_DENOM_MIN = 1.0


# =============================================================================
# CSU engineered features
# =============================================================================


# claim_intensity_ratio was dropped post-audit (b3_engineered_audit_20260511):
# z=40.83 on CSU n=9607 / n_pos=1743 with max_input_z=9.78 = 4.2x
# amplification beyond inputs. The amplification heuristic flags this as
# leakage (ratios can manufacture signal not present in either component).
# See docs/calibration/b3_engineered_audit_20260511.json + amended pre-spec.
CSU_ENGINEERED_FEATURES: Tuple[str, ...] = (
    "age_x_insurance_interaction",
    "engagement_per_visit",
    "treatment_diversity_intensity",
    "severity_engagement_product",
)


def _engineer_csu_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute CSU engineered features.

    Returns (df_with_engineered, materialized_feature_names). Features
    whose input columns are missing from the DataFrame are silently
    skipped — this is a best-effort enrichment, not a contract. Missing
    inputs are logged at INFO level so operators can diagnose drift.

    All transforms are deterministic and use only pre-anchor inputs per
    the CSU manifest (csu_feature_manifest.py). NaN inputs propagate
    via standard pandas semantics; ratio denominators are clipped to
    >= _RATIO_DENOM_MIN to avoid divide-by-zero.

    Note: The returned DataFrame is the same object as the input, with
    engineered columns assigned via ``df[col] = ...``. Callers that
    require an immutable copy should ``df.copy()`` before invocation.
    """
    materialized: List[str] = []

    # C1: age × insurance interaction (categorical → numeric via codes)
    if "age_continuous" in df.columns and "insurance_type" in df.columns:
        insurance_codes = _categorical_to_codes(df["insurance_type"])
        df["age_x_insurance_interaction"] = (
            pd.to_numeric(df["age_continuous"], errors="coerce") * insurance_codes
        )
        materialized.append("age_x_insurance_interaction")
    else:
        logger.info(
            "csu FE: skipping age_x_insurance_interaction "
            "(missing age_continuous or insurance_type)"
        )

    # C2 (claim_intensity_ratio) DROPPED post-audit — see module docstring.

    # C3: engagement per visit
    if "engagement_score" in df.columns and "hcp_visits" in df.columns:
        eng = pd.to_numeric(df["engagement_score"], errors="coerce")
        visits = pd.to_numeric(df["hcp_visits"], errors="coerce")
        denom = visits.clip(lower=_RATIO_DENOM_MIN)
        df["engagement_per_visit"] = eng / denom
        materialized.append("engagement_per_visit")
    else:
        logger.info("csu FE: skipping engagement_per_visit (missing inputs)")

    # C4: treatment diversity intensity = prior_treatments × log1p(days_on_therapy)
    if "prior_treatments" in df.columns and "days_on_therapy" in df.columns:
        pt = pd.to_numeric(df["prior_treatments"], errors="coerce").fillna(0.0)
        dot = pd.to_numeric(df["days_on_therapy"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["treatment_diversity_intensity"] = pt * np.log1p(dot)
        materialized.append("treatment_diversity_intensity")
    else:
        logger.info("csu FE: skipping treatment_diversity_intensity (missing inputs)")

    # C5: severity × engagement product
    if "disease_severity" in df.columns and "engagement_score" in df.columns:
        sev = pd.to_numeric(df["disease_severity"], errors="coerce")
        eng = pd.to_numeric(df["engagement_score"], errors="coerce")
        df["severity_engagement_product"] = sev * eng
        materialized.append("severity_engagement_product")
    else:
        logger.info("csu FE: skipping severity_engagement_product (missing inputs)")

    return df, materialized


# =============================================================================
# Optum engineered features
# =============================================================================


OPTUM_ENGINEERED_FEATURES: Tuple[str, ...] = (
    "comorbidity_load_total",
    "csu_dx_intensity",
    "polypharmacy_breadth",
    "lab_workup_completeness",
    "specialist_visit_interaction",
)


_OPTUM_COMORBIDITY_FLAGS: Tuple[str, ...] = (
    "has_atopic_dermatitis",
    "has_asthma",
    "has_allergic_rhinitis",
    "has_anxiety",
    "has_depression",
    "has_thyroid_autoimmune",
    "has_nsaid_hypersensitivity",
    "has_angioedema",
)

_OPTUM_DRUG_EVER_FLAGS: Tuple[str, ...] = (
    "h1_1g_ever_filled",
    "h1_2g_ever_filled",
    "h2_ever_filled",
    "ltra_ever_filled",
    "sys_steroid_ever_filled",
    "top_steroid_ever_filled",
    "immunosupp_ever_filled",
)

_OPTUM_LAB_TESTED_FLAGS: Tuple[str, ...] = (
    "ige_total_tested",
    "eosinophil_tested",
    "crp_tested",
    "tpo_ab_tested",
    "free_t4_tested",
    "tsh_tested",
    "ana_tested",
    "cbc_tested",
)


def _engineer_optum_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute Optum engineered features. See _engineer_csu_features for contract."""
    materialized: List[str] = []

    # O1: total comorbidity load (sum of 8 has_X flags)
    available_comorb = [c for c in _OPTUM_COMORBIDITY_FLAGS if c in df.columns]
    if len(available_comorb) >= 2:
        df["comorbidity_load_total"] = (
            df[available_comorb].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        )
        materialized.append("comorbidity_load_total")
    else:
        logger.info(
            "optum FE: skipping comorbidity_load_total "
            f"(only {len(available_comorb)} of 8 has_X flags present)"
        )

    # O2: CSU dx intensity = dx_total_csu / max(months_since_first_dx, 1)
    if "dx_total_csu" in df.columns and "months_since_first_dx" in df.columns:
        dx_total = pd.to_numeric(df["dx_total_csu"], errors="coerce").fillna(0.0)
        months = pd.to_numeric(df["months_since_first_dx"], errors="coerce")
        denom = months.clip(lower=_RATIO_DENOM_MIN)
        df["csu_dx_intensity"] = dx_total / denom
        materialized.append("csu_dx_intensity")
    else:
        logger.info("optum FE: skipping csu_dx_intensity (missing inputs)")

    # O3: polypharmacy breadth (sum of 7 drug-class ever-filled flags)
    available_drugs = [c for c in _OPTUM_DRUG_EVER_FLAGS if c in df.columns]
    if len(available_drugs) >= 2:
        df["polypharmacy_breadth"] = (
            df[available_drugs].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        )
        materialized.append("polypharmacy_breadth")
    else:
        logger.info(
            "optum FE: skipping polypharmacy_breadth "
            f"(only {len(available_drugs)} of 7 drug-class flags present)"
        )

    # O4: lab workup completeness (sum of 8 lab-panel tested flags)
    available_labs = [c for c in _OPTUM_LAB_TESTED_FLAGS if c in df.columns]
    if len(available_labs) >= 2:
        df["lab_workup_completeness"] = (
            df[available_labs].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        )
        materialized.append("lab_workup_completeness")
    else:
        logger.info(
            "optum FE: skipping lab_workup_completeness "
            f"(only {len(available_labs)} of 8 lab-tested flags present)"
        )

    # O5: specialist visit interaction
    if "office_visits_allergist" in df.columns and "office_visits_dermatology" in df.columns:
        allerg = pd.to_numeric(df["office_visits_allergist"], errors="coerce").fillna(0.0)
        derm = pd.to_numeric(df["office_visits_dermatology"], errors="coerce").fillna(0.0)
        df["specialist_visit_interaction"] = allerg * derm
        materialized.append("specialist_visit_interaction")
    else:
        logger.info("optum FE: skipping specialist_visit_interaction (missing inputs)")

    return df, materialized


# =============================================================================
# Dispatch
# =============================================================================


_DISPATCH: Mapping[str, _EngineerFn] = {
    "csu": _engineer_csu_features,
    "optum": _engineer_optum_features,
}


def _categorical_to_codes(series: pd.Series) -> pd.Series:
    """Encode a categorical/object series to numeric codes deterministically.

    Uses pandas ``factorize`` with a stable sort so the same string values
    map to the same codes across train/val/test splits (codes are derived
    from the per-split unique values; this is acceptable for the C1
    interaction because the interaction is the SAME ordinal partition
    within each split — the model trains on within-split signal, not
    cross-split code identity).

    NaN values are encoded as -1 by factorize; we preserve them as NaN
    in the output so downstream imputation handles them uniformly with
    other missing inputs.

    Args:
        series: pandas Series of object/categorical/string dtype.

    Returns:
        Float Series with category codes (NaN where input was NaN).
    """
    codes, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
    coded = pd.Series(codes, index=series.index, dtype=float)
    coded[coded < 0] = np.nan
    return coded


def engineer_features(
    df: pd.DataFrame,
    manifest_source: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute engineered features for the given DataFrame and cohort.

    Pure function — caller controls copying. Returns (df, materialized).
    When manifest_source is None or unknown, returns (df, []) with no
    modification — engineering is gated on a known cohort manifest.

    Args:
        df: input DataFrame (mutated in place; caller copies if needed).
        manifest_source: "csu" / "optum" / other. Determines which family
            of engineered features to compute.

    Returns:
        (df, materialized_feature_names). ``materialized`` lists only the
        features that were actually added (inputs present); missing inputs
        cause a feature to be skipped silently with an INFO log.
    """
    if not manifest_source:
        return df, []
    impl = _DISPATCH.get(manifest_source)
    if impl is None:
        logger.info("feature_engineering: unknown manifest_source %r — no-op", manifest_source)
        return df, []
    return impl(df)


async def engineer_features_node(state: DataPreparerState) -> Dict[str, Any]:
    """LangGraph node: engineer features on train/val/test splits.

    Gated on ``state["enable_feature_engineering"]`` (default False).
    When False, returns an empty dict (no state mutation). When True,
    reads ``scope_spec.feature_manifest_source`` to dispatch the
    correct family of transforms and applies them in-place to
    train_df / validation_df / test_df / holdout_df.

    The materialized feature names are surfaced under
    ``state["engineered_features"]`` so downstream nodes (adaptive_validity_check,
    leakage_remediation, transform_data) can audit them.

    Args:
        state: DataPreparerState dict.

    Returns:
        Dict patch for LangGraph (engineered_features, engineered_dispatch_source).
    """
    enabled = bool(state.get("enable_feature_engineering", False))
    if not enabled:
        return {}

    scope = state.get("scope_spec") or {}
    if isinstance(scope, dict):
        manifest_source = scope.get("feature_manifest_source")
    else:
        manifest_source = getattr(scope, "feature_manifest_source", None)

    materialized_per_split: Dict[str, List[str]] = {}
    for split_key in ("train_df", "validation_df", "test_df", "holdout_df"):
        df = state.get(split_key)
        if df is None:
            continue
        # Guard against non-DataFrame state values (legacy callers can pass
        # numpy arrays or dicts in some test paths). Type check is intentional.
        if not isinstance(df, pd.DataFrame):
            logger.warning(
                "engineer_features_node: %s is not a DataFrame (got %s); skipping",
                split_key,
                type(df).__name__,
            )
            continue
        # Mutate in place — this matches the pattern of transform_data which
        # also reuses the DataFrame object on state.
        _, materialized = engineer_features(df, manifest_source)
        materialized_per_split[split_key] = materialized

    # All splits should materialize the same feature names since they share
    # the same column schema; surface the train-split list canonically.
    canonical = materialized_per_split.get("train_df", [])

    logger.info(
        "engineer_features_node: manifest_source=%r added %d features: %s",
        manifest_source,
        len(canonical),
        canonical,
    )

    return {
        "engineered_features": canonical,
        "engineered_dispatch_source": manifest_source,
    }
