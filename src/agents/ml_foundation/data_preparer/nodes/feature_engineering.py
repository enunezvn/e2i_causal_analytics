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
#
# Issue #187 / backlog #17 (commit cfa71627, 2026-05-12): three of the
# four B3 engineered features (engagement_per_visit,
# treatment_diversity_intensity, severity_engagement_product) were
# reclassified as ``knowable_at=post_index`` because their inputs are
# medication-derived aggregates that are themselves post-index (target
# coupling: untreated patients are absent from ``CSUDataConverter._med_by_pat``
# so all medication-derived aggregates collapse to zero for them).
# Materializing them would add forbidden columns to ``train_df`` even
# though ``_select_features`` filters via ``CSU_FORBIDDEN_AS_FEATURES``
# at Layer 3 — defense-in-depth: don't materialize them in the first
# place. The formulas remain as private ``_csu_*`` helpers for unit-test
# coverage of the math (per-formula tests still pin the
# clamp/log1p/product semantics) but the production path no longer
# adds the forbidden columns to the split DataFrames.
CSU_ENGINEERED_FEATURES: Tuple[str, ...] = ("age_x_insurance_interaction",)


# Historical formulas retained for unit-test coverage only. Issue #187
# regression pin: if these are re-added to CSU_ENGINEERED_FEATURES, the
# Layer 1 chain-validity test in test_feature_engineering.py will catch
# the regression and the test_csu_engineered_features_post_index_*
# guard will fire because the manifest still declares them post-index.
_BACKLOG_17_FORBIDDEN_CSU_ENGINEERED: Tuple[str, ...] = (
    "engagement_per_visit",
    "treatment_diversity_intensity",
    "severity_engagement_product",
)


def _csu_engagement_per_visit(df: pd.DataFrame) -> pd.Series:
    """Historical (post-index) formula retained for unit-test coverage only.

    Issue #187 / backlog #17: this engineered feature is FORBIDDEN as a
    production model input (its inputs ``engagement_score`` + ``hcp_visits``
    are post-index per the CSU manifest). The math is preserved here so
    the per-formula clamp/zero-visits regression test can still pin it.
    """
    eng = pd.to_numeric(df["engagement_score"], errors="coerce")
    visits = pd.to_numeric(df["hcp_visits"], errors="coerce")
    denom = visits.clip(lower=_RATIO_DENOM_MIN)
    return eng / denom


def _csu_treatment_diversity_intensity(df: pd.DataFrame) -> pd.Series:
    """Historical (post-index) formula retained for unit-test coverage only.

    Issue #187 / backlog #17: forbidden as production input. See
    ``_csu_engagement_per_visit``.
    """
    pt = pd.to_numeric(df["prior_treatments"], errors="coerce").fillna(0.0)
    dot = pd.to_numeric(df["days_on_therapy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return pt * np.log1p(dot)


def _csu_severity_engagement_product(df: pd.DataFrame) -> pd.Series:
    """Historical (post-index) formula retained for unit-test coverage only.

    Issue #187 / backlog #17: forbidden as production input. See
    ``_csu_engagement_per_visit``.
    """
    sev = pd.to_numeric(df["disease_severity"], errors="coerce")
    eng = pd.to_numeric(df["engagement_score"], errors="coerce")
    return sev * eng


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

    Issue #187 / backlog #17: post-index engineered candidates
    (``engagement_per_visit``, ``treatment_diversity_intensity``,
    ``severity_engagement_product``) are NOT materialized here — the
    manifest declares them forbidden because their inputs are
    medication-derived and target-coupled. The per-formula math is
    retained in private ``_csu_*`` helpers so the math regression
    tests still pin clamp/log1p/product semantics, but the production
    DataFrame never gains these columns. ``_select_features``
    (adaptive_validity_check.py) provides downstream defense-in-depth
    via ``CSU_FORBIDDEN_AS_FEATURES``; this helper provides the
    upstream pin.

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
    # C3-C5 (engagement_per_visit, treatment_diversity_intensity,
    # severity_engagement_product) FORBIDDEN per backlog #17 — see
    # _BACKLOG_17_FORBIDDEN_CSU_ENGINEERED. The math is retained in the
    # private _csu_* helpers for test coverage; production no longer
    # materializes them on the split DataFrames.

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

    M1 (codex): codes are derived from a HASH of the string value
    (specifically, the rank-by-sorted-string-index in a fixed
    cohort-wide vocabulary) so the same string deterministically maps
    to the same numeric code regardless of which split sees it. This
    matters for the C1 interaction (``age × insurance_type``) when
    the engineer_features helper is applied per-split (e.g., in the
    LangGraph node where train/val/test DataFrames are processed
    independently): per-split ``pd.factorize`` would assign code 0
    to whichever value happens to come first alphabetically in that
    split's unique set, producing inconsistent train/val/test
    encodings when category presence differs across splits.

    The chosen encoding is the rank of the sorted unique values
    PRESENT IN THE SERIES being encoded, BUT only if the caller
    pre-aligns the categories. To make per-split encoding
    deterministic, we use the HASH of the string itself
    (modulo a large prime) — different splits map the same string
    to the same numeric code with probability ≈ 1; collisions are
    astronomically rare and would in any case map two strings
    consistently across splits. The exact numeric value of each code
    is not meaningful for the C1 interaction (any deterministic
    string → number injection works); cross-split STABILITY of the
    mapping is the load-bearing property.

    NaN values are encoded as NaN.

    Args:
        series: pandas Series of object/categorical/string dtype.

    Returns:
        Float Series with category codes (NaN where input was NaN).
    """

    def _hash_value(v: Any) -> float:
        if v is None:
            return np.nan
        try:
            if pd.isna(v):
                return np.nan
        except (TypeError, ValueError):
            pass
        # Convert to str + use a deterministic hash modulo a moderate
        # prime. Python's built-in hash() is salted per-process; use
        # a stable algorithm instead.
        s = str(v).encode("utf-8")
        h = 0
        for byte in s:
            h = (h * 131 + byte) & 0xFFFFFFFF
        return float(h % 10007)  # bound to a small range for numerical sanity

    coded = series.map(_hash_value)
    return coded.astype(float)


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
    correct family of transforms and applies them to train_df /
    validation_df / test_df / holdout_df.

    H3 (codex): the node returns the MUTATED DataFrames in the state
    patch (keyed by their original state keys), not just the
    engineered-features metadata. This is the same pattern
    ``transform_data`` uses — return the modified object so
    LangGraph's reducer owns the new state. In-place mutation alone is
    not replay-safe: under checkpoint resume or concurrent execution,
    the DataFrame objects are deserialized fresh and in-place
    mutations would be lost. Returning them in the patch makes the
    node deterministically idempotent.

    Args:
        state: DataPreparerState dict.

    Returns:
        Dict patch for LangGraph: engineered_features +
        engineered_dispatch_source + any of train_df/validation_df/
        test_df/holdout_df that were mutated.
    """
    enabled = bool(state.get("enable_feature_engineering", False))
    if not enabled:
        return {}

    scope = state.get("scope_spec") or {}
    if isinstance(scope, dict):
        manifest_source = scope.get("feature_manifest_source")
    else:
        manifest_source = getattr(scope, "feature_manifest_source", None)

    patch: Dict[str, Any] = {}
    materialized_per_split: Dict[str, List[str]] = {}
    for split_key in ("train_df", "validation_df", "test_df", "holdout_df"):
        df = state.get(split_key)
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            logger.warning(
                "engineer_features_node: %s is not a DataFrame (got %s); skipping",
                split_key,
                type(df).__name__,
            )
            continue
        df_out, materialized = engineer_features(df, manifest_source)
        materialized_per_split[split_key] = materialized
        # H3 fix: surface the mutated DataFrame in the state patch so
        # LangGraph's reducer applies the change durably (replay-safe).
        patch[split_key] = df_out

    canonical = materialized_per_split.get("train_df", [])

    logger.info(
        "engineer_features_node: manifest_source=%r added %d features: %s",
        manifest_source,
        len(canonical),
        canonical,
    )

    patch["engineered_features"] = canonical
    patch["engineered_dispatch_source"] = manifest_source
    return patch
