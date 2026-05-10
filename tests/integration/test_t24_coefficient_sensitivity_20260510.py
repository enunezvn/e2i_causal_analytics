"""Plan v4 Gate G5 — T2.4 coefficient-sensitivity integration tests.

Closes the v3 §6 T2.4 acceptance criterion ("coefficient sensitivity tests
pass on Optum and CSU") that was unaddressed prior to PR #125 + this commit.

The test workflow per cohort:

1. Load the cohort's ``patient_journeys`` artifact via the FileIngestor
   used in production (``_load_from_files`` semantics) — same code path
   the tier0 runner uses.
2. Build a numeric feature matrix X (numeric dtype only — categoricals
   are excluded from the coefficient-sensitivity audit because their
   imputed coefficients are not directly comparable across strategies).
3. Call ``compute_imputation_audit`` on X to derive per-feature
   recommended imputation strategies.
4. Call ``compute_coefficient_sensitivity(X, y, recommendations)`` to
   fit baseline + imputed models and produce the per-feature comparison.
5. Assert ``passes_pre_spec`` is True on the helper's output, which wraps
   the three pre-specified thresholds T1 / T2 / T3 from
   ``docs/specs/g5_coefficient_sensitivity_prespec_20260510.md``.

The thresholds are imported from the helper's constants — NOT
re-declared in the test file — so the spec memo's threshold values are
the only place they live (no drift possible).

Skip-if-data-missing: if a cohort artifact is absent (e.g., CSU not on
disk in the current checkout), the test SKIPS that cohort rather than
fails. CI runs both cohorts; local dev may run only one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.ingestion import FileIngestor
from src.agents.ml_foundation.data_preparer.nodes.coefficient_sensitivity import (
    G5_EFFECT_SIZE_CV_MAX,
    G5_FLIPS_PER_FEATURE_MAX,
    G5_FRACTION_SIGNIFICANT_FLIPPED_MAX,
    G5_SIGNIFICANCE_SIGMA_MULTIPLE,
    compute_coefficient_sensitivity,
)
from src.agents.ml_foundation.data_preparer.nodes.imputation_audit import (
    compute_imputation_audit,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Cohort artifact paths
# ---------------------------------------------------------------------------

CSU_DATA_DIR = REPO_ROOT / "data" / "rwd" / "csu"
CSU_JOURNEYS_PATH = CSU_DATA_DIR / "e2i_ml_v3_patient_journeys.json"

OPTUM_INITIATION_DIR = REPO_ROOT / "data" / "rwd" / "optum" / "initiation"
OPTUM_INITIATION_JOURNEYS_PATH = OPTUM_INITIATION_DIR / "e2i_ml_v3_patient_journeys.parquet"


# ---------------------------------------------------------------------------
# Cohort-specific target columns (drop from X, become y).
# ---------------------------------------------------------------------------

CSU_TARGET = "treatment_initiated"
OPTUM_TARGET = "treatment_initiated"


# ---------------------------------------------------------------------------
# Columns to exclude from X regardless of dtype:
#  - identifiers (patient_id, journey_id) — high-cardinality numeric
#    identifiers would dominate any logistic-regression coefficient
#    landscape with spurious magnitudes.
#  - splits / partition columns
#  - target columns (handled per-cohort).
# ---------------------------------------------------------------------------

EXCLUDED_COLUMNS_BASE = frozenset(
    {
        "patient_id",
        "patient_journey_id",
        "data_split",
        "treatment_initiated",
        "discontinuation_flag",
        "treatment_persistence",
        "journey_status",
        "journey_stage",
    }
)


def _load_patient_journeys(directory: Path) -> pd.DataFrame:
    """Use the production FileIngestor to load patient_journeys from a
    cohort directory. Returns the raw DataFrame; downstream code is
    responsible for selecting numeric features and the target."""
    ingestor = FileIngestor()
    frames = ingestor.ingest_directory(directory)
    if "patient_journeys" not in frames:
        raise RuntimeError(f"Cohort directory {directory} did not yield a patient_journeys frame")
    return frames["patient_journeys"]


def _build_features_and_target(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract numeric feature matrix + binary target series.

    Casts the target column to int (0/1) and drops it from X. Excludes
    other identifier / split columns that would corrupt the coefficient
    landscape.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Target column {target_col!r} not present in cohort journeys: "
            f"{sorted(df.columns.tolist())[:10]}..."
        )

    excluded = EXCLUDED_COLUMNS_BASE
    X = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
    # Restrict to numeric columns; drop bool because bool dtype with NaN
    # cells doesn't survive ``fillna(0.0)``.
    numeric_cols = [
        c
        for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    X = X[numeric_cols].copy()
    if X.shape[1] == 0:
        raise ValueError(
            f"Cohort journeys had no numeric feature columns after excluding "
            f"{sorted(excluded)}; got dtypes: "
            f"{df.dtypes.value_counts().to_dict()}"
        )

    # Binary target. Allow NaN-as-False semantics (untreated patients
    # may have NaN in the target column when joined post-filter).
    y_raw = df[target_col].fillna(0)
    # Coerce numeric / bool to int 0/1.
    if pd.api.types.is_bool_dtype(y_raw):
        y = y_raw.astype(np.int64)
    else:
        y = (y_raw.astype(np.float64) > 0.5).astype(np.int64)
    y = pd.Series(y.to_numpy(), name=target_col)

    return X, y


def _train_split(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the train rows if a ``data_split`` column is present.
    Otherwise return the full DataFrame (the helper accepts any binary
    classification matrix; train-only is the convention)."""
    if "data_split" in df.columns:
        train_mask = df["data_split"] == "train"
        if train_mask.sum() > 0:
            return df[train_mask].reset_index(drop=True)
    return df


def _run_g5_on_cohort(directory: Path, target_col: str, seed: int = 42) -> dict:
    """Full G5 workflow on a cohort directory: load → audit → sensitivity."""
    df = _load_patient_journeys(directory)
    df_train = _train_split(df)
    X, y = _build_features_and_target(df_train, target_col)

    # Step 1: audit missingness on X to derive per-feature recommendations.
    audit = compute_imputation_audit(X)
    recommendations = audit["imputation_audit_recommendations"]

    # Step 2: run the coefficient-sensitivity audit.
    return compute_coefficient_sensitivity(X, y, recommendations, seed=seed)


# ---------------------------------------------------------------------------
# Module-scoped fixtures: amortize cohort-loading + audit cost across all
# assertions per cohort.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def csu_sensitivity() -> dict:
    """Run G5 on real CSU patient_journeys. Skips if data not on disk."""
    if not CSU_JOURNEYS_PATH.exists():
        pytest.skip(
            f"CSU journeys file not present at {CSU_JOURNEYS_PATH}; "
            "skipping CSU coefficient-sensitivity assertions."
        )
    return _run_g5_on_cohort(CSU_DATA_DIR, CSU_TARGET)


@pytest.fixture(scope="module")
def optum_initiation_sensitivity() -> dict:
    """Run G5 on real Optum initiation cohort. Skips if data not on disk."""
    if not OPTUM_INITIATION_JOURNEYS_PATH.exists():
        pytest.skip(
            f"Optum initiation journeys file not present at "
            f"{OPTUM_INITIATION_JOURNEYS_PATH}; skipping Optum "
            "coefficient-sensitivity assertions."
        )
    return _run_g5_on_cohort(OPTUM_INITIATION_DIR, OPTUM_TARGET)


# ---------------------------------------------------------------------------
# Threshold drift detector — pins the pre-spec memo's load-bearing values.
# Mirror of the unit test's TestG5Thresholds class so the integration suite
# also fails LOUDLY if anyone mutates the constants without refreshing the
# spec doc.
# ---------------------------------------------------------------------------


class TestG5PreSpecConstantsDriftDetector:
    """If these constants drift without a refreshed spec memo, the
    threshold-shopping invariant has been violated."""

    def test_t1_flips_per_feature_max(self) -> None:
        assert G5_FLIPS_PER_FEATURE_MAX == 1, (
            "G5_FLIPS_PER_FEATURE_MAX drifted from the pre-spec memo's "
            "T1 value of 1. Per v3 §8 anti-threshold-shopping invariant, "
            "any threshold change requires a NEW pre-spec memo at a fresh "
            "date. Do NOT edit this constant in place."
        )

    def test_t2_effect_size_cv_max(self) -> None:
        assert G5_EFFECT_SIZE_CV_MAX == 0.5, (
            "G5_EFFECT_SIZE_CV_MAX drifted from the pre-spec memo's "
            "T2 value of 0.5. See test_t1_flips_per_feature_max for "
            "the resolution protocol."
        )

    def test_t3_fraction_significant_flipped_max(self) -> None:
        assert G5_FRACTION_SIGNIFICANT_FLIPPED_MAX == 0.10, (
            "G5_FRACTION_SIGNIFICANT_FLIPPED_MAX drifted from the "
            "pre-spec memo's T3 value of 0.10. See test_t1 for the "
            "resolution protocol."
        )

    def test_significance_sigma_multiple(self) -> None:
        assert G5_SIGNIFICANCE_SIGMA_MULTIPLE == 1.0


# ---------------------------------------------------------------------------
# Per-cohort assertions: the load-bearing G5 acceptance criteria.
# Each cohort gets its own test class so failure messages identify the
# cohort precisely.
# ---------------------------------------------------------------------------


class TestCSUCoefficientSensitivity:
    """G5 acceptance on the CSU cohort.

    CSU's known properties (per the codex CSU-benchmark research +
    iter6 cohort-growth memory):
      - n ≈ 9607 patients (sufficient for stable LR coefficients).
      - Treatment-initiation rate ≈ 18% (moderate imbalance).
      - Most numeric features have low missingness (< 30%); a few
        engagement / claim-count fields are 30-70% missing.
    """

    @pytest.mark.integration
    def test_passes_pre_spec(self, csu_sensitivity: dict) -> None:
        """T1 + T2 + T3 all hold on CSU."""
        assert csu_sensitivity["passes_pre_spec"] is True, (
            f"CSU G5 pre-spec FAILED: violations="
            f"{csu_sensitivity['violations']!r}. "
            f"Per v3 §8, the response is NOT to relax the threshold; "
            f"investigate the imputation strategy or feature surface. "
            f"aggregate={csu_sensitivity['aggregate']!r}"
        )

    @pytest.mark.integration
    def test_t1_flips_per_feature(self, csu_sensitivity: dict) -> None:
        """Per-feature flip count ≤ G5_FLIPS_PER_FEATURE_MAX (T1)."""
        max_flips = csu_sensitivity["aggregate"]["max_flips_per_feature_significant"]
        assert max_flips <= G5_FLIPS_PER_FEATURE_MAX, (
            f"CSU T1 violated: max_flips_per_feature_significant={max_flips} "
            f"exceeds pre-spec ceiling {G5_FLIPS_PER_FEATURE_MAX}."
        )

    @pytest.mark.integration
    def test_t2_effect_size_cv(self, csu_sensitivity: dict) -> None:
        """Max effect-size CV across significant features ≤ T2."""
        max_cv = csu_sensitivity["aggregate"]["max_effect_size_variance_significant"]
        assert max_cv <= G5_EFFECT_SIZE_CV_MAX, (
            f"CSU T2 violated: max_effect_size_variance_significant={max_cv:.3f} "
            f"exceeds pre-spec ceiling {G5_EFFECT_SIZE_CV_MAX}."
        )

    @pytest.mark.integration
    def test_t3_fraction_significant_flipped(self, csu_sensitivity: dict) -> None:
        """Aggregate fraction of significant features flipping ≤ T3."""
        fraction = csu_sensitivity["aggregate"]["fraction_significant_flipped"]
        assert fraction <= G5_FRACTION_SIGNIFICANT_FLIPPED_MAX, (
            f"CSU T3 violated: fraction_significant_flipped={fraction:.3f} "
            f"exceeds pre-spec ceiling {G5_FRACTION_SIGNIFICANT_FLIPPED_MAX}."
        )

    @pytest.mark.integration
    def test_n_features_audited_nonzero(self, csu_sensitivity: dict) -> None:
        """Sanity: the helper actually saw features. If n_features=0,
        the cohort schema regressed (numeric columns disappeared) and
        the rest of the assertions are vacuously passing."""
        assert csu_sensitivity["n_features"] > 0, (
            "CSU yielded no numeric features for the sensitivity audit; "
            "the rest of the G5 assertions are vacuously passing. "
            "Investigate the cohort schema."
        )


class TestOptumInitiationCoefficientSensitivity:
    """G5 acceptance on the Optum initiation cohort.

    Optum-initiation's known properties (per the optum_revalidation memory):
      - n=1294 patients (default-window) — small but ML-viable.
      - Significant imbalance + small positive class (~31:1 ratio).
      - Several Rx claim / lookback-window features have moderate missingness.
    """

    @pytest.mark.integration
    def test_passes_pre_spec(self, optum_initiation_sensitivity: dict) -> None:
        """T1 + T2 + T3 all hold on Optum-initiation."""
        assert optum_initiation_sensitivity["passes_pre_spec"] is True, (
            f"Optum-initiation G5 pre-spec FAILED: violations="
            f"{optum_initiation_sensitivity['violations']!r}. "
            f"Per v3 §8, the response is NOT to relax the threshold; "
            f"investigate the imputation strategy or feature surface. "
            f"aggregate={optum_initiation_sensitivity['aggregate']!r}"
        )

    @pytest.mark.integration
    def test_t1_flips_per_feature(self, optum_initiation_sensitivity: dict) -> None:
        """Per-feature flip count ≤ T1 on Optum."""
        max_flips = optum_initiation_sensitivity["aggregate"]["max_flips_per_feature_significant"]
        assert max_flips <= G5_FLIPS_PER_FEATURE_MAX, (
            f"Optum T1 violated: max_flips_per_feature_significant={max_flips} "
            f"exceeds pre-spec ceiling {G5_FLIPS_PER_FEATURE_MAX}."
        )

    @pytest.mark.integration
    def test_t2_effect_size_cv(self, optum_initiation_sensitivity: dict) -> None:
        """Max effect-size CV ≤ T2 on Optum."""
        max_cv = optum_initiation_sensitivity["aggregate"]["max_effect_size_variance_significant"]
        assert max_cv <= G5_EFFECT_SIZE_CV_MAX, (
            f"Optum T2 violated: max_effect_size_variance_significant={max_cv:.3f} "
            f"exceeds pre-spec ceiling {G5_EFFECT_SIZE_CV_MAX}."
        )

    @pytest.mark.integration
    def test_t3_fraction_significant_flipped(self, optum_initiation_sensitivity: dict) -> None:
        """Aggregate fraction of significant features flipping ≤ T3 on Optum."""
        fraction = optum_initiation_sensitivity["aggregate"]["fraction_significant_flipped"]
        assert fraction <= G5_FRACTION_SIGNIFICANT_FLIPPED_MAX, (
            f"Optum T3 violated: fraction_significant_flipped={fraction:.3f} "
            f"exceeds pre-spec ceiling {G5_FRACTION_SIGNIFICANT_FLIPPED_MAX}."
        )

    @pytest.mark.integration
    def test_n_features_audited_nonzero(self, optum_initiation_sensitivity: dict) -> None:
        """Sanity: the helper actually saw features."""
        assert optum_initiation_sensitivity["n_features"] > 0, (
            "Optum yielded no numeric features for the sensitivity audit; "
            "the rest of the G5 assertions are vacuously passing. "
            "Investigate the cohort schema."
        )
