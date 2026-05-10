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

Cohort-data presence (G5 codex M2 closure):
  - In CI (env ``CI=true``): both cohort artifacts MUST be present;
    missing data = test FAILS, not skips. Skipping silently in CI
    removes the load-bearing cohort gate without surfacing the gap.
  - Locally: missing data SKIPS the cohort assertions. The
    ``RUN_LOCAL_ONLY=1`` env var is honored as an explicit opt-in for
    local-only runs that intentionally skip cohort assertions.
"""

from __future__ import annotations

import logging
import os
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


def _cohort_artifact_missing_action(artifact_path: Path, cohort_name: str) -> None:
    """G5 codex M2 closure: in CI, missing cohort artifacts FAIL the
    test (so the cohort gate cannot be silently elided). Locally,
    skip with a clear pointer.

    Honors:
      - ``CI=true`` (GitHub Actions / GitLab CI / generic CI marker)
        → fail with pytest.fail (artifact missing in CI is unacceptable
        because the cohort gate is required by v3 §6 T2.4 acceptance).
      - ``RUN_LOCAL_ONLY=1`` → explicit local-only opt-in; skip with
        a different message identifying the intentional opt-out.
      - default (no env vars) → skip locally.
    """
    is_ci = os.environ.get("CI", "").lower() in ("true", "1", "yes")
    is_local_only = os.environ.get("RUN_LOCAL_ONLY", "") == "1"

    if is_ci:
        pytest.fail(
            f"{cohort_name} cohort artifact missing at {artifact_path} in CI. "
            "G5 v3 §6 T2.4 acceptance criterion requires both cohort "
            "assertions to run. Provision the artifact or unset CI to skip "
            "locally. (To explicitly run locally without cohort data, set "
            "RUN_LOCAL_ONLY=1.)",
            pytrace=False,
        )

    if is_local_only:
        pytest.skip(
            f"{cohort_name} cohort artifact missing at {artifact_path}; "
            "RUN_LOCAL_ONLY=1 set, skipping cohort assertion intentionally."
        )

    pytest.skip(
        f"{cohort_name} journeys file not present at {artifact_path}; "
        f"skipping {cohort_name} coefficient-sensitivity assertions. "
        "(Set CI=true to fail on missing artifacts.)"
    )


@pytest.fixture(scope="module")
def csu_sensitivity() -> dict:
    """Run G5 on real CSU patient_journeys.

    Skip semantics (M2): CI fails on missing artifact; local skips.
    """
    if not CSU_JOURNEYS_PATH.exists():
        _cohort_artifact_missing_action(CSU_JOURNEYS_PATH, "CSU")
    return _run_g5_on_cohort(CSU_DATA_DIR, CSU_TARGET)


@pytest.fixture(scope="module")
def optum_initiation_sensitivity() -> dict:
    """Run G5 on real Optum initiation cohort.

    Skip semantics (M2): CI fails on missing artifact; local skips.
    """
    if not OPTUM_INITIATION_JOURNEYS_PATH.exists():
        _cohort_artifact_missing_action(OPTUM_INITIATION_JOURNEYS_PATH, "Optum-initiation")
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


# ---------------------------------------------------------------------------
# Failure-path coverage (G5 codex H4 closure).
#
# Every prior cohort assertion checks ``passes_pre_spec is True``. Without a
# negative-path test, a regression that makes T1/T2/T3 unreachable would go
# undetected (e.g., the H1 regression where T1 was trivially passable). These
# tests use _DeterministicCoefEstimator to construct controlled coefficient
# sequences that PROVE each threshold can fail when violated.
# ---------------------------------------------------------------------------


from typing import Iterator, List, Optional

from sklearn.base import BaseEstimator

from src.agents.ml_foundation.data_preparer.nodes.coefficient_sensitivity import (
    compute_coefficient_sensitivity,
)


class _DeterministicCoefEstimator(BaseEstimator):
    """Mock estimator: returns pre-determined coefs per fit call. Mirrors
    the unit-test helper. Local copy avoids cross-test-tier import."""

    def __init__(self, coef_sequence: List[np.ndarray]) -> None:
        self._coef_sequence = coef_sequence
        self._iter: Optional[Iterator[np.ndarray]] = None
        self.coef_: Optional[np.ndarray] = None

    def _ensure_iter(self) -> Iterator[np.ndarray]:
        if self._iter is None:
            self._iter = iter(self._coef_sequence)
        return self._iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DeterministicCoefEstimator":
        next_coef = next(self._ensure_iter())
        self.coef_ = np.atleast_2d(np.asarray(next_coef, dtype=np.float64))
        return self


def _failure_fixture_X_y(n_features: int = 4, n_rows: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    """Build a tiny X (with NaN cells in every column so each gets a
    per-feature re-fit) + arbitrary balanced y. The mock estimator
    ignores both — what matters is that the helper iterates over
    ``compared_features`` once for baseline and once per feature."""
    rng = np.random.default_rng(0)
    cols = {}
    for i in range(n_features):
        col = rng.standard_normal(size=n_rows)
        # Inject 30% NaN so the per-feature re-fit fires.
        mask = rng.uniform(size=n_rows) < 0.30
        col[mask] = np.nan
        cols[f"feat_{i}"] = col
    X = pd.DataFrame(cols)
    y = pd.Series(rng.integers(0, 2, size=n_rows))
    return X, y


class TestG5FailurePathCoverage:
    """H4 closure: prove T1, T2, T3 can each FAIL when violated.

    Without a negative-path test, the green cohort assertions could
    pass vacuously (e.g., if a regression makes a threshold unreachable
    or the helper silently zeroes out flips). These tests construct
    deterministic coefficient sequences that GUARANTEE each threshold
    is violated.
    """

    @pytest.mark.integration
    def test_single_flip_in_single_strategy_mode_does_not_fire_t1_but_fires_t3(
        self,
    ) -> None:
        """G5 codex pass-2 NEW MED — helper-vs-memo threshold alignment.

        Spec memo: "flips_per_feature ≤ 1" → 1 flip is tolerated; failure
        is strictly ``> 1``. In single-strategy mode flip_count is
        bounded by {0, 1} so T1 cannot fire here (forward-compat for
        multi-strategy sweeps). The single-flip violation IS still
        caught — by T3 (cohort fraction flipped > 0.10). This test
        pins both behaviors:

          - max_flips_per_feature_significant <= 1 (T1 NOT violated)
          - 1 flip / 1 significant feature = 1.0 > 0.10 → T3 violated
          - passes_pre_spec is False (overall failure via T3)

        Construct a coef sequence where the baseline has one large-
        magnitude (significant) feature and the per-feature re-fit on
        that feature flips its sign.
        """
        X, y = _failure_fixture_X_y(n_features=4)
        # Baseline: feat_0 has the largest magnitude, well above 1σ.
        # Other features have small symmetric magnitudes so feat_0 is
        # uniquely "significant".
        baseline = np.array([5.0, 0.1, -0.1, 0.05])
        # Per-feature re-fits (4 features → 4 re-fits in column order):
        feat_0_refit = np.array([-5.0, 0.1, -0.1, 0.05])  # FLIP
        feat_1_refit = np.array([5.0, 0.12, -0.1, 0.05])
        feat_2_refit = np.array([5.0, 0.1, -0.12, 0.05])
        feat_3_refit = np.array([5.0, 0.1, -0.1, 0.06])
        mock = _DeterministicCoefEstimator(
            [baseline, feat_0_refit, feat_1_refit, feat_2_refit, feat_3_refit]
        )

        recs = dict.fromkeys(X.columns, "drop_row_or_mean")
        result = compute_coefficient_sensitivity(X, y, recs, estimator=mock)

        # passes_pre_spec is False (T3 fires) — overall failure surfaced.
        assert result["passes_pre_spec"] is False, (
            f"T3 failure path expected passes_pre_spec=False; got True. "
            f"violations={result['violations']!r}"
        )
        violation_text = " ".join(result["violations"])
        # T1 must NOT fire (single flip is within tolerance per memo).
        assert "T1 violated" not in violation_text, (
            f"T1 fired on a single flip; spec memo allows ≤ 1 flip. "
            f"violations={result['violations']!r}"
        )
        # T3 must fire (1 flip / 1 sig feature = 1.0 > 0.10).
        assert "T3 violated" in violation_text, (
            f"Expected T3 violation in {result['violations']!r}; not found."
        )
        # Per-feature contract: feat_0 flipped, others did not.
        assert result["per_feature"]["feat_0"]["sign_flip"] is True
        assert result["per_feature"]["feat_0"]["flip_count"] == 1
        # max_flips_per_feature_significant must be <= 1 in single-strategy
        # mode (T1's tolerance is ≤ 1).
        assert result["aggregate"]["max_flips_per_feature_significant"] == 1
        assert (
            result["aggregate"]["max_flips_per_feature_significant"]
            <= G5_FLIPS_PER_FEATURE_MAX
        )

    @pytest.mark.integration
    def test_t2_can_fail_when_effect_size_cv_exceeds_ceiling(self) -> None:
        """T2 fails when a significant feature's effect-size CV exceeds
        G5_EFFECT_SIZE_CV_MAX=0.5. Engineer baseline=+10.0,
        post_impute=+0.5 → mean=5.25, std=4.75 → CV=0.905 > 0.5.
        No sign flip (both positive), so T1 + T3 do NOT fail. T2 alone
        is the load-bearing violation."""
        X, y = _failure_fixture_X_y(n_features=3)
        baseline = np.array([10.0, 0.05, -0.05])
        feat_0_refit = np.array([0.5, 0.05, -0.05])  # CV ≈ 0.905
        feat_1_refit = np.array([10.0, 0.05, -0.05])
        feat_2_refit = np.array([10.0, 0.05, -0.05])
        mock = _DeterministicCoefEstimator([baseline, feat_0_refit, feat_1_refit, feat_2_refit])

        recs = dict.fromkeys(X.columns, "drop_row_or_mean")
        result = compute_coefficient_sensitivity(X, y, recs, estimator=mock)

        assert result["passes_pre_spec"] is False, (
            f"T2 failure path expected passes_pre_spec=False; got True. "
            f"violations={result['violations']!r}, "
            f"aggregate={result['aggregate']!r}"
        )
        violation_text = " ".join(result["violations"])
        assert "T2 violated" in violation_text, (
            f"Expected T2 violation in {result['violations']!r}; not found."
        )
        # Verify CV is computed correctly: std([10.0, 0.5]) / |mean([10.0, 0.5])|
        expected_cv = float(np.std([10.0, 0.5], ddof=0)) / abs(float(np.mean([10.0, 0.5])))
        assert result["aggregate"]["max_effect_size_variance_significant"] == pytest.approx(
            expected_cv, rel=1e-6
        )
        assert result["aggregate"]["max_effect_size_variance_significant"] > G5_EFFECT_SIZE_CV_MAX
        # T1 and T3 must NOT fire (no flip, fraction_flipped=0).
        assert result["per_feature"]["feat_0"]["sign_flip"] is False
        assert result["aggregate"]["fraction_significant_flipped"] == 0.0

    @pytest.mark.integration
    def test_t3_can_fail_when_too_many_significant_features_flip(self) -> None:
        """T3 fails when fraction_significant_flipped > 0.10. Construct
        a 12-feature fixture where 8 features are "significant" (>1σ)
        and 2 of them flip → 2/8 = 0.25 > 0.10. T1 also fails (any
        flip in single-strategy mode triggers T1), but T3's failure is
        the load-bearing observation here."""
        X, y = _failure_fixture_X_y(n_features=12)
        # Build a baseline where 8 features are large-magnitude and 4
        # are noise. Std(|baseline|) determines sigma; pick magnitudes
        # so the 8 large-magnitude features cross the 1σ threshold.
        baseline = np.array(
            [
                5.0,
                4.5,
                4.8,
                5.2,
                4.7,
                5.1,
                4.9,
                5.0,  # 8 "significant" (>1σ when sigma ≈ 2.5)
                0.05,
                -0.05,
                0.08,
                -0.06,  # 4 noise
            ]
        )
        # Per-feature re-fits: feat_0 and feat_1 flip; others stay.
        feat_0_refit = baseline.copy()
        feat_0_refit[0] = -5.0  # flip
        feat_1_refit = baseline.copy()
        feat_1_refit[1] = -4.5  # flip
        # The remaining 10 re-fits are no-op coefs (same as baseline,
        # nudged by 0.001 to avoid CV degeneracy that would also trip T2).
        nudge = 1e-3
        no_op_refits = [baseline + nudge for _ in range(10)]

        mock = _DeterministicCoefEstimator([baseline, feat_0_refit, feat_1_refit, *no_op_refits])

        recs = dict.fromkeys(X.columns, "drop_row_or_mean")
        result = compute_coefficient_sensitivity(X, y, recs, estimator=mock)

        assert result["passes_pre_spec"] is False, (
            f"T3 failure path expected passes_pre_spec=False; got True. "
            f"violations={result['violations']!r}, "
            f"aggregate={result['aggregate']!r}"
        )
        violation_text = " ".join(result["violations"])
        assert "T3 violated" in violation_text, (
            f"Expected T3 violation in {result['violations']!r}; not found."
        )
        # 2 of 8 significant features flipped → 0.25 > 0.10.
        assert (
            result["aggregate"]["fraction_significant_flipped"]
            > G5_FRACTION_SIGNIFICANT_FLIPPED_MAX
        )
        assert result["aggregate"]["fraction_significant_flipped"] == pytest.approx(0.25, abs=1e-6)
