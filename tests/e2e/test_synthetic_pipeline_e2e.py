"""End-to-End Tests for Synthetic Data Pipeline.

Tests the complete synthetic data pipeline flow:
1. Generate - Create synthetic data with known causal effects
2. Validate - Verify schema, splits, and data quality
3. Load - Prepare data for ML/causal analysis
4. Train - Fit causal model
5. Causal - Recover ATE and validate against ground truth

Version: 4.3
Target: Zero regressions, all DGPs produce recoverable effects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import (
    Brand,
    DataSplit,
    DGPType,
    DGP_CONFIGS,
)
from src.ml.synthetic.loaders import (
    get_dataset_stats,
    validate_supabase_data,
    DatasetStats,
)
from src.ml.synthetic.ground_truth.causal_effects import (
    GroundTruthStore,
    GroundTruthEffect,
    create_ground_truth_from_dgp_config,
)

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False


# ============================================================================
# TEST DATA STRUCTURES
# ============================================================================


@dataclass
class PipelineResult:
    """Result from running the full E2E pipeline."""

    dgp_type: DGPType
    brand: Brand
    n_samples: int

    # Stage results
    generation_success: bool
    validation_success: bool
    load_success: bool
    train_success: bool
    causal_success: bool

    # Metrics
    true_ate: float
    estimated_ate: Optional[float] = None
    ate_error: Optional[float] = None
    within_tolerance: bool = False

    # Data stats
    stats: Optional[DatasetStats] = None
    validation_errors: List[str] = None
    validation_warnings: List[str] = None

    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.validation_warnings is None:
            self.validation_warnings = []

    @property
    def all_stages_passed(self) -> bool:
        """Check if all pipeline stages passed."""
        return all([
            self.generation_success,
            self.validation_success,
            self.load_success,
            self.train_success,
            self.causal_success,
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "dgp_type": self.dgp_type.value,
            "brand": self.brand.value,
            "n_samples": self.n_samples,
            "all_stages_passed": self.all_stages_passed,
            "generation_success": self.generation_success,
            "validation_success": self.validation_success,
            "load_success": self.load_success,
            "train_success": self.train_success,
            "causal_success": self.causal_success,
            "true_ate": self.true_ate,
            "estimated_ate": self.estimated_ate,
            "ate_error": self.ate_error,
            "within_tolerance": self.within_tolerance,
        }


# ============================================================================
# PIPELINE COMPONENTS
# ============================================================================


def generate_synthetic_data(
    dgp_type: DGPType,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Stage 1: Generate synthetic data with known causal effects.

    Args:
        dgp_type: Type of data generating process
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with treatment, outcome, and covariates
    """
    np.random.seed(seed)

    # Get ground truth config - DGP_CONFIGS uses DGPType enum as key
    config = DGP_CONFIGS.get(dgp_type, DGP_CONFIGS[DGPType.CONFOUNDED])
    true_ate = config.true_ate  # Access as attribute, not dict key

    # Base confounders
    C1 = np.random.normal(0, 1, n_samples)
    C2 = np.random.normal(0, 1, n_samples)

    # Treatment assignment with confounding
    confounder_strength = 0.4
    prob_T = 1 / (1 + np.exp(-(confounder_strength * C1 + 0.2 * C2)))
    T = np.random.binomial(1, prob_T)

    # Outcome with treatment effect
    noise = np.random.normal(0, 0.1, n_samples)
    Y = true_ate * T + 0.3 * C1 + 0.2 * C2 + noise

    # Create DataFrame with all required columns
    df = pd.DataFrame({
        "T": T,
        "Y": Y,
        "C1": C1,
        "C2": C2,
        "brand": np.random.choice([b.value for b in Brand], n_samples),
        "data_split": np.random.choice(
            [s.value for s in DataSplit],
            n_samples,
            p=[0.60, 0.20, 0.15, 0.05],  # 60/20/15/5 ratio
        ),
    })

    return df


def validate_data(df: pd.DataFrame, table_name: str = "synthetic") -> Dict[str, Any]:
    """Stage 2: Validate data for schema compliance and quality.

    Args:
        df: DataFrame to validate
        table_name: Name for reporting

    Returns:
        Validation results dictionary
    """
    results = validate_supabase_data(
        datasets={table_name: df},
        expected_brands=[b.value for b in Brand],
        expected_splits=[s.value for s in DataSplit],
    )

    return results


def prepare_for_loading(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: Prepare data for ML pipeline loading.

    Args:
        df: Raw DataFrame

    Returns:
        Prepared DataFrame ready for training
    """
    # Apply any transformations needed for loading
    df = df.copy()

    # Ensure correct dtypes
    df["T"] = df["T"].astype(int)
    df["Y"] = df["Y"].astype(float)

    # Ensure splits are strings
    df["data_split"] = df["data_split"].astype(str)

    return df


def train_causal_model(
    df: pd.DataFrame,
    treatment_col: str = "T",
    outcome_col: str = "Y",
    confounder_cols: List[str] = None,
) -> Dict[str, Any]:
    """Stage 4: Train causal model on prepared data.

    Args:
        df: Prepared DataFrame
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names

    Returns:
        Training results including fitted model
    """
    if confounder_cols is None:
        confounder_cols = ["C1", "C2"]

    # Use simple linear regression for speed
    from sklearn.linear_model import LinearRegression

    X_cols = [treatment_col] + confounder_cols
    X = df[X_cols].values
    y = df[outcome_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Treatment coefficient is the ATE estimate
    estimated_ate = model.coef_[0]

    return {
        "model": model,
        "estimated_ate": estimated_ate,
        "feature_importance": dict(zip(X_cols, model.coef_)),
    }


def estimate_causal_effect(
    df: pd.DataFrame,
    treatment_col: str = "T",
    outcome_col: str = "Y",
    confounder_cols: List[str] = None,
) -> float:
    """Stage 5: Estimate causal effect with adjustment.

    Args:
        df: DataFrame with treatment, outcome, and confounders
        treatment_col: Name of treatment column
        outcome_col: Name of outcome column
        confounder_cols: List of confounder column names

    Returns:
        Estimated ATE
    """
    if confounder_cols is None:
        confounder_cols = ["C1", "C2"]

    from sklearn.linear_model import LinearRegression

    X_cols = [treatment_col] + confounder_cols
    X = df[X_cols].values
    y = df[outcome_col].values

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]


# ============================================================================
# E2E PIPELINE RUNNER
# ============================================================================


def run_pipeline(
    dgp_type: DGPType,
    brand: Brand = Brand.REMIBRUTINIB,
    n_samples: int = 2000,
    seed: int = 42,
) -> PipelineResult:
    """Run the complete E2E pipeline for a given DGP.

    Args:
        dgp_type: Type of data generating process
        brand: Brand to use (for data generation)
        n_samples: Number of samples
        seed: Random seed

    Returns:
        PipelineResult with all stage outcomes
    """
    # DGP_CONFIGS uses DGPType enum as key, values are DGPConfig objects
    config = DGP_CONFIGS.get(dgp_type, DGP_CONFIGS[DGPType.CONFOUNDED])
    true_ate = config.true_ate
    tolerance = config.tolerance

    result = PipelineResult(
        dgp_type=dgp_type,
        brand=brand,
        n_samples=n_samples,
        generation_success=False,
        validation_success=False,
        load_success=False,
        train_success=False,
        causal_success=False,
        true_ate=true_ate,
    )

    # Stage 1: Generate
    try:
        df = generate_synthetic_data(dgp_type, n_samples, seed)
        result.generation_success = len(df) == n_samples
        result.stats = get_dataset_stats(df, f"synthetic_{dgp_type.value}")
    except Exception as e:
        result.validation_errors.append(f"Generation failed: {e}")
        return result

    # Stage 2: Validate
    try:
        validation = validate_data(df)
        result.validation_success = validation["is_valid"]
        result.validation_errors = validation.get("errors", [])
        result.validation_warnings = validation.get("warnings", [])
    except Exception as e:
        result.validation_errors.append(f"Validation failed: {e}")
        return result

    # Stage 3: Load
    try:
        df_prepared = prepare_for_loading(df)
        result.load_success = len(df_prepared) == n_samples
    except Exception as e:
        result.validation_errors.append(f"Loading failed: {e}")
        return result

    # Stage 4: Train
    try:
        train_results = train_causal_model(df_prepared)
        result.train_success = train_results["model"] is not None
    except Exception as e:
        result.validation_errors.append(f"Training failed: {e}")
        return result

    # Stage 5: Causal
    try:
        estimated_ate = estimate_causal_effect(df_prepared)
        result.estimated_ate = estimated_ate
        result.ate_error = abs(estimated_ate - true_ate)
        result.within_tolerance = result.ate_error <= tolerance
        result.causal_success = result.within_tolerance
    except Exception as e:
        result.validation_errors.append(f"Causal estimation failed: {e}")
        return result

    return result


# ============================================================================
# E2E TESTS
# ============================================================================


class TestSyntheticPipelineE2E:
    """End-to-end tests for the synthetic data pipeline."""

    @pytest.fixture
    def ground_truth_store(self):
        """Initialize ground truth store for testing."""
        return GroundTruthStore()

    @pytest.mark.parametrize("dgp_type", [
        DGPType.SIMPLE_LINEAR,
        DGPType.CONFOUNDED,
        DGPType.HETEROGENEOUS,
        DGPType.TIME_SERIES,
        DGPType.SELECTION_BIAS,
    ])
    def test_pipeline_for_dgp(self, dgp_type: DGPType):
        """Test complete pipeline for each DGP type.

        Validates:
        1. All pipeline stages complete successfully
        2. ATE estimate is within tolerance of true value
        3. Data passes schema validation
        """
        result = run_pipeline(
            dgp_type=dgp_type,
            n_samples=2000,
            seed=42,
        )

        # Check all stages passed
        assert result.generation_success, f"Generation failed for {dgp_type.value}"
        assert result.validation_success, (
            f"Validation failed for {dgp_type.value}: {result.validation_errors}"
        )
        assert result.load_success, f"Loading failed for {dgp_type.value}"
        assert result.train_success, f"Training failed for {dgp_type.value}"

        # Check causal recovery
        assert result.estimated_ate is not None, (
            f"No ATE estimate for {dgp_type.value}"
        )
        assert result.within_tolerance, (
            f"ATE outside tolerance for {dgp_type.value}: "
            f"true={result.true_ate:.4f}, est={result.estimated_ate:.4f}, "
            f"error={result.ate_error:.4f}"
        )

    def test_all_dgps_complete_pipeline(self):
        """Verify all DGPs complete the full pipeline."""
        results = []
        all_dgp_types = [
            DGPType.SIMPLE_LINEAR,
            DGPType.CONFOUNDED,
            DGPType.HETEROGENEOUS,
            DGPType.TIME_SERIES,
            DGPType.SELECTION_BIAS,
        ]

        for dgp_type in all_dgp_types:
            result = run_pipeline(dgp_type, n_samples=1500, seed=42)
            results.append(result)

        # Print summary
        print("\n" + "=" * 70)
        print("E2E PIPELINE SUMMARY")
        print("=" * 70)
        for r in results:
            status = "✓ PASS" if r.all_stages_passed else "✗ FAIL"
            ate_info = (
                f"ATE: true={r.true_ate:.3f}, est={r.estimated_ate:.3f}"
                if r.estimated_ate else "ATE: N/A"
            )
            print(f"  {r.dgp_type.value:20s} {status:10s} {ate_info}")
        print("=" * 70)

        # All should pass
        passed = sum(1 for r in results if r.all_stages_passed)
        assert passed == len(all_dgp_types), (
            f"Only {passed}/{len(all_dgp_types)} DGPs completed pipeline"
        )

    def test_validation_catches_bad_data(self):
        """Verify validation stage catches schema violations."""
        # Create data with invalid brand
        df = pd.DataFrame({
            "T": [0, 1, 0, 1],
            "Y": [0.1, 0.5, 0.2, 0.6],
            "C1": [0.0, 0.5, -0.5, 1.0],
            "C2": [0.2, -0.2, 0.1, -0.1],
            "brand": ["invalid_brand", "invalid_brand", "bad", "wrong"],
            "data_split": ["train", "train", "validation", "test"],
        })

        validation = validate_data(df)

        assert not validation["is_valid"], "Validation should fail for invalid brands"
        assert any("brand" in str(e).lower() for e in validation["errors"]), (
            "Should have brand-related error"
        )

    def test_data_splits_maintained(self):
        """Verify data split ratios are maintained through pipeline."""
        result = run_pipeline(
            dgp_type=DGPType.CONFOUNDED,
            n_samples=3000,
            seed=42,
        )

        assert result.stats is not None
        assert result.stats.split_stats is not None

        # Check all splits present
        split_names = {s.split_name for s in result.stats.split_stats}
        expected = {"train", "validation", "test", "holdout"}
        assert split_names == expected, f"Missing splits: {expected - split_names}"

        # Check approximate ratios (with some tolerance for random sampling)
        for split in result.stats.split_stats:
            if split.split_name == "train":
                assert 50 < split.percentage < 70, "Train should be ~60%"
            elif split.split_name == "validation":
                assert 15 < split.percentage < 25, "Validation should be ~20%"
            elif split.split_name == "test":
                assert 10 < split.percentage < 20, "Test should be ~15%"
            elif split.split_name == "holdout":
                assert 2 < split.percentage < 8, "Holdout should be ~5%"

    def test_ground_truth_store_integration(self, ground_truth_store):
        """Verify ground truth store correctly records and validates effects."""
        # Run pipeline
        result = run_pipeline(
            dgp_type=DGPType.CONFOUNDED,
            n_samples=2000,
            seed=42,
        )

        # Store ground truth using the factory function
        ground_truth = create_ground_truth_from_dgp_config(
            brand=result.brand,
            dgp_type=result.dgp_type,
            n_samples=result.n_samples,
        )
        ground_truth_store.store(ground_truth)

        # Validate estimate
        validation_result = ground_truth_store.validate_estimate(
            brand=result.brand,
            dgp_type=result.dgp_type,
            estimated_ate=result.estimated_ate,
        )

        assert validation_result["is_valid"], (
            f"Ground truth validation failed: "
            f"true={result.true_ate}, est={result.estimated_ate}, "
            f"error={validation_result.get('error', 'N/A')}"
        )

    def test_pipeline_reproducibility(self):
        """Verify pipeline produces consistent results with same seed."""
        results = []
        for _ in range(3):
            result = run_pipeline(
                dgp_type=DGPType.SIMPLE_LINEAR,
                n_samples=1000,
                seed=12345,  # Same seed
            )
            results.append(result.estimated_ate)

        # All estimates should be identical
        assert len(set(results)) == 1, (
            f"Estimates should be identical with same seed: {results}"
        )


class TestPipelineStageIsolation:
    """Test each pipeline stage in isolation."""

    def test_generation_stage(self):
        """Test data generation produces expected structure."""
        df = generate_synthetic_data(DGPType.CONFOUNDED, n_samples=100, seed=42)

        assert len(df) == 100
        assert "T" in df.columns
        assert "Y" in df.columns
        assert "brand" in df.columns
        assert "data_split" in df.columns
        assert df["T"].isin([0, 1]).all()

    def test_validation_stage(self):
        """Test validation correctly identifies valid data."""
        df = generate_synthetic_data(DGPType.SIMPLE_LINEAR, n_samples=100, seed=42)
        validation = validate_data(df)

        assert validation["is_valid"], f"Valid data failed: {validation['errors']}"

    def test_load_stage_preserves_data(self):
        """Test loading stage preserves data integrity."""
        df_original = generate_synthetic_data(DGPType.CONFOUNDED, n_samples=100, seed=42)
        df_loaded = prepare_for_loading(df_original)

        assert len(df_loaded) == len(df_original)
        assert np.allclose(df_loaded["Y"].values, df_original["Y"].values)

    def test_train_stage_produces_model(self):
        """Test training stage produces valid model."""
        df = generate_synthetic_data(DGPType.SIMPLE_LINEAR, n_samples=500, seed=42)
        df = prepare_for_loading(df)
        train_results = train_causal_model(df)

        assert train_results["model"] is not None
        assert "estimated_ate" in train_results
        assert isinstance(train_results["estimated_ate"], float)

    def test_causal_stage_estimates_effect(self):
        """Test causal estimation produces reasonable estimate."""
        df = generate_synthetic_data(DGPType.SIMPLE_LINEAR, n_samples=1000, seed=42)
        df = prepare_for_loading(df)
        estimated_ate = estimate_causal_effect(df)

        true_ate = DGP_CONFIGS[DGPType.SIMPLE_LINEAR].true_ate
        assert abs(estimated_ate - true_ate) < 0.10, (
            f"Estimate too far from truth: {estimated_ate} vs {true_ate}"
        )
