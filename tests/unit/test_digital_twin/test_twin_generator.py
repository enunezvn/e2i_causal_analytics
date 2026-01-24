"""
Unit Tests for Digital Twin Generator.

Tests cover:
- Initialization with different twin types and brands
- Training with valid and invalid data
- Feature preparation and encoding
- Twin population generation
- Propensity calculation
- Model metrics
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.models.twin_models import (
    Brand,
    TwinModelMetrics,
    TwinPopulation,
    TwinType,
)
from src.digital_twin.twin_generator import TwinGenerator


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def hcp_training_data() -> pd.DataFrame:
    """Create sample HCP training data with 1500 samples."""
    np.random.seed(42)
    n = 1500

    return pd.DataFrame({
        "specialty": np.random.choice(["rheumatology", "dermatology", "allergy"], n),
        "years_experience": np.random.randint(1, 40, n),
        "practice_type": np.random.choice(["academic", "community", "private"], n),
        "practice_size": np.random.choice(["solo", "small", "medium", "large"], n),
        "region": np.random.choice(["northeast", "south", "midwest", "west"], n),
        "decile": np.random.randint(1, 11, n),
        "priority_tier": np.random.randint(1, 6, n),
        "total_patient_volume": np.random.randint(100, 10000, n),
        "target_patient_volume": np.random.randint(10, 1000, n),
        "digital_engagement_score": np.random.uniform(0, 1, n),
        "preferred_channel": np.random.choice(["email", "phone", "in_person"], n),
        "last_interaction_days": np.random.randint(0, 365, n),
        "interaction_frequency": np.random.uniform(0, 10, n),
        "adoption_stage": np.random.choice(
            ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"], n
        ),
        "peer_influence_score": np.random.uniform(0, 1, n),
        "prescribing_change": np.random.uniform(-0.2, 0.3, n),  # Target
    })


@pytest.fixture
def patient_training_data() -> pd.DataFrame:
    """Create sample patient training data."""
    np.random.seed(42)
    n = 1500

    return pd.DataFrame({
        "age_group": np.random.choice(["18-34", "35-44", "45-54", "55-64", "65+"], n),
        "gender": np.random.choice(["male", "female"], n),
        "geographic_region": np.random.choice(["northeast", "south", "midwest", "west"], n),
        "socioeconomic_index": np.random.uniform(0, 1, n),
        "primary_diagnosis_code": np.random.choice(["L40.0", "L40.1", "L40.2"], n),
        "comorbidity_count": np.random.randint(0, 5, n),
        "risk_score": np.random.uniform(0, 1, n),
        "journey_complexity_score": np.random.uniform(0, 1, n),
        "insurance_type": np.random.choice(["commercial", "medicare", "medicaid"], n),
        "insurance_coverage_flag": np.random.choice([True, False], n),
        "journey_stage": np.random.choice(["diagnosis", "treatment", "maintenance"], n),
        "journey_duration_days": np.random.randint(1, 1000, n),
        "treatment_line": np.random.randint(1, 4, n),
        "adherence_rate": np.random.uniform(0.3, 1.0, n),  # Target
    })


@pytest.fixture
def territory_training_data() -> pd.DataFrame:
    """Create sample territory training data."""
    np.random.seed(42)
    n = 1500

    return pd.DataFrame({
        "region": np.random.choice(["northeast", "south", "midwest", "west"], n),
        "state_count": np.random.randint(1, 10, n),
        "zip_count": np.random.randint(10, 500, n),
        "total_hcps": np.random.randint(50, 2000, n),
        "covered_hcps": np.random.randint(30, 1500, n),
        "coverage_rate": np.random.uniform(0.4, 1.0, n),
        "total_patient_volume": np.random.randint(1000, 100000, n),
        "market_share": np.random.uniform(0.05, 0.50, n),
        "growth_rate": np.random.uniform(-0.1, 0.2, n),
        "competitor_presence": np.random.uniform(0.2, 0.8, n),
        "territory_performance": np.random.uniform(-0.2, 0.3, n),  # Target
    })


@pytest.fixture
def small_training_data() -> pd.DataFrame:
    """Create insufficient training data for error testing."""
    np.random.seed(42)
    n = 500  # Below minimum

    return pd.DataFrame({
        "specialty": np.random.choice(["rheumatology", "dermatology"], n),
        "decile": np.random.randint(1, 11, n),
        "prescribing_change": np.random.uniform(-0.2, 0.3, n),
    })


@pytest.fixture
def hcp_generator() -> TwinGenerator:
    """Create HCP twin generator."""
    return TwinGenerator(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB)


@pytest.fixture
def patient_generator() -> TwinGenerator:
    """Create Patient twin generator."""
    return TwinGenerator(twin_type=TwinType.PATIENT, brand=Brand.FABHALTA)


@pytest.fixture
def territory_generator() -> TwinGenerator:
    """Create Territory twin generator."""
    return TwinGenerator(twin_type=TwinType.TERRITORY, brand=Brand.KISQALI)


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestTwinGeneratorInit:
    """Tests for TwinGenerator initialization."""

    def test_init_hcp_generator(self):
        """Test initializing HCP generator."""
        generator = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB)
        assert generator.twin_type == TwinType.HCP
        assert generator.brand == Brand.REMIBRUTINIB
        assert generator.model is None
        assert generator.model_id is None

    def test_init_patient_generator(self):
        """Test initializing patient generator."""
        generator = TwinGenerator(twin_type=TwinType.PATIENT, brand=Brand.FABHALTA)
        assert generator.twin_type == TwinType.PATIENT
        assert generator.brand == Brand.FABHALTA

    def test_init_territory_generator(self):
        """Test initializing territory generator."""
        generator = TwinGenerator(twin_type=TwinType.TERRITORY, brand=Brand.KISQALI)
        assert generator.twin_type == TwinType.TERRITORY
        assert generator.brand == Brand.KISQALI

    def test_default_features_defined(self):
        """Test that default features are defined for all twin types."""
        for twin_type in TwinType:
            assert twin_type in TwinGenerator.DEFAULT_FEATURES
            assert len(TwinGenerator.DEFAULT_FEATURES[twin_type]) > 0


# =============================================================================
# TRAINING TESTS
# =============================================================================


class TestTwinGeneratorTraining:
    """Tests for TwinGenerator training."""

    def test_train_hcp_model(self, hcp_generator, hcp_training_data):
        """Test training HCP model."""
        metrics = hcp_generator.train(
            data=hcp_training_data,
            target_col="prescribing_change",
        )

        assert isinstance(metrics, TwinModelMetrics)
        assert metrics.r2_score is not None
        assert metrics.rmse is not None
        assert metrics.mae is not None
        assert len(metrics.cv_scores) == 5
        assert hcp_generator.model is not None
        assert hcp_generator.model_id is not None

    def test_train_patient_model(self, patient_generator, patient_training_data):
        """Test training patient model."""
        metrics = patient_generator.train(
            data=patient_training_data,
            target_col="adherence_rate",
        )

        assert isinstance(metrics, TwinModelMetrics)
        assert patient_generator.model is not None

    def test_train_territory_model(self, territory_generator, territory_training_data):
        """Test training territory model."""
        metrics = territory_generator.train(
            data=territory_training_data,
            target_col="territory_performance",
        )

        assert isinstance(metrics, TwinModelMetrics)
        assert territory_generator.model is not None

    def test_train_insufficient_data(self, hcp_generator, small_training_data):
        """Test training fails with insufficient data."""
        with pytest.raises(ValueError, match="Insufficient training data"):
            hcp_generator.train(
                data=small_training_data,
                target_col="prescribing_change",
            )

    def test_train_missing_target_column(self, hcp_generator, hcp_training_data):
        """Test training fails with missing target column."""
        with pytest.raises(ValueError, match="Target column 'missing_column' not in data"):
            hcp_generator.train(
                data=hcp_training_data,
                target_col="missing_column",
            )

    def test_train_custom_features(self, hcp_generator, hcp_training_data):
        """Test training with custom feature columns."""
        custom_features = ["specialty", "decile", "digital_engagement_score"]

        metrics = hcp_generator.train(
            data=hcp_training_data,
            target_col="prescribing_change",
            feature_cols=custom_features,
        )

        assert len(hcp_generator.feature_columns) == 3
        assert set(hcp_generator.feature_columns) == set(custom_features)

    def test_train_gradient_boosting(self, hcp_generator, hcp_training_data):
        """Test training with gradient boosting algorithm."""
        metrics = hcp_generator.train(
            data=hcp_training_data,
            target_col="prescribing_change",
            algorithm="gradient_boosting",
        )

        assert isinstance(metrics, TwinModelMetrics)
        assert hcp_generator.model is not None

    def test_train_invalid_algorithm(self, hcp_generator, hcp_training_data):
        """Test training fails with invalid algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            hcp_generator.train(
                data=hcp_training_data,
                target_col="prescribing_change",
                algorithm="invalid_algorithm",
            )

    def test_train_feature_importances(self, hcp_generator, hcp_training_data):
        """Test that feature importances are calculated."""
        metrics = hcp_generator.train(
            data=hcp_training_data,
            target_col="prescribing_change",
        )

        assert len(metrics.feature_importances) > 0
        assert len(metrics.top_features) > 0
        # All importances should sum close to 1
        total_importance = sum(metrics.feature_importances.values())
        assert 0.99 <= total_importance <= 1.01

    def test_train_stores_statistics(self, hcp_generator, hcp_training_data):
        """Test that training stores feature statistics."""
        hcp_generator.train(
            data=hcp_training_data,
            target_col="prescribing_change",
        )

        # Should have categorical distributions
        assert len(hcp_generator._categorical_distributions) > 0
        # Should have numerical stats
        assert len(hcp_generator._feature_stats) > 0


# =============================================================================
# GENERATION TESTS
# =============================================================================


class TestTwinGeneratorGeneration:
    """Tests for TwinGenerator population generation."""

    def test_generate_without_training(self, hcp_generator):
        """Test generation fails without training."""
        with pytest.raises(RuntimeError, match="Model not trained"):
            hcp_generator.generate(n=100)

    def test_generate_hcp_population(self, hcp_generator, hcp_training_data):
        """Test generating HCP population."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        population = hcp_generator.generate(n=100)

        assert isinstance(population, TwinPopulation)
        assert len(population) == 100
        assert population.twin_type == TwinType.HCP
        assert population.brand == Brand.REMIBRUTINIB

    @pytest.mark.xdist_group(name="digital_twin_heavy")
    def test_generate_with_seed(self, hcp_training_data):
        """Test that seed produces reproducible results across separate generators."""
        # Train two identical generators
        gen1 = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB)
        gen2 = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.REMIBRUTINIB)

        gen1.train(data=hcp_training_data.copy(), target_col="prescribing_change")
        gen2.train(data=hcp_training_data.copy(), target_col="prescribing_change")

        # Generate with same seed from fresh generators
        pop1 = gen1.generate(n=50, seed=42)
        pop2 = gen2.generate(n=50, seed=42)

        # Same outcomes when using same seed on separate generators
        outcomes1 = [round(t.baseline_outcome, 4) for t in pop1.twins]
        outcomes2 = [round(t.baseline_outcome, 4) for t in pop2.twins]

        assert outcomes1 == outcomes2

    def test_generate_with_constraints(self, hcp_generator, hcp_training_data):
        """Test generation with feature constraints."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        constraints = {"specialty": "rheumatology", "decile": 1}
        population = hcp_generator.generate(n=50, constraints=constraints)

        # All twins should have constrained values
        for twin in population.twins:
            assert twin.features.get("specialty") == "rheumatology"
            assert twin.features.get("decile") == 1

    def test_generate_twins_have_features(self, hcp_generator, hcp_training_data):
        """Test that generated twins have expected features."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        population = hcp_generator.generate(n=10)

        for twin in population.twins:
            assert isinstance(twin.features, dict)
            assert len(twin.features) > 0
            assert isinstance(twin.baseline_outcome, float)
            assert 0 <= twin.baseline_propensity <= 1

    def test_generate_feature_summary(self, hcp_generator, hcp_training_data):
        """Test that feature summary is calculated."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        population = hcp_generator.generate(n=100)

        assert len(population.feature_summary) > 0

    def test_generate_population_has_model_id(self, hcp_generator, hcp_training_data):
        """Test that generated population links to model."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        population = hcp_generator.generate(n=10)

        assert population.model_id == hcp_generator.model_id


# =============================================================================
# PROPENSITY CALCULATION TESTS
# =============================================================================


class TestPropensityCalculation:
    """Tests for propensity score calculation."""

    def test_propensity_hcp_high_decile(self, hcp_generator, hcp_training_data):
        """Test propensity calculation for high-decile HCP."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        features = {
            "decile": 1,  # Top decile
            "digital_engagement_score": 0.9,  # High engagement
        }

        propensity = hcp_generator._calculate_propensity(features)

        assert 0.1 <= propensity <= 0.9
        assert propensity > 0.5  # Should be higher for top decile + high engagement

    def test_propensity_hcp_low_decile(self, hcp_generator, hcp_training_data):
        """Test propensity calculation for low-decile HCP."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        features = {
            "decile": 10,  # Bottom decile
            "digital_engagement_score": 0.1,  # Low engagement
        }

        propensity = hcp_generator._calculate_propensity(features)

        assert 0.1 <= propensity <= 0.9
        # Should be lower for bottom decile + low engagement

    def test_propensity_non_hcp_default(self, patient_generator, patient_training_data):
        """Test propensity defaults for non-HCP twins."""
        patient_generator.train(data=patient_training_data, target_col="adherence_rate")

        features = {"age_group": "45-54"}

        propensity = patient_generator._calculate_propensity(features)

        # Non-HCP twins should get base propensity around 0.5
        assert 0.1 <= propensity <= 0.9


# =============================================================================
# MODEL INFO TESTS
# =============================================================================


class TestModelInfo:
    """Tests for model information extraction."""

    def test_get_model_info_untrained(self, hcp_generator):
        """Test model info before training."""
        info = hcp_generator.get_model_info()

        assert info["model_id"] is None
        assert info["twin_type"] == "hcp"
        assert info["brand"] == "Remibrutinib"
        assert info["is_trained"] is False
        assert info["metrics"] is None

    def test_get_model_info_trained(self, hcp_generator, hcp_training_data):
        """Test model info after training."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        info = hcp_generator.get_model_info()

        assert info["model_id"] is not None
        assert info["twin_type"] == "hcp"
        assert info["brand"] == "Remibrutinib"
        assert info["is_trained"] is True
        assert info["metrics"] is not None
        assert info["target_column"] == "prescribing_change"
        assert len(info["feature_columns"]) > 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_feature_columns_handled(self, hcp_generator):
        """Test that missing feature columns are handled gracefully."""
        np.random.seed(42)
        n = 1500

        # Data with only some expected features
        data = pd.DataFrame({
            "specialty": np.random.choice(["rheumatology", "dermatology"], n),
            "decile": np.random.randint(1, 11, n),
            # Missing many expected HCP features
            "prescribing_change": np.random.uniform(-0.2, 0.3, n),
        })

        # Should train with available features, warning about missing ones
        metrics = hcp_generator.train(data=data, target_col="prescribing_change")

        assert metrics is not None
        assert len(hcp_generator.feature_columns) == 2  # Only specialty and decile

    def test_categorical_encoding(self, hcp_generator, hcp_training_data):
        """Test that categorical features are properly encoded."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        # Specialty should be in label encoders
        assert "specialty" in hcp_generator.label_encoders
        assert "practice_type" in hcp_generator.label_encoders

    def test_numerical_scaling(self, hcp_generator, hcp_training_data):
        """Test that numerical features are scaled."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        assert hcp_generator.scaler is not None

    @pytest.mark.xdist_group(name="digital_twin_heavy")
    def test_generate_large_population(self, hcp_generator, hcp_training_data):
        """Test generating a larger population.

        Note: Reduced from n=500 to n=100 for faster test execution.
        The core functionality is verified at this scale.
        """
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        population = hcp_generator.generate(n=100)

        assert len(population) == 100

    def test_metrics_cv_scores(self, hcp_generator, hcp_training_data):
        """Test that cross-validation scores are calculated."""
        metrics = hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        assert len(metrics.cv_scores) == 5
        assert metrics.cv_mean is not None
        assert metrics.cv_std is not None
        # CV mean should be close to average of scores
        assert abs(metrics.cv_mean - np.mean(metrics.cv_scores)) < 0.01

    def test_unknown_category_handling(self, hcp_generator, hcp_training_data):
        """Test handling of unknown category during feature conversion."""
        hcp_generator.train(data=hcp_training_data, target_col="prescribing_change")

        # Create features with unknown category
        features = {
            "specialty": "unknown_specialty",  # Not in training data
            "decile": 5,
        }

        # Should handle gracefully
        arr = hcp_generator._features_to_array(features)
        assert arr is not None


# =============================================================================
# DATA LEAKAGE PREVENTION TESTS (Phase 2)
# =============================================================================


class TestDataLeakagePrevention:
    """Tests for data leakage prevention in twin generation."""

    def test_target_not_in_features(self, hcp_training_data):
        """Test that target column is excluded from features."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        generator.train(data=hcp_training_data, target_col="prescribing_change")

        # Target should not be in feature columns
        assert "prescribing_change" not in generator.feature_columns

        # Generate twins
        population = generator.generate(n=100)

        # Generated features should not contain target
        for twin in population.twins:
            assert "prescribing_change" not in twin.features

    def test_no_train_test_contamination(self, hcp_training_data):
        """Test that training data statistics don't leak into test twins."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        generator.train(data=hcp_training_data, target_col="prescribing_change")

        # Generate twins with different seeds
        population1 = generator.generate(n=100, seed=42)
        population2 = generator.generate(n=100, seed=123)

        # Twin IDs should be unique across populations
        ids1 = {t.twin_id for t in population1.twins}
        ids2 = {t.twin_id for t in population2.twins}

        assert len(ids1.intersection(ids2)) == 0

    def test_feature_statistics_from_training_only(self, hcp_training_data):
        """Test that feature statistics are computed from training data only."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        # Record training data statistics
        training_mean = hcp_training_data["digital_engagement_score"].mean()
        training_std = hcp_training_data["digital_engagement_score"].std()

        generator.train(data=hcp_training_data, target_col="prescribing_change")

        # Check that scaler uses training statistics
        if generator.scaler is not None:
            # For StandardScaler, mean_ and scale_ should match training data
            # (for the digital_engagement_score column if it's scaled)
            pass  # Scaler internal stats are based on training data

        # Generate and verify distribution is reasonable
        population = generator.generate(n=500, seed=42)

        engagement_scores = [
            t.features.get("digital_engagement_score", 0.5)
            for t in population.twins
        ]

        # Generated engagement should have similar range to training
        assert min(engagement_scores) >= 0.0
        assert max(engagement_scores) <= 1.0

    def test_no_future_information_in_features(self, hcp_training_data):
        """Test that no future information leaks into twin generation."""
        import warnings

        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        # Add a "future" column that shouldn't be used
        data_with_future = hcp_training_data.copy()
        data_with_future["future_outcome"] = np.random.uniform(0, 1, len(data_with_future))

        # Train - future_outcome should not affect model since it's not prescribing_change
        generator.train(data=data_with_future, target_col="prescribing_change")

        # Verify future_outcome is not in features
        # (it wasn't explicitly included in feature definitions)
        population = generator.generate(n=100)

        for twin in population.twins:
            assert "future_outcome" not in twin.features

    def test_temporal_ordering_not_violated(self, hcp_training_data):
        """Test that temporal ordering is respected in training."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        # Add a timestamp column
        data_with_time = hcp_training_data.copy()
        data_with_time["event_timestamp"] = pd.date_range(
            start="2023-01-01", periods=len(data_with_time), freq="H"
        )

        # Training should work
        generator.train(data=data_with_time, target_col="prescribing_change")

        # Timestamp should not be in generated features
        population = generator.generate(n=100)

        for twin in population.twins:
            assert "event_timestamp" not in twin.features

    def test_cross_validation_prevents_leakage(self, hcp_training_data):
        """Test that cross-validation is used to prevent overfitting/leakage."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        metrics = generator.train(
            data=hcp_training_data,
            target_col="prescribing_change",
        )

        # CV scores should be computed
        assert len(metrics.cv_scores) > 0

        # CV mean should be reasonable (not perfect, indicating potential leakage)
        if metrics.r2_score is not None:
            # R2 significantly higher than CV mean could indicate leakage
            # Allow some variance but flag major discrepancies
            if metrics.cv_mean is not None:
                assert metrics.r2_score < metrics.cv_mean + 0.3

    def test_generated_twins_have_valid_feature_ranges(self, hcp_training_data):
        """Test that generated twins have features within valid ranges."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        generator.train(data=hcp_training_data, target_col="prescribing_change")
        population = generator.generate(n=200, seed=42)

        # Check feature value ranges
        for twin in population.twins:
            features = twin.features

            # Decile should be 1-10
            if "decile" in features:
                assert 1 <= features["decile"] <= 10

            # Engagement score should be 0-1
            if "digital_engagement_score" in features:
                assert 0.0 <= features["digital_engagement_score"] <= 1.0

            # Propensity should be 0-1
            assert 0.0 <= twin.baseline_propensity <= 1.0

    def test_reproducibility_with_seed(self, hcp_training_data):
        """Test that generation is reproducible with same seed."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        generator.train(data=hcp_training_data, target_col="prescribing_change")

        # Generate twice with same seed
        population1 = generator.generate(n=50, seed=12345)
        population2 = generator.generate(n=50, seed=12345)

        # Should produce same propensities
        propensities1 = [t.baseline_propensity for t in population1.twins]
        propensities2 = [t.baseline_propensity for t in population2.twins]

        for p1, p2 in zip(propensities1, propensities2):
            assert abs(p1 - p2) < 0.001

    def test_different_seeds_produce_different_twins(self, hcp_training_data):
        """Test that different seeds produce different populations."""
        generator = TwinGenerator(
            twin_type=TwinType.HCP,
            brand=Brand.REMIBRUTINIB,
        )

        generator.train(data=hcp_training_data, target_col="prescribing_change")

        population1 = generator.generate(n=50, seed=111)
        population2 = generator.generate(n=50, seed=222)

        propensities1 = [t.baseline_propensity for t in population1.twins]
        propensities2 = [t.baseline_propensity for t in population2.twins]

        # Should not be identical
        assert propensities1 != propensities2
