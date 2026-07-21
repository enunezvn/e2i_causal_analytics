"""
Tests for SplitValidator.

Tests ML data split validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.validators.split_validator import (
    LeakageInfo,
    SplitValidationResult,
    SplitValidator,
)


class TestSplitValidator:
    """Test suite for SplitValidator."""

    @pytest.fixture
    def validator(self):
        """Create a SplitValidator instance."""
        return SplitValidator()

    @pytest.fixture
    def valid_patient_df(self):
        """
        Create a valid patient DataFrame with proper splits.

        - No entity overlap across splits
        - Temporal ordering preserved
        - Correct ratios
        """
        np.random.seed(42)

        # Create patients with chronological dates and isolated splits
        patients = []

        # Train: 60% (2022-01-01 to 2023-06-30)
        for i in range(600):
            patients.append(
                {
                    "patient_id": f"pt_{i:05d}",
                    "patient_journey_id": f"journey_{i:05d}",
                    "journey_start_date": "2022-06-15",
                    "data_split": "train",
                }
            )

        # Validation: 20% (2023-07-01 to 2024-03-31)
        for i in range(600, 800):
            patients.append(
                {
                    "patient_id": f"pt_{i:05d}",
                    "patient_journey_id": f"journey_{i:05d}",
                    "journey_start_date": "2023-10-15",
                    "data_split": "validation",
                }
            )

        # Test: 10% (#44 holdout enlargement: 15% → 10%)
        for i in range(800, 900):
            patients.append(
                {
                    "patient_id": f"pt_{i:05d}",
                    "patient_journey_id": f"journey_{i:05d}",
                    "journey_start_date": "2024-06-15",
                    "data_split": "test",
                }
            )

        # Holdout: 10% (#44 holdout enlargement: 5% → 10%)
        for i in range(900, 1000):
            patients.append(
                {
                    "patient_id": f"pt_{i:05d}",
                    "patient_journey_id": f"journey_{i:05d}",
                    "journey_start_date": "2024-11-15",
                    "data_split": "holdout",
                }
            )

        return pd.DataFrame(patients)

    @pytest.fixture
    def entity_overlap_df(self):
        """Create a DataFrame with entity overlap (leakage)."""
        return pd.DataFrame(
            {
                "patient_id": ["pt_001", "pt_001", "pt_002", "pt_002"],
                "patient_journey_id": ["j1", "j2", "j3", "j4"],
                "journey_start_date": ["2022-01-01", "2023-07-01", "2022-01-01", "2024-10-01"],
                "data_split": ["train", "validation", "train", "holdout"],  # pt_001 in both!
            }
        )

    @pytest.fixture
    def temporal_overlap_df(self):
        """Create a DataFrame with temporal overlap."""
        return pd.DataFrame(
            {
                "patient_id": ["pt_001", "pt_002", "pt_003"],
                "patient_journey_id": ["j1", "j2", "j3"],
                "journey_start_date": ["2023-08-01", "2023-06-01", "2024-01-01"],
                "data_split": ["train", "validation", "test"],  # Train date after validation!
            }
        )

    def test_validate_valid_splits(self, validator, valid_patient_df):
        """Test validation of properly split data."""
        result = validator.validate(
            df=valid_patient_df,
            entity_column="patient_id",
            date_column="journey_start_date",
            split_column="data_split",
        )

        assert result.is_valid is True
        assert result.total_records == 1000
        assert not result.has_critical_leakage()

    def test_validate_split_ratios(self, validator, valid_patient_df):
        """Test that split ratios are calculated correctly."""
        result = validator.validate(
            df=valid_patient_df,
            entity_column="patient_id",
            split_column="data_split",
        )

        assert result.split_counts["train"] == 600
        assert result.split_counts["validation"] == 200
        assert result.split_counts["test"] == 100
        assert result.split_counts["holdout"] == 100

        assert abs(result.split_ratios["train"] - 0.60) < 0.01
        assert abs(result.split_ratios["validation"] - 0.20) < 0.01
        assert abs(result.split_ratios["test"] - 0.10) < 0.01
        assert abs(result.split_ratios["holdout"] - 0.10) < 0.01
        # A design-conforming frame carries no ratio warnings.
        assert result.ratio_errors["test"] < 0.01
        assert result.ratio_errors["holdout"] < 0.01

    def test_legacy_15_5_frame_fails_ratio_validation(self, validator):
        """Red-first #44 pin: a frame seeded with the PRE-enlargement 60/20/15/5
        quota must now trip ratio warnings for test AND holdout — the 5pp gap to
        the new 10/10 design exceeds the tightened tolerance. Guards against the
        validator silently blessing an old-quota substrate (the previous 0.05
        tolerance made 15%-vs-10% pass borderline)."""
        rows = []
        pid = 0
        for split, n in [("train", 600), ("validation", 200), ("test", 150), ("holdout", 50)]:
            for _ in range(n):
                rows.append({"patient_id": f"pt_{pid:05d}", "data_split": split})
                pid += 1
        df = pd.DataFrame(rows)

        result = validator.validate(df=df, entity_column="patient_id", split_column="data_split")

        assert result.ratio_errors["test"] == pytest.approx(0.05, abs=0.001)
        assert result.ratio_errors["holdout"] == pytest.approx(0.05, abs=0.001)
        ratio_warnings = [w for w in result.warnings if "differs from expected" in w]
        assert any("'test'" in w for w in ratio_warnings)
        assert any("'holdout'" in w for w in ratio_warnings)

    def test_detect_entity_overlap(self, validator, entity_overlap_df):
        """Test detection of entity overlap across splits."""
        result = validator.validate(
            df=entity_overlap_df,
            entity_column="patient_id",
            split_column="data_split",
        )

        assert result.is_valid is False
        assert result.has_critical_leakage()

        entity_leakages = [l for l in result.leakages if l.leakage_type == "entity_overlap"]
        assert len(entity_leakages) > 0

    def test_detect_temporal_overlap(self, validator, temporal_overlap_df):
        """Test detection of temporal overlap between splits."""
        result = validator.validate(
            df=temporal_overlap_df,
            entity_column="patient_id",
            date_column="journey_start_date",
            split_column="data_split",
        )

        temporal_leakages = [l for l in result.leakages if l.leakage_type == "temporal_overlap"]
        # Should detect temporal issues
        assert len(temporal_leakages) > 0 or len(result.warnings) > 0

    def test_validate_missing_split_column(self, validator, valid_patient_df):
        """Test validation with missing split column."""
        df = valid_patient_df.drop(columns=["data_split"])

        result = validator.validate(
            df=df,
            entity_column="patient_id",
            split_column="data_split",
        )

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_multiple_datasets(self, validator, valid_patient_df):
        """Test validation of multiple datasets."""
        # Create a related engagement events DataFrame
        engagement_df = pd.DataFrame(
            {
                "engagement_id": [f"e_{i}" for i in range(100)],
                "patient_id": [f"pt_{i % 1000:05d}" for i in range(100)],
                "data_split": ["train"] * 60
                + ["validation"] * 20
                + ["test"] * 10
                + ["holdout"] * 10,
            }
        )

        datasets = {
            "patient_journeys": valid_patient_df,
            "engagement_events": engagement_df,
        }

        results = validator.validate_multiple_datasets(
            datasets=datasets,
            entity_column="patient_id",
            split_column="data_split",
        )

        assert len(results) == 2
        assert results["patient_journeys"].is_valid is True

    def test_get_validation_summary(self, validator, valid_patient_df):
        """Test validation summary generation."""
        results = validator.validate_multiple_datasets(
            datasets={"patient_journeys": valid_patient_df},
            entity_column="patient_id",
            split_column="data_split",
        )

        summary = validator.get_validation_summary(results)

        assert summary["all_valid"] is True
        assert summary["total_tables"] == 1
        assert summary["total_leakages"] == 0

    def test_temporal_overlap_invalidates_split(self, validator, temporal_overlap_df):
        """M-fo5a: a detected temporal overlap is a leakage that must fail the
        split. Previously it was recorded with severity='warning', which never
        flipped is_valid, so chronologically-leaking splits passed validation.
        """
        result = validator.validate(
            df=temporal_overlap_df,
            entity_column="patient_id",
            date_column="journey_start_date",
            split_column="data_split",
        )

        temporal_leakages = [l for l in result.leakages if l.leakage_type == "temporal_overlap"]
        assert len(temporal_leakages) > 0
        assert all(l.severity == "critical" for l in temporal_leakages)
        assert result.temporal_violations >= 1
        assert result.is_valid is False
        assert result.has_critical_leakage() is True

    def test_leakage_detector_exception_fails_closed(self, valid_patient_df):
        """M-fo5b: if the wired LeakageDetector raises, the validator must NOT
        silently pass. The failure is recorded in errors and is_valid=False
        (fail-closed), instead of being swallowed into a warning.
        """

        class _ExplodingDetector:
            def detect_leakage(self, *args, **kwargs):
                raise RuntimeError("detector boom")

        validator = SplitValidator()
        validator._leakage_detector = _ExplodingDetector()

        result = validator.validate(
            df=valid_patient_df,
            entity_column="patient_id",
            date_column="journey_start_date",
            split_column="data_split",
        )

        assert result.is_valid is False
        assert any("LeakageDetector" in e for e in result.errors)
        assert any("detector boom" in e for e in result.errors)
        # The failure must NOT be downgraded to a mere warning.
        assert not any("LeakageDetector check failed" in w for w in result.warnings)

    def test_validator_cluster_modules_marked_test_only(self):
        """M-reach2: the synthetic-validator cluster has no production consumers
        and is documented as test-only-by-design. Guard the docstring annotation
        so it is not silently dropped.
        """
        import src.ml.synthetic.validators as validators_pkg
        from src.ml.synthetic.validators import (
            causal_validator,
            schema_validator,
            split_validator,
        )

        for module in (
            validators_pkg,
            causal_validator,
            schema_validator,
            split_validator,
        ):
            assert module.__doc__ is not None
            assert "Test-only-by-design" in module.__doc__


class TestSplitValidationResult:
    """Test suite for SplitValidationResult."""

    def test_add_leakage_critical(self):
        """Test that critical leakage sets is_valid to False."""
        result = SplitValidationResult(is_valid=True, total_records=100)

        result.add_leakage(
            LeakageInfo(
                leakage_type="entity_overlap",
                severity="critical",
                description="Test leakage",
            )
        )

        assert result.is_valid is False
        assert result.has_critical_leakage() is True

    def test_add_leakage_warning(self):
        """Test that warning leakage doesn't affect is_valid."""
        result = SplitValidationResult(is_valid=True, total_records=100)

        result.add_leakage(
            LeakageInfo(
                leakage_type="temporal_overlap",
                severity="warning",
                description="Test warning",
            )
        )

        assert result.is_valid is True
        assert result.has_critical_leakage() is False
        assert len(result.leakages) == 1
