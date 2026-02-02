"""Tests for CohortConstructor core logic."""

import pandas as pd
import pytest

from src.agents.cohort_constructor import (
    CohortConfig,
    CohortConstructor,
    CohortExecutionResult,
    Criterion,
    CriterionType,
    Operator,
    TemporalRequirements,
)
from src.agents.cohort_constructor.constructor import (
    compare_cohorts,
    create_cohort_quick,
)


class TestCohortConstructorInit:
    """Tests for CohortConstructor initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with a valid configuration."""
        constructor = CohortConstructor(sample_config)

        assert constructor.config == sample_config
        assert constructor.eligibility_log == []
        assert constructor.patient_assignments == []
        assert constructor.validation_errors == []

    def test_init_sets_latency_counters(self, sample_config):
        """Test that latency counters are initialized to zero."""
        constructor = CohortConstructor(sample_config)

        assert constructor._validate_config_ms == 0
        assert constructor._apply_criteria_ms == 0
        assert constructor._validate_temporal_ms == 0
        assert constructor._generate_metadata_ms == 0


class TestCohortConstructorValidation:
    """Tests for input validation."""

    def test_validate_required_fields_success(self, sample_config, sample_patient_df):
        """Test validation passes when required fields are present."""
        constructor = CohortConstructor(sample_config)

        # Should not raise
        constructor._validate_required_fields(sample_patient_df)
        assert len(constructor.validation_errors) == 0

    def test_validate_required_fields_missing(self, sample_config):
        """Test validation fails when required fields are missing."""
        df = pd.DataFrame({"patient_journey_id": ["P001"]})
        constructor = CohortConstructor(sample_config)

        with pytest.raises(ValueError, match="Missing required fields"):
            constructor._validate_required_fields(df)

        assert len(constructor.validation_errors) == 1
        assert constructor.validation_errors[0]["type"] == "missing_fields"


class TestCohortConstructorCriteriaApplication:
    """Tests for criteria application logic."""

    def test_apply_inclusion_criteria(self, sample_config, sample_patient_df):
        """Test applying inclusion criteria filters patients correctly."""
        constructor = CohortConstructor(sample_config)

        # Apply inclusion criteria (age >= 18)
        eligible = constructor._apply_inclusion_criteria(sample_patient_df)

        # All patients are >= 18 in sample data
        assert len(eligible) == len(sample_patient_df)

    def test_apply_inclusion_filters_out_patients(self, sample_patient_df):
        """Test inclusion criteria correctly filters patients."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[
                Criterion(
                    field="age_at_diagnosis",
                    operator=Operator.GREATER,
                    value=40,  # Only keeps ages > 40
                    criterion_type=CriterionType.INCLUSION,
                    description="Adults over 40",
                )
            ],
            exclusion_criteria=[],
            required_fields=["patient_journey_id", "age_at_diagnosis"],
        )

        constructor = CohortConstructor(config)
        eligible = constructor._apply_inclusion_criteria(sample_patient_df)

        # Only patients with age > 40 should remain
        assert all(eligible["age_at_diagnosis"] > 40)
        assert len(constructor.eligibility_log) == 1

    def test_apply_exclusion_criteria(self, sample_patient_df):
        """Test applying exclusion criteria removes correct patients."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[
                Criterion(
                    field="gender",
                    operator=Operator.EQUAL,
                    value="M",  # Exclude males
                    criterion_type=CriterionType.EXCLUSION,
                    description="Exclude males",
                )
            ],
            required_fields=["patient_journey_id"],
        )

        constructor = CohortConstructor(config)
        eligible = constructor._apply_exclusion_criteria(sample_patient_df)

        # No males should remain
        assert all(eligible["gender"] != "M")


class TestCohortConstructorOperators:
    """Tests for individual operators."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame."""
        return pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5],
                "text": ["a", "ab", "abc", "b", "c"],
                "category": ["X", "Y", "Z", "X", "Y"],
            }
        )

    def test_operator_equal(self, basic_df):
        """Test EQUAL operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[
                Criterion(
                    field="value",
                    operator=Operator.EQUAL,
                    value=3,
                    criterion_type=CriterionType.INCLUSION,
                    description="Equal to 3",
                )
            ],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.EQUAL, 3)

        assert mask.tolist() == [False, False, True, False, False]

    def test_operator_not_equal(self, basic_df):
        """Test NOT_EQUAL operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.NOT_EQUAL, 3)

        assert mask.tolist() == [True, True, False, True, True]

    def test_operator_greater(self, basic_df):
        """Test GREATER operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.GREATER, 3)

        assert mask.tolist() == [False, False, False, True, True]

    def test_operator_greater_equal(self, basic_df):
        """Test GREATER_EQUAL operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.GREATER_EQUAL, 3)

        assert mask.tolist() == [False, False, True, True, True]

    def test_operator_less(self, basic_df):
        """Test LESS operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.LESS, 3)

        assert mask.tolist() == [True, True, False, False, False]

    def test_operator_less_equal(self, basic_df):
        """Test LESS_EQUAL operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.LESS_EQUAL, 3)

        assert mask.tolist() == [True, True, True, False, False]

    def test_operator_in(self, basic_df):
        """Test IN operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["category"], Operator.IN, ["X", "Y"])

        assert mask.tolist() == [True, True, False, True, True]

    def test_operator_not_in(self, basic_df):
        """Test NOT_IN operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["category"], Operator.NOT_IN, ["X", "Y"])

        assert mask.tolist() == [False, False, True, False, False]

    def test_operator_between(self, basic_df):
        """Test BETWEEN operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["value"], Operator.BETWEEN, (2, 4))

        assert mask.tolist() == [False, True, True, True, False]

    def test_operator_contains(self, basic_df):
        """Test CONTAINS operator."""
        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            required_fields=[],
        )

        constructor = CohortConstructor(config)
        mask = constructor._compute_mask(basic_df["text"], Operator.CONTAINS, "b")

        assert mask.tolist() == [False, True, True, True, False]


class TestCohortConstructorExecution:
    """Tests for full cohort construction execution."""

    def test_construct_cohort_success(self, sample_config, sample_patient_df):
        """Test successful cohort construction."""
        constructor = CohortConstructor(sample_config)
        eligible_df, result = constructor.construct_cohort(sample_patient_df)

        assert isinstance(result, CohortExecutionResult)
        assert result.status == "success"
        assert result.error_message is None
        assert len(eligible_df) > 0

    def test_construct_cohort_returns_execution_id(self, sample_config, sample_patient_df):
        """Test that execution returns a unique execution ID."""
        constructor = CohortConstructor(sample_config)
        _, result = constructor.construct_cohort(sample_patient_df)

        assert result.execution_id.startswith("exec_")
        assert len(result.execution_id) > 10

    def test_construct_cohort_tracks_latencies(self, sample_config, sample_patient_df):
        """Test that execution tracks node latencies."""
        constructor = CohortConstructor(sample_config)
        _, result = constructor.construct_cohort(sample_patient_df)

        metadata = result.execution_metadata
        assert "validate_config_ms" in metadata
        assert "apply_criteria_ms" in metadata
        assert "validate_temporal_ms" in metadata
        assert "generate_metadata_ms" in metadata
        assert metadata["execution_time_ms"] >= 0

    def test_construct_cohort_missing_fields_error(self, sample_config):
        """Test error handling for missing required fields."""
        df = pd.DataFrame({"patient_journey_id": ["P001"]})
        constructor = CohortConstructor(sample_config)

        eligible_df, result = constructor.construct_cohort(df)

        assert result.status == "failed"
        assert result.error_code == "CC_002"
        assert "Missing required fields" in result.error_message
        assert len(eligible_df) == 0

    def test_construct_cohort_empty_input(self, sample_config, empty_patient_df):
        """Test handling of empty input DataFrame."""
        constructor = CohortConstructor(sample_config)
        eligible_df, result = constructor.construct_cohort(empty_patient_df)

        # Should handle empty input without crashing
        assert len(eligible_df) == 0
        assert result.eligibility_stats["total_input_patients"] == 0

    def test_construct_cohort_eligibility_log(self, sample_config, sample_patient_df):
        """Test that eligibility log is populated."""
        constructor = CohortConstructor(sample_config)
        _, result = constructor.construct_cohort(sample_patient_df)

        assert len(result.eligibility_log) >= len(sample_config.inclusion_criteria)

    def test_construct_cohort_patient_assignments(self, sample_config, sample_patient_df):
        """Test that patient assignments are tracked when requested."""
        constructor = CohortConstructor(sample_config)
        _, result = constructor.construct_cohort(sample_patient_df, track_assignments=True)

        assert len(result.patient_assignments) == len(sample_patient_df)

    def test_construct_cohort_no_patient_assignments_when_disabled(
        self, sample_config, sample_patient_df
    ):
        """Test that patient assignments are not tracked when disabled."""
        constructor = CohortConstructor(sample_config)
        _, result = constructor.construct_cohort(sample_patient_df, track_assignments=False)

        assert len(result.patient_assignments) == 0


class TestCohortConstructorTemporalValidation:
    """Tests for temporal validation."""

    def test_temporal_validation_with_lookback(self):
        """Test temporal validation with lookback period."""
        df = pd.DataFrame(
            {
                "patient_journey_id": ["P001", "P002", "P003"],
                "age_at_diagnosis": [30, 40, 50],
                "diagnosis_date": ["2023-06-01", "2023-06-01", "2023-06-01"],
                "journey_start_date": [
                    "2023-01-01",
                    "2023-05-01",
                    "2023-05-20",
                ],  # P003 has only 12 days lookback
                "follow_up_days": [120, 120, 120],
            }
        )

        config = CohortConfig(
            cohort_name="Test",
            brand="test",
            indication="test",
            inclusion_criteria=[],
            exclusion_criteria=[],
            temporal_requirements=TemporalRequirements(
                lookback_days=30,
                followup_days=90,
                index_date_field="diagnosis_date",
            ),
            required_fields=["patient_journey_id", "age_at_diagnosis"],
        )

        constructor = CohortConstructor(config)
        eligible = constructor._validate_temporal_eligibility(df)

        # P003 should be excluded due to insufficient lookback
        assert "P001" in eligible["patient_journey_id"].values
        assert "P002" in eligible["patient_journey_id"].values
        assert "P003" not in eligible["patient_journey_id"].values


class TestCohortConstructorUtilities:
    """Tests for utility methods."""

    def test_is_eligible(self, sample_config, sample_patient_df):
        """Test single patient eligibility check."""
        constructor = CohortConstructor(sample_config)

        # Test with an eligible patient
        patient = sample_patient_df.iloc[0]
        result = constructor.is_eligible(patient)

        assert isinstance(result, bool)

    def test_summary_report(self, sample_config, sample_patient_df):
        """Test summary report generation."""
        constructor = CohortConstructor(sample_config)
        _, result = constructor.construct_cohort(sample_patient_df)

        report = constructor.summary_report(result)

        assert "COHORT CONSTRUCTION SUMMARY" in report
        assert sample_config.cohort_name in report
        assert sample_config.brand.upper() in report

    def test_reset_state(self, sample_config, sample_patient_df):
        """Test state reset between constructions."""
        constructor = CohortConstructor(sample_config)

        # First construction
        constructor.construct_cohort(sample_patient_df)
        assert len(constructor.eligibility_log) > 0

        # Reset
        constructor._reset_state()

        assert constructor.eligibility_log == []
        assert constructor.patient_assignments == []
        assert constructor._criterion_order == 0


class TestCohortConstructorModuleFunctions:
    """Tests for module-level utility functions."""

    def test_create_cohort_quick(self, remibrutinib_patient_df):
        """Test quick cohort creation function."""
        eligible_df, result = create_cohort_quick(
            brand="remibrutinib",
            indication="csu",
            df=remibrutinib_patient_df,
        )

        assert isinstance(result, CohortExecutionResult)
        assert result.status == "success"

    def test_compare_cohorts(self, sample_patient_df):
        """Test cohort comparison function."""
        configs = [
            CohortConfig(
                cohort_name="Strict Cohort",
                brand="test",
                indication="test",
                inclusion_criteria=[
                    Criterion(
                        field="age_at_diagnosis",
                        operator=Operator.GREATER,
                        value=50,
                        criterion_type=CriterionType.INCLUSION,
                        description="Age > 50",
                    )
                ],
                exclusion_criteria=[],
                required_fields=["patient_journey_id", "age_at_diagnosis"],
            ),
            CohortConfig(
                cohort_name="Relaxed Cohort",
                brand="test",
                indication="test",
                inclusion_criteria=[
                    Criterion(
                        field="age_at_diagnosis",
                        operator=Operator.GREATER_EQUAL,
                        value=18,
                        criterion_type=CriterionType.INCLUSION,
                        description="Age >= 18",
                    )
                ],
                exclusion_criteria=[],
                required_fields=["patient_journey_id", "age_at_diagnosis"],
            ),
        ]

        comparison = compare_cohorts(sample_patient_df, configs)

        assert len(comparison) == 2
        assert "cohort_name" in comparison.columns
        assert "eligible_population" in comparison.columns
        assert (
            comparison.iloc[1]["eligible_population"] >= comparison.iloc[0]["eligible_population"]
        )
