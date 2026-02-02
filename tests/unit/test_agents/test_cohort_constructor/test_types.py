"""Tests for CohortConstructor types and data structures."""

from src.agents.cohort_constructor import (
    CohortConfig,
    CohortExecutionResult,
    Criterion,
    CriterionType,
    EligibilityLogEntry,
    Operator,
    PatientAssignment,
    TemporalRequirements,
)


class TestOperator:
    """Tests for Operator enum."""

    def test_all_operators_defined(self):
        """Test that all expected operators are defined."""
        expected = [
            "EQUAL",
            "NOT_EQUAL",
            "GREATER",
            "GREATER_EQUAL",
            "LESS",
            "LESS_EQUAL",
            "IN",
            "NOT_IN",
            "BETWEEN",
            "CONTAINS",
        ]
        for op in expected:
            assert hasattr(Operator, op), f"Missing operator: {op}"

    def test_operator_values(self):
        """Test operator string values."""
        assert Operator.EQUAL.value == "=="
        assert Operator.NOT_EQUAL.value == "!="
        assert Operator.GREATER.value == ">"
        assert Operator.GREATER_EQUAL.value == ">="
        assert Operator.LESS.value == "<"
        assert Operator.LESS_EQUAL.value == "<="
        assert Operator.IN.value == "in"
        assert Operator.NOT_IN.value == "not_in"
        assert Operator.BETWEEN.value == "between"
        assert Operator.CONTAINS.value == "contains"


class TestCriterionType:
    """Tests for CriterionType enum."""

    def test_criterion_types(self):
        """Test criterion type values."""
        assert CriterionType.INCLUSION.value == "inclusion"
        assert CriterionType.EXCLUSION.value == "exclusion"


class TestCriterion:
    """Tests for Criterion dataclass."""

    def test_create_criterion(self):
        """Test creating a criterion."""
        criterion = Criterion(
            field="age",
            operator=Operator.GREATER_EQUAL,
            value=18,
            criterion_type=CriterionType.INCLUSION,
            description="Adult patients",
        )

        assert criterion.field == "age"
        assert criterion.operator == Operator.GREATER_EQUAL
        assert criterion.value == 18
        assert criterion.criterion_type == CriterionType.INCLUSION
        assert criterion.description == "Adult patients"

    def test_criterion_to_dict(self):
        """Test converting criterion to dictionary."""
        criterion = Criterion(
            field="age",
            operator=Operator.GREATER_EQUAL,
            value=18,
            criterion_type=CriterionType.INCLUSION,
            description="Adult patients",
        )

        d = criterion.to_dict()
        assert d["field"] == "age"
        assert d["operator"] == ">="
        assert d["value"] == 18
        assert d["criterion_type"] == "inclusion"
        assert d["description"] == "Adult patients"

    def test_criterion_with_list_value(self):
        """Test criterion with list value (for IN operator)."""
        criterion = Criterion(
            field="diagnosis_code",
            operator=Operator.IN,
            value=["L50.1", "L50.8", "L50.9"],
            criterion_type=CriterionType.INCLUSION,
            description="CSU diagnosis codes",
        )

        assert criterion.operator == Operator.IN
        assert isinstance(criterion.value, list)
        assert len(criterion.value) == 3

    def test_criterion_with_tuple_value(self):
        """Test criterion with tuple value (for BETWEEN operator)."""
        criterion = Criterion(
            field="age",
            operator=Operator.BETWEEN,
            value=(18, 65),
            criterion_type=CriterionType.INCLUSION,
            description="Adult working age",
        )

        assert criterion.operator == Operator.BETWEEN
        assert isinstance(criterion.value, tuple)
        assert criterion.value == (18, 65)


class TestTemporalRequirements:
    """Tests for TemporalRequirements dataclass."""

    def test_default_values(self):
        """Test default temporal requirement values."""
        temporal = TemporalRequirements()
        assert temporal.lookback_days == 180
        assert temporal.followup_days == 90
        assert temporal.index_date_field == "diagnosis_date"

    def test_custom_values(self):
        """Test custom temporal requirement values."""
        temporal = TemporalRequirements(
            lookback_days=365,
            followup_days=180,
            index_date_field="treatment_start_date",
        )
        assert temporal.lookback_days == 365
        assert temporal.followup_days == 180
        assert temporal.index_date_field == "treatment_start_date"

    def test_attributes_accessible(self):
        """Test temporal requirements attributes are accessible."""
        temporal = TemporalRequirements(lookback_days=365, followup_days=180)

        # TemporalRequirements is a simple dataclass without to_dict
        # Verify attributes are accessible
        assert temporal.lookback_days == 365
        assert temporal.followup_days == 180
        assert temporal.index_date_field == "diagnosis_date"


class TestCohortConfig:
    """Tests for CohortConfig dataclass."""

    def test_create_config(self, sample_criterion):
        """Test creating a cohort configuration."""
        config = CohortConfig(
            cohort_name="Test Cohort",
            brand="test",
            indication="test_indication",
            inclusion_criteria=[sample_criterion],
            exclusion_criteria=[],
        )

        assert config.cohort_name == "Test Cohort"
        assert config.brand == "test"
        assert config.indication == "test_indication"
        assert len(config.inclusion_criteria) == 1
        assert len(config.exclusion_criteria) == 0

    def test_config_with_temporal_requirements(self, sample_criterion):
        """Test config with custom temporal requirements."""
        temporal = TemporalRequirements(lookback_days=365)
        config = CohortConfig(
            cohort_name="Test Cohort",
            brand="test",
            indication="test",
            inclusion_criteria=[sample_criterion],
            exclusion_criteria=[],
            temporal_requirements=temporal,
        )

        assert config.temporal_requirements.lookback_days == 365

    def test_config_to_dict(self, sample_criterion):
        """Test converting config to dictionary."""
        config = CohortConfig(
            cohort_name="Test Cohort",
            brand="test",
            indication="test",
            inclusion_criteria=[sample_criterion],
            exclusion_criteria=[],
            version="1.0.0",
        )

        d = config.to_dict()
        assert d["cohort_name"] == "Test Cohort"
        assert d["brand"] == "test"
        assert d["indication"] == "test"
        assert len(d["inclusion_criteria"]) == 1
        assert d["version"] == "1.0.0"


class TestEligibilityLogEntry:
    """Tests for EligibilityLogEntry dataclass."""

    def test_create_log_entry(self):
        """Test creating an eligibility log entry."""
        entry = EligibilityLogEntry(
            criterion_name="age",
            criterion_type="inclusion",
            criterion_order=1,
            operator=">=",
            value=18,
            removed_count=5,
            remaining_count=95,
            description="Adult patients",
        )

        assert entry.criterion_name == "age"
        assert entry.removed_count == 5
        assert entry.remaining_count == 95

    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary."""
        entry = EligibilityLogEntry(
            criterion_name="age",
            criterion_type="inclusion",
            criterion_order=1,
            operator=">=",
            value=18,
            removed_count=5,
            remaining_count=95,
        )

        d = entry.to_dict()
        assert d["criterion_name"] == "age"
        assert d["removed_count"] == 5
        assert d["remaining_count"] == 95


class TestPatientAssignment:
    """Tests for PatientAssignment dataclass."""

    def test_create_eligible_assignment(self):
        """Test creating an eligible patient assignment."""
        assignment = PatientAssignment(
            patient_journey_id="P001",
            is_eligible=True,
        )

        assert assignment.patient_journey_id == "P001"
        assert assignment.is_eligible is True
        assert assignment.failed_criteria == []

    def test_create_ineligible_assignment(self):
        """Test creating an ineligible patient assignment."""
        assignment = PatientAssignment(
            patient_journey_id="P001",
            is_eligible=False,
            failed_criteria=["age < 18", "missing diagnosis"],
            lookback_complete=False,
        )

        assert assignment.is_eligible is False
        assert len(assignment.failed_criteria) == 2
        assert assignment.lookback_complete is False


class TestCohortExecutionResult:
    """Tests for CohortExecutionResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful execution result."""
        result = CohortExecutionResult(
            cohort_id="cohort_001",
            execution_id="exec_001",
            eligible_patient_ids=["P001", "P002", "P003"],
            eligibility_stats={"initial_population": 100, "final_eligible": 50},
            eligibility_log=[],
            patient_assignments=[],
            execution_metadata={"execution_time_ms": 150},
            status="success",
        )

        assert result.cohort_id == "cohort_001"
        assert len(result.eligible_patient_ids) == 3
        assert result.status == "success"
        assert result.error_message is None

    def test_create_failed_result(self):
        """Test creating a failed execution result."""
        result = CohortExecutionResult(
            cohort_id="cohort_001",
            execution_id="exec_001",
            eligible_patient_ids=[],
            eligibility_stats={},
            eligibility_log=[],
            patient_assignments=[],
            execution_metadata={},
            status="failed",
            error_message="Empty cohort: all patients excluded",
            error_code="CC_003",
        )

        assert result.status == "failed"
        assert result.error_code == "CC_003"
        assert "Empty cohort" in result.error_message

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = CohortExecutionResult(
            cohort_id="cohort_001",
            execution_id="exec_001",
            eligible_patient_ids=["P001"],
            eligibility_stats={"final_eligible": 1},
            eligibility_log=[],
            patient_assignments=[],
            execution_metadata={},
            status="success",
        )

        d = result.to_dict()
        assert d["cohort_id"] == "cohort_001"
        assert d["status"] == "success"
        assert len(d["eligible_patient_ids"]) == 1
