"""Fixtures for CohortConstructor agent tests."""

from datetime import datetime

import pandas as pd
import pytest


@pytest.fixture
def sample_patient_df():
    """Create a sample patient DataFrame for testing."""
    return pd.DataFrame(
        {
            "patient_journey_id": ["P001", "P002", "P003", "P004", "P005"],
            "age_at_diagnosis": [25, 45, 62, 18, 55],
            "gender": ["F", "M", "F", "M", "F"],
            "diagnosis_code": ["L50.1", "L50.1", "L50.1", "L50.1", "L50.1"],
            "diagnosis_date": ["2023-01-15"] * 5,
            "first_observation_date": ["2020-01-01"] * 5,
            "last_observation_date": ["2024-12-01"] * 5,
        }
    )


@pytest.fixture
def remibrutinib_patient_df():
    """Create patient data for Remibrutinib CSU testing."""
    return pd.DataFrame(
        {
            "patient_journey_id": ["P001", "P002", "P003", "P004", "P005", "P006"],
            "age_at_diagnosis": [25, 45, 62, 18, 15, 55],  # P005 is under 18
            "gender": ["F", "M", "F", "M", "F", "M"],
            "diagnosis_code": ["L50.1", "L50.1", "L50.1", "L50.1", "L50.1", "L50.1"],
            "diagnosis_date": ["2023-01-15"] * 6,
            "urticaria_severity_uas7": [18, 12, 25, 20, 22, 8],  # P002 and P006 have UAS7 < 16
            "antihistamine_refractory": [
                True,
                False,
                True,
                True,
                True,
                False,
            ],  # P002 and P006 not refractory
            "prior_antihistamine_therapy": [True, True, True, True, True, True],
            "first_observation_date": ["2020-01-01"] * 6,
            "last_observation_date": ["2024-12-01"] * 6,
        }
    )


@pytest.fixture
def fabhalta_patient_df():
    """Create patient data for Fabhalta PNH testing."""
    return pd.DataFrame(
        {
            "patient_journey_id": ["P001", "P002", "P003", "P004", "P005"],
            "age_at_diagnosis": [35, 45, 28, 55, 42],
            "gender": ["F", "M", "F", "M", "F"],
            "diagnosis_code": ["D59.5", "D59.5", "D59.5", "D59.5", "D59.5"],  # PNH ICD-10 code
            "diagnosis_date": ["2023-01-15"] * 5,
            "complement_inhibitor_status": [
                "naive",
                "experienced",
                "naive",
                "experienced",
                "naive",
            ],
            "ldh_level": [2.5, 1.2, 3.0, 0.8, 2.2],  # P002 and P004 have LDH < 1.5x ULN
            "pnh_clone_size": [15, 5, 20, 8, 12],  # P002 has <10% clone
            "first_observation_date": ["2020-01-01"] * 5,
            "last_observation_date": ["2024-12-01"] * 5,
        }
    )


@pytest.fixture
def kisqali_patient_df():
    """Create patient data for Kisqali HR+/HER2- BC testing."""
    return pd.DataFrame(
        {
            "patient_journey_id": ["P001", "P002", "P003", "P004", "P005"],
            "age_at_diagnosis": [55, 62, 48, 70, 45],
            "gender": ["F", "F", "F", "F", "F"],
            "diagnosis_code": ["C50.9", "C50.9", "C50.9", "C50.9", "C50.9"],
            "diagnosis_date": ["2023-01-15"] * 5,
            "hr_status": [
                "positive",
                "positive",
                "negative",
                "positive",
                "positive",
            ],  # P003 is HR-
            "her2_status": [
                "negative",
                "positive",
                "negative",
                "negative",
                "negative",
            ],  # P002 is HER2+
            "ecog_performance_status": [0, 1, 0, 3, 1],  # P004 has ECOG > 2
            "disease_stage": ["metastatic", "advanced", "metastatic", "advanced", "metastatic"],
            "first_observation_date": ["2020-01-01"] * 5,
            "last_observation_date": ["2024-12-01"] * 5,
        }
    )


@pytest.fixture
def empty_patient_df():
    """Create an empty patient DataFrame."""
    return pd.DataFrame(
        columns=[
            "patient_journey_id",
            "age_at_diagnosis",
            "gender",
            "diagnosis_code",
            "diagnosis_date",
            "first_observation_date",
            "last_observation_date",
        ]
    )


@pytest.fixture
def large_patient_df():
    """Create a larger patient DataFrame for performance testing."""
    n_patients = 1000
    base_date = datetime(2023, 1, 15)

    return pd.DataFrame(
        {
            "patient_journey_id": [f"P{i:05d}" for i in range(n_patients)],
            "age_at_diagnosis": [20 + (i % 60) for i in range(n_patients)],
            "gender": ["F" if i % 2 == 0 else "M" for i in range(n_patients)],
            "diagnosis_code": ["L50.1"] * n_patients,
            "diagnosis_date": [base_date.strftime("%Y-%m-%d")] * n_patients,
            "urticaria_severity_uas7": [10 + (i % 25) for i in range(n_patients)],
            "antihistamine_refractory": [i % 3 != 0 for i in range(n_patients)],
            "prior_antihistamine_therapy": [True] * n_patients,
            "first_observation_date": ["2020-01-01"] * n_patients,
            "last_observation_date": ["2024-12-01"] * n_patients,
        }
    )


@pytest.fixture
def sample_criterion():
    """Create a sample criterion for testing."""
    from src.agents.cohort_constructor import Criterion, CriterionType, Operator

    return Criterion(
        field="age_at_diagnosis",
        operator=Operator.GREATER_EQUAL,
        value=18,
        criterion_type=CriterionType.INCLUSION,
        description="Adult patients only",
    )


@pytest.fixture
def sample_config():
    """Create a sample cohort configuration for testing."""
    from src.agents.cohort_constructor import (
        CohortConfig,
        Criterion,
        CriterionType,
        Operator,
        TemporalRequirements,
    )

    return CohortConfig(
        cohort_name="Test Cohort",
        brand="test",
        indication="test_indication",
        inclusion_criteria=[
            Criterion(
                field="age_at_diagnosis",
                operator=Operator.GREATER_EQUAL,
                value=18,
                criterion_type=CriterionType.INCLUSION,
                description="Adult patients",
            ),
        ],
        exclusion_criteria=[],
        temporal_requirements=TemporalRequirements(
            lookback_days=180,
            followup_days=90,
        ),
        required_fields=["patient_journey_id", "age_at_diagnosis"],
    )


@pytest.fixture
def mock_mlflow_logger(mocker):
    """Mock CohortMLflowLogger for testing."""
    mock = mocker.MagicMock()
    mock.log_cohort_execution.return_value = "test-run-id"
    mock.log_sla_compliance.return_value = None
    return mock


@pytest.fixture
def mock_opik_tracer(mocker):
    """Mock CohortOpikTracer for testing."""
    mock = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock.trace_cohort_construction.return_value.__enter__ = mocker.MagicMock(
        return_value=mock_context
    )
    mock.trace_cohort_construction.return_value.__exit__ = mocker.MagicMock(return_value=None)
    return mock
