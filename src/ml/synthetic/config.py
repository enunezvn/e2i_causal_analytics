"""
E2I Synthetic Data Configuration

Centralized configuration for synthetic data generation with:
- Ground truth causal effects (TRUE_ATE, CATE)
- ML-compliant split boundaries
- Entity volume configurations
- DGP (Data Generating Process) parameters
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional


# =============================================================================
# ENUMS
# =============================================================================


class Brand(str, Enum):
    """Supported pharmaceutical brands."""

    REMIBRUTINIB = "remibrutinib"
    FABHALTA = "fabhalta"
    KISQALI = "kisqali"


class DGPType(str, Enum):
    """Data Generating Process types with known causal effects."""

    SIMPLE_LINEAR = "simple_linear"
    CONFOUNDED = "confounded"
    HETEROGENEOUS = "heterogeneous"
    TIME_SERIES = "time_series"
    SELECTION_BIAS = "selection_bias"


class DataSplit(str, Enum):
    """ML data splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    HOLDOUT = "holdout"


# Convenience exports
BRANDS = [b.value for b in Brand]
DGP_TYPES = [d.value for d in DGPType]


# =============================================================================
# GROUND TRUTH CONFIGURATIONS
# =============================================================================


@dataclass
class DGPConfig:
    """
    Configuration for a Data Generating Process.

    Each DGP has a known TRUE_ATE that the pipeline must recover
    within the specified tolerance.
    """

    dgp_type: DGPType
    true_ate: float
    tolerance: float = 0.05
    confounders: List[str] = field(default_factory=list)
    treatment_variable: str = "engagement_score"
    outcome_variable: str = "treatment_initiated"
    description: str = ""

    # For heterogeneous DGP
    cate_by_segment: Optional[Dict[str, float]] = None

    # For time-series DGP
    lag_periods: int = 1
    include_seasonality: bool = False


# Pre-defined DGP configurations
DGP_CONFIGS: Dict[DGPType, DGPConfig] = {
    DGPType.SIMPLE_LINEAR: DGPConfig(
        dgp_type=DGPType.SIMPLE_LINEAR,
        true_ate=0.40,
        tolerance=0.05,
        confounders=[],
        description="Simple linear effect, no confounding (baseline test)",
    ),
    DGPType.CONFOUNDED: DGPConfig(
        dgp_type=DGPType.CONFOUNDED,
        true_ate=0.25,
        tolerance=0.05,
        confounders=["disease_severity", "academic_hcp"],
        description="Confounded effect requiring adjustment",
    ),
    DGPType.HETEROGENEOUS: DGPConfig(
        dgp_type=DGPType.HETEROGENEOUS,
        true_ate=0.30,  # Average ATE
        tolerance=0.05,
        confounders=["disease_severity", "academic_hcp"],
        cate_by_segment={
            "high_severity": 0.50,
            "medium_severity": 0.30,
            "low_severity": 0.15,
        },
        description="Heterogeneous treatment effects (CATE by segment)",
    ),
    DGPType.TIME_SERIES: DGPConfig(
        dgp_type=DGPType.TIME_SERIES,
        true_ate=0.30,
        tolerance=0.05,
        confounders=["disease_severity"],
        lag_periods=2,
        include_seasonality=True,
        description="Time-series with lag effects and seasonality",
    ),
    DGPType.SELECTION_BIAS: DGPConfig(
        dgp_type=DGPType.SELECTION_BIAS,
        true_ate=0.35,
        tolerance=0.05,
        confounders=["disease_severity", "academic_hcp"],
        description="Selection bias requiring IPW correction",
    ),
}


# =============================================================================
# SPLIT BOUNDARIES
# =============================================================================


@dataclass
class SplitBoundaries:
    """
    Chronological split boundaries for ML-compliant data splitting.

    Splits are patient-level and temporal:
    - Train: 60% (2022-01-01 to 2023-06-30)
    - Validation: 20% (2023-07-01 to 2024-03-31)
    - Test: 15% (2024-04-01 to 2024-09-30)
    - Holdout: 5% (2024-10-01 to 2024-12-31)
    """

    data_start_date: date = date(2022, 1, 1)
    data_end_date: date = date(2024, 12, 31)

    # Split boundaries
    train_end_date: date = date(2023, 6, 30)
    validation_end_date: date = date(2024, 3, 31)
    test_end_date: date = date(2024, 9, 30)
    # Holdout is everything after test_end_date

    # Ratios (for validation)
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    test_ratio: float = 0.15
    holdout_ratio: float = 0.05

    # Gap between splits to prevent temporal leakage
    temporal_gap_days: int = 7

    def get_split_for_date(self, dt: date) -> DataSplit:
        """Determine the split for a given date."""
        if dt <= self.train_end_date:
            return DataSplit.TRAIN
        elif dt <= self.validation_end_date:
            return DataSplit.VALIDATION
        elif dt <= self.test_end_date:
            return DataSplit.TEST
        else:
            return DataSplit.HOLDOUT


# =============================================================================
# ENTITY VOLUME CONFIGURATION
# =============================================================================


@dataclass
class EntityVolumes:
    """
    Target record counts for each entity type.

    These are per-brand defaults, scaled for 3 brands total.
    """

    hcp_profiles_per_brand: int = 5000
    patient_journeys_per_brand: int = 28333  # ~85K / 3
    treatment_events_per_patient: tuple = (10, 40)  # (min, max)
    engagement_events_per_patient: tuple = (2, 10)
    ml_predictions_per_patient: tuple = (1, 5)
    triggers_per_hcp: tuple = (5, 20)
    outcomes_per_patient: tuple = (0, 3)

    @property
    def total_hcps(self) -> int:
        return self.hcp_profiles_per_brand * 3

    @property
    def total_patients(self) -> int:
        return self.patient_journeys_per_brand * 3


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================


@dataclass
class SyntheticDataConfig:
    """
    Master configuration for synthetic data generation.

    Combines DGP configs, split boundaries, and entity volumes.
    """

    # Configuration identity
    config_name: str = "e2i_synthetic_v1"
    config_version: str = "1.0.0"

    # Sub-configurations
    split_boundaries: SplitBoundaries = field(default_factory=SplitBoundaries)
    entity_volumes: EntityVolumes = field(default_factory=EntityVolumes)

    # Processing settings
    batch_size: int = 1000
    random_seed: int = 42

    # Validation thresholds
    min_refutation_pass_rate: float = 0.60
    ate_tolerance: float = 0.05

    # Missingness patterns (for realism)
    missingness_rates: Dict[str, float] = field(
        default_factory=lambda: {
            "insurance_type": 0.08,
            "age_at_diagnosis": 0.05,
            "comorbidity_score": 0.12,
            "prior_therapy_count": 0.15,
            "engagement_quality": 0.10,
        }
    )

    # Measurement error (for realism)
    measurement_error_std: Dict[str, float] = field(
        default_factory=lambda: {
            "disease_severity": 0.10,
            "engagement_score": 0.15,
            "outcome_score": 0.12,
        }
    )

    def get_dgp_config(self, dgp_type: DGPType) -> DGPConfig:
        """Get the DGP configuration for a specific type."""
        return DGP_CONFIGS[dgp_type]

    def get_all_dgp_configs(self) -> Dict[DGPType, DGPConfig]:
        """Get all DGP configurations."""
        return DGP_CONFIGS


# =============================================================================
# SCHEMA ENUMS (Match Supabase exactly)
# =============================================================================


class SpecialtyEnum(str, Enum):
    """HCP specialties matching database ENUM."""

    DERMATOLOGY = "dermatology"
    HEMATOLOGY = "hematology"
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    RHEUMATOLOGY = "rheumatology"
    INTERNAL_MEDICINE = "internal_medicine"
    GENERAL_PRACTICE = "general_practice"
    ALLERGY_IMMUNOLOGY = "allergy_immunology"


class PracticeTypeEnum(str, Enum):
    """Practice types matching database ENUM."""

    ACADEMIC = "academic"
    COMMUNITY = "community"
    PRIVATE = "private"


class RegionEnum(str, Enum):
    """Geographic regions matching database ENUM."""

    NORTHEAST = "northeast"
    SOUTH = "south"
    MIDWEST = "midwest"
    WEST = "west"


class InsuranceTypeEnum(str, Enum):
    """Insurance types matching database ENUM."""

    COMMERCIAL = "commercial"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"


class EngagementTypeEnum(str, Enum):
    """Engagement types matching database ENUM."""

    DETAIL_VISIT = "detail_visit"
    DIGITAL = "digital"
    SPEAKER_PROGRAM = "speaker_program"
    SAMPLE_REQUEST = "sample_request"
    WEBINAR = "webinar"


# =============================================================================
# DEFAULT CONFIG INSTANCE
# =============================================================================

DEFAULT_CONFIG = SyntheticDataConfig()
