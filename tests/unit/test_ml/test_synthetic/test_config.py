"""
Unit tests for src/ml/synthetic/config.py

Tests configuration dataclasses and enums for synthetic data generation.
"""

from datetime import date

import pytest

from src.ml.synthetic.config import (
    BRANDS,
    DEFAULT_CONFIG,
    DGP_CONFIGS,
    DGP_TYPES,
    Brand,
    DataSplit,
    DGPConfig,
    DGPType,
    EngagementTypeEnum,
    EntityVolumes,
    InsuranceTypeEnum,
    PracticeTypeEnum,
    RegionEnum,
    SpecialtyEnum,
    SplitBoundaries,
    SyntheticDataConfig,
)


@pytest.mark.unit
class TestEnums:
    """Test enum definitions."""

    def test_brand_enum_values(self):
        """Test Brand enum has expected values."""
        assert Brand.REMIBRUTINIB.value == "Remibrutinib"
        assert Brand.FABHALTA.value == "Fabhalta"
        assert Brand.KISQALI.value == "Kisqali"

    def test_brand_enum_capitalized(self):
        """Test Brand values are capitalized (Supabase requirement)."""
        for brand in Brand:
            assert brand.value[0].isupper()

    def test_dgp_type_enum_values(self):
        """Test DGPType enum has expected values."""
        assert DGPType.SIMPLE_LINEAR.value == "simple_linear"
        assert DGPType.CONFOUNDED.value == "confounded"
        assert DGPType.HETEROGENEOUS.value == "heterogeneous"
        assert DGPType.TIME_SERIES.value == "time_series"
        assert DGPType.SELECTION_BIAS.value == "selection_bias"

    def test_data_split_enum_values(self):
        """Test DataSplit enum has expected values."""
        assert DataSplit.TRAIN.value == "train"
        assert DataSplit.VALIDATION.value == "validation"
        assert DataSplit.TEST.value == "test"
        assert DataSplit.HOLDOUT.value == "holdout"

    def test_specialty_enum_values(self):
        """Test SpecialtyEnum has medical specialties."""
        assert SpecialtyEnum.DERMATOLOGY.value == "dermatology"
        assert SpecialtyEnum.HEMATOLOGY.value == "hematology"
        assert SpecialtyEnum.ONCOLOGY.value == "oncology"
        assert SpecialtyEnum.NEUROLOGY.value == "neurology"

    def test_practice_type_enum_values(self):
        """Test PracticeTypeEnum values."""
        assert PracticeTypeEnum.ACADEMIC.value == "academic"
        assert PracticeTypeEnum.COMMUNITY.value == "community"
        assert PracticeTypeEnum.PRIVATE.value == "private"

    def test_region_enum_values(self):
        """Test RegionEnum has US regions."""
        assert RegionEnum.NORTHEAST.value == "northeast"
        assert RegionEnum.SOUTH.value == "south"
        assert RegionEnum.MIDWEST.value == "midwest"
        assert RegionEnum.WEST.value == "west"

    def test_insurance_type_enum_values(self):
        """Test InsuranceTypeEnum values."""
        assert InsuranceTypeEnum.COMMERCIAL.value == "commercial"
        assert InsuranceTypeEnum.MEDICARE.value == "medicare"
        assert InsuranceTypeEnum.MEDICAID.value == "medicaid"

    def test_engagement_type_enum_values(self):
        """Test EngagementTypeEnum values."""
        assert EngagementTypeEnum.DETAIL_VISIT.value == "detail_visit"
        assert EngagementTypeEnum.DIGITAL.value == "digital"
        assert EngagementTypeEnum.SPEAKER_PROGRAM.value == "speaker_program"
        assert EngagementTypeEnum.SAMPLE_REQUEST.value == "sample_request"
        assert EngagementTypeEnum.WEBINAR.value == "webinar"

    def test_brands_convenience_list(self):
        """Test BRANDS convenience list."""
        assert isinstance(BRANDS, list)
        assert len(BRANDS) == 3
        assert "Remibrutinib" in BRANDS
        assert "Fabhalta" in BRANDS
        assert "Kisqali" in BRANDS

    def test_dgp_types_convenience_list(self):
        """Test DGP_TYPES convenience list."""
        assert isinstance(DGP_TYPES, list)
        assert len(DGP_TYPES) == 5
        assert "simple_linear" in DGP_TYPES
        assert "confounded" in DGP_TYPES


@pytest.mark.unit
class TestDGPConfig:
    """Test DGPConfig dataclass."""

    def test_dgp_config_creation(self):
        """Test DGPConfig creation."""
        config = DGPConfig(
            dgp_type=DGPType.SIMPLE_LINEAR,
            true_ate=0.40,
            tolerance=0.05,
        )

        assert config.dgp_type == DGPType.SIMPLE_LINEAR
        assert config.true_ate == 0.40
        assert config.tolerance == 0.05
        assert config.treatment_variable == "engagement_score"  # Default
        assert config.outcome_variable == "treatment_initiated"  # Default

    def test_dgp_config_with_confounders(self):
        """Test DGPConfig with confounders."""
        config = DGPConfig(
            dgp_type=DGPType.CONFOUNDED,
            true_ate=0.25,
            confounders=["disease_severity", "academic_hcp"],
        )

        assert len(config.confounders) == 2
        assert "disease_severity" in config.confounders
        assert "academic_hcp" in config.confounders

    def test_dgp_config_with_cate(self):
        """Test DGPConfig with CATE segments."""
        cate_map = {
            "high_severity": 0.50,
            "medium_severity": 0.30,
            "low_severity": 0.15,
        }

        config = DGPConfig(
            dgp_type=DGPType.HETEROGENEOUS,
            true_ate=0.30,
            cate_by_segment=cate_map,
        )

        assert config.cate_by_segment == cate_map
        assert config.cate_by_segment["high_severity"] == 0.50

    def test_dgp_config_time_series_params(self):
        """Test DGPConfig with time series parameters."""
        config = DGPConfig(
            dgp_type=DGPType.TIME_SERIES,
            true_ate=0.30,
            lag_periods=2,
            include_seasonality=True,
        )

        assert config.lag_periods == 2
        assert config.include_seasonality is True

    def test_dgp_config_default_values(self):
        """Test DGPConfig default values."""
        config = DGPConfig(
            dgp_type=DGPType.SIMPLE_LINEAR,
            true_ate=0.40,
        )

        assert config.tolerance == 0.05
        assert config.confounders == []
        assert config.treatment_variable == "engagement_score"
        assert config.outcome_variable == "treatment_initiated"
        assert config.description == ""
        assert config.cate_by_segment is None
        assert config.lag_periods == 1
        assert config.include_seasonality is False


@pytest.mark.unit
class TestDGPConfigs:
    """Test pre-defined DGP configurations."""

    def test_all_dgp_types_configured(self):
        """Test that all DGP types have configurations."""
        for dgp_type in DGPType:
            assert dgp_type in DGP_CONFIGS

    def test_simple_linear_config(self):
        """Test SIMPLE_LINEAR DGP configuration."""
        config = DGP_CONFIGS[DGPType.SIMPLE_LINEAR]

        assert config.dgp_type == DGPType.SIMPLE_LINEAR
        assert config.true_ate == 0.40
        assert config.tolerance == 0.05
        assert config.confounders == []

    def test_confounded_config(self):
        """Test CONFOUNDED DGP configuration."""
        config = DGP_CONFIGS[DGPType.CONFOUNDED]

        assert config.dgp_type == DGPType.CONFOUNDED
        assert config.true_ate == 0.25
        assert len(config.confounders) > 0
        assert "disease_severity" in config.confounders

    def test_heterogeneous_config(self):
        """Test HETEROGENEOUS DGP configuration."""
        config = DGP_CONFIGS[DGPType.HETEROGENEOUS]

        assert config.dgp_type == DGPType.HETEROGENEOUS
        assert config.true_ate == 0.30
        assert config.cate_by_segment is not None
        assert "high_severity" in config.cate_by_segment
        assert "medium_severity" in config.cate_by_segment
        assert "low_severity" in config.cate_by_segment

    def test_time_series_config(self):
        """Test TIME_SERIES DGP configuration."""
        config = DGP_CONFIGS[DGPType.TIME_SERIES]

        assert config.dgp_type == DGPType.TIME_SERIES
        assert config.lag_periods == 2
        assert config.include_seasonality is True

    def test_selection_bias_config(self):
        """Test SELECTION_BIAS DGP configuration."""
        config = DGP_CONFIGS[DGPType.SELECTION_BIAS]

        assert config.dgp_type == DGPType.SELECTION_BIAS
        assert config.true_ate == 0.35
        assert len(config.confounders) > 0

    def test_all_configs_have_descriptions(self):
        """Test that all DGP configs have descriptions."""
        for _dgp_type, config in DGP_CONFIGS.items():
            assert config.description != ""
            assert isinstance(config.description, str)


@pytest.mark.unit
class TestSplitBoundaries:
    """Test SplitBoundaries dataclass."""

    def test_split_boundaries_defaults(self):
        """Test SplitBoundaries default values."""
        boundaries = SplitBoundaries()

        assert boundaries.data_start_date == date(2022, 1, 1)
        assert boundaries.data_end_date == date(2024, 12, 31)
        assert boundaries.train_end_date == date(2023, 6, 30)
        assert boundaries.validation_end_date == date(2024, 3, 31)
        assert boundaries.test_end_date == date(2024, 9, 30)

    def test_split_boundaries_ratios(self):
        """Test SplitBoundaries ratio validation."""
        boundaries = SplitBoundaries()

        assert boundaries.train_ratio == 0.60
        assert boundaries.validation_ratio == 0.20
        assert boundaries.test_ratio == 0.15
        assert boundaries.holdout_ratio == 0.05

        # Ratios should sum to 1.0
        total = (
            boundaries.train_ratio
            + boundaries.validation_ratio
            + boundaries.test_ratio
            + boundaries.holdout_ratio
        )
        assert abs(total - 1.0) < 0.001

    def test_split_boundaries_temporal_gap(self):
        """Test temporal gap configuration."""
        boundaries = SplitBoundaries()

        assert boundaries.temporal_gap_days == 7

    def test_get_split_for_date_train(self):
        """Test get_split_for_date returns train for early dates."""
        boundaries = SplitBoundaries()

        assert boundaries.get_split_for_date(date(2022, 1, 1)) == DataSplit.TRAIN
        assert boundaries.get_split_for_date(date(2023, 6, 30)) == DataSplit.TRAIN

    def test_get_split_for_date_validation(self):
        """Test get_split_for_date returns validation for mid dates."""
        boundaries = SplitBoundaries()

        assert boundaries.get_split_for_date(date(2023, 7, 1)) == DataSplit.VALIDATION
        assert boundaries.get_split_for_date(date(2024, 3, 31)) == DataSplit.VALIDATION

    def test_get_split_for_date_test(self):
        """Test get_split_for_date returns test for later dates."""
        boundaries = SplitBoundaries()

        assert boundaries.get_split_for_date(date(2024, 4, 1)) == DataSplit.TEST
        assert boundaries.get_split_for_date(date(2024, 9, 30)) == DataSplit.TEST

    def test_get_split_for_date_holdout(self):
        """Test get_split_for_date returns holdout for latest dates."""
        boundaries = SplitBoundaries()

        assert boundaries.get_split_for_date(date(2024, 10, 1)) == DataSplit.HOLDOUT
        assert boundaries.get_split_for_date(date(2024, 12, 31)) == DataSplit.HOLDOUT

    def test_custom_split_boundaries(self):
        """Test custom SplitBoundaries."""
        boundaries = SplitBoundaries(
            data_start_date=date(2020, 1, 1),
            train_end_date=date(2020, 6, 30),
            temporal_gap_days=14,
        )

        assert boundaries.data_start_date == date(2020, 1, 1)
        assert boundaries.train_end_date == date(2020, 6, 30)
        assert boundaries.temporal_gap_days == 14


@pytest.mark.unit
class TestEntityVolumes:
    """Test EntityVolumes dataclass."""

    def test_entity_volumes_defaults(self):
        """Test EntityVolumes default values."""
        volumes = EntityVolumes()

        assert volumes.hcp_profiles_per_brand == 5000
        assert volumes.patient_journeys_per_brand == 28333
        assert isinstance(volumes.treatment_events_per_patient, tuple)
        assert isinstance(volumes.engagement_events_per_patient, tuple)

    def test_entity_volumes_total_hcps(self):
        """Test total_hcps property."""
        volumes = EntityVolumes(hcp_profiles_per_brand=1000)

        assert volumes.total_hcps == 3000  # 1000 * 3 brands

    def test_entity_volumes_total_patients(self):
        """Test total_patients property."""
        volumes = EntityVolumes(patient_journeys_per_brand=10000)

        assert volumes.total_patients == 30000  # 10000 * 3 brands

    def test_entity_volumes_custom_values(self):
        """Test EntityVolumes with custom values."""
        volumes = EntityVolumes(
            hcp_profiles_per_brand=100,
            patient_journeys_per_brand=500,
            treatment_events_per_patient=(5, 20),
        )

        assert volumes.hcp_profiles_per_brand == 100
        assert volumes.patient_journeys_per_brand == 500
        assert volumes.treatment_events_per_patient == (5, 20)
        assert volumes.total_hcps == 300
        assert volumes.total_patients == 1500


@pytest.mark.unit
class TestSyntheticDataConfig:
    """Test SyntheticDataConfig dataclass."""

    def test_synthetic_data_config_defaults(self):
        """Test SyntheticDataConfig default values."""
        config = SyntheticDataConfig()

        assert config.config_name == "e2i_synthetic_v1"
        assert config.config_version == "1.0.0"
        assert isinstance(config.split_boundaries, SplitBoundaries)
        assert isinstance(config.entity_volumes, EntityVolumes)
        assert config.batch_size == 1000
        assert config.random_seed == 42

    def test_synthetic_data_config_validation_thresholds(self):
        """Test validation threshold settings."""
        config = SyntheticDataConfig()

        assert config.min_refutation_pass_rate == 0.60
        assert config.ate_tolerance == 0.05

    def test_synthetic_data_config_missingness_rates(self):
        """Test missingness rate configuration."""
        config = SyntheticDataConfig()

        assert isinstance(config.missingness_rates, dict)
        assert "insurance_type" in config.missingness_rates
        assert 0 <= config.missingness_rates["insurance_type"] <= 1

    def test_synthetic_data_config_measurement_error(self):
        """Test measurement error configuration."""
        config = SyntheticDataConfig()

        assert isinstance(config.measurement_error_std, dict)
        assert "disease_severity" in config.measurement_error_std
        assert config.measurement_error_std["disease_severity"] > 0

    def test_get_dgp_config(self):
        """Test get_dgp_config method."""
        config = SyntheticDataConfig()

        dgp_config = config.get_dgp_config(DGPType.SIMPLE_LINEAR)

        assert isinstance(dgp_config, DGPConfig)
        assert dgp_config.dgp_type == DGPType.SIMPLE_LINEAR
        assert dgp_config.true_ate == 0.40

    def test_get_all_dgp_configs(self):
        """Test get_all_dgp_configs method."""
        config = SyntheticDataConfig()

        all_configs = config.get_all_dgp_configs()

        assert isinstance(all_configs, dict)
        assert len(all_configs) == 5
        assert all(isinstance(k, DGPType) for k in all_configs.keys())
        assert all(isinstance(v, DGPConfig) for v in all_configs.values())

    def test_custom_synthetic_data_config(self):
        """Test SyntheticDataConfig with custom values."""
        custom_boundaries = SplitBoundaries(train_ratio=0.70)
        custom_volumes = EntityVolumes(hcp_profiles_per_brand=1000)

        config = SyntheticDataConfig(
            config_name="custom_v1",
            config_version="2.0.0",
            split_boundaries=custom_boundaries,
            entity_volumes=custom_volumes,
            batch_size=500,
            random_seed=123,
        )

        assert config.config_name == "custom_v1"
        assert config.config_version == "2.0.0"
        assert config.split_boundaries.train_ratio == 0.70
        assert config.entity_volumes.hcp_profiles_per_brand == 1000
        assert config.batch_size == 500
        assert config.random_seed == 123


@pytest.mark.unit
class TestDefaultConfig:
    """Test DEFAULT_CONFIG instance."""

    def test_default_config_exists(self):
        """Test that DEFAULT_CONFIG is created."""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, SyntheticDataConfig)

    def test_default_config_values(self):
        """Test DEFAULT_CONFIG has expected values."""
        assert DEFAULT_CONFIG.config_name == "e2i_synthetic_v1"
        assert DEFAULT_CONFIG.random_seed == 42

    def test_default_config_is_singleton(self):
        """Test DEFAULT_CONFIG is a module-level singleton."""
        from src.ml.synthetic.config import DEFAULT_CONFIG as dc2

        # Should be the same object
        assert DEFAULT_CONFIG is dc2


@pytest.mark.unit
class TestConfigConsistency:
    """Test configuration consistency and relationships."""

    def test_all_dgp_configs_have_valid_ates(self):
        """Test all DGP configs have valid ATE values."""
        for _dgp_type, config in DGP_CONFIGS.items():
            assert 0 <= config.true_ate <= 1
            assert 0 < config.tolerance < 1

    def test_heterogeneous_cate_consistency(self):
        """Test HETEROGENEOUS CATE segments average to ATE."""
        config = DGP_CONFIGS[DGPType.HETEROGENEOUS]

        # Average of CATE values should be close to ATE (assuming equal weights)
        cate_values = list(config.cate_by_segment.values())
        avg_cate = sum(cate_values) / len(cate_values)

        # Should be within tolerance
        assert abs(avg_cate - config.true_ate) <= 0.10

    def test_split_boundaries_chronological(self):
        """Test split boundaries are chronological."""
        boundaries = SplitBoundaries()

        assert boundaries.data_start_date < boundaries.train_end_date
        assert boundaries.train_end_date < boundaries.validation_end_date
        assert boundaries.validation_end_date < boundaries.test_end_date
        assert boundaries.test_end_date < boundaries.data_end_date

    def test_entity_volumes_ranges_valid(self):
        """Test entity volume ranges are valid."""
        volumes = EntityVolumes()

        # Ranges should be (min, max) with min < max
        assert volumes.treatment_events_per_patient[0] < volumes.treatment_events_per_patient[1]
        assert volumes.engagement_events_per_patient[0] < volumes.engagement_events_per_patient[1]
        assert volumes.ml_predictions_per_patient[0] < volumes.ml_predictions_per_patient[1]
