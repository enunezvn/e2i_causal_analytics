"""Tests for brand configuration factory."""

import pytest

from src.agents.cohort_constructor import (
    CohortConfig,
    Criterion,
    CriterionType,
    Operator,
    TemporalRequirements,
)
from src.agents.cohort_constructor.configs import (
    get_brand_config,
    list_available_configs,
    get_config_for_brand_indication,
    _get_remibrutinib_csu_config,
    _get_fabhalta_pnh_config,
    _get_fabhalta_c3g_config,
    _get_kisqali_hr_her2_bc_config,
)


class TestGetBrandConfig:
    """Tests for get_brand_config factory function."""

    def test_get_remibrutinib_config(self):
        """Test getting Remibrutinib configuration."""
        config = get_brand_config("remibrutinib")

        assert config.brand == "remibrutinib"
        assert config.indication == "csu"
        assert len(config.inclusion_criteria) > 0
        assert len(config.exclusion_criteria) > 0

    def test_get_remibrutinib_with_indication(self):
        """Test getting Remibrutinib with explicit indication."""
        config = get_brand_config("remibrutinib", "csu")

        assert config.brand == "remibrutinib"
        assert config.indication == "csu"

    def test_get_fabhalta_pnh_config(self):
        """Test getting Fabhalta PNH configuration."""
        config = get_brand_config("fabhalta", "pnh")

        assert config.brand == "fabhalta"
        assert config.indication == "pnh"

    def test_get_fabhalta_c3g_config(self):
        """Test getting Fabhalta C3G configuration."""
        config = get_brand_config("fabhalta", "c3g")

        assert config.brand == "fabhalta"
        assert config.indication == "c3g"

    def test_get_fabhalta_default_indication(self):
        """Test Fabhalta default is PNH."""
        config = get_brand_config("fabhalta")

        assert config.brand == "fabhalta"
        assert config.indication == "pnh"  # Default indication

    def test_get_kisqali_config(self):
        """Test getting Kisqali configuration."""
        config = get_brand_config("kisqali")

        assert config.brand == "kisqali"
        assert config.indication == "hr_her2_bc"

    def test_case_insensitive_brand(self):
        """Test brand name is case insensitive."""
        config1 = get_brand_config("REMIBRUTINIB")
        config2 = get_brand_config("Remibrutinib")
        config3 = get_brand_config("remibrutinib")

        assert config1.brand == config2.brand == config3.brand == "remibrutinib"

    def test_unsupported_brand_raises_error(self):
        """Test error for unsupported brand."""
        with pytest.raises(ValueError, match="Unsupported brand"):
            get_brand_config("unknown_brand")


class TestListAvailableConfigs:
    """Tests for list_available_configs function."""

    def test_list_configs_returns_dict(self):
        """Test that list returns a dictionary."""
        configs = list_available_configs()

        assert isinstance(configs, dict)
        assert len(configs) > 0

    def test_list_configs_contains_expected_brands(self):
        """Test that expected brands are in the list."""
        configs = list_available_configs()

        assert "remibrutinib" in configs
        assert "fabhalta" in configs
        assert "kisqali" in configs

    def test_list_configs_has_required_keys(self):
        """Test that config summaries have required keys."""
        configs = list_available_configs()

        for key, summary in configs.items():
            assert "cohort_name" in summary
            assert "brand" in summary
            assert "indication" in summary
            assert "version" in summary
            assert "inclusion_count" in summary
            assert "exclusion_count" in summary


class TestGetConfigForBrandIndication:
    """Tests for get_config_for_brand_indication function."""

    def test_explicit_brand_indication(self):
        """Test getting config with explicit brand and indication."""
        config = get_config_for_brand_indication("fabhalta", "pnh")

        assert config.brand == "fabhalta"
        assert config.indication == "pnh"


class TestRemibrutinibCSUConfig:
    """Tests for Remibrutinib CSU configuration details."""

    @pytest.fixture
    def remibrutinib_config(self):
        """Get Remibrutinib configuration."""
        return _get_remibrutinib_csu_config()

    def test_cohort_name(self, remibrutinib_config):
        """Test cohort name is set correctly."""
        assert "Remibrutinib" in remibrutinib_config.cohort_name
        assert "CSU" in remibrutinib_config.cohort_name

    def test_inclusion_criteria_count(self, remibrutinib_config):
        """Test expected number of inclusion criteria."""
        assert len(remibrutinib_config.inclusion_criteria) == 4

    def test_exclusion_criteria_count(self, remibrutinib_config):
        """Test expected number of exclusion criteria."""
        assert len(remibrutinib_config.exclusion_criteria) == 4

    def test_age_criterion(self, remibrutinib_config):
        """Test age inclusion criterion."""
        age_criteria = [c for c in remibrutinib_config.inclusion_criteria if c.field == "age_at_diagnosis"]

        assert len(age_criteria) == 1
        assert age_criteria[0].operator == Operator.GREATER_EQUAL
        assert age_criteria[0].value == 18

    def test_diagnosis_code_criterion(self, remibrutinib_config):
        """Test diagnosis code criterion includes CSU codes."""
        dx_criteria = [c for c in remibrutinib_config.inclusion_criteria if c.field == "diagnosis_code"]

        assert len(dx_criteria) == 1
        assert dx_criteria[0].operator == Operator.IN
        assert "L50.1" in dx_criteria[0].value  # Idiopathic urticaria

    def test_uas7_criterion(self, remibrutinib_config):
        """Test UAS7 severity criterion."""
        uas7_criteria = [c for c in remibrutinib_config.inclusion_criteria if c.field == "urticaria_severity_uas7"]

        assert len(uas7_criteria) == 1
        assert uas7_criteria[0].operator == Operator.GREATER_EQUAL
        assert uas7_criteria[0].value == 16  # Moderate-to-severe threshold

    def test_temporal_requirements(self, remibrutinib_config):
        """Test temporal requirements are set."""
        assert remibrutinib_config.temporal_requirements.lookback_days == 180
        assert remibrutinib_config.temporal_requirements.followup_days == 90

    def test_required_fields(self, remibrutinib_config):
        """Test required fields include key clinical fields."""
        assert "patient_journey_id" in remibrutinib_config.required_fields
        assert "age_at_diagnosis" in remibrutinib_config.required_fields
        assert "urticaria_severity_uas7" in remibrutinib_config.required_fields


class TestFabhaltaPNHConfig:
    """Tests for Fabhalta PNH configuration details."""

    @pytest.fixture
    def fabhalta_pnh_config(self):
        """Get Fabhalta PNH configuration."""
        return _get_fabhalta_pnh_config()

    def test_cohort_name(self, fabhalta_pnh_config):
        """Test cohort name is set correctly."""
        assert "Fabhalta" in fabhalta_pnh_config.cohort_name
        assert "PNH" in fabhalta_pnh_config.cohort_name

    def test_diagnosis_code_criterion(self, fabhalta_pnh_config):
        """Test PNH diagnosis codes."""
        dx_criteria = [c for c in fabhalta_pnh_config.inclusion_criteria if c.field == "diagnosis_code"]

        assert len(dx_criteria) == 1
        assert "D59.5" in dx_criteria[0].value  # PNH code

    def test_ldh_criterion(self, fabhalta_pnh_config):
        """Test LDH ratio criterion."""
        ldh_criteria = [c for c in fabhalta_pnh_config.inclusion_criteria if c.field == "ldh_ratio"]

        assert len(ldh_criteria) == 1
        assert ldh_criteria[0].value == 1.5

    def test_complement_inhibitor_criterion(self, fabhalta_pnh_config):
        """Test complement inhibitor status criterion."""
        ci_criteria = [c for c in fabhalta_pnh_config.inclusion_criteria if c.field == "complement_inhibitor_status"]

        assert len(ci_criteria) == 1
        assert "current" in ci_criteria[0].value
        assert "prior" in ci_criteria[0].value

    def test_meningococcal_exclusion(self, fabhalta_pnh_config):
        """Test meningococcal vaccination exclusion criterion."""
        mening_criteria = [c for c in fabhalta_pnh_config.exclusion_criteria if c.field == "meningococcal_vaccination_current"]

        assert len(mening_criteria) == 1
        assert mening_criteria[0].value == False  # Excluded if NOT current

    def test_temporal_requirements(self, fabhalta_pnh_config):
        """Test temporal requirements for PNH."""
        assert fabhalta_pnh_config.temporal_requirements.lookback_days == 365
        assert fabhalta_pnh_config.temporal_requirements.followup_days == 180


class TestFabhaltaC3GConfig:
    """Tests for Fabhalta C3G configuration details."""

    @pytest.fixture
    def fabhalta_c3g_config(self):
        """Get Fabhalta C3G configuration."""
        return _get_fabhalta_c3g_config()

    def test_cohort_name(self, fabhalta_c3g_config):
        """Test cohort name is set correctly."""
        assert "Fabhalta" in fabhalta_c3g_config.cohort_name
        assert "C3G" in fabhalta_c3g_config.cohort_name

    def test_indication(self, fabhalta_c3g_config):
        """Test indication is c3g."""
        assert fabhalta_c3g_config.indication == "c3g"

    def test_proteinuria_criterion(self, fabhalta_c3g_config):
        """Test proteinuria criterion."""
        prot_criteria = [c for c in fabhalta_c3g_config.inclusion_criteria if c.field == "proteinuria_g_day"]

        assert len(prot_criteria) == 1
        assert prot_criteria[0].value == 1.0

    def test_egfr_criterion(self, fabhalta_c3g_config):
        """Test eGFR criterion."""
        egfr_criteria = [c for c in fabhalta_c3g_config.inclusion_criteria if c.field == "egfr"]

        assert len(egfr_criteria) == 1
        assert egfr_criteria[0].value == 30


class TestKisqaliConfig:
    """Tests for Kisqali HR+/HER2- BC configuration details."""

    @pytest.fixture
    def kisqali_config(self):
        """Get Kisqali configuration."""
        return _get_kisqali_hr_her2_bc_config()

    def test_cohort_name(self, kisqali_config):
        """Test cohort name is set correctly."""
        assert "Kisqali" in kisqali_config.cohort_name

    def test_indication(self, kisqali_config):
        """Test indication."""
        assert kisqali_config.indication == "hr_her2_bc"

    def test_hr_status_criterion(self, kisqali_config):
        """Test HR status criterion."""
        hr_criteria = [c for c in kisqali_config.inclusion_criteria if c.field == "hr_status"]

        assert len(hr_criteria) == 1
        assert hr_criteria[0].value == "positive"

    def test_her2_status_criterion(self, kisqali_config):
        """Test HER2 status criterion."""
        her2_criteria = [c for c in kisqali_config.inclusion_criteria if c.field == "her2_status"]

        assert len(her2_criteria) == 1
        assert her2_criteria[0].value == "negative"

    def test_ecog_criterion(self, kisqali_config):
        """Test ECOG performance status criterion."""
        ecog_criteria = [c for c in kisqali_config.inclusion_criteria if c.field == "ecog_performance_status"]

        assert len(ecog_criteria) == 1
        assert ecog_criteria[0].operator == Operator.LESS_EQUAL
        assert ecog_criteria[0].value == 1

    def test_disease_stage_criterion(self, kisqali_config):
        """Test disease stage criterion."""
        stage_criteria = [c for c in kisqali_config.inclusion_criteria if c.field == "disease_stage"]

        assert len(stage_criteria) == 1
        assert "metastatic" in stage_criteria[0].value
        assert "advanced" in stage_criteria[0].value

    def test_cdk46_inhibitor_exclusion(self, kisqali_config):
        """Test prior CDK4/6 inhibitor exclusion."""
        cdk_criteria = [c for c in kisqali_config.exclusion_criteria if c.field == "prior_cdk46_inhibitor"]

        assert len(cdk_criteria) == 1
        assert cdk_criteria[0].value == True  # Excluded if prior CDK4/6

    def test_qtc_exclusion(self, kisqali_config):
        """Test QTc prolongation exclusion."""
        qtc_criteria = [c for c in kisqali_config.exclusion_criteria if c.field == "qtc_prolongation"]

        assert len(qtc_criteria) == 1


class TestConfigConsistency:
    """Tests for configuration consistency across brands."""

    def test_all_configs_have_required_fields(self):
        """Test all configurations have required fields defined."""
        configs = [
            get_brand_config("remibrutinib"),
            get_brand_config("fabhalta", "pnh"),
            get_brand_config("fabhalta", "c3g"),
            get_brand_config("kisqali"),
        ]

        for config in configs:
            assert len(config.required_fields) > 0
            assert "patient_journey_id" in config.required_fields

    def test_all_configs_have_adult_age_criterion(self):
        """Test all configurations require adult patients."""
        configs = [
            get_brand_config("remibrutinib"),
            get_brand_config("fabhalta", "pnh"),
            get_brand_config("fabhalta", "c3g"),
            get_brand_config("kisqali"),
        ]

        for config in configs:
            age_criteria = [c for c in config.inclusion_criteria if c.field == "age_at_diagnosis"]
            assert len(age_criteria) >= 1, f"{config.brand} missing age criterion"
            assert age_criteria[0].value >= 18

    def test_all_configs_have_diagnosis_criterion(self):
        """Test all configurations require diagnosis code."""
        configs = [
            get_brand_config("remibrutinib"),
            get_brand_config("fabhalta", "pnh"),
            get_brand_config("fabhalta", "c3g"),
            get_brand_config("kisqali"),
        ]

        for config in configs:
            dx_criteria = [c for c in config.inclusion_criteria if c.field == "diagnosis_code"]
            assert len(dx_criteria) >= 1, f"{config.brand} missing diagnosis criterion"

    def test_all_configs_have_version(self):
        """Test all configurations have version."""
        configs = list_available_configs()

        for key, summary in configs.items():
            assert summary["version"] is not None
            assert summary["version"] != ""

    def test_all_configs_have_temporal_requirements(self):
        """Test all configurations have temporal requirements."""
        configs = [
            get_brand_config("remibrutinib"),
            get_brand_config("fabhalta", "pnh"),
            get_brand_config("fabhalta", "c3g"),
            get_brand_config("kisqali"),
        ]

        for config in configs:
            assert config.temporal_requirements is not None
            assert config.temporal_requirements.lookback_days > 0
            assert config.temporal_requirements.followup_days > 0
