"""Tests for Feast entity definitions.

Validates that all entities are properly defined with correct:
- Join keys
- Descriptions
- Tags

Note: These tests require the feast package to be installed.
They are skipped if feast is not available.
"""

import sys
from pathlib import Path

import pytest

# Check if feast is installed
try:
    import feast  # noqa: F401

    HAS_FEAST = True
except ImportError:
    HAS_FEAST = False

# Skip entire module if feast is not installed
pytestmark = pytest.mark.skipif(
    not HAS_FEAST, reason="feast package not installed - install with: pip install feast"
)

# Add feature_repo to path for imports
feature_repo_path = Path(__file__).parent.parent.parent.parent / "feature_repo"
sys.path.insert(0, str(feature_repo_path))


class TestEntityDefinitions:
    """Test Feast entity definitions."""

    def test_entity_imports(self):
        """Test that all entities can be imported."""
        from entities import (
            brand,
            hcp,
            hcp_brand,
            hcp_territory,
            patient,
            patient_brand,
            territory,
            trigger,
        )

        # Verify entities are not None
        assert hcp is not None
        assert patient is not None
        assert territory is not None
        assert brand is not None
        assert trigger is not None
        assert hcp_brand is not None
        assert patient_brand is not None
        assert hcp_territory is not None

    def test_hcp_entity(self):
        """Test HCP entity definition."""
        from entities import hcp

        assert hcp.name == "hcp"
        # Feast 0.58.0 uses join_key (singular) returning a string
        assert hcp.join_key == "hcp_id"
        assert "Healthcare Provider" in hcp.description
        assert hcp.tags.get("domain") == "commercial"
        assert hcp.tags.get("pii") == "false"

    def test_patient_entity(self):
        """Test Patient entity definition."""
        from entities import patient

        assert patient.name == "patient"
        assert patient.join_key == "patient_id"
        assert "Anonymized" in patient.description
        assert patient.tags.get("pii") == "pseudonymized"
        assert patient.tags.get("retention_days") == "365"

    def test_territory_entity(self):
        """Test Territory entity definition."""
        from entities import territory

        assert territory.name == "territory"
        assert territory.join_key == "territory_id"
        assert "sales territory" in territory.description.lower()
        assert territory.tags.get("owner") == "sales-ops"

    def test_brand_entity(self):
        """Test Brand entity definition."""
        from entities import brand

        assert brand.name == "brand"
        assert brand.join_key == "brand_id"
        # Should mention at least one of our brands
        assert any(b in brand.description for b in ["Remibrutinib", "Fabhalta", "Kisqali"])

    def test_trigger_entity(self):
        """Test Trigger entity definition."""
        from entities import trigger

        assert trigger.name == "trigger"
        assert trigger.join_key == "trigger_id"
        assert "Marketing" in trigger.description
        assert trigger.tags.get("owner") == "marketing"

    def test_composite_entities(self):
        """Test composite entity definitions.

        Note: Feast 0.58.0 only supports single join keys, so composite entities
        use single composite key strings (e.g., "hcp_brand_id" = "{hcp_id}_{brand_id}").
        """
        from entities import hcp_brand, hcp_territory, patient_brand

        # HCP-Brand composite (uses single composite key)
        assert hcp_brand.name == "hcp_brand"
        assert hcp_brand.join_key == "hcp_brand_id"
        assert hcp_brand.tags.get("composite") == "true"
        assert "key_format" in hcp_brand.tags

        # Patient-Brand composite
        assert patient_brand.name == "patient_brand"
        assert patient_brand.join_key == "patient_brand_id"
        assert patient_brand.tags.get("pii") == "pseudonymized"

        # HCP-Territory composite
        assert hcp_territory.name == "hcp_territory"
        assert hcp_territory.join_key == "hcp_territory_id"

    def test_entity_registry(self):
        """Test entity registry functions."""
        from entities import ALL_ENTITIES, ENTITY_MAP, get_entity

        # All entities in list
        assert len(ALL_ENTITIES) == 8

        # Entity map has correct entries
        assert len(ENTITY_MAP) == 8
        assert "hcp" in ENTITY_MAP
        assert "patient" in ENTITY_MAP

        # get_entity works
        hcp = get_entity("hcp")
        assert hcp.name == "hcp"

        # get_entity raises for unknown
        with pytest.raises(KeyError) as exc_info:
            get_entity("unknown_entity")
        assert "not found" in str(exc_info.value)

    def test_all_entities_have_required_tags(self):
        """Test that all entities have required tags."""
        from entities import ALL_ENTITIES

        required_tags = ["domain", "owner"]

        for entity in ALL_ENTITIES:
            for tag in required_tags:
                assert tag in entity.tags, f"Entity {entity.name} missing tag: {tag}"


class TestDataSourceDefinitions:
    """Test Feast data source definitions."""

    def test_data_source_imports(self):
        """Test that all data sources can be imported."""
        from data_sources import (
            business_metrics_source,
            hcp_profiles_source,
            patient_journey_source,
            territory_metrics_source,
            triggers_source,
        )

        assert business_metrics_source is not None
        assert patient_journey_source is not None
        assert triggers_source is not None
        assert hcp_profiles_source is not None
        assert territory_metrics_source is not None

    def test_business_metrics_source(self):
        """Test business metrics data source."""
        from data_sources import business_metrics_source

        assert business_metrics_source.name == "business_metrics_source"
        # Feast 0.58.0 uses event_timestamp_column or timestamp_field internally
        assert hasattr(business_metrics_source, "name")

    def test_patient_journey_source(self):
        """Test patient journey data source."""
        from data_sources import patient_journey_source

        assert patient_journey_source.name == "patient_journey_source"
        assert hasattr(patient_journey_source, "name")

    def test_triggers_source(self):
        """Test triggers data source."""
        from data_sources import triggers_source

        assert triggers_source.name == "triggers_source"
        assert hasattr(triggers_source, "name")

    def test_source_registry(self):
        """Test source registry functions."""
        from data_sources import ALL_SOURCES, get_source

        # Check source count
        assert len(ALL_SOURCES) >= 5

        # Source map works
        source = get_source("business_metrics_source")
        assert source.name == "business_metrics_source"

        # Unknown source raises
        with pytest.raises(KeyError):
            get_source("unknown_source")


class TestFeatureViewDefinitions:
    """Test Feast feature view definitions."""

    def test_feature_view_imports(self):
        """Test that all feature views can be imported."""
        from features import (
            hcp_conversion_fv,
            hcp_profile_fv,
            market_dynamics_fv,
            patient_journey_fv,
            trigger_effectiveness_fv,
        )

        assert hcp_conversion_fv is not None
        assert hcp_profile_fv is not None
        assert patient_journey_fv is not None
        assert trigger_effectiveness_fv is not None
        assert market_dynamics_fv is not None

    def test_hcp_conversion_feature_view(self):
        """Test HCP conversion feature view."""
        from features import hcp_conversion_fv

        assert hcp_conversion_fv.name == "hcp_conversion_features"
        assert hcp_conversion_fv.online is True

        # Check tags
        assert hcp_conversion_fv.tags.get("use_case") == "hcp_conversion"
        assert hcp_conversion_fv.tags.get("model_type") == "binary_classification"

        # Check has expected fields
        field_names = [f.name for f in hcp_conversion_fv.schema]
        assert "trx_count" in field_names
        assert "market_share" in field_names
        assert "engagement_score" in field_names

    def test_patient_journey_feature_view(self):
        """Test patient journey feature view."""
        from features import patient_journey_fv

        assert patient_journey_fv.name == "patient_journey_features"
        assert patient_journey_fv.tags.get("use_case") == "churn_prediction"

        # Check has expected fields
        field_names = [f.name for f in patient_journey_fv.schema]
        assert "adherence_rate" in field_names
        assert "churn_risk_score" in field_names

    def test_trigger_effectiveness_feature_view(self):
        """Test trigger effectiveness feature view."""
        from features import trigger_effectiveness_fv

        assert trigger_effectiveness_fv.name == "trigger_effectiveness_features"
        assert trigger_effectiveness_fv.tags.get("use_case") == "trigger_effectiveness"

        field_names = [f.name for f in trigger_effectiveness_fv.schema]
        assert "trigger_type" in field_names
        assert "conversion_flag" in field_names

    def test_market_dynamics_feature_view(self):
        """Test market dynamics feature view."""
        from features import market_dynamics_fv

        assert market_dynamics_fv.name == "market_dynamics_features"
        assert market_dynamics_fv.tags.get("use_case") == "roi_prediction"
        assert market_dynamics_fv.tags.get("model_type") == "regression"

    def test_feature_view_registry(self):
        """Test feature view registry."""
        from features import FEATURE_VIEW_MAP, get_feature_view

        # Check registry has all views
        assert "hcp_conversion" in FEATURE_VIEW_MAP
        assert "patient_journey" in FEATURE_VIEW_MAP
        assert "trigger_effectiveness" in FEATURE_VIEW_MAP
        assert "market_dynamics" in FEATURE_VIEW_MAP

        # get_feature_view works
        fv = get_feature_view("hcp_conversion")
        assert fv.name == "hcp_conversion_features"

        # Unknown raises
        with pytest.raises(KeyError):
            get_feature_view("unknown_view")

    def test_all_feature_views_have_online_enabled(self):
        """Test that all feature views have online serving enabled."""
        from features import FEATURE_VIEW_MAP

        for name, fv in FEATURE_VIEW_MAP.items():
            assert fv.online is True, f"Feature view {name} should have online=True"

    def test_all_feature_views_have_use_case_tag(self):
        """Test that all feature views have use_case tag."""
        from features import FEATURE_VIEW_MAP

        for name, fv in FEATURE_VIEW_MAP.items():
            assert "use_case" in fv.tags, f"Feature view {name} missing use_case tag"
