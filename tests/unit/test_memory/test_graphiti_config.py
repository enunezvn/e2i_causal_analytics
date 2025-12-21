"""
Unit tests for E2I Graphiti Configuration.

Tests the configuration loading and entity/relationship type mappings.
"""

import os
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch, MagicMock

from src.memory.graphiti_config import (
    E2IEntityType,
    E2IRelationshipType,
    GraphitiEntityConfig,
    GraphitiRelationshipConfig,
    GraphitiConfig,
    load_graphiti_config,
    get_graphiti_config,
    clear_graphiti_config_cache,
    DEFAULT_ENTITY_CONFIGS,
    DEFAULT_RELATIONSHIP_CONFIGS,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear config cache before and after each test."""
    clear_graphiti_config_cache()
    yield
    clear_graphiti_config_cache()


@pytest.fixture
def sample_config_yaml():
    """Sample YAML configuration content."""
    return """
environment: local_pilot

memory_backends:
  semantic:
    local_pilot:
      graph_name: test_semantic_graph
      graphity:
        enabled: true
        model: claude-3-5-sonnet-latest
        entity_types:
          - Patient
          - HCP
          - Brand
        relationship_types:
          - CAUSES
          - IMPACTS
          - PRESCRIBES
      cache:
        enabled: true

cognitive_workflow:
  reflector:
    graphity_batch_size: 10
    min_confidence_for_learning: 0.7

performance:
  cache:
    semantic_cache_ttl_minutes: 10
"""


@pytest.fixture
def temp_config_file(sample_config_yaml):
    """Create a temporary config file."""
    with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sample_config_yaml)
        f.flush()
        yield Path(f.name)
    # Cleanup
    os.unlink(f.name)


# ============================================================================
# Entity Type Enum Tests
# ============================================================================

class TestE2IEntityType:
    """Tests for E2IEntityType enum."""

    def test_all_entity_types_defined(self):
        """Test all expected entity types are defined."""
        expected_types = [
            "Patient", "HCP", "Brand", "Region", "KPI",
            "CausalPath", "Trigger", "Agent", "Episode", "Community"
        ]
        actual_types = [e.value for e in E2IEntityType]
        for expected in expected_types:
            assert expected in actual_types

    def test_entity_type_values(self):
        """Test entity type string values."""
        assert E2IEntityType.PATIENT.value == "Patient"
        assert E2IEntityType.HCP.value == "HCP"
        assert E2IEntityType.BRAND.value == "Brand"
        assert E2IEntityType.KPI.value == "KPI"
        assert E2IEntityType.EPISODE.value == "Episode"

    def test_entity_type_is_string_enum(self):
        """Test entity types are string enums."""
        assert isinstance(E2IEntityType.PATIENT.value, str)
        # String enum allows direct comparison
        assert E2IEntityType.PATIENT == "Patient"


# ============================================================================
# Relationship Type Enum Tests
# ============================================================================

class TestE2IRelationshipType:
    """Tests for E2IRelationshipType enum."""

    def test_all_relationship_types_defined(self):
        """Test all expected relationship types are defined."""
        expected_types = [
            "TREATED_BY", "PRESCRIBED", "PRESCRIBES", "CAUSES",
            "IMPACTS", "INFLUENCES", "DISCOVERED", "GENERATED",
            "MENTIONS", "MEMBER_OF", "RELATES_TO"
        ]
        actual_types = [r.value for r in E2IRelationshipType]
        for expected in expected_types:
            assert expected in actual_types

    def test_relationship_type_values(self):
        """Test relationship type string values."""
        assert E2IRelationshipType.CAUSES.value == "CAUSES"
        assert E2IRelationshipType.IMPACTS.value == "IMPACTS"
        assert E2IRelationshipType.PRESCRIBES.value == "PRESCRIBES"
        assert E2IRelationshipType.DISCOVERED.value == "DISCOVERED"

    def test_relationship_type_is_string_enum(self):
        """Test relationship types are string enums."""
        assert isinstance(E2IRelationshipType.CAUSES.value, str)
        assert E2IRelationshipType.CAUSES == "CAUSES"


# ============================================================================
# Entity Config Dataclass Tests
# ============================================================================

class TestGraphitiEntityConfig:
    """Tests for GraphitiEntityConfig dataclass."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = GraphitiEntityConfig(
            name="TestEntity",
            label="TestEntity",
        )
        assert config.name == "TestEntity"
        assert config.label == "TestEntity"
        assert config.description == ""
        assert config.properties == []
        assert config.indexes == []

    def test_create_with_all_fields(self):
        """Test creating config with all fields."""
        config = GraphitiEntityConfig(
            name="HCP",
            label="HCP",
            description="Healthcare Provider",
            properties=["hcp_id", "npi", "specialty"],
            indexes=["hcp_id", "npi"],
        )
        assert config.description == "Healthcare Provider"
        assert len(config.properties) == 3
        assert "npi" in config.indexes


class TestGraphitiRelationshipConfig:
    """Tests for GraphitiRelationshipConfig dataclass."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = GraphitiRelationshipConfig(
            name="RELATES_TO",
            label="RELATES_TO",
        )
        assert config.name == "RELATES_TO"
        assert config.description == ""
        assert config.properties == []

    def test_create_with_all_fields(self):
        """Test creating config with all fields."""
        config = GraphitiRelationshipConfig(
            name="CAUSES",
            label="CAUSES",
            description="Causal relationship",
            properties=["effect_size", "confidence", "p_value"],
        )
        assert config.description == "Causal relationship"
        assert "confidence" in config.properties


# ============================================================================
# GraphitiConfig Tests
# ============================================================================

class TestGraphitiConfig:
    """Tests for GraphitiConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GraphitiConfig()
        assert config.enabled is True
        assert config.model == "claude-3-5-sonnet-latest"
        assert config.graph_name == "e2i_semantic"
        assert config.falkordb_host == "localhost"
        assert config.falkordb_port == 6380
        assert config.episode_batch_size == 5
        assert config.min_confidence_for_extraction == 0.6
        assert config.cache_enabled is True
        assert config.cache_ttl_minutes == 5

    def test_entity_types_default(self):
        """Test default entity types include all types."""
        config = GraphitiConfig()
        assert len(config.entity_types) == len(E2IEntityType)

    def test_relationship_types_default(self):
        """Test default relationship types include all types."""
        config = GraphitiConfig()
        assert len(config.relationship_types) == len(E2IRelationshipType)

    def test_get_entity_label_from_config(self):
        """Test getting entity label from config."""
        config = GraphitiConfig(
            entity_configs={
                "HCP": GraphitiEntityConfig(name="HCP", label="HealthcareProvider")
            }
        )
        label = config.get_entity_label(E2IEntityType.HCP)
        assert label == "HealthcareProvider"

    def test_get_entity_label_default(self):
        """Test getting entity label falls back to value."""
        config = GraphitiConfig(entity_configs={})
        label = config.get_entity_label(E2IEntityType.PATIENT)
        assert label == "Patient"

    def test_get_relationship_label_from_config(self):
        """Test getting relationship label from config."""
        config = GraphitiConfig(
            relationship_configs={
                "CAUSES": GraphitiRelationshipConfig(name="CAUSES", label="CAUSAL_LINK")
            }
        )
        label = config.get_relationship_label(E2IRelationshipType.CAUSES)
        assert label == "CAUSAL_LINK"

    def test_get_relationship_label_default(self):
        """Test getting relationship label falls back to value."""
        config = GraphitiConfig(relationship_configs={})
        label = config.get_relationship_label(E2IRelationshipType.IMPACTS)
        assert label == "IMPACTS"


# ============================================================================
# Default Config Tests
# ============================================================================

class TestDefaultConfigs:
    """Tests for default entity and relationship configurations."""

    def test_default_entity_configs_coverage(self):
        """Test all entity types have default configs."""
        for entity_type in E2IEntityType:
            assert entity_type.value in DEFAULT_ENTITY_CONFIGS

    def test_default_relationship_configs_coverage(self):
        """Test all relationship types have default configs."""
        for rel_type in E2IRelationshipType:
            assert rel_type.value in DEFAULT_RELATIONSHIP_CONFIGS

    def test_hcp_entity_config(self):
        """Test HCP entity configuration."""
        config = DEFAULT_ENTITY_CONFIGS["HCP"]
        assert config.name == "HCP"
        assert "npi" in config.properties
        assert "specialty" in config.properties
        assert "npi" in config.indexes

    def test_patient_entity_config(self):
        """Test Patient entity configuration."""
        config = DEFAULT_ENTITY_CONFIGS["Patient"]
        assert config.name == "Patient"
        assert "patient_id" in config.properties
        assert "journey_stage" in config.properties

    def test_causes_relationship_config(self):
        """Test CAUSES relationship configuration."""
        config = DEFAULT_RELATIONSHIP_CONFIGS["CAUSES"]
        assert config.name == "CAUSES"
        assert "effect_size" in config.properties
        assert "confidence" in config.properties

    def test_prescribes_relationship_config(self):
        """Test PRESCRIBES relationship configuration."""
        config = DEFAULT_RELATIONSHIP_CONFIGS["PRESCRIBES"]
        assert config.name == "PRESCRIBES"
        assert "frequency" in config.properties


# ============================================================================
# Config Loading Tests
# ============================================================================

class TestLoadGraphitiConfig:
    """Tests for load_graphiti_config function."""

    def test_load_from_file(self, temp_config_file):
        """Test loading config from YAML file."""
        config = load_graphiti_config(temp_config_file)

        assert config.enabled is True
        assert config.graph_name == "test_semantic_graph"
        assert config.episode_batch_size == 10
        assert config.min_confidence_for_extraction == 0.7
        assert config.cache_ttl_minutes == 10

    def test_load_entity_types_from_file(self, temp_config_file):
        """Test entity types are loaded from config."""
        config = load_graphiti_config(temp_config_file)

        # Should have only the 3 types from YAML
        assert len(config.entity_types) == 3
        assert E2IEntityType.PATIENT in config.entity_types
        assert E2IEntityType.HCP in config.entity_types
        assert E2IEntityType.BRAND in config.entity_types

    def test_load_relationship_types_from_file(self, temp_config_file):
        """Test relationship types are loaded from config."""
        config = load_graphiti_config(temp_config_file)

        assert len(config.relationship_types) == 3
        assert E2IRelationshipType.CAUSES in config.relationship_types
        assert E2IRelationshipType.IMPACTS in config.relationship_types
        assert E2IRelationshipType.PRESCRIBES in config.relationship_types

    def test_load_nonexistent_file_uses_defaults(self):
        """Test loading from nonexistent file uses defaults."""
        config = load_graphiti_config(Path("/nonexistent/config.yaml"))

        # Should have default values
        assert config.graph_name == "e2i_semantic"
        assert len(config.entity_types) == len(E2IEntityType)

    def test_load_with_environment_override(self, temp_config_file):
        """Test environment variable overrides config."""
        with patch.dict(os.environ, {"E2I_ENVIRONMENT": "local_pilot"}):
            config = load_graphiti_config(temp_config_file)
            assert config.graph_name == "test_semantic_graph"

    def test_load_with_falkordb_env_vars(self, temp_config_file):
        """Test FalkorDB settings from environment variables."""
        with patch.dict(os.environ, {
            "FALKORDB_HOST": "custom-host",
            "FALKORDB_PORT": "6381",
        }):
            config = load_graphiti_config(temp_config_file)
            assert config.falkordb_host == "custom-host"
            assert config.falkordb_port == 6381

    def test_load_invalid_entity_type_warning(self, temp_config_file):
        """Test invalid entity types are logged as warnings."""
        # Create config with invalid entity type
        invalid_yaml = """
environment: local_pilot
memory_backends:
  semantic:
    local_pilot:
      graphity:
        enabled: true
        entity_types:
          - Patient
          - InvalidType
"""
        with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_graphiti_config(config_path)
            # Should only have Patient (InvalidType should be skipped)
            assert len(config.entity_types) == 1
            assert E2IEntityType.PATIENT in config.entity_types
        finally:
            os.unlink(config_path)


class TestGetGraphitiConfig:
    """Tests for get_graphiti_config singleton function."""

    def test_returns_config(self):
        """Test function returns a GraphitiConfig."""
        with patch('src.memory.graphiti_config.load_graphiti_config') as mock_load:
            mock_load.return_value = GraphitiConfig()
            config = get_graphiti_config()
            assert isinstance(config, GraphitiConfig)

    def test_caches_config(self):
        """Test config is cached after first call."""
        with patch('src.memory.graphiti_config.load_graphiti_config') as mock_load:
            mock_load.return_value = GraphitiConfig()

            config1 = get_graphiti_config()
            config2 = get_graphiti_config()

            # Should only load once
            assert mock_load.call_count == 1
            assert config1 is config2


class TestClearGraphitiConfigCache:
    """Tests for clear_graphiti_config_cache function."""

    def test_clear_cache(self):
        """Test cache clearing causes reload."""
        with patch('src.memory.graphiti_config.load_graphiti_config') as mock_load:
            mock_load.return_value = GraphitiConfig()

            # First call
            get_graphiti_config()
            assert mock_load.call_count == 1

            # Clear and call again
            clear_graphiti_config_cache()
            get_graphiti_config()

            # Should have loaded again
            assert mock_load.call_count == 2


# ============================================================================
# Model Configuration Tests
# ============================================================================

class TestModelConfiguration:
    """Tests for model configuration."""

    def test_default_model_is_sonnet(self):
        """Test default model is Claude 3.5 Sonnet."""
        config = GraphitiConfig()
        assert config.model == "claude-3-5-sonnet-latest"

    def test_model_from_yaml(self, temp_config_file):
        """Test model can be loaded from YAML."""
        config = load_graphiti_config(temp_config_file)
        assert config.model == "claude-3-5-sonnet-latest"


# ============================================================================
# Port Configuration Tests
# ============================================================================

class TestPortConfiguration:
    """Tests for FalkorDB port configuration."""

    def test_default_port_is_6380(self):
        """Test default port matches docker-compose external port."""
        config = GraphitiConfig()
        assert config.falkordb_port == 6380

    def test_port_from_environment(self):
        """Test port can be overridden via environment."""
        with patch.dict(os.environ, {"FALKORDB_PORT": "6399"}):
            # Need to clear cache to pick up env change
            clear_graphiti_config_cache()

            # Create a temp config file
            with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("environment: local_pilot\n")
                f.flush()
                config_path = Path(f.name)

            try:
                config = load_graphiti_config(config_path)
                assert config.falkordb_port == 6399
            finally:
                os.unlink(config_path)
