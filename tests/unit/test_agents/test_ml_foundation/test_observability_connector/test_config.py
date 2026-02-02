"""Unit tests for Observability Configuration Loader.

Tests configuration loading from YAML with:
- Default values
- YAML file loading
- Environment variable substitution
- Environment-specific overrides
- Validation and error handling

Phase 3.4 Configuration tests.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.agents.ml_foundation.observability_connector.config import (
    BatchingSettings,
    CacheSettings,
    CacheTTLSettings,
    CircuitBreakerFallback,
    CircuitBreakerSettings,
    ObservabilityConfig,
    OpikSettings,
    RetrySettings,
    SamplingSettings,
    get_observability_config,
    reset_observability_config,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_config_yaml():
    """Sample YAML configuration."""
    return {
        "opik": {
            "enabled": True,
            "api_key_env": "OPIK_API_KEY",
            "workspace": "test-workspace",
            "project_name": "test-project",
            "use_local": False,
        },
        "sampling": {
            "default_rate": 0.5,
            "production_rate": 0.1,
            "always_sample_errors": True,
            "always_sample_agents": ["orchestrator"],
            "agent_overrides": {"drift_monitor": 0.2},
        },
        "batching": {
            "enabled": True,
            "max_batch_size": 50,
            "max_wait_seconds": 3.0,
            "retry": {
                "enabled": True,
                "max_retries": 5,
            },
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 3,
            "reset_timeout_seconds": 60.0,
            "fallback": {
                "log_to_database": True,
                "log_to_file": True,
            },
        },
        "cache": {
            "backend": "redis",
            "key_prefix": "test_metrics",
            "ttl": {
                "1h": 30,
                "24h": 150,
                "7d": 300,
            },
        },
        "retention": {
            "span_ttl_days": 60,
            "cleanup": {
                "enabled": True,
                "batch_size": 500,
            },
        },
        "database": {
            "spans_table": "test_spans",
            "batch_insert": True,
        },
        "spans": {
            "include_input": True,
            "max_input_size_bytes": 5000,
            "redact_patterns": ["password", "secret"],
        },
        "agent_tiers": {
            "tier_0": {
                "sample_rate": 0.5,
                "agents": ["data_preparer", "model_trainer"],
            },
            "tier_1": {
                "sample_rate": 1.0,
                "detailed_logging": True,
                "agents": ["orchestrator"],
            },
        },
        "logging": {
            "level": "DEBUG",
            "format": "json",
        },
        "environments": {
            "production": {
                "sampling": {
                    "default_rate": 0.05,
                },
                "cache": {
                    "backend": "redis",
                },
            },
        },
    }


@pytest.fixture
def temp_config_file(sample_config_yaml):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_yaml, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_observability_config()
    yield
    reset_observability_config()


# ============================================================================
# OPIK SETTINGS TESTS
# ============================================================================


class TestOpikSettings:
    """Tests for OpikSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = OpikSettings()
        assert settings.enabled is True
        assert settings.api_key_env == "OPIK_API_KEY"
        assert settings.workspace == "default"
        assert settings.project_name == "e2i-causal-analytics"
        assert settings.use_local is False

    def test_custom_values(self):
        """Test custom values."""
        settings = OpikSettings(
            enabled=False,
            workspace="custom",
            project_name="my-project",
        )
        assert settings.enabled is False
        assert settings.workspace == "custom"
        assert settings.project_name == "my-project"

    def test_api_key_from_env(self):
        """Test API key from environment."""
        settings = OpikSettings()
        with patch.dict(os.environ, {"OPIK_API_KEY": "test-key"}):
            assert settings.api_key == "test-key"

    def test_api_key_missing(self):
        """Test missing API key."""
        settings = OpikSettings()
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPIK_API_KEY if it exists
            os.environ.pop("OPIK_API_KEY", None)
            assert settings.api_key is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        settings = OpikSettings(workspace="test")
        result = settings.to_dict()
        assert result["workspace"] == "test"
        assert "enabled" in result
        assert "has_api_key" in result


# ============================================================================
# SAMPLING SETTINGS TESTS
# ============================================================================


class TestSamplingSettings:
    """Tests for SamplingSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = SamplingSettings()
        assert settings.default_rate == 1.0
        assert settings.production_rate == 0.1
        assert settings.always_sample_errors is True

    def test_get_rate_default(self):
        """Test default rate."""
        settings = SamplingSettings(default_rate=0.5)
        assert settings.get_rate() == 0.5

    def test_get_rate_with_override(self):
        """Test rate with agent override."""
        settings = SamplingSettings(
            default_rate=0.5,
            agent_overrides={"test_agent": 0.8},
        )
        assert settings.get_rate("test_agent") == 0.8
        assert settings.get_rate("other_agent") == 0.5

    def test_get_rate_always_sample_agents(self):
        """Test always sample agents."""
        settings = SamplingSettings(
            default_rate=0.1,
            always_sample_agents=["important_agent"],
        )
        assert settings.get_rate("important_agent") == 1.0
        assert settings.get_rate("other_agent") == 0.1

    def test_get_rate_production_environment(self):
        """Test production rate selection."""
        settings = SamplingSettings(
            default_rate=1.0,
            production_rate=0.1,
            environment_aware=True,
        )
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert settings.get_rate() == 0.1

    def test_get_rate_development_environment(self):
        """Test development rate selection."""
        settings = SamplingSettings(
            default_rate=1.0,
            production_rate=0.1,
            environment_aware=True,
        )
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert settings.get_rate() == 1.0


# ============================================================================
# BATCHING SETTINGS TESTS
# ============================================================================


class TestBatchingSettings:
    """Tests for BatchingSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = BatchingSettings()
        assert settings.enabled is True
        assert settings.max_batch_size == 100
        assert settings.max_wait_seconds == 5.0

    def test_custom_values(self):
        """Test custom values."""
        settings = BatchingSettings(
            max_batch_size=50,
            max_wait_seconds=2.0,
        )
        assert settings.max_batch_size == 50
        assert settings.max_wait_seconds == 2.0

    def test_retry_settings(self):
        """Test nested retry settings."""
        retry = RetrySettings(max_retries=5)
        settings = BatchingSettings(retry=retry)
        assert settings.retry.max_retries == 5
        assert settings.retry.enabled is True


# ============================================================================
# CIRCUIT BREAKER SETTINGS TESTS
# ============================================================================


class TestCircuitBreakerSettings:
    """Tests for CircuitBreakerSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = CircuitBreakerSettings()
        assert settings.enabled is True
        assert settings.failure_threshold == 5
        assert settings.reset_timeout_seconds == 30.0

    def test_fallback_settings(self):
        """Test fallback settings."""
        fallback = CircuitBreakerFallback(log_to_file=True)
        settings = CircuitBreakerSettings(fallback=fallback)
        assert settings.fallback.log_to_file is True
        assert settings.fallback.log_to_database is True


# ============================================================================
# CACHE SETTINGS TESTS
# ============================================================================


class TestCacheSettings:
    """Tests for CacheSettings dataclass."""

    def test_default_values(self):
        """Test default values."""
        settings = CacheSettings()
        assert settings.backend == "memory"
        assert settings.key_prefix == "obs_metrics"

    def test_ttl_get_ttl(self):
        """Test TTL retrieval."""
        ttl = CacheTTLSettings(window_1h=30, window_24h=150)
        assert ttl.get_ttl("1h") == 30
        assert ttl.get_ttl("24h") == 150
        assert ttl.get_ttl("7d") == 600  # default
        assert ttl.get_ttl("unknown") == 120  # default


# ============================================================================
# OBSERVABILITY CONFIG TESTS
# ============================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ObservabilityConfig()
        assert config.opik.enabled is True
        assert config.sampling.default_rate == 1.0
        assert config.batching.max_batch_size == 100
        assert config.circuit_breaker.failure_threshold == 5

    def test_from_yaml_with_file(self, temp_config_file):
        """Test loading from YAML file."""
        config = ObservabilityConfig.from_yaml(temp_config_file)

        assert config.opik.workspace == "test-workspace"
        assert config.opik.project_name == "test-project"
        assert config.sampling.default_rate == 0.5
        assert config.batching.max_batch_size == 50
        assert config.circuit_breaker.failure_threshold == 3
        assert config.cache.backend == "redis"

    def test_from_yaml_missing_file(self):
        """Test loading from missing file returns defaults."""
        config = ObservabilityConfig.from_yaml("/nonexistent/path.yaml")
        assert config.opik.enabled is True  # default value

    def test_from_yaml_environment_override(self, sample_config_yaml):
        """Test environment-specific overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config_yaml, f)
            temp_path = f.name

        try:
            # Load with production environment
            config = ObservabilityConfig.from_yaml(temp_path, "production")
            assert config.sampling.default_rate == 0.05  # overridden
            assert config.cache.backend == "redis"  # overridden
        finally:
            os.unlink(temp_path)

    def test_get_agent_tier(self, temp_config_file):
        """Test getting agent tier settings."""
        config = ObservabilityConfig.from_yaml(temp_config_file)

        tier = config.get_agent_tier("orchestrator")
        assert tier is not None
        assert tier.sample_rate == 1.0
        assert tier.detailed_logging is True

        tier = config.get_agent_tier("data_preparer")
        assert tier is not None
        assert tier.sample_rate == 0.5

        tier = config.get_agent_tier("unknown_agent")
        assert tier is None

    def test_get_sample_rate(self, temp_config_file):
        """Test sample rate retrieval."""
        config = ObservabilityConfig.from_yaml(temp_config_file)

        # Agent with tier setting
        rate = config.get_sample_rate("orchestrator")
        assert rate == 1.0

        # Agent with override
        rate = config.get_sample_rate("drift_monitor")
        assert rate == 0.2

        # Unknown agent - falls back to default
        rate = config.get_sample_rate("unknown")
        assert rate == 0.5  # default from config file

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = ObservabilityConfig()
        result = config.to_dict()

        assert "opik" in result
        assert "sampling" in result
        assert "batching" in result
        assert "circuit_breaker" in result


# ============================================================================
# SINGLETON TESTS
# ============================================================================


class TestSingleton:
    """Tests for singleton access pattern."""

    def test_get_observability_config_default(self):
        """Test getting default config."""
        config = get_observability_config()
        assert config is not None
        assert isinstance(config, ObservabilityConfig)

    def test_get_observability_config_singleton(self):
        """Test singleton behavior."""
        config1 = get_observability_config()
        config2 = get_observability_config()
        assert config1 is config2

    def test_get_observability_config_with_path(self, temp_config_file):
        """Test singleton with custom path."""
        config = get_observability_config(temp_config_file)
        assert config.opik.workspace == "test-workspace"

    def test_force_reload(self, temp_config_file):
        """Test force reload."""
        # Get initial config (defaults)
        get_observability_config()

        # Force reload with file
        config2 = get_observability_config(temp_config_file, force_reload=True)

        # Should be different due to force reload
        assert config2.opik.workspace == "test-workspace"

    def test_reset_singleton(self):
        """Test resetting singleton."""
        config1 = get_observability_config()
        reset_observability_config()
        config2 = get_observability_config()

        # Should be new instance but equal values (both defaults)
        assert config1 is not config2


# ============================================================================
# NESTED SETTINGS TESTS
# ============================================================================


class TestNestedSettings:
    """Tests for nested configuration parsing."""

    def test_retry_settings_parsing(self, temp_config_file):
        """Test retry settings are parsed correctly."""
        config = ObservabilityConfig.from_yaml(temp_config_file)
        assert config.batching.retry.max_retries == 5
        assert config.batching.retry.enabled is True

    def test_fallback_settings_parsing(self, temp_config_file):
        """Test fallback settings are parsed correctly."""
        config = ObservabilityConfig.from_yaml(temp_config_file)
        assert config.circuit_breaker.fallback.log_to_database is True
        assert config.circuit_breaker.fallback.log_to_file is True

    def test_ttl_settings_parsing(self, temp_config_file):
        """Test TTL settings are parsed correctly."""
        config = ObservabilityConfig.from_yaml(temp_config_file)
        assert config.cache.ttl.get_ttl("1h") == 30
        assert config.cache.ttl.get_ttl("24h") == 150
        assert config.cache.ttl.get_ttl("7d") == 300

    def test_agent_tiers_parsing(self, temp_config_file):
        """Test agent tiers are parsed correctly."""
        config = ObservabilityConfig.from_yaml(temp_config_file)
        assert "tier_0" in config.agent_tiers
        assert "tier_1" in config.agent_tiers
        assert "data_preparer" in config.agent_tiers["tier_0"].agents
        assert config.agent_tiers["tier_1"].detailed_logging is True


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_yaml(self):
        """Test loading empty YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = ObservabilityConfig.from_yaml(temp_path)
            # Should return defaults
            assert config.opik.enabled is True
        finally:
            os.unlink(temp_path)

    def test_partial_yaml(self):
        """Test loading partial YAML."""
        partial_config = {
            "opik": {"enabled": False},
            # Other sections missing
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name

        try:
            config = ObservabilityConfig.from_yaml(temp_path)
            assert config.opik.enabled is False
            # Other sections should have defaults
            assert config.sampling.default_rate == 1.0
        finally:
            os.unlink(temp_path)

    def test_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            config = ObservabilityConfig.from_yaml(temp_path)
            # Should return defaults on error
            assert config.opik.enabled is True
        finally:
            os.unlink(temp_path)

    def test_unknown_fields_ignored(self):
        """Test unknown fields are ignored."""
        config_with_unknown = {
            "opik": {
                "enabled": True,
                "unknown_field": "value",
                "another_unknown": 123,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_with_unknown, f)
            temp_path = f.name

        try:
            config = ObservabilityConfig.from_yaml(temp_path)
            assert config.opik.enabled is True
            # Should not raise error for unknown fields
        finally:
            os.unlink(temp_path)

    def test_config_path_stored(self, temp_config_file):
        """Test config path is stored."""
        config = ObservabilityConfig.from_yaml(temp_config_file)
        assert config._config_path == temp_config_file


# ============================================================================
# INTEGRATION WITH OPIK CONFIG
# ============================================================================


class TestOpikConfigIntegration:
    """Tests for OpikConfig.from_config_file integration."""

    def test_opik_config_from_file(self, temp_config_file):
        """Test OpikConfig loading from config file."""
        from src.mlops.opik_connector import OpikConfig

        config = OpikConfig.from_config_file(temp_config_file)
        assert config.workspace == "test-workspace"
        assert config.project_name == "test-project"
        assert config.sample_rate == 0.5
        assert config.always_sample_errors is True

    def test_opik_config_fallback_to_env(self):
        """Test OpikConfig falls back to env on import error."""
        from src.mlops.opik_connector import OpikConfig

        # When config file doesn't exist, ObservabilityConfig returns defaults
        # and doesn't raise an exception, so we get default values
        config = OpikConfig.from_config_file("/nonexistent/path.yaml")
        # Should get defaults since file was not found (no exception raised)
        assert config.workspace == "default"  # default value
        assert config.project_name == "e2i-causal-analytics"  # default value

    def test_opik_config_from_env(self):
        """Test OpikConfig.from_env directly."""
        from src.mlops.opik_connector import OpikConfig

        with patch.dict(
            os.environ,
            {"OPIK_WORKSPACE": "env-workspace", "OPIK_PROJECT_NAME": "env-project"},
        ):
            config = OpikConfig.from_env()
            assert config.workspace == "env-workspace"
            assert config.project_name == "env-project"
