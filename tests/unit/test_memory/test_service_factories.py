"""
Unit tests for E2I Agentic Memory service factories.

Tests focus on:
- Config loading and parsing
- Factory function behavior
- Error handling for missing env vars
- Service abstraction interfaces
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.memory.services.config import (
    EmbeddingConfig,
    EpisodicMemoryConfig,
    LLMConfig,
    MemoryConfig,
    ProceduralMemoryConfig,
    SemanticMemoryConfig,
    WorkingMemoryConfig,
    clear_config_cache,
    get_config,
    load_memory_config,
)
from src.memory.services.factories import (
    AnthropicLLMService,
    BedrockEmbeddingService,
    BedrockLLMService,
    OpenAIEmbeddingService,
    ServiceConnectionError,
    get_embedding_service,
    get_llm_service,
    get_supabase_client,
    reset_all_clients,
)

# ============================================================================
# CONFIG TESTS
# ============================================================================


class TestConfigLoader:
    """Tests for configuration loading."""

    def setup_method(self):
        """Reset config cache before each test."""
        clear_config_cache()

    def test_load_memory_config_returns_memory_config(self):
        """load_memory_config should return a MemoryConfig instance."""
        config = load_memory_config()
        assert isinstance(config, MemoryConfig)

    def test_config_has_environment(self):
        """Config should have environment set."""
        config = load_memory_config()
        assert config.environment in ("local_pilot", "aws_production")

    def test_config_has_working_memory_config(self):
        """Config should have working memory configuration."""
        config = load_memory_config()
        assert isinstance(config.working, WorkingMemoryConfig)
        assert config.working.backend == "redis"
        assert config.working.ttl_seconds > 0

    def test_config_has_episodic_memory_config(self):
        """Config should have episodic memory configuration."""
        config = load_memory_config()
        assert isinstance(config.episodic, EpisodicMemoryConfig)
        assert config.episodic.backend == "supabase"
        assert config.episodic.table == "episodic_memories"

    def test_config_has_semantic_memory_config(self):
        """Config should have semantic memory configuration."""
        config = load_memory_config()
        assert isinstance(config.semantic, SemanticMemoryConfig)
        assert config.semantic.backend == "falkordb"
        assert config.semantic.graph_name == "e2i_semantic"

    def test_config_has_procedural_memory_config(self):
        """Config should have procedural memory configuration."""
        config = load_memory_config()
        assert isinstance(config.procedural, ProceduralMemoryConfig)
        assert config.procedural.backend == "supabase"
        assert config.procedural.table == "procedural_memories"

    def test_config_has_embedding_config(self):
        """Config should have embedding configuration."""
        config = load_memory_config()
        assert isinstance(config.embeddings, EmbeddingConfig)
        assert config.embeddings.dimensions == 1536

    def test_config_has_llm_config(self):
        """Config should have LLM configuration."""
        config = load_memory_config()
        assert isinstance(config.llm, LLMConfig)
        assert config.llm.max_tokens > 0

    def test_get_config_returns_cached_singleton(self):
        """get_config should return the same instance on multiple calls."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_clear_config_cache_allows_reload(self):
        """clear_config_cache should allow reloading config."""
        config1 = get_config()
        clear_config_cache()
        config2 = get_config()
        # Should be equal but not the same object
        assert config1.environment == config2.environment
        assert config1 is not config2

    def test_config_get_raw_returns_nested_value(self):
        """get_raw should return nested config values."""
        config = load_memory_config()
        ttl = config.get_raw("memory_backends.working.local_pilot.ttl_seconds")
        assert ttl is not None
        assert isinstance(ttl, int)

    def test_config_get_raw_returns_default_for_missing(self):
        """get_raw should return default for missing keys."""
        config = load_memory_config()
        value = config.get_raw("non.existent.path", "default_value")
        assert value == "default_value"

    def test_config_not_found_raises_error(self):
        """load_memory_config should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_memory_config(Path("/nonexistent/config.yaml"))

    @patch.dict(os.environ, {"E2I_ENVIRONMENT": "aws_production"})
    def test_environment_override_via_env_var(self):
        """Environment can be overridden via E2I_ENVIRONMENT env var."""
        clear_config_cache()
        config = load_memory_config()
        assert config.environment == "aws_production"


# ============================================================================
# FACTORY TESTS - MOCK MODE
# ============================================================================


class TestFactoriesWithMocks:
    """Tests for factory functions using mocks (no real connections)."""

    def setup_method(self):
        """Reset all clients before each test."""
        reset_all_clients()

    def teardown_method(self):
        """Reset all clients after each test."""
        reset_all_clients()

    def test_service_connection_error_has_service_attribute(self):
        """ServiceConnectionError should store the service name."""
        error = ServiceConnectionError("TestService", "Test message")
        assert error.service == "TestService"

    def test_supabase_client_raises_without_url(self):
        """get_supabase_client should raise if SUPABASE_URL is not set."""
        # Save current values
        saved_url = os.environ.pop("SUPABASE_URL", None)
        saved_key = os.environ.pop("SUPABASE_ANON_KEY", None)
        reset_all_clients()

        try:
            with pytest.raises(ServiceConnectionError) as exc_info:
                get_supabase_client()
            assert "SUPABASE_URL" in str(exc_info.value)
        finally:
            # Restore values
            if saved_url:
                os.environ["SUPABASE_URL"] = saved_url
            if saved_key:
                os.environ["SUPABASE_ANON_KEY"] = saved_key

    def test_supabase_client_raises_without_key(self):
        """get_supabase_client should raise if SUPABASE_ANON_KEY is not set."""
        # Save current values
        saved_url = os.environ.get("SUPABASE_URL")
        saved_key = os.environ.pop("SUPABASE_ANON_KEY", None)
        reset_all_clients()

        # Set URL but not key
        os.environ["SUPABASE_URL"] = "https://example.supabase.co"

        try:
            with pytest.raises(ServiceConnectionError) as exc_info:
                get_supabase_client()
            assert "SUPABASE_ANON_KEY" in str(exc_info.value)
        finally:
            # Restore values
            if saved_url:
                os.environ["SUPABASE_URL"] = saved_url
            elif "SUPABASE_URL" in os.environ:
                del os.environ["SUPABASE_URL"]
            if saved_key:
                os.environ["SUPABASE_ANON_KEY"] = saved_key


# ============================================================================
# EMBEDDING SERVICE TESTS
# ============================================================================


class TestEmbeddingServices:
    """Tests for embedding service implementations."""

    def setup_method(self):
        """Reset services before each test."""
        reset_all_clients()

    def test_get_embedding_service_returns_openai_for_local(self):
        """get_embedding_service should return OpenAI for local_pilot."""
        service = get_embedding_service("local_pilot")
        assert isinstance(service, OpenAIEmbeddingService)

    def test_get_embedding_service_returns_bedrock_for_production(self):
        """get_embedding_service should return Bedrock for aws_production."""
        reset_all_clients()
        service = get_embedding_service("aws_production")
        assert isinstance(service, BedrockEmbeddingService)

    @patch.dict(os.environ, {"E2I_ENVIRONMENT": "local_pilot"})
    def test_get_embedding_service_uses_env_var(self):
        """get_embedding_service should use E2I_ENVIRONMENT env var."""
        reset_all_clients()
        service = get_embedding_service()
        assert isinstance(service, OpenAIEmbeddingService)

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_openai_service_raises_without_api_key(self):
        """OpenAI service should raise if OPENAI_API_KEY is not set."""
        service = OpenAIEmbeddingService()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with pytest.raises(ServiceConnectionError) as exc_info:
                service._get_client()
            assert "OPENAI_API_KEY" in str(exc_info.value)


# ============================================================================
# LLM SERVICE TESTS
# ============================================================================


class TestLLMServices:
    """Tests for LLM service implementations."""

    def setup_method(self):
        """Reset services before each test."""
        reset_all_clients()

    def test_get_llm_service_returns_anthropic_for_local(self):
        """get_llm_service should return Anthropic for local_pilot."""
        service = get_llm_service("local_pilot")
        assert isinstance(service, AnthropicLLMService)

    def test_get_llm_service_returns_bedrock_for_production(self):
        """get_llm_service should return Bedrock for aws_production."""
        reset_all_clients()
        service = get_llm_service("aws_production")
        assert isinstance(service, BedrockLLMService)

    @patch.dict(os.environ, {"E2I_ENVIRONMENT": "local_pilot"})
    def test_get_llm_service_uses_env_var(self):
        """get_llm_service should use E2I_ENVIRONMENT env var."""
        reset_all_clients()
        service = get_llm_service()
        assert isinstance(service, AnthropicLLMService)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""})
    def test_anthropic_service_raises_without_api_key(self):
        """Anthropic service should raise if ANTHROPIC_API_KEY is not set."""
        service = AnthropicLLMService()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with pytest.raises(ServiceConnectionError) as exc_info:
                service._get_client()
            assert "ANTHROPIC_API_KEY" in str(exc_info.value)


# ============================================================================
# SERVICE CONNECTION ERROR TESTS
# ============================================================================


class TestServiceConnectionError:
    """Tests for ServiceConnectionError."""

    def test_service_connection_error_contains_service_name(self):
        """ServiceConnectionError should include service name."""
        error = ServiceConnectionError("Redis", "Connection failed")
        assert error.service == "Redis"
        assert "Redis" in str(error)

    def test_service_connection_error_contains_message(self):
        """ServiceConnectionError should include error message."""
        error = ServiceConnectionError("Redis", "Connection failed")
        assert "Connection failed" in str(error)

    def test_service_connection_error_preserves_original(self):
        """ServiceConnectionError should preserve original exception."""
        original = ValueError("Original error")
        error = ServiceConnectionError("Redis", "Connection failed", original)
        assert error.original_error is original


# ============================================================================
# RESET FUNCTION TESTS
# ============================================================================


class TestResetFunctions:
    """Tests for client reset functionality."""

    def test_reset_all_clients_clears_caches(self):
        """reset_all_clients should clear all cached clients."""
        # Get services to populate caches
        get_embedding_service("local_pilot")
        get_llm_service("local_pilot")

        # Reset
        reset_all_clients()

        # Verify caches are cleared by getting new instances
        service1 = get_embedding_service("local_pilot")
        reset_all_clients()
        service2 = get_embedding_service("local_pilot")

        # After reset, should be different instances
        assert service1 is not service2
