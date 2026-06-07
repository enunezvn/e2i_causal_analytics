"""
Unit tests for E2I Agentic Memory service factories.

Tests focus on:
- Config loading and parsing
- Factory function behavior
- Error handling for missing env vars
- Service abstraction interfaces
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

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
    OpenAILLMService,
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
        assert config.semantic.graph_name == "e2i_causal"

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
        """get_supabase_client should raise if NO Supabase key is set.

        After M9 (#703) the factory prefers a service-role key and falls back to
        anon, so a missing-key assertion must clear ALL three key vars (not just
        anon) — otherwise a service key present in the env would make the call
        succeed instead of raising.
        """
        # Save current values
        saved_url = os.environ.get("SUPABASE_URL")
        saved = {
            k: os.environ.pop(k, None)
            for k in ("SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY")
        }
        reset_all_clients()

        # Set URL but no key of any kind
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
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


# ============================================================================
# M9 (#703): backend Supabase clients authenticate as SERVICE-ROLE, not anon
# ============================================================================


class TestSupabaseClientUsesServiceRole:
    """get_supabase_client / get_async_supabase_client must PREFER the
    service-role key over the anon key.

    The backend is a trusted server-side caller. Migration 058 REVOKEs the
    anon/authenticated table+view grants, so an anon-key client would lose
    access; the backend must authenticate as service-role (bypasses RLS, retains
    full grants). Falls back to the anon key only when no service key is set
    (keeps dev/test green). Mirrors the existing get_async_supabase_service_client
    resolution: SERVICE_ROLE_KEY > SERVICE_KEY > ANON_KEY.
    """

    def setup_method(self):
        reset_all_clients()

    def teardown_method(self):
        reset_all_clients()

    def test_sync_prefers_service_key_over_anon(self, monkeypatch):
        """get_supabase_client uses SUPABASE_SERVICE_KEY when present."""
        monkeypatch.setenv("SUPABASE_URL", "https://svc.example.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key-sentinel")
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key-sentinel")
        reset_all_clients()
        with patch("supabase.create_client") as mock_create:
            mock_create.return_value = object()
            get_supabase_client()
        assert mock_create.call_args.args[1] == "svc-key-sentinel", (
            "get_supabase_client must authenticate with the service-role key, not anon"
        )

    def test_sync_prefers_service_role_key_var(self, monkeypatch):
        """SUPABASE_SERVICE_ROLE_KEY also satisfies the service-role preference."""
        monkeypatch.setenv("SUPABASE_URL", "https://svc.example.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "svc-role-sentinel")
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key-sentinel")
        reset_all_clients()
        with patch("supabase.create_client") as mock_create:
            mock_create.return_value = object()
            get_supabase_client()
        assert mock_create.call_args.args[1] == "svc-role-sentinel"

    def test_sync_falls_back_to_anon_without_service_key(self, monkeypatch):
        """Without any service key, get_supabase_client falls back to anon."""
        monkeypatch.setenv("SUPABASE_URL", "https://anon.example.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-only-sentinel")
        reset_all_clients()
        with patch("supabase.create_client") as mock_create:
            mock_create.return_value = object()
            get_supabase_client()
        assert mock_create.call_args.args[1] == "anon-only-sentinel"

    def test_async_prefers_service_key_over_anon(self, monkeypatch):
        """get_async_supabase_client uses the service-role key when present."""
        from src.memory.services.factories import get_async_supabase_client

        monkeypatch.setenv("SUPABASE_URL", "https://svc.example.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key-sentinel")
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key-sentinel")
        reset_all_clients()
        with patch("supabase.acreate_client", new_callable=AsyncMock) as mock_acreate:
            mock_acreate.return_value = object()
            asyncio.run(get_async_supabase_client())
        assert mock_acreate.call_args.args[1] == "svc-key-sentinel", (
            "get_async_supabase_client must authenticate with the service-role key, not anon"
        )

    def test_async_falls_back_to_anon_without_service_key(self, monkeypatch):
        """Without any service key, get_async_supabase_client falls back to anon."""
        from src.memory.services.factories import get_async_supabase_client

        monkeypatch.setenv("SUPABASE_URL", "https://anon.example.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
        monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-only-sentinel")
        reset_all_clients()
        with patch("supabase.acreate_client", new_callable=AsyncMock) as mock_acreate:
            mock_acreate.return_value = object()
            asyncio.run(get_async_supabase_client())
        assert mock_acreate.call_args.args[1] == "anon-only-sentinel"


# ============================================================================
# EMBEDDING SERVICE TESTS
# ============================================================================


class TestEmbeddingServices:
    """Tests for embedding service implementations."""

    def setup_method(self):
        """Reset services before each test."""
        reset_all_clients()

    def test_get_embedding_service_returns_openai_for_local(self):
        """get_embedding_service should return OpenAI for local_pilot if available."""
        pytest.importorskip("openai", reason="OpenAI package not installed")
        service = get_embedding_service("local_pilot")
        # Accept either OpenAI or Fallback depending on environment
        from src.memory.services.factories import FallbackEmbeddingService

        assert isinstance(service, (OpenAIEmbeddingService, FallbackEmbeddingService))

    def test_get_embedding_service_returns_bedrock_for_production(self):
        """get_embedding_service should return Bedrock for aws_production if available."""
        pytest.importorskip("boto3", reason="boto3 package not installed")
        reset_all_clients()
        service = get_embedding_service("aws_production")
        # Accept either Bedrock or Fallback depending on environment
        from src.memory.services.factories import FallbackEmbeddingService

        assert isinstance(service, (BedrockEmbeddingService, FallbackEmbeddingService))

    @patch.dict(os.environ, {"E2I_ENVIRONMENT": "local_pilot"})
    def test_get_embedding_service_uses_env_var(self):
        """get_embedding_service should use E2I_ENVIRONMENT env var."""
        pytest.importorskip("openai", reason="OpenAI package not installed")
        reset_all_clients()
        service = get_embedding_service()
        # Accept either OpenAI or Fallback depending on environment
        from src.memory.services.factories import FallbackEmbeddingService

        assert isinstance(service, (OpenAIEmbeddingService, FallbackEmbeddingService))

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""})
    def test_openai_service_raises_without_api_key(self):
        """OpenAI service should raise if OPENAI_API_KEY is not set."""
        service = OpenAIEmbeddingService()
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with pytest.raises(ServiceConnectionError) as exc_info:
                service._get_client()
            assert "OPENAI_API_KEY" in str(exc_info.value)


# ============================================================================
# M5a: get_embedding_service wires MemoryConfig.embeddings.model
# ============================================================================


class TestEmbeddingServiceConfigWiring:
    """get_embedding_service should read embeddings.model from config.

    Precedence: explicit-arg > env-var > config-value > hardcoded-default.
    """

    def setup_method(self):
        reset_all_clients()
        clear_config_cache()

    def teardown_method(self):
        reset_all_clients()
        clear_config_cache()

    def _model_of(self, service):
        """Resolve the concrete OpenAI embedding model from a service."""
        from src.memory.services.factories import (
            FallbackEmbeddingService,
            OpenAIEmbeddingService,
        )

        if isinstance(service, FallbackEmbeddingService):
            primary = service._get_primary()
            assert isinstance(primary, OpenAIEmbeddingService)
            return primary.model
        assert isinstance(service, OpenAIEmbeddingService)
        return service.model

    def test_uses_config_model_when_no_env_or_arg(self):
        """With no env/arg override, the config embeddings.model is used."""
        sentinel_model = "text-embedding-from-config-sentinel"

        class _Cfg:
            class embeddings:  # noqa: N801 - mimic dataclass attr access
                model = sentinel_model

        # Ensure no env override is present.
        os.environ.pop("E2I_EMBEDDING_MODEL", None)

        # _resolve_embedding_model imports get_config lazily from the config
        # module, so patch it at its source.
        with patch("src.memory.services.config.get_config", return_value=_Cfg()):
            service = get_embedding_service("local_pilot", use_fallback=False)
            assert self._model_of(service) == sentinel_model

    @patch.dict(os.environ, {"E2I_EMBEDDING_MODEL": "text-embedding-from-env"})
    def test_env_var_overrides_config(self):
        """E2I_EMBEDDING_MODEL env var takes precedence over config value."""

        class _Cfg:
            class embeddings:  # noqa: N801
                model = "text-embedding-from-config-sentinel"

        with patch("src.memory.services.config.get_config", return_value=_Cfg()):
            service = get_embedding_service("local_pilot", use_fallback=False)
            assert self._model_of(service) == "text-embedding-from-env"

    @patch.dict(os.environ, {"E2I_EMBEDDING_MODEL": "text-embedding-from-env"})
    def test_explicit_arg_overrides_env_and_config(self):
        """An explicit model= arg wins over env var and config value."""

        class _Cfg:
            class embeddings:  # noqa: N801
                model = "text-embedding-from-config-sentinel"

        with patch("src.memory.services.config.get_config", return_value=_Cfg()):
            service = get_embedding_service(
                "local_pilot", use_fallback=False, model="text-embedding-explicit-arg"
            )
            assert self._model_of(service) == "text-embedding-explicit-arg"


# ============================================================================
# M6: get_embedding_service is a cached singleton keyed on (env, use_fallback)
# ============================================================================


class TestEmbeddingServiceCaching:
    """get_embedding_service must cache instances keyed on (env, use_fallback)."""

    def setup_method(self):
        reset_all_clients()

    def teardown_method(self):
        reset_all_clients()

    def test_same_key_returns_same_instance(self):
        """Two calls with the same (env, fallback) return the SAME object."""
        a = get_embedding_service("local_pilot", use_fallback=True)
        b = get_embedding_service("local_pilot", use_fallback=True)
        assert a is b

    def test_different_key_returns_different_instance(self):
        """Different (env, fallback) keys produce distinct instances."""
        a = get_embedding_service("local_pilot", use_fallback=True)
        b = get_embedding_service("local_pilot", use_fallback=False)
        assert a is not b

    def test_reset_builds_fresh_instance(self):
        """After reset_all_clients(), a fresh instance is built."""
        a = get_embedding_service("local_pilot", use_fallback=True)
        reset_all_clients()
        b = get_embedding_service("local_pilot", use_fallback=True)
        assert a is not b


# ============================================================================
# LLM SERVICE TESTS
# ============================================================================


class TestLLMServices:
    """Tests for LLM service implementations."""

    def setup_method(self):
        """Reset services before each test."""
        reset_all_clients()

    def test_get_llm_service_returns_anthropic_for_local(self):
        """get_llm_service returns Anthropic when provider='anthropic'."""
        service = get_llm_service("local_pilot", provider="anthropic")
        assert isinstance(service, AnthropicLLMService)

    def test_get_llm_service_returns_openai_when_specified(self):
        """get_llm_service returns OpenAI when provider='openai'."""
        service = get_llm_service("local_pilot", provider="openai")
        assert isinstance(service, OpenAILLMService)

    def test_get_llm_service_returns_bedrock_for_production(self):
        """get_llm_service should return Bedrock for aws_production."""
        reset_all_clients()
        service = get_llm_service("aws_production")
        assert isinstance(service, BedrockLLMService)

    def test_get_llm_service_production_ignores_provider(self):
        """aws_production always uses Bedrock regardless of provider."""
        reset_all_clients()
        service = get_llm_service("aws_production", provider="openai")
        assert isinstance(service, BedrockLLMService)

    @patch.dict(os.environ, {"E2I_ENVIRONMENT": "local_pilot", "LLM_PROVIDER": "anthropic"})
    def test_get_llm_service_uses_env_var(self):
        """get_llm_service should use E2I_ENVIRONMENT and LLM_PROVIDER env vars."""
        reset_all_clients()
        service = get_llm_service()
        assert isinstance(service, AnthropicLLMService)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""})
    def test_anthropic_service_raises_without_api_key(self):
        """Anthropic service should raise if ANTHROPIC_API_KEY is not set."""
        pytest.importorskip("anthropic", reason="anthropic package not installed")
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
        get_llm_service("local_pilot", provider="anthropic")

        # Reset
        reset_all_clients()

        # Verify caches are cleared by getting new instances
        service1 = get_embedding_service("local_pilot")
        reset_all_clients()
        service2 = get_embedding_service("local_pilot")

        # After reset, should be different instances
        assert service1 is not service2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getter_name",
    ["get_async_supabase_client", "get_async_supabase_service_client"],
)
async def test_async_supabase_getter_concurrent_init_creates_one(monkeypatch, getter_name):
    """L14 (#694): concurrent first-time callers must create exactly ONE async
    client. Without the init lock, every coroutine passes the ``is None`` check
    across the ``await acreate_client`` and creates a duplicate (orphaned httpx
    pool). Faithful: the real getter runs; only the SDK ``acreate_client`` is
    replaced with a slow stub that yields, forcing the race. Both async getters
    share the lock, so both are exercised (codex LOW)."""
    import src.memory.services.factories as factories

    getter = getattr(factories, getter_name)

    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    reset_all_clients()

    create_calls: list = []

    async def _slow_acreate(url, key, options=None):
        create_calls.append((url, key))
        await asyncio.sleep(0)  # real suspension -> concurrent callers interleave
        return object()

    try:
        with patch("supabase.acreate_client", _slow_acreate):
            clients = await asyncio.gather(*[getter() for _ in range(20)])
        assert len(create_calls) == 1, (
            f"{getter_name}: expected exactly 1 client creation under concurrency, "
            f"got {len(create_calls)}"
        )
        # Every caller receives the same cached instance.
        assert len({id(c) for c in clients}) == 1
    finally:
        reset_all_clients()
