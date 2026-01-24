"""
Unit tests for embedding service fallback functionality.

Tests:
- LocalEmbeddingService
- FallbackEmbeddingService
- get_embedding_service() factory with fallback options
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.services.factories import (
    EmbeddingService,
    FallbackEmbeddingService,
    LocalEmbeddingService,
    OpenAIEmbeddingService,
    ServiceConnectionError,
    get_embedding_service,
    reset_all_clients,
)


class TestLocalEmbeddingService:
    """Tests for LocalEmbeddingService."""

    def setup_method(self):
        """Reset clients before each test."""
        reset_all_clients()

    @pytest.mark.asyncio
    async def test_embed_returns_list(self):
        """Test that embed returns a list of floats."""
        with patch("src.memory.services.factories.LocalEmbeddingService._get_model") as mock_model:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = MagicMock(
                tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
            )
            mock_model.return_value = mock_encoder

            service = LocalEmbeddingService()
            result = await service.embed("test text")

            assert isinstance(result, list)
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_caches_results(self):
        """Test that repeated calls use cache."""
        with patch("src.memory.services.factories.LocalEmbeddingService._get_model") as mock_model:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = MagicMock(
                tolist=MagicMock(return_value=[0.1, 0.2, 0.3])
            )
            mock_model.return_value = mock_encoder

            service = LocalEmbeddingService()

            # First call
            result1 = await service.embed("test text")
            # Second call (should use cache)
            result2 = await service.embed("test text")

            assert result1 == result2
            # encode should only be called once due to caching
            assert mock_encoder.encode.call_count == 1

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        with patch("src.memory.services.factories.LocalEmbeddingService._get_model") as mock_model:
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = MagicMock(
                tolist=MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
            )
            mock_model.return_value = mock_encoder

            service = LocalEmbeddingService()
            results = await service.embed_batch(["text1", "text2"])

            assert isinstance(results, list)
            assert len(results) == 2


class TestFallbackEmbeddingService:
    """Tests for FallbackEmbeddingService."""

    def setup_method(self):
        """Reset clients before each test."""
        reset_all_clients()

    @pytest.mark.asyncio
    async def test_uses_primary_when_available(self):
        """Test that primary service is used when available."""
        service = FallbackEmbeddingService(environment="local_pilot")

        with patch.object(OpenAIEmbeddingService, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]

            result = await service.embed("test")

            assert result == [0.1, 0.2, 0.3]
            assert not service.is_using_fallback

    @pytest.mark.asyncio
    async def test_falls_back_on_primary_failure(self):
        """Test fallback activation when primary fails."""
        service = FallbackEmbeddingService(environment="local_pilot")

        with patch.object(
            OpenAIEmbeddingService, "embed", new_callable=AsyncMock
        ) as mock_primary:
            mock_primary.side_effect = ServiceConnectionError("OpenAI", "API error")

            with patch.object(
                LocalEmbeddingService, "embed", new_callable=AsyncMock
            ) as mock_fallback:
                mock_fallback.return_value = [0.4, 0.5, 0.6]

                result = await service.embed("test")

                assert result == [0.4, 0.5, 0.6]
                assert service.is_using_fallback

    @pytest.mark.asyncio
    async def test_retries_primary_after_interval(self):
        """Test that primary is retried after retry interval."""
        service = FallbackEmbeddingService(environment="local_pilot")
        service._primary_retry_interval = 0.1  # Short interval for testing

        # First call fails, activates fallback
        with patch.object(
            OpenAIEmbeddingService, "embed", new_callable=AsyncMock
        ) as mock_primary:
            mock_primary.side_effect = ServiceConnectionError("OpenAI", "API error")

            with patch.object(
                LocalEmbeddingService, "embed", new_callable=AsyncMock
            ) as mock_fallback:
                mock_fallback.return_value = [0.1, 0.2]
                await service.embed("test")
                assert service.is_using_fallback

        # Wait for retry interval
        import asyncio
        await asyncio.sleep(0.15)

        # Second call should try primary again
        with patch.object(
            OpenAIEmbeddingService, "embed", new_callable=AsyncMock
        ) as mock_primary:
            mock_primary.return_value = [0.3, 0.4]
            result = await service.embed("test2")

            assert result == [0.3, 0.4]
            assert not service.is_using_fallback

    def test_status_property(self):
        """Test status property returns expected fields."""
        service = FallbackEmbeddingService(environment="local_pilot")
        status = service.status

        assert "using_fallback" in status
        assert "fallback_duration_seconds" in status
        assert "environment" in status
        assert status["environment"] == "local_pilot"


class TestGetEmbeddingService:
    """Tests for get_embedding_service factory."""

    def setup_method(self):
        """Reset clients before each test."""
        reset_all_clients()

    def test_returns_fallback_service_by_default(self):
        """Test that fallback is enabled by default."""
        with patch.dict(os.environ, {"E2I_EMBEDDING_FALLBACK": "true"}, clear=False):
            service = get_embedding_service()
            assert isinstance(service, FallbackEmbeddingService)

    def test_returns_openai_when_fallback_disabled(self):
        """Test that OpenAI is returned when fallback disabled."""
        service = get_embedding_service(use_fallback=False, environment="local_pilot")
        assert isinstance(service, OpenAIEmbeddingService)

    def test_respects_environment_variable(self):
        """Test that E2I_EMBEDDING_FALLBACK env var is respected."""
        with patch.dict(os.environ, {"E2I_EMBEDDING_FALLBACK": "false"}, clear=False):
            service = get_embedding_service()
            assert not isinstance(service, FallbackEmbeddingService)

    def test_explicit_parameter_overrides_env(self):
        """Test that explicit parameter overrides environment."""
        with patch.dict(os.environ, {"E2I_EMBEDDING_FALLBACK": "false"}, clear=False):
            service = get_embedding_service(use_fallback=True)
            assert isinstance(service, FallbackEmbeddingService)
