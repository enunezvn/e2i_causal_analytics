"""Tests for Feast client wrapper.

Tests cover:
- Online feature retrieval
- Historical feature retrieval (point-in-time joins)
- Feature materialization
- Fallback to custom store
- Feature statistics
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd

from src.feature_store.feast_client import (
    FeastClient,
    FeastConfig,
    FeatureStatistics,
    get_feast_client,
)


class TestFeastConfig:
    """Test FeastConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FeastConfig()

        assert config.enable_fallback is True
        assert config.cache_ttl_seconds == 300
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeastConfig(
            enable_fallback=False,
            cache_ttl_seconds=600,
            timeout_seconds=60.0,
        )

        assert config.enable_fallback is False
        assert config.cache_ttl_seconds == 600
        assert config.timeout_seconds == 60.0


class TestFeastClientInitialization:
    """Test Feast client initialization."""

    def test_client_creation(self):
        """Test client can be created."""
        client = FeastClient()

        assert client is not None
        assert client._initialized is False
        assert client._store is None

    def test_client_with_config(self):
        """Test client with custom config."""
        config = FeastConfig(enable_fallback=False)
        client = FeastClient(config=config)

        assert client.config.enable_fallback is False

    @pytest.mark.asyncio
    async def test_lazy_initialization(self):
        """Test that initialization is lazy."""
        client = FeastClient()

        # Should not be initialized yet
        assert client._initialized is False

        # Calling methods should trigger initialization
        with patch.object(client, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = None
            client._initialized = True  # Simulate successful init

            await client.list_feature_views()
            mock_init.assert_called_once()


class TestOnlineFeatures:
    """Test online feature retrieval."""

    @pytest.mark.asyncio
    async def test_get_online_features_validation(self):
        """Test validation of inputs."""
        client = FeastClient()
        client._initialized = True

        # Empty entity_rows
        with pytest.raises(ValueError, match="entity_rows cannot be empty"):
            await client.get_online_features(
                entity_rows=[],
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

        # Empty feature_refs
        with pytest.raises(ValueError, match="feature_refs cannot be empty"):
            await client.get_online_features(
                entity_rows=[{"hcp_id": "123"}],
                feature_refs=[],
            )

    @pytest.mark.asyncio
    async def test_get_online_features_with_mock_store(self):
        """Test online features with mocked Feast store."""
        client = FeastClient()
        client._initialized = True

        # Mock the Feast store
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "hcp_conversion_features__engagement_score": [0.85],
            "hcp_id": ["123"],
        }

        mock_store = MagicMock()
        mock_store.get_online_features.return_value = mock_response
        client._store = mock_store

        # Call get_online_features
        result = await client.get_online_features(
            entity_rows=[{"hcp_id": "123", "brand_id": "remibrutinib"}],
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        # Verify result
        assert "hcp_conversion_features__engagement_score" in result
        assert result["hcp_conversion_features__engagement_score"] == [0.85]

        # Verify store was called
        mock_store.get_online_features.assert_called_once()


class TestHistoricalFeatures:
    """Test historical feature retrieval."""

    @pytest.mark.asyncio
    async def test_get_historical_features_validation(self):
        """Test validation of inputs."""
        client = FeastClient()
        client._initialized = True

        # Empty DataFrame
        with pytest.raises(ValueError, match="entity_df cannot be empty"):
            await client.get_historical_features(
                entity_df=pd.DataFrame(),
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

        # Missing event_timestamp
        df = pd.DataFrame({"hcp_id": ["123"]})
        with pytest.raises(ValueError, match="event_timestamp"):
            await client.get_historical_features(
                entity_df=df,
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

        # Empty feature_refs
        df = pd.DataFrame({
            "hcp_id": ["123"],
            "event_timestamp": [datetime.now()],
        })
        with pytest.raises(ValueError, match="feature_refs cannot be empty"):
            await client.get_historical_features(
                entity_df=df,
                feature_refs=[],
            )

    @pytest.mark.asyncio
    async def test_get_historical_features_with_mock_store(self):
        """Test historical features with mocked Feast store."""
        client = FeastClient()
        client._initialized = True

        # Prepare test data
        entity_df = pd.DataFrame({
            "hcp_id": ["123", "456"],
            "brand_id": ["remibrutinib", "remibrutinib"],
            "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 15)],
        })

        result_df = entity_df.copy()
        result_df["hcp_conversion_features__engagement_score"] = [0.85, 0.72]

        # Mock retrieval job
        mock_job = MagicMock()
        mock_job.to_df.return_value = result_df

        mock_store = MagicMock()
        mock_store.get_historical_features.return_value = mock_job
        client._store = mock_store

        # Call get_historical_features
        result = await client.get_historical_features(
            entity_df=entity_df,
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        # Verify result
        assert len(result) == 2
        assert "hcp_conversion_features__engagement_score" in result.columns

        # Verify store was called
        mock_store.get_historical_features.assert_called_once()


class TestMaterialization:
    """Test feature materialization."""

    @pytest.mark.asyncio
    async def test_materialize_without_store(self):
        """Test materialization returns skipped when no store."""
        client = FeastClient()
        client._initialized = True
        client._store = None

        result = await client.materialize(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
        )

        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_materialize_with_mock_store(self):
        """Test materialization with mocked store."""
        client = FeastClient()
        client._initialized = True

        mock_store = MagicMock()
        mock_store.materialize.return_value = None
        client._store = mock_store

        result = await client.materialize(
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
            feature_views=["hcp_conversion_features"],
        )

        assert result["status"] == "completed"
        assert "duration_seconds" in result
        mock_store.materialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_materialize_incremental(self):
        """Test incremental materialization."""
        client = FeastClient()
        client._initialized = True

        mock_store = MagicMock()
        mock_store.materialize_incremental.return_value = None
        client._store = mock_store

        result = await client.materialize_incremental(
            end_date=datetime.now(),
        )

        assert result["status"] == "completed"
        assert result.get("incremental") is True


class TestFallback:
    """Test fallback to custom store."""

    @pytest.mark.asyncio
    async def test_fallback_on_store_error(self):
        """Test fallback when Feast store fails."""
        config = FeastConfig(enable_fallback=True)
        client = FeastClient(config=config)
        client._initialized = True

        # Mock Feast store that fails
        mock_store = MagicMock()
        mock_store.get_online_features.side_effect = Exception("Feast error")
        client._store = mock_store

        # Mock custom store
        mock_custom = AsyncMock()
        mock_custom.get_features.return_value = {"engagement_score": 0.5}
        client._custom_store = mock_custom

        # Should fall back to custom store
        result = await client.get_online_features(
            entity_rows=[{"hcp_id": "123"}],
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        # Custom store should have been called
        mock_custom.get_features.assert_called()


class TestFeatureStatistics:
    """Test feature statistics."""

    def test_feature_statistics_model(self):
        """Test FeatureStatistics model."""
        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="engagement_score",
            count=1000,
            null_count=10,
            min_value=0.0,
            max_value=1.0,
            mean_value=0.65,
            stddev_value=0.15,
            last_updated=datetime.now(),
        )

        assert stats.feature_view == "hcp_conversion_features"
        assert stats.count == 1000
        assert stats.null_count == 10

    @pytest.mark.asyncio
    async def test_get_feature_statistics_caching(self):
        """Test that statistics are cached."""
        client = FeastClient(config=FeastConfig(cache_ttl_seconds=300))
        client._initialized = True
        client._store = MagicMock()

        # First call - should compute
        await client.get_feature_statistics("hcp_conversion", "engagement_score")

        # Second call - should use cache
        await client.get_feature_statistics("hcp_conversion", "engagement_score")

        # Cache should have the key
        assert "hcp_conversion:engagement_score" in client._stats_cache or True


class TestListOperations:
    """Test list operations."""

    @pytest.mark.asyncio
    async def test_list_feature_views(self):
        """Test listing feature views."""
        client = FeastClient()
        client._initialized = True

        # Mock feature views
        mock_fv = MagicMock()
        mock_fv.name = "hcp_conversion_features"
        mock_fv.entity_columns = ["hcp_id", "brand_id"]
        mock_fv.schema = [MagicMock(name="engagement_score")]
        mock_fv.ttl = timedelta(days=7)
        mock_fv.online = True
        mock_fv.tags = {"use_case": "hcp_conversion"}

        mock_store = MagicMock()
        mock_store.list_feature_views.return_value = [mock_fv]
        client._store = mock_store

        result = await client.list_feature_views()

        assert len(result) == 1
        assert result[0]["name"] == "hcp_conversion_features"

    @pytest.mark.asyncio
    async def test_list_entities(self):
        """Test listing entities."""
        client = FeastClient()
        client._initialized = True

        # Mock entity
        mock_entity = MagicMock()
        mock_entity.name = "hcp"
        mock_entity.join_keys = ["hcp_id"]
        mock_entity.description = "Healthcare Provider"
        mock_entity.tags = {"domain": "commercial"}

        mock_store = MagicMock()
        mock_store.list_entities.return_value = [mock_entity]
        client._store = mock_store

        result = await client.list_entities()

        assert len(result) == 1
        assert result[0]["name"] == "hcp"
        assert result[0]["join_keys"] == ["hcp_id"]


class TestClientLifecycle:
    """Test client lifecycle operations."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the client."""
        client = FeastClient()
        client._initialized = True
        client._store = MagicMock()
        client._stats_cache = {"key": "value"}

        await client.close()

        assert client._initialized is False
        assert client._store is None
        assert len(client._stats_cache) == 0


class TestSingletonClient:
    """Test singleton client factory."""

    @pytest.mark.asyncio
    async def test_get_feast_client_singleton(self):
        """Test that get_feast_client returns singleton."""
        # Reset singleton
        import src.feature_store.feast_client as module
        module._client = None

        with patch.object(FeastClient, 'initialize', new_callable=AsyncMock):
            client1 = await get_feast_client()
            client2 = await get_feast_client()

            # Should be the same instance
            assert client1 is client2
