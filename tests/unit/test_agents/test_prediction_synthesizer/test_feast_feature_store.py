"""Unit tests for FeastFeatureStore adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.prediction_synthesizer.nodes.feast_feature_store import (
    FeastFeatureStore,
    get_feast_feature_store,
)


@pytest.fixture
def mock_adapter():
    """Create mock FeatureAnalyzerAdapter."""
    adapter = MagicMock()
    adapter.get_feature_importance = AsyncMock(
        return_value={
            "call_frequency": 0.85,
            "prescription_count": 0.72,
            "visit_recency": 0.65,
        }
    )
    adapter.get_online_features = AsyncMock(
        return_value={
            "hcp_id": "hcp_123",
            "call_frequency": 15.0,
            "prescription_count": 42.0,
            "visit_recency": 7.0,
            "event_timestamp": "2024-12-25T00:00:00",
        }
    )
    adapter.get_online_features_batch = AsyncMock(
        return_value=[
            {"hcp_id": "hcp_001", "call_frequency": 10.0},
            {"hcp_id": "hcp_002", "call_frequency": 20.0},
        ]
    )
    adapter.check_entity_freshness = AsyncMock(
        return_value={
            "fresh": True,
            "stale_features": [],
            "last_updated": "2024-12-25T00:00:00Z",
        }
    )
    return adapter


class TestFeastFeatureStoreInit:
    """Tests for FeastFeatureStore initialization."""

    def test_init_with_adapter(self, mock_adapter):
        """Test initialization with provided adapter."""
        store = FeastFeatureStore(adapter=mock_adapter)
        assert store._adapter is mock_adapter
        assert store._default_feature_view == "hcp_features"
        assert store._entity_key == "hcp_id"

    def test_init_with_custom_config(self, mock_adapter):
        """Test initialization with custom config."""
        store = FeastFeatureStore(
            adapter=mock_adapter,
            default_feature_view="custom_view",
            entity_key="custom_id",
        )
        assert store._default_feature_view == "custom_view"
        assert store._entity_key == "custom_id"

    def test_init_without_adapter(self):
        """Test initialization without adapter - lazy loading."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = FeastFeatureStore()
            assert store._adapter is None
            assert not store.is_available

    def test_is_available_with_adapter(self, mock_adapter):
        """Test is_available returns True when adapter exists."""
        store = FeastFeatureStore(adapter=mock_adapter)
        assert store.is_available is True


class TestFeastFeatureStoreGetImportance:
    """Tests for get_importance method."""

    @pytest.mark.asyncio
    async def test_get_importance_success(self, mock_adapter):
        """Test successful feature importance retrieval."""
        store = FeastFeatureStore(adapter=mock_adapter)

        importance = await store.get_importance("churn_xgb")

        assert "call_frequency" in importance
        assert importance["call_frequency"] == 0.85
        mock_adapter.get_feature_importance.assert_called_once_with(
            model_id="churn_xgb",
            feature_view="hcp_features",
        )

    @pytest.mark.asyncio
    async def test_get_importance_no_adapter(self):
        """Test get_importance returns empty when no adapter."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = FeastFeatureStore()
            importance = await store.get_importance("model_1")
            assert importance == {}

    @pytest.mark.asyncio
    async def test_get_importance_adapter_error(self, mock_adapter):
        """Test graceful handling of adapter errors."""
        mock_adapter.get_feature_importance = AsyncMock(side_effect=Exception("Feast error"))
        store = FeastFeatureStore(adapter=mock_adapter)

        importance = await store.get_importance("model_1")
        assert importance == {}

    @pytest.mark.asyncio
    async def test_get_importance_method_not_available(self):
        """Test when adapter doesn't have get_feature_importance."""
        adapter = MagicMock(spec=[])  # No methods
        store = FeastFeatureStore(adapter=adapter)

        importance = await store.get_importance("model_1")
        assert importance == {}


class TestFeastFeatureStoreGetOnlineFeatures:
    """Tests for get_online_features method."""

    @pytest.mark.asyncio
    async def test_get_online_features_success(self, mock_adapter):
        """Test successful online feature retrieval."""
        store = FeastFeatureStore(adapter=mock_adapter)

        features = await store.get_online_features("hcp_123")

        assert "call_frequency" in features
        assert features["call_frequency"] == 15.0
        # Entity key and timestamp should be removed
        assert "hcp_id" not in features
        assert "event_timestamp" not in features

    @pytest.mark.asyncio
    async def test_get_online_features_with_refs(self, mock_adapter):
        """Test online features with specific feature refs."""
        store = FeastFeatureStore(adapter=mock_adapter)

        await store.get_online_features(
            entity_id="hcp_123",
            feature_refs=["hcp_features:call_frequency"],
        )

        mock_adapter.get_online_features.assert_called_once_with(
            entity_dict={"hcp_id": "hcp_123"},
            feature_refs=["hcp_features:call_frequency"],
        )

    @pytest.mark.asyncio
    async def test_get_online_features_custom_view(self, mock_adapter):
        """Test online features with custom feature view."""
        store = FeastFeatureStore(adapter=mock_adapter)

        await store.get_online_features(
            entity_id="hcp_123",
            feature_view="custom_view",
        )

        mock_adapter.get_online_features.assert_called_once_with(
            entity_dict={"hcp_id": "hcp_123"},
            feature_refs=["custom_view:*"],
        )

    @pytest.mark.asyncio
    async def test_get_online_features_no_adapter(self):
        """Test returns empty when no adapter."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = FeastFeatureStore()
            features = await store.get_online_features("hcp_123")
            assert features == {}

    @pytest.mark.asyncio
    async def test_get_online_features_error(self, mock_adapter):
        """Test graceful error handling."""
        mock_adapter.get_online_features = AsyncMock(side_effect=Exception("Connection error"))
        store = FeastFeatureStore(adapter=mock_adapter)

        features = await store.get_online_features("hcp_123")
        assert features == {}


class TestFeastFeatureStoreGetOnlineFeaturesBatch:
    """Tests for batch online feature retrieval."""

    @pytest.mark.asyncio
    async def test_get_online_features_batch_success(self, mock_adapter):
        """Test successful batch retrieval."""
        store = FeastFeatureStore(adapter=mock_adapter)

        result = await store.get_online_features_batch(["hcp_001", "hcp_002"])

        assert "hcp_001" in result
        assert "hcp_002" in result
        # Entity key should be removed from each result
        assert "hcp_id" not in result["hcp_001"]

    @pytest.mark.asyncio
    async def test_get_online_features_batch_no_adapter(self):
        """Test batch returns empty dicts when no adapter."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = FeastFeatureStore()
            result = await store.get_online_features_batch(["hcp_001", "hcp_002"])
            assert result == {"hcp_001": {}, "hcp_002": {}}


class TestFeastFeatureStoreCheckFreshness:
    """Tests for feature freshness checking."""

    @pytest.mark.asyncio
    async def test_check_freshness_success(self, mock_adapter):
        """Test successful freshness check."""
        store = FeastFeatureStore(adapter=mock_adapter)

        freshness = await store.check_feature_freshness("hcp_123")

        assert freshness["fresh"] is True
        assert freshness["stale_features"] == []
        mock_adapter.check_entity_freshness.assert_called_once_with(
            entity_id="hcp_123",
            entity_key="hcp_id",
            max_staleness_hours=24.0,
        )

    @pytest.mark.asyncio
    async def test_check_freshness_custom_staleness(self, mock_adapter):
        """Test freshness check with custom staleness."""
        store = FeastFeatureStore(adapter=mock_adapter)

        await store.check_feature_freshness("hcp_123", max_staleness_hours=48.0)

        mock_adapter.check_entity_freshness.assert_called_once_with(
            entity_id="hcp_123",
            entity_key="hcp_id",
            max_staleness_hours=48.0,
        )

    @pytest.mark.asyncio
    async def test_check_freshness_no_adapter(self):
        """Test freshness returns fresh when no adapter."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = FeastFeatureStore()
            freshness = await store.check_feature_freshness("hcp_123")
            assert freshness["fresh"] is True

    @pytest.mark.asyncio
    async def test_check_freshness_method_not_available(self):
        """Test when adapter doesn't have check_entity_freshness."""
        adapter = MagicMock(spec=[])
        store = FeastFeatureStore(adapter=adapter)

        freshness = await store.check_feature_freshness("hcp_123")
        assert freshness["fresh"] is True


class TestGetFeastFeatureStoreFactory:
    """Tests for factory function."""

    def test_factory_default_config(self):
        """Test factory with default config."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = get_feast_feature_store()
            assert store._default_feature_view == "hcp_features"
            assert store._entity_key == "hcp_id"

    def test_factory_custom_config(self):
        """Test factory with custom config."""
        with patch(
            "src.agents.prediction_synthesizer.nodes.feast_feature_store._get_feature_analyzer_adapter",
            return_value=None,
        ):
            store = get_feast_feature_store(
                default_feature_view="custom_view",
                entity_key="custom_id",
            )
            assert store._default_feature_view == "custom_view"
            assert store._entity_key == "custom_id"
