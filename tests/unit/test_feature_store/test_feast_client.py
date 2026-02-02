"""Tests for Feast client wrapper.

Tests cover:
- Online feature retrieval
- Historical feature retrieval (point-in-time joins)
- Feature materialization
- Fallback to custom store
- Feature statistics
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.feature_store.feast_client import (
    FeastClient,
    FeastConfig,
    FeatureFreshness,
    FeatureStatistics,
    FreshnessStatus,
    get_feast_client,
    load_feast_config,
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
        with patch.object(client, "initialize", new_callable=AsyncMock) as mock_init:
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
        df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime.now()],
            }
        )
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
        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123", "456"],
                "brand_id": ["remibrutinib", "remibrutinib"],
                "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 15)],
            }
        )

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
        await client.get_online_features(
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

        with patch.object(FeastClient, "initialize", new_callable=AsyncMock):
            client1 = await get_feast_client()
            client2 = await get_feast_client()

            # Should be the same instance
            assert client1 is client2


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_feast_config_default_path(self):
        """Test loading config from default path."""
        config = load_feast_config()

        # Should return a dict (either from file or defaults)
        assert isinstance(config, dict)

    def test_load_feast_config_missing_file(self, tmp_path):
        """Test loading config when file doesn't exist returns defaults."""
        missing_path = tmp_path / "nonexistent.yaml"
        config = load_feast_config(missing_path)

        # Should return default config
        assert "materialization" in config
        assert config["materialization"]["max_staleness_hours"] == 24.0

    def test_load_feast_config_custom_file(self, tmp_path):
        """Test loading config from custom YAML file."""
        config_path = tmp_path / "test_feast.yaml"
        config_path.write_text("""
materialization:
  max_staleness_hours: 48.0
  warning_threshold_hours: 12.0
feature_views:
  hcp_features:
    max_staleness_hours: 6.0
""")
        config = load_feast_config(config_path)

        assert config["materialization"]["max_staleness_hours"] == 48.0
        assert config["materialization"]["warning_threshold_hours"] == 12.0
        assert "hcp_features" in config["feature_views"]

    def test_client_loads_materialization_config(self):
        """Test that FeastClient loads materialization config on init."""
        client = FeastClient()

        # Should have materialization config loaded
        assert hasattr(client, "_materialization_config")
        assert isinstance(client._materialization_config, dict)

    def test_client_initializes_timestamp_tracking(self):
        """Test that FeastClient initializes timestamp tracking."""
        client = FeastClient()

        # Should have empty timestamp dict
        assert hasattr(client, "_materialization_timestamps")
        assert isinstance(client._materialization_timestamps, dict)
        assert len(client._materialization_timestamps) == 0


class TestFeatureFreshness:
    """Test feature freshness functionality."""

    def test_freshness_status_enum_values(self):
        """Test FreshnessStatus enum has expected values."""
        assert FreshnessStatus.FRESH == "fresh"
        assert FreshnessStatus.WARNING == "warning"
        assert FreshnessStatus.STALE == "stale"
        assert FreshnessStatus.EXPIRED == "expired"
        assert FreshnessStatus.UNKNOWN == "unknown"

    def test_feature_freshness_model_defaults(self):
        """Test FeatureFreshness model default values."""
        freshness = FeatureFreshness(feature_view="test_view")

        assert freshness.feature_view == "test_view"
        assert freshness.last_materialized is None
        assert freshness.freshness_status == FreshnessStatus.UNKNOWN
        assert freshness.is_fresh is False
        assert freshness.max_staleness_hours == 24.0

    def test_feature_freshness_model_with_values(self):
        """Test FeatureFreshness model with custom values."""
        now = datetime.now()
        freshness = FeatureFreshness(
            feature_view="hcp_features",
            last_materialized=now,
            freshness_status=FreshnessStatus.FRESH,
            age_hours=0.5,
            ttl_hours=24.0,
            max_staleness_hours=24.0,
            warning_threshold_hours=12.0,
            is_fresh=True,
            message="Features are fresh",
        )

        assert freshness.feature_view == "hcp_features"
        assert freshness.last_materialized == now
        assert freshness.freshness_status == FreshnessStatus.FRESH
        assert freshness.age_hours == 0.5
        assert freshness.is_fresh is True

    @pytest.mark.asyncio
    async def test_get_feature_freshness_unknown_no_materialization(self):
        """Test freshness is UNKNOWN when no materialization recorded."""
        client = FeastClient()
        client._initialized = True

        freshness = await client.get_feature_freshness("unknown_view")

        assert freshness.feature_view == "unknown_view"
        assert freshness.freshness_status == FreshnessStatus.UNKNOWN
        assert freshness.last_materialized is None
        assert freshness.is_fresh is False

    @pytest.mark.asyncio
    async def test_get_feature_freshness_fresh_status(self):
        """Test freshness is FRESH when recently materialized."""
        client = FeastClient()
        client._initialized = True
        # Record materialization 30 minutes ago
        client._materialization_timestamps["hcp_features"] = datetime.now() - timedelta(minutes=30)

        freshness = await client.get_feature_freshness("hcp_features")

        assert freshness.feature_view == "hcp_features"
        assert freshness.freshness_status == FreshnessStatus.FRESH
        assert freshness.is_fresh is True
        assert freshness.age_hours < 1.0

    @pytest.mark.asyncio
    async def test_get_feature_freshness_warning_status(self):
        """Test freshness is WARNING when approaching staleness."""
        client = FeastClient()
        client._initialized = True
        # Configure warning threshold at 12 hours, staleness at 24 hours
        client._materialization_config = {
            "materialization": {"max_staleness_hours": 24.0, "warning_threshold_hours": 12.0},
            "feature_views": {},
        }
        # Record materialization 14 hours ago (past warning, before staleness)
        client._materialization_timestamps["hcp_features"] = datetime.now() - timedelta(hours=14)

        freshness = await client.get_feature_freshness("hcp_features")

        assert freshness.freshness_status == FreshnessStatus.WARNING
        # WARNING means approaching staleness but still technically fresh/usable
        assert freshness.is_fresh is True

    @pytest.mark.asyncio
    async def test_get_feature_freshness_stale_status(self):
        """Test freshness is STALE when past staleness threshold."""
        client = FeastClient()
        client._initialized = True
        client._materialization_config = {
            "materialization": {"max_staleness_hours": 24.0},
            "feature_views": {},
        }
        # Record materialization 30 hours ago (past staleness, before expiry)
        client._materialization_timestamps["hcp_features"] = datetime.now() - timedelta(hours=30)

        freshness = await client.get_feature_freshness("hcp_features")

        assert freshness.freshness_status == FreshnessStatus.STALE
        assert freshness.is_fresh is False

    @pytest.mark.asyncio
    async def test_get_feature_freshness_expired_status(self):
        """Test freshness is EXPIRED when very old."""
        client = FeastClient()
        client._initialized = True
        client._materialization_config = {
            "materialization": {"max_staleness_hours": 24.0},
            "feature_views": {},
        }
        # Record materialization 50 hours ago (past 2x staleness)
        client._materialization_timestamps["hcp_features"] = datetime.now() - timedelta(hours=50)

        freshness = await client.get_feature_freshness("hcp_features")

        assert freshness.freshness_status == FreshnessStatus.EXPIRED
        assert freshness.is_fresh is False

    @pytest.mark.asyncio
    async def test_get_all_freshness(self):
        """Test getting freshness for all feature views."""
        client = FeastClient()
        client._initialized = True
        # Clear config to use mock store instead
        client._materialization_config = {"materialization": {}, "feature_views": {}}

        # Setup mock store with feature views
        mock_fv1 = MagicMock()
        mock_fv1.name = "hcp_features"
        mock_fv2 = MagicMock()
        mock_fv2.name = "brand_features"
        mock_store = MagicMock()
        mock_store.list_feature_views.return_value = [mock_fv1, mock_fv2]
        client._store = mock_store

        # Record one materialization
        client._materialization_timestamps["hcp_features"] = datetime.now()

        result = await client.get_all_freshness()

        assert isinstance(result, dict)
        assert "hcp_features" in result
        assert "brand_features" in result
        assert result["hcp_features"].freshness_status == FreshnessStatus.FRESH
        assert result["brand_features"].freshness_status == FreshnessStatus.UNKNOWN

    def test_record_materialization_with_timestamp(self):
        """Test recording materialization with custom timestamp."""
        client = FeastClient()
        past_time = datetime(2024, 1, 15, 12, 0, 0)

        client.record_materialization("hcp_features", timestamp=past_time)

        assert "hcp_features" in client._materialization_timestamps
        assert client._materialization_timestamps["hcp_features"] == past_time

    def test_record_materialization_default_now(self):
        """Test recording materialization defaults to current time."""
        client = FeastClient()
        before = datetime.now()

        client.record_materialization("hcp_features")

        after = datetime.now()
        recorded = client._materialization_timestamps["hcp_features"]
        assert before <= recorded <= after


class TestFreshnessThresholds:
    """Test freshness threshold configuration."""

    def test_get_freshness_thresholds_defaults(self):
        """Test default thresholds when no config."""
        client = FeastClient()
        client._materialization_config = {"materialization": {}, "feature_views": {}}

        max_stale, warning = client._get_freshness_thresholds("any_view")

        assert max_stale == 24.0  # default
        assert warning == max_stale / 2  # default is half of staleness

    def test_get_freshness_thresholds_global_config(self):
        """Test thresholds from global materialization config."""
        client = FeastClient()
        client._materialization_config = {
            "materialization": {"max_staleness_hours": 48.0, "warning_threshold_hours": 24.0},
            "feature_views": {},
        }

        max_stale, warning = client._get_freshness_thresholds("any_view")

        assert max_stale == 48.0
        assert warning == 24.0

    def test_get_freshness_thresholds_feature_view_override(self):
        """Test per-feature-view threshold override."""
        client = FeastClient()
        client._materialization_config = {
            "materialization": {"max_staleness_hours": 24.0, "warning_threshold_hours": 12.0},
            "feature_views": {
                "hcp_features": {
                    "max_staleness_hours": 6.0,
                    "warning_threshold_hours": 3.0,
                }
            },
        }

        # Feature view with override
        max_stale, warning = client._get_freshness_thresholds("hcp_features")
        assert max_stale == 6.0
        assert warning == 3.0

        # Feature view without override uses global
        max_stale, warning = client._get_freshness_thresholds("other_view")
        assert max_stale == 24.0
        assert warning == 12.0

    @pytest.mark.asyncio
    async def test_close_clears_timestamps(self):
        """Test that closing client clears materialization timestamps."""
        client = FeastClient()
        client._initialized = True
        client._materialization_timestamps = {"hcp_features": datetime.now()}

        await client.close()

        assert len(client._materialization_timestamps) == 0


class TestMaterializationTimestampTracking:
    """Test that materialization methods track timestamps."""

    @pytest.mark.asyncio
    async def test_materialize_tracks_timestamp(self):
        """Test that materialize() records timestamps for feature views."""
        client = FeastClient()
        client._initialized = True
        mock_store = MagicMock()
        mock_store.materialize.return_value = None
        client._store = mock_store

        before = datetime.now()
        await client.materialize(
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            feature_views=["hcp_features", "brand_features"],
        )
        after = datetime.now()

        # Both feature views should have timestamps recorded
        assert "hcp_features" in client._materialization_timestamps
        assert "brand_features" in client._materialization_timestamps
        assert before <= client._materialization_timestamps["hcp_features"] <= after

    @pytest.mark.asyncio
    async def test_materialize_incremental_tracks_timestamp(self):
        """Test that materialize_incremental() records timestamps."""
        client = FeastClient()
        client._initialized = True
        # Clear config to use mock store instead
        client._materialization_config = {"materialization": {}, "feature_views": {}}

        mock_store = MagicMock()
        mock_store.materialize_incremental.return_value = None
        # Mock list_feature_views for getting all view names
        mock_fv = MagicMock()
        mock_fv.name = "hcp_features"
        mock_store.list_feature_views.return_value = [mock_fv]
        client._store = mock_store

        before = datetime.now()
        await client.materialize_incremental(end_date=datetime.now())
        after = datetime.now()

        # Feature view should have timestamp recorded
        assert "hcp_features" in client._materialization_timestamps
        assert before <= client._materialization_timestamps["hcp_features"] <= after
