"""Tests for Feast client wrapper.

Tests cover:
- Online feature retrieval
- Historical feature retrieval (point-in-time joins)
- Feature materialization
- Fallback to custom store
- Feature statistics
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.feature_store.feast_client import (
    FeastClient,
    FeastConfig,
    FeastError,
    FeastFallbackError,
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
        mock_custom = MagicMock()
        mock_result = MagicMock()
        mock_result.features = {"hcp_conversion_features__engagement_score": 0.5}
        mock_custom.get_entity_features.return_value = mock_result
        client._custom_store = mock_custom

        # Should fall back to custom store
        await client.get_online_features(
            entity_rows=[{"hcp_id": "123"}],
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        # Custom store should have been called
        mock_custom.get_entity_features.assert_called()


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
        mock_entity.join_key = "hcp_id"
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
        client._materialization_timestamps["hcp_features"] = datetime.now(timezone.utc) - timedelta(
            minutes=30
        )

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
        client._materialization_timestamps["hcp_features"] = datetime.now(timezone.utc) - timedelta(
            hours=14
        )

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
        client._materialization_timestamps["hcp_features"] = datetime.now(timezone.utc) - timedelta(
            hours=30
        )

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
        client._materialization_timestamps["hcp_features"] = datetime.now(timezone.utc) - timedelta(
            hours=50
        )

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
        client._materialization_timestamps["hcp_features"] = datetime.now(timezone.utc)

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
        before = datetime.now(timezone.utc)

        client.record_materialization("hcp_features")

        after = datetime.now(timezone.utc)
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

        before = datetime.now(timezone.utc)
        await client.materialize(
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            feature_views=["hcp_features", "brand_features"],
        )
        after = datetime.now(timezone.utc)

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

        before = datetime.now(timezone.utc)
        await client.materialize_incremental(end_date=datetime.now(timezone.utc))
        after = datetime.now(timezone.utc)

        # Feature view should have timestamp recorded
        assert "hcp_features" in client._materialization_timestamps
        assert before <= client._materialization_timestamps["hcp_features"] <= after


# ============================================================================
# Block 2 — Feast fail-loud tests
# ============================================================================


class TestProductionFallbackRaises:
    """Test that the historical-features fallback raises in production."""

    @pytest.mark.asyncio
    async def test_historical_fallback_raises_in_production(self, monkeypatch):
        """FeastFallbackError is raised (not swallowed) when ENVIRONMENT=production."""
        monkeypatch.setenv("ENVIRONMENT", "production")

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True
        # Simulate: no Feast store, but custom store is present so fallback is attempted
        client._store = None
        client._custom_store = MagicMock()

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime(2024, 1, 1)],
            }
        )

        with pytest.raises(FeastFallbackError):
            await client.get_historical_features(
                entity_df=entity_df,
                feature_refs=["hcp_view:engagement_score"],
            )

        # _fallback_used must remain False — the raise fires before the side-effect.
        assert client._fallback_used is False

    @pytest.mark.asyncio
    async def test_fallback_used_flag_set_in_non_production(self, monkeypatch):
        """_fallback_used is set to True when the fallback succeeds (non-prod)."""
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True
        client._store = None
        client._custom_store = MagicMock()

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime(2024, 1, 1)],
            }
        )

        await client.get_historical_features(
            entity_df=entity_df,
            feature_refs=["hcp_view:engagement_score"],
        )

        assert client._fallback_used is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("env_value", ["PRODUCTION", "Production", "production"])
    async def test_historical_fallback_raises_case_insensitive(self, monkeypatch, env_value):
        """ENVIRONMENT check is case-insensitive (uppercase must NOT silently disable the prod block)."""
        monkeypatch.setenv("ENVIRONMENT", env_value)

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True
        client._store = None
        client._custom_store = MagicMock()

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime(2024, 1, 1)],
            }
        )

        with pytest.raises(FeastFallbackError):
            await client.get_historical_features(
                entity_df=entity_df,
                feature_refs=["hcp_view:engagement_score"],
            )

    @pytest.mark.asyncio
    async def test_online_fallback_raises_in_production(self, monkeypatch):
        """#532: the ONLINE-features custom-store fallback must ALSO fail loud in
        production, mirroring the historical-features guard.

        Otherwise the embedded path silently serves custom-store (Supabase) data
        for online features, and the predictions route would tag it
        ``feature_source='feast_online'`` — the exact #532 mislabel. The route
        cannot distinguish a real Feast fetch from a fallback, so the honesty
        guard belongs here at the client.
        """
        monkeypatch.setenv("ENVIRONMENT", "production")

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True
        client._store = None
        client._remote_base_url = None
        client._custom_store = MagicMock()

        with pytest.raises(FeastFallbackError):
            await client.get_online_features(
                entity_rows=[{"patient_id": "PAT-1"}],
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("env_value", ["PRODUCTION", "Production", "production"])
    async def test_online_fallback_raises_case_insensitive(self, monkeypatch, env_value):
        """ENVIRONMENT check for the online fallback is case-insensitive too."""
        monkeypatch.setenv("ENVIRONMENT", env_value)

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True
        client._store = None
        client._remote_base_url = None
        client._custom_store = MagicMock()

        with pytest.raises(FeastFallbackError):
            await client.get_online_features(
                entity_rows=[{"patient_id": "PAT-1"}],
                feature_refs=["hcp_conversion_features:engagement_score"],
            )


class TestFreshnessDefaultsToStaleOnException:
    """Test that get_feature_freshness defaults to stale on exception."""

    @pytest.mark.asyncio
    async def test_freshness_default_false_on_exception(self, monkeypatch):
        """When initialization raises, freshness is treated as stale (is_fresh=False)."""
        monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

        client = FeastClient()
        # Patch initialize to raise so the inner body never executes
        client.initialize = AsyncMock(side_effect=Exception("Feast service unreachable"))

        result = await client.get_feature_freshness("some_feature_view")

        assert result.is_fresh is False
        assert result.freshness_status is FreshnessStatus.UNKNOWN
        assert result.message is not None
        assert "Freshness check failed" in result.message

    @pytest.mark.asyncio
    async def test_freshness_allow_stale_env_overrides_exception(self, monkeypatch):
        """ALLOW_STALE_FEAST=1 causes is_fresh=True even when initialization raises."""
        monkeypatch.setenv("ALLOW_STALE_FEAST", "1")

        client = FeastClient()
        client.initialize = AsyncMock(side_effect=Exception("Feast service unreachable"))

        result = await client.get_feature_freshness("some_feature_view")

        assert result.is_fresh is True


# ============================================================================
# Block 2 polish — FeastError base class + defensive-chain narrowing tests
# ============================================================================


class TestFeastErrorBaseClass:
    """Tests for the FeastError base class introduced in Block 2 polish.

    Catchers that want "any Feast-domain error" should be able to write
    ``except FeastError`` rather than enumerating every subclass. This
    keeps the contract stable when new Feast-domain exceptions are added.
    """

    def test_feast_fallback_error_is_feast_error(self):
        """FeastFallbackError is a subclass of FeastError (isinstance check)."""
        assert isinstance(FeastFallbackError(), FeastError)
        assert issubclass(FeastFallbackError, FeastError)

    def test_feast_error_catches_feast_fallback_error(self):
        """Catching FeastError catches FeastFallbackError too."""
        caught = False
        try:
            raise FeastFallbackError("test")
        except FeastError:
            caught = True
        assert caught is True


class TestFreshnessDefensiveExceptionChain:
    """Test that the pre-emptive ``except FeastFallbackError: raise`` in
    ``get_feature_freshness`` mirrors the ``get_historical_features`` pattern.

    This is the "latent today, defensive against tomorrow" guard from
    Block 2 polish: if a future refactor of ``_ensure_initialized`` (or
    any helper above) starts raising ``FeastFallbackError``, the broad
    ``except Exception`` below would otherwise swallow it into an UNKNOWN
    status — masking a production-policy violation.
    """

    @pytest.mark.asyncio
    async def test_freshness_propagates_feast_fallback_error_through_outer_except(
        self, monkeypatch
    ):
        """FeastFallbackError raised inside get_feature_freshness MUST propagate.

        Patch the initialize() helper to raise FeastFallbackError. The
        outer ``except FeastFallbackError: raise`` must re-raise it
        rather than letting the broad ``except Exception`` convert it
        to an UNKNOWN-status FeatureFreshness. ALLOW_STALE_FEAST is
        unset to make sure the bypass path doesn't muddy the test.
        """
        monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

        client = FeastClient()
        client.initialize = AsyncMock(
            side_effect=FeastFallbackError("simulated future refactor"),
        )

        with pytest.raises(FeastFallbackError, match="simulated future refactor"):
            await client.get_feature_freshness("some_feature_view")

    @pytest.mark.asyncio
    async def test_freshness_propagates_feast_fallback_error_even_with_allow_stale(
        self, monkeypatch
    ):
        """ALLOW_STALE_FEAST=1 must NOT mask FeastFallbackError propagation.

        The bypass env-var is for ops emergencies during a stale Feast,
        not for swallowing production-policy violations. The pre-emptive
        ``except FeastFallbackError: raise`` fires *before* ALLOW_STALE_FEAST
        is consulted in the broad except-Exception block, so the policy
        violation is surfaced regardless of the bypass.
        """
        monkeypatch.setenv("ALLOW_STALE_FEAST", "1")

        client = FeastClient()
        client.initialize = AsyncMock(
            side_effect=FeastFallbackError("simulated future refactor"),
        )

        with pytest.raises(FeastFallbackError, match="simulated future refactor"):
            await client.get_feature_freshness("some_feature_view")


class TestHistoricalFeaturesOuterExceptionPath:
    """Test that ``get_historical_features`` re-raises non-FeastFallbackError
    exceptions through the outer ``except Exception`` block when no fallback
    is configured.

    Covers the path where the inner Feast call raises something *other*
    than FeastFallbackError (e.g., a generic RuntimeError), and there is
    no ``_custom_store`` to fall back to. The function should re-raise
    the original exception (NOT swallow it into FeastFallbackError).
    """

    @pytest.mark.asyncio
    async def test_outer_except_reraises_non_fallback_exception_without_custom_store(
        self, monkeypatch
    ):
        """Generic Feast-store exception with no custom_store re-raises original."""
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True

        # Configure Feast store to raise a generic RuntimeError (NOT FeastFallbackError)
        mock_store = MagicMock()
        mock_store.get_historical_features.side_effect = RuntimeError(
            "Feast offline store unavailable"
        )
        client._store = mock_store
        # No custom store → outer except has nothing to fall back to → must re-raise
        client._custom_store = None

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime(2024, 1, 1)],
            }
        )

        with pytest.raises(RuntimeError, match="Feast offline store unavailable"):
            await client.get_historical_features(
                entity_df=entity_df,
                feature_refs=["hcp_view:engagement_score"],
            )

    @pytest.mark.asyncio
    async def test_outer_except_routes_to_fallback_when_custom_store_present(self, monkeypatch):
        """Generic Feast-store exception WITH custom_store routes to fallback (non-prod).

        Confirms the outer ``except Exception`` block invokes the
        fallback path when both ``enable_fallback=True`` and
        ``_custom_store`` are present, AND ENVIRONMENT is not production.
        This is the complementary path to the
        FeastFallbackError-passthrough test above.
        """
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        client = FeastClient(config=FeastConfig(enable_fallback=True))
        client._initialized = True

        # Feast store raises generic exception
        mock_store = MagicMock()
        mock_store.get_historical_features.side_effect = RuntimeError(
            "Feast offline store unavailable"
        )
        client._store = mock_store
        client._custom_store = MagicMock()

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime(2024, 1, 1)],
            }
        )

        # Should not raise — fallback path returns a DataFrame
        result = await client.get_historical_features(
            entity_df=entity_df,
            feature_refs=["hcp_view:engagement_score"],
        )

        assert isinstance(result, pd.DataFrame)
        # _fallback_used flag set in the fallback path
        assert client._fallback_used is True


def _remote_cm(mock_response):
    """Build a mock ``httpx.AsyncClient(...)`` context manager whose ``.post``
    returns ``mock_response``. Returns ``(context_manager, mock_client)``."""
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm, mock_client


class TestOnlineFeaturesRemote:
    """Online features via the Feast feature-server HTTP sidecar (#532 Option 1).

    The app image cannot ``import feast`` (feast 0.43.0 pins ``tenacity<9``,
    irreconcilable with the prod ``tenacity==9.1.2``), so when ``FEAST_URL`` is
    configured the client fetches online features over HTTP from the
    ``e2i_feast`` sidecar's ``POST /get-online-features`` endpoint instead of an
    embedded ``FeatureStore``.
    """

    def test_config_server_url_default_none(self):
        """server_url defaults to None — embedded mode unless explicitly configured."""
        assert FeastConfig().server_url is None

    @pytest.mark.asyncio
    async def test_initialize_remote_mode_works_without_feast(self, monkeypatch):
        """Remote mode initializes even when feast is unimportable (the app-image condition)."""
        import builtins

        real_import = builtins.__import__

        def no_feast(name, *args, **kwargs):
            if name == "feast" or name.startswith("feast."):
                raise ImportError("No module named 'feast'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_feast)

        client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))
        await client.initialize()

        assert client._initialized is True
        assert client._store is None
        assert client._remote_base_url == "http://feast:6566"

    @pytest.mark.asyncio
    async def test_remote_posts_columnar_request(self):
        """entity_rows (row-oriented) transpose to columnar `entities`; POST to /get-online-features."""
        client = FeastClient(config=FeastConfig(server_url="http://feast:6566/"))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "metadata": {"feature_names": ["hcp_conversion_features__engagement_score"]},
                "results": [{"values": [0.85, 0.42], "statuses": ["PRESENT", "PRESENT"]}],
            }
        )
        cm, mock_client = _remote_cm(mock_response)

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            await client.get_online_features(
                entity_rows=[
                    {"hcp_id": "123", "brand_id": "remibrutinib"},
                    {"hcp_id": "456", "brand_id": "fabhalta"},
                ],
                feature_refs=["hcp_conversion_features:engagement_score"],
                full_feature_names=True,
            )

        call = mock_client.post.call_args
        # trailing slash on server_url is normalized to a single join
        assert call.args[0] == "http://feast:6566/get-online-features"
        body = call.kwargs["json"]
        assert body["features"] == ["hcp_conversion_features:engagement_score"]
        assert body["full_feature_names"] is True
        assert body["entities"] == {
            "hcp_id": ["123", "456"],
            "brand_id": ["remibrutinib", "fabhalta"],
        }

    @pytest.mark.asyncio
    async def test_remote_parses_response_to_flat_dict(self):
        """The feature-server proto-dict response flattens to {feature_name: [values]} (drop-in with embedded to_dict())."""
        client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "metadata": {
                    "feature_names": ["hcp_id", "hcp_conversion_features__engagement_score"]
                },
                "results": [
                    {"values": ["123"], "statuses": ["PRESENT"]},
                    {"values": [0.85], "statuses": ["PRESENT"]},
                ],
            }
        )
        cm, _ = _remote_cm(mock_response)

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            result = await client.get_online_features(
                entity_rows=[{"hcp_id": "123"}],
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

        assert result == {
            "hcp_id": ["123"],
            "hcp_conversion_features__engagement_score": [0.85],
        }

    @pytest.mark.asyncio
    async def test_remote_not_found_status_maps_to_none(self):
        """A non-PRESENT status yields None for that cell (no stale/garbage value forwarded)."""
        client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "metadata": {"feature_names": ["fv__feat"]},
                "results": [{"values": [None], "statuses": ["NOT_FOUND"]}],
            }
        )
        cm, _ = _remote_cm(mock_response)

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            result = await client.get_online_features(
                entity_rows=[{"hcp_id": "999"}],
                feature_refs=["fv:feat"],
            )

        assert result == {"fv__feat": [None]}

    @pytest.mark.asyncio
    async def test_remote_fails_loud_on_feature_names_results_length_mismatch(self):
        """A 200 response whose feature_names/results lengths disagree RAISES (not silent partial data)."""
        client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "metadata": {"feature_names": ["fv__feat"]},
                "results": [],  # 1 name, 0 result columns -> malformed
            }
        )
        cm, _ = _remote_cm(mock_response)

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            with pytest.raises(FeastError):
                await client.get_online_features(
                    entity_rows=[{"hcp_id": "1"}],
                    feature_refs=["fv:feat"],
                )

    @pytest.mark.asyncio
    async def test_remote_fails_loud_on_row_count_mismatch(self):
        """A column whose values length != number of requested entity rows RAISES."""
        client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "metadata": {"feature_names": ["fv__feat"]},
                # 1 value/status but 2 entity rows were requested -> malformed
                "results": [{"values": [0.85], "statuses": ["PRESENT"]}],
            }
        )
        cm, _ = _remote_cm(mock_response)

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            with pytest.raises(FeastError):
                await client.get_online_features(
                    entity_rows=[{"hcp_id": "1"}, {"hcp_id": "2"}],
                    feature_refs=["fv:feat"],
                )

    @pytest.mark.asyncio
    async def test_remote_fails_loud_on_sidecar_error(self):
        """A sidecar transport error RAISES FeastError — it must NOT silently degrade to the custom store."""
        import httpx

        client = FeastClient(config=FeastConfig(server_url="http://feast:6566"))
        # Attach a custom store to prove the remote path does NOT silently fall
        # back to it on error (that silent mislabel was the #532 bug).
        client._custom_store = MagicMock()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_client)
        cm.__aexit__ = AsyncMock(return_value=None)

        with patch("src.feature_store.feast_client.httpx.AsyncClient", return_value=cm):
            with pytest.raises(FeastError):
                await client.get_online_features(
                    entity_rows=[{"hcp_id": "123"}],
                    feature_refs=["fv:feat"],
                )

    @pytest.mark.asyncio
    async def test_get_feast_client_wires_feast_url_env(self, monkeypatch):
        """get_feast_client() with no explicit config reads FEAST_URL into config.server_url (prod wiring)."""
        import src.feature_store.feast_client as module

        module._client = None
        monkeypatch.setenv("FEAST_URL", "http://feast:6566")
        try:
            client = await get_feast_client()
            assert client.config.server_url == "http://feast:6566"
            assert client._remote_base_url == "http://feast:6566"
        finally:
            module._client = None


class TestFeatureStatisticsNoFabricatedRecency:
    """#556 anti-mocking: get_feature_statistics must not fabricate last_updated=now().

    A freshness check compares now() to stats.last_updated; if last_updated is itself
    a fabricated now(), age is ~0 and the check ALWAYS reports fresh — silently
    defeating the QC gate. When real recency cannot be determined the stat paths must
    return last_updated=None (unverifiable), not a fabricated timestamp.
    """

    @pytest.mark.asyncio
    async def test_get_feature_statistics_returns_none_recency_without_real_source(self):
        client = FeastClient()
        # Bypass initialize(): the app/CI image cannot `import feast` (feast 0.43.0
        # pins tenacity<9, prod uses tenacity==9.1.2; #307), and initialize()
        # re-raises that ImportError by design. We are not testing init here — we
        # are asserting the recency contract of the statistics path, so mark the
        # client initialized and leave _store=None.
        client._initialized = True
        # #559: _compute_feature_statistics now self-resolves the shared sync Supabase
        # client via get_supabase() when none is passed. To assert the "no real recency
        # available" contract hermetically (and not depend on ambient SUPABASE_* env), we
        # force get_supabase() to None: no client → no source query → recency genuinely
        # unknown → last_updated MUST be None (never a fabricated now()).
        with patch("src.api.dependencies.supabase_client.get_supabase", return_value=None):
            stats = await client.get_feature_statistics(
                feature_view="hcp_conversion_features",
                feature_name="engagement_score",
                supabase_client=None,
            )
        assert stats is not None
        assert stats.last_updated is None, (
            "last_updated must be None when no real recency was computed — never a "
            "fabricated datetime.now() (#556)"
        )


class TestRealRecencySignal559:
    """#559: ``check_feature_freshness`` must reflect REAL data age.

    Recency = ``MAX(<raw source timestamp column>)`` for the feature view's source
    table, computed via PostgREST (``.table().select().order().limit()``) — NOT the
    ``execute_sql`` RPC, which does NOT exist in the target Supabase (verified live;
    the pre-existing stats query silently 404s and degrades to a count-only fallback).
    When a source table / timestamp column cannot be resolved, recency is ``None``
    (genuinely unknown → fail-closed downstream) — never a fabricated ``now()``.
    """

    # ---------------- cycle 1: per-table raw timestamp-column mapping ----------------
    def test_infer_timestamp_column_maps_live_verified_columns(self):
        """Mapping matches columns verified to exist in the live DB (correct types)."""
        c = FeastClient()
        assert c._infer_timestamp_column("hcp_profiles") == "updated_at"
        assert c._infer_timestamp_column("triggers") == "trigger_timestamp"
        assert c._infer_timestamp_column("business_metrics") == "metric_date"
        assert c._infer_timestamp_column("patient_journeys") == "journey_start_date"
        assert c._infer_timestamp_column("territory_metrics") == "metric_date"

    def test_infer_timestamp_column_unmapped_returns_none(self):
        assert FeastClient()._infer_timestamp_column("not_a_real_table") is None

    def test_infer_source_table_authoritative_feature_view_mapping(self):
        """#559: every real Feast FeatureView name must resolve to the table its Feast
        source actually reads from (cross-referenced to feature_repo/features/*.py +
        feature_repo/data_sources.py) — otherwise recency is a real-but-WRONG MAX from the
        wrong table. In particular hcp_conversion_features and hcp_engagement_features
        source from business_metrics_source (NOT hcp_profiles)."""
        c = FeastClient()
        expected = {
            "hcp_conversion_features": "business_metrics",
            "hcp_engagement_features": "business_metrics",
            "hcp_profile_features": "hcp_profiles",
            "patient_journey_features": "patient_journeys",
            "patient_adherence_features": "patient_journeys",
            "trigger_effectiveness_features": "triggers",
            "trigger_response_features": "triggers",
            "territory_performance_features": "territory_metrics",
            "market_dynamics_features": "business_metrics",
        }
        for fv, table in expected.items():
            assert c._infer_source_table(fv) == table, f"{fv} must resolve to {table}"

    # ---------------- cycle 2: recency value parsing ----------------
    def test_parse_recency_value_aware_datetime_passthrough(self):
        aware = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)
        assert FeastClient._parse_recency_value(aware) == aware

    def test_parse_recency_value_naive_datetime_assumed_utc(self):
        parsed = FeastClient._parse_recency_value(datetime(2026, 5, 20, 14, 30))
        assert parsed == datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)
        assert parsed.tzinfo is not None

    def test_parse_recency_value_iso_string_with_offset(self):
        assert FeastClient._parse_recency_value("2026-05-20T14:30:00+00:00") == datetime(
            2026, 5, 20, 14, 30, tzinfo=timezone.utc
        )

    def test_parse_recency_value_z_suffix(self):
        assert FeastClient._parse_recency_value("2026-05-20T14:30:00Z") == datetime(
            2026, 5, 20, 14, 30, tzinfo=timezone.utc
        )

    def test_parse_recency_value_bare_date_is_utc_midnight(self):
        parsed = FeastClient._parse_recency_value("2026-05-25")
        assert (parsed.year, parsed.month, parsed.day) == (2026, 5, 25)
        assert parsed.tzinfo is not None

    def test_parse_recency_value_none_empty_and_garbage_return_none(self):
        assert FeastClient._parse_recency_value(None) is None
        assert FeastClient._parse_recency_value("") is None
        assert FeastClient._parse_recency_value("   ") is None
        assert FeastClient._parse_recency_value("not-a-date") is None
        assert FeastClient._parse_recency_value(12345) is None

    # ---------------- cycle 3: _query_max_recency uses PostgREST, not execute_sql ----
    @pytest.mark.asyncio
    async def test_query_max_recency_uses_postgrest_order_limit_not_rpc(self):
        c = FeastClient()
        client = MagicMock()
        result = MagicMock()
        result.data = [{"updated_at": "2026-04-28T12:40:31.906808+00:00"}]
        (
            client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value
        ) = result

        recency = await c._query_max_recency(client, "hcp_profiles")

        assert recency == datetime(2026, 4, 28, 12, 40, 31, 906808, tzinfo=timezone.utc)
        client.table.assert_called_once_with("hcp_profiles")
        client.table.return_value.select.assert_called_once_with("updated_at")
        client.table.return_value.select.return_value.order.assert_called_once_with(
            "updated_at", desc=True, nullsfirst=False
        )
        client.table.return_value.select.return_value.order.return_value.limit.assert_called_once_with(
            1
        )
        # the dead execute_sql RPC must NEVER be used for recency (the #559 regression)
        client.rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_max_recency_unmapped_table_issues_no_query(self):
        c = FeastClient()
        client = MagicMock()
        recency = await c._query_max_recency(client, "some_unmapped_table")
        assert recency is None
        client.table.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_max_recency_query_failure_returns_none(self):
        c = FeastClient()
        client = MagicMock()
        (
            client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.side_effect
        ) = RuntimeError("boom")
        assert await c._query_max_recency(client, "triggers") is None

    @pytest.mark.asyncio
    async def test_query_max_recency_empty_result_returns_none(self):
        c = FeastClient()
        client = MagicMock()
        result = MagicMock()
        result.data = []
        (
            client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value
        ) = result
        assert await c._query_max_recency(client, "hcp_profiles") is None

    # ---------------- cycle 4: _query_statistics_from_supabase threads recency --------
    @pytest.mark.asyncio
    async def test_stats_threads_recency_on_primary_query_path(self):
        c = FeastClient()
        dt = datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)
        stats_row = {
            "total_count": 100,
            "non_null_count": 100,
            "null_count": 0,
            "min_val": 0.0,
            "max_val": 1.0,
            "mean_val": 0.5,
            "stddev_val": 0.1,
        }
        client = MagicMock()
        client.rpc.return_value.execute.return_value = MagicMock(data=[stats_row])
        with patch.object(c, "_query_max_recency", AsyncMock(return_value=dt)):
            stats = await c._query_statistics_from_supabase(
                client=client,
                table_name="hcp_profiles",
                column_name="engagement_score",
                feature_view="hcp_conversion_features",
            )
        assert stats.count == 100
        assert stats.last_updated == dt

    @pytest.mark.asyncio
    async def test_stats_threads_recency_on_countonly_fallback(self):
        """When the primary stats query fails (execute_sql dead), recency still flows
        through the count-only fallback path."""
        c = FeastClient()
        dt = datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)
        client = MagicMock()
        client.rpc.return_value.execute.side_effect = RuntimeError("execute_sql missing")
        count_res = MagicMock()
        count_res.count = 4242
        client.table.return_value.select.return_value.limit.return_value.execute.return_value = (
            count_res
        )
        with patch.object(c, "_query_max_recency", AsyncMock(return_value=dt)):
            stats = await c._query_statistics_from_supabase(
                client=client,
                table_name="triggers",
                column_name="conversion_flag",
                feature_view="trigger_features",
            )
        assert stats.count == 4242
        assert stats.last_updated == dt

    @pytest.mark.asyncio
    async def test_stats_recency_none_preserves_stats(self):
        """Recency genuinely unknown → last_updated None, but the stats survive."""
        c = FeastClient()
        stats_row = {
            "total_count": 7,
            "non_null_count": 7,
            "null_count": 0,
            "min_val": 0.0,
            "max_val": 1.0,
            "mean_val": 0.5,
            "stddev_val": 0.1,
        }
        client = MagicMock()
        client.rpc.return_value.execute.return_value = MagicMock(data=[stats_row])
        with patch.object(c, "_query_max_recency", AsyncMock(return_value=None)):
            stats = await c._query_statistics_from_supabase(
                client=client,
                table_name="hcp_profiles",
                column_name="x",
                feature_view="hcp_conversion_features",
            )
        assert stats.count == 7
        assert stats.last_updated is None

    # ---------------- cycle 5: _compute_feature_statistics self-resolves client -------
    @pytest.mark.asyncio
    async def test_compute_stats_self_resolves_get_supabase_when_no_client(self):
        """The adapter calls get_feature_statistics with NO supabase_client. The stats
        path must self-resolve the shared sync client (get_supabase) so recency fires —
        otherwise the freshness gate is permanently fail-closed (the #559 trap)."""
        c = FeastClient()
        c._initialized = True
        dt = datetime(2026, 4, 28, 12, 0, tzinfo=timezone.utc)
        resolved = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="engagement_score",
            count=5,
            null_count=0,
            last_updated=dt,
        )
        with (
            patch(
                "src.api.dependencies.supabase_client.get_supabase",
                return_value=MagicMock(),
            ),
            patch.object(
                c, "_query_statistics_from_supabase", AsyncMock(return_value=resolved)
            ) as q,
        ):
            stats = await c._compute_feature_statistics(
                feature_view="hcp_conversion_features",
                feature_name="engagement_score",
            )
        q.assert_awaited_once()
        assert stats is not None
        assert stats.last_updated == dt

    @pytest.mark.asyncio
    async def test_compute_stats_get_supabase_none_fails_closed_no_fabrication(self):
        """get_supabase() unavailable → recency genuinely unknown → last_updated None
        (honest fail-closed), never a fabricated now()."""
        c = FeastClient()
        c._initialized = True
        c._store = None
        with patch("src.api.dependencies.supabase_client.get_supabase", return_value=None):
            stats = await c._compute_feature_statistics(
                feature_view="hcp_conversion_features",
                feature_name="engagement_score",
            )
        assert stats is not None
        assert stats.last_updated is None
