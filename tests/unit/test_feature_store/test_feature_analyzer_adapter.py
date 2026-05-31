"""Tests for Feature Analyzer Adapter with Feast integration.

Tests cover:
- Feast lazy initialization
- Point-in-time correct training data retrieval
- Feature discovery from registry
- Feature freshness checking
- Fallback to custom store
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from src.feature_store.feast_client import FeastClient, FeatureStatistics
from src.feature_store.feature_analyzer_adapter import (
    PYTHON_TO_FEATURE_TYPE,
    FeatureAnalyzerAdapter,
    get_feature_analyzer_adapter,
)
from src.feature_store.models import FeatureValueType


class TestFeatureAnalyzerAdapterInit:
    """Test adapter initialization."""

    def test_adapter_creation_with_defaults(self):
        """Test adapter can be created with defaults."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client)

        assert adapter.fs_client is mock_fs_client
        assert adapter.auto_create_groups is True
        assert adapter.enable_feast is True
        assert adapter._feast_client is None
        assert adapter._feast_initialized is False

    def test_adapter_creation_with_feast_disabled(self):
        """Test adapter with Feast disabled."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            enable_feast=False,
        )

        assert adapter.enable_feast is False

    def test_adapter_creation_with_feast_client(self):
        """Test adapter with provided Feast client."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        assert adapter._feast_client is mock_feast_client


class TestFeastInitialization:
    """Test Feast lazy initialization."""

    @pytest.mark.asyncio
    async def test_ensure_feast_initialized_when_disabled(self):
        """Test that initialization returns False when Feast disabled."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client, enable_feast=False)

        result = await adapter._ensure_feast_initialized()

        assert result is False
        assert adapter._feast_initialized is False

    @pytest.mark.asyncio
    async def test_ensure_feast_initialized_success(self):
        """Test successful Feast initialization."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter._ensure_feast_initialized()

        assert result is True
        assert adapter._feast_initialized is True
        mock_feast_client.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_feast_initialized_already_initialized(self):
        """Test that initialization is skipped if already done."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )
        adapter._feast_initialized = True

        result = await adapter._ensure_feast_initialized()

        assert result is True
        mock_feast_client.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_feast_initialized_failure(self):
        """Test handling of initialization failure."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock(side_effect=Exception("Init failed"))

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter._ensure_feast_initialized()

        assert result is False
        assert adapter._feast_initialized is False


class TestGetTrainingFeatures:
    """Test point-in-time correct training feature retrieval."""

    @pytest.mark.asyncio
    async def test_get_training_features_validation_empty_df(self):
        """Test validation of empty entity DataFrame."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client)

        with pytest.raises(ValueError, match="entity_df cannot be empty"):
            await adapter.get_training_features(
                entity_df=pd.DataFrame(),
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

    @pytest.mark.asyncio
    async def test_get_training_features_validation_missing_timestamp(self):
        """Test validation of missing event_timestamp column."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client)

        entity_df = pd.DataFrame({"hcp_id": ["123"]})

        with pytest.raises(ValueError, match="event_timestamp"):
            await adapter.get_training_features(
                entity_df=entity_df,
                feature_refs=["hcp_conversion_features:engagement_score"],
            )

    @pytest.mark.asyncio
    async def test_get_training_features_validation_empty_refs(self):
        """Test validation of empty feature refs."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client)

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime.now()],
            }
        )

        with pytest.raises(ValueError, match="feature_refs cannot be empty"):
            await adapter.get_training_features(
                entity_df=entity_df,
                feature_refs=[],
            )

    @pytest.mark.asyncio
    async def test_get_training_features_from_feast(self):
        """Test successful feature retrieval from Feast."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123", "456"],
                "brand_id": ["remibrutinib", "remibrutinib"],
                "event_timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 15)],
            }
        )

        result_df = entity_df.copy()
        result_df["hcp_conversion_features__engagement_score"] = [0.85, 0.72]

        mock_feast_client.get_historical_features = AsyncMock(return_value=result_df)

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.get_training_features(
            entity_df=entity_df,
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        assert len(result) == 2
        assert "hcp_conversion_features__engagement_score" in result.columns
        mock_feast_client.get_historical_features.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_training_features_fallback_on_feast_error(self):
        """Test fallback to custom store when Feast fails."""
        mock_fs_client = MagicMock()
        mock_fs_client.get_features.return_value = {"engagement_score": 0.5}

        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        mock_feast_client.get_historical_features = AsyncMock(side_effect=Exception("Feast error"))

        entity_df = pd.DataFrame(
            {
                "hcp_id": ["123"],
                "event_timestamp": [datetime.now()],
            }
        )

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.get_training_features(
            entity_df=entity_df,
            feature_refs=["hcp_conversion_features:engagement_score"],
        )

        # Should return result from fallback
        assert result is not None


class TestDiscoverFeatures:
    """Test feature discovery from registry."""

    @pytest.mark.asyncio
    async def test_discover_features_from_feast(self):
        """Test feature discovery from Feast registry."""
        mock_fs_client = MagicMock()
        mock_fs_client.list_feature_groups.return_value = []

        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        mock_feast_client.list_feature_views = AsyncMock(
            return_value=[
                {
                    "name": "hcp_conversion_features",
                    "entities": ["hcp", "brand"],
                    "tags": {"use_case": "hcp_conversion"},
                    "online": True,
                    "ttl_days": 7,
                    "schema": [
                        {"name": "engagement_score", "dtype": "FLOAT64"},
                        {"name": "trx_count", "dtype": "INT64"},
                    ],
                }
            ]
        )

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.discover_features()

        assert len(result) == 2
        assert result[0]["feature_view"] == "hcp_conversion_features"
        assert result[0]["feature_name"] == "engagement_score"

    @pytest.mark.asyncio
    async def test_discover_features_with_use_case_filter(self):
        """Test feature discovery with use_case filter."""
        mock_fs_client = MagicMock()
        mock_fs_client.list_feature_groups.return_value = []

        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        mock_feast_client.list_feature_views = AsyncMock(
            return_value=[
                {
                    "name": "hcp_conversion_features",
                    "entities": ["hcp"],
                    "tags": {"use_case": "hcp_conversion"},
                    "schema": [{"name": "engagement_score", "dtype": "FLOAT64"}],
                },
                {
                    "name": "patient_journey_features",
                    "entities": ["patient"],
                    "tags": {"use_case": "churn_prediction"},
                    "schema": [{"name": "adherence_rate", "dtype": "FLOAT64"}],
                },
            ]
        )

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.discover_features(use_case="hcp_conversion")

        assert len(result) == 1
        assert result[0]["feature_view"] == "hcp_conversion_features"

    @pytest.mark.asyncio
    async def test_discover_features_with_entity_filter(self):
        """Test feature discovery with entity filter."""
        mock_fs_client = MagicMock()
        mock_fs_client.list_feature_groups.return_value = []

        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        mock_feast_client.list_feature_views = AsyncMock(
            return_value=[
                {
                    "name": "hcp_conversion_features",
                    "entities": ["hcp", "brand"],
                    "tags": {},
                    "schema": [{"name": "engagement_score", "dtype": "FLOAT64"}],
                },
                {
                    "name": "patient_journey_features",
                    "entities": ["patient"],
                    "tags": {},
                    "schema": [{"name": "adherence_rate", "dtype": "FLOAT64"}],
                },
            ]
        )

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.discover_features(entity_names=["patient"])

        assert len(result) == 1
        assert result[0]["feature_view"] == "patient_journey_features"


class TestCheckFeatureFreshness:
    """Test feature freshness checking."""

    @pytest.mark.asyncio
    async def test_check_freshness_feast_unavailable_fails_closed(self):
        """#556: when Feast is unavailable the freshness signal must FAIL CLOSED
        (fresh=False), not silently report fresh=True. Otherwise the QC/registrar
        gate that blocks on stale features is defeated by an empty/down store.
        The ALLOW_STALE_FEAST escape hatch (in feast_registrar) lets intentional
        no-Feast environments proceed.
        """
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client, enable_feast=False)

        result = await adapter.check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"]
        )

        assert result["fresh"] is False
        assert "Feast not available" in result["recommendations"][0]

    @pytest.mark.asyncio
    async def test_check_freshness_unverifiable_feature_fails_closed(self):
        """#556: a feature whose freshness cannot be verified (no statistics, or
        the stats lookup errors) must NOT be reported fresh — an unverifiable
        feature is treated as stale so the gate stays honest."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        # No statistics available for the feature.
        mock_feast_client.get_feature_statistics = AsyncMock(return_value=None)

        adapter = FeatureAnalyzerAdapter(mock_fs_client, feast_client=mock_feast_client)

        result = await adapter.check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"],
            max_staleness_hours=24.0,
        )

        assert result["fresh"] is False
        assert "hcp_conversion_features:engagement_score" in result["stale_features"]

    @pytest.mark.asyncio
    async def test_check_freshness_all_fresh(self):
        """Test freshness check when all features are fresh."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()

        # Feature updated 1 hour ago
        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="engagement_score",
            count=1000,
            null_count=0,
            last_updated=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        mock_feast_client.get_feature_statistics = AsyncMock(return_value=stats)

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"],
            max_staleness_hours=24.0,
        )

        assert result["fresh"] is True
        assert len(result["stale_features"]) == 0

    @pytest.mark.asyncio
    async def test_check_freshness_stale_features(self):
        """Test freshness check when features are stale."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()

        # Feature updated 48 hours ago
        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="engagement_score",
            count=1000,
            null_count=0,
            last_updated=datetime.now(timezone.utc) - timedelta(hours=48),
        )
        mock_feast_client.get_feature_statistics = AsyncMock(return_value=stats)

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"],
            max_staleness_hours=24.0,
        )

        assert result["fresh"] is False
        assert "hcp_conversion_features:engagement_score" in result["stale_features"]
        assert len(result["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_check_freshness_null_last_updated_fails_closed(self):
        """#556: get_feature_statistics no longer fabricates last_updated=now(); when
        recency is unknown it returns last_updated=None. The freshness check must
        treat that as unverifiable (fresh=False), NOT compute age against a fake
        'now' and report fresh — that fabricated-now was the silent mock that
        defeated the gate."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        stats = FeatureStatistics(
            feature_view="hcp_conversion_features",
            feature_name="engagement_score",
            count=1000,
            null_count=0,
            last_updated=None,  # recency unknown — must NOT be treated as fresh
        )
        mock_feast_client.get_feature_statistics = AsyncMock(return_value=stats)

        adapter = FeatureAnalyzerAdapter(mock_fs_client, feast_client=mock_feast_client)

        result = await adapter.check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"],
            max_staleness_hours=24.0,
        )

        assert result["fresh"] is False
        assert "hcp_conversion_features:engagement_score" in result["stale_features"]


class TestSyncFeaturesToFeast:
    """Test feature sync to Feast."""

    @pytest.mark.asyncio
    async def test_sync_feast_unavailable(self):
        """Test sync when Feast is unavailable."""
        mock_fs_client = MagicMock()
        adapter = FeatureAnalyzerAdapter(mock_fs_client, enable_feast=False)

        result = await adapter.sync_features_to_feast(
            experiment_id="exp_001",
            entity_key="hcp_id",
        )

        assert result["synced"] is False
        assert "Feast not available" in result["errors"]

    @pytest.mark.asyncio
    async def test_sync_with_materialization(self):
        """Test sync with materialization."""
        mock_fs_client = MagicMock()

        # Mock feature list
        mock_feature = MagicMock()
        mock_feature.name = "engagement_score"
        mock_fs_client.list_features.return_value = [mock_feature]

        mock_feast_client = MagicMock(spec=FeastClient)
        mock_feast_client.initialize = AsyncMock()
        mock_feast_client.materialize_incremental = AsyncMock(return_value={"status": "completed"})

        adapter = FeatureAnalyzerAdapter(
            mock_fs_client,
            feast_client=mock_feast_client,
        )

        result = await adapter.sync_features_to_feast(
            experiment_id="exp_001",
            entity_key="hcp_id",
            materialize=True,
        )

        assert result["synced"] is True
        assert result["features_synced"] == 1
        assert result["materialized"] is True


class TestFactoryFunction:
    """Test factory function."""

    def test_get_feature_analyzer_adapter_default(self):
        """Test factory with defaults."""
        mock_fs_client = MagicMock()
        adapter = get_feature_analyzer_adapter(mock_fs_client)

        assert adapter.fs_client is mock_fs_client
        assert adapter.enable_feast is True

    def test_get_feature_analyzer_adapter_with_feast(self):
        """Test factory with Feast client."""
        mock_fs_client = MagicMock()
        mock_feast_client = MagicMock(spec=FeastClient)

        adapter = get_feature_analyzer_adapter(
            mock_fs_client,
            feast_client=mock_feast_client,
            enable_feast=True,
        )

        assert adapter._feast_client is mock_feast_client
        assert adapter.enable_feast is True

    def test_get_feature_analyzer_adapter_feast_disabled(self):
        """Test factory with Feast disabled."""
        mock_fs_client = MagicMock()

        adapter = get_feature_analyzer_adapter(
            mock_fs_client,
            enable_feast=False,
        )

        assert adapter.enable_feast is False


class TestRealRecencyFreshnessEndToEnd559:
    """#559: a REAL MAX(<timestamp col>) flows through a REAL FeastClient into the REAL
    ``adapter.check_feature_freshness`` (fresh vs stale), with NO stubbing of
    ``get_feature_statistics``. The adapter calls ``get_feature_statistics`` WITHOUT a
    ``supabase_client`` (the prod path), so the stats path must self-resolve the shared
    client via ``get_supabase()`` for recency to fire. We patch ``get_supabase`` to a mock
    answering the real two-mechanism shape: a PostgREST ``.order().limit()`` recency query
    and the count-only fallback (the ``execute_sql`` stats RPC fails, as it does live).
    This proves the always-fail-closed chain is broken by a GENUINE signal, not a
    fabricated one.
    """

    @staticmethod
    def _supabase_mock(recency: datetime) -> MagicMock:
        sb = MagicMock()
        # Real recency mechanism: .table(t).select(col).order(col, desc, nullslast).limit(1).
        # Key by every mapped timestamp column so the row is correct regardless of which
        # source table the feature view resolves to (hcp_conversion_features → business_metrics
        # → metric_date, etc.) — _query_max_recency reads rows[0].get(<selected column>).
        rec_res = MagicMock()
        rec_res.data = [
            {
                "updated_at": recency.isoformat(),
                "metric_date": recency.isoformat(),
                "trigger_timestamp": recency.isoformat(),
                "journey_start_date": recency.isoformat(),
            }
        ]
        (
            sb.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value
        ) = rec_res
        # The execute_sql stats RPC does not exist in the target Supabase → it fails →
        # the count-only PostgREST fallback runs. Recency must still be threaded through.
        sb.rpc.return_value.execute.side_effect = RuntimeError("execute_sql does not exist")
        count_res = MagicMock()
        count_res.count = 1000
        sb.table.return_value.select.return_value.limit.return_value.execute.return_value = (
            count_res
        )
        return sb

    async def _run(self, recency: datetime, monkeypatch) -> dict:
        feast_client = FeastClient()
        # app/CI image cannot `import feast` (tenacity pin, #307); we are not testing init.
        feast_client._initialized = True
        monkeypatch.setattr(
            "src.api.dependencies.supabase_client.get_supabase",
            lambda: self._supabase_mock(recency),
        )
        adapter = FeatureAnalyzerAdapter(MagicMock(), feast_client=feast_client)
        adapter._feast_initialized = True  # skip real feast init in _ensure_feast_initialized
        return await adapter.check_feature_freshness(
            feature_refs=["hcp_conversion_features:engagement_score"],
            max_staleness_hours=24.0,
        )

    @pytest.mark.asyncio
    async def test_recent_real_recency_reports_fresh(self, monkeypatch):
        result = await self._run(datetime.now(timezone.utc) - timedelta(hours=2), monkeypatch)
        assert result["fresh"] is True
        assert result["stale_features"] == []
        age = result["feature_ages"]["hcp_conversion_features:engagement_score"]
        assert age is not None and 0 <= age < 24

    @pytest.mark.asyncio
    async def test_old_real_recency_reports_stale(self, monkeypatch):
        result = await self._run(datetime.now(timezone.utc) - timedelta(hours=72), monkeypatch)
        assert result["fresh"] is False
        assert "hcp_conversion_features:engagement_score" in result["stale_features"]


class TestTypeMapping:
    """Test Python to feature type mapping."""

    def test_python_type_mapping(self):
        """Test Python dtype to FeatureValueType mapping."""
        assert PYTHON_TO_FEATURE_TYPE["int64"] == FeatureValueType.INT64
        assert PYTHON_TO_FEATURE_TYPE["float64"] == FeatureValueType.FLOAT64
        assert PYTHON_TO_FEATURE_TYPE["object"] == FeatureValueType.STRING
        assert PYTHON_TO_FEATURE_TYPE["bool"] == FeatureValueType.BOOL
        assert PYTHON_TO_FEATURE_TYPE["datetime64[ns]"] == FeatureValueType.TIMESTAMP
