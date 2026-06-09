"""
Unit tests for ML Data Loader - Phase 1: Data Loading Foundation.

Tests:
- Data loading from Supabase
- Temporal splitting
- Filter application
- Error handling
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.repositories.ml_data_loader import MLDataLoader, MLDataset, get_ml_data_loader


class TestMLDataset:
    """Tests for MLDataset container."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample MLDataset."""
        return MLDataset(
            train=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}),
            val=pd.DataFrame({"a": [7], "b": [8]}),
            test=pd.DataFrame({"a": [9, 10], "b": [11, 12]}),
            metadata={"table": "test", "loaded_at": "2024-01-01T00:00:00"},
        )

    def test_train_size(self, sample_dataset):
        """Test train_size property."""
        assert sample_dataset.train_size == 3

    def test_val_size(self, sample_dataset):
        """Test val_size property."""
        assert sample_dataset.val_size == 1

    def test_test_size(self, sample_dataset):
        """Test test_size property."""
        assert sample_dataset.test_size == 2

    def test_total_size(self, sample_dataset):
        """Test total_size property."""
        assert sample_dataset.total_size == 6

    def test_summary(self, sample_dataset):
        """Test summary method."""
        summary = sample_dataset.summary()
        assert summary["train_size"] == 3
        assert summary["val_size"] == 1
        assert summary["test_size"] == 2
        assert "split_ratios" in summary
        assert summary["table"] == "test"


class TestMLDataLoader:
    """Tests for MLDataLoader."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def loader(self, mock_client):
        """Create loader with mock client."""
        return MLDataLoader(supabase_client=mock_client)

    @pytest.fixture
    def sample_data(self):
        """Sample data rows."""
        now = datetime.now()
        return [
            {
                "id": "1",
                "metric_name": "TRx_volume",
                "brand": "Kisqali",
                "value": 100,
                "created_at": (now - timedelta(days=90)).isoformat(),
            },
            {
                "id": "2",
                "metric_name": "TRx_volume",
                "brand": "Kisqali",
                "value": 110,
                "created_at": (now - timedelta(days=60)).isoformat(),
            },
            {
                "id": "3",
                "metric_name": "TRx_volume",
                "brand": "Kisqali",
                "value": 120,
                "created_at": (now - timedelta(days=10)).isoformat(),
            },
        ]


class TestLoadForTraining(TestMLDataLoader):
    """Tests for load_for_training method."""

    @pytest.mark.asyncio
    async def test_raises_error_for_unsupported_table(self, loader):
        """Test that unsupported tables raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await loader.load_for_training(table="unsupported_table")
        assert "not supported" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_mldataset(self, loader, mock_client, sample_data):
        """Test that load_for_training returns MLDataset."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.data = sample_data
        mock_query = MagicMock()
        mock_query.execute = MagicMock(return_value=mock_result)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.gte = MagicMock(return_value=mock_query)
        mock_query.lt = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_client.table.return_value.select.return_value = mock_query

        result = await loader.load_for_training(
            table="business_metrics",
            filters={"brand": "Kisqali"},
        )

        assert isinstance(result, MLDataset)
        assert "table" in result.metadata

    @pytest.mark.asyncio
    async def test_applies_filters(self, loader, mock_client, sample_data):
        """Test that filters are applied to query."""
        mock_result = MagicMock()
        mock_result.data = sample_data
        mock_query = MagicMock()
        mock_query.execute = MagicMock(return_value=mock_result)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.gte = MagicMock(return_value=mock_query)
        mock_query.lt = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_client.table.return_value.select.return_value = mock_query

        await loader.load_for_training(
            table="business_metrics",
            filters={"brand": "Kisqali", "region": "US"},
        )

        # Verify eq was called for filters
        assert mock_query.eq.called

    @pytest.mark.asyncio
    async def test_handles_empty_response(self, loader, mock_client):
        """Test handling of empty data response."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_query = MagicMock()
        mock_query.execute = MagicMock(return_value=mock_result)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.gte = MagicMock(return_value=mock_query)
        mock_query.lt = MagicMock(return_value=mock_query)
        mock_query.order = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_client.table.return_value.select.return_value = mock_query

        result = await loader.load_for_training(table="business_metrics")

        assert result.train_size == 0
        assert result.val_size == 0
        assert result.test_size == 0


class TestLoadTableSample(TestMLDataLoader):
    """Tests for load_table_sample method."""

    @pytest.mark.asyncio
    async def test_raises_error_for_unsupported_table(self, loader):
        """Test that unsupported tables raise ValueError."""
        with pytest.raises(ValueError):
            await loader.load_table_sample(table="unsupported_table")

    @pytest.mark.asyncio
    async def test_returns_dataframe(self, loader, mock_client, sample_data):
        """Test that load_table_sample returns DataFrame."""
        mock_result = MagicMock()
        mock_result.data = sample_data
        mock_query = MagicMock()
        mock_query.execute = MagicMock(return_value=mock_result)
        mock_query.eq = MagicMock(return_value=mock_query)
        mock_query.limit = MagicMock(return_value=mock_query)
        mock_client.table.return_value.select.return_value = mock_query

        result = await loader.load_table_sample(table="business_metrics", limit=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestNoClientBehavior(TestMLDataLoader):
    """Tests for behavior when no client is available."""

    @pytest.fixture
    def loader_no_client(self):
        """Create loader without client."""
        loader = MLDataLoader.__new__(MLDataLoader)
        loader.client = None
        return loader

    @pytest.mark.asyncio
    async def test_load_for_training_returns_empty(self, loader_no_client):
        """Test that empty dataset is returned without client."""
        result = await loader_no_client.load_for_training(table="business_metrics")
        assert result.train_size == 0

    @pytest.mark.asyncio
    async def test_load_table_sample_returns_empty(self, loader_no_client):
        """Test that empty DataFrame is returned without client."""
        result = await loader_no_client.load_table_sample(table="business_metrics")
        assert len(result) == 0


class TestGetMLDataLoader:
    """Tests for get_ml_data_loader function."""

    def test_returns_loader_instance(self):
        """Test that function returns MLDataLoader."""
        loader = get_ml_data_loader(supabase_client=MagicMock())
        assert isinstance(loader, MLDataLoader)


# ----------------------------------------------------- provenance (R7)
class _RecordingQuery:
    """Fluent fake query builder recording every ``.eq`` call."""

    def __init__(self, eq_log):
        self._eq_log = eq_log

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        self._eq_log.append(a)
        return self

    def gte(self, *a, **k):
        return self

    def lt(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        class _R:
            data: list = []

        return _R()


class _RecordingClient:
    def __init__(self, eq_log):
        self._eq_log = eq_log

    def table(self, name):
        return _RecordingQuery(self._eq_log)


class TestProvenanceDefaultExclude:
    """R7: taggable-table reads default-exclude is_synthetic; untagged are skipped."""

    @pytest.mark.asyncio
    async def test_load_table_sample_excludes_synthetic_on_taggable_table(self):
        eq_log: list = []
        loader = MLDataLoader(supabase_client=_RecordingClient(eq_log))
        await loader.load_table_sample("business_metrics", columns=["hcp_id", "trx_count"])
        assert ("is_synthetic", False) in eq_log

    @pytest.mark.asyncio
    async def test_load_table_sample_skips_predicate_on_nontaggable_table(self):
        eq_log: list = []
        loader = MLDataLoader(supabase_client=_RecordingClient(eq_log))
        await loader.load_table_sample("agent_activities", columns=["id"])
        assert ("is_synthetic", False) not in eq_log

    @pytest.mark.asyncio
    async def test_load_table_sample_includes_synthetic_when_opted_in(self):
        eq_log: list = []
        loader = MLDataLoader(supabase_client=_RecordingClient(eq_log))
        await loader.load_table_sample(
            "business_metrics", columns=["hcp_id"], include_synthetic=True
        )
        assert ("is_synthetic", False) not in eq_log

    @pytest.mark.asyncio
    async def test_load_date_range_excludes_synthetic_on_taggable_table(self):
        eq_log: list = []
        loader = MLDataLoader(supabase_client=_RecordingClient(eq_log))
        await loader._load_date_range(
            table="triggers", filters=None, date_column="trigger_timestamp"
        )
        assert ("is_synthetic", False) in eq_log

    @pytest.mark.asyncio
    async def test_count_records_excludes_synthetic_on_taggable_table(self):
        eq_log: list = []
        loader = MLDataLoader(supabase_client=_RecordingClient(eq_log))
        await loader.count_records("patient_journeys")
        assert ("is_synthetic", False) in eq_log

    @pytest.mark.asyncio
    async def test_get_date_range_excludes_synthetic_on_taggable_table(self):
        eq_log: list = []
        loader = MLDataLoader(supabase_client=_RecordingClient(eq_log))
        await loader.get_date_range("business_metrics", date_column="metric_date")
        assert ("is_synthetic", False) in eq_log


class TestHeterogeneousConnectorProvenance:
    """R9: heterogeneous connector threads include_synthetic to MLDataLoader."""

    @pytest.mark.asyncio
    async def test_query_forwards_default_exclude(self):
        from unittest.mock import AsyncMock

        from src.agents.heterogeneous_optimizer.connectors.supabase_connector import (
            HeterogeneousOptimizerDataConnector,
        )

        connector = HeterogeneousOptimizerDataConnector()
        connector._initialized = True
        connector._loader = MagicMock()
        connector._loader.load_table_sample = AsyncMock(return_value=pd.DataFrame())

        await connector.query("business_metrics", columns=["hcp_id"])

        _, kwargs = connector._loader.load_table_sample.call_args
        assert kwargs["include_synthetic"] is False

    @pytest.mark.asyncio
    async def test_query_forwards_opt_in(self):
        from unittest.mock import AsyncMock

        from src.agents.heterogeneous_optimizer.connectors.supabase_connector import (
            HeterogeneousOptimizerDataConnector,
        )

        connector = HeterogeneousOptimizerDataConnector()
        connector._initialized = True
        connector._loader = MagicMock()
        connector._loader.load_table_sample = AsyncMock(return_value=pd.DataFrame())

        await connector.query("business_metrics", columns=["hcp_id"], include_synthetic=True)

        _, kwargs = connector._loader.load_table_sample.call_args
        assert kwargs["include_synthetic"] is True
