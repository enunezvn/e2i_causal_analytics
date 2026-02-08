"""
Comprehensive unit tests for src/feature_store/client.py

Tests cover:
- FeatureStoreClient initialization
- Feature group management (create, get, list)
- Feature management (create, get, list)
- Feature retrieval (online, historical)
- Feature writing (single, batch)
- Monitoring and statistics
- MLflow integration
- Health checks
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.feature_store.client import FeatureStoreClient
from src.feature_store.models import (
    EntityFeatures,
    Feature,
    FeatureGroup,
    FeatureStatistics,
    FeatureValue,
)

# =============================================================================
# Test Initialization
# =============================================================================


@patch("src.feature_store.client.create_client")
@patch("src.feature_store.client.redis.from_url")
def test_client_init_success(mock_redis, mock_supabase):
    """Test successful client initialization."""
    mock_supabase_client = MagicMock()
    mock_supabase.return_value = mock_supabase_client

    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_redis.return_value = mock_redis_client

    client = FeatureStoreClient(
        supabase_url="http://test",
        supabase_key="test-key",
        redis_url="redis://test:6379",
    )

    assert client.supabase == mock_supabase_client
    assert client.redis_client == mock_redis_client
    assert client.enable_cache is True


@patch("src.feature_store.client.create_client")
def test_client_init_without_redis(mock_supabase):
    """Test initialization when Redis is unavailable."""
    mock_supabase_client = MagicMock()
    mock_supabase.return_value = mock_supabase_client

    with patch(
        "src.feature_store.client.redis.from_url", side_effect=Exception("Connection failed")
    ):
        client = FeatureStoreClient(
            supabase_url="http://test",
            supabase_key="test-key",
            enable_cache=True,
        )

        assert client.redis_client is None
        assert client.enable_cache is False


@patch.dict("os.environ", {}, clear=True)
def test_client_init_missing_credentials():
    """Test initialization fails without credentials."""
    with pytest.raises(ValueError, match="Supabase URL and key are required"):
        FeatureStoreClient()


@patch.dict("os.environ", {"SUPABASE_URL": "http://test", "SUPABASE_KEY": "test-key"})
@patch("src.feature_store.client.create_client")
@patch("src.feature_store.client.redis.from_url")
def test_client_init_from_env(mock_redis, mock_supabase):
    """Test initialization from environment variables."""
    mock_supabase_client = MagicMock()
    mock_supabase.return_value = mock_supabase_client

    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_redis.return_value = mock_redis_client

    FeatureStoreClient()

    mock_supabase.assert_called_once_with("http://test", "test-key")


# =============================================================================
# Test Feature Group Management
# =============================================================================


@pytest.fixture
def mock_client():
    """Create a mock FeatureStoreClient."""
    with patch("src.feature_store.client.create_client"):
        with patch("src.feature_store.client.redis.from_url"):
            client = FeatureStoreClient(
                supabase_url="http://test",
                supabase_key="test-key",
                enable_cache=False,
            )
            yield client


def test_create_feature_group(mock_client):
    """Test creating a feature group."""
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": str(uuid4()),
            "name": "test_group",
            "description": "Test group",
            "owner": "test_owner",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    mock_table.insert.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    feature_group = mock_client.create_feature_group(
        name="test_group",
        description="Test group",
        owner="test_owner",
    )

    assert isinstance(feature_group, FeatureGroup)
    assert feature_group.name == "test_group"
    mock_table.insert.assert_called_once()


def test_get_feature_group(mock_client):
    """Test getting a feature group."""
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": str(uuid4()),
            "name": "test_group",
            "description": "Test group",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    mock_table.select.return_value.eq.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    feature_group = mock_client.get_feature_group("test_group")

    assert feature_group is not None
    assert feature_group.name == "test_group"


def test_get_feature_group_not_found(mock_client):
    """Test getting a non-existent feature group."""
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = []
    mock_table.select.return_value.eq.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    feature_group = mock_client.get_feature_group("nonexistent")

    assert feature_group is None


def test_list_feature_groups(mock_client):
    """Test listing feature groups."""
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": str(uuid4()),
            "name": "group1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid4()),
            "name": "group2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    mock_table.select.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    feature_groups = mock_client.list_feature_groups()

    assert len(feature_groups) == 2
    assert all(isinstance(fg, FeatureGroup) for fg in feature_groups)


# =============================================================================
# Test Feature Management
# =============================================================================


def test_create_feature(mock_client):
    """Test creating a feature."""
    # Mock get_feature_group
    mock_client.get_feature_group = MagicMock(
        return_value=FeatureGroup(
            id=uuid4(),
            name="test_group",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    # Mock table insert
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {
            "id": str(uuid4()),
            "feature_group_id": str(uuid4()),
            "name": "test_feature",
            "value_type": "float64",
            "entity_keys": ["hcp_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    mock_table.upsert.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    feature = mock_client.create_feature(
        feature_group_name="test_group",
        name="test_feature",
        value_type="float64",
        entity_keys=["hcp_id"],
    )

    assert isinstance(feature, Feature)
    assert feature.name == "test_feature"


def test_create_feature_invalid_group(mock_client):
    """Test creating feature with invalid group."""
    mock_client.get_feature_group = MagicMock(return_value=None)

    with pytest.raises(ValueError, match="Feature group not found"):
        mock_client.create_feature(
            feature_group_name="nonexistent",
            name="test_feature",
            value_type="float64",
            entity_keys=["hcp_id"],
        )


# =============================================================================
# Test Feature Retrieval
# =============================================================================


def test_get_entity_features(mock_client):
    """Test getting entity features."""
    mock_retriever = MagicMock()
    mock_retriever.get_entity_features.return_value = EntityFeatures(
        entity_values={"hcp_id": "HCP123"},
        features={},
        feature_group=None,
        retrieved_at=datetime.now(timezone.utc),
        stale_features=[],
    )
    mock_client.retriever = mock_retriever

    features = mock_client.get_entity_features(
        entity_values={"hcp_id": "HCP123"},
        feature_group="hcp_demographics",
    )

    assert isinstance(features, EntityFeatures)
    mock_retriever.get_entity_features.assert_called_once()


def test_get_historical_features(mock_client):
    """Test getting historical features."""
    mock_retriever = MagicMock()
    mock_retriever.get_historical_features.return_value = [
        {
            "feature_name": "specialty",
            "value": "Oncology",
            "event_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ]
    mock_client.retriever = mock_retriever

    features = mock_client.get_historical_features(
        entity_values={"hcp_id": "HCP123"},
        feature_names=["specialty"],
    )

    assert len(features) >= 0
    mock_retriever.get_historical_features.assert_called_once()


# =============================================================================
# Test Feature Writing
# =============================================================================


def test_write_feature_value(mock_client):
    """Test writing a single feature value."""
    mock_writer = MagicMock()
    mock_writer.write_feature_value.return_value = FeatureValue(
        id=uuid4(),
        feature_id=uuid4(),
        entity_values={"hcp_id": "HCP123"},
        value="Oncology",
        event_timestamp=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )
    mock_client.writer = mock_writer

    feature_value = mock_client.write_feature_value(
        feature_name="specialty",
        entity_values={"hcp_id": "HCP123"},
        value="Oncology",
        event_timestamp=datetime.now(timezone.utc),
    )

    assert isinstance(feature_value, FeatureValue)
    mock_writer.write_feature_value.assert_called_once()


def test_write_batch_features(mock_client):
    """Test writing batch features."""
    mock_writer = MagicMock()
    mock_writer.write_batch_features.return_value = 3
    mock_client.writer = mock_writer

    count = mock_client.write_batch_features(
        feature_values=[
            {
                "feature_name": "specialty",
                "entity_values": {"hcp_id": "HCP123"},
                "value": "Oncology",
                "event_timestamp": datetime.now(timezone.utc),
            },
            {
                "feature_name": "specialty",
                "entity_values": {"hcp_id": "HCP456"},
                "value": "Cardiology",
                "event_timestamp": datetime.now(timezone.utc),
            },
        ]
    )

    assert count == 3
    mock_writer.write_batch_features.assert_called_once()


# =============================================================================
# Test Statistics
# =============================================================================


def test_get_feature_statistics(mock_client):
    """Test getting feature statistics."""
    # Mock get_feature
    mock_client.get_feature = MagicMock(
        return_value=Feature(
            id=uuid4(),
            feature_group_id=uuid4(),
            name="test_feature",
            value_type="float64",
            entity_keys=["hcp_id"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    # Mock Supabase query
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = [
        {"value": 10.5, "event_timestamp": datetime.now(timezone.utc).isoformat()},
        {"value": 20.3, "event_timestamp": datetime.now(timezone.utc).isoformat()},
        {"value": 15.7, "event_timestamp": datetime.now(timezone.utc).isoformat()},
    ]
    mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    stats = mock_client.get_feature_statistics("test_group", "test_feature")

    assert isinstance(stats, FeatureStatistics)
    assert stats.count == 3
    assert stats.mean is not None


def test_get_feature_statistics_no_data(mock_client):
    """Test getting statistics for feature with no data."""
    mock_client.get_feature = MagicMock(
        return_value=Feature(
            id=uuid4(),
            feature_group_id=uuid4(),
            name="test_feature",
            value_type="float64",
            entity_keys=["hcp_id"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = []
    mock_table.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    stats = mock_client.get_feature_statistics("test_group", "test_feature")

    assert stats.count == 0
    assert stats.mean is None


# =============================================================================
# Test Health Check
# =============================================================================


def test_health_check_all_healthy(mock_client):
    """Test health check when all services are healthy."""
    # Mock Supabase
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = []
    mock_table.select.return_value.limit.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    # Mock Redis
    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_client.redis_client = mock_redis_client

    # Mock MLflow
    with patch("src.feature_store.client.mlflow.get_tracking_uri", return_value="http://test"):
        health = mock_client.health_check()

    assert health["supabase"] is True
    assert health["redis"] is True
    assert health["mlflow"] is True


def test_health_check_partial_failure(mock_client):
    """Test health check with some services failing."""
    # Mock Supabase (success)
    mock_table = MagicMock()
    mock_result = MagicMock()
    mock_result.data = []
    mock_table.select.return_value.limit.return_value.execute.return_value = mock_result
    mock_client.supabase.table = MagicMock(return_value=mock_table)

    # Redis is None (failed during init)
    mock_client.redis_client = None

    # MLflow fails
    with patch(
        "src.feature_store.client.mlflow.get_tracking_uri", side_effect=Exception("MLflow down")
    ):
        health = mock_client.health_check()

    assert health["supabase"] is True
    assert health["redis"] is False
    assert health["mlflow"] is False


# =============================================================================
# Test Close
# =============================================================================


def test_close_with_redis(mock_client):
    """Test closing client with Redis."""
    mock_redis_client = MagicMock()
    mock_client.redis_client = mock_redis_client

    mock_client.close()

    mock_redis_client.close.assert_called_once()


def test_close_without_redis(mock_client):
    """Test closing client without Redis."""
    mock_client.redis_client = None

    # Should not raise
    mock_client.close()
