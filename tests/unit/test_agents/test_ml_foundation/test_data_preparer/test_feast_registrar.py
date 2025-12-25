"""Unit tests for feast_registrar node."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.feast_registrar import (
    _check_feature_freshness,
    register_features_in_feast,
)


@pytest.fixture
def mock_state_with_train_data():
    """Create mock state with training data."""
    train_df = pd.DataFrame(
        {
            "hcp_id": ["hcp_001", "hcp_002", "hcp_003"],
            "feature1": np.random.randn(3),
            "feature2": np.random.randn(3),
            "target": [0, 1, 0],
        }
    )

    return {
        "experiment_id": "exp_feast_test_123",
        "train_df": train_df,
        "data_source": "hcp_features",
        "scope_spec": {
            "experiment_id": "exp_feast_test_123",
            "required_features": ["feature1", "feature2"],
            "entity_key": "hcp_id",
            "prediction_target": "target",
        },
    }


@pytest.fixture
def mock_state_minimal():
    """Create minimal mock state without train data."""
    return {
        "experiment_id": "exp_minimal_123",
        "scope_spec": {
            "experiment_id": "exp_minimal_123",
            "required_features": ["feature1"],
        },
    }


@pytest.fixture
def mock_adapter():
    """Create mock FeatureAnalyzerAdapter."""
    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={
            "feature_group_created": True,
            "features_registered": 2,
            "features_skipped": 0,
            "errors": [],
        }
    )
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": True,
            "stale_features": [],
            "feature_ages": {"feature_analyzer_exp_feast_test_123:feature1": 1.5},
            "recommendations": [],
        }
    )
    return adapter


@pytest.mark.asyncio
async def test_register_features_when_adapter_unavailable(mock_state_with_train_data):
    """Test registration when adapter is unavailable."""
    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=None,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_registration_status"] == "skipped"
    assert result["feast_features_registered"] == 0
    assert any("not available" in w for w in result["feast_warnings"])


@pytest.mark.asyncio
async def test_register_features_when_no_train_data(mock_state_minimal):
    """Test registration when train data is missing."""
    mock_adapter = MagicMock()
    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=mock_adapter,
    ):
        result = await register_features_in_feast(mock_state_minimal)

    assert result["feast_registration_status"] == "skipped"
    assert any("No training data" in w for w in result["feast_warnings"])


@pytest.mark.asyncio
async def test_register_features_success(mock_state_with_train_data, mock_adapter):
    """Test successful feature registration."""
    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=mock_adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_registration_status"] == "completed"
    assert result["feast_features_registered"] == 2
    assert result["feast_registered_at"] is not None

    # Verify adapter was called correctly
    mock_adapter.register_features_from_state.assert_called_once()
    call_kwargs = mock_adapter.register_features_from_state.call_args[1]
    assert call_kwargs["experiment_id"] == "exp_feast_test_123"
    assert call_kwargs["entity_key"] == "hcp_id"
    assert call_kwargs["owner"] == "data_preparer"


@pytest.mark.asyncio
async def test_register_features_with_adapter_errors(mock_state_with_train_data):
    """Test registration when adapter returns errors."""
    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={
            "feature_group_created": True,
            "features_registered": 1,
            "features_skipped": 0,
            "errors": [{"feature": "feature2", "error": "Registration failed"}],
        }
    )
    adapter.check_feature_freshness = AsyncMock(return_value=None)

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_registration_status"] == "completed"
    assert result["feast_features_registered"] == 1
    assert any("Registration error" in w for w in result["feast_warnings"])


@pytest.mark.asyncio
async def test_register_features_freshness_check(mock_state_with_train_data, mock_adapter):
    """Test that freshness check is included in registration."""
    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=mock_adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_freshness_check"] is not None
    assert result["feast_freshness_check"]["fresh"] is True

    # Verify freshness check was called
    mock_adapter.check_feature_freshness.assert_called_once()


@pytest.mark.asyncio
async def test_register_features_stale_features_warning(mock_state_with_train_data):
    """Test that stale features generate warnings."""
    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={
            "features_registered": 2,
            "errors": [],
        }
    )
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": False,
            "stale_features": ["feature_view:feature1"],
            "feature_ages": {"feature_view:feature1": 48.5},
            "recommendations": ["Run materialization for feature_view"],
        }
    )

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_freshness_check"]["fresh"] is False
    assert any("Freshness" in w for w in result["feast_warnings"])


@pytest.mark.asyncio
async def test_register_features_handles_exception(mock_state_with_train_data):
    """Test that exceptions are handled gracefully."""
    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(side_effect=Exception("Feast unavailable"))

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_registration_status"] == "error"
    assert any("Registration error" in w for w in result["feast_warnings"])


@pytest.mark.asyncio
async def test_register_features_empty_result(mock_state_with_train_data):
    """Test registration when no features are registered."""
    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={
            "features_registered": 0,
            "errors": [],
        }
    )
    adapter.check_feature_freshness = AsyncMock(return_value=None)

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    assert result["feast_registration_status"] == "empty"
    assert result["feast_features_registered"] == 0


@pytest.mark.asyncio
async def test_check_feature_freshness_helper():
    """Test the _check_feature_freshness helper function."""
    adapter = MagicMock()
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": True,
            "stale_features": [],
            "feature_ages": {},
        }
    )

    result = await _check_feature_freshness(
        adapter=adapter,
        experiment_id="exp_test",
        feature_names=["feature1", "feature2"],
        max_staleness_hours=24.0,
    )

    assert result["fresh"] is True
    adapter.check_feature_freshness.assert_called_once()


@pytest.mark.asyncio
async def test_check_feature_freshness_handles_exception():
    """Test that freshness check handles exceptions gracefully."""
    adapter = MagicMock()
    adapter.check_feature_freshness = AsyncMock(
        side_effect=Exception("Feast not responding")
    )

    result = await _check_feature_freshness(
        adapter=adapter,
        experiment_id="exp_test",
        feature_names=["feature1"],
        max_staleness_hours=24.0,
    )

    # Should return None on failure (non-critical)
    assert result is None


@pytest.mark.asyncio
async def test_register_features_timestamp_format(mock_state_with_train_data, mock_adapter):
    """Test that registered_at timestamp is in valid ISO format."""
    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=mock_adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    # Should be valid ISO timestamp
    timestamp = result["feast_registered_at"]
    assert timestamp is not None
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
