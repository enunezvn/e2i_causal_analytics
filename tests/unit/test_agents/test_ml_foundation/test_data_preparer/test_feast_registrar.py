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
async def test_register_features_stale_features_warning(mock_state_with_train_data, monkeypatch):
    """Stale features generate warnings AND the new hard-block contract.

    Defense-in-depth alongside ``test_qc_gate_blocks_on_stale_feast`` — this
    test exercises the same path with slightly different fixture data so we
    fail closed if either assertion regresses.
    """
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

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
    # Without this, ``adapter._feast_client`` would be an auto-spawned
    # MagicMock and ``getattr(mock, "_fallback_used", False)`` would
    # return another truthy MagicMock — so the explicit
    # ``feast_fallback_used is False`` assertion below would fail under
    # the (correct) post-Block-2-polish direct-attribute access.
    adapter._feast_client = None

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    # Warning still surfaced
    assert result["feast_freshness_check"]["fresh"] is False
    assert any("Freshness" in w for w in result["feast_warnings"])
    # Block 2 hard-block contract
    assert result["feast_blocked"] is True
    assert result["feast_registration_status"] == "blocked_stale_features"
    # blocking_issues must be appended so the QC gate forces gate_passed=False
    assert any("Feast features stale" in issue for issue in result.get("blocking_issues", []))
    # Explicit type check — adapter._feast_client=None means the
    # ``if feast_client is not None`` branch in the registrar is skipped,
    # so the key may be absent. When present (e.g. in fallback paths), it
    # must be a real bool, never a MagicMock. (Block 2 polish)
    if "feast_fallback_used" in result:
        assert result["feast_fallback_used"] is False


@pytest.mark.asyncio
async def test_stale_features_advisory_for_file_sourced_run(
    mock_state_with_train_data, monkeypatch
):
    """FIX 3 (leakage over-drop investigation, 2026-06-03): a --data-dir run
    sources its features straight from parquet, NOT from the Feast online store,
    so a stale/unreachable Feast is irrelevant to that data. The freshness check
    must be ADVISORY (warning) for a file-sourced run, never a hard block — even
    without ALLOW_STALE_FEAST. (Genuine Feast-serving runs still hard-block, per
    test_register_features_stale_features_warning.)
    """
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)
    # Mark the run as file-sourced (features loaded from disk, not Feast).
    mock_state_with_train_data["data_source"] = {
        "type": "file_dir",
        "path": "data/rwd/optum_gap_enriched/initiation",
    }

    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={"features_registered": 2, "errors": []}
    )
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": False,
            "stale_features": ["feature_view:feature1"],
            "recommendations": ["Run materialization for feature_view"],
        }
    )
    adapter._feast_client = None

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    # Advisory warning is still surfaced (transparency)...
    assert any("Freshness" in w for w in result["feast_warnings"])
    # ...but it does NOT hard-block a file-sourced run.
    assert result.get("feast_blocked") is False
    assert not any(
        "Feast features stale" in issue for issue in (result.get("blocking_issues") or [])
    )


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
async def test_check_feature_freshness_handles_exception(monkeypatch):
    """On exception, freshness check returns stale dict (fresh=False) by default."""
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

    adapter = MagicMock()
    adapter.check_feature_freshness = AsyncMock(side_effect=Exception("Feast not responding"))

    result = await _check_feature_freshness(
        adapter=adapter,
        experiment_id="exp_test",
        feature_names=["feature1"],
        max_staleness_hours=24.0,
    )

    # Returns a stale dict — not None — so callers can react to the failure.
    assert result is not None
    assert result["fresh"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_check_feature_freshness_allow_stale_on_exception(monkeypatch):
    """ALLOW_STALE_FEAST=1 makes _check_feature_freshness return fresh=True on exception."""
    monkeypatch.setenv("ALLOW_STALE_FEAST", "1")

    adapter = MagicMock()
    adapter.check_feature_freshness = AsyncMock(side_effect=Exception("Feast not responding"))

    result = await _check_feature_freshness(
        adapter=adapter,
        experiment_id="exp_test",
        feature_names=["feature1"],
        max_staleness_hours=24.0,
    )

    assert result is not None
    assert result["fresh"] is True
    assert "warning" in result


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


# ============================================================================
# Block 2 — QC gate tests
# ============================================================================


@pytest.mark.asyncio
async def test_qc_gate_blocks_on_stale_feast(mock_state_with_train_data, monkeypatch):
    """Stale features hard-block training unless ALLOW_STALE_FEAST=1."""
    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={"features_registered": 2, "errors": []}
    )
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": False,
            "error": "stale",
            "recommendations": ["Run materialization for feature_analyzer_exp_feast_test_123"],
        }
    )
    # Simulate adapter with no backing FeastClient so fallback flag stays False
    adapter._feast_client = None

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    # Hard block must be set
    assert result.get("feast_blocked") is True
    assert result.get("feast_registration_status") == "blocked_stale_features"
    # Freshness warning propagated
    assert any("Freshness" in w for w in result["feast_warnings"])
    # blocking_issues appended so _finalize_output's gate logic forces gate_passed=False
    assert "blocking_issues" in result
    assert any("Feast features stale" in issue for issue in result["blocking_issues"])


@pytest.mark.asyncio
async def test_qc_gate_allows_with_allow_stale_env(mock_state_with_train_data, monkeypatch):
    """ALLOW_STALE_FEAST=1 bypasses the hard block (warnings only)."""
    monkeypatch.setenv("ALLOW_STALE_FEAST", "1")

    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={"features_registered": 2, "errors": []}
    )
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": False,
            "error": "stale",
            "recommendations": ["Run materialization for feature_analyzer_exp_feast_test_123"],
        }
    )
    adapter._feast_client = None

    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        result = await register_features_in_feast(mock_state_with_train_data)

    # No hard block when ALLOW_STALE_FEAST is set
    assert result.get("feast_blocked") is False
    assert result.get("feast_registration_status") != "blocked_stale_features"
    # blocking_issues NOT appended in the bypass path
    assert not any(
        "Feast features stale" in issue for issue in (result.get("blocking_issues", []) or [])
    )


@pytest.mark.asyncio
async def test_stale_feast_blocks_finalize_output_gate(mock_state_with_train_data, monkeypatch):
    """End-to-end: stale features → registrar appends blocking_issues → finalize_output flips gate_passed=False."""
    from src.agents.ml_foundation.data_preparer.graph import finalize_output

    monkeypatch.delenv("ALLOW_STALE_FEAST", raising=False)

    adapter = MagicMock()
    adapter.register_features_from_state = AsyncMock(
        return_value={"features_registered": 2, "errors": []}
    )
    adapter.check_feature_freshness = AsyncMock(
        return_value={
            "fresh": False,
            "error": "stale",
            "recommendations": ["Run materialization"],
        }
    )
    adapter._feast_client = None

    # Run the registrar
    with patch(
        "src.agents.ml_foundation.data_preparer.nodes.feast_registrar._get_feature_analyzer_adapter",
        return_value=adapter,
    ):
        registrar_updates = await register_features_in_feast(mock_state_with_train_data)

    # Build the post-registrar state (simulate LangGraph state merge)
    post_state = {**mock_state_with_train_data, **registrar_updates}
    # Simulate other QC checks having passed cleanly so the only blocker is Feast
    post_state["qc_status"] = "passed"
    post_state["overall_score"] = 0.95
    post_state["train_df"] = pd.DataFrame({"feature1": [1.0]})
    post_state["validation_df"] = pd.DataFrame({"feature1": [1.0]})
    post_state["test_df"] = pd.DataFrame({"feature1": [1.0]})
    post_state["holdout_df"] = pd.DataFrame({"feature1": [1.0]})

    # Run finalize_output
    final_updates = await finalize_output(post_state)

    # Gate must be blocked because of the Feast blocking_issue
    assert final_updates["gate_passed"] is False
    assert final_updates["qc_passed"] is False
    assert any("Feast features stale" in blocker for blocker in final_updates["blockers"])
