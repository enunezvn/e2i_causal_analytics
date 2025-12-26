"""
Unit tests for ShapAnalysisRepository.

Tests CRUD operations for ml_shap_analyses table.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.shap_analysis import ShapAnalysisRepository, get_shap_analysis_repository


class TestShapAnalysisRepository:
    """Tests for ShapAnalysisRepository."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Supabase client."""
        client = MagicMock()
        return client

    @pytest.fixture
    def repo(self, mock_client):
        """Create repository with mock client."""
        return ShapAnalysisRepository(supabase_client=mock_client)

    @pytest.fixture
    def sample_analysis_dict(self):
        """Sample SHAP analysis data."""
        return {
            "experiment_id": "exp-123",
            "feature_importance": [
                {"feature": "days_since_last_visit", "importance": 0.35},
                {"feature": "therapy_adherence_score", "importance": 0.28},
                {"feature": "total_hcp_interactions", "importance": 0.15},
            ],
            "interactions": [
                {
                    "features": ["days_since_last_visit", "therapy_adherence_score"],
                    "interaction_strength": 0.12,
                    "interpretation": "Strong positive interaction",
                },
            ],
            "interpretation": "Key drivers are recency and adherence metrics.",
            "top_features": ["days_since_last_visit", "therapy_adherence_score", "total_hcp_interactions"],
            "samples_analyzed": 500,
            "computation_time_seconds": 45.2,
            "explainer_type": "TreeExplainer",
            "model_version": "v2.3.1",
        }

    @pytest.fixture
    def sample_db_record(self):
        """Sample database record as returned from Supabase."""
        return {
            "id": str(uuid.uuid4()),
            "model_registry_id": "reg-456",
            "analysis_type": "global",
            "global_importance": {
                "days_since_last_visit": 0.35,
                "therapy_adherence_score": 0.28,
                "total_hcp_interactions": 0.15,
            },
            "top_interactions": [
                {
                    "feature_1": "days_since_last_visit",
                    "feature_2": "therapy_adherence_score",
                    "interaction_strength": 0.12,
                    "interpretation": "Strong positive interaction",
                },
            ],
            "natural_language_explanation": "Key drivers are recency and adherence metrics.",
            "key_drivers": ["days_since_last_visit", "therapy_adherence_score", "total_hcp_interactions"],
            "sample_size": 500,
            "computation_duration_seconds": 45,
            "computation_method": "TreeExplainer",
            "computed_at": "2025-01-15T10:30:00Z",
        }


class TestStoreAnalysis(TestShapAnalysisRepository):
    """Tests for store_analysis method."""

    @pytest.mark.asyncio
    async def test_stores_analysis_successfully(self, repo, mock_client, sample_analysis_dict, sample_db_record):
        """Test that analysis is stored correctly."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.data = [sample_db_record]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_insert = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_client.table.return_value.insert = mock_insert

        # Execute
        result = await repo.store_analysis(
            analysis_dict=sample_analysis_dict,
            model_registry_id="reg-456",
        )

        # Verify
        assert result is not None
        assert result["id"] == sample_db_record["id"]
        mock_client.table.assert_called_with("ml_shap_analyses")
        mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_stores_analysis_without_registry_id(self, repo, mock_client, sample_analysis_dict, sample_db_record):
        """Test that analysis can be stored without model_registry_id."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.data = [sample_db_record]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_insert = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_client.table.return_value.insert = mock_insert

        # Execute
        result = await repo.store_analysis(analysis_dict=sample_analysis_dict)

        # Verify
        assert result is not None
        mock_insert.assert_called_once()
        # Check that model_registry_id is not in the call if None
        call_args = mock_insert.call_args[0][0]
        assert "model_registry_id" not in call_args or call_args["model_registry_id"] is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self, sample_analysis_dict):
        """Test that None is returned when no client is available."""
        repo = ShapAnalysisRepository(supabase_client=None)

        result = await repo.store_analysis(analysis_dict=sample_analysis_dict)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_insert_failure(self, repo, mock_client, sample_analysis_dict):
        """Test that None is returned when insert fails."""
        # Setup mock to return empty data
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)
        mock_insert = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_client.table.return_value.insert = mock_insert

        result = await repo.store_analysis(analysis_dict=sample_analysis_dict)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, repo, mock_client, sample_analysis_dict):
        """Test that exceptions are handled gracefully."""
        # Setup mock to raise exception
        mock_execute = AsyncMock(side_effect=Exception("Database error"))
        mock_insert = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_client.table.return_value.insert = mock_insert

        result = await repo.store_analysis(analysis_dict=sample_analysis_dict)

        assert result is None

    @pytest.mark.asyncio
    async def test_builds_correct_db_record(self, repo, mock_client, sample_analysis_dict):
        """Test that the database record is built correctly from analysis dict."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.data = [{"id": "test-id"}]
        mock_execute = AsyncMock(return_value=mock_result)
        mock_insert = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_client.table.return_value.insert = mock_insert

        await repo.store_analysis(
            analysis_dict=sample_analysis_dict,
            model_registry_id="reg-123",
        )

        # Get the record that was inserted
        call_args = mock_insert.call_args[0][0]

        # Verify field mappings
        assert call_args["analysis_type"] == "global"
        assert call_args["model_registry_id"] == "reg-123"
        assert call_args["global_importance"]["days_since_last_visit"] == 0.35
        assert call_args["computation_duration_seconds"] == 45  # Integer, not float
        assert call_args["computation_method"] == "TreeExplainer"
        assert call_args["sample_size"] == 500
        assert len(call_args["key_drivers"]) == 3


class TestGetByModelRegistryId(TestShapAnalysisRepository):
    """Tests for get_by_model_registry_id method."""

    @pytest.mark.asyncio
    async def test_returns_analyses_for_model(self, repo, mock_client, sample_db_record):
        """Test that analyses are returned for a model."""
        # Setup mock chain
        mock_result = MagicMock()
        mock_result.data = [sample_db_record]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_order = MagicMock(return_value=MagicMock(limit=mock_limit))
        mock_eq2 = MagicMock(return_value=MagicMock(order=mock_order))
        mock_eq1 = MagicMock(return_value=MagicMock(eq=mock_eq2))
        mock_select = MagicMock(return_value=MagicMock(eq=mock_eq1))
        mock_client.table.return_value.select = mock_select

        result = await repo.get_by_model_registry_id(
            model_registry_id="reg-456",
            analysis_type="global",
            limit=10,
        )

        assert len(result) == 1
        assert result[0]["id"] == sample_db_record["id"]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_client(self):
        """Test that empty list is returned when no client."""
        repo = ShapAnalysisRepository(supabase_client=None)

        result = await repo.get_by_model_registry_id(model_registry_id="reg-456")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self, repo, mock_client):
        """Test that empty list is returned on error."""
        mock_execute = AsyncMock(side_effect=Exception("Database error"))
        mock_limit = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_order = MagicMock(return_value=MagicMock(limit=mock_limit))
        mock_eq2 = MagicMock(return_value=MagicMock(order=mock_order))
        mock_eq1 = MagicMock(return_value=MagicMock(eq=mock_eq2))
        mock_select = MagicMock(return_value=MagicMock(eq=mock_eq1))
        mock_client.table.return_value.select = mock_select

        result = await repo.get_by_model_registry_id(model_registry_id="reg-456")

        assert result == []


class TestGetLatestForModel(TestShapAnalysisRepository):
    """Tests for get_latest_for_model method."""

    @pytest.mark.asyncio
    async def test_returns_latest_analysis(self, repo, mock_client, sample_db_record):
        """Test that the latest analysis is returned."""
        # Setup mock chain
        mock_result = MagicMock()
        mock_result.data = [sample_db_record]
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_order = MagicMock(return_value=MagicMock(limit=mock_limit))
        mock_eq = MagicMock(return_value=MagicMock(order=mock_order))
        mock_select = MagicMock(return_value=MagicMock(eq=mock_eq))
        mock_client.table.return_value.select = mock_select

        result = await repo.get_latest_for_model(model_registry_id="reg-456")

        assert result is not None
        assert result["id"] == sample_db_record["id"]

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data(self, repo, mock_client):
        """Test that None is returned when no data exists."""
        mock_result = MagicMock()
        mock_result.data = []
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_order = MagicMock(return_value=MagicMock(limit=mock_limit))
        mock_eq = MagicMock(return_value=MagicMock(order=mock_order))
        mock_select = MagicMock(return_value=MagicMock(eq=mock_eq))
        mock_client.table.return_value.select = mock_select

        result = await repo.get_latest_for_model(model_registry_id="reg-456")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_client(self):
        """Test that None is returned when no client."""
        repo = ShapAnalysisRepository(supabase_client=None)

        result = await repo.get_latest_for_model(model_registry_id="reg-456")

        assert result is None


class TestGetFeatureImportanceTrends(TestShapAnalysisRepository):
    """Tests for get_feature_importance_trends method."""

    @pytest.mark.asyncio
    async def test_returns_trends(self, repo, mock_client):
        """Test that feature importance trends are returned."""
        trend_data = [
            {
                "id": "id-1",
                "global_importance": {"feature_a": 0.35, "feature_b": 0.25},
                "computed_at": "2025-01-15T10:00:00Z",
            },
            {
                "id": "id-2",
                "global_importance": {"feature_a": 0.32, "feature_b": 0.28},
                "computed_at": "2025-01-14T10:00:00Z",
            },
        ]

        mock_result = MagicMock()
        mock_result.data = trend_data
        mock_execute = AsyncMock(return_value=mock_result)

        mock_limit = MagicMock(return_value=MagicMock(execute=mock_execute))
        mock_order = MagicMock(return_value=MagicMock(limit=mock_limit))
        mock_eq2 = MagicMock(return_value=MagicMock(order=mock_order))
        mock_eq1 = MagicMock(return_value=MagicMock(eq=mock_eq2))
        mock_select = MagicMock(return_value=MagicMock(eq=mock_eq1))
        mock_client.table.return_value.select = mock_select

        result = await repo.get_feature_importance_trends(
            model_registry_id="reg-456",
            limit=30,
        )

        assert len(result) == 2
        assert result[0]["global_importance"]["feature_a"] == 0.35

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_client(self):
        """Test that empty list is returned when no client."""
        repo = ShapAnalysisRepository(supabase_client=None)

        result = await repo.get_feature_importance_trends(model_registry_id="reg-456")

        assert result == []


class TestGetShapAnalysisRepository:
    """Tests for get_shap_analysis_repository singleton function."""

    def test_returns_repository_instance(self, monkeypatch):
        """Test that a repository instance is returned."""
        # Reset singleton
        import src.repositories.shap_analysis as module
        monkeypatch.setattr(module, "_shap_analysis_repository", None)

        repo = get_shap_analysis_repository()

        assert isinstance(repo, ShapAnalysisRepository)

    def test_returns_same_instance(self, monkeypatch):
        """Test that the same instance is returned on subsequent calls."""
        import src.repositories.shap_analysis as module
        monkeypatch.setattr(module, "_shap_analysis_repository", None)

        repo1 = get_shap_analysis_repository()
        repo2 = get_shap_analysis_repository()

        assert repo1 is repo2
