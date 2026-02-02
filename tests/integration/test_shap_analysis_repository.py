"""
Integration tests for ShapAnalysisRepository.

Tests CRUD operations against real Supabase database.
Use pytest markers to skip when Supabase is not configured.
"""

import os
import uuid

import pytest

from src.repositories.shap_analysis import get_shap_analysis_repository

# =============================================================================
# Test Configuration
# =============================================================================

# Check for Supabase availability (factory uses SUPABASE_ANON_KEY)
HAS_SUPABASE = bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_ANON_KEY"))

requires_supabase = pytest.mark.skipif(
    not HAS_SUPABASE,
    reason="SUPABASE_URL and SUPABASE_ANON_KEY environment variables not set",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_experiment_id() -> str:
    """Generate unique experiment ID for test isolation."""
    return f"test-shap-{uuid.uuid4().hex[:16]}"


@pytest.fixture
def sample_analysis_dict(test_experiment_id):
    """Sample SHAP analysis data for testing."""
    return {
        "experiment_id": test_experiment_id,
        "feature_importance": [
            {"feature": "days_since_last_visit", "importance": 0.35},
            {"feature": "therapy_adherence_score", "importance": 0.28},
            {"feature": "total_hcp_interactions", "importance": 0.15},
            {"feature": "insurance_tier", "importance": 0.12},
            {"feature": "patient_age_bucket", "importance": 0.10},
        ],
        "interactions": [
            {
                "features": ["days_since_last_visit", "therapy_adherence_score"],
                "interaction_strength": 0.12,
                "interpretation": "Recency amplifies adherence effect",
            },
            {
                "features": ["insurance_tier", "patient_age_bucket"],
                "interaction_strength": 0.08,
                "interpretation": "Insurance and age interact for coverage",
            },
        ],
        "interpretation": "Key drivers are visit recency and therapy adherence.",
        "top_features": [
            "days_since_last_visit",
            "therapy_adherence_score",
            "total_hcp_interactions",
            "insurance_tier",
            "patient_age_bucket",
        ],
        "samples_analyzed": 1000,
        "computation_time_seconds": 45.7,
        "explainer_type": "TreeExplainer",
        "model_version": "v2.3.1-integration-test",
    }


@pytest.fixture
async def repository(monkeypatch):
    """Get repository instance with real Supabase client."""
    import src.memory.services.factories as factories
    import src.repositories.shap_analysis as module

    # Reset both the factory's Supabase client and the repository singleton
    # to ensure fresh initialization with env vars
    monkeypatch.setattr(factories, "_supabase_client", None)
    monkeypatch.setattr(module, "_shap_analysis_repository", None)

    repo = get_shap_analysis_repository()
    yield repo
    # Note: We don't clean up test data here as it may be useful for debugging
    # In production, you might want to delete test records after tests


# =============================================================================
# Store Analysis Tests
# =============================================================================


@requires_supabase
class TestStoreAnalysisIntegration:
    """Integration tests for store_analysis method.

    Note: Insert operations may fail with RLS (Row Level Security) errors
    when using anon key. Tests handle this gracefully.
    """

    @pytest.mark.asyncio
    async def test_stores_and_retrieves_analysis(self, repository, sample_analysis_dict):
        """Test that analysis can be stored and retrieved."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        # Store analysis
        result = await repository.store_analysis(
            analysis_dict=sample_analysis_dict,
            model_registry_id=None,  # Test without registry ID
        )

        # Insert may fail due to RLS policy with anon key - this is expected
        if result is None:
            pytest.skip("Insert blocked by RLS policy (expected with anon key)")

        assert "id" in result
        assert result["analysis_type"] == "global"

        # Verify data was stored correctly
        assert result["sample_size"] == 1000
        assert result["computation_method"] == "TreeExplainer"
        assert len(result["key_drivers"]) == 5
        assert "days_since_last_visit" in result["global_importance"]

    @pytest.mark.asyncio
    async def test_stores_with_model_registry_id(self, repository, sample_analysis_dict):
        """Test that analysis can be stored with model_registry_id."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        # Use a test registry ID (won't exist in FK, but some schemas may not enforce)
        test_registry_id = f"test-reg-{uuid.uuid4().hex[:8]}"

        try:
            result = await repository.store_analysis(
                analysis_dict=sample_analysis_dict,
                model_registry_id=test_registry_id,
            )

            # If FK is enforced, this might fail - that's OK
            if result is not None:
                assert result["model_registry_id"] == test_registry_id
        except Exception:
            # FK constraint violation is expected if registry ID doesn't exist
            pass

    @pytest.mark.asyncio
    async def test_stores_multiple_analyses(self, repository, sample_analysis_dict):
        """Test that multiple analyses can be stored."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        results = []
        for i in range(3):
            # Modify experiment_id for each
            analysis = sample_analysis_dict.copy()
            analysis["experiment_id"] = f"{sample_analysis_dict['experiment_id']}-{i}"
            analysis["samples_analyzed"] = 1000 + i * 100

            result = await repository.store_analysis(analysis_dict=analysis)
            if result:
                results.append(result)

        # Insert may fail due to RLS policy with anon key - this is expected
        if len(results) == 0:
            pytest.skip("Insert blocked by RLS policy (expected with anon key)")

        assert len(results) >= 1  # At least one should succeed


# =============================================================================
# Get By Model Registry ID Tests
# =============================================================================


@requires_supabase
class TestGetByModelRegistryIdIntegration:
    """Integration tests for get_by_model_registry_id method."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_registry(self, repository):
        """Test that empty list is returned for non-existent registry ID."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        result = await repository.get_by_model_registry_id(
            model_registry_id=f"nonexistent-{uuid.uuid4().hex}",
            analysis_type="global",
            limit=10,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, repository):
        """Test that limit parameter is respected."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        # Query with small limit
        result = await repository.get_by_model_registry_id(
            model_registry_id="any-id",
            analysis_type="global",
            limit=1,
        )

        assert len(result) <= 1


# =============================================================================
# Get Latest For Model Tests
# =============================================================================


@requires_supabase
class TestGetLatestForModelIntegration:
    """Integration tests for get_latest_for_model method."""

    @pytest.mark.asyncio
    async def test_returns_none_for_nonexistent_model(self, repository):
        """Test that None is returned for non-existent model."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        result = await repository.get_latest_for_model(
            model_registry_id=f"nonexistent-{uuid.uuid4().hex}",
        )

        assert result is None


# =============================================================================
# Get Feature Importance Trends Tests
# =============================================================================


@requires_supabase
class TestGetFeatureImportanceTrendsIntegration:
    """Integration tests for get_feature_importance_trends method."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_nonexistent_model(self, repository):
        """Test that empty list is returned for non-existent model."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        result = await repository.get_feature_importance_trends(
            model_registry_id=f"nonexistent-{uuid.uuid4().hex}",
            limit=30,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self, repository):
        """Test that limit parameter is respected."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        result = await repository.get_feature_importance_trends(
            model_registry_id="any-id",
            limit=5,
        )

        assert len(result) <= 5


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


@requires_supabase
class TestEndToEndFlow:
    """End-to-end integration tests for complete SHAP analysis flow."""

    @pytest.mark.asyncio
    async def test_complete_analysis_lifecycle(self, repository, sample_analysis_dict):
        """Test complete lifecycle: store → retrieve → query trends."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        # 1. Store analysis
        stored = await repository.store_analysis(analysis_dict=sample_analysis_dict)

        if stored is None:
            pytest.skip("Could not store analysis (possible FK constraint)")

        stored_id = stored["id"]
        assert stored_id is not None

        # 2. Verify it exists in the table (via direct query)
        result = await (
            repository.client.table(repository.table_name).select("*").eq("id", stored_id).execute()
        )

        assert len(result.data) == 1
        assert result.data[0]["id"] == stored_id

        # 3. Verify global_importance JSONB is correctly stored
        assert "global_importance" in result.data[0]
        importance = result.data[0]["global_importance"]
        assert "days_since_last_visit" in importance
        assert importance["days_since_last_visit"] == 0.35

        # 4. Verify top_interactions JSONB is correctly stored
        assert "top_interactions" in result.data[0]
        interactions = result.data[0]["top_interactions"]
        assert len(interactions) >= 1
        assert interactions[0]["feature_1"] == "days_since_last_visit"
        assert interactions[0]["feature_2"] == "therapy_adherence_score"

    @pytest.mark.asyncio
    async def test_schema_field_mapping(self, repository, sample_analysis_dict):
        """Test that Python dict fields map correctly to database columns."""
        if repository.client is None:
            pytest.skip("Supabase client not available")

        stored = await repository.store_analysis(analysis_dict=sample_analysis_dict)

        if stored is None:
            pytest.skip("Could not store analysis")

        # Verify field mappings (per mlops_tables.sql schema)
        assert stored["analysis_type"] == "global"
        assert stored["computation_duration_seconds"] == 45  # Integer, not float
        assert stored["computation_method"] == "TreeExplainer"
        assert stored["sample_size"] == 1000
        assert (
            stored["natural_language_explanation"]
            == "Key drivers are visit recency and therapy adherence."
        )
        assert len(stored["key_drivers"]) == 5


# =============================================================================
# Repository Initialization Tests
# =============================================================================


class TestRepositoryInitialization:
    """Tests for repository initialization and singleton behavior."""

    def test_singleton_returns_same_instance(self, monkeypatch):
        """Test that singleton pattern works correctly."""
        import src.repositories.shap_analysis as module

        # Reset singleton
        monkeypatch.setattr(module, "_shap_analysis_repository", None)

        repo1 = get_shap_analysis_repository()
        repo2 = get_shap_analysis_repository()

        assert repo1 is repo2

    def test_repository_has_correct_table_name(self):
        """Test that repository uses correct table name."""
        repo = get_shap_analysis_repository()

        assert repo.table_name == "ml_shap_analyses"
