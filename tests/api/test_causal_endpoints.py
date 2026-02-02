"""
Tests for Causal API endpoints.

Phase 1C of API Audit - Causal Inference API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 1C.1: Hierarchical Analysis (POST /hierarchical/analyze, GET /hierarchical/{id}, GET /estimators)
- Batch 1C.2: Routing & Health (POST /route, GET /health, POST /validate)
- Batch 1C.3: Pipeline Execution (POST /pipeline/sequential, POST /pipeline/parallel, GET /pipeline/{id})
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def hierarchical_analysis_request():
    """Sample hierarchical analysis request."""
    return {
        "treatment_var": "promotion_received",
        "outcome_var": "trx_count",
        "effect_modifiers": ["region", "specialty"],
        "n_segments": 4,
        "segmentation_method": "quantile",
        "estimator_type": "causal_forest",
        "min_segment_size": 50,
        "confidence_level": 0.95,
        "aggregation_method": "variance_weighted",
        "timeout_seconds": 60,
    }


@pytest.fixture
def mock_segment_result():
    """Mock segment CATE result."""
    result = MagicMock()
    result.segment_id = 1  # Must be int per SegmentCATEResult schema
    result.segment_name = "High Responders"
    result.n_samples = 125
    result.uplift_range = (0.7, 1.0)
    result.cate_mean = 0.15
    result.cate_std = 0.03
    result.cate_ci_lower = 0.09
    result.cate_ci_upper = 0.21
    result.success = True
    result.error_message = None
    return result


@pytest.fixture
def mock_hierarchical_result(mock_segment_result):
    """Mock hierarchical analysis result."""
    result = MagicMock()
    result.segment_results = [mock_segment_result]
    result.overall_ate = 0.12
    result.overall_ate_ci_lower = 0.08
    result.overall_ate_ci_upper = 0.16
    result.segment_heterogeneity = 0.25
    result.n_segments = 4
    result.warnings = []
    result.errors = []
    return result


@pytest.fixture
def mock_nested_ci_result():
    """Mock nested CI calculation result."""
    result = MagicMock()
    result.aggregate_ate = 0.13
    result.aggregate_ci_lower = 0.09
    result.aggregate_ci_upper = 0.17
    result.aggregate_std = 0.02
    result.confidence_level = 0.95
    result.aggregation_method = "variance_weighted"
    result.segment_contributions = {"seg_001": 0.4, "seg_002": 0.6}
    result.i_squared = 0.35
    result.tau_squared = 0.005
    result.n_segments_included = 4
    result.total_sample_size = 500
    return result


@pytest.fixture
def route_query_request():
    """Sample route query request."""
    return {
        "query": "What is the causal effect of detailing on prescriptions?",
        "prefer_library": None,
    }


@pytest.fixture
def cross_validation_request():
    """Sample cross-validation request."""
    return {
        "primary_library": "econml",
        "validation_library": "dowhy",
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "confounders": ["region", "specialty"],
        "agreement_threshold": 0.7,
    }


@pytest.fixture
def sequential_pipeline_request():
    """Sample sequential pipeline request."""
    return {
        "stages": [
            {"library": "dowhy", "estimator": "propensity_score_matching"},
            {"library": "econml", "estimator": "causal_forest"},
        ],
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "stop_on_failure": True,
    }


@pytest.fixture
def parallel_pipeline_request():
    """Sample parallel pipeline request."""
    return {
        "libraries": ["econml", "dowhy"],
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "consensus_method": "variance_weighted",
        "timeout_seconds": 30,
    }


# =============================================================================
# BATCH 1C.1 - HIERARCHICAL ANALYSIS TESTS
# =============================================================================


class TestRunHierarchicalAnalysis:
    """Tests for POST /causal/hierarchical/analyze."""

    def test_hierarchical_analysis_sync_success(
        self, hierarchical_analysis_request, mock_hierarchical_result, mock_nested_ci_result
    ):
        """Should run hierarchical analysis synchronously."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze = AsyncMock(return_value=mock_hierarchical_result)

        mock_nested_ci = MagicMock()
        mock_nested_ci.compute = MagicMock(return_value=mock_nested_ci_result)

        # Patch multiple imports in the causal engine
        with (
            patch(
                "src.causal_engine.hierarchical.HierarchicalAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "src.causal_engine.hierarchical.HierarchicalConfig",
            ),
            patch(
                "src.causal_engine.hierarchical.NestedCIConfig",
            ),
            patch(
                "src.causal_engine.hierarchical.NestedConfidenceInterval",
                return_value=mock_nested_ci,
            ),
            patch(
                "src.causal_engine.hierarchical.analyzer.SegmentationMethod",
            ),
            patch(
                "src.causal_engine.hierarchical.nested_ci.SegmentEstimate",
            ),
        ):
            response = client.post(
                "/api/causal/hierarchical/analyze",
                json=hierarchical_analysis_request,
            )

        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "completed"
        assert "segment_results" in data
        assert data["segmentation_method"] == "quantile"
        assert data["estimator_type"] == "causal_forest"

    def test_hierarchical_analysis_async_mode(self, hierarchical_analysis_request):
        """Should return pending status in async mode."""
        response = client.post(
            "/api/causal/hierarchical/analyze",
            params={"async_mode": "true"},
            json=hierarchical_analysis_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert "analysis_id" in data

    def test_hierarchical_analysis_invalid_request(self):
        """Should return 422 for invalid request."""
        response = client.post(
            "/api/causal/hierarchical/analyze",
            json={"treatment_var": "x"},  # Missing required fields
        )

        assert response.status_code == 422


class TestGetHierarchicalAnalysis:
    """Tests for GET /causal/hierarchical/{analysis_id}."""

    def test_get_analysis_success(self, hierarchical_analysis_request):
        """Should return analysis result from cache."""
        # First create an analysis to populate the cache
        response = client.post(
            "/api/causal/hierarchical/analyze",
            params={"async_mode": "true"},
            json=hierarchical_analysis_request,
        )
        analysis_id = response.json()["analysis_id"]

        # Now fetch it
        response = client.get(f"/api/causal/hierarchical/{analysis_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["analysis_id"] == analysis_id

    def test_get_analysis_not_found(self):
        """Should return 404 for missing analysis."""
        response = client.get("/api/causal/hierarchical/nonexistent-id-12345")

        assert response.status_code == 404


class TestListEstimators:
    """Tests for GET /causal/estimators."""

    def test_list_all_estimators(self):
        """Should list all available estimators."""
        response = client.get("/api/causal/estimators")

        assert response.status_code == 200
        data = response.json()
        assert "estimators" in data
        assert "total" in data
        assert "by_library" in data
        assert data["total"] > 0

    def test_list_estimators_filtered_by_library(self):
        """Should filter estimators by library."""
        response = client.get("/api/causal/estimators", params={"library": "econml"})

        assert response.status_code == 200
        data = response.json()
        for estimator in data["estimators"]:
            assert estimator["library"] == "econml"

    def test_list_estimators_dowhy(self):
        """Should list DoWhy estimators."""
        response = client.get("/api/causal/estimators", params={"library": "dowhy"})

        assert response.status_code == 200
        data = response.json()
        assert all(e["library"] == "dowhy" for e in data["estimators"])
        # Check for known DoWhy estimators
        estimator_names = [e["name"] for e in data["estimators"]]
        assert "propensity_score_matching" in estimator_names


# =============================================================================
# BATCH 1C.2 - ROUTING & HEALTH TESTS
# =============================================================================


class TestRouteCausalQuery:
    """Tests for POST /causal/route."""

    def test_route_causal_effect_question(self):
        """Should route causal effect question to DoWhy."""
        response = client.post(
            "/api/causal/route",
            json={"query": "Does detailing cause an increase in prescriptions?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question_type"] == "causal_effect"
        assert data["primary_library"] == "dowhy"
        assert "routing_confidence" in data
        assert "routing_rationale" in data

    def test_route_heterogeneity_question(self):
        """Should route heterogeneity question to EconML."""
        # Note: Query must avoid causal_effect keywords (cause, causes, effect of, impact of, does)
        # Use heterogeneity keywords: vary, heterogen, different, segment, subgroup
        response = client.post(
            "/api/causal/route",
            json={"query": "Show variation in treatment response by segment and subgroup"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question_type"] == "effect_heterogeneity"
        assert data["primary_library"] == "econml"

    def test_route_targeting_question(self):
        """Should route targeting question to CausalML."""
        response = client.post(
            "/api/causal/route",
            json={"query": "Which HCPs should we target for the campaign?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question_type"] == "targeting"
        assert data["primary_library"] == "causalml"

    def test_route_network_question(self):
        """Should route network question to NetworkX."""
        # Note: Query must use network keywords without triggering causal_effect keywords first
        response = client.post(
            "/api/causal/route",
            json={"query": "How do changes propagate through the network dependencies?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question_type"] == "system_dependencies"
        assert data["primary_library"] == "networkx"

    def test_route_with_library_preference(self):
        """Should respect library preference in routing."""
        response = client.post(
            "/api/causal/route",
            json={
                "query": "What is the effect?",
                "prefer_library": "causalml",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["primary_library"] == "causalml"
        assert data["routing_confidence"] == 0.9


class TestCausalHealthCheck:
    """Tests for GET /causal/health."""

    def test_health_check_success(self):
        """Should return health status with library availability."""
        response = client.get("/api/causal/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "libraries_available" in data
        assert "dowhy" in data["libraries_available"]
        assert "econml" in data["libraries_available"]
        assert "estimators_loaded" in data

    def test_health_check_returns_library_status(self):
        """Should indicate which libraries are available."""
        response = client.get("/api/causal/health")

        assert response.status_code == 200
        data = response.json()
        # At least one library should be available
        libs = data["libraries_available"]
        assert isinstance(libs, dict)
        assert all(isinstance(v, bool) for v in libs.values())


class TestCrossValidation:
    """Tests for POST /causal/validate."""

    def test_cross_validation_success(self, cross_validation_request):
        """Should run cross-library validation."""
        response = client.post(
            "/api/causal/validate",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "validation_id" in data
        assert data["primary_library"] == "econml"
        assert data["validation_library"] == "dowhy"
        assert "primary_effect" in data
        assert "validation_effect" in data
        assert "agreement_score" in data
        assert "validation_passed" in data
        assert "recommendations" in data

    def test_cross_validation_agreement_threshold(self, cross_validation_request):
        """Should check against agreement threshold."""
        cross_validation_request["agreement_threshold"] = 0.9
        response = client.post(
            "/api/causal/validate",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agreement_threshold"] == 0.9


# =============================================================================
# BATCH 1C.3 - PIPELINE EXECUTION TESTS
# =============================================================================


class TestSequentialPipeline:
    """Tests for POST /causal/pipeline/sequential."""

    def test_sequential_pipeline_sync_success(self, sequential_pipeline_request):
        """Should run sequential pipeline synchronously."""
        response = client.post(
            "/api/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert data["status"] in ["completed", "pending"]
        assert data["stages_total"] == 2
        assert "stage_results" in data
        assert "consensus_effect" in data

    def test_sequential_pipeline_async_mode(self, sequential_pipeline_request):
        """Should return pending status in async mode."""
        response = client.post(
            "/api/causal/pipeline/sequential",
            params={"async_mode": "true"},
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert "pipeline_id" in data

    def test_sequential_pipeline_computes_consensus(self, sequential_pipeline_request):
        """Should compute consensus effect from stages."""
        response = client.post(
            "/api/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        if data["status"] == "completed":
            assert data["consensus_effect"] is not None
            assert "library_agreement_score" in data


class TestParallelPipeline:
    """Tests for POST /causal/pipeline/parallel."""

    def test_parallel_pipeline_success(self, parallel_pipeline_request):
        """Should run parallel pipeline across libraries."""
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert "libraries_succeeded" in data
        assert "libraries_failed" in data
        assert "library_results" in data
        assert "consensus_effect" in data

    def test_parallel_pipeline_consensus(self, parallel_pipeline_request):
        """Should compute consensus from parallel results."""
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        if data["libraries_succeeded"]:
            assert data["consensus_effect"] is not None
            assert data["consensus_method"] == "variance_weighted"

    def test_parallel_pipeline_with_timeout(self, parallel_pipeline_request):
        """Should respect timeout configuration."""
        # Note: timeout_seconds must be >= 30 per schema constraint
        parallel_pipeline_request["timeout_seconds"] = 30
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )

        # Should complete within timeout
        assert response.status_code == 200


class TestGetPipelineStatus:
    """Tests for GET /causal/pipeline/{pipeline_id}."""

    def test_get_pipeline_status_success(self, sequential_pipeline_request):
        """Should return pipeline status from cache."""
        # First create a pipeline to populate the cache
        response = client.post(
            "/api/causal/pipeline/sequential",
            params={"async_mode": "true"},
            json=sequential_pipeline_request,
        )
        pipeline_id = response.json()["pipeline_id"]

        # Now fetch status
        response = client.get(f"/api/causal/pipeline/{pipeline_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_id"] == pipeline_id

    def test_get_pipeline_status_not_found(self):
        """Should return 404 for missing pipeline."""
        response = client.get("/api/causal/pipeline/nonexistent-pipeline-12345")

        assert response.status_code == 404
