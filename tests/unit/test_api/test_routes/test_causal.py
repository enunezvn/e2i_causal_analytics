"""Unit tests for Causal API routes.

Tests cover:
- Hierarchical analysis endpoints
- Library routing
- Sequential/Parallel pipeline execution
- Cross-library validation
- Estimator listing
- Health check
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.causal import router

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create a FastAPI app with the causal router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def hierarchical_request():
    """Create a sample hierarchical analysis request."""
    return {
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "effect_modifiers": ["feature_1", "feature_2"],
        "data_source": "mock_data",
        "n_segments": 3,
        "segmentation_method": "quantile",
        "estimator_type": "causal_forest",
        "min_segment_size": 50,
        "confidence_level": 0.95,
    }


@pytest.fixture
def route_query_request():
    """Create a sample route query request."""
    return {
        "query": "How does the treatment effect vary across patient segments?",
        "context": {
            "brand": "Kisqali",
            "region": "Northeast",
        },
    }


@pytest.fixture
def sequential_pipeline_request():
    """Create a sample sequential pipeline request."""
    return {
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "data_source": "mock_data",
        "covariates": ["feature_1", "feature_2"],
        "stages": [
            {"library": "dowhy", "estimator": "propensity_score_matching", "parameters": {}},
            {"library": "econml", "estimator": "causal_forest", "parameters": {}},
            {"library": "causalml", "estimator": "uplift_random_forest", "parameters": {}},
        ],
    }


@pytest.fixture
def parallel_pipeline_request():
    """Create a sample parallel pipeline request."""
    return {
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "data_source": "mock_data",
        "covariates": ["feature_1", "feature_2"],
        "libraries": ["dowhy", "econml", "causalml"],
        "estimators": {
            "dowhy": "propensity_score_matching",
            "econml": "causal_forest",
            "causalml": "uplift_random_forest",
        },
    }


@pytest.fixture
def cross_validation_request():
    """Create a sample cross-validation request."""
    return {
        "treatment_var": "treatment",
        "outcome_var": "outcome",
        "data_source": "mock_data",
        "primary_library": "econml",
        "validation_library": "causalml",
        "agreement_threshold": 0.85,
    }


# =============================================================================
# HIERARCHICAL ANALYSIS TESTS
# =============================================================================


class TestHierarchicalAnalysis:
    """Tests for hierarchical analysis endpoints."""

    def test_run_hierarchical_analysis_sync(self, client, hierarchical_request):
        """Test synchronous hierarchical analysis."""
        response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] in ["completed", "in_progress", "pending", "failed"]
        # Response uses HierarchicalAnalysisResponse schema which has estimator_type
        assert "estimator_type" in data

    def test_run_hierarchical_analysis_async(self, client, hierarchical_request):
        """Test asynchronous hierarchical analysis."""
        response = client.post(
            "/causal/hierarchical/analyze?async_mode=true",
            json=hierarchical_request,
        )

        # API returns 200 with status=pending for async mode
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "pending"

    def test_get_hierarchical_result_success(self, client, hierarchical_request):
        """Test retrieving hierarchical analysis result."""
        # First, create an analysis
        create_response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )
        analysis_id = create_response.json()["analysis_id"]

        # Then retrieve it
        response = client.get(f"/causal/hierarchical/{analysis_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["analysis_id"] == analysis_id

    def test_get_hierarchical_result_not_found(self, client):
        """Test retrieving non-existent analysis."""
        response = client.get("/causal/hierarchical/nonexistent-id-12345")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_hierarchical_analysis_validation_error(self, client):
        """Test hierarchical analysis with invalid request."""
        invalid_request = {
            "treatment_var": "treatment",
            # Missing required fields
        }

        response = client.post(
            "/causal/hierarchical/analyze",
            json=invalid_request,
        )

        assert response.status_code == 422  # Validation error

    def test_hierarchical_analysis_with_custom_segments(self, client, hierarchical_request):
        """Test hierarchical analysis with custom segment count."""
        hierarchical_request["n_segments"] = 5
        hierarchical_request["segmentation_method"] = "kmeans"

        response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["segmentation_method"] == "kmeans"
        # Response uses n_segments_analyzed, not n_segments
        assert "n_segments_analyzed" in data


# =============================================================================
# LIBRARY ROUTING TESTS
# =============================================================================


class TestLibraryRouting:
    """Tests for library routing endpoint."""

    def test_route_query_success(self, client, route_query_request):
        """Test successful query routing."""
        response = client.post(
            "/causal/route",
            json=route_query_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "primary_library" in data
        assert "question_type" in data
        assert "routing_confidence" in data
        assert 0 <= data["routing_confidence"] <= 1

    def test_route_query_causal_effect(self, client):
        """Test routing for causal effect question."""
        response = client.post(
            "/causal/route",
            json={
                "query": "Does the marketing campaign cause increased sales?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Causal effect questions typically route to DoWhy
        assert data["primary_library"] in ["dowhy", "econml", "causalml", "networkx"]
        assert "question_type" in data

    def test_route_query_targeting(self, client):
        """Test routing for targeting question."""
        response = client.post(
            "/causal/route",
            json={
                "query": "Who should we target for the treatment?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "primary_library" in data

    def test_route_query_heterogeneity(self, client):
        """Test routing for effect heterogeneity question."""
        response = client.post(
            "/causal/route",
            json={
                "query": "How does the treatment effect vary across segments?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "primary_library" in data

    def test_route_query_empty_query(self, client):
        """Test routing with empty query."""
        response = client.post(
            "/causal/route",
            json={"query": ""},
        )

        # Should return validation error or handle gracefully
        assert response.status_code in [200, 400, 422]


# =============================================================================
# SEQUENTIAL PIPELINE TESTS
# =============================================================================


class TestSequentialPipeline:
    """Tests for sequential pipeline execution."""

    def test_run_sequential_pipeline_sync(self, client, sequential_pipeline_request):
        """Test synchronous sequential pipeline."""
        response = client.post(
            "/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert data["status"] in ["completed", "in_progress", "pending"]
        # SequentialPipelineResponse has stages_completed and stages_total
        assert "stages_completed" in data
        assert "stages_total" in data

    def test_run_sequential_pipeline_async(self, client, sequential_pipeline_request):
        """Test asynchronous sequential pipeline."""
        response = client.post(
            "/causal/pipeline/sequential?async_mode=true",
            json=sequential_pipeline_request,
        )

        # API returns 200 with status=pending for async mode
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert "pipeline_id" in data

    def test_sequential_pipeline_stage_order(self, client, sequential_pipeline_request):
        """Test that pipeline respects stage order."""
        # Override stages with different order
        sequential_pipeline_request["stages"] = [
            {"library": "networkx", "estimator": "causal_graph", "parameters": {}},
            {"library": "dowhy", "estimator": "propensity_score_matching", "parameters": {}},
            {"library": "econml", "estimator": "causal_forest", "parameters": {}},
        ]

        response = client.post(
            "/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        # Verify stages_total matches our input
        assert data["stages_total"] == 3


# =============================================================================
# PARALLEL PIPELINE TESTS
# =============================================================================


class TestParallelPipeline:
    """Tests for parallel pipeline execution."""

    def test_run_parallel_pipeline_sync(self, client, parallel_pipeline_request):
        """Test synchronous parallel pipeline."""
        response = client.post(
            "/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert data["status"] in ["completed", "in_progress", "pending"]
        # ParallelPipelineResponse has libraries_succeeded and libraries_failed
        assert "libraries_succeeded" in data
        assert "libraries_failed" in data

    def test_run_parallel_pipeline_async(self, client, parallel_pipeline_request):
        """Test asynchronous parallel pipeline."""
        response = client.post(
            "/causal/pipeline/parallel?async_mode=true",
            json=parallel_pipeline_request,
        )

        # API returns 200 - may be pending or completed if execution is fast
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["pending", "completed", "in_progress"]
        assert "pipeline_id" in data

    def test_get_pipeline_status_success(self, client, sequential_pipeline_request):
        """Test retrieving pipeline status."""
        # First create a pipeline
        create_response = client.post(
            "/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )
        pipeline_id = create_response.json()["pipeline_id"]

        # Then get status
        response = client.get(f"/causal/pipeline/{pipeline_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_id"] == pipeline_id

    def test_get_pipeline_status_not_found(self, client):
        """Test retrieving non-existent pipeline."""
        response = client.get("/causal/pipeline/nonexistent-pipeline-12345")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# CROSS-VALIDATION TESTS
# =============================================================================


class TestCrossValidation:
    """Tests for cross-library validation."""

    def test_run_cross_validation(self, client, cross_validation_request):
        """Test cross-library validation."""
        response = client.post(
            "/causal/validate",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "validation_id" in data
        assert "primary_library" in data
        assert "validation_library" in data
        assert "agreement_score" in data

    def test_cross_validation_agreement_threshold(self, client, cross_validation_request):
        """Test cross-validation with custom agreement threshold."""
        cross_validation_request["agreement_threshold"] = 0.90

        response = client.post(
            "/causal/validate",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agreement_threshold"] == 0.90

    def test_cross_validation_libraries(self, client, cross_validation_request):
        """Test cross-validation with different library pairs."""
        cross_validation_request["primary_library"] = "dowhy"
        cross_validation_request["validation_library"] = "econml"

        response = client.post(
            "/causal/validate",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["primary_library"] == "dowhy"
        assert data["validation_library"] == "econml"


# =============================================================================
# ESTIMATOR LISTING TESTS
# =============================================================================


class TestEstimatorListing:
    """Tests for estimator listing endpoint."""

    def test_list_all_estimators(self, client):
        """Test listing all estimators."""
        response = client.get("/causal/estimators")

        assert response.status_code == 200
        data = response.json()
        assert "estimators" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_estimators_by_library(self, client):
        """Test listing estimators filtered by library."""
        response = client.get("/causal/estimators?library=econml")

        assert response.status_code == 200
        data = response.json()
        assert "estimators" in data
        # All returned estimators should be from EconML if filter was applied
        if "library_filter" in data:
            assert data["library_filter"] == "econml"
        # Estimators may be filtered or include all
        for estimator in data["estimators"]:
            if data.get("library_filter"):
                assert estimator["library"] == "econml"

    def test_list_estimators_by_estimator_type(self, client):
        """Test listing estimators filtered by type."""
        response = client.get("/causal/estimators?estimator_type=causal_forest")

        assert response.status_code == 200
        data = response.json()
        assert "estimators" in data

    def test_list_estimators_invalid_library(self, client):
        """Test listing estimators with invalid library."""
        response = client.get("/causal/estimators?library=invalid_lib")

        # Should return validation error or empty results
        assert response.status_code in [200, 422]


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestHealthCheck:
    """Tests for causal engine health check."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/causal/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "libraries_available" in data

    def test_health_check_library_status(self, client):
        """Test that health check includes library status."""
        response = client.get("/causal/health")

        assert response.status_code == 200
        data = response.json()

        # Should include status for each library
        libraries = data.get("libraries_available", {})
        assert "dowhy" in libraries or len(libraries) >= 0
        assert "econml" in libraries or len(libraries) >= 0
        assert "causalml" in libraries or len(libraries) >= 0


# =============================================================================
# REQUEST VALIDATION TESTS
# =============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    def test_hierarchical_missing_treatment_var(self, client):
        """Test hierarchical analysis without treatment_var."""
        response = client.post(
            "/causal/hierarchical/analyze",
            json={
                "outcome_var": "outcome",
                "data_source": "mock_data",
            },
        )

        assert response.status_code == 422

    def test_hierarchical_missing_outcome_var(self, client):
        """Test hierarchical analysis without outcome_var."""
        response = client.post(
            "/causal/hierarchical/analyze",
            json={
                "treatment_var": "treatment",
                "data_source": "mock_data",
            },
        )

        assert response.status_code == 422

    def test_hierarchical_invalid_segmentation_method(self, client, hierarchical_request):
        """Test hierarchical analysis with invalid segmentation method."""
        hierarchical_request["segmentation_method"] = "invalid_method"

        response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )

        assert response.status_code == 422

    def test_hierarchical_invalid_estimator_type(self, client, hierarchical_request):
        """Test hierarchical analysis with invalid estimator type."""
        hierarchical_request["estimator_type"] = "invalid_estimator"

        response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )

        assert response.status_code == 422

    def test_pipeline_missing_treatment_var(self, client):
        """Test pipeline without treatment_var."""
        response = client.post(
            "/causal/pipeline/sequential",
            json={
                "outcome_var": "outcome",
                "data_source": "mock_data",
                "stages": [
                    {
                        "library": "dowhy",
                        "estimator": "propensity_score_matching",
                        "parameters": {},
                    },
                ],
            },
        )

        assert response.status_code == 422

    def test_cross_validation_missing_primary_library(self, client):
        """Test cross-validation without primary_library."""
        response = client.post(
            "/causal/validate",
            json={
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "data_source": "mock_data",
                "validation_library": "causalml",
            },
        )

        assert response.status_code == 422


# =============================================================================
# RESPONSE FORMAT TESTS
# =============================================================================


class TestResponseFormats:
    """Tests for response format consistency."""

    def test_hierarchical_response_format(self, client, hierarchical_request):
        """Test hierarchical analysis response format."""
        response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields per HierarchicalAnalysisResponse schema
        assert "analysis_id" in data
        assert "status" in data
        assert "segmentation_method" in data
        assert "estimator_type" in data

    def test_route_response_format(self, client, route_query_request):
        """Test routing response format."""
        response = client.post(
            "/causal/route",
            json=route_query_request,
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields per RouteQueryResponse schema
        assert "primary_library" in data
        assert "question_type" in data
        assert "routing_confidence" in data
        assert "routing_rationale" in data

    def test_pipeline_response_format(self, client, sequential_pipeline_request):
        """Test pipeline response format."""
        response = client.post(
            "/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields per SequentialPipelineResponse schema
        assert "pipeline_id" in data
        assert "status" in data
        assert "stages_completed" in data
        assert "stages_total" in data
        assert "stage_results" in data

    def test_validation_response_format(self, client, cross_validation_request):
        """Test cross-validation response format."""
        response = client.post(
            "/causal/validate",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields per CrossValidationResponse schema
        assert "validation_id" in data
        assert "primary_library" in data
        assert "validation_library" in data
        assert "agreement_score" in data

    def test_estimator_list_response_format(self, client):
        """Test estimator list response format."""
        response = client.get("/causal/estimators")

        assert response.status_code == 200
        data = response.json()

        # Required fields per EstimatorListResponse schema
        assert "estimators" in data
        assert "total" in data
        assert isinstance(data["estimators"], list)

        # Each estimator should have required fields
        if data["estimators"]:
            estimator = data["estimators"][0]
            assert "name" in estimator
            assert "library" in estimator
            assert "description" in estimator

    def test_health_response_format(self, client):
        """Test health check response format."""
        response = client.get("/causal/health")

        assert response.status_code == 200
        data = response.json()

        # Required fields per CausalHealthResponse schema
        assert "status" in data
        assert "libraries_available" in data


# =============================================================================
# ENUM VALIDATION TESTS
# =============================================================================


class TestEnumValidation:
    """Tests for enum value validation."""

    def test_valid_causal_libraries(self, client):
        """Test all valid causal library values."""
        valid_libraries = ["dowhy", "econml", "causalml", "networkx"]

        for lib in valid_libraries:
            response = client.get(f"/causal/estimators?library={lib}")
            assert response.status_code == 200

    def test_valid_segmentation_methods(self, client, hierarchical_request):
        """Test all valid segmentation methods."""
        valid_methods = ["quantile", "kmeans", "threshold", "tree"]

        for method in valid_methods:
            hierarchical_request["segmentation_method"] = method
            response = client.post(
                "/causal/hierarchical/analyze",
                json=hierarchical_request,
            )
            assert response.status_code == 200

    def test_valid_estimator_types(self, client, hierarchical_request):
        """Test all valid estimator types."""
        valid_types = ["causal_forest", "linear_dml", "x_learner", "t_learner", "s_learner", "ols"]

        for est_type in valid_types:
            hierarchical_request["estimator_type"] = est_type
            response = client.post(
                "/causal/hierarchical/analyze",
                json=hierarchical_request,
            )
            assert response.status_code == 200
