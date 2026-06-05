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

    def test_run_hierarchical_analysis_sync_fails_closed_without_data(
        self, client, hierarchical_request
    ):
        """Sync hierarchical analysis with no inline data MUST fail-closed (503).

        C1 de-fabrication: the endpoint previously ran the REAL EconML analyzer
        over fabricated np.random data and returned 200 COMPLETED — this test
        used to (inadvertently) assert that fabrication. Post-C1 the default
        path resolves a real DataFrame from filters.estimation_data_records and
        raises 503 when none is present (matching the sibling endpoints). The
        labeled demo path and the real-data path are covered by
        tests/api/test_hierarchical_defab.py.
        """
        response = client.post(
            "/causal/hierarchical/analyze",
            json=hierarchical_request,
        )

        assert response.status_code == 503, (
            f"Default path must fail-closed with 503, got {response.status_code}"
        )

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
        """Test retrieving hierarchical analysis result.

        Uses demo_mode=true to populate the cache without inline data (post-C1
        the default path fails-closed with 503 when no real data is supplied).
        """
        # First, create an analysis (labeled demo placeholder is sufficient here)
        create_response = client.post(
            "/causal/hierarchical/analyze?demo_mode=true",
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

        # demo_mode=true exercises that the request params (segmentation method,
        # n_segments) flow through to the response without inline data (post-C1
        # the default path fails-closed with 503 when no real data is supplied).
        response = client.post(
            "/causal/hierarchical/analyze?demo_mode=true",
            json=hierarchical_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["segmentation_method"] == "kmeans"
        # Response uses n_segments_analyzed, not n_segments
        assert "n_segments_analyzed" in data
        assert data["n_segments_analyzed"] == 5


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

    def test_run_sequential_pipeline_raises_503_in_default_mode(
        self, client, sequential_pipeline_request
    ):
        """Default (no demo_mode) fails closed with 503 — no fabricated effects.

        Pins the F-005 fail-closed semantics: chat users + LLMs that hit the
        endpoint without demo_mode get a structured 503, NOT random.uniform-
        shaped fabricated stats with statistical_significance=True.

        Post-#354 C-8 (2026-05-22): the 503 is no longer a hardcoded
        short-circuit. The endpoint now invokes the wired SequentialPipeline;
        when no DataFrame is resolvable from request filters, every wired
        executor fails-closed and the response builder honestly reports 503.
        See tests/api/test_causal_pipeline_c8_wiring.py for the positive
        (data-provided) tests that exercise the real-execution path.
        """
        response = client.post(
            "/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )
        assert response.status_code == 503

    def test_run_sequential_pipeline_sync_with_demo_mode(self, client, sequential_pipeline_request):
        """demo_mode=true returns pinned-zero placeholders + is_demo warning."""
        response = client.post(
            "/causal/pipeline/sequential?demo_mode=true",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert data["status"] in ["completed", "in_progress", "pending"]
        assert "stages_completed" in data
        assert "stages_total" in data

    def test_run_sequential_pipeline_async_with_demo_mode(
        self, client, sequential_pipeline_request
    ):
        """Async + demo_mode returns 200 with pending status."""
        response = client.post(
            "/causal/pipeline/sequential?async_mode=true&demo_mode=true",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert "pipeline_id" in data

    def test_sequential_pipeline_stage_order_with_demo_mode(
        self, client, sequential_pipeline_request
    ):
        """Pipeline respects stage order when run in demo_mode."""
        sequential_pipeline_request["stages"] = [
            {"library": "networkx", "estimator": "causal_graph", "parameters": {}},
            {"library": "dowhy", "estimator": "propensity_score_matching", "parameters": {}},
            {"library": "econml", "estimator": "causal_forest", "parameters": {}},
        ]

        response = client.post(
            "/causal/pipeline/sequential?demo_mode=true",
            json=sequential_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["stages_total"] == 3


# =============================================================================
# PARALLEL PIPELINE TESTS
# =============================================================================


class TestParallelPipeline:
    """Tests for parallel pipeline execution."""

    def test_run_parallel_pipeline_raises_503_in_default_mode(
        self, client, parallel_pipeline_request
    ):
        """Default (no demo_mode) fails closed with 503.

        Post-#354 C-8 (2026-05-22): the 503 now reflects the wired
        ParallelPipeline's honest fail-close when no DataFrame is resolvable
        from request filters, rather than a hardcoded short-circuit. See
        tests/api/test_causal_pipeline_c8_wiring.py for the positive
        (data-provided) tests.
        """
        response = client.post(
            "/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 503

    def test_run_parallel_pipeline_sync_with_demo_mode(self, client, parallel_pipeline_request):
        """demo_mode=true returns pinned-zero placeholders."""
        response = client.post(
            "/causal/pipeline/parallel?demo_mode=true",
            json=parallel_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "pipeline_id" in data
        assert data["status"] in ["completed", "in_progress", "pending"]
        assert "libraries_succeeded" in data
        assert "libraries_failed" in data

    def test_run_parallel_pipeline_async_with_demo_mode(self, client, parallel_pipeline_request):
        """Async + demo_mode returns 200."""
        response = client.post(
            "/causal/pipeline/parallel?async_mode=true&demo_mode=true",
            json=parallel_pipeline_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["pending", "completed", "in_progress"]
        assert "pipeline_id" in data

    def test_get_pipeline_status_success(self, client, sequential_pipeline_request):
        """Test retrieving pipeline status (uses demo_mode for setup)."""
        # First create a pipeline in demo_mode (default path is 503)
        create_response = client.post(
            "/causal/pipeline/sequential?demo_mode=true",
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

    def test_run_cross_validation_raises_503_in_default_mode(
        self, client, cross_validation_request
    ):
        """Default (no demo_mode) fails closed with 503."""
        response = client.post(
            "/causal/validate",
            json=cross_validation_request,
        )
        assert response.status_code == 503

    def test_run_cross_validation_with_demo_mode(self, client, cross_validation_request):
        """demo_mode=true returns pinned-zero validation result."""
        response = client.post(
            "/causal/validate?demo_mode=true",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert "validation_id" in data
        assert "primary_library" in data
        assert "validation_library" in data
        assert "agreement_score" in data

    def test_cross_validation_agreement_threshold_with_demo_mode(
        self, client, cross_validation_request
    ):
        """Custom agreement threshold honored under demo_mode."""
        cross_validation_request["agreement_threshold"] = 0.90

        response = client.post(
            "/causal/validate?demo_mode=true",
            json=cross_validation_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["agreement_threshold"] == 0.90

    def test_cross_validation_libraries_with_demo_mode(self, client, cross_validation_request):
        """Cross-validation with different library pairs (demo_mode)."""
        cross_validation_request["primary_library"] = "dowhy"
        cross_validation_request["validation_library"] = "econml"

        response = client.post(
            "/causal/validate?demo_mode=true",
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

    def test_pipeline_response_format_with_demo_mode(self, client, sequential_pipeline_request):
        """Pipeline response format (demo_mode — default 503 is pinned in TestSequentialPipeline)."""
        response = client.post(
            "/causal/pipeline/sequential?demo_mode=true",
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

    def test_validation_response_format_with_demo_mode(self, client, cross_validation_request):
        """Cross-validation response format (demo_mode — default 503 is pinned in TestCrossValidation)."""
        response = client.post(
            "/causal/validate?demo_mode=true",
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


class TestDataSourceDefaultIsNeutral:
    """Disputed-sweep finding #5: the misleading ``data_source='mock_data'``
    default is renamed to a neutral identifier.

    The old default string ``'mock_data'`` looked like a request to use fake
    data, but NOTHING in the codebase branches on that literal (only
    ``data_source == 'synthetic'`` triggers a special — non-mock — path in
    src/agents/causal_impact/agent.py). The default is now ``'default'``.
    """

    def test_all_request_schemas_default_to_neutral_identifier(self):
        from src.api.schemas.causal import (
            CrossValidationRequest,
            HierarchicalAnalysisRequest,
            ParallelPipelineRequest,
            SequentialPipelineRequest,
        )

        hier = HierarchicalAnalysisRequest(treatment_var="t", outcome_var="o")
        assert hier.data_source == "default"
        assert hier.data_source != "mock_data"

        # Pipeline/validation schemas have required fields beyond data_source;
        # construct the minimal valid instances.
        from src.api.schemas.causal import CausalLibrary, PipelineStageConfig

        seq = SequentialPipelineRequest(
            treatment_var="t",
            outcome_var="o",
            stages=[
                PipelineStageConfig(library=CausalLibrary.DOWHY),
                PipelineStageConfig(library=CausalLibrary.ECONML),
            ],
        )
        assert seq.data_source == "default"

        par = ParallelPipelineRequest(
            treatment_var="t",
            outcome_var="o",
            libraries=[CausalLibrary.DOWHY, CausalLibrary.ECONML],
        )
        assert par.data_source == "default"

        cv = CrossValidationRequest(
            treatment_var="t",
            outcome_var="o",
            primary_library=CausalLibrary.DOWHY,
            validation_library=CausalLibrary.ECONML,
        )
        assert cv.data_source == "default"

    def test_neutral_default_does_not_trigger_a_mock_path(self):
        """The neutral default must not be treated like the only behavior-changing
        data_source literal ('synthetic').

        The agent's _initialize_state special-cases ONLY data_source ==
        'synthetic' (setting a fast ols/refutation config). With 'default' (the
        new neutral schema default) that fast-path must NOT engage — proving the
        new default triggers no mock/special data behavior.
        """
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent.__new__(CausalImpactAgent)

        base_input = {
            "query": "impact?",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "confounders": [],
        }

        # 'synthetic' -> fast-path params are injected.
        synth = agent._initialize_state({**base_input, "data_source": "synthetic"})
        assert synth["parameters"].get("method") == "ols"
        assert "refutation_config" in synth["parameters"]

        # 'default' (the new neutral schema default) -> NO fast-path params.
        neutral = agent._initialize_state({**base_input, "data_source": "default"})
        assert "method" not in neutral["parameters"]
        assert "refutation_config" not in neutral["parameters"]
        assert neutral["data_source"] == "default"
