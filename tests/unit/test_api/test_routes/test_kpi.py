"""Unit tests for KPI API routes.

Tests cover:
- Listing KPIs and workstreams
- Single KPI calculation
- Batch KPI calculation
- Cache invalidation
- Health check
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.kpi import get_kpi_calculator, router
from src.kpi.models import (
    CalculationType,
    CausalLibrary,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    KPIThreshold,
    Workstream,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_kpi_metadata():
    """Create a mock KPI metadata."""
    return KPIMetadata(
        id="WS1-DQ-001",
        name="Data Completeness Rate",
        definition="Percentage of records with all required fields populated",
        formula="complete_records / total_records * 100",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_DATA_QUALITY,
        tables=["hcp_profiles", "patient_journeys"],
        columns=["data_completeness_score"],
        threshold=KPIThreshold(target=95.0, warning=85.0, critical=75.0),
        unit="%",
        primary_causal_library=CausalLibrary.NONE,
    )


@pytest.fixture
def mock_kpi_result():
    """Create a mock KPI result."""
    return KPIResult(
        kpi_id="WS1-DQ-001",
        value=92.5,
        status=KPIStatus.WARNING,
        calculated_at=datetime.now(timezone.utc),
        cached=False,
    )


@pytest.fixture
def mock_calculator(mock_kpi_metadata, mock_kpi_result):
    """Create a mock KPI calculator."""
    calculator = MagicMock()

    # Mock list_kpis to return list of KPI metadata
    calculator.list_kpis.return_value = [mock_kpi_metadata]

    # Mock get_kpi_metadata to return a single KPI
    calculator.get_kpi_metadata.return_value = mock_kpi_metadata

    # Mock calculate (sync method)
    calculator.calculate.return_value = mock_kpi_result

    # Mock calculate_batch (sync method)
    mock_batch = MagicMock()
    mock_batch.results = [mock_kpi_result]
    mock_batch.total_kpis = 1
    mock_batch.successful = 1
    mock_batch.failed = 0
    mock_batch.workstream = None
    mock_batch.calculated_at = datetime.now(timezone.utc)
    calculator.calculate_batch.return_value = mock_batch

    # Mock invalidate_cache (sync method)
    calculator.invalidate_cache.return_value = 1

    # Mock cache
    calculator._cache = MagicMock()
    calculator._cache.enabled = True
    calculator._cache.size.return_value = 10

    # Mock database connection
    calculator._db = MagicMock()

    return calculator


@pytest.fixture
def app(mock_calculator):
    """Create a FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Override dependency
    app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


# =============================================================================
# LIST KPIs TESTS
# =============================================================================


class TestListKPIs:
    """Tests for GET /api/kpis endpoint."""

    def test_list_kpis_success(self, client, mock_calculator):
        """Test successful KPI listing."""
        response = client.get("/api/kpis")

        assert response.status_code == 200
        data = response.json()
        assert "kpis" in data
        assert "total" in data
        assert len(data["kpis"]) == 1
        assert data["kpis"][0]["id"] == "WS1-DQ-001"

    def test_list_kpis_by_workstream(self, client, mock_calculator):
        """Test KPI listing filtered by workstream."""
        response = client.get("/api/kpis?workstream=ws1_data_quality")

        assert response.status_code == 200
        data = response.json()
        assert data["workstream"] == "ws1_data_quality"
        # Verify list_kpis was called with workstream parameter
        mock_calculator.list_kpis.assert_called()

    def test_list_kpis_by_causal_library(self, client, mock_calculator):
        """Test KPI listing filtered by causal library."""
        response = client.get("/api/kpis?causal_library=dowhy")

        assert response.status_code == 200
        data = response.json()
        assert data["causal_library"] == "dowhy"


# =============================================================================
# LIST WORKSTREAMS TESTS
# =============================================================================


class TestListWorkstreams:
    """Tests for GET /api/kpis/workstreams endpoint."""

    def test_list_workstreams_success(self, client, mock_calculator, mock_kpi_metadata):
        """Test successful workstream listing.

        Route iterates through ALL Workstream enum values (6 total)
        and calls calculator.list_kpis(workstream=ws) for each.
        """
        # Mock returns our test KPI for any workstream query
        mock_calculator.list_kpis.return_value = [mock_kpi_metadata]

        response = client.get("/api/kpis/workstreams")

        assert response.status_code == 200
        data = response.json()
        assert "workstreams" in data
        assert "total" in data
        # Route returns ALL 6 Workstream enum values
        assert data["total"] == 6
        assert len(data["workstreams"]) == 6
        # Each workstream should have a kpi_count
        for ws in data["workstreams"]:
            assert "id" in ws
            assert "name" in ws
            assert "kpi_count" in ws


# =============================================================================
# GET KPI METADATA TESTS
# =============================================================================


class TestGetKPIMetadata:
    """Tests for GET /api/kpis/{kpi_id}/metadata endpoint."""

    def test_get_metadata_success(self, client, mock_calculator):
        """Test successful metadata retrieval."""
        response = client.get("/api/kpis/WS1-DQ-001/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "WS1-DQ-001"
        assert data["name"] == "Data Completeness Rate"
        assert "threshold" in data

    def test_get_metadata_not_found(self, client, mock_calculator):
        """Test metadata retrieval for non-existent KPI."""
        # Route calls calculator.get_kpi_metadata() which returns None for unknown KPIs
        mock_calculator.get_kpi_metadata.return_value = None

        response = client.get("/api/kpis/INVALID-KPI/metadata")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# GET KPI VALUE TESTS
# =============================================================================


class TestGetKPIValue:
    """Tests for GET /api/kpis/{kpi_id} endpoint."""

    def test_get_value_success(self, client, mock_calculator):
        """Test successful KPI value retrieval."""
        response = client.get("/api/kpis/WS1-DQ-001")

        assert response.status_code == 200
        data = response.json()
        assert data["kpi_id"] == "WS1-DQ-001"
        assert data["value"] == 92.5
        assert data["status"] == "warning"

    def test_get_value_with_cache_disabled(self, client, mock_calculator):
        """Test KPI value retrieval with cache disabled."""
        response = client.get("/api/kpis/WS1-DQ-001?use_cache=false")

        assert response.status_code == 200
        mock_calculator.calculate.assert_called()

    def test_get_value_not_found(self, client, mock_calculator):
        """Test value retrieval for non-existent KPI."""
        # Route calls calculator.calculate() and checks result.error for "not found"
        mock_result = MagicMock()
        mock_result.error = "KPI not found: INVALID-KPI"
        mock_calculator.calculate.return_value = mock_result

        response = client.get("/api/kpis/INVALID-KPI")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# CALCULATE KPI TESTS
# =============================================================================


class TestCalculateKPI:
    """Tests for POST /api/kpis/calculate endpoint."""

    def test_calculate_success(self, client, mock_calculator):
        """Test successful KPI calculation."""
        response = client.post(
            "/api/kpis/calculate",
            json={
                "kpi_id": "WS1-DQ-001",
                "use_cache": True,
                "force_refresh": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["kpi_id"] == "WS1-DQ-001"
        assert data["value"] == 92.5

    def test_calculate_with_context(self, client, mock_calculator):
        """Test KPI calculation with context."""
        response = client.post(
            "/api/kpis/calculate",
            json={
                "kpi_id": "WS1-DQ-001",
                "context": {
                    "brand": "remibrutinib",
                    "territory": "US",
                },
            },
        )

        assert response.status_code == 200
        mock_calculator.calculate.assert_called()

    def test_calculate_not_found(self, client, mock_calculator):
        """Test calculation for non-existent KPI."""
        # Route calls calculator.calculate() and checks result.error for "not found"
        mock_result = MagicMock()
        mock_result.error = "KPI not found: INVALID-KPI"
        mock_calculator.calculate.return_value = mock_result

        response = client.post(
            "/api/kpis/calculate",
            json={"kpi_id": "INVALID-KPI"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# BATCH CALCULATE TESTS
# =============================================================================


class TestBatchCalculate:
    """Tests for POST /api/kpis/batch endpoint."""

    def test_batch_calculate_by_ids(self, client, mock_calculator):
        """Test batch calculation by KPI IDs."""
        response = client.post(
            "/api/kpis/batch",
            json={
                "kpi_ids": ["WS1-DQ-001", "WS1-DQ-002"],
                "use_cache": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_kpis" in data
        assert "successful" in data

    def test_batch_calculate_by_workstream(self, client, mock_calculator):
        """Test batch calculation by workstream."""
        response = client.post(
            "/api/kpis/batch",
            json={
                "workstream": "ws1_data_quality",
                "use_cache": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["workstream"] == "ws1_data_quality"

    def test_batch_calculate_no_specification(self, client, mock_calculator):
        """Test batch calculation with empty specification.

        Note: Route delegates to calculator.calculate_batch() which handles empty specs.
        The API returns whatever the calculator returns, including empty results.
        """
        # Mock empty batch result
        mock_batch = MagicMock()
        mock_batch.results = []
        mock_batch.total_kpis = 0
        mock_batch.successful = 0
        mock_batch.failed = 0
        mock_batch.calculated_at = datetime.now(timezone.utc)
        mock_calculator.calculate_batch.return_value = mock_batch

        response = client.post(
            "/api/kpis/batch",
            json={"use_cache": True},
        )

        # Route returns 200 with empty results when no KPIs specified
        assert response.status_code == 200
        data = response.json()
        assert data["total_kpis"] == 0
        assert data["results"] == []


# =============================================================================
# CACHE INVALIDATION TESTS
# =============================================================================


class TestCacheInvalidation:
    """Tests for POST /api/kpis/invalidate endpoint."""

    def test_invalidate_single_kpi(self, client, mock_calculator):
        """Test cache invalidation for single KPI."""
        response = client.post(
            "/api/kpis/invalidate",
            json={"kpi_id": "WS1-DQ-001"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "invalidated_count" in data

    def test_invalidate_workstream(self, client, mock_calculator):
        """Test cache invalidation for workstream."""
        response = client.post(
            "/api/kpis/invalidate",
            json={"workstream": "ws1_data_quality"},
        )

        assert response.status_code == 200

    def test_invalidate_all(self, client, mock_calculator):
        """Test invalidation of all cache entries."""
        response = client.post(
            "/api/kpis/invalidate",
            json={"invalidate_all": True},
        )

        assert response.status_code == 200

    def test_invalidate_no_specification(self, client, mock_calculator):
        """Test invalidation with no specification returns success with message.

        Note: Route returns 200 with message 'No invalidation criteria specified'
        rather than returning a 400 error.
        """
        response = client.post(
            "/api/kpis/invalidate",
            json={},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["invalidated_count"] == 0
        assert "no invalidation criteria" in data["message"].lower()


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestHealthCheck:
    """Tests for GET /api/kpis/health endpoint."""

    def test_health_check_healthy(self, client, mock_calculator, mock_kpi_metadata):
        """Test healthy system status."""
        # Route calls calculator.list_kpis() and checks len > 0 for healthy status
        # Also checks calculator._db is not None
        mock_calculator.list_kpis.return_value = [mock_kpi_metadata] * 10
        mock_calculator._db = MagicMock()  # Non-None means connected

        response = client.get("/api/kpis/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["registry_loaded"] is True
        assert data["total_kpis"] == 10
        assert "cache_size" in data

    def test_health_check_degraded(self, client, mock_calculator):
        """Test degraded system status when no KPIs loaded."""
        # Route returns "degraded" when len(all_kpis) == 0
        mock_calculator.list_kpis.return_value = []
        mock_calculator._db = MagicMock()

        response = client.get("/api/kpis/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["registry_loaded"] is False
        assert data["total_kpis"] == 0

    def test_health_check_error(self, client, mock_calculator):
        """Test unhealthy system status when exception occurs."""
        # Route returns "unhealthy" when an exception is raised
        mock_calculator.list_kpis.side_effect = Exception("Registry error")

        response = client.get("/api/kpis/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["error"] is not None
        assert "registry error" in data["error"].lower()
