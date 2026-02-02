"""
Tests for KPI API endpoints.

Phase 2B of API Audit - KPI Calculation API
Tests organized by batch as per api-endpoints-audit-plan.md

Endpoints covered:
- Batch 2B.1: Retrieval (GET /api/kpis, GET /api/kpis/{id}, GET /api/kpis/{id}/metadata, GET /api/kpis/workstreams)
- Batch 2B.2: Calculation (POST /api/kpis/calculate, POST /api/kpis/batch, POST /api/kpis/invalidate, GET /api/kpis/health)
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes.kpi import get_kpi_calculator

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_kpi_metadata():
    """Mock KPI metadata object."""
    kpi = MagicMock()
    kpi.id = "data_freshness_lag"
    kpi.name = "Data Freshness Lag"
    kpi.definition = "Time since last data update"
    kpi.formula = "NOW() - MAX(updated_at)"
    kpi.calculation_type = MagicMock(value="aggregation")
    kpi.workstream = MagicMock(value="ws1_data_quality")
    kpi.tables = ["business_metrics"]
    kpi.columns = ["updated_at"]
    kpi.view = None
    kpi.threshold = MagicMock(target=1.0, warning=4.0, critical=8.0)
    kpi.unit = "hours"
    kpi.frequency = "hourly"
    kpi.primary_causal_library = MagicMock(value="none")
    kpi.brand = None
    kpi.note = None
    return kpi


@pytest.fixture
def mock_kpi_result():
    """Mock KPI calculation result."""
    result = MagicMock()
    result.kpi_id = "data_freshness_lag"
    result.value = 2.5
    result.status = MagicMock(value="normal")
    result.calculated_at = datetime.now(timezone.utc)
    result.cached = False
    result.cache_expires_at = None
    result.error = None
    result.causal_library_used = None
    result.confidence_interval = None
    result.p_value = None
    result.effect_size = None
    result.metadata = {}
    return result


@pytest.fixture
def mock_batch_result(mock_kpi_result):
    """Mock batch calculation result."""
    batch = MagicMock()
    batch.results = [mock_kpi_result]
    batch.calculated_at = datetime.now(timezone.utc)
    batch.total_kpis = 1
    batch.successful = 1
    batch.failed = 0
    return batch


@pytest.fixture
def mock_calculator(mock_kpi_metadata, mock_kpi_result, mock_batch_result):
    """Mock KPICalculator instance."""
    calculator = MagicMock()
    calculator.list_kpis = MagicMock(return_value=[mock_kpi_metadata])
    calculator.get_kpi_metadata = MagicMock(return_value=mock_kpi_metadata)
    calculator.calculate = MagicMock(return_value=mock_kpi_result)
    calculator.calculate_batch = MagicMock(return_value=mock_batch_result)
    calculator.invalidate_cache = MagicMock(return_value=5)
    calculator._db = MagicMock()
    calculator._cache = MagicMock(enabled=True, size=MagicMock(return_value=10))
    return calculator


@pytest.fixture(autouse=True)
def cleanup_overrides():
    """Clean up dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


# =============================================================================
# BATCH 2B.1 - RETRIEVAL TESTS
# =============================================================================


class TestListKPIs:
    """Tests for GET /api/kpis."""

    def test_list_kpis_success(self, mock_calculator):
        """Should list all KPIs."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis")

        assert response.status_code == 200
        data = response.json()
        assert "kpis" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["kpis"]) == data["total"]

    def test_list_kpis_filter_by_workstream(self, mock_calculator):
        """Should filter KPIs by workstream."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis", params={"workstream": "ws1_data_quality"})

        assert response.status_code == 200
        data = response.json()
        assert data["workstream"] == "ws1_data_quality"
        mock_calculator.list_kpis.assert_called()

    def test_list_kpis_filter_by_causal_library(self, mock_calculator):
        """Should filter KPIs by causal library."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis", params={"causal_library": "econml"})

        assert response.status_code == 200
        data = response.json()
        assert data["causal_library"] == "econml"


class TestGetKPIValue:
    """Tests for GET /api/kpis/{kpi_id}."""

    def test_get_kpi_value_success(self, mock_calculator):
        """Should return calculated KPI value."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag")

        assert response.status_code == 200
        data = response.json()
        assert data["kpi_id"] == "data_freshness_lag"
        assert "value" in data
        assert "status" in data
        assert "calculated_at" in data

    def test_get_kpi_value_with_cache(self, mock_calculator):
        """Should use cached value when requested."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag", params={"use_cache": "true"})

        assert response.status_code == 200
        mock_calculator.calculate.assert_called_with(
            kpi_id="data_freshness_lag",
            use_cache=True,
            force_refresh=False,
            context={},
        )

    def test_get_kpi_value_force_refresh(self, mock_calculator):
        """Should force recalculation when requested."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag", params={"force_refresh": "true"})

        assert response.status_code == 200
        mock_calculator.calculate.assert_called_with(
            kpi_id="data_freshness_lag",
            use_cache=True,
            force_refresh=True,
            context={},
        )

    def test_get_kpi_value_not_found(self, mock_calculator):
        """Should return 404 for missing KPI."""
        mock_result = MagicMock()
        mock_result.error = "KPI not found: invalid_kpi"
        mock_calculator.calculate.return_value = mock_result

        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/invalid_kpi")

        assert response.status_code == 404

    def test_get_kpi_value_with_brand_filter(self, mock_calculator):
        """Should filter by brand when provided."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag", params={"brand": "Remibrutinib"})

        assert response.status_code == 200
        mock_calculator.calculate.assert_called_with(
            kpi_id="data_freshness_lag",
            use_cache=True,
            force_refresh=False,
            context={"brand": "Remibrutinib"},
        )


class TestGetKPIMetadata:
    """Tests for GET /api/kpis/{kpi_id}/metadata."""

    def test_get_kpi_metadata_success(self, mock_calculator):
        """Should return KPI metadata."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "data_freshness_lag"
        assert data["name"] == "Data Freshness Lag"
        assert "definition" in data
        assert "formula" in data
        assert "threshold" in data
        assert "workstream" in data

    def test_get_kpi_metadata_not_found(self, mock_calculator):
        """Should return 404 for missing KPI."""
        mock_calculator.get_kpi_metadata.return_value = None

        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/invalid_kpi/metadata")

        assert response.status_code == 404


class TestListWorkstreams:
    """Tests for GET /api/kpis/workstreams."""

    def test_list_workstreams_success(self, mock_calculator):
        """Should list all workstreams."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/workstreams")

        assert response.status_code == 200
        data = response.json()
        assert "workstreams" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_workstreams_includes_kpi_counts(self, mock_calculator):
        """Should include KPI counts per workstream."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/workstreams")

        assert response.status_code == 200
        data = response.json()
        for ws in data["workstreams"]:
            assert "id" in ws
            assert "name" in ws
            assert "kpi_count" in ws


# =============================================================================
# BATCH 2B.2 - CALCULATION TESTS
# =============================================================================


class TestCalculateKPI:
    """Tests for POST /api/kpis/calculate."""

    def test_calculate_kpi_success(self, mock_calculator):
        """Should calculate single KPI."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/calculate",
            json={
                "kpi_id": "data_freshness_lag",
                "use_cache": False,
                "force_refresh": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["kpi_id"] == "data_freshness_lag"
        assert "value" in data

    def test_calculate_kpi_with_context(self, mock_calculator):
        """Should calculate KPI with context."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/calculate",
            json={
                "kpi_id": "data_freshness_lag",
                "context": {
                    "brand": "Kisqali",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
            },
        )

        assert response.status_code == 200

    def test_calculate_kpi_not_found(self, mock_calculator):
        """Should return 404 for missing KPI."""
        mock_result = MagicMock()
        mock_result.error = "KPI not found: invalid_kpi"
        mock_calculator.calculate.return_value = mock_result

        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/calculate",
            json={"kpi_id": "invalid_kpi"},
        )

        assert response.status_code == 404


class TestBatchCalculateKPIs:
    """Tests for POST /api/kpis/batch."""

    def test_batch_calculate_success(self, mock_calculator):
        """Should calculate multiple KPIs."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/batch",
            json={
                "kpi_ids": ["data_freshness_lag", "data_completeness"],
                "use_cache": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_kpis" in data
        assert "successful" in data
        assert "failed" in data

    def test_batch_calculate_by_workstream(self, mock_calculator):
        """Should calculate all KPIs in a workstream."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/batch",
            json={"workstream": "ws1_data_quality"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["workstream"] == "ws1_data_quality"

    def test_batch_calculate_with_context(self, mock_calculator):
        """Should apply context to batch calculation."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/batch",
            json={
                "kpi_ids": ["data_freshness_lag"],
                "context": {"brand": "Fabhalta"},
            },
        )

        assert response.status_code == 200


class TestInvalidateCache:
    """Tests for POST /api/kpis/invalidate."""

    def test_invalidate_all_cache(self, mock_calculator):
        """Should invalidate all cached KPIs."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/invalidate",
            json={"invalidate_all": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["invalidated_count"] == 5
        assert "All KPI cache entries invalidated" in data["message"]

    def test_invalidate_single_kpi_cache(self, mock_calculator):
        """Should invalidate cache for single KPI."""
        mock_calculator.invalidate_cache.return_value = 1

        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/invalidate",
            json={"kpi_id": "data_freshness_lag"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["invalidated_count"] == 1
        assert "data_freshness_lag" in data["message"]

    def test_invalidate_workstream_cache(self, mock_calculator):
        """Should invalidate cache for workstream."""
        mock_calculator.invalidate_cache.return_value = 10

        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/invalidate",
            json={"workstream": "ws1_data_quality"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["invalidated_count"] == 10

    def test_invalidate_invalid_workstream(self, mock_calculator):
        """Should return 400 for invalid workstream."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/invalidate",
            json={"workstream": "invalid_workstream"},
        )

        assert response.status_code == 400


class TestKPIHealthCheck:
    """Tests for GET /api/kpis/health."""

    def test_health_check_healthy(self, mock_calculator):
        """Should return healthy status."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        with patch("src.api.routes.kpi.get_registry"):
            response = client.get("/api/kpis/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["registry_loaded"] is True
        assert data["total_kpis"] > 0
        assert data["cache_enabled"] is True
        assert data["database_connected"] is True

    def test_health_check_degraded_no_kpis(self, mock_calculator):
        """Should return degraded status when no KPIs loaded."""
        mock_calculator.list_kpis.return_value = []

        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        with patch("src.api.routes.kpi.get_registry"):
            response = client.get("/api/kpis/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "No KPIs loaded" in data["error"]

    def test_health_check_includes_workstreams(self, mock_calculator):
        """Should include available workstreams."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        with patch("src.api.routes.kpi.get_registry"):
            response = client.get("/api/kpis/health")

        assert response.status_code == 200
        data = response.json()
        assert "workstreams_available" in data
