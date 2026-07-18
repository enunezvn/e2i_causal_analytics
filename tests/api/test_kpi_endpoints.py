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

    def test_get_kpi_value_with_segment_filter(self, mock_calculator):
        """Should filter by severity tier when segment is provided (migration 105)."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag", params={"segment": "low_severity"})

        assert response.status_code == 200
        mock_calculator.calculate.assert_called_with(
            kpi_id="data_freshness_lag",
            use_cache=True,
            force_refresh=False,
            context={"segment": "low_severity"},
        )

    def test_get_kpi_value_with_therapy_line_filter(self, mock_calculator):
        """Should filter by line of therapy when therapy_line is provided (migration 105)."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.get("/api/kpis/data_freshness_lag", params={"therapy_line": "0"})

        assert response.status_code == 200
        mock_calculator.calculate.assert_called_with(
            kpi_id="data_freshness_lag",
            use_cache=True,
            force_refresh=False,
            context={"therapy_line": "0"},
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

    def test_calculate_kpi_with_segment_and_therapy_line_context(self, mock_calculator):
        """Should thread segment and therapy_line from context into the calculator call
        (migration 105 patient-segment axes)."""
        app.dependency_overrides[get_kpi_calculator] = lambda: mock_calculator
        response = client.post(
            "/api/kpis/calculate",
            json={
                "kpi_id": "data_freshness_lag",
                "context": {
                    "brand": "Kisqali",
                    "segment": "high_severity",
                    "therapy_line": "2",
                },
            },
        )

        assert response.status_code == 200
        mock_calculator.calculate.assert_called_with(
            kpi_id="data_freshness_lag",
            use_cache=True,
            force_refresh=False,
            context={
                "brand": "Kisqali",
                "segment": "high_severity",
                "therapy_line": "2",
            },
        )

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


# =============================================================================
# KPI HISTORY (time-series KPI-history view; migration 079 + history_backfill)
# =============================================================================

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402


class _FakeQuery:
    """Fluent stand-in for the async Supabase query builder used by the
    kpi_history repo + the ROI backfill handler."""

    def __init__(self, rows):
        self._rows = rows

    def table(self, *a, **k):
        return self

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def gte(self, *a, **k):
        return self

    def lte(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    @property
    def not_(self):
        return self

    def is_(self, *a, **k):
        return self

    async def execute(self):
        return SimpleNamespace(data=self._rows)


class TestKPIHistoryEndpoint:
    """GET /api/kpis/{kpi_id}/history."""

    def test_history_returns_points(self):
        rows = [
            {"metric_date": "2026-05-01", "value": 1.83, "status": "warning"},
            {"metric_date": "2026-06-01", "value": 1.85, "status": "warning"},
        ]
        fake_repo = SimpleNamespace(get_history=AsyncMock(return_value=rows))
        with patch(
            "src.repositories.kpi_history.get_kpi_history_repository",
            new=AsyncMock(return_value=fake_repo),
        ):
            resp = client.get("/api/kpis/WS3-BI-010/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["kpi_id"] == "WS3-BI-010"
        assert body["count"] == 2
        assert [p["metric_date"] for p in body["points"]] == ["2026-05-01", "2026-06-01"]
        assert body["points"][0]["value"] == 1.83

    def test_history_empty_for_point_in_time_kpi(self):
        fake_repo = SimpleNamespace(get_history=AsyncMock(return_value=[]))
        with patch(
            "src.repositories.kpi_history.get_kpi_history_repository",
            new=AsyncMock(return_value=fake_repo),
        ):
            resp = client.get("/api/kpis/WS1-DQ-001/history")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["points"] == []


class TestKPIHistoryCoverageEndpoint:
    """GET /api/kpis/history/coverage."""

    def test_coverage_groups_scopes_per_kpi(self):
        rows = [
            {
                "kpi_id": "WS3-BI-007",
                "brand": "Kisqali",
                "points": 35,
                "first_date": "2023-08-01",
                "last_date": "2026-06-01",
            },
            {
                "kpi_id": "WS3-BI-007",
                "brand": "Fabhalta",
                "points": 35,
                "first_date": "2023-09-01",
                "last_date": "2026-05-01",
            },
            {
                "kpi_id": "WS3-BI-010",
                "brand": "",
                "points": 163,
                "first_date": "2013-01-01",
                "last_date": "2026-07-01",
            },
        ]
        fake_repo = SimpleNamespace(get_coverage=AsyncMock(return_value=rows))
        with patch(
            "src.repositories.kpi_history.get_kpi_history_repository",
            new=AsyncMock(return_value=fake_repo),
        ):
            resp = client.get("/api/kpis/history/coverage")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        by_id = {e["kpi_id"]: e for e in body["coverage"]}
        # Per-brand-only KPI: no '' scope, brands sorted, points summed,
        # date span = min/max across scopes.
        nbrx = by_id["WS3-BI-007"]
        assert nbrx["brands"] == ["Fabhalta", "Kisqali"]
        assert nbrx["points"] == 70
        assert nbrx["first_date"] == "2023-08-01"
        assert nbrx["last_date"] == "2026-06-01"
        # Global KPI keeps its '' scope visible.
        assert by_id["WS3-BI-010"]["brands"] == [""]

    def test_coverage_empty_when_no_history(self):
        fake_repo = SimpleNamespace(get_coverage=AsyncMock(return_value=[]))
        with patch(
            "src.repositories.kpi_history.get_kpi_history_repository",
            new=AsyncMock(return_value=fake_repo),
        ):
            resp = client.get("/api/kpis/history/coverage")
        assert resp.status_code == 200
        assert resp.json() == {"coverage": [], "total": 0}


class TestWeeklyCapture:
    """src.kpi.history_capture — append-only capture of present-state KPIs."""

    def _result(self, value, status="good", error=None):
        return SimpleNamespace(value=value, status=SimpleNamespace(value=status), error=error)

    def test_capture_writes_todays_point_and_skips_failures(self):
        from src.kpi import history_capture

        def fake_calculate(kpi_id, use_cache=True, force_refresh=False):
            if kpi_id == "WS3-BI-004":
                return self._result(None, error="KPI WS3-BI-004 unavailable: no data")
            return self._result(0.87)

        calculator = SimpleNamespace(calculate=fake_calculate)
        upserted: list = []

        async def fake_upsert(points):
            upserted.extend(points)
            return len(points)

        fake_repo = SimpleNamespace(upsert_points=fake_upsert)
        with (
            patch("src.api.routes.kpi.get_kpi_calculator", return_value=calculator),
            patch(
                "src.repositories.kpi_history.get_kpi_history_repository",
                new=AsyncMock(return_value=fake_repo),
            ),
        ):
            summary = asyncio.run(history_capture.run_capture(["WS1-DQ-001", "WS3-BI-004"]))

        # The healthy KPI wrote exactly one append-only point for today.
        assert summary["written"] == {"WS1-DQ-001": 0.87}
        assert len(upserted) == 1
        point = upserted[0]
        assert point["kpi_id"] == "WS1-DQ-001"
        assert point["source"] == "weekly_capture"
        assert point["metric_date"] == summary["date"]
        assert point["brand"] == "" and point["region"] == ""
        assert point["is_synthetic"] is True
        # The failing KPI wrote NOTHING and surfaced its error honestly.
        assert "WS3-BI-004" in summary["errors"]
        assert all(p["kpi_id"] != "WS3-BI-004" for p in upserted)

    def test_purge_deletes_only_weekly_capture_source(self):
        from src.kpi import history_capture

        deleted_pairs: list = []

        async def fake_delete(kpi_id, source):
            deleted_pairs.append((kpi_id, source))
            return 1

        fake_repo = SimpleNamespace(delete_source=fake_delete)
        with patch(
            "src.repositories.kpi_history.get_kpi_history_repository",
            new=AsyncMock(return_value=fake_repo),
        ):
            asyncio.run(history_capture.purge_captures())

        assert deleted_pairs, "purge must target every capture KPI"
        assert {s for _, s in deleted_pairs} == {"weekly_capture"}
        assert {k for k, _ in deleted_pairs} == set(history_capture.CAPTURE_KPI_IDS)


class TestROIHistoryHandler:
    """The ROI backfill handler aggregates real business_metrics.roi monthly."""

    def test_aggregates_mean_roi_per_month_and_brand(self):
        from src.kpi.history_backfill import _backfill_roi

        rows = [
            {"metric_date": "2026-05-01", "brand": "Kisqali", "roi": 1.8},
            {"metric_date": "2026-05-01", "brand": "Fabhalta", "roi": 2.0},
            {"metric_date": "2026-06-01", "brand": "Kisqali", "roi": 1.9},
        ]
        kpi_meta = SimpleNamespace(id="WS3-BI-010", threshold=None)
        points = asyncio.run(_backfill_roi(_FakeQuery(rows), kpi_meta))

        # Global (brand='') = mean roi per month across brands.
        glob = {p["metric_date"]: p for p in points if p["brand"] == ""}
        assert glob["2026-05-01"]["value"] == pytest.approx((1.8 + 2.0) / 2)
        assert glob["2026-06-01"]["value"] == pytest.approx(1.9)
        assert glob["2026-05-01"]["source"] == "business_metrics.roi"
        assert glob["2026-05-01"]["region"] == ""
        # Per-brand series present too.
        kis = {p["metric_date"]: p for p in points if p["brand"] == "Kisqali"}
        assert kis["2026-05-01"]["value"] == pytest.approx(1.8)
        # Every point carries the KPI id + synthetic provenance.
        assert all(p["kpi_id"] == "WS3-BI-010" and p["is_synthetic"] for p in points)


class TestKPIHistorySegmentedEndpoint:
    """GET /api/kpis/{kpi_id}/history/segmented (migration 110 live compute)."""

    @staticmethod
    def _rows():
        # Real kpi_query output shape (dry-run validated 2026-07-18).
        base = {"data_min": "2026-01-01", "data_max": "2026-06-30"}
        return [
            {"month_start": "2026-01-01", "bucket": "low_severity", "value": 3, **base},
            {"month_start": "2026-01-01", "bucket": "medium_severity", "value": 7, **base},
            {"month_start": "2026-01-01", "bucket": "high_severity", "value": 5, **base},
            {"month_start": "2026-02-01", "bucket": "low_severity", "value": 4, **base},
        ]

    def test_segment_axis_returns_ordered_zero_filled_series(self):
        with patch(
            "src.kpi.segmented_history.fetch_segmented_rows",
            new=AsyncMock(return_value=self._rows()),
        ) as fetch:
            resp = client.get(
                "/api/kpis/WS3-BI-005/history/segmented",
                params={"axis": "segment", "brand": "Remibrutinib"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kpi_id"] == "WS3-BI-005"
        assert body["brand"] == "Remibrutinib"
        assert body["axis"] == "segment"
        assert body["data_through"] == "2026-06-30"
        assert [s["key"] for s in body["series"]] == [
            "low_severity",
            "medium_severity",
            "high_severity",
        ]
        assert body["series"][0]["label"] == "Low severity"
        # Feb has no medium/high rows -> genuine zeros, not missing points.
        medium = body["series"][1]
        assert medium["points"][1] == {"metric_date": "2026-02-01", "value": 0.0, "status": None}
        fetch.assert_awaited_once_with("WS3-BI-005", axis="segment", brand="Remibrutinib")

    def test_therapy_line_axis_and_value_filter(self):
        rows = [
            {
                "month_start": "2026-01-01",
                "bucket": "2",
                "value": 9,
                "data_min": "2026-01-01",
                "data_max": "2026-01-31",
            }
        ]
        with patch(
            "src.kpi.segmented_history.fetch_segmented_rows",
            new=AsyncMock(return_value=rows),
        ):
            resp = client.get(
                "/api/kpis/WS3-BI-006/history/segmented",
                params={"axis": "therapy_line", "value": "2"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert [s["key"] for s in body["series"]] == ["2"]
        assert body["series"][0]["label"] == "2 prior lines"
        assert body["series"][0]["points"] == [
            {"metric_date": "2026-01-01", "value": 9.0, "status": None}
        ]

    def test_unsupported_kpi_is_422_not_empty(self):
        resp = client.get("/api/kpis/WS3-BI-008/history/segmented", params={"axis": "segment"})
        assert resp.status_code == 422
        # The app's StarletteHTTPException handler wraps details in the
        # structured error envelope; assert on the surfaced message text.
        assert "WS3-BI-005" in resp.text

    def test_unknown_axis_is_422(self):
        resp = client.get("/api/kpis/WS3-BI-005/history/segmented", params={"axis": "region"})
        assert resp.status_code == 422

    def test_unknown_bucket_value_is_422(self):
        resp = client.get(
            "/api/kpis/WS3-BI-005/history/segmented",
            params={"axis": "segment", "value": "extreme"},
        )
        assert resp.status_code == 422

    def test_empty_rows_yield_empty_series(self):
        with patch(
            "src.kpi.segmented_history.fetch_segmented_rows",
            new=AsyncMock(return_value=[]),
        ):
            resp = client.get("/api/kpis/WS3-BI-007/history/segmented", params={"axis": "segment"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["series"] == []
        assert body["count"] == 0
        assert body["data_through"] is None
