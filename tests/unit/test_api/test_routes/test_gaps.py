"""
Unit tests for src/api/routes/gaps.py

Tests cover:
- Gap analysis endpoints (run_gap_analysis, get_gap_analysis, list_opportunities, get_gap_health)
- Happy paths, error paths, edge cases
- Mock all external dependencies (GapAnalyzerAgent, in-memory storage)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from src.api.routes.gaps import (
    AnalysisStatus,
    GapType,
    ImplementationDifficulty,
    RunGapAnalysisRequest,
    _analyses_store,
    _convert_opportunities,
    _execute_gap_analysis,
    _generate_mock_response,
    get_gap_analysis,
    get_gap_health,
    list_opportunities,
    run_gap_analysis,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_user():
    """Mock authenticated user with analyst role."""
    return {"user_id": "test_user", "role": "analyst"}


@pytest.fixture
def sample_gap_request():
    """Sample gap analysis request."""
    return RunGapAnalysisRequest(
        query="Identify performance gaps for Kisqali in Q4",
        brand="kisqali",
        metrics=["trx", "market_share"],
        segments=["region"],
        gap_type=GapType.ALL,
    )


@pytest.fixture(autouse=True)
def clear_analyses_store():
    """Clear the analyses store before each test."""
    global _analyses_store
    _analyses_store.clear()
    yield
    _analyses_store.clear()


# =============================================================================
# Endpoint Tests
# =============================================================================


class TestRunGapAnalysisEndpoint:
    """Tests for /gaps/analyze endpoint."""

    @pytest.mark.asyncio
    async def test_run_analysis_async_mode(self, sample_gap_request, mock_user):
        """Test gap analysis in async mode."""
        response = await run_gap_analysis(
            sample_gap_request,
            BackgroundTasks(),
            async_mode=True,
            user=mock_user,
        )

        assert response.status == AnalysisStatus.PENDING
        assert response.analysis_id.startswith("gap_")
        assert response.brand == "kisqali"

    @pytest.mark.asyncio
    async def test_run_analysis_sync_mode_with_mock_data(self, sample_gap_request, mock_user):
        """Test gap analysis in sync mode (uses mock data)."""
        with patch("src.api.routes.gaps._execute_gap_analysis") as mock_execute:
            mock_execute.return_value = MagicMock(
                analysis_id="test-id",
                status=AnalysisStatus.COMPLETED,
                brand="kisqali",
                metrics_analyzed=["trx"],
                segments_analyzed=4,
            )

            response = await run_gap_analysis(
                sample_gap_request,
                BackgroundTasks(),
                async_mode=False,
                user=mock_user,
            )

            assert response.status == AnalysisStatus.COMPLETED
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_analysis_sync_mode_error(self, sample_gap_request, mock_user):
        """Test gap analysis error handling in sync mode."""
        with patch(
            "src.api.routes.gaps._execute_gap_analysis", side_effect=Exception("Analysis failed")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await run_gap_analysis(
                    sample_gap_request,
                    BackgroundTasks(),
                    async_mode=False,
                    user=mock_user,
                )

            assert exc_info.value.status_code == 500


class TestGetGapAnalysisEndpoint:
    """Tests for /gaps/{analysis_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_analysis_success(self):
        """Test retrieving gap analysis by ID."""
        # Add test analysis to store
        from src.api.routes.gaps import GapAnalysisResponse

        test_analysis = GapAnalysisResponse(
            analysis_id="test-id",
            status=AnalysisStatus.COMPLETED,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=4,
        )
        _analyses_store["test-id"] = test_analysis

        response = await get_gap_analysis("test-id")

        assert response.analysis_id == "test-id"
        assert response.status == AnalysisStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self):
        """Test analysis not found error."""
        with pytest.raises(HTTPException) as exc_info:
            await get_gap_analysis("nonexistent-id")

        assert exc_info.value.status_code == 404


class TestListOpportunitiesEndpoint:
    """Tests for /gaps/opportunities endpoint."""

    @pytest.mark.asyncio
    async def test_list_opportunities_empty(self):
        """Test listing opportunities when none exist."""
        response = await list_opportunities(
            brand=None,
            min_roi=None,
            difficulty=None,
            limit=20,
        )

        assert response.total_count == 0
        assert len(response.opportunities) == 0

    @pytest.mark.asyncio
    async def test_list_opportunities_with_filters(self):
        """Test listing opportunities with filters."""
        # Add test analysis with opportunities
        from src.api.routes.gaps import (
            GapAnalysisResponse,
            PerformanceGap,
            PrioritizedOpportunity,
            ROIEstimate,
        )

        test_opp = PrioritizedOpportunity(
            rank=1,
            gap=PerformanceGap(
                gap_id="gap1",
                metric="trx",
                segment="region",
                segment_value="Northeast",
                current_value=85.0,
                target_value=100.0,
                gap_size=15.0,
                gap_percentage=15.0,
                gap_type="vs_target",
            ),
            roi_estimate=ROIEstimate(
                gap_id="gap1",
                estimated_revenue_impact=500000.0,
                estimated_cost_to_close=100000.0,
                expected_roi=5.0,
                risk_adjusted_roi=4.0,
                payback_period_months=6,
                attribution_level="partial",
                attribution_rate=0.7,
                confidence=0.8,
            ),
            recommended_action="Increase coverage",
            implementation_difficulty=ImplementationDifficulty.LOW,
            time_to_impact="3-6 months",
        )

        test_analysis = GapAnalysisResponse(
            analysis_id="test-id",
            status=AnalysisStatus.COMPLETED,
            brand="kisqali",
            metrics_analyzed=["trx"],
            segments_analyzed=4,
            prioritized_opportunities=[test_opp],
        )
        _analyses_store["test-id"] = test_analysis

        response = await list_opportunities(
            brand="kisqali",
            min_roi=2.0,
            difficulty=ImplementationDifficulty.LOW,
            limit=20,
        )

        assert response.total_count == 1
        assert len(response.opportunities) == 1


class TestGetGapHealthEndpoint:
    """Tests for /gaps/health endpoint."""

    @pytest.mark.asyncio
    async def test_gap_health_agent_available(self):
        """Test health check when agent is available."""
        with patch("src.agents.gap_analyzer.GapAnalyzerAgent"):
            response = await get_gap_health()

            assert response.status == "healthy"
            assert response.agent_available is True

    @pytest.mark.asyncio
    async def test_gap_health_agent_unavailable(self):
        """Test health check when agent is not available."""
        # Mock the import to fail by making the module unavailable
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "src.agents.gap_analyzer" or name.startswith("src.agents.gap_analyzer."):
                raise ImportError("Module not available")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            response = await get_gap_health()

            assert response.status == "degraded"
            assert response.agent_available is False


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_convert_opportunities_valid_data(self):
        """Test converting agent output to API format."""
        opportunities = [
            {
                "rank": 1,
                "gap": {
                    "gap_id": "gap1",
                    "metric": "trx",
                    "segment": "region",
                    "segment_value": "Northeast",
                    "current_value": 85.0,
                    "target_value": 100.0,
                    "gap_size": 15.0,
                    "gap_percentage": 15.0,
                    "gap_type": "vs_target",
                },
                "roi_estimate": {
                    "gap_id": "gap1",
                    "estimated_revenue_impact": 500000.0,
                    "estimated_cost_to_close": 100000.0,
                    "expected_roi": 5.0,
                    "risk_adjusted_roi": 4.0,
                    "payback_period_months": 6,
                    "attribution_level": "partial",
                    "attribution_rate": 0.7,
                    "confidence": 0.8,
                },
                "recommended_action": "Increase coverage",
                "implementation_difficulty": "low",
                "time_to_impact": "3-6 months",
            }
        ]

        result = _convert_opportunities(opportunities)

        assert len(result) == 1
        assert result[0].rank == 1
        assert result[0].gap.gap_id == "gap1"

    def test_convert_opportunities_invalid_data(self):
        """Test converting with invalid data."""
        opportunities = [
            {"rank": 1, "gap": {}, "roi_estimate": {}}  # Missing required fields
        ]

        result = _convert_opportunities(opportunities)

        # Empty dicts successfully convert to objects with default values
        assert len(result) == 1
        assert result[0].gap.gap_id == ""  # Default value
        assert result[0].gap.metric == ""  # Default value

    def test_generate_mock_response(self, sample_gap_request):
        """Test mock response generation."""
        import time

        start_time = time.time()

        response = _generate_mock_response(sample_gap_request, start_time)

        assert response.brand == "kisqali"
        assert response.status == AnalysisStatus.COMPLETED
        assert len(response.warnings) > 0
        assert "mock data" in response.warnings[0].lower()


class TestExecuteGapAnalysis:
    """Tests for _execute_gap_analysis function."""

    @pytest.mark.asyncio
    async def test_execute_with_agent(self, sample_gap_request):
        """Test execution with real Gap Analyzer agent."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "status": "completed",
                "segments_analyzed": 4,
                "prioritized_opportunities": [],
                "quick_wins": [],
                "strategic_bets": [],
                "total_addressable_value": 0.0,
                "total_gap_value": 0.0,
                "executive_summary": "Test summary",
                "key_insights": [],
                "warnings": [],
                "detection_latency_ms": 100,
                "roi_latency_ms": 150,
            }
        )

        with patch(
            "src.agents.gap_analyzer.graph.create_gap_analyzer_graph", return_value=mock_graph
        ):
            response = await _execute_gap_analysis(sample_gap_request)

            assert response.status == AnalysisStatus.COMPLETED
            assert response.brand == "kisqali"

    @pytest.mark.asyncio
    async def test_execute_maps_errors_to_failed_even_if_status_says_completed(
        self, sample_gap_request
    ):
        """F2 belt-and-suspenders: a result carrying errors must map to FAILED.

        Even if the agent graph (incorrectly) reports status='completed', the route
        must NOT return a green HTTP-200 "no gaps" response when ``errors`` is present.
        The route forces AnalysisStatus.FAILED whenever ``result['errors']`` is non-empty.
        """
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                # Worst case: a stale/buggy node still claims completed...
                "status": "completed",
                # ...but a real terminal error was accumulated upstream.
                "errors": [{"node": "gap_detector", "error": "'region'"}],
                "segments_analyzed": 0,
                "prioritized_opportunities": [],
                "quick_wins": [],
                "strategic_bets": [],
                "total_addressable_value": 0.0,
                "total_gap_value": 0.0,
                "executive_summary": "No significant performance gaps.",
                "key_insights": [],
                "warnings": [],
                "detection_latency_ms": 5,
                "roi_latency_ms": 0,
            }
        )

        with patch(
            "src.agents.gap_analyzer.graph.create_gap_analyzer_graph", return_value=mock_graph
        ):
            response = await _execute_gap_analysis(sample_gap_request)

            assert response.status == AnalysisStatus.FAILED, (
                "route must force FAILED when the agent result carries errors, "
                f"got {response.status}"
            )

    @pytest.mark.asyncio
    async def test_execute_clean_no_gaps_stays_completed(self, sample_gap_request):
        """Regression guard: a clean completed result with NO errors stays COMPLETED."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "status": "completed",
                "errors": [],
                "segments_analyzed": 4,
                "prioritized_opportunities": [],
                "quick_wins": [],
                "strategic_bets": [],
                "total_addressable_value": 0.0,
                "total_gap_value": 0.0,
                "executive_summary": "No significant performance gaps.",
                "key_insights": [],
                "warnings": [],
                "detection_latency_ms": 100,
                "roi_latency_ms": 150,
            }
        )

        with patch(
            "src.agents.gap_analyzer.graph.create_gap_analyzer_graph", return_value=mock_graph
        ):
            response = await _execute_gap_analysis(sample_gap_request)

            assert response.status == AnalysisStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_mock_when_explicitly_allowed(
        self, sample_gap_request, monkeypatch
    ):
        """Mock-fallback is gated on E2I_REQUIRE_AGENT_IMPORT=0 (closed-by-default policy)."""
        monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "0")
        with patch(
            "src.agents.gap_analyzer.graph.create_gap_analyzer_graph", side_effect=ImportError
        ):
            response = await _execute_gap_analysis(sample_gap_request)

            assert response.status == AnalysisStatus.COMPLETED
            assert len(response.warnings) > 0

    @pytest.mark.asyncio
    async def test_execute_raises_503_when_mock_disabled(self, sample_gap_request, monkeypatch):
        """Closed-by-default: ImportError must raise 503 when mock-fallback is disabled."""
        monkeypatch.setenv("E2I_REQUIRE_AGENT_IMPORT", "1")
        with patch(
            "src.agents.gap_analyzer.graph.create_gap_analyzer_graph", side_effect=ImportError
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _execute_gap_analysis(sample_gap_request)
            assert exc_info.value.status_code == 503
            assert exc_info.value.detail["error"] == "agent_unavailable"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_list_opportunities_max_limit(self):
        """Test listing opportunities with max limit."""
        response = await list_opportunities(
            brand=None,
            min_roi=None,
            difficulty=None,
            limit=100,  # Max allowed
        )

        assert response.total_count <= 100

    @pytest.mark.asyncio
    async def test_run_analysis_all_gap_types(self, sample_gap_request, mock_user):
        """Test analysis with all gap types."""
        for gap_type in [
            GapType.VS_TARGET,
            GapType.VS_BENCHMARK,
            GapType.VS_POTENTIAL,
            GapType.TEMPORAL,
            GapType.ALL,
        ]:
            sample_gap_request.gap_type = gap_type

            response = await run_gap_analysis(
                sample_gap_request,
                BackgroundTasks(),
                async_mode=True,
                user=mock_user,
            )

            assert response is not None

    @pytest.mark.asyncio
    async def test_run_analysis_max_opportunities(self, sample_gap_request, mock_user):
        """Test analysis with max opportunities limit."""
        sample_gap_request.max_opportunities = 50  # Max allowed

        response = await run_gap_analysis(
            sample_gap_request,
            BackgroundTasks(),
            async_mode=True,
            user=mock_user,
        )

        assert response is not None
