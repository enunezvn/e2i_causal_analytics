"""#1834 — ``POST /gaps/analyze`` validates the ``time_period`` grammar (422 that
lists the accepted forms) and returns the RESOLVED window the analysis compared.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Module-level on purpose (same pattern as test_route_shadow_regression.py and
# test_cors_configuration.py): importing src.api.main costs ~12-13 s warm and
# more on a cold cache; at module level it is paid once at collection, outside
# pytest-timeout's per-test budget (CI Unit Tests: --timeout=30,
# timeout_method=thread, which os._exit()s the xdist worker on overrun).
from src.api.dependencies.auth import require_analyst
from src.api.main import app
from src.api.routes.gaps import (
    GapAnalysisResponse,
    GapAnalysisStatus,
    RunGapAnalysisRequest,
    _execute_gap_analysis,
)

RESOLVED = {
    "time_period": "current_quarter",
    "period_start": "2026-07-01",
    "period_end": "2026-08-30",
    "prior_start": "2026-04-01",
    "prior_end": "2026-06-30",
}


# ---------------------------------------------------------------------------
# Request model: the grammar is enforced at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_request_model_rejects_unknown_time_period_listing_accepted_forms():
    with pytest.raises(ValidationError) as excinfo:
        RunGapAnalysisRequest(query="q", brand="kisqali", time_period="bogus")

    errors = excinfo.value.errors()
    assert [e["loc"] for e in errors] == [("time_period",)]
    msg = errors[0]["msg"]
    for form in ("current_quarter", "previous_quarter", "Q#_YYYY", "YYYY-Q#", "YTD", "MTD"):
        assert form in msg, f"{form!r} missing from 422 message: {msg}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "label",
    [
        "current_quarter",
        "previous_quarter",
        "last_quarter",
        "Q3_2026",
        "2024-Q3",
        "YTD",
        "MTD",
        "2026-07-01_2026-08-30",
    ],
)
def test_request_model_accepts_every_documented_form(label):
    req = RunGapAnalysisRequest(query="q", brand="kisqali", time_period=label)
    assert req.time_period == label


@pytest.mark.unit
def test_request_default_is_current_quarter_and_valid():
    req = RunGapAnalysisRequest(query="q", brand="kisqali")
    assert req.time_period == "current_quarter"


# ---------------------------------------------------------------------------
# HTTP boundary: the real app's 422 envelope
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_http_422_on_bogus_time_period_names_the_field_and_the_accepted_forms():
    app.dependency_overrides[require_analyst] = lambda: {"user_id": "t", "role": "analyst"}
    try:
        # Deliberately NOT `with TestClient(app)`: the context manager runs main.py's
        # full lifespan (BentoML/Redis/FalkorDB connection retries), which measured
        # ~23 s with no reachable services — past the CI Unit Tests job's
        # `--timeout=30` with `timeout_method=thread`, which os._exit()s the xdist
        # worker instead of failing (CI job 99277685267). The 422 envelope under
        # test is produced by the REAL app's RequestValidationError handler before
        # any endpoint or startup state is touched, so the lifespan is not needed.
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/gaps/analyze",
            json={"query": "q", "brand": "kisqali", "time_period": "bogus"},
        )
    finally:
        app.dependency_overrides.pop(require_analyst, None)

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["category"] == "validation"
    schema_errors = body["details"]["schema_errors"]
    assert [e["field"] for e in schema_errors] == ["body.time_period"]
    assert "current_quarter" in schema_errors[0]["message"]
    assert "YYYY-MM-DD_YYYY-MM-DD" in schema_errors[0]["message"]


# ---------------------------------------------------------------------------
# Response: the resolved window is a first-class field
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_response_model_has_optional_resolved_period():
    resp = GapAnalysisResponse(
        analysis_id="gap_x",
        status=GapAnalysisStatus.COMPLETED,
        brand="kisqali",
        metrics_analyzed=["trx"],
        segments_analyzed=1,
        resolved_period=RESOLVED,
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["resolved_period"] == RESOLVED

    pending = GapAnalysisResponse(
        analysis_id="gap_y",
        status=GapAnalysisStatus.PENDING,
        brand="kisqali",
        metrics_analyzed=["trx"],
        segments_analyzed=0,
    )
    assert pending.resolved_period is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_maps_the_graph_resolved_period_into_the_response():
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
            "executive_summary": "s",
            "key_insights": [],
            "warnings": [],
            "detection_latency_ms": 1,
            "roi_latency_ms": 1,
            "resolved_period": dict(RESOLVED),
        }
    )
    request = RunGapAnalysisRequest(query="q", brand="kisqali")

    with patch("src.agents.gap_analyzer.graph.create_gap_analyzer_graph", return_value=mock_graph):
        response = await _execute_gap_analysis(request)

    assert response.status == GapAnalysisStatus.COMPLETED
    assert response.resolved_period is not None
    assert response.resolved_period.model_dump(mode="json") == RESOLVED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_leaves_resolved_period_empty_on_a_failed_run():
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "status": "failed",
            "errors": [{"node": "gap_detector", "error": "Unsupported time_period 'x'"}],
            "warnings": [],
        }
    )
    request = RunGapAnalysisRequest(query="q", brand="kisqali")

    with patch("src.agents.gap_analyzer.graph.create_gap_analyzer_graph", return_value=mock_graph):
        response = await _execute_gap_analysis(request)

    assert response.status == GapAnalysisStatus.FAILED
    assert response.resolved_period is None
