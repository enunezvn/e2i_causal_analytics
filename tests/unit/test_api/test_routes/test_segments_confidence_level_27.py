"""RED-FIRST test for #27: /api/segments echoes the CATE CI confidence level.

The segment-analysis route accepts a configurable ``significance_level`` (alpha)
and forwards it into the heterogeneous optimizer, whose CATE CIs are computed at
that level. But the SegmentAnalysisResponse previously (a) documented
``cate_ci_lower/upper`` as a hardcoded "95% CI" and (b) never echoed the level,
so a frontend could not tell whether the bounds were 95%, 90%, etc.

This test pins the corrected contract: the response carries
``confidence_level == 1 - significance_level``. It exercises the REAL
``_generate_mock_response`` builder with a real ``RunSegmentAnalysisRequest`` --
the CI VALUES are illustrative (mock-data path, explicitly labeled), but the
confidence_level field under test is derived from real request plumbing.
"""

import time

from src.api.routes.segments import (
    RunSegmentAnalysisRequest,
    SegmentAnalysisResponse,
    _generate_mock_response,
)


def _request(significance_level: float) -> RunSegmentAnalysisRequest:
    return RunSegmentAnalysisRequest(
        query="Which HCP segments respond best to rep visits?",
        treatment_var="rep_visits",
        outcome_var="trx",
        segment_vars=["region", "specialty"],
        significance_level=significance_level,
    )


def test_segment_response_schema_has_confidence_level_default_095():
    resp = SegmentAnalysisResponse(analysis_id="seg_x", status="completed")
    assert resp.confidence_level == 0.95


def test_mock_response_echoes_default_095_level():
    resp = _generate_mock_response(_request(0.05), time.time())
    assert resp.confidence_level == 0.95


def test_mock_response_echoes_090_for_alpha_010():
    """significance_level=0.10 (alpha) => confidence_level 0.90, not 0.95."""
    resp = _generate_mock_response(_request(0.10), time.time())
    assert abs(resp.confidence_level - 0.90) < 1e-9, (
        f"alpha=0.10 must echo confidence_level=0.90, got {resp.confidence_level}"
    )


def test_full_request_significance_range_does_not_break_response_validation():
    """No previously-valid significance_level may fail response validation (#27 regression).

    The request allows significance_level in (0.0, 0.5), so confidence_level =
    1 - alpha spans (0.5, 1.0). The response field bound must accept that whole
    inverse range -- a too-narrow [0.80, 0.99] would 500 on a valid 0.30 request.
    """
    for alpha, expected_cl in [(0.30, 0.70), (0.001, 0.999), (0.499, 0.501)]:
        resp = _generate_mock_response(_request(alpha), time.time())
        assert abs(resp.confidence_level - expected_cl) < 1e-9, (
            f"alpha={alpha} must echo confidence_level={expected_cl}, got {resp.confidence_level}"
        )
