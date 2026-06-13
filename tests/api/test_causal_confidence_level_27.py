"""RED-FIRST route-level tests for #27: confidence_level emitted in responses.

The frontend cannot honestly label "95% CI" unless the API reports the level
used. These tests hit the REAL demo-mode route (TestClient, no mocking) and
assert the response carries confidence_level, and that requesting a non-default
level (0.90) is echoed back so the UI labels the interval truthfully.

The demo path's consensus interval is a pinned-zero placeholder (every library
returns effect_estimate=0.0 -> std=0.0 -> CI = [0, 0] at ANY z). So the test
asserts the *label* is honest, not a non-zero half-width -- see the PR body for
the structural note that the demo CI is meaningless by construction and the real
path does not emit a consensus CI today.
"""

import pytest


@pytest.fixture
def parallel_pipeline_request():
    return {
        "libraries": ["econml", "dowhy"],
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "consensus_method": "variance_weighted",
        "timeout_seconds": 30,
    }


class TestParallelResponseConfidenceLevel:
    def test_demo_response_includes_confidence_level_default_095(self, parallel_pipeline_request):
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/parallel?demo_mode=true",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 200, response.text[:300]
        data = response.json()
        assert "confidence_level" in data, (
            "Response must expose confidence_level for honest CI labeling"
        )
        assert data["confidence_level"] == 0.95

    def test_demo_response_echoes_requested_090_level(self, parallel_pipeline_request):
        """Requesting confidence_level=0.90 must be echoed so the UI labels a 90% CI."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        req = dict(parallel_pipeline_request)
        req["confidence_level"] = 0.90

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/parallel?demo_mode=true",
            json=req,
        )
        assert response.status_code == 200, response.text[:300]
        data = response.json()
        assert data["confidence_level"] == 0.90, (
            f"Requested 0.90 must be echoed, got {data.get('confidence_level')}"
        )
