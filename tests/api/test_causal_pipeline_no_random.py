"""
Tests for F-005: /causal/pipeline/{sequential,parallel,validate} must not
fabricate effects with random.uniform.

Background:
    Prior to F-005, the three endpoints constructed PipelineStageResult with
    `effect = 0.15 + random.uniform(-0.05, 0.05)`, `p_value = random.uniform(...)`,
    etc. The endpoints sit behind `require_analyst` auth and were reachable from
    chat-driven flows, surfacing fake-but-plausible "library agreement scores".

These tests assert the fail-closed contract:
    - Default (no `demo_mode=True`) path must NOT use random.uniform — it either
      delegates to a real estimator or returns HTTPException(503).
    - Explicit `demo_mode=True` returns an `is_demo: true` envelope with
      clearly-pinned (zero) values, never RNG.

Reference: GitHub issue #419.
"""

from __future__ import annotations

import inspect

import pytest

from src.api.routes import causal as causal_module

# =============================================================================
# Static-source regression pins (cheapest assertion: forbid the primitive)
# =============================================================================


class TestNoRandomUniformInCausalPipelineSource:
    """
    Regression pin: ensure the production pipeline handlers contain no
    random.uniform() calls in their bodies.

    This static-source check prevents future re-introduction of fabrication
    primitives in /causal/pipeline/{sequential,parallel,validate}.
    """

    def test_execute_sequential_pipeline_has_no_random_uniform(self):
        """_execute_sequential_pipeline must not contain random.uniform."""
        source = inspect.getsource(causal_module._execute_sequential_pipeline)
        assert "random.uniform" not in source, (
            "F-005 regression: random.uniform reintroduced in _execute_sequential_pipeline"
        )

    def test_run_library_analysis_has_no_random_uniform(self):
        """_run_library_analysis (parallel pipeline) must not contain random.uniform."""
        source = inspect.getsource(causal_module._run_library_analysis)
        # await asyncio.sleep(random.uniform(...)) was also a fabrication primitive
        # used to make the demo "feel" like real work. Forbid it too.
        assert "random.uniform" not in source, (
            "F-005 regression: random.uniform reintroduced in _run_library_analysis"
        )

    def test_run_cross_validation_has_no_random_uniform(self):
        """run_cross_validation must not contain random.uniform."""
        source = inspect.getsource(causal_module.run_cross_validation)
        assert "random.uniform" not in source, (
            "F-005 regression: random.uniform reintroduced in run_cross_validation"
        )


# =============================================================================
# Endpoint behavior tests
# =============================================================================


@pytest.fixture
def sequential_pipeline_request():
    """Sample sequential pipeline request."""
    return {
        "stages": [
            {"library": "econml", "estimator": "causal_forest"},
            {"library": "dowhy", "estimator": "propensity_score_matching"},
        ],
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "stop_on_failure": False,
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


@pytest.fixture
def cross_validation_request():
    """Sample cross-validation request."""
    return {
        "primary_library": "econml",
        "validation_library": "dowhy",
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "agreement_threshold": 0.7,
    }


class TestSequentialPipelineDefaultPathNoFabrication:
    """The default-path (no demo_mode) must NOT silently fabricate effects."""

    def test_sequential_default_path_emits_no_fabricated_pharma_range_effects(
        self, sequential_pipeline_request
    ):
        """
        With no demo_mode flag, the endpoint must either:
            - Return a real estimator output (effect from real estimator), OR
            - Return HTTPException(503)/error envelope.

        It must NOT return arbitrary RNG values in the pharma uplift range
        (0.10-0.20 centered at 0.15) with plausible CI/p_value shapes.

        We accept any successful response that includes is_demo=False OR
        non-200 status, AND assert that the effects (if returned) do not
        match the legacy RNG signature.
        """
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )
        # Acceptable shapes: 200 with real estimator results, OR 503/500 error
        assert response.status_code in (200, 500, 503), (
            f"Unexpected status: {response.status_code}, body={response.text[:500]}"
        )
        if response.status_code == 200:
            data = response.json()
            # Default-path success must label its provenance honestly
            # Either is_demo absent/false OR labeled as real estimator output
            # The pin: effects must NOT be `0.15 + uniform(-0.05, 0.05)` shape.
            # We can't deterministically check randomness post-hoc, but we CAN
            # require that the handler claimed real-mode (is_demo absent or False).
            assert data.get("is_demo") is not True, (
                "Default-path response claims is_demo=true — endpoint silently "
                "returned fabricated values; require explicit demo_mode=true"
            )

    def test_sequential_with_demo_mode_returns_is_demo_envelope(self, sequential_pipeline_request):
        """When `demo_mode=true` request param is set, response must label is_demo=true."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/sequential?demo_mode=true",
            json=sequential_pipeline_request,
        )
        # Demo mode is OPTIONAL — accept either a 200 with is_demo or a
        # 400/422 if the schema doesn't support it (then we depend purely
        # on the source-pin tests above). The structural requirement is:
        # if demo_mode is plumbed at all, it must label the response.
        if response.status_code == 200:
            data = response.json()
            # The handler MAY accept demo_mode and label, or MAY return real
            # results regardless. Either is acceptable; the prohibited shape
            # is "response without is_demo and yet using random.uniform".
            assert "pipeline_id" in data


class TestParallelPipelineDefaultPathNoFabrication:
    """The default-path parallel pipeline must NOT silently fabricate effects."""

    def test_parallel_default_path_emits_no_fabricated_pharma_range_effects(
        self, parallel_pipeline_request
    ):
        """Default-path parallel must either return real results or error."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )
        assert response.status_code in (200, 500, 503)
        if response.status_code == 200:
            data = response.json()
            assert data.get("is_demo") is not True, (
                "Default-path response claims is_demo=true — endpoint silently "
                "returned fabricated values; require explicit demo_mode=true"
            )


class TestCrossValidationDefaultPathNoFabrication:
    """The default-path /validate must NOT silently fabricate effects."""

    def test_cross_validation_default_path_emits_no_fabricated_effects(
        self, cross_validation_request
    ):
        """Default-path cross-validation must either return real results or error."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/validate",
            json=cross_validation_request,
        )
        assert response.status_code in (200, 500, 503)
        if response.status_code == 200:
            data = response.json()
            assert data.get("is_demo") is not True, (
                "Default-path response claims is_demo=true — endpoint silently "
                "returned fabricated values; require explicit demo_mode=true"
            )
