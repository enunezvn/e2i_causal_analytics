"""
Tests for F-005: /causal/pipeline/{sequential,parallel,validate} must not
fabricate effects with random.uniform OR synthetic-data-backed real estimators.

Background:
    Prior to F-005, the three endpoints constructed PipelineStageResult with
    `effect = 0.15 + random.uniform(-0.05, 0.05)`, `p_value = random.uniform(...)`,
    etc. The endpoints sit behind `require_analyst` auth and were reachable from
    chat-driven flows, surfacing fake-but-plausible "library agreement scores".

    The first iteration of the fix moved the RNG out but left a synthetic
    seeded dataset feeding the real energy-score estimator. Codex iter-1
    flagged that as a labeling fabrication (HIGH-1): the response numbers
    are real estimates of a synthetic dataset, but consumers cannot tell
    they came from synthetic data. The fix is to fail-closed in the default
    (non-demo) path entirely.

These tests assert the fail-closed contract:
    - Default (no `demo_mode=True`) path MUST return HTTPException(503).
    - Explicit `demo_mode=True` returns clearly-labeled pinned-zero values.
    - Source-pin tests cover all helpers, not just the top-level handlers.

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
    Regression pin: ensure the pipeline handlers AND helpers contain no
    random.uniform() calls in their bodies.

    Coverage extended to helpers (per F-005 iter-1 MEDIUM-1): a future
    re-introduction of random.uniform inside _demo_stage_placeholder would
    otherwise bypass the static pins on just the top-level handlers.

    NOTE: _build_synthetic_pipeline_data was DELETED in iter-3 per F-005
    iter-2 codex HIGH-1 (dead-code-with-zero-production-consumers footgun).
    DELETE > LABEL per [[feedback-no-mocking-no-patching]].
    """

    def test_no_synthetic_pipeline_data_helper(self):
        """_build_synthetic_pipeline_data must remain DELETED (no resurrection)."""
        assert not hasattr(causal_module, "_build_synthetic_pipeline_data"), (
            "F-005 regression: _build_synthetic_pipeline_data has been resurrected. "
            "It was a synthetic-data helper with zero production consumers — deleted "
            "in iter-3 because LABEL-style preservation of dead-code is a footgun "
            "(see [[feedback-no-mocking-no-patching]])."
        )

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

    def test_demo_stage_placeholder_has_no_random_uniform(self):
        """Demo placeholder helper must emit pinned zeros, not RNG."""
        source = inspect.getsource(causal_module._demo_stage_placeholder)
        assert "random.uniform" not in source, (
            "F-005 regression: random.uniform reintroduced in _demo_stage_placeholder"
        )

    def test_run_sequential_pipeline_endpoint_has_no_random_uniform(self):
        """Sequential endpoint handler must not contain random.uniform."""
        source = inspect.getsource(causal_module.run_sequential_pipeline)
        assert "random.uniform" not in source

    def test_run_parallel_pipeline_endpoint_has_no_random_uniform(self):
        """Parallel endpoint handler must not contain random.uniform."""
        source = inspect.getsource(causal_module.run_parallel_pipeline)
        assert "random.uniform" not in source


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


class TestSequentialPipelineDefaultPath503:
    """Default-path /pipeline/sequential MUST fail-closed with 503.

    Post-#354 C-8 reasoning (2026-05-22): the 503 is no longer a hardcoded
    short-circuit. The endpoint now invokes the wired SequentialPipeline
    (C-1..C-6); when no DataFrame is resolvable from request filters, every
    wired executor fails-closed (refuses to fabricate synthetic data), and
    the response builder honestly reports 503 with _NO_RESOLVABLE_DATA_DETAIL.
    The behavioral invariant — "no real data → 503" — is preserved.
    """

    def test_sequential_default_path_returns_503(self, sequential_pipeline_request):
        """No demo_mode + no inline DataFrame → 503.

        The structured error response goes through src.api.main's global handler
        which wraps HTTPException(503) in DependencyError. The original detail
        is only surfaced in debug mode via `original_error`. The structural
        guarantee tested here is the 503 status — the detail is verified by
        the source-pin test on the _NO_REAL_DATA_BACKEND_DETAIL constant.

        Post-C-8: the 503 reflects the wired pipeline's honest fail-close
        after invoking every executor and receiving success=False from each.
        """
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )
        assert response.status_code == 503, (
            f"Default path must fail-closed with 503, got {response.status_code}: "
            f"{response.text[:500]}"
        )

    def test_no_real_data_backend_constant_explains_intent(self):
        """Source-level pin: the 503 detail constant must explicitly explain why."""
        from src.api.routes.causal import _NO_REAL_DATA_BACKEND_DETAIL

        detail_lower = _NO_REAL_DATA_BACKEND_DETAIL.lower()
        assert "real data" in detail_lower, (
            f"Constant must mention 'real data', got: {_NO_REAL_DATA_BACKEND_DETAIL}"
        )
        assert "demo_mode" in detail_lower, (
            f"Constant must mention 'demo_mode' (escape hatch), got: {_NO_REAL_DATA_BACKEND_DETAIL}"
        )

    def test_sequential_async_default_path_still_503(self, sequential_pipeline_request):
        """async_mode=true must also fail-closed when demo_mode is absent.

        The endpoint creates a pending response then schedules background work
        — but the background work itself will fail-closed. The endpoint should
        either reject up-front OR schedule and let the cached result reflect
        the failure. We accept both shapes (200 pending OR 503 up-front).
        """
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/sequential?async_mode=true",
            json=sequential_pipeline_request,
        )
        # Acceptable shapes: 200 pending OR 503 up-front
        assert response.status_code in (200, 503)


class TestSequentialPipelineDemoMode:
    """With demo_mode=true, response must label is_demo=true at every level."""

    def test_sequential_demo_mode_returns_pinned_zeros(self, sequential_pipeline_request):
        """demo_mode=true → 200 with stage_results[*].additional_results.is_demo=true."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/sequential?demo_mode=true",
            json=sequential_pipeline_request,
        )
        assert response.status_code == 200, (
            f"demo_mode=true must succeed, got {response.status_code}: {response.text[:300]}"
        )
        data = response.json()
        assert "pipeline_id" in data
        # Every stage must be labeled is_demo=true
        for stage in data.get("stage_results", []):
            assert stage.get("additional_results", {}).get("is_demo") is True, (
                f"Stage {stage} missing is_demo=true label"
            )
            assert stage.get("effect_estimate") == 0.0, (
                "Demo stage must have pinned-zero effect_estimate"
            )
        # Warning must explicitly call out demo_mode
        warnings = data.get("warnings", [])
        assert any("demo_mode" in w.lower() or "is_demo" in w.lower() for w in warnings), (
            f"Demo response must include demo_mode warning, got: {warnings}"
        )


class TestParallelPipelineDefaultPath503:
    """Default-path /pipeline/parallel MUST fail-closed with 503.

    Post-#354 C-8 reasoning (2026-05-22): same as the sequential class
    docstring — the endpoint now invokes the wired ParallelPipeline; the
    503 is the pipeline's honest fail-close when no DataFrame is resolvable
    from request filters.
    """

    def test_parallel_default_path_returns_503(self, parallel_pipeline_request):
        """No demo_mode + no inline DataFrame → 503 (post-C-8 honest fail-close)."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 503, (
            f"Default path must fail-closed with 503, got {response.status_code}"
        )


class TestParallelPipelineDemoMode:
    """With demo_mode=true, parallel pipeline must label is_demo=true."""

    def test_parallel_demo_mode_returns_is_demo(self, parallel_pipeline_request):
        """demo_mode=true → 200 with library_results[*].is_demo=true."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/pipeline/parallel?demo_mode=true",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 200
        data = response.json()
        # Every library result must be labeled is_demo=true
        for lib_name, lib_result in data.get("library_results", {}).items():
            assert lib_result.get("is_demo") is True, (
                f"Library {lib_name} missing is_demo=true label: {lib_result}"
            )
            assert lib_result.get("effect_estimate") == 0.0
        # Warning must explicitly call out demo_mode
        warnings = data.get("warnings", [])
        assert any("demo_mode" in w.lower() or "is_demo" in w.lower() for w in warnings)


class TestCrossValidationDefaultPath503:
    """Default-path /validate MUST fail-closed with 503."""

    def test_cross_validation_default_path_returns_503(self, cross_validation_request):
        """No demo_mode → 503."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/validate",
            json=cross_validation_request,
        )
        assert response.status_code == 503, (
            f"Default path must fail-closed with 503, got {response.status_code}"
        )


class TestCrossValidationDemoMode:
    """With demo_mode=true, /validate must label is_demo=true in recommendations.

    CrossValidationResponse has no is_demo field — codex iter-1 HIGH-3
    requires the demo label to be machine-readable. We encode it as the
    first item in the recommendations array with the literal token
    "is_demo=true:".
    """

    def test_cross_validation_demo_mode_labels_is_demo_in_recommendations(
        self, cross_validation_request
    ):
        """demo_mode=true → 200 with first recommendation containing 'is_demo=true'."""
        from fastapi.testclient import TestClient

        from src.api.main import app

        client = TestClient(app)
        response = client.post(
            "/api/causal/validate?demo_mode=true",
            json=cross_validation_request,
        )
        assert response.status_code == 200
        data = response.json()
        recommendations = data.get("recommendations", [])
        assert recommendations, "Demo response must include recommendations"
        # The first recommendation must contain the is_demo=true machine-readable label
        assert "is_demo=true" in recommendations[0].lower(), (
            f"First recommendation must include 'is_demo=true' label, got: {recommendations}"
        )
        # Effect values must be pinned zeros
        assert data["primary_effect"] == 0.0
        assert data["validation_effect"] == 0.0
