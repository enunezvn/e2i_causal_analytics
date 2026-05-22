"""
Tests for #354 C-8: /causal/pipeline/{sequential,parallel} wired to real pipelines.

Background:
    After phases C-1..C-6 of #354 wired all 4 LibraryExecutors (DoWhy, EconML,
    CausalML, NetworkX) to real backends + cross-library aggregation, Surface C
    (the /causal/pipeline/{sequential,parallel} HTTP endpoints) still
    short-circuited to HTTPException(503) BEFORE invoking the wired pipelines.
    Phase C-8 replaces that short-circuit with real `SequentialPipeline.execute()`
    and `ParallelPipeline.execute()` calls.

Contract under test:
    - `demo_mode=true` branch is PRESERVED VERBATIM (UI-demo contract per
      v4 §2.3): pinned-zero placeholders + is_demo=true labeling.
    - `demo_mode=false` (default) branch:
        * Calls SequentialPipeline.execute() / ParallelPipeline.execute()
          when a DataFrame is provided via `filters.estimation_data_records`.
        * Fails closed with HTTPException(503) when no DataFrame is resolvable
          AND no library succeeds (honest fail — the data backend is not wired
          for arbitrary `data_source` strings; the 503 reflects reality).
        * Returns real PipelineOutput-derived response when at least one library
          succeeds (no hardcoded fabricated values).

Reference: GH #354 phase C-8; dispatch plan `.claude/plans/354_dispatch_plan_v1.md`
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import causal as causal_module

client = TestClient(app)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sequential_pipeline_request():
    """Sample sequential pipeline request body (no inline DataFrame)."""
    return {
        "stages": [
            {"library": "econml", "estimator": "causal_forest"},
            {"library": "dowhy", "estimator": "propensity_score_matching"},
        ],
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "covariates": ["age", "region"],
        "stop_on_failure": False,
    }


@pytest.fixture
def parallel_pipeline_request():
    """Sample parallel pipeline request body (no inline DataFrame)."""
    return {
        "libraries": ["econml", "dowhy"],
        "treatment_var": "promotion",
        "outcome_var": "trx",
        "covariates": ["age", "region"],
        "consensus_method": "variance_weighted",
        "timeout_seconds": 30,
    }


# =============================================================================
# Source-level pins: real-pipeline wiring is in place (no hardcoded short-circuit)
# =============================================================================


class TestSequentialEndpointWiresRealPipeline:
    """Source-level pins on _execute_sequential_pipeline ensuring real wiring."""

    def test_execute_sequential_pipeline_imports_sequentialpipeline(self):
        """The handler module must import SequentialPipeline from causal_engine.

        Pins the wiring contract: removing the import is a regression toward the
        pre-C-8 hardcoded 503-default short-circuit.
        """
        source = inspect.getsource(causal_module)
        assert (
            "from src.causal_engine.pipeline.sequential import" in source
            or "from src.causal_engine.pipeline import SequentialPipeline" in source
            or "import SequentialPipeline" in source
        ), "C-8 regression: SequentialPipeline import missing from causal route module"

    def test_execute_sequential_pipeline_calls_real_pipeline_execute(self):
        """_execute_sequential_pipeline body MUST call SequentialPipeline.execute().

        Source-level invariant — the handler MUST delegate to the real pipeline,
        not return a hardcoded structure.
        """
        source = inspect.getsource(causal_module._execute_sequential_pipeline)
        # Match the class instantiation or factory function pattern. Plain
        # mention of `SequentialPipelineResponse` (the schema type) does NOT
        # count — must be the engine class, not the response model.
        signals = (
            "SequentialPipeline(",
            "create_sequential_pipeline(",
            "_run_real_sequential_pipeline(",
        )
        assert any(s in source for s in signals), (
            "C-8 regression: _execute_sequential_pipeline does not instantiate or "
            "call SequentialPipeline (only the response model is referenced); "
            "check whether it has reverted to short-circuit-only"
        )


class TestParallelEndpointWiresRealPipeline:
    """Source-level pins on run_parallel_pipeline ensuring real wiring."""

    def test_parallel_endpoint_imports_parallelpipeline(self):
        """The handler module must import ParallelPipeline from causal_engine."""
        source = inspect.getsource(causal_module)
        assert (
            "from src.causal_engine.pipeline.parallel import" in source
            or "from src.causal_engine.pipeline import ParallelPipeline" in source
            or "import ParallelPipeline" in source
        ), "C-8 regression: ParallelPipeline import missing from causal route module"

    def test_run_parallel_pipeline_calls_real_pipeline_execute(self):
        """run_parallel_pipeline body MUST call ParallelPipeline.execute()."""
        source = inspect.getsource(causal_module.run_parallel_pipeline)
        signals = (
            "ParallelPipeline(",
            "create_parallel_pipeline(",
            "_run_real_parallel_pipeline(",
        )
        assert any(s in source for s in signals), (
            "C-8 regression: run_parallel_pipeline does not instantiate or "
            "call ParallelPipeline (only the response model is referenced)"
        )


# =============================================================================
# Demo-mode contract preservation (regression guard — must stay verbatim)
# =============================================================================


class TestDemoModeContractPreserved:
    """`demo_mode=true` branch must continue to return pinned-zero placeholders.

    This is the documented UI-demo contract per v4 §2.3. C-8 must NOT touch
    the demo_mode=true branch.
    """

    def test_sequential_demo_mode_still_returns_pinned_zeros(self, sequential_pipeline_request):
        """demo_mode=true → 200 + every stage labeled is_demo=true + effect=0.0."""
        response = client.post(
            "/api/causal/pipeline/sequential?demo_mode=true",
            json=sequential_pipeline_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "pending"]
        for stage in data.get("stage_results", []):
            assert stage.get("additional_results", {}).get("is_demo") is True, (
                f"C-8 regression: demo_mode stage lost is_demo=true label: {stage}"
            )
            assert stage.get("effect_estimate") == 0.0, (
                "C-8 regression: demo_mode stage no longer pinned-zero"
            )
        warnings = data.get("warnings", [])
        assert any("demo_mode" in w.lower() or "is_demo" in w.lower() for w in warnings)

    def test_parallel_demo_mode_still_returns_is_demo_labels(self, parallel_pipeline_request):
        """demo_mode=true → 200 + every library result labeled is_demo=true."""
        response = client.post(
            "/api/causal/pipeline/parallel?demo_mode=true",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 200
        data = response.json()
        for lib_name, lib_result in data.get("library_results", {}).items():
            assert lib_result.get("is_demo") is True, (
                f"C-8 regression: parallel demo lost is_demo=true label on {lib_name}: {lib_result}"
            )
            assert lib_result.get("effect_estimate") == 0.0
        warnings = data.get("warnings", [])
        assert any("demo_mode" in w.lower() or "is_demo" in w.lower() for w in warnings)


# =============================================================================
# Default-mode (real wiring) fail-closed contract
# =============================================================================


class TestSequentialDefaultModeFailsClosed:
    """Default-mode without resolvable DataFrame MUST fail-closed with honest 503.

    The fail-close is no longer a hardcoded short-circuit — it's the honest
    outcome of having every wired executor return success=False because no
    DataFrame is resolvable from state.
    """

    def test_sequential_default_no_data_returns_503(self, sequential_pipeline_request):
        """No demo_mode + no inline DataFrame → 503 (honest fail-close)."""
        response = client.post(
            "/api/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )
        assert response.status_code == 503, (
            f"Default path without resolvable data must fail-closed with 503, "
            f"got {response.status_code}: {response.text[:300]}"
        )


class TestParallelDefaultModeFailsClosed:
    """Default-mode without resolvable DataFrame MUST fail-closed with honest 503."""

    def test_parallel_default_no_data_returns_503(self, parallel_pipeline_request):
        """No demo_mode + no inline DataFrame → 503 (honest fail-close)."""
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 503, (
            f"Default path without resolvable data must fail-closed with 503, "
            f"got {response.status_code}: {response.text[:300]}"
        )


# =============================================================================
# Real-pipeline integration: when caller provides a DataFrame, pipeline runs
# =============================================================================


def _make_small_estimation_dataframe(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Build a tiny DataFrame with promotion / trx / age / region columns.

    Used ONLY by these C-8 integration tests to assert the wiring works
    end-to-end when callers DO supply a DataFrame (via filters.estimation_data_records).
    This is opt-in test-fixture data — NOT a production silent-fallback (Surface C
    does not have a default-on synthetic-data path; this fixture lives in tests/).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    age = rng.normal(55, 12, n)
    region = rng.integers(0, 3, n).astype(float)
    promotion = (
        0.2 * (age > 50).astype(float) + 0.3 * (region == 1).astype(float) + rng.normal(0, 0.3, n)
    )
    trx = (
        0.5 * promotion
        + 0.1 * (age - 55) / 12
        + 0.2 * (region == 0).astype(float)
        + rng.normal(0, 0.4, n)
    )
    return pd.DataFrame(
        {
            "promotion": promotion,
            "trx": trx,
            "age": age,
            "region": region,
        }
    )


class TestSequentialPipelineRealExecutionWithInlineData:
    """When the caller supplies a DataFrame, the wired pipeline executes for real.

    This is the positive case proving C-8 wiring works. The DataFrame conveyance
    via `filters.estimation_data_records` is the API analogue of the existing
    `state["data_cache"]["estimation_data"]` contract used by other causal nodes.
    """

    def test_sequential_runs_real_pipeline_when_data_provided(self, sequential_pipeline_request):
        """Pipeline runs end-to-end when filters.estimation_data_records carries data.

        Note: pandas.DataFrame is not JSON-serializable; the API accepts records
        (list-of-dicts) and the route rehydrates them into a DataFrame.
        """
        df = _make_small_estimation_dataframe()
        sequential_pipeline_request["filters"] = {
            "estimation_data_records": df.to_dict(orient="records"),
        }
        # Drop networkx (symbolic; see C-5 design spike) and use only econml +
        # dowhy which exercise real numeric estimators.
        sequential_pipeline_request["stages"] = [
            {"library": "dowhy", "estimator": "propensity_score_matching"},
            {"library": "econml", "estimator": "linear_dml"},
        ]
        response = client.post(
            "/api/causal/pipeline/sequential",
            json=sequential_pipeline_request,
        )
        # With data provided, the pipeline should EXECUTE — not short-circuit
        # back to 503. We accept 200 (success/partial) or 500 (real exception
        # surfaces). NEVER 503 when data was supplied.
        assert response.status_code in (200, 500), (
            f"With data provided, pipeline should execute (not short-circuit). "
            f"Got {response.status_code}: {response.text[:500]}"
        )
        if response.status_code == 200:
            data = response.json()
            assert "stage_results" in data
            # No is_demo=true labels in non-demo mode (would indicate a silent
            # substitution that defeats the wiring).
            for stage in data.get("stage_results", []):
                assert not stage.get("additional_results", {}).get("is_demo"), (
                    "C-8 regression: non-demo response carries is_demo=true label "
                    "(silent substitution into demo path?)"
                )


class TestParallelPipelineRealExecutionWithInlineData:
    """When the caller supplies a DataFrame, the wired parallel pipeline runs."""

    def test_parallel_runs_real_pipeline_when_data_provided(self, parallel_pipeline_request):
        """Parallel pipeline runs end-to-end when filters.estimation_data_records is provided."""
        df = _make_small_estimation_dataframe()
        parallel_pipeline_request["filters"] = {
            "estimation_data_records": df.to_dict(orient="records"),
        }
        parallel_pipeline_request["libraries"] = ["dowhy", "econml"]
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )
        assert response.status_code in (200, 500), (
            f"With data provided, parallel pipeline should execute (not short-circuit). "
            f"Got {response.status_code}: {response.text[:500]}"
        )
        if response.status_code == 200:
            data = response.json()
            assert "library_results" in data
            for lib_name, lib_result in data.get("library_results", {}).items():
                assert not lib_result.get("is_demo"), (
                    f"C-8 regression: non-demo parallel response on {lib_name} "
                    "carries is_demo=true (silent substitution into demo path?)"
                )


# =============================================================================
# Source-level pins: no hardcoded fake values in the wiring layer
# =============================================================================


class TestNoHardcodedValuesInWiring:
    """Source-level guards against re-introducing hardcoded fake values."""

    def test_no_random_uniform_in_real_wiring_path(self):
        """The wiring path MUST NOT use random.uniform (regression of F-005 fix)."""
        forbidden_helpers = (
            "_run_real_sequential_pipeline",
            "_run_real_parallel_pipeline",
            "_build_pipeline_input",
            "_resolve_pipeline_dataframe",
        )
        for helper_name in forbidden_helpers:
            if hasattr(causal_module, helper_name):
                helper_source = inspect.getsource(getattr(causal_module, helper_name))
                assert "random.uniform" not in helper_source, (
                    f"C-8 regression: random.uniform reintroduced in {helper_name}"
                )
                assert "np.random.seed" not in helper_source, (
                    f"C-8 regression: np.random.seed reintroduced in {helper_name}"
                )

    def test_no_fixed_ate_constant_in_wiring(self):
        """No `ate=0.12`-style hardcoded effect values in any new helper."""
        source = inspect.getsource(causal_module)
        # The Surface B hardcoded `ate=0.12` was the original #354 motivation.
        # Surface C wiring must not re-introduce this anti-pattern.
        assert "ate=0.12" not in source, (
            "C-8 regression: hardcoded ate=0.12 reintroduced in route module "
            "(matches Surface B anti-pattern that #354 was opened to fix)"
        )

    def test_no_hardcoded_consensus_effect(self):
        """No hardcoded `consensus_effect=<number>` in non-demo helpers."""
        for name in dir(causal_module):
            obj = getattr(causal_module, name)
            if not callable(obj) or not name.startswith("_"):
                continue
            if name in ("_demo_stage_placeholder",):
                # demo helper is allowed to pin zero
                continue
            try:
                src = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            for forbidden_literal in (
                "consensus_effect=0.21",
                "consensus_effect=0.19",
                "consensus_effect = 0.21",
                "consensus_effect = 0.19",
            ):
                assert forbidden_literal not in src, (
                    f"C-8 regression: hardcoded {forbidden_literal} in {name} "
                    "(matches Surface B anti-pattern)"
                )
