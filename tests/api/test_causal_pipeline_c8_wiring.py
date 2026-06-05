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
from src.api.schemas.causal import (
    ParallelPipelineResponse,
    SequentialPipelineResponse,
)

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
    DataFrame is resolvable from state (plus the data-required guard so
    NetworkX's symbolic success cannot mask a missing effect estimate).
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

    def test_sequential_networkx_plus_dowhy_no_data_still_returns_503(self):
        """NetworkX succeeds symbolically; DoWhy needs data → 503 (codex iter-2 HIGH).

        Per codex iter-2: a request that includes a data-required library
        (DoWhy/EconML/CausalML) but supplies no data should fail-close even
        when NetworkX succeeds. Otherwise NetworkX's symbolic graph-analysis
        success would mask the missing effect estimate the user asked for.
        """
        request_body = {
            "stages": [
                {"library": "networkx", "estimator": None},
                {"library": "dowhy", "estimator": "propensity_score_matching"},
            ],
            "treatment_var": "promotion",
            "outcome_var": "trx",
            "covariates": ["age"],
            "stop_on_failure": False,
        }
        response = client.post("/api/causal/pipeline/sequential", json=request_body)
        assert response.status_code == 503, (
            f"NetworkX + DoWhy without data must fail-close (DoWhy data-required); "
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

    def test_parallel_networkx_plus_econml_no_data_still_returns_503(self):
        """NetworkX succeeds symbolically; EconML needs data → 503 (codex iter-2 HIGH)."""
        request_body = {
            "libraries": ["networkx", "econml"],
            "treatment_var": "promotion",
            "outcome_var": "trx",
            "covariates": ["age"],
            "consensus_method": "variance_weighted",
            "timeout_seconds": 30,
        }
        response = client.post("/api/causal/pipeline/parallel", json=request_body)
        assert response.status_code == 503, (
            f"NetworkX + EconML without data must fail-close (EconML data-required); "
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
        """Pipeline runs end-to-end and returns 200 when data is provided.

        Note: pandas.DataFrame is not JSON-serializable; the API accepts records
        (list-of-dicts) and the route rehydrates them into a DataFrame.

        Per codex iter-1 MEDIUM: this test requires 200, not 200-or-500. A
        500 here would mean the pipeline fails internally during real
        execution — a real bug, not an acceptable outcome. The 503-removed
        invariant is enforced separately via the source-level pins above.
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
        assert response.status_code == 200, (
            f"With data provided, pipeline must execute and return 200. "
            f"Got {response.status_code}: {response.text[:500]}"
        )
        data = response.json()
        assert "stage_results" in data
        # At least one stage must report a real effect_estimate (proof that
        # the wired executor produced real output, not a hardcoded value).
        stage_results = data.get("stage_results", [])
        assert stage_results, "non-empty stage_results required"
        completed_with_effect = [
            s
            for s in stage_results
            if s.get("status") == "completed" and isinstance(s.get("effect_estimate"), (int, float))
        ]
        assert completed_with_effect, (
            f"At least one stage must produce a real effect_estimate; got: {stage_results}"
        )
        # No is_demo=true labels in non-demo mode (would indicate a silent
        # substitution that defeats the wiring).
        for stage in stage_results:
            assert not stage.get("additional_results", {}).get("is_demo"), (
                "C-8 regression: non-demo response carries is_demo=true label "
                "(silent substitution into demo path?)"
            )


class TestParallelPipelineRealExecutionWithInlineData:
    """When the caller supplies a DataFrame, the wired parallel pipeline runs."""

    def test_parallel_runs_real_pipeline_when_data_provided(self, parallel_pipeline_request):
        """Parallel pipeline runs end-to-end and returns 200 when data is provided.

        Per codex iter-1 MEDIUM: this test requires 200, not 200-or-500.
        """
        df = _make_small_estimation_dataframe()
        parallel_pipeline_request["filters"] = {
            "estimation_data_records": df.to_dict(orient="records"),
        }
        parallel_pipeline_request["libraries"] = ["dowhy", "econml"]
        response = client.post(
            "/api/causal/pipeline/parallel",
            json=parallel_pipeline_request,
        )
        assert response.status_code == 200, (
            f"With data provided, parallel pipeline must execute and return 200. "
            f"Got {response.status_code}: {response.text[:500]}"
        )
        data = response.json()
        assert "library_results" in data
        # At least one library must succeed AND surface a real effect_estimate.
        succeeded = data.get("libraries_succeeded") or []
        assert succeeded, (
            f"At least one library must succeed in parallel mode; "
            f"got succeeded={succeeded}, failed={data.get('libraries_failed')}"
        )
        # Validate that each "succeeded" entry has real data (not a labeling-only
        # response). Per codex iter-1 HIGH: a non-primary successful library
        # must NOT return just {"library": <name>} — its real payload must be
        # surfaced via state["<lib>_result"]["result"].
        library_results = data.get("library_results", {})
        for lib_name in succeeded:
            payload = library_results.get(lib_name, {})
            assert not payload.get("is_demo"), (
                f"C-8 regression: non-demo parallel response on {lib_name} "
                "carries is_demo=true (silent substitution into demo path?)"
            )
            # A real successful library must carry SOME real data beyond
            # just its name (effect_estimate, ate, auuc, qini, or n_nodes/n_edges
            # for networkx). If the payload is just {"library": lib_name} the
            # adapter dropped real data on the floor.
            payload_keys = set(payload.keys()) - {"library", "method"}
            assert payload_keys, (
                f"C-8 regression: succeeded library {lib_name} has empty payload "
                f"(only contains {set(payload.keys())}); the adapter likely read "
                "primary_result instead of state['<lib>_result']['result']."
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


class TestRobustnessValidationFlagDefaults:
    """M-reach3: pipeline responses must carry an explicit robustness-unvalidated flag.

    /causal/pipeline/{sequential,parallel} never run refutation (DoWhy executor
    hardcodes refutation_results={}), so the response models MUST expose a
    fail-safe boolean that defaults to False, plus a warning string, so a
    consumer cannot mistake an unrefuted ATE for a validated one.
    """

    def test_sequential_response_has_robustness_flag_defaulting_false(self):
        resp = SequentialPipelineResponse(
            pipeline_id="p1",
            status="completed",
            stages_completed=1,
            stages_total=1,
            total_latency_ms=10,
            created_at="2026-06-05T00:00:00Z",
        )
        # Field exists and is fail-safe by default.
        assert resp.robustness_validation_performed is False
        # Warning field exists (None by default; populated by the builder).
        assert resp.robustness_warning is None

    def test_parallel_response_has_robustness_flag_defaulting_false(self):
        resp = ParallelPipelineResponse(
            pipeline_id="p2",
            status="completed",
            consensus_method="variance_weighted",
            total_latency_ms=10,
            created_at="2026-06-05T00:00:00Z",
        )
        assert resp.robustness_validation_performed is False
        assert resp.robustness_warning is None

    def test_robustness_flag_is_serialized_in_model_dump(self):
        # The flag must survive model_dump() since the route caches responses
        # via .model_dump() (causal.py:711/718) and returns the dict shape.
        resp = ParallelPipelineResponse(
            pipeline_id="p3",
            status="completed",
            consensus_method="variance_weighted",
            total_latency_ms=10,
            created_at="2026-06-05T00:00:00Z",
        )
        dumped = resp.model_dump()
        assert dumped["robustness_validation_performed"] is False
        assert "robustness_warning" in dumped


class TestRobustnessUnvalidatedLabelingOnRealPath:
    """M-reach3: the real (non-demo) pipeline must label its ATE as unrefuted."""

    def test_sequential_real_response_is_labeled_unvalidated(self, sequential_pipeline_request):
        df = _make_small_estimation_dataframe()
        sequential_pipeline_request["filters"] = {
            "estimation_data_records": df.to_dict(orient="records"),
        }
        sequential_pipeline_request["stages"] = [
            {"library": "dowhy", "estimator": "propensity_score_matching"},
            {"library": "econml", "estimator": "linear_dml"},
        ]
        response = client.post("/api/causal/pipeline/sequential", json=sequential_pipeline_request)
        assert response.status_code == 200, response.text[:500]
        data = response.json()
        # Real ATE present but explicitly flagged as NOT robustness-validated.
        assert data["robustness_validation_performed"] is False
        assert data["robustness_warning"], "unvalidated response must carry a warning"
        assert "refut" in data["robustness_warning"].lower()
        # The caveat is also surfaced in the warnings list consumers already read.
        assert any("refut" in w.lower() for w in data.get("warnings", []))

    def test_parallel_real_response_is_labeled_unvalidated(self, parallel_pipeline_request):
        df = _make_small_estimation_dataframe()
        parallel_pipeline_request["filters"] = {
            "estimation_data_records": df.to_dict(orient="records"),
        }
        parallel_pipeline_request["libraries"] = ["dowhy", "econml"]
        response = client.post("/api/causal/pipeline/parallel", json=parallel_pipeline_request)
        assert response.status_code == 200, response.text[:500]
        data = response.json()
        assert data["robustness_validation_performed"] is False
        assert data["robustness_warning"], "unvalidated response must carry a warning"
        assert "refut" in data["robustness_warning"].lower()
        assert any("refut" in w.lower() for w in data.get("warnings", []))
