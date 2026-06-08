"""Red-first tests for CausalMLExecutor real-library wiring (phase C-4 of GH #354).

These assertions are authored "red-first" per the dispatch plan's TDD protocol —
they FAIL against the current placeholder body in
`src/causal_engine/pipeline/executors/causalml.py` (which returns
`auuc=0.0, qini=0.0, confidence=0.78, model="UpliftRandomForest"` regardless of
input), and they go GREEN once the executor body is rewired to call the real
uplift module (`UpliftRandomForest` / `UpliftTree` / `UpliftGradientBoosting`
from `src.causal_engine.uplift`, which themselves wrap
`causalml.inference.tree.UpliftRandomForestClassifier`,
`causalml.inference.tree.UpliftTreeClassifier`, and
`causalml.inference.meta.{BaseT,BaseX,BaseS}Classifier`).

Cross-refs:
- Dispatch plan: .claude/plans/354_dispatch_plan_v1.md §0 (V-05, V-20, V-23), §2.2 (C-4 brief)
- Design plan: .claude/plans/causal_engine_canonical_routing_v4.md §1.3 (CausalML maturity), §5.1 C-4
- Wrap point (V-05): `src/causal_engine/uplift/random_forest.py:54,182`;
  `src/causal_engine/uplift/gradient_boosting.py:163,177,191`
- Production-wiring reference: `src/agents/heterogeneous_optimizer/nodes/uplift_analyzer.py:358-383`
  shows `UpliftRandomForest(config).estimate(X, treatment, y)` usage.

Forbidden patterns this test file pins against (Wave-3 pattern #3 / R2):
- `np.random.seed`, `random.uniform`, hardcoded synthetic data feed
- All-default/all-zero `LibraryExecutionResult.result` on data unavailability
  (must raise/return success=False instead — Wave-3 pattern #4)
- Hardcoded `auuc=0.0, qini=0.0, confidence=0.78` (the C-1 stub behavior)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.pipeline.executors.causalml import (
    CausalMLExecutor,
    _resolve_control_name,
)
from src.causal_engine.pipeline.router import CausalLibrary
from src.causal_engine.pipeline.state import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
)

# =============================================================================
# Fixtures
# =============================================================================


def _make_pipeline_state(
    *,
    treatment_var: Optional[str] = "marketing_spend",
    outcome_var: Optional[str] = "sales",
    filters: Optional[Dict[str, Any]] = None,
    confounders: Optional[list] = None,
) -> PipelineState:
    """Build a minimal PipelineState for executor tests.

    `filters` is the documented escape-hatch a caller can use to inject a
    real DataFrame (`filters["dataframe"]`) until C-6 lands a proper data
    backend hook on PipelineState. Without `filters["dataframe"]`, the
    executor must fail-closed (no synthetic-data fallback).
    """
    config: PipelineConfig = {
        "mode": "sequential",
        "libraries_enabled": ["causalml"],
        "primary_library": "causalml",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": True,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }

    return PipelineState(
        query="Which segments respond most to marketing?",
        question_type="targeting_optimization",
        treatment_var=treatment_var,
        outcome_var=outcome_var,
        confounders=confounders or ["region", "age_group"],
        effect_modifiers=None,
        data_source="test_data",
        filters=filters,
        config=config,
        routed_libraries=["causalml"],
        routing_confidence=0.9,
        routing_rationale="Test routing",
        networkx_result=None,
        causal_graph=None,
        graph_metrics=None,
        dowhy_result=None,
        causal_effect=None,
        refutation_results=None,
        identification_method=None,
        econml_result=None,
        cate_by_segment=None,
        overall_ate=None,
        heterogeneity_score=None,
        causalml_result=None,
        uplift_scores=None,
        auuc=None,
        qini=None,
        targeting_recommendations=None,
        consensus_effect=None,
        consensus_confidence=None,
        library_agreement=None,
        nested_cate=None,
        segment_confidence_intervals=None,
        executive_summary=None,
        key_insights=None,
        recommended_actions=None,
        current_stage=PipelineStage.PENDING,
        stage_latencies={},
        total_latency_ms=0,
        libraries_executed=[],
        libraries_skipped=[],
        errors=[],
        warnings=[],
        status="pending",
    )


def _make_pipeline_config() -> PipelineConfig:
    return {
        "mode": "sequential",
        "libraries_enabled": ["causalml"],
        "primary_library": "causalml",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": True,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }


def _make_real_uplift_data(
    n: int = 600,
    seed: int = 7,
) -> Dict[str, Any]:
    """Build a real, deterministic uplift dataset for the network-gated test.

    This data is the TEST FIXTURE'S input to `state["filters"]["dataframe"]`
    — it is NOT synthesized inside the executor. The executor MUST receive
    real DataFrame data from the caller. The seed here makes the test
    deterministic; using a seed in the test fixture (not the production code)
    is the standard pattern, and the C-1 design-pushback paragraph protects
    against the FORBIDDEN inversion (seed inside the executor body).

    Treatment effect is heterogeneous: positive in `region=high_income`,
    zero in `region=low_income`. The real CausalML model should recover a
    positive ATE and non-trivial ATT/ATC.

    Columns: marketing_spend (treatment), sales (outcome), age, income,
    region (str), age_group (str). The string columns match the default
    confounders in `_make_pipeline_state` so the executor's column-validation
    gate (codex iter-1 HIGH-2) does not trip on unrelated tests.
    """
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size=n)
    age = rng.normal(50.0, 10.0, size=n)
    income = rng.normal(60000.0, 15000.0, size=n)
    # Binary outcome with treatment effect ~0.2 amongst treated above-median income.
    base_p = 0.3
    above_median = (income > np.median(income)).astype(float)
    treat_effect = 0.2 * treatment * above_median
    noise = rng.normal(0.0, 0.05, size=n)
    p_y = np.clip(base_p + treat_effect + noise, 0.01, 0.99)
    y = (rng.random(size=n) < p_y).astype(int)
    # Numeric encodings for string-like confounders so the CausalML tree
    # ensemble doesn't reject non-numeric features. The categorical
    # interpretation is preserved by the column names.
    region_idx = (above_median > 0).astype(int)  # 0/1
    age_group_idx = (age > 50).astype(int)  # 0/1
    df = pd.DataFrame(
        {
            "marketing_spend": treatment,
            "sales": y,
            "age": age,
            "income": income,
            "region": region_idx,
            "age_group": age_group_idx,
        }
    )
    return {"dataframe": df}


# =============================================================================
# Helper: control_name resolution (codex iter-1 closes HIGH-1)
# =============================================================================


class TestResolveControlName:
    """`_resolve_control_name` picks the right control label for `UpliftConfig`.

    Closes codex iter-0 HIGH-1: `UpliftConfig.control_name` defaults to
    `"control"`, which breaks against binary 0/1 treatments (the common case).
    The helper must pick the lexicographically smallest stringified unique
    treatment value so CausalML's `UpliftRandomForestClassifier` can find the
    control group.
    """

    def test_binary_numeric_treatment_picks_zero(self):
        treatment = np.array([0, 1, 0, 1, 1, 0])
        assert _resolve_control_name(treatment) == "0"

    def test_explicit_control_treat_labels_picks_control(self):
        treatment = np.array(["treat", "control", "treat", "control"])
        # "control" < "treat" lexicographically.
        assert _resolve_control_name(treatment) == "control"

    def test_multi_arm_picks_first_lexicographic(self):
        treatment = np.array(["B", "A", "C", "A", "B"])
        assert _resolve_control_name(treatment) == "A"

    def test_single_unique_value_does_not_crash(self):
        treatment = np.array([1, 1, 1])
        # No control group present, but helper must still return a string.
        assert _resolve_control_name(treatment) == "1"


# =============================================================================
# Contract preservation (locked in C-1)
# =============================================================================


class TestCausalMLExecutorContractPreserved:
    """ABC contract + locked-in-C-1 invariants are NOT relaxed by C-4 wiring."""

    def test_library_property_returns_causalml_enum(self):
        executor = CausalMLExecutor()
        assert executor.library == CausalLibrary.CAUSALML

    def test_inherits_from_library_executor_abc(self):
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert issubclass(CausalMLExecutor, LibraryExecutor)

    def test_validate_input_passes_with_treatment_and_outcome(self):
        executor = CausalMLExecutor()
        state = _make_pipeline_state()
        is_valid, error = executor.validate_input(state)
        assert is_valid is True
        assert error == ""

    def test_validate_input_fails_without_treatment_var(self):
        executor = CausalMLExecutor()
        state = _make_pipeline_state(treatment_var=None)
        is_valid, error = executor.validate_input(state)
        assert is_valid is False
        assert "CausalML requires treatment_var" in error

    def test_validate_input_fails_without_outcome_var(self):
        executor = CausalMLExecutor()
        state = _make_pipeline_state(outcome_var=None)
        is_valid, error = executor.validate_input(state)
        assert is_valid is False
        assert "CausalML requires outcome_var" in error


# =============================================================================
# Fail-closed semantics — no synthetic-data fallback (R2, R9)
# =============================================================================


class TestCausalMLExecutorFailsClosedWhenDataUnavailable:
    """Without a real DataFrame from the caller, executor MUST fail-closed.

    These tests pin the FORBIDDEN-pattern guard: the executor must NEVER fall
    back to seeded synthetic data, hardcoded constants, or silent substitution
    when `state["filters"]["dataframe"]` is missing or unusable.
    """

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_filters_is_none(self):
        executor = CausalMLExecutor()
        state = _make_pipeline_state(filters=None)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["library"] == "causalml"
        assert result["success"] is False
        assert result["error"] is not None
        assert "data" in result["error"].lower()
        assert result["result"] is None
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_filters_missing_dataframe_key(self):
        executor = CausalMLExecutor()
        state = _make_pipeline_state(filters={"some_other_key": "x"})
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_dataframe_missing_treatment_column(self):
        executor = CausalMLExecutor()
        bad_df = pd.DataFrame({"sales": [0, 1, 0, 1], "age": [25.0, 35.0, 45.0, 55.0]})
        state = _make_pipeline_state(filters={"dataframe": bad_df})
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        assert "marketing_spend" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_dataframe_missing_outcome_column(self):
        executor = CausalMLExecutor()
        bad_df = pd.DataFrame({"marketing_spend": [0, 1, 0, 1], "age": [25.0, 35.0, 45.0, 55.0]})
        state = _make_pipeline_state(filters={"dataframe": bad_df})
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        assert "sales" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_dataframe_is_empty(self):
        executor = CausalMLExecutor()
        empty_df = pd.DataFrame({"marketing_spend": [], "sales": [], "age": [], "income": []})
        state = _make_pipeline_state(filters={"dataframe": empty_df})
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_execute_fails_closed_does_not_return_placeholder_zeros(self):
        """When data unavailable, result MUST be None (not {'auuc': 0.0, 'qini': 0.0, ...}).

        Pins against the C-1 stub behavior that silently returned zeros with
        success=True. C-4 must replace that with explicit fail-closed.
        """
        executor = CausalMLExecutor()
        state = _make_pipeline_state(filters=None)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        # Crucial: result is None, NOT a dict with zero placeholders.
        assert result["result"] is None
        # Confidence is 0.0 on failure (not the old 0.78 placeholder).
        assert result["confidence"] != 0.78

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_declared_confounder_column_missing(self):
        """Closes codex iter-0 HIGH-2: when the caller declares confounders
        that ARE NOT present in `filters["dataframe"]`, executor must raise
        ExecutorDataUnavailable rather than silently fitting on the remaining
        columns (which would be "all-default on missing input" — Wave-3
        pattern #4).
        """
        executor = CausalMLExecutor()
        df = pd.DataFrame(
            {
                "marketing_spend": [0, 1, 0, 1, 0, 1] * 20,
                "sales": [0, 1, 0, 1, 0, 1] * 20,
                "age": [25.0, 35.0, 45.0, 55.0, 30.0, 40.0] * 20,
                # Note: `income`, `nps_score` are listed in confounders below
                # but NOT present in this DataFrame.
            }
        )
        state = _make_pipeline_state(
            filters={"dataframe": df},
            confounders=["age", "income", "nps_score"],
        )
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        # Error message must name the missing declared columns to make
        # diagnosis trivial.
        assert "income" in result["error"]
        assert "nps_score" in result["error"]
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_execute_fails_closed_when_declared_effect_modifier_missing(self):
        """Same fail-closed for effect_modifiers (codex iter-0 HIGH-2)."""
        executor = CausalMLExecutor()
        df = pd.DataFrame(
            {
                "marketing_spend": [0, 1, 0, 1] * 25,
                "sales": [0, 1, 0, 1] * 25,
                "age": [25.0, 35.0, 45.0, 55.0] * 25,
            }
        )
        state = _make_pipeline_state(
            filters={"dataframe": df},
            confounders=["age"],
        )
        # Add an effect_modifier that's missing from the DataFrame.
        state["effect_modifiers"] = ["channel"]
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is False
        assert result["error"] is not None
        assert "channel" in result["error"]
        assert result["result"] is None


# =============================================================================
# Source-code anti-mocking guards (R2)
# =============================================================================


class TestCausalMLExecutorSourceCodeIsFabricationFree:
    """grep the executor source to catch synthetic-data smells before they ship.

    Detects the FORBIDDEN patterns (Wave-3 pattern #3) at the source-code level
    so a future regression that re-introduces synthetic data, `np.random.seed`,
    or hardcoded plausible-wrong constants will fail this test rather than
    silently passing a behavior assertion that happens to be loose enough.
    """

    def _executor_source(self) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[4]
            / "src"
            / "causal_engine"
            / "pipeline"
            / "executors"
            / "causalml.py"
        ).read_text()

    def test_no_random_uniform_call(self):
        """No call-site invocation of `random.uniform(...)` (synthetic-data fabrication).

        Checks for the literal call pattern `random.uniform(` — docstring
        references that mention the forbidden pattern by name (to explain
        the contract) are explicitly allowed.
        """
        src = self._executor_source()
        assert "random.uniform(" not in src, (
            "FORBIDDEN: random.uniform() call in executor body (synthetic-data fabrication)"
        )

    def test_no_np_random_seed_call(self):
        """No call-site invocation of `np.random.seed(...)` (silent-fabrication trap)."""
        src = self._executor_source()
        assert "np.random.seed(" not in src, (
            "FORBIDDEN: np.random.seed() call in executor body (silent-fabrication trap)"
        )

    def test_no_np_random_default_rng_call(self):
        """No call-site invocation of `np.random.default_rng(...)` (synthetic-data fallback)."""
        src = self._executor_source()
        assert "np.random.default_rng(" not in src, (
            "FORBIDDEN: np.random.default_rng() call in executor body (synthetic-data fallback)"
        )

    def test_no_placeholder_marker_comment(self):
        src = self._executor_source()
        assert "Placeholder implementation - actual" not in src, (
            "C-1 placeholder marker must be removed after C-4 rewire"
        )

    def test_no_hardcoded_confidence_0_78(self):
        """The stub returned confidence=0.78 on every success. Real-wired must compute it."""
        src = self._executor_source()
        # Allow `0.78` to appear in comments/docstrings but not as a `confidence=0.78` literal.
        assert "confidence=0.78" not in src, (
            "FORBIDDEN: hardcoded confidence=0.78 (C-1 stub behavior)"
        )

    def test_executor_imports_real_uplift_module(self):
        """The executor MUST import the production uplift wrappers as Python imports.

        Closes codex iter-0 MEDIUM: the prior loose check (any string match
        on `UpliftRandomForest`/`UpliftGradientBoosting`/`UpliftTree`)
        would have been satisfied by a docstring alone. We use Python's
        AST to parse the executor source and confirm there is at least
        one real ``from src.causal_engine.uplift import ...`` (top-level
        or nested inside a function body for lazy import) that imports
        one of the production wrappers. This guarantees the executor
        actually CALLS the real uplift module rather than just naming
        it in a docstring.
        """
        import ast

        src = self._executor_source()
        tree = ast.parse(src)

        production_wrappers = {
            "UpliftRandomForest",
            "UpliftTree",
            "UpliftGradientBoosting",
        }
        found_real_import = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            # Allow both `src.causal_engine.uplift` and the relative form
            # (`..uplift` from within `pipeline.executors`); the latter is
            # not how the production-wired uplift_analyzer reaches it, but
            # accepting it keeps the rule tied to behavior, not pathing.
            if module not in {
                "src.causal_engine.uplift",
                "causal_engine.uplift",
            } and not module.endswith(".uplift"):
                continue
            imported_names = {alias.name for alias in node.names}
            if imported_names & production_wrappers:
                found_real_import = True
                break

        assert found_real_import, (
            "Executor must contain a real Python import of at least one of "
            f"{sorted(production_wrappers)} from src.causal_engine.uplift "
            "(docstring references alone are insufficient)."
        )


# =============================================================================
# Real-library success path — wrapped via uplift module
# =============================================================================


class TestCausalMLExecutorRealLibraryWiring:
    """Executor invokes the real uplift module and produces non-placeholder results.

    Marked `slow` because CausalML model fitting can take seconds-to-tens-of-seconds
    on the synthetic-but-real dataset used here. We feed REAL data (from the test
    fixture, not from inside the executor) through the production wrapper.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_execute_resolves_control_name_from_binary_treatment(self):
        """Closes codex iter-0 HIGH-1 + iter-1 MEDIUM:

        - With binary 0/1 treatments the resolved `control_name` must be `"0"`
          (the lexicographically smallest stringified unique value). UpliftConfig
          default `"control"` would NOT match and the real-library success path
          would fail closed.
        - The `treatment_groups` field in the result payload carries the real
          `UpliftResult.treatment_groups` (the non-control arms, excluding the
          configured control_name). For binary 0/1 with control "0" this is
          `["1"]`. The pre-fit raw observed labels live in
          `observed_treatment_groups`.
        """
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=300, seed=31)
        state = _make_pipeline_state(filters=filters)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is True, (
            f"Expected success after control_name resolution; got error: {result.get('error')}"
        )
        assert result["result"]["control_name"] == "0"
        # Non-control arms from real UpliftResult; control "0" is EXCLUDED.
        assert result["result"]["treatment_groups"] == ["1"]
        # Raw observed labels (incl. control) remain available for callers.
        assert "0" in result["result"]["observed_treatment_groups"]
        assert "1" in result["result"]["observed_treatment_groups"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_execute_returns_real_uplift_result_when_data_available(self):
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=400, seed=11)
        state = _make_pipeline_state(filters=filters)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["library"] == "causalml"
        assert result["success"] is True, f"Expected success, got error: {result.get('error')}"
        assert result["error"] is None
        assert result["latency_ms"] >= 0
        # Result MUST be populated with real outputs — not None, not all-zero.
        assert result["result"] is not None
        # The shape required for `_update_state_with_result` to propagate state:
        assert "auuc" in result["result"]
        assert "qini" in result["result"]
        # NEW real-wired fields: ATE / ATT / ATC from UpliftResult
        assert "ate" in result["result"]
        # M-stat4: the honesty marker must survive the executor flattening so a
        # downstream consumer of the flat payload cannot mistake the mean
        # model-predicted uplift for an identification-validated ATE/ATT/ATC.
        assert (
            result["result"]["data_provenance"]
            == "model_predicted_uplift_not_identification_validated"
        )
        # Per-sample uplift scores (or aggregate summary) — must be present, NOT empty
        assert "uplift_scores_summary" in result["result"]
        # Real-wired confidence is computed from sample size / agreement, not the
        # C-1 hardcoded 0.78. Allow any finite [0, 1] value EXCEPT the stub.
        assert 0.0 < result["confidence"] <= 1.0
        assert result["confidence"] != 0.78

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_execute_records_real_uplift_metrics_finite(self):
        """auuc/qini from real CausalML estimator are finite numbers (not the 0.0 stub)."""
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=400, seed=13)
        state = _make_pipeline_state(filters=filters)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is True
        # auuc / qini can be None if the metrics helper raises, but if present
        # they MUST be finite floats — never the all-zero placeholder.
        if result["result"]["auuc"] is not None:
            assert isinstance(result["result"]["auuc"], float)
            assert np.isfinite(result["result"]["auuc"])
        if result["result"]["qini"] is not None:
            assert isinstance(result["result"]["qini"], float)
            assert np.isfinite(result["result"]["qini"])

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_execute_records_real_ate_finite(self):
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=400, seed=17)
        state = _make_pipeline_state(filters=filters)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is True
        ate = result["result"].get("ate")
        assert ate is not None, "Real CausalML execution must produce an ATE"
        assert isinstance(ate, float)
        assert np.isfinite(ate)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_execute_records_model_type_from_uplift_result(self):
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=300, seed=19)
        state = _make_pipeline_state(filters=filters)
        config = _make_pipeline_config()

        result = await executor.execute(state, config)

        assert result["success"] is True
        # `model` field carries the actual UpliftModelType used (e.g.
        # 'uplift_random_forest') rather than the hardcoded 'UpliftRandomForest'
        # stub string.
        assert "model" in result["result"]
        assert result["result"]["model"] in {
            "uplift_random_forest",
            "uplift_tree",
            "uplift_gradient_boosting",
            "causal_tree",
        }

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_execute_emits_warning_when_metrics_fail_but_keeps_success(self):
        """If auuc/qini calculation fails after a successful fit, mark fields as
        unavailable and add a warning — but DO NOT silent-substitute."""
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=200, seed=23)
        state = _make_pipeline_state(filters=filters)
        config = _make_pipeline_config()

        # Force the metrics helper to raise; the executor should keep success
        # (the FIT itself succeeded) but flag auuc/qini as unavailable + warn.
        with patch(
            "src.causal_engine.pipeline.executors.causalml._compute_uplift_metrics_safe",
            side_effect=RuntimeError("metrics failed"),
        ):
            result = await executor.execute(state, config)

        assert result["success"] is True
        # Either the fields are None (unavailable) or marked with `_available=False`.
        # Either way, they MUST NOT be silently substituted with a different signal.
        assert result["result"].get("auuc") is None
        assert result["result"].get("qini") is None
        # Warning indicating metrics unavailable is propagated up.
        assert any("metric" in w.lower() for w in result["warnings"])


# =============================================================================
# Exception handling (R1 — preserves try/except shape, but no silent fallback)
# =============================================================================


class TestCausalMLExecutorExceptionHandling:
    """When a downstream library call raises, executor returns success=False
    with error captured — no placeholder result, no synthetic fallback."""

    @pytest.mark.asyncio
    async def test_execute_handles_uplift_model_exception(self):
        executor = CausalMLExecutor()
        filters = _make_real_uplift_data(n=200, seed=29)
        # Use confounders that match the real-data fixture columns to avoid
        # tripping the HIGH-2 declared-column fail-closed gate (added in
        # iter-1). The point of this test is to exercise the uplift-model
        # exception path, not the column-validation path.
        state = _make_pipeline_state(filters=filters, confounders=["age", "income"])
        config = _make_pipeline_config()

        # Patch the uplift wrapper to raise; verify executor returns success=False
        # without falling back to a plausible-wrong placeholder.
        with patch(
            "src.causal_engine.pipeline.executors.causalml._fit_uplift_model",
            side_effect=RuntimeError("uplift fit failed"),
        ):
            result = await executor.execute(state, config)

        assert result["success"] is False
        assert "uplift fit failed" in (result["error"] or "")
        assert result["result"] is None
        assert result["confidence"] == 0.0
