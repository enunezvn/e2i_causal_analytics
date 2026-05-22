"""Red-first real-pipeline assertions for `causal_effect_estimator` (phase C-7 of GH #354).

This file is RED-FIRST per the TDD protocol in `.claude/dispatch/354_executor_brief_template.md`.

On the placeholder body (hardcoded `EffectEstimate(ate=0.12, ci_lower=0.08, ci_upper=0.16,
p_value=0.001, method=method, n_samples=10000)` at `tool_registrations.py:385`), these
assertions FAIL — they assert the tool:

1. CALLS `SequentialPipeline.execute()` (the real multi-library aggregation pipeline
   wired through C-1..C-6).
2. RETURNS PIPELINE-DERIVED outputs (`ate` from `PipelineOutput.consensus_effect`,
   confidence-derived `p_value` and `ci_*`) — NOT the hardcoded `0.12/0.08/0.16/0.001`
   placeholder constants.
3. FAILS CLOSED when the caller does NOT supply a DataFrame (raises a descriptive
   `RuntimeError`; never falls back to `ate=0.12` or any default).
4. FAILS CLOSED when the pipeline raises `ExecutorDataUnavailable` from a downstream
   executor (propagates as `RuntimeError`; no silent default).
5. FAILS CLOSED when the pipeline returns `status='failed'` (raises; never silently
   returns a placeholder `EffectEstimate`).
6. FAILS CLOSED when the pipeline returns success but `consensus_effect` is `None`
   (no successful library produced a finite ATE) — marks SKIPPED, raises; never
   silently substitutes a different signal (Wave-3 anti-mocking pattern #4).
7. CONSTRUCTS `PipelineInput` from the tool's declared kwargs (treatment, outcome,
   confounders, method) plus the caller-supplied DataFrame.
8. ROUTES the DataFrame through the canonical `data_resolver` slots so all 4 Wave-1
   executors (DoWhy, EconML, CausalML, NetworkX-graph) can find it.

The forbidden patterns this file catches:
- Hardcoded `ate=0.12` (or any plausible-but-fake constant) anywhere in the new code path.
- Silent fallback to a default ATE/CI when pipeline raises.
- `np.random.seed` / `random.uniform` anywhere in the tool body.
- Synthetic data fed to the pipeline from inside the tool.
- Silent substitution when `consensus_effect=None` (must mark skipped + fail-closed).
- Keeping the old `ate=0.12` line as a commented-out "fallback for future reference."

Cross-refs:
- Dispatch plan: .claude/plans/354_dispatch_plan_v1.md §2.4 C-7
- Design plan: .claude/plans/causal_engine_canonical_routing_v4.md §1-§5
- Brief template: .claude/dispatch/354_executor_brief_template.md
- Data resolver (C-6): src/causal_engine/pipeline/data_resolver.py
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer.tool_registrations import (
    EffectEstimate,
    causal_effect_estimator,
)

# ============================================================================
# Helpers
# ============================================================================


def _build_real_dataframe(*, n: int = 400, true_ate: float = 1.5, seed: int = 13) -> pd.DataFrame:
    """Build a DataFrame with a known causal effect for asserting pipeline ATE recovery.

    FIXTURE assembled inside the TEST (not the tool body) — the anti-pattern is
    synthetic data fed to the real pipeline from INSIDE the tool. Tests are
    allowed (and expected) to construct DataFrames the tool then consumes via
    kwargs.
    """
    rng = np.random.default_rng(seed)
    confounder_a = rng.normal(0.0, 1.0, n)
    treatment = 0.5 * confounder_a + rng.normal(0.0, 1.0, n)
    outcome = true_ate * treatment + 0.7 * confounder_a + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"treatment": treatment, "outcome": outcome, "confounder_a": confounder_a})


def _build_pipeline_output(
    *,
    consensus_effect: Optional[float] = 0.42,
    consensus_confidence: Optional[float] = 0.83,
    status: str = "completed",
    libraries_used: Optional[List[str]] = None,
    primary_result: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a `PipelineOutput`-shaped dict for mocking `SequentialPipeline.execute`.

    Returns a plain dict (matches the TypedDict at runtime; tests don't need to
    cast). Defaults represent a successful 3-library consensus run.
    """
    return {
        "question_type": "causal_effect",
        "primary_result": primary_result or {"ate": consensus_effect},
        "libraries_used": libraries_used or ["dowhy", "econml", "causalml"],
        "consensus_effect": consensus_effect,
        "consensus_confidence": consensus_confidence,
        "executive_summary": "Test pipeline summary",
        "key_insights": [],
        "recommended_actions": [],
        "total_latency_ms": 123,
        "status": status,
        "warnings": [],
        "errors": errors or [],
    }


# ============================================================================
# Tests
# ============================================================================


class TestCausalEffectEstimatorRealWiring:
    """C-7 real-pipeline wrap assertions for `causal_effect_estimator`."""

    def test_invocation_calls_sequential_pipeline_execute(self) -> None:
        """The tool MUST invoke `SequentialPipeline.execute(input_data)`.

        Asserts (on a mocked SequentialPipeline) that the tool reaches
        `execute()` exactly once with a `PipelineInput`-shaped argument carrying
        the treatment + outcome + confounders + data_source the caller supplied.
        """
        df = _build_real_dataframe()
        with patch(
            "src.agents.tool_composer.tool_registrations.SequentialPipeline"
        ) as mock_pipeline_cls:
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.execute = AsyncMock(
                return_value=_build_pipeline_output(consensus_effect=0.42)
            )

            result = causal_effect_estimator(
                treatment="treatment",
                outcome="outcome",
                confounders=["confounder_a"],
                method="backdoor.linear_regression",
                data=df,
            )

            # Pipeline.execute was called exactly once.
            assert mock_instance.execute.call_count == 1

            # The call carried a PipelineInput-shaped dict with the caller's
            # treatment / outcome / confounders / data_source.
            call_args = mock_instance.execute.call_args
            assert call_args is not None, "execute() must be called with arguments"
            input_data = call_args.args[0] if call_args.args else call_args.kwargs.get("input_data")
            assert isinstance(input_data, dict)
            assert input_data.get("treatment_var") == "treatment"
            assert input_data.get("outcome_var") == "outcome"
            assert input_data.get("confounders") == ["confounder_a"]
            # Post-#458: the DataFrame travels through the first-class
            # `PipelineInput.estimation_data` field. The orchestrator copies
            # it into `state["estimation_data"]` and every executor reads it
            # via `resolve_estimation_dataframe(state)`.
            assert input_data.get("estimation_data") is df, (
                "PipelineInput.estimation_data must hold the caller's "
                "DataFrame (the canonical first-class field after #458). "
                f"Got input_data keys={sorted(input_data.keys())!r}."
            )

            # The returned EffectEstimate's ATE matches the pipeline's
            # consensus_effect, NOT the hardcoded 0.12.
            assert isinstance(result, EffectEstimate)
            assert result.ate == pytest.approx(0.42)
            assert result.method == "backdoor.linear_regression"

    def test_estimation_data_first_class_field_carries_dataframe(self) -> None:
        """Post-#458: the DataFrame is conveyed via the first-class
        ``PipelineInput.estimation_data`` field, not the legacy
        ``state["data_cache"]`` subclass-injection hack.

        The orchestrator copies ``input_data["estimation_data"]`` into
        ``state["estimation_data"]`` (see ``orchestrator._create_initial_state``)
        so all four executors can resolve it via
        ``resolve_estimation_dataframe(state)`` — no per-library key drift.
        This test pins the contract end-to-end: a real orchestrator run
        on a real ``SequentialPipeline`` reaches the state mutation site
        with the caller's DataFrame at the first-class slot.
        """
        from src.causal_engine.pipeline import SequentialPipeline
        from src.causal_engine.pipeline.router import (
            CausalLibrary,
            QuestionType,
            RoutingDecision,
        )

        df = _build_real_dataframe()
        pipeline = SequentialPipeline()

        pipeline_input: Dict[str, Any] = {
            "query": "test",
            "treatment_var": "treatment",
            "outcome_var": "outcome",
            "confounders": ["confounder_a"],
            "effect_modifiers": None,
            "data_source": "test",
            "filters": None,
            "estimation_data": df,
        }
        routing = RoutingDecision(
            question_type=QuestionType.CAUSAL_RELATIONSHIP,
            primary_library=CausalLibrary.DOWHY,
            secondary_libraries=[CausalLibrary.ECONML, CausalLibrary.CAUSALML],
            recommended_mode="sequential",
            confidence=0.9,
            rationale="test",
        )

        state = pipeline._create_initial_state(pipeline_input, routing)  # type: ignore[arg-type]

        assert state.get("estimation_data") is df, (  # type: ignore[typeddict-item]
            "PipelineState['estimation_data'] must hold the caller's "
            "DataFrame after orchestrator initial-state construction "
            "(#458 contract)."
        )

    def test_returned_ate_comes_from_consensus_effect_not_hardcoded(self) -> None:
        """Returned `EffectEstimate.ate` MUST equal `PipelineOutput.consensus_effect`.

        Pins the rewiring: regardless of what number the pipeline returns, the
        tool surfaces THAT number — never the historical `0.12` constant. We
        parameterize on multiple distinct values to catch a "stub returns
        whatever the test asked for" failure mode that a single fixed value
        would miss.
        """
        df = _build_real_dataframe()
        for synthetic_pipeline_effect in [-0.5, 0.0, 0.07, 1.234, 5.0]:
            with patch(
                "src.agents.tool_composer.tool_registrations.SequentialPipeline"
            ) as mock_pipeline_cls:
                mock_instance = mock_pipeline_cls.return_value
                mock_instance.execute = AsyncMock(
                    return_value=_build_pipeline_output(
                        consensus_effect=synthetic_pipeline_effect,
                        consensus_confidence=0.8,
                    )
                )
                result = causal_effect_estimator(
                    treatment="treatment",
                    outcome="outcome",
                    confounders=["confounder_a"],
                    method="backdoor.linear_regression",
                    data=df,
                )
                assert result.ate == pytest.approx(synthetic_pipeline_effect), (
                    f"Expected ATE={synthetic_pipeline_effect}, got {result.ate}. "
                    f"This means the tool is returning a constant instead of the "
                    f"pipeline's consensus_effect."
                )
                assert result.ate != pytest.approx(0.12) or synthetic_pipeline_effect == 0.12, (
                    "ATE happens to equal the historical placeholder 0.12 in "
                    "a non-0.12 test case — likely the tool is hardcoded."
                )

    def test_ci_derived_from_pipeline_output_not_hardcoded(self) -> None:
        """CI bounds MUST be derived from the pipeline's confidence/effect, not 0.08/0.16.

        The pipeline returns `consensus_effect` and `consensus_confidence`; the
        tool derives `ci_lower`/`ci_upper` from those (any documented formula
        is acceptable as long as it varies with the inputs). The historical
        constants (0.08, 0.16) MUST NOT appear regardless of the pipeline's
        outputs.
        """
        df = _build_real_dataframe()
        with patch(
            "src.agents.tool_composer.tool_registrations.SequentialPipeline"
        ) as mock_pipeline_cls:
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.execute = AsyncMock(
                return_value=_build_pipeline_output(consensus_effect=2.5, consensus_confidence=0.95)
            )
            result = causal_effect_estimator(
                treatment="treatment",
                outcome="outcome",
                confounders=["confounder_a"],
                method="backdoor.linear_regression",
                data=df,
            )

            # CIs must NOT be the placeholder constants.
            assert result.ci_lower != pytest.approx(0.08), (
                "ci_lower=0.08 is the hardcoded placeholder; real wiring must "
                "derive CI bounds from pipeline outputs."
            )
            assert result.ci_upper != pytest.approx(0.16), (
                "ci_upper=0.16 is the hardcoded placeholder; real wiring must "
                "derive CI bounds from pipeline outputs."
            )
            # CIs must be ordered and bracket-or-touch the estimate.
            assert result.ci_lower <= result.ate <= result.ci_upper, (
                f"CIs must bracket the point estimate; got "
                f"({result.ci_lower}, {result.ate}, {result.ci_upper})"
            )
            # n_samples should reflect the input data (NOT the hardcoded 10000
            # constant for a 400-row fixture).
            assert result.n_samples != 10000, (
                "n_samples=10000 is the hardcoded placeholder; for a "
                f"{len(df)}-row DataFrame the tool must report the real count."
            )
            assert result.n_samples == len(df)


class TestCausalEffectEstimatorFailClosed:
    """C-7 fail-closed assertions — never silently return placeholder values."""

    def test_fail_closed_when_no_dataframe_supplied(self) -> None:
        """No DataFrame in kwargs ⇒ raise (never return ate=0.12)."""
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            causal_effect_estimator(
                treatment="treatment",
                outcome="outcome",
                confounders=["confounder_a"],
                method="backdoor.linear_regression",
            )
        # The error message must mention the missing data — operators must be
        # able to read the message and know what to fix.
        message = str(exc_info.value).lower()
        assert "data" in message or "dataframe" in message or "estimation" in message, (
            f"Fail-closed error message must mention the missing data; got {exc_info.value!r}"
        )

    def test_fail_closed_when_pipeline_raises_executor_data_unavailable(self) -> None:
        """Pipeline raises ⇒ tool propagates as exception (never returns ate=0.12)."""
        # Import the canonical ExecutorDataUnavailable (defined in causalml.py
        # in C-1 but conceptually a pipeline-package contract).
        from src.causal_engine.pipeline.executors.causalml import (
            ExecutorDataUnavailable,
        )

        df = _build_real_dataframe()
        with patch(
            "src.agents.tool_composer.tool_registrations.SequentialPipeline"
        ) as mock_pipeline_cls:
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.execute = AsyncMock(
                side_effect=ExecutorDataUnavailable(
                    "CausalMLExecutor: declared feature columns missing."
                )
            )
            with pytest.raises((RuntimeError, ExecutorDataUnavailable)):
                causal_effect_estimator(
                    treatment="treatment",
                    outcome="outcome",
                    confounders=["confounder_a"],
                    method="backdoor.linear_regression",
                    data=df,
                )

    def test_fail_closed_when_pipeline_status_failed(self) -> None:
        """Pipeline returns status='failed' ⇒ raise (never return placeholder)."""
        df = _build_real_dataframe()
        with patch(
            "src.agents.tool_composer.tool_registrations.SequentialPipeline"
        ) as mock_pipeline_cls:
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.execute = AsyncMock(
                return_value=_build_pipeline_output(
                    consensus_effect=None,
                    consensus_confidence=None,
                    status="failed",
                    errors=[{"library": "dowhy", "error": "boom"}],
                )
            )
            with pytest.raises(RuntimeError):
                causal_effect_estimator(
                    treatment="treatment",
                    outcome="outcome",
                    confounders=["confounder_a"],
                    method="backdoor.linear_regression",
                    data=df,
                )

    def test_fail_closed_when_consensus_effect_is_none(self) -> None:
        """Pipeline completes but no library produced a finite ATE ⇒ raise.

        Pattern #4 from Wave-3: silent-substitution forbidden. When the
        executor "succeeded" but produced no result, fail-closed; never fall
        back to a different signal that answers a different question.
        """
        df = _build_real_dataframe()
        with patch(
            "src.agents.tool_composer.tool_registrations.SequentialPipeline"
        ) as mock_pipeline_cls:
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.execute = AsyncMock(
                return_value=_build_pipeline_output(
                    consensus_effect=None,
                    consensus_confidence=None,
                    status="completed",
                )
            )
            with pytest.raises(RuntimeError) as exc_info:
                causal_effect_estimator(
                    treatment="treatment",
                    outcome="outcome",
                    confounders=["confounder_a"],
                    method="backdoor.linear_regression",
                    data=df,
                )
            message = str(exc_info.value).lower()
            assert "consensus" in message or "effect" in message or "available" in message, (
                "Fail-closed error must mention the missing consensus_effect; "
                f"got {exc_info.value!r}"
            )

    def test_fail_closed_when_consensus_effect_is_nan_or_inf(self) -> None:
        """Pipeline returns non-finite consensus_effect ⇒ raise (treat as missing)."""
        df = _build_real_dataframe()
        for bad_value in [float("nan"), float("inf"), float("-inf")]:
            with patch(
                "src.agents.tool_composer.tool_registrations.SequentialPipeline"
            ) as mock_pipeline_cls:
                mock_instance = mock_pipeline_cls.return_value
                mock_instance.execute = AsyncMock(
                    return_value=_build_pipeline_output(
                        consensus_effect=bad_value, consensus_confidence=0.9
                    )
                )
                with pytest.raises(RuntimeError):
                    causal_effect_estimator(
                        treatment="treatment",
                        outcome="outcome",
                        confounders=["confounder_a"],
                        method="backdoor.linear_regression",
                        data=df,
                    )


class TestCausalEffectEstimatorAntiMocking:
    """C-7 anti-mocking guards — source-code-level forbidden-pattern checks.

    These tests inspect the source of `causal_effect_estimator` itself to
    catch regressions where someone re-introduces the hardcoded placeholder
    or a random-data fallback. They are intentionally redundant with the
    behavioral tests above so that *deleting* the behavioral tests cannot
    re-open the door silently.
    """

    def _read_source(self) -> str:
        # The @composable_tool decorator wraps the function; `getsourcefile`
        # on the wrapper returns the decorator's file (`tool_registry/registry.py`).
        # Unwrap via `__wrapped__` (the canonical functools.wraps attribute) to
        # find the real definition file.
        target = getattr(causal_effect_estimator, "__wrapped__", causal_effect_estimator)
        path = Path(inspect.getsourcefile(target) or "")
        return path.read_text()

    def test_source_does_not_hardcode_ate_012(self) -> None:
        """Executable code in the tool's call graph MUST NOT contain the
        historical `ate=0.12` placeholder.

        Scope (codex iter-0 LOW expansion): scans `causal_effect_estimator`
        AND all of its module-level helpers reachable from the tool body
        (`_extract_dataframe_from_kwargs`, `_run_pipeline_sync`,
        `_derive_ci_and_p_value`, `_derive_p_value_from_confidence`, and the
        `_DataAwareSequentialPipeline` subclass). The original AST guard only
        scanned the public function — codex flagged that a regression could
        hide in a helper. We now check the entire call graph in the module.

        Uses AST parsing to scan ONLY executable code (excludes docstrings),
        so a docstring that legitimately documents "we replaced ate=0.12 with
        SequentialPipeline" doesn't trigger a false positive. The forbidden
        patterns surface as `keyword=Constant` AST nodes; we scan for the
        specific (keyword_arg, value) pairs.
        """
        import ast

        src = self._read_source()
        tree = ast.parse(src)

        # The tool's call graph: the entry point plus every module-level
        # helper it directly invokes. If a future refactor adds another
        # helper, add it here too (the test failure message tells you the
        # specific function that contained the offender, so the next dev
        # knows where to look). Post-#458 the helper-only subclass
        # `_DataAwareSequentialPipeline` is deleted (DataFrame travels
        # via `PipelineInput.estimation_data` directly).
        in_scope_functions = {
            "causal_effect_estimator",
            "_extract_dataframe_from_kwargs",
            "_run_pipeline_sync",
            "_derive_ci_and_p_value",
            "_derive_p_value_from_confidence",
        }
        in_scope_classes: set[str] = set()

        # Forbidden (keyword_name, value) pairs.
        forbidden_pairs = {
            ("ate", 0.12),
            ("ci_lower", 0.08),
            ("ci_upper", 0.16),
            ("p_value", 0.001),
            ("n_samples", 10000),
        }

        offenders: List[str] = []

        def _scan_function_body(func_node: ast.FunctionDef) -> None:
            body_nodes: List[ast.stmt] = list(func_node.body)
            # Drop the docstring so the function's own docstring doesn't
            # falsely match. A docstring is an Expr at body[0] holding a
            # Constant string value.
            if (
                body_nodes
                and isinstance(body_nodes[0], ast.Expr)
                and isinstance(body_nodes[0].value, ast.Constant)
                and isinstance(body_nodes[0].value.value, str)
            ):
                body_nodes = body_nodes[1:]
            for stmt in body_nodes:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.keyword) and isinstance(sub.value, ast.Constant):
                        key = sub.arg
                        val = sub.value.value
                        if (key, val) in forbidden_pairs:
                            offenders.append(f"{func_node.name}: {key}={val!r}")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in in_scope_functions:
                _scan_function_body(node)
            elif isinstance(node, ast.ClassDef) and node.name in in_scope_classes:
                # Scan every method body inside the class (covers __init__
                # plus overrides like _create_initial_state).
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        _scan_function_body(sub_node)

        assert not offenders, (
            f"causal_effect_estimator call graph contains historical "
            f"hardcoded placeholder values in executable code: {offenders}. "
            f"Per C-7 these must be derived from SequentialPipeline outputs, "
            f"not hardcoded."
        )

    def test_source_does_not_use_random_seeding(self) -> None:
        """Forbidden patterns: `np.random.seed`, `random.uniform`, etc.

        Wave-3 anti-mocking pattern #3 — synthetic data fed to a real
        estimator is a silent fabrication. The tool body must rely on
        caller-supplied data only.
        """
        src = self._read_source()
        pattern = re.compile(
            r"^def causal_effect_estimator\(.*?(?=^@composable_tool|^def\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        m = pattern.search(src)
        assert m is not None
        body = m.group(0)
        forbidden_patterns = [
            "np.random.seed",
            "np.random.default_rng",
            "random.uniform",
            "random.seed",
        ]
        offenders = [p for p in forbidden_patterns if p in body]
        assert not offenders, (
            f"causal_effect_estimator function body uses forbidden randomness "
            f"sources: {offenders}. Real data must come from caller kwargs."
        )

    def test_source_imports_sequentialpipeline(self) -> None:
        """The tool's module MUST import SequentialPipeline (the wired path)."""
        src = self._read_source()
        assert "SequentialPipeline" in src, (
            "tool_registrations.py MUST import SequentialPipeline to invoke "
            "the C-1..C-6 wired pipeline. Without this import the tool "
            "cannot have replaced the placeholder body."
        )
