"""Red-first tests for issue #458.

Acceptance criteria (from issue body):
  1. PipelineState and PipelineInput declare canonical DataFrame field(s).
  2. resolve_estimation_dataframe reads the first-class field first; legacy
     key reads retain back-compat for one release with a DeprecationWarning.
  3. All 4 executors migrate to read via the resolver (drift-guard regression).
  4. Three subclass workarounds deleted; call sites updated.
  5. Drift-guard test asserts no executor reads DataFrame via legacy keys.
  6. Fail-closed semantics preserved end-to-end.

These tests must FAIL on the pre-#458 codebase (RED) and pass once the
issue's work is implemented (GREEN).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

from src.causal_engine.pipeline.data_resolver import resolve_estimation_dataframe
from src.causal_engine.pipeline.state import PipelineInput, PipelineState

# =============================================================================
# AC #1 — PipelineState / PipelineInput declare estimation_data first-class
# =============================================================================


class TestFirstClassDataframeField:
    """estimation_data must be a declared NotRequired key on both TypedDicts."""

    def test_pipeline_state_declares_estimation_data_optional_key(self):
        # `NotRequired[...]` entries land in __optional_keys__ on Python 3.11+.
        assert "estimation_data" in PipelineState.__optional_keys__

    def test_pipeline_input_declares_estimation_data_optional_key(self):
        assert "estimation_data" in PipelineInput.__optional_keys__

    def test_pipeline_state_accepts_dataframe_at_top_level(self):
        # Authoring a state literal with the new field must type-check at
        # runtime (TypedDict is structural; we exercise it via dict literal).
        df = pd.DataFrame({"t": [0, 1], "y": [0.1, 0.2]})
        state: PipelineState = {  # type: ignore[typeddict-item]
            "estimation_data": df,  # the new first-class field
        }
        assert state["estimation_data"] is df  # type: ignore[typeddict-item]


# =============================================================================
# AC #2 — resolver prefers first-class field; legacy keys emit DeprecationWarning
# =============================================================================


def _make_state_with(**overrides) -> dict:
    """Return a minimal dict castable to PipelineState for resolver tests.

    The resolver only inspects a handful of keys; we don't fabricate the full
    TypedDict shape (the resolver duck-types via .get).
    """
    base: dict = {}
    base.update(overrides)
    return base


class TestResolverFirstClassPriority:
    """When state['estimation_data'] is set, it MUST win over legacy keys."""

    def test_first_class_field_wins_over_data_cache(self):
        first_class = pd.DataFrame({"x": [1]})
        legacy = pd.DataFrame({"x": [99]})
        state = _make_state_with(
            estimation_data=first_class,
            data_cache={"estimation_data": legacy},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        # First-class wins.
        assert result is first_class

    def test_first_class_field_wins_over_filters_estimation_data(self):
        first_class = pd.DataFrame({"x": [1]})
        legacy = pd.DataFrame({"x": [99]})
        state = _make_state_with(
            estimation_data=first_class,
            filters={"estimation_data": legacy},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        assert result is first_class

    def test_first_class_field_wins_over_filters_dataframe(self):
        first_class = pd.DataFrame({"x": [1]})
        legacy = pd.DataFrame({"x": [99]})
        state = _make_state_with(
            estimation_data=first_class,
            filters={"dataframe": legacy},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        assert result is first_class

    def test_first_class_path_emits_no_deprecation_warning(self):
        df = pd.DataFrame({"x": [1]})
        state = _make_state_with(estimation_data=df)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        assert result is df
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


class TestResolverLegacyDeprecationWarning:
    """Each legacy path must emit a DeprecationWarning on resolve."""

    def test_data_cache_legacy_path_emits_deprecation_warning(self):
        df = pd.DataFrame({"x": [1]})
        state = _make_state_with(data_cache={"estimation_data": df})
        with pytest.warns(DeprecationWarning, match="data_cache"):
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        assert result is df

    def test_filters_estimation_data_legacy_path_emits_deprecation_warning(self):
        df = pd.DataFrame({"x": [1]})
        state = _make_state_with(filters={"estimation_data": df})
        with pytest.warns(DeprecationWarning, match="filters"):
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        assert result is df

    def test_filters_dataframe_legacy_path_emits_deprecation_warning(self):
        df = pd.DataFrame({"x": [1]})
        state = _make_state_with(filters={"dataframe": df})
        with pytest.warns(DeprecationWarning, match="filters"):
            result = resolve_estimation_dataframe(state)  # type: ignore[arg-type]
        assert result is df


# =============================================================================
# AC #3 + #5 — Drift-guard: no executor reads DataFrame via legacy keys directly
# =============================================================================


_EXECUTOR_DIR = (
    Path(__file__).resolve().parents[4] / "src" / "causal_engine" / "pipeline" / "executors"
)

# Patterns that indicate a direct legacy-key read in code (NOT in docstrings or
# comments). We grep on full source then strip out lines whose first non-ws
# character starts a comment, and lines that fall inside triple-quoted blocks.
_LEGACY_PATTERNS = (
    re.compile(r"""state\[(['"])data_cache\1\]"""),
    re.compile(r"""state\[(['"])filters\1\]\[\s*(['"])estimation_data\2\s*\]"""),
    re.compile(r"""state\[(['"])filters\1\]\[\s*(['"])dataframe\2\s*\]"""),
    re.compile(r"""state\[(['"])filters\1\]\[\s*(['"])data\2\s*\]"""),
)


def _strip_docstrings_and_comments(source: str) -> str:
    """Best-effort strip of triple-quoted blocks and # comments.

    Drift-guard is a regression check, not a parser. The intent is: a future
    edit that re-introduces a direct legacy read in executable code triggers
    a failure; references inside docstrings (e.g., 'see data_cache key') do not.
    """
    # Strip triple-quoted blocks (handles both """ and ''').
    no_docstrings = re.sub(r'("""|\'\'\').*?\1', "", source, flags=re.DOTALL)
    # Strip end-of-line comments (simple heuristic; does not handle # inside strings).
    no_comments = re.sub(r"(^|\s)#[^\n]*", r"\1", no_docstrings)
    return no_comments


class TestExecutorDriftGuard:
    """Regression guard: executors must not read DataFrames via legacy keys.

    Once all 4 executors migrate to ``resolve_estimation_dataframe(state)``,
    no executable line should mention the legacy ``state[...][...]`` shapes.
    """

    @pytest.mark.parametrize(
        "executor_filename",
        ["base.py", "networkx.py", "dowhy.py", "econml.py", "causalml.py"],
    )
    def test_executor_does_not_read_legacy_keys_directly(self, executor_filename):
        path = _EXECUTOR_DIR / executor_filename
        source = path.read_text(encoding="utf-8")
        code_only = _strip_docstrings_and_comments(source)
        for pattern in _LEGACY_PATTERNS:
            match = pattern.search(code_only)
            assert match is None, (
                f"{executor_filename}: legacy DataFrame key read found in code "
                f"(pattern={pattern.pattern!r}, match={match.group(0)!r}). "
                "Executors must read via resolve_estimation_dataframe(state)."
            )


# =============================================================================
# AC #4 — Subclass workarounds for DataFrame injection removed
# =============================================================================


class TestSubclassWorkaroundsRemoved:
    """The three private subclasses authored solely for DataFrame injection
    are deleted once the first-class field exists.

    Note: ``_SurfaceCSequentialPipeline`` / ``_SurfaceCParallelPipeline``
    also currently capture ``last_state`` for per-library aggregation; that
    secondary concern is independent of #458's scope. The contract this test
    enforces is the *DataFrame-injection* mechanism. The implementation may
    keep a thinner subclass for ``last_state`` capture IF necessary; what
    MUST go is the ``dataframe=`` constructor kwarg + the
    ``_create_initial_state`` override that writes ``state["data_cache"]``.
    """

    def test_data_aware_sequential_pipeline_is_removed(self):
        # Symbol should not exist in tool_registrations after cleanup.
        import src.agents.tool_composer.tool_registrations as tr

        assert not hasattr(tr, "_DataAwareSequentialPipeline"), (
            "_DataAwareSequentialPipeline must be deleted once PipelineState "
            "declares estimation_data first-class (#458 AC #4)."
        )

    def test_surface_c_pipelines_no_longer_take_dataframe_kwarg(self):
        # If a thinner subclass remains for `last_state` capture, it MUST NOT
        # accept a `dataframe=` kwarg (that mechanism is replaced by the
        # first-class PipelineInput.estimation_data field).
        import inspect

        import src.api.routes.causal as causal_routes

        for sym in ("_SurfaceCSequentialPipeline", "_SurfaceCParallelPipeline"):
            cls = getattr(causal_routes, sym, None)
            if cls is None:
                continue  # fully deleted is also acceptable
            sig = inspect.signature(cls.__init__)
            assert "dataframe" not in sig.parameters, (
                f"{sym}: 'dataframe=' constructor kwarg must be removed; "
                "DataFrame travels through PipelineInput.estimation_data (#458 AC #4)."
            )
