"""Red-first import-path tests for the C-1 executors package split.

This test file is part of phase C-1 SETUP of GH #354 (NetworkX → DoWhy → EconML →
CausalML canonical-routing). It asserts the per-executor file layout the
refactor produces and pins the public-surface guarantee that Wave-1 (C-2..C-5)
parallel dispatchers depend on.

Authored "red-first" per dispatch plan TDD protocol — assertions FAIL on this
commit (no executors/ package yet); they go GREEN once
`src/causal_engine/pipeline/orchestrator.py` is split into
`src/causal_engine/pipeline/executors/{base,networkx,dowhy,econml,causalml}.py`.

Cross-refs:
- Dispatch plan: .claude/plans/354_dispatch_plan_v1.md §2.1 (C-1 SETUP scope)
- Design plan: .claude/plans/causal_engine_canonical_routing_v4.md §1-§5
"""

from abc import ABC

import pytest

from src.causal_engine.pipeline.router import CausalLibrary


class TestExecutorsPackageImports:
    """Per-executor module import-path assertions (red-first)."""

    def test_base_module_exports_library_executor_abc(self):
        """`from src.causal_engine.pipeline.executors.base import LibraryExecutor` succeeds."""
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert LibraryExecutor is not None

    def test_networkx_module_exports_networkx_executor(self):
        """`from src.causal_engine.pipeline.executors.networkx import NetworkXExecutor` succeeds."""
        from src.causal_engine.pipeline.executors.networkx import NetworkXExecutor

        assert NetworkXExecutor is not None

    def test_dowhy_module_exports_dowhy_executor(self):
        """`from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor` succeeds."""
        from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor

        assert DoWhyExecutor is not None

    def test_econml_module_exports_econml_executor(self):
        """`from src.causal_engine.pipeline.executors.econml import EconMLExecutor` succeeds."""
        from src.causal_engine.pipeline.executors.econml import EconMLExecutor

        assert EconMLExecutor is not None

    def test_causalml_module_exports_causalml_executor(self):
        """`from src.causal_engine.pipeline.executors.causalml import CausalMLExecutor` succeeds."""
        from src.causal_engine.pipeline.executors.causalml import CausalMLExecutor

        assert CausalMLExecutor is not None

    def test_executors_package_reexports_all_executors(self):
        """`from src.causal_engine.pipeline.executors import <Executor>` succeeds for all 5 names."""
        from src.causal_engine.pipeline.executors import (
            CausalMLExecutor,
            DoWhyExecutor,
            EconMLExecutor,
            LibraryExecutor,
            NetworkXExecutor,
        )

        assert LibraryExecutor is not None
        assert NetworkXExecutor is not None
        assert DoWhyExecutor is not None
        assert EconMLExecutor is not None
        assert CausalMLExecutor is not None


class TestPipelinePackagePublicSurfacePreserved:
    """`pipeline/__init__.py` public surface guarantee for external callers."""

    def test_pipeline_package_reexports_executors(self):
        """External callers using `from src.causal_engine.pipeline import <Executor>` still work."""
        from src.causal_engine.pipeline import (
            CausalMLExecutor,
            DoWhyExecutor,
            EconMLExecutor,
            LibraryExecutor,
            NetworkXExecutor,
        )

        assert LibraryExecutor is not None
        assert NetworkXExecutor is not None
        assert DoWhyExecutor is not None
        assert EconMLExecutor is not None
        assert CausalMLExecutor is not None

    def test_pipeline_package_all_includes_executors(self):
        """`__all__` still lists the 5 executor names so `from X import *` keeps working."""
        from src.causal_engine import pipeline

        assert "LibraryExecutor" in pipeline.__all__
        assert "NetworkXExecutor" in pipeline.__all__
        assert "DoWhyExecutor" in pipeline.__all__
        assert "EconMLExecutor" in pipeline.__all__
        assert "CausalMLExecutor" in pipeline.__all__

    def test_pipeline_orchestrator_reexports_executors_for_backward_compat(self):
        """Existing tests + callers using `pipeline.orchestrator.<Executor>` still resolve.

        Backward-compat guarantee: the existing test_orchestrator.py imports executors
        directly from `src.causal_engine.pipeline.orchestrator` (V-02), and external
        callers may have done the same. The refactor MUST preserve this import path
        even though the canonical location moves to `pipeline.executors.<lib>`.
        """
        from src.causal_engine.pipeline.orchestrator import (
            CausalMLExecutor,
            DoWhyExecutor,
            EconMLExecutor,
            LibraryExecutor,
            NetworkXExecutor,
        )

        assert LibraryExecutor is not None
        assert NetworkXExecutor is not None
        assert DoWhyExecutor is not None
        assert EconMLExecutor is not None
        assert CausalMLExecutor is not None

    def test_pipeline_package_object_identity_across_paths(self):
        """The same class object is reachable via canonical (executors/) and legacy (orchestrator)
        paths.

        Object identity guarantees that `isinstance(x, NetworkXExecutor)` works regardless of
        which import path the caller used. Prevents subtle bugs where two import paths produce
        two different class objects (Python's import caches make this rare but possible if
        rebound).
        """
        from src.causal_engine.pipeline import NetworkXExecutor as PipelineNetworkX
        from src.causal_engine.pipeline.executors.networkx import (
            NetworkXExecutor as ExecutorsNetworkX,
        )
        from src.causal_engine.pipeline.orchestrator import (
            NetworkXExecutor as OrchestratorNetworkX,
        )

        assert PipelineNetworkX is ExecutorsNetworkX
        assert PipelineNetworkX is OrchestratorNetworkX


class TestLibraryExecutorABCContract:
    """LibraryExecutor ABC contract preservation (no method add/remove/rename)."""

    def test_library_executor_is_abc(self):
        """`LibraryExecutor` IS an abstract class (cannot be instantiated directly)."""
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert issubclass(LibraryExecutor, ABC)
        with pytest.raises(TypeError):
            LibraryExecutor()  # type: ignore[abstract]

    def test_library_executor_has_library_abstract_property(self):
        """`LibraryExecutor.library` is an abstractmethod (property)."""
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert hasattr(LibraryExecutor, "library")
        # `library` is decorated as @property + @abstractmethod
        library_attr = LibraryExecutor.__dict__.get("library")
        assert library_attr is not None
        # When decorated as @property @abstractmethod, the descriptor exposes
        # __isabstractmethod__ = True
        assert getattr(library_attr, "__isabstractmethod__", False) is True

    def test_library_executor_has_execute_abstract_method(self):
        """`LibraryExecutor.execute` is an abstract async method."""
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert hasattr(LibraryExecutor, "execute")
        execute_attr = LibraryExecutor.__dict__.get("execute")
        assert execute_attr is not None
        assert getattr(execute_attr, "__isabstractmethod__", False) is True

    def test_library_executor_has_validate_input_abstract_method(self):
        """`LibraryExecutor.validate_input` is an abstract method."""
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert hasattr(LibraryExecutor, "validate_input")
        validate_attr = LibraryExecutor.__dict__.get("validate_input")
        assert validate_attr is not None
        assert getattr(validate_attr, "__isabstractmethod__", False) is True

    def test_library_executor_abstract_method_set_is_exhaustive(self):
        """The set of abstract methods is exactly {library, execute, validate_input}.

        This pins the ABC contract for Wave-1 (C-2..C-5) dispatchers. If a future
        change adds or removes an abstract method, this test fails and forces an
        explicit decision (likely a fresh dispatch-plan iteration).
        """
        from src.causal_engine.pipeline.executors.base import LibraryExecutor

        assert LibraryExecutor.__abstractmethods__ == frozenset(
            {"library", "execute", "validate_input"}
        )


class TestConcreteExecutorsInheritAndReportCorrectLibrary:
    """Each concrete executor inherits from `LibraryExecutor` and reports correct enum value."""

    def test_networkx_executor_inherits_from_library_executor(self):
        from src.causal_engine.pipeline.executors.base import LibraryExecutor
        from src.causal_engine.pipeline.executors.networkx import NetworkXExecutor

        assert issubclass(NetworkXExecutor, LibraryExecutor)

    def test_networkx_executor_reports_networkx_enum(self):
        from src.causal_engine.pipeline.executors.networkx import NetworkXExecutor

        executor = NetworkXExecutor()
        assert executor.library == CausalLibrary.NETWORKX

    def test_dowhy_executor_inherits_from_library_executor(self):
        from src.causal_engine.pipeline.executors.base import LibraryExecutor
        from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor

        assert issubclass(DoWhyExecutor, LibraryExecutor)

    def test_dowhy_executor_reports_dowhy_enum(self):
        from src.causal_engine.pipeline.executors.dowhy import DoWhyExecutor

        executor = DoWhyExecutor()
        assert executor.library == CausalLibrary.DOWHY

    def test_econml_executor_inherits_from_library_executor(self):
        from src.causal_engine.pipeline.executors.base import LibraryExecutor
        from src.causal_engine.pipeline.executors.econml import EconMLExecutor

        assert issubclass(EconMLExecutor, LibraryExecutor)

    def test_econml_executor_reports_econml_enum(self):
        from src.causal_engine.pipeline.executors.econml import EconMLExecutor

        executor = EconMLExecutor()
        assert executor.library == CausalLibrary.ECONML

    def test_causalml_executor_inherits_from_library_executor(self):
        from src.causal_engine.pipeline.executors.base import LibraryExecutor
        from src.causal_engine.pipeline.executors.causalml import CausalMLExecutor

        assert issubclass(CausalMLExecutor, LibraryExecutor)

    def test_causalml_executor_reports_causalml_enum(self):
        from src.causal_engine.pipeline.executors.causalml import CausalMLExecutor

        executor = CausalMLExecutor()
        assert executor.library == CausalLibrary.CAUSALML
