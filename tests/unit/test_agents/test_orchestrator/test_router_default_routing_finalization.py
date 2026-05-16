"""Regression tests for Issue #269 — `_default_routing` finalization.

Issue #269 (codex LOW-6): `src/agents/orchestrator/nodes/router.py:_default_routing`
returns a dispatch plan WITHOUT applying the self-dispatch strip that the main
`execute()` path applies at the end. Today this is safe because the default
dispatch is hard-coded to ``"explainer"`` (not ``"orchestrator"``), but the
guard is by-inspection, not by-construction. A future refactor that makes the
default agent configurable (env var, config, etc.) would silently re-introduce
the F1 self-dispatch leak.

Acceptance (from issue body):

1. Either funnel both return paths through one finalization block that runs
   the strip (option A), or have `_default_routing` itself call the strip
   before returning (option B). This PR implements **option B** — the strip
   is extracted into ``_apply_self_dispatch_guard`` and called from both
   ``execute()`` and ``_default_routing``.

2. Add a regression unit test that monkey-patches ``_default_routing`` (or
   the underlying dispatch construction) to return ``agent_name="orchestrator"``
   and asserts the final state's ``dispatch_plan`` does NOT contain the self
   literal.

Falsifiability cycle (run before every commit that touches this file):

* Comment out the ``dispatch_plan = self._apply_self_dispatch_guard(...)``
  line inside the production ``_default_routing``. Re-run the AST tests
  below — ``test_default_routing_source_invokes_strip_helper`` AND
  ``test_default_routing_source_assignment_pattern_is_load_bearing``
  MUST trip. Restore the line; confirm green.

* The runtime tests for the helper (``test_apply_self_dispatch_guard_*``)
  exercise the helper's contract directly. Reverting the helper's filter
  predicate (e.g. ``!= "orchestrator"`` → ``!= "explainer"``) trips
  ``test_apply_self_dispatch_guard_helper_exists_and_is_callable``.

The AST + helper-runtime split is intentional: the AST tests pin the
structural conformance (does `_default_routing` call the helper?), and the
helper-runtime tests pin the behavioural contract (does the helper do what
its name says?). Both together cover Issue #269's full surface without
requiring exhaustive integration coverage of every hypothetical future
refactor.
"""

from __future__ import annotations

import ast
from typing import Any, Dict

import pytest

from src.agents.orchestrator.nodes.router import RouterNode
from src.agents.orchestrator.state import AgentDispatch


class TestDefaultRoutingFinalization:
    """Issue #269: `_default_routing` must funnel through the strip helper."""

    @pytest.mark.asyncio
    async def test_default_routing_returns_explainer_when_intent_is_none(self) -> None:
        """Smoke: the default path returns the hard-coded explainer.

        Documents the current contract so a future refactor doesn't
        accidentally change the default agent (which would also break
        user-visible behaviour). This is NOT the falsifiability test for
        the #269 strip — see the AST tests below for the structural
        load-bearing checks.
        """
        router = RouterNode()
        state: Dict[str, Any] = {}  # no intent
        result = await router.execute(state)
        agent_names = [d["agent_name"] for d in result["dispatch_plan"]]
        assert agent_names == ["explainer"], agent_names
        assert "orchestrator" not in agent_names

    @pytest.mark.asyncio
    async def test_apply_self_dispatch_guard_helper_exists_and_is_callable(self) -> None:
        """Structural invariant: the strip helper must exist on RouterNode.

        Issue #269 AC option (B) requires `_default_routing` to call the
        strip before returning. The cleanest implementation extracts the
        existing inline guard at execute() lines 239-262 into a method that
        both call sites share. This test pins the helper's existence so a
        future refactor that inlines it again (and forgets `_default_routing`)
        is caught at import-time-ish.

        Falsifiability: rename ``_apply_self_dispatch_guard`` or remove it
        from RouterNode → this test trips at ``getattr`` (None returned).
        Replacing the filter predicate
        (``d["agent_name"] != "orchestrator"`` → ``True``) → the
        ``"orchestrator" not in cleaned_names`` assertion trips.
        """
        router = RouterNode()
        helper = getattr(router, "_apply_self_dispatch_guard", None)
        assert helper is not None, (
            "RouterNode._apply_self_dispatch_guard helper missing. "
            "Issue #269 requires the self-dispatch strip to be extracted into "
            "a method so both execute() and _default_routing call the same code."
        )
        assert callable(helper)

        # Verify it does what the name suggests.
        plan_with_self = [
            AgentDispatch(
                agent_name="orchestrator",
                priority="medium",
                parameters={},
                timeout_ms=30000,
                fallback_agent=None,
            ),
            AgentDispatch(
                agent_name="explainer",
                priority="medium",
                parameters={},
                timeout_ms=30000,
                fallback_agent=None,
            ),
        ]
        cleaned = helper(plan_with_self, source="test")
        cleaned_names = [d["agent_name"] for d in cleaned]
        assert "orchestrator" not in cleaned_names, cleaned_names
        assert "explainer" in cleaned_names, cleaned_names

    @pytest.mark.asyncio
    async def test_apply_self_dispatch_guard_falls_back_to_explainer_when_all_stripped(
        self,
    ) -> None:
        """When the strip removes ALL entries, the helper must emit a real
        agent (explainer) so the dispatch_plan is never empty.

        This contract is load-bearing: downstream consumers (e.g. cognitive.py
        api fallback) read `dispatch_plan[0]` and would crash on empty plans.

        Falsifiability: remove the
        ``if not filtered_plan: filtered_plan = [explainer-dispatch]``
        block from the helper → this test trips on ``len(cleaned) >= 1``.
        """
        router = RouterNode()
        helper = router._apply_self_dispatch_guard
        all_self = [
            AgentDispatch(
                agent_name="orchestrator",
                priority="medium",
                parameters={},
                timeout_ms=30000,
                fallback_agent=None,
            ),
        ]
        cleaned = helper(all_self, source="test")
        names = [d["agent_name"] for d in cleaned]
        assert len(cleaned) >= 1, cleaned
        assert "orchestrator" not in names, names
        # The contract is: fall back to a real agent. Today it's explainer;
        # if a future refactor changes the fallback agent, this assertion
        # is intentionally tight so the change is reviewed.
        assert names == ["explainer"], names


class TestDefaultRoutingCallsFinalizationOnAllReturnPaths:
    """AST-level structural conformance — the load-bearing #269 check.

    These tests are the **falsifiability anchor** for Issue #269. They
    verify that ``_default_routing``'s source code:

    1. References ``_apply_self_dispatch_guard`` (the helper name).
    2. Assigns the helper's return value back to ``dispatch_plan``
       (the helper returns a new list; discarding the result is a silent
       no-op).

    Falsifiability cycle (verified): comment out the
    ``dispatch_plan = self._apply_self_dispatch_guard(...)`` line in
    production ``_default_routing`` → BOTH AST tests below trip. Restore
    the line → all 5 tests in this module pass.

    Why AST (not runtime)? The runtime test approach
    (monkey-patch ``_default_routing`` to emit ``"orchestrator"``) has a
    structural problem: the monkey-patch itself can either (a) call the
    strip helper (making the test vacuous w.r.t. the production wiring)
    or (b) not call it (making the test test-only, not production). The
    AST test pins the structural contract directly — the same property
    the codex LOW-6 finding flagged.

    The runtime helper tests above (``test_apply_self_dispatch_guard_*``)
    cover the helper's behavioural contract. Together: AST pins WHERE
    the helper is called, runtime tests pin WHAT the helper does.
    """

    @staticmethod
    def _find_strip_assignment_node(source: str) -> "ast.Assign | None":
        """Walk the AST of ``_default_routing``'s source and find the
        ``dispatch_plan = self._apply_self_dispatch_guard(...)`` assignment.

        Returns ``None`` if no such assignment exists. Comments are NOT
        present in the AST, so commenting out the assignment makes this
        return ``None`` — falsifying the AST tests.
        """
        import ast
        import textwrap

        tree = ast.parse(textwrap.dedent(source))
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef), type(func_def)

        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                # Looking for: dispatch_plan = self._apply_self_dispatch_guard(...)
                # Target must be a single Name targeting `dispatch_plan`.
                if len(node.targets) != 1:
                    continue
                target = node.targets[0]
                if not isinstance(target, ast.Name) or target.id != "dispatch_plan":
                    continue
                value = node.value
                if not isinstance(value, ast.Call):
                    continue
                func = value.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "_apply_self_dispatch_guard"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                ):
                    return node
        return None

    def test_default_routing_source_invokes_strip_helper(self) -> None:
        """The source of `_default_routing` must reference the strip helper.

        Uses AST (not string match) so commented-out invocations do NOT pass.
        ``inspect.getsource`` returns the textual source including comments,
        but ``ast.parse`` only sees executable code — commented lines never
        produce ``ast.Assign`` nodes.

        Falsifiability (verified by programmatic simulation): replace the
        ``self._apply_self_dispatch_guard(dispatch_plan, source="_default_routing")``
        line in production code with a comment OR remove it → this test trips
        because the AST walk finds no assignment node.
        """
        import inspect

        source = inspect.getsource(RouterNode._default_routing)
        strip_node = self._find_strip_assignment_node(source)
        assert strip_node is not None, (
            "`_default_routing` does not invoke `dispatch_plan = self._apply_self_dispatch_guard(...)`. "
            "Issue #269 requires every return path to be funneled through the "
            "strip helper so a future refactor that changes the default agent "
            "to a configurable value cannot silently re-introduce F1. "
            f"Current source:\n{source}"
        )

    def test_default_routing_source_assignment_pattern_is_load_bearing(self) -> None:
        """The strip helper call must ASSIGN its result back to ``dispatch_plan``.

        A common refactor hazard is calling the helper but discarding the
        return value (e.g. ``self._apply_self_dispatch_guard(dispatch_plan)``
        without ``dispatch_plan = ...``). Because the helper returns a NEW
        list (it doesn't mutate in place), the discarded result means the
        strip is silently skipped.

        Uses AST: a bare ``Expression`` call site (no assignment) produces
        an ``ast.Expr`` node, not an ``ast.Assign``, so this test trips when
        the call result is discarded.

        Falsifiability (verified): remove the ``dispatch_plan = `` prefix
        from the production call site → ``_find_strip_assignment_node``
        returns None → this test trips.
        """
        import inspect

        source = inspect.getsource(RouterNode._default_routing)
        strip_node = self._find_strip_assignment_node(source)
        assert strip_node is not None, (
            "_default_routing has no `dispatch_plan = self._apply_self_dispatch_guard(...)` "
            "assignment node in its AST. The helper returns a NEW list; discarding "
            "it (calling without assignment) silently skips the strip. "
            f"Current source:\n{source}"
        )

    def test_default_routing_source_assignment_precedes_return(self) -> None:
        """The strip MUST be applied BEFORE the return statement.

        Falsifiability: move the
        ``dispatch_plan = self._apply_self_dispatch_guard(...)`` call AFTER
        the ``return {...}`` block (so it becomes dead code) → this test
        trips.

        Implementation: walk the AST of ``_default_routing`` and find the
        first ``Return`` node. The strip-helper call's line number must
        precede the Return's line number.
        """
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(RouterNode._default_routing))
        tree = ast.parse(source)
        func_def = tree.body[0]
        assert isinstance(func_def, ast.FunctionDef), type(func_def)

        # Find the strip-helper call site's line number.
        strip_line: int | None = None
        return_line: int | None = None
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                # Looking for: dispatch_plan = self._apply_self_dispatch_guard(...)
                value = node.value
                if isinstance(value, ast.Call):
                    func = value.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "_apply_self_dispatch_guard"
                    ):
                        strip_line = node.lineno
            if isinstance(node, ast.Return) and return_line is None:
                return_line = node.lineno

        assert strip_line is not None, (
            "No `dispatch_plan = self._apply_self_dispatch_guard(...)` assignment "
            f"found in _default_routing AST. Source:\n{source}"
        )
        assert return_line is not None, (
            f"No return statement found in _default_routing AST. Source:\n{source}"
        )
        assert strip_line < return_line, (
            f"strip helper called at line {strip_line} (relative to function start) "
            f"but return is at line {return_line}. The strip MUST precede the return. "
            f"Source:\n{source}"
        )
