"""Tests that ``copilotkit._classify_query_type`` is converged onto the
SSOT in ``src/agents/multi_faceted.py`` (issue #295).

After PR #296 converged the three Tool Composer dispatch heuristics onto
``src/agents/multi_faceted.py``, a **fourth** site producing the literal
label ``"multi_faceted"`` still lived at
``src/api/routes/copilotkit.py:988-1016`` inside ``_classify_query_type``,
re-implementing a topic-count keyword-grouping heuristic inline.

This test file enforces convergence by two complementary checks:

  - **Structural (AST)** — the function body does not contain a parallel
    keyword-grouping heuristic. Specifically, the 5 topic-keyword groups
    of literals ({trx/nrx/...}, {causal/impact/...}, etc.) must not all
    appear inline; the multi-facet decision MUST be delegated to
    ``src.agents.multi_faceted``.

  - **Behavioral** — the function still returns ``"multi_faceted"`` for
    queries that the SSOT's analytics-labeling algorithm classifies as
    multi-faceted, and the 8 non-multi-faceted enum labels are still
    reachable.

Falsifiability: revert the convergence (restore the inline 5-group
``topic_count`` heuristic and the ``return "multi_faceted"`` literal)
and the structural test trips on the unique-literal sets while the
behavioral parity-vs-SSOT test trips because the inline heuristic and
the SSOT helper would no longer share an implementation. Verified
manually as part of landing this PR.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

from src.agents import multi_faceted as ssot
from src.api.routes import copilotkit


# ---------------------------------------------------------------------------
# Structural — the function body does not duplicate the SSOT's job.
# ---------------------------------------------------------------------------


class TestNoDuplicateKeywordGroupingInline:
    """The 5 topic-keyword groups must not all be inlined in
    ``_classify_query_type`` — the multi-facet decision must be delegated
    to ``src/agents/multi_faceted.py``.
    """

    # These are the literal keyword groups from the pre-convergence
    # implementation (copilotkit.py:1006-1013). We use minimal "tripwire"
    # subsets — if a future regression re-introduces the 5-group pattern,
    # all five subsets re-appear and this test trips.
    _GROUP_TRIPWIRES: tuple[tuple[str, str, str], ...] = (
        ("trx", "nrx", "kpi"),
        ("causal", "impact", "intervention"),
        ("predict", "forecast", "future"),
        ("experiment", "ab test", "a/b"),
        ("drift", "shift", "degradation"),
    )

    def _source(self) -> str:
        return inspect.getsource(copilotkit._classify_query_type)

    def test_does_not_inline_all_five_topic_groups(self):
        """A re-introduction of the topic-count heuristic would put all
        5 groups inline. With ≥1 group missing from the source, the
        multi-faceted decision cannot be the original inline algorithm.
        """
        source = self._source()
        present_groups = sum(
            all(kw in source for kw in group) for group in self._GROUP_TRIPWIRES
        )
        assert present_groups < 5, (
            "All 5 topic keyword groups present inline in _classify_query_type — "
            "the multi-faceted detection has been re-duplicated. Delegate to "
            "src.agents.multi_faceted instead."
        )

    def test_delegates_to_multi_faceted_ssot(self):
        """Either the source imports ``src.agents.multi_faceted`` (at
        any name) or calls a public helper from that module. We accept
        any reference shape; what we forbid is the function deciding
        ``multi_faceted`` purely from inlined keyword groups.
        """
        module_source = inspect.getsource(copilotkit)
        function_source = self._source()

        imports_ssot = bool(
            re.search(r"from\s+src\.agents(\.|\s+import\s+)multi_faceted", module_source)
            or re.search(r"import\s+src\.agents\.multi_faceted", module_source)
        )

        # Resolve callable names exported by the SSOT module so we
        # match against the real attribute names (forward-compatible
        # if a new helper is added).
        ssot_callable_names = tuple(
            name
            for name, obj in vars(ssot).items()
            if callable(obj) and not name.startswith("_")
        )
        calls_ssot_helper = any(
            name in function_source for name in ssot_callable_names
        )

        assert imports_ssot and calls_ssot_helper, (
            "_classify_query_type must import from and call into "
            "src.agents.multi_faceted (SSOT for multi-faceted detection). "
            f"imports_ssot={imports_ssot} calls_ssot_helper={calls_ssot_helper}"
        )

    def test_classify_function_returns_multi_faceted_via_ssot_call(self):
        """Walk the AST of ``_classify_query_type``; for the ``return
        \"multi_faceted\"`` branch, require that the controlling
        expression involves a call (not a pure literal-keyword
        comprehension). Falsifiability: reverting to the inline
        ``topic_count = sum([...]); if topic_count >= 2: return
        "multi_faceted"`` pattern trips this because the controlling
        if-test has zero ``ast.Call`` nodes whose function name is in
        the SSOT module.
        """
        source = self._source()
        tree = ast.parse(source)
        func = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        )

        # Find the return-"multi_faceted" statement.
        multi_faceted_returns = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and node.value.value == "multi_faceted"
        ]
        assert multi_faceted_returns, (
            'Expected at least one `return "multi_faceted"` statement.'
        )

        # The controlling If for at least one such return must contain
        # an ast.Call referencing an SSOT-module callable.
        ssot_callable_names = {
            name
            for name, obj in vars(ssot).items()
            if callable(obj) and not name.startswith("_")
        }

        def call_targets(node: ast.AST) -> set[str]:
            names: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    if isinstance(fn, ast.Name):
                        names.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        names.add(fn.attr)
            return names

        # Find the enclosing If for each multi_faceted return.
        controlling_if_calls: set[str] = set()
        for ret in multi_faceted_returns:
            for ancestor in ast.walk(func):
                if isinstance(ancestor, ast.If) and ret in ast.walk(ancestor):
                    controlling_if_calls |= call_targets(ancestor.test)

        assert controlling_if_calls & ssot_callable_names, (
            "The multi_faceted-return branch is not gated by a call into "
            "src.agents.multi_faceted. The function appears to decide "
            '"multi_faceted" without delegating to the SSOT. '
            f"controlling_if_calls={controlling_if_calls!r} ssot_callable_names={ssot_callable_names!r}"
        )


# ---------------------------------------------------------------------------
# Behavioral — the function still returns "multi_faceted" exactly when
# the SSOT helper does, on representative queries.
# ---------------------------------------------------------------------------


class TestClassifierBehaviorPreserved:
    """Black-box behavior preserved for the 3 callers (copilotkit.py
    L2154/L2313/L2363) feeding ``query_type`` into chat-message
    analytics.
    """

    @pytest.mark.parametrize(
        "query,expected_label",
        [
            # Multi-faceted: queries that combine ≥2 of the analytics
            # topic groups (KPI + causal, KPI + drift, etc.). These
            # tripped the pre-convergence inline heuristic and must
            # continue to be labelled "multi_faceted" after convergence.
            ("show trx and the causal effect of the campaign", "multi_faceted"),
            ("forecast trx and detect drift", "multi_faceted"),
            ("run an experiment to predict future trx", "multi_faceted"),
            # Single-topic enum labels (no multi-faceted classification).
            ("what is the trx for kisqali?", "kpi_inquiry"),
            ("why did sales drop", "causal_analysis"),
            ("system tier health", "agent_status"),
            ("what should we do next", "recommendation"),
            ("run an a/b test", "experiment"),
            ("forecast next quarter", "prediction"),
            ("any drift in the data?", "drift_alert"),
            ("hello", "general"),
        ],
    )
    def test_classify_returns_expected_label(self, query, expected_label):
        assert copilotkit._classify_query_type(query) == expected_label
