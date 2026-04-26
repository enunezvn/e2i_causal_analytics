"""Block 5B (#10): unit tests for the placeholder cost-matrix CLI plumbing.

Block 5 wired ``cost_matrix`` end-to-end (scope_definer → scope_spec →
model_trainer → evaluator) but no caller of ``scripts/run_tier0_test.py``
populated one, so a default ``python scripts/run_tier0_test.py`` run never
emitted ``business_utility``. Block 5B closes that verification gap by
auto-injecting a unit-shape placeholder unless the new
``--no-demo-cost-matrix`` flag is passed.

These tests assert the CLI plumbing in isolation (the helpers +
argparse), without spinning up the full pipeline:

  - ``_default_demo_cost_matrix()`` returns the unit-shape contract.
  - ``_build_parser()`` produces a parser that recognises
    ``--no-demo-cost-matrix`` (default ``False``).
  - ``_should_inject_demo_cost_matrix(scope_spec, inject)`` returns
    ``True`` only when injection is requested AND the scope_spec has
    no usable cost_matrix.

Block 6B-polish chunk 2 / 5B-I-2 + I-3: the previous version mocked
the inject branch with a hand-rolled re-implementation and read the
parser via subprocess ``--help``. Both are now replaced by direct
imports of the real helpers from ``scripts/run_tier0_test`` so a drift
in the production decision rule is caught here, not just in the
synthetic e2e gate.

The full e2e check that ``business_utility`` lands in
``validation_metrics``/``test_metrics`` is covered separately by the
synthetic test in ``tests/synthetic/test_business_utility_emitted.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tier0_test import (  # noqa: E402
    _build_parser,
    _default_demo_cost_matrix,
    _should_inject_demo_cost_matrix,
)

# ---------------------------------------------------------------------------
# Helper contract
# ---------------------------------------------------------------------------


class TestDefaultDemoCostMatrix:
    """``_default_demo_cost_matrix`` is the single source of truth for the
    placeholder shape; the CLI wiring and the synthetic e2e test both
    consume it. If this contract drifts, the synthetic test's exact
    arithmetic check will catch it loudly — these tests pin the shape so
    the drift is caught at the unit level too.
    """

    def test_returns_unit_shape(self):
        """The shape MUST be unit-scaled, not dollar-denominated.

        - tp = +1.0  : a true-positive prediction is worth one unit
        - fn = -1.0  : a missed target costs the same one unit
        - fp = -0.05 : a false-positive costs 5 % of a unit
        - tn =  0.0  : a true-negative is neutral

        Production callers MUST supply real per-brand dollar values; this
        default is for the dev runner only.
        """
        cm = _default_demo_cost_matrix()
        assert cm == {"tp": 1.0, "fp": -0.05, "fn": -1.0, "tn": 0.0}

    def test_values_are_floats(self):
        """All four values must be plain floats — the validator at
        ``scope_builder._validate_cost_matrix`` rejects ints-with-bool
        and other surprises, so the helper produces a fully-typed dict
        that can flow through unchanged."""
        cm = _default_demo_cost_matrix()
        for key, value in cm.items():
            assert isinstance(value, float), (
                f"cost_matrix[{key!r}] is {type(value).__name__}, expected float"
            )

    def test_returns_fresh_dict_each_call(self):
        """The helper must NOT return a shared mutable singleton — callers
        push the result onto ``scope_spec`` and downstream code is free to
        mutate the spec dict. Aliasing across runs would create cross-run
        contamination that's painful to debug."""
        a = _default_demo_cost_matrix()
        b = _default_demo_cost_matrix()
        assert a == b
        assert a is not b
        a["tp"] = 999.0
        assert b["tp"] == 1.0


# ---------------------------------------------------------------------------
# Argparse plumbing — exercise the REAL parser via _build_parser
# ---------------------------------------------------------------------------


class TestArgparseFlag:
    """The ``--no-demo-cost-matrix`` flag is a boolean opt-out: default
    ``False`` (auto-inject ON), present in argv → ``True`` (auto-inject
    OFF). 5B-I-3: tests now exercise the real parser, no subprocess."""

    def test_flag_default_is_false(self):
        parser = _build_parser()
        # parse only the args we care about — argparse will accept []
        # and use defaults for everything.
        ns = parser.parse_args([])
        assert ns.no_demo_cost_matrix is False

    def test_flag_when_passed_is_true(self):
        parser = _build_parser()
        ns = parser.parse_args(["--no-demo-cost-matrix"])
        assert ns.no_demo_cost_matrix is True

    def test_real_parser_recognises_flag(self):
        """Sanity check: the real parser MUST register the flag.

        5B-I-3: replaces the subprocess-with-magic-number ``--help``
        test from the previous iteration; we now query the parser
        directly for the action."""
        parser = _build_parser()
        flag_actions = [
            a for a in parser._actions if "--no-demo-cost-matrix" in a.option_strings
        ]
        assert len(flag_actions) == 1, (
            "Expected exactly one --no-demo-cost-matrix action; "
            f"found {len(flag_actions)}: {[a.option_strings for a in flag_actions]}"
        )
        action = flag_actions[0]
        # The flag is a store_true boolean opt-out.
        assert action.dest == "no_demo_cost_matrix"
        assert action.default is False
        # store_true actions have const=True (so passing the flag yields True).
        assert action.const is True


# ---------------------------------------------------------------------------
# Auto-inject decision — exercise the REAL helper from run_tier0_test
# ---------------------------------------------------------------------------


class TestAutoInjectBranch:
    """5B-I-2: the inject decision is now ``_should_inject_demo_cost_matrix``
    inside ``scripts/run_tier0_test``. Tests import and exercise the real
    helper rather than mocking the branch logic. Search the script for
    "Block 5B (#10): auto-inject" to find the call site that consumes
    this helper.
    """

    def test_flag_absent_injects_when_no_existing_matrix(self):
        """Default behaviour: inject=True + no existing matrix → True."""
        scope_spec: dict = {"experiment_id": "exp_001"}
        assert _should_inject_demo_cost_matrix(scope_spec, inject=True) is True

    def test_flag_present_suppresses_inject(self):
        """``--no-demo-cost-matrix`` passed → False even with no matrix."""
        scope_spec: dict = {"experiment_id": "exp_002"}
        assert _should_inject_demo_cost_matrix(scope_spec, inject=False) is False

    def test_existing_cost_matrix_is_preserved(self):
        """If scope_definer already produced a cost_matrix (e.g. a
        future LLM-driven path), the helper returns False so the
        existing matrix is preserved."""
        existing = {"tp": 250.0, "fp": -25.0, "fn": -200.0, "tn": 0.0}
        scope_spec: dict = {"experiment_id": "exp_003", "cost_matrix": existing}
        assert _should_inject_demo_cost_matrix(scope_spec, inject=True) is False

    def test_scope_definer_default_none_value_triggers_inject(self):
        """ScopeDefinerAgent emits ``cost_matrix=None`` by default. The
        helper MUST treat present-but-None as un-set (the previous bug
        was that ``"cost_matrix" not in scope_spec`` returned False
        against ``{"cost_matrix": None}`` and the placeholder never
        landed)."""
        scope_spec: dict = {"experiment_id": "exp_004", "cost_matrix": None}
        assert _should_inject_demo_cost_matrix(scope_spec, inject=True) is True

    def test_empty_dict_cost_matrix_triggers_inject(self):
        """Edge case: ``cost_matrix={}`` is falsy and should trigger
        inject (an empty dict provides no signal to the evaluator)."""
        scope_spec: dict = {"experiment_id": "exp_005", "cost_matrix": {}}
        assert _should_inject_demo_cost_matrix(scope_spec, inject=True) is True
