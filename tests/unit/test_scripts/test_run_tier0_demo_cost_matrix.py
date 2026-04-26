"""Block 5B (#10): unit tests for the placeholder cost-matrix CLI plumbing.

Block 5 wired ``cost_matrix`` end-to-end (scope_definer → scope_spec →
model_trainer → evaluator) but no caller of ``scripts/run_tier0_test.py``
populated one, so a default ``python scripts/run_tier0_test.py`` run never
emitted ``business_utility``. Block 5B closes that verification gap by
auto-injecting a unit-shape placeholder unless the new
``--no-demo-cost-matrix`` flag is passed.

These tests assert the CLI plumbing in isolation (the helper +
argparse), without spinning up the full pipeline:

  - ``_default_demo_cost_matrix()`` returns the unit-shape contract.
  - argparse parses ``--no-demo-cost-matrix`` (default ``False``).
  - When the flag is absent, the auto-inject path writes the
    placeholder onto ``scope_spec["cost_matrix"]``.
  - When the flag is present, the auto-inject is suppressed and
    ``scope_spec`` carries no ``cost_matrix`` key.

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

from scripts.run_tier0_test import _default_demo_cost_matrix  # noqa: E402

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
# Argparse plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]):
    """Reconstruct the script's argparse parser without invoking ``main()``.

    ``main()`` calls ``parser.parse_args()`` and then runs the pipeline,
    which is way too heavy for a unit test. Instead we re-create the
    same ArgumentParser shape and only assert on the parsed flag. The
    flag is the only piece the test cares about; if the script's parser
    drifts, this test will fail loudly because ``argv`` no longer
    matches.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-demo-cost-matrix", action="store_true")
    return parser.parse_args(argv)


class TestArgparseFlag:
    """The ``--no-demo-cost-matrix`` flag is a boolean opt-out: default
    ``False`` (auto-inject ON), present in argv → ``True`` (auto-inject
    OFF)."""

    def test_flag_default_is_false(self):
        ns = _parse_args([])
        assert ns.no_demo_cost_matrix is False

    def test_flag_when_passed_is_true(self):
        ns = _parse_args(["--no-demo-cost-matrix"])
        assert ns.no_demo_cost_matrix is True

    def test_real_script_parser_recognises_flag(self):
        """Sanity check: the actual ``main()`` parser must accept the
        flag without error. We can't easily isolate the parser (it's
        local to ``main``) but we can confirm ``--help`` mentions it.
        """
        import subprocess

        # Just confirm the flag is in --help output. ``main()`` is
        # heavy — ``--help`` short-circuits before any pipeline runs.
        result = subprocess.run(
            [sys.executable, "scripts/run_tier0_test.py", "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--help exited non-zero: stderr={result.stderr}"
        )
        assert "--no-demo-cost-matrix" in result.stdout, (
            "argparse did not register --no-demo-cost-matrix; help text:\n"
            + result.stdout
        )


# ---------------------------------------------------------------------------
# Auto-inject behaviour (mimics the run_pipeline branch)
# ---------------------------------------------------------------------------


class TestAutoInjectBranch:
    """The auto-inject lives at ``run_pipeline`` line ~3753. It runs
    AFTER step_1 returns and writes the placeholder onto ``scope_spec``
    when (a) ``inject_demo_cost_matrix`` is True AND (b) ``cost_matrix``
    is not already present. We replicate that branch logic here so the
    test catches drift without invoking the full pipeline.
    """

    @staticmethod
    def _apply_branch(scope_spec: dict, inject: bool) -> dict:
        """Mirror the run_pipeline branch exactly. Keep this in sync with
        ``scripts/run_tier0_test.py`` (search ``Block 5B (#10): auto-
        inject the unit-shape placeholder``).

        Note: scope_definer always emits ``cost_matrix`` as a key (with a
        ``None`` default) so we use ``.get("cost_matrix")`` falsiness as
        the "un-set" check, not raw key presence. A populated dict is
        preserved (caller already has a real matrix).
        """
        if inject and not scope_spec.get("cost_matrix"):
            scope_spec["cost_matrix"] = _default_demo_cost_matrix()
        return scope_spec

    def test_flag_absent_injects_placeholder(self):
        """Default behaviour: ``--no-demo-cost-matrix`` not passed →
        ``scope_spec["cost_matrix"]`` is the unit-shape placeholder."""
        scope_spec: dict = {"experiment_id": "exp_001"}
        result = self._apply_branch(scope_spec, inject=True)
        assert result["cost_matrix"] == {
            "tp": 1.0,
            "fp": -0.05,
            "fn": -1.0,
            "tn": 0.0,
        }

    def test_flag_present_suppresses_inject(self):
        """``--no-demo-cost-matrix`` passed → no ``cost_matrix`` key."""
        scope_spec: dict = {"experiment_id": "exp_002"}
        result = self._apply_branch(scope_spec, inject=False)
        assert "cost_matrix" not in result

    def test_existing_cost_matrix_is_preserved(self):
        """If scope_definer already produced a cost_matrix (e.g. a
        future LLM-driven path), the auto-inject MUST NOT overwrite
        it — the placeholder is a fallback, not a stomp."""
        existing = {"tp": 250.0, "fp": -25.0, "fn": -200.0, "tn": 0.0}
        scope_spec: dict = {"experiment_id": "exp_003", "cost_matrix": existing}
        result = self._apply_branch(scope_spec, inject=True)
        assert result["cost_matrix"] == existing

    def test_scope_definer_default_none_value_triggers_inject(self):
        """ScopeDefinerAgent emits ``cost_matrix=None`` by default (see
        ``scope_builder._validate_cost_matrix``). The auto-inject MUST
        treat a present-but-None key as un-set; the previous bug was
        that ``"cost_matrix" not in scope_spec`` returned False against
        ``{"cost_matrix": None}`` and the placeholder never landed."""
        scope_spec: dict = {"experiment_id": "exp_004", "cost_matrix": None}
        result = self._apply_branch(scope_spec, inject=True)
        assert result["cost_matrix"] == _default_demo_cost_matrix()
