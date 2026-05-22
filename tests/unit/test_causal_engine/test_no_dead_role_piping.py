"""Drift-guard regression test (issue #237).

Asserts that ``src/causal_engine/refutation_runner.py`` does not
re-introduce the diagnostic role-attribution piping that PR #365
added and this PR retires.

Rationale
---------
Per the deep-research plan ``.claude/plans/causal_role_engine_consumption_v0.md``
+ user direction, ``causal_engine`` does not consume Layer-4
``causal_role`` for confidence propagation. The diagnostic helper
``_enrich_offending_features`` and its ``role_attributions`` /
``offending_features`` parameter chain in
``RefutationRunner.run_all_tests`` / ``_run_placebo_test`` was dead
code (no production caller ever passed those kwargs), and has been
removed.

This test pins the post-removal state so a future PR can't silently
re-introduce dead piping without the CI failing first.

What this test does NOT pin
---------------------------
This test does NOT forbid all role-aware code in ``causal_engine``
forever. If a future PR wires a real consumer in
``refutation_runner.py`` from a new engine-layer policy plan
(the predecessor engine-layer adjustment-set policy lift was
permanently REJECTED in plan v2 — see
``.claude/plans/237_engine_role_consumption_v2.md``), the new code
MUST also update or replace this test with one that pins the new
wiring's actual contract. The point is to fail loudly on accidental
re-add, not to prevent intentional future work.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[3]
REFUTATION_RUNNER = REPO_ROOT / "src" / "causal_engine" / "refutation_runner.py"

# Symbols + parameter names that the dead diagnostic piping introduced.
# Re-introducing any of these requires updating this test (i.e.,
# documenting the new consumer that justifies the wiring).
_FORBIDDEN_SYMBOLS: tuple[str, ...] = ("_enrich_offending_features",)
_FORBIDDEN_PARAMS: tuple[str, ...] = ("role_attributions", "offending_features")
_GATED_FUNCTIONS: tuple[str, ...] = ("run_all_tests", "_run_placebo_test")


def _source_tree() -> ast.Module:
    assert REFUTATION_RUNNER.is_file(), f"refutation_runner.py not found at {REFUTATION_RUNNER!s}"
    return ast.parse(REFUTATION_RUNNER.read_text(encoding="utf-8"), filename=str(REFUTATION_RUNNER))


def test_no_enrich_offending_features_symbol() -> None:
    """The diagnostic enrichment helper must not exist."""
    tree = _source_tree()
    findings: List[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in _FORBIDDEN_SYMBOLS
        ):
            findings.append(f"line {node.lineno}: function {node.name!r} reintroduced")
    assert not findings, (
        "Forbidden helper(s) re-introduced in refutation_runner.py — these were "
        "removed per #237 (no production consumer):\n  - " + "\n  - ".join(findings)
    )


def test_no_role_attributions_or_offending_features_params_on_gated_functions() -> None:
    """``run_all_tests`` and ``_run_placebo_test`` must not declare the dead-code kwargs."""
    tree = _source_tree()
    findings: List[str] = []

    # Walk to find class methods named in _GATED_FUNCTIONS.
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in _GATED_FUNCTIONS
        ):
            param_names = {arg.arg for arg in node.args.args} | {
                arg.arg for arg in node.args.kwonlyargs
            }
            for forbidden in _FORBIDDEN_PARAMS:
                if forbidden in param_names:
                    findings.append(
                        f"line {node.lineno}: function {node.name!r} re-declares parameter {forbidden!r}"
                    )

    assert not findings, (
        "Forbidden parameter(s) re-introduced on gated functions — these were "
        "removed per #237 (no production caller passed them):\n  - " + "\n  - ".join(findings)
    )
