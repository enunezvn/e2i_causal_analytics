"""#1662: a function-local ``import X`` shadows the module-level ``X`` for the
WHOLE function, so any earlier use of ``X`` in that function raises
``UnboundLocalError`` at runtime.

This is not a style rule. It took down the live AG-UI surface. PR #1647 (branch
``fix/trx-substrate-fence-1640``, for issue #1640) hoisted ``import time`` to
module scope across two commits -- ``6a460f1e7`` added the module-level import,
``4e794fa34`` removed SIX local copies including the one at the TOP of
``LangGraphAgent.execute`` -- and left FOUR behind further down the same
function. Python's scoping rules then made ``time`` local to ``execute`` in its
entirety, so ``start_time = time.time()`` at the top of the function -- the
first thing it does -- raised

    UnboundLocalError: cannot access local variable 'time' where it is not
    associated with a value

on EVERY ``agent/run`` request, from the moment ``6eafcf4ae`` merged it to main.

Note the failure mode, because it is worse than an error status and will not be
found by searching for one: the client gets **HTTP 200 with an empty body**.
``StreamingResponse`` commits the status line before the body generator is
iterated, so by the time ``execute`` raises there is nothing left to signal
with -- the stream simply ends. Measured live before the fix: a real
``agent/run`` returned ``HTTP 200 frames=0 elapsed=0.6s`` while ``e2i_api``
logged the traceback at ``copilotkit.py:846``; 16 occurrences over the
container's ~2h lifetime.

Two properties make this worth a guard rather than a one-line fix:

* it is **invisible to normal review** -- the deleted import and the surviving
  ones are hundreds of lines apart, and each surviving one looks locally
  harmless;
* it is **invisible to mypy and ruff** -- neither flags a redundant local import
  that shadows a module-level name, because in isolation each is legal.

So the guard is an AST scan, and it is deliberately repo-wide rather than
scoped to the file that broke: the defect is a property of Python scoping, not
of ``copilotkit.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"


def _module_level_import_names(tree: ast.Module) -> set[str]:
    """Names bound by imports at module scope."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _shadowing_offenders(path: Path) -> list[str]:
    """Function-local imports that shadow a module-level name ALREADY LOADED
    earlier in the same function.

    Only that ordering is a bug. A function-local import of a module that is
    never used before it is a legitimate and widely-used pattern here -- this
    repo deliberately defers heavy imports (importing any ``src.rag`` leaf pulls
    in dspy at ~714MB), so a blanket ban would be wrong.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    module_names = _module_level_import_names(tree)
    offenders: list[str] = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        local_import_line: dict[str, int] = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound in module_names:
                        line = local_import_line.get(bound)
                        if line is None or node.lineno < line:
                            local_import_line[bound] = node.lineno

        if not local_import_line:
            continue

        for node in ast.walk(func):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in local_import_line
                and node.lineno < local_import_line[node.id]
            ):
                offenders.append(
                    f"{path.relative_to(SRC.parent)}:{node.lineno} loads "
                    f"{node.id!r} in {func.name}() before the function-local "
                    f"'import {node.id}' at line {local_import_line[node.id]} "
                    f"-> UnboundLocalError at runtime"
                )

    return offenders


def test_copilotkit_execute_has_no_shadowing_local_import() -> None:
    """The exact regression. Fails before the fix with the two real loads."""
    offenders = _shadowing_offenders(SRC / "api" / "routes" / "copilotkit.py")
    assert not offenders, "\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize(
    "path",
    sorted(SRC.rglob("*.py")),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_no_module_shadowing_local_imports_anywhere(path: Path) -> None:
    """Repo-wide: the defect is a property of Python scoping, not of one file."""
    offenders = _shadowing_offenders(path)
    assert not offenders, "\n  " + "\n  ".join(offenders)
