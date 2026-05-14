"""Static lint: forbid bare ``asyncio.run(...)`` inside ``tests/integration/``
synchronous test bodies (issue #220 — companion to #218's src/ lint).

Companion to ``tests/integration/test_no_unconditional_nest_asyncio_apply.py``.
That lint covers ``src/``; this one covers ``tests/integration/``. The
motivation is the same monkey-patch bug class: once *any* test on an xdist
worker triggers ``nest_asyncio.apply()`` (via the RAGAS polluter at
``ragas/async_utils.py:49`` identified by PR #219, or any future caller),
every subsequent bare ``asyncio.run(coro)`` on that worker routes through
``nest_asyncio.run`` against a possibly-closed loop and raises
``RuntimeError: Event loop is closed`` (issue #215 victim pattern).

The legitimate forms inside an integration test are:

  (a) ``@pytest.mark.asyncio`` + ``async def test_…`` + ``await coro`` —
      pytest-asyncio drives the loop; no bare ``asyncio.run`` needed.
  (b) Explicit-loop pattern (canonical post-PR #217 / #220 migration)::

          loop = asyncio.new_event_loop()
          try:
              loop.run_until_complete(coro)
          finally:
              loop.close()

  (c) ``asyncio.run`` *inside an async-def helper that itself is awaited*
      — exotic but not regressive; we allow because the scan focuses on
      top-level / sync-test bodies where the closed-loop bug actually fires.

What this lint REJECTS:

  - Bare ``asyncio.run(coro)`` inside a sync ``def test_*`` body.
  - Bare ``asyncio.run(coro)`` inside a sync top-level helper that a
    sync test calls (transitive risk).

What it does NOT reject (by design — runtime probe in
``tests/conftest.py`` is the authoritative fallback):

  - Calls in ``async def`` functions (those run inside a managed loop).
  - Calls inside docstrings, strings, or comments.
  - Calls in ``tests/unit/`` (out of scope; this lint is integration-only
    because the RAGAS pollution chain manifests on the integration-tests
    xdist lane).

If a file legitimately needs an unmigrated callsite (e.g. it is a
regression pin documenting the old pattern for some upstream reason),
add it to ``_EXEMPT_FILES`` with a comment justifying the carve-out and
a tracking issue link.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_INTEGRATION_ROOT = REPO_ROOT / "tests" / "integration"

# Files explicitly exempted. Empty by default — every callsite under
# ``tests/integration/`` must be either migrated or accompanied by a
# carve-out entry here. The exemption operates at file granularity to
# keep the lint cheap; tighten to per-line if a real partial case appears.
_EXEMPT_FILES: frozenset[str] = frozenset()


class _BareAsyncioRunScanner(ast.NodeVisitor):
    """AST walker that flags ``asyncio.run(...)`` calls inside *synchronous*
    function bodies.

    Tracks the enclosing-function async/sync mode via a stack so calls
    inside ``async def`` helpers are correctly skipped, while calls
    inside sync ``def test_…`` bodies are flagged.

    Also tracks ``asyncio`` import bindings so renamed imports
    (``import asyncio as aio``) are caught.
    """

    def __init__(self) -> None:
        super().__init__()
        self._asyncio_module_bindings: set[str] = {"asyncio"}
        # Stack of booleans: True == inside an async function, False == sync.
        self._async_scope_stack: list[bool] = []
        self.unguarded: list[tuple[int, str]] = []  # (lineno, source line)
        self._lines: list[str] = []

    def set_source(self, source: str) -> None:
        self._lines = source.splitlines()

    def _record(self, lineno: int) -> None:
        line = self._lines[lineno - 1] if 0 < lineno <= len(self._lines) else ""
        self.unguarded.append((lineno, line))

    def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
        for alias in node.names:
            if alias.name == "asyncio":
                self._asyncio_module_bindings.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        self._async_scope_stack.append(False)
        try:
            self.generic_visit(node)
        finally:
            self._async_scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        self._async_scope_stack.append(True)
        try:
            self.generic_visit(node)
        finally:
            self._async_scope_stack.pop()

    def _is_asyncio_run(self, call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "run":
            value = func.value
            return isinstance(value, ast.Name) and value.id in self._asyncio_module_bindings
        return False

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        if self._is_asyncio_run(node):
            # Skip if we're inside an async function (legitimate context).
            in_async = bool(self._async_scope_stack and self._async_scope_stack[-1])
            if not in_async:
                self._record(node.lineno)
        self.generic_visit(node)


def _scan_file(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return list of (lineno, source line) for unguarded ``asyncio.run``
    calls in ``path``."""

    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    scanner = _BareAsyncioRunScanner()
    scanner.set_source(source)
    scanner.visit(tree)
    return scanner.unguarded


def _iter_test_files() -> list[pathlib.Path]:
    """All ``tests/integration/**/*.py`` test source files."""

    return [p for p in TESTS_INTEGRATION_ROOT.rglob("*.py") if p.name != "__init__.py"]


def test_no_bare_asyncio_run_in_integration_tests() -> None:
    """Issue #220: forbid bare ``asyncio.run(coro)`` inside synchronous
    test bodies under ``tests/integration/``.

    Once an xdist test on the worker triggers ``nest_asyncio.apply()``
    (the RAGAS polluter at ``ragas/async_utils.py:49``, or any future
    third-party caller), all subsequent bare ``asyncio.run`` calls on
    the same worker route through ``nest_asyncio.run`` and may surface
    ``RuntimeError: Event loop is closed`` against a since-closed loop.
    The fix is the explicit-loop pattern (PR #217 commit ``a321b64f``):

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    Or convert to ``async def test_…`` + ``await coro`` (the repo runs
    pytest-asyncio with ``asyncio_mode = "auto"``, so the async form
    just works).
    """

    violations: list[tuple[pathlib.Path, int, str]] = []
    for path in _iter_test_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _EXEMPT_FILES:
            continue
        for lineno, line in _scan_file(path):
            violations.append((path, lineno, line))

    if violations:
        msg_lines = [
            f"{len(violations)} bare ``asyncio.run(...)`` call(s) in "
            f"synchronous tests/integration/ bodies — latent victims of "
            f"the RAGAS pollution chain (issue #220 / #218 / #215):",
        ]
        for path, lineno, line in violations:
            rel = path.relative_to(REPO_ROOT).as_posix()
            msg_lines.append(f"  {rel}:{lineno}: {line.strip()}")
        msg_lines.append(
            "Migrate to the explicit-loop pattern (see PR #217 commit "
            "``a321b64f`` for the canonical example) or convert the test "
            "to ``async def`` + ``await``. The repo's pytest-asyncio "
            "auto-mode handles the latter."
        )
        pytest.fail("\n".join(msg_lines))


# =============================================================================
# Scanner self-coverage
# =============================================================================


def _scan_fragment(tmp_path: pathlib.Path, name: str, source: str) -> list[tuple[int, str]]:
    """Helper: write ``source`` to ``tmp_path / name`` and return unguarded."""

    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return _scan_file(path)


def test_scanner_flags_bare_asyncio_run_in_sync_test(tmp_path: pathlib.Path) -> None:
    """Baseline: ``asyncio.run`` in a sync ``def test_*`` body is a violation."""

    source = "import asyncio\n\ndef test_x():\n    asyncio.run(some_coro())\n"
    unguarded = _scan_fragment(tmp_path, "frag_baseline.py", source)
    assert [(ln, _) for ln, _ in unguarded] and unguarded[0][0] == 4, unguarded


def test_scanner_allows_asyncio_run_in_async_function(tmp_path: pathlib.Path) -> None:
    """``asyncio.run`` inside an async function is allowed — that helper
    won't be invoked from a sync test directly (and if it is, it'd be via
    ``await``, not via the closed-loop path)."""

    source = "import asyncio\n\nasync def helper():\n    return asyncio.run(some_coro())\n"
    assert _scan_fragment(tmp_path, "frag_async.py", source) == []


def test_scanner_catches_aliased_asyncio_import(tmp_path: pathlib.Path) -> None:
    """``import asyncio as aio; aio.run(...)`` must also be flagged."""

    source = "import asyncio as aio\n\ndef test_x():\n    aio.run(some_coro())\n"
    unguarded = _scan_fragment(tmp_path, "frag_alias.py", source)
    assert unguarded and unguarded[0][0] == 4, unguarded


def test_scanner_accepts_explicit_loop_pattern(tmp_path: pathlib.Path) -> None:
    """The canonical migration target — ``new_event_loop`` +
    ``run_until_complete`` + ``close`` — must NOT be flagged."""

    source = (
        "import asyncio\n"
        "\n"
        "def test_x():\n"
        "    loop = asyncio.new_event_loop()\n"
        "    try:\n"
        "        loop.run_until_complete(some_coro())\n"
        "    finally:\n"
        "        loop.close()\n"
    )
    assert _scan_fragment(tmp_path, "frag_explicit_loop.py", source) == []


def test_scanner_skips_unparseable_files(tmp_path: pathlib.Path) -> None:
    """A broken test file should not crash the lint."""

    broken = tmp_path / "frag_broken.py"
    broken.write_text("def test_x(:\n    asyncio.run(\n", encoding="utf-8")
    assert _scan_file(broken) == []


def test_scanner_flags_nested_sync_def_inside_async_module_top_level(
    tmp_path: pathlib.Path,
) -> None:
    """A sync ``def`` defined at module top level (even alongside async
    helpers) is sync context — its ``asyncio.run`` is flagged."""

    source = (
        "import asyncio\n"
        "\n"
        "async def setup_helper():\n"
        "    return None\n"
        "\n"
        "def test_x():\n"
        "    asyncio.run(setup_helper())\n"
    )
    unguarded = _scan_fragment(tmp_path, "frag_mixed.py", source)
    assert unguarded and unguarded[0][0] == 7, unguarded
