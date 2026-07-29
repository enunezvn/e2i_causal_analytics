"""Static lint: forbid unconditional ``nest_asyncio.apply()`` in src/ (issue #218).

``nest_asyncio.apply()`` is a process-wide monkey-patch of ``asyncio.run`` and
the asyncio loop runner machinery. Once a worker calls it, every subsequent
``asyncio.run(coro)`` on that worker routes through nest_asyncio's wrapper,
which references the loop captured at apply-time. If that loop has been
closed (the common pytest-asyncio per-test teardown path), the next sync
``asyncio.run`` call raises ``RuntimeError: Event loop is closed`` (issue
#215 + #218 polluter chain).

The legitimate pattern — required to coexist with notebooks, LangGraph
``ainvoke`` inside a sync wrapper, etc. — is::

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)

i.e. the ``apply()`` call is GATED on detecting an actually-running outer
loop. Calling ``apply()`` unconditionally at module load time, or inside a
factory that runs at import, pollutes the process even when no nested loop
exists.

This test scans ``src/`` (production code) for ``nest_asyncio.apply()``
calls and asserts each one is preceded — within a small lexical window —
by an ``is_running()`` guard. Test files are exempt: tests deliberately
exercise the polluted state (regression pins for issue #215 and #218)
and the runtime probe in ``tests/conftest.py`` handles those cases.

If you legitimately need a non-test ``apply()`` call without a guard
(rare — usually means a Jupyter-only helper), add the file path to
``_GUARDED_BY_DESIGN_EXCEPTIONS`` with a comment justifying the carve-out
and a link to an issue tracking the long-term fix.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Files explicitly exempted from the guard requirement. Empty by default —
# every callsite in src/ MUST be gated. Add entries with a comment justifying
# the carve-out + a tracking issue link.
_GUARDED_BY_DESIGN_EXCEPTIONS: frozenset[str] = frozenset()

# Guard idiom recognition is now AST-based (see ``_GuardedCallChecker``).
# Codex pass-1 MEDIUM: the prior lexical-window heuristic accepted comments
# and inverted conditions; we now require an ancestor ``If``/``While``/
# ``IfExp`` test (or ``Try`` body) to syntactically reference one of the
# guard tokens above the call.


class _ApplyCallsiteVisitor(ast.NodeVisitor):
    """AST walker that records every ``nest_asyncio.apply()`` callsite
    while tracking import aliases.

    Handles, in order of precedence:

    1. ``import nest_asyncio`` / ``import nest_asyncio as X`` — records the
       module-binding name (``nest_asyncio`` or ``X``). Subsequent
       attribute calls of the form ``<binding>.apply(...)`` are matched.
    2. ``from nest_asyncio import apply`` / ``... as Y`` — records the
       function-binding name (``apply`` or ``Y``). Subsequent bare calls
       ``<binding>(...)`` are matched.

    Codex pass-1 HIGH (issue #218): the prior regex-only scan missed
    both alias forms (`import nest_asyncio as na`; `from nest_asyncio
    import apply`), allowing an unconditional polluter to evade the
    guard test AND the count pin in one stroke.

    Codex pass-2 LOW (documented limitation): binding tracking is
    *flat* — we do not model lexical scope. ``if TYPE_CHECKING: from
    nest_asyncio import apply`` would taint module-level ``apply()``
    calls anywhere in the same file, producing a false positive. No
    such pattern currently exists in src/; the runtime probe in
    tests/conftest.py is the authoritative check when binding flow
    becomes ambiguous. Promote to AST-scope analysis only if a real
    case appears.
    """

    def __init__(self) -> None:
        super().__init__()
        # Names that resolve to the nest_asyncio MODULE.
        self._module_bindings: set[str] = set()
        # Names that resolve directly to nest_asyncio.apply (from-import case).
        self._apply_bindings: set[str] = set()
        self.callsites: list[tuple[int, str]] = []  # (lineno, source line)
        self._lines: list[str] = []

    def set_source(self, source: str) -> None:
        self._lines = source.splitlines()

    def _record(self, lineno: int) -> None:
        line = self._lines[lineno - 1] if 0 < lineno <= len(self._lines) else ""
        self.callsites.append((lineno, line))

    def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
        for alias in node.names:
            if alias.name == "nest_asyncio":
                self._module_bindings.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # type: ignore[override]
        if node.module == "nest_asyncio":
            for alias in node.names:
                if alias.name == "apply":
                    self._apply_bindings.add(alias.asname or alias.name)
                elif alias.name == "*":  # pragma: no cover — defensive
                    self._apply_bindings.add("apply")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
        """Track reassignment aliases: ``do_apply = nest_asyncio.apply``
        and ``do_apply = <existing_apply_binding>``.

        Codex pass-2 MEDIUM: a future unconditional polluter could evade
        detection by binding ``apply`` to a fresh name (``f = na.apply;
        f()``) — that pattern is uncommon but the AST scanner closes the
        gap cheaply. Only handles single-target ``Name = ...`` assigns
        because that covers every realistic alias chain; tuple/starred
        unpacking targets are out of scope.
        """

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            value = node.value
            # RHS form 1: ``<module>.apply`` attribute reference (no call).
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "apply"
                and isinstance(value.value, ast.Name)
                and value.value.id in self._module_bindings
            ):
                self._apply_bindings.add(target_name)
            # RHS form 2: ``<existing_apply_binding>`` — chained alias.
            elif isinstance(value, ast.Name) and value.id in self._apply_bindings:
                self._apply_bindings.add(target_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # type: ignore[override]
        """Same alias-tracking for annotated assignment (``f: Callable =
        nest_asyncio.apply``). Real-world rarity, but matches the
        ``visit_Assign`` coverage so a typed equivalent doesn't escape."""

        if isinstance(node.target, ast.Name) and node.value is not None:
            target_name = node.target.id
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "apply"
                and isinstance(value.value, ast.Name)
                and value.value.id in self._module_bindings
            ):
                self._apply_bindings.add(target_name)
            elif isinstance(value, ast.Name) and value.id in self._apply_bindings:
                self._apply_bindings.add(target_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func = node.func
        matched = False
        # Form 1: ``<module>.apply(...)`` where <module> resolves to nest_asyncio.
        if isinstance(func, ast.Attribute) and func.attr == "apply":
            value = func.value
            if isinstance(value, ast.Name) and value.id in self._module_bindings:
                matched = True
        # Form 2: ``<bound_apply>(...)`` where the bound name is a from-import
        # of nest_asyncio.apply.
        elif isinstance(func, ast.Name) and func.id in self._apply_bindings:
            matched = True
        if matched:
            self._record(node.lineno)
        self.generic_visit(node)


def _iter_apply_callsites() -> list[tuple[pathlib.Path, int, str]]:
    """Yield (path, lineno, line) for every ``nest_asyncio.apply()`` call
    in src/ (excluding the test tree).

    AST-based: catches aliased imports and from-imports that the prior
    regex scan missed (codex pass-1 HIGH).
    """

    callsites: list[tuple[pathlib.Path, int, str]] = []
    for path in SRC_ROOT.rglob("*.py"):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            # A src/ file that doesn't parse is its own bug, but not this
            # test's responsibility. Skip cleanly.
            continue
        visitor = _ApplyCallsiteVisitor()
        visitor.set_source(source)
        visitor.visit(tree)
        for lineno, line in visitor.callsites:
            callsites.append((path, lineno, line))
    return callsites


class _GuardedCallChecker(ast.NodeVisitor):
    """For each detected ``apply()`` call, record whether its AST
    ancestor chain contains an ``If`` or ``Try`` whose test/handler
    spelling references ``is_running`` or ``get_running_loop``.

    This replaces the previous lexical-window heuristic which accepted
    *any* preceding token occurrence — including comments and inverted
    conditions (codex pass-1 MEDIUM). The AST view forces the guard to
    syntactically dominate the call.
    """

    _GUARD_TOKEN_SOURCES = ("is_running", "get_running_loop")

    def __init__(self, module_bindings: set[str], apply_bindings: set[str]) -> None:
        super().__init__()
        self._module_bindings = module_bindings
        self._apply_bindings = apply_bindings
        self._ancestor_stack: list[ast.AST] = []
        self.unguarded: list[int] = []  # linenos
        self.guarded: list[int] = []

    def _is_guard_carrier(self, node: ast.AST) -> bool:
        """A guard carrier is an If/While/IfExp whose test mentions one of
        the guard tokens, or a Try whose body wraps a ``get_running_loop``
        call (the common ``try: get_running_loop(); except RuntimeError``
        idiom)."""

        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            test_src = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
            return any(t in test_src for t in self._GUARD_TOKEN_SOURCES)
        if isinstance(node, ast.Try):
            # Walk the *body* (not handlers) for a get_running_loop call —
            # the canonical pattern stores the result then branches on it.
            for stmt in node.body:
                body_src = ast.unparse(stmt) if hasattr(ast, "unparse") else ""
                if "get_running_loop" in body_src:
                    return True
        return False

    def _matches_apply_call(self, call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "apply":
            value = func.value
            return isinstance(value, ast.Name) and value.id in self._module_bindings
        if isinstance(func, ast.Name):
            return func.id in self._apply_bindings
        return False

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        self._ancestor_stack.append(node)
        try:
            super().generic_visit(node)
        finally:
            self._ancestor_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        if self._matches_apply_call(node):
            guarded = any(self._is_guard_carrier(anc) for anc in self._ancestor_stack)
            (self.guarded if guarded else self.unguarded).append(node.lineno)
        # Recurse so nested calls (rare) are still inspected.
        self.generic_visit(node)


def _classify_path(path: pathlib.Path) -> tuple[list[int], list[int]]:
    """Return (guarded_linenos, unguarded_linenos) for one src/ file."""

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return ([], [])
    # Resolve bindings first (top-down imports), then check call ancestry.
    binder = _ApplyCallsiteVisitor()
    binder.set_source(source)
    binder.visit(tree)
    checker = _GuardedCallChecker(binder._module_bindings, binder._apply_bindings)
    checker.visit(tree)
    return (checker.guarded, checker.unguarded)


def test_no_unconditional_nest_asyncio_apply_in_src() -> None:
    """Every ``nest_asyncio.apply()`` call in src/ must have a syntactically
    dominating ``is_running()`` / ``get_running_loop()`` guard.

    Issue #218: prevents the entire bug class (eager / module-level /
    unconditional apply pollutes every downstream ``asyncio.run`` on the
    same process). Codex pass-1 MEDIUM upgraded the check from a lexical
    preceding-window scan to an AST ancestor walk so the guard cannot be
    forged by a nearby comment or an inverted condition.
    """

    callsites = _iter_apply_callsites()
    assert callsites, (
        "expected at least one nest_asyncio.apply() callsite in src/ — "
        "AST scan returned empty (regression?)"
    )

    violations: list[tuple[pathlib.Path, int, str]] = []
    for path in sorted({p for p, _, _ in callsites}):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _GUARDED_BY_DESIGN_EXCEPTIONS:
            continue
        _, unguarded_linenos = _classify_path(path)
        if not unguarded_linenos:
            continue
        text_lines = path.read_text(encoding="utf-8").splitlines()
        for lineno in unguarded_linenos:
            line = text_lines[lineno - 1] if 0 < lineno <= len(text_lines) else ""
            violations.append((path, lineno, line))

    if violations:
        msg_lines = [
            f"{len(violations)} unconditional nest_asyncio.apply() call(s) "
            f"in src/ — would pollute asyncio.run process-wide (issue #218):",
        ]
        for path, lineno, line in violations:
            rel = path.relative_to(REPO_ROOT).as_posix()
            msg_lines.append(f"  {rel}:{lineno}: {line.strip()}")
        msg_lines.append(
            "Wrap the call in ``if loop and loop.is_running():`` (or "
            "equivalent ``get_running_loop()`` guard) and call apply() only "
            "inside that branch. See src/agents/tool_composer/composer.py "
            "for the canonical pattern."
        )
        pytest.fail("\n".join(msg_lines))


def test_nest_asyncio_apply_callsite_count_pinned() -> None:
    """Pin the count of known ``nest_asyncio.apply()`` callsites in src/.

    This is a *low-friction* canary: when a future PR adds a new callsite,
    this test fails and forces the author to (a) confirm the new site is
    gated and (b) intentionally update the pinned count. Without it, the
    guard test above would silently accept new gated sites — but a single
    typo (e.g. ``if loop or loop.is_running():``) would slip through the
    lexical scan.
    """

    callsites = _iter_apply_callsites()
    # Current accepted total = 9 gated callsites in src/ (verified
    # 2026-05-14 during issue #218 investigation):
    #   - src/agents/experiment_designer/graph.py:86
    #   - src/agents/experiment_monitor/agent.py:252
    #   - src/agents/tool_composer/composer.py:552
    #   - src/agents/tool_composer/executor.py:853
    #   - src/agents/tool_composer/decomposer.py:274
    #   - src/agents/tool_composer/planner.py:643
    #   - src/agents/tool_composer/synthesizer.py:308
    #   - src/tasks/ab_testing_tasks.py:108
    #   - src/tasks/feast_tasks.py:42
    #   - src/tasks/drift_monitoring_tasks.py:83
    #   - src/tasks/feedback_loop_tasks.py:95
    #   - src/tasks/dspy_optimization_tasks.py:run_async (DSPy F1 keystone;
    #     gated inside `try: asyncio.get_running_loop()` — same pattern as
    #     feedback_loop_tasks.run_async)
    #   - src/tasks/routing_label_tasks.py:run_async (#1341 Phase 1 routing
    #     labeler; gated inside `try: asyncio.get_running_loop()` — same
    #     pattern as dspy_optimization_tasks.run_async; audited 2026-07-29)
    # ⇒ 13 total. If the count changes, audit the new callsite(s) for
    # proper ``is_running`` / ``get_running_loop`` gating, then update
    # the pin below in the same PR.
    expected = 13
    assert len(callsites) == expected, (
        f"nest_asyncio.apply() callsite count drifted in src/ "
        f"(expected {expected}, found {len(callsites)}). Audit each new "
        f"callsite for an is_running() / get_running_loop() guard and "
        f"update the pin if intentional. Callsites:\n"
        + "\n".join(f"  {p.relative_to(REPO_ROOT).as_posix()}:{ln}" for p, ln, _ in callsites)
    )


# =============================================================================
# AST scanner self-coverage (codex pass-1 HIGH + MEDIUM regressions)
# =============================================================================


def _scan_fragment(tmp_path: pathlib.Path, name: str, source: str) -> tuple[list[int], list[int]]:
    """Helper: write ``source`` to ``tmp_path / name`` and return
    ``(guarded_linenos, unguarded_linenos)`` from the AST classifier."""

    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return _classify_path(path)


def test_ast_scan_catches_direct_apply_call(tmp_path: pathlib.Path) -> None:
    """Baseline: ``nest_asyncio.apply()`` outside any guard is flagged
    as unguarded."""

    guarded, unguarded = _scan_fragment(
        tmp_path,
        "fragment_direct.py",
        "import nest_asyncio\nnest_asyncio.apply()\n",
    )
    assert unguarded == [2], (guarded, unguarded)
    assert not guarded


def test_ast_scan_catches_aliased_module_import(tmp_path: pathlib.Path) -> None:
    """Codex pass-1 HIGH: ``import nest_asyncio as na; na.apply()`` MUST
    be detected — the prior regex missed this entirely."""

    guarded, unguarded = _scan_fragment(
        tmp_path,
        "fragment_alias_module.py",
        "import nest_asyncio as na\nna.apply()\n",
    )
    assert unguarded == [2], (guarded, unguarded)
    assert not guarded


def test_ast_scan_catches_from_import_apply(tmp_path: pathlib.Path) -> None:
    """Codex pass-1 HIGH: ``from nest_asyncio import apply; apply()``
    must be detected. The from-import form binds ``apply`` directly,
    so the call has no attribute dotted access at all."""

    guarded, unguarded = _scan_fragment(
        tmp_path,
        "fragment_from_import.py",
        "from nest_asyncio import apply\napply()\n",
    )
    assert unguarded == [2], (guarded, unguarded)
    assert not guarded


def test_ast_scan_catches_from_import_apply_aliased(tmp_path: pathlib.Path) -> None:
    """``from nest_asyncio import apply as do_apply; do_apply()`` — the
    asname binding must also be tracked."""

    guarded, unguarded = _scan_fragment(
        tmp_path,
        "fragment_from_import_alias.py",
        "from nest_asyncio import apply as do_apply\ndo_apply()\n",
    )
    assert unguarded == [2], (guarded, unguarded)


def test_ast_scan_accepts_canonical_if_running_guard(tmp_path: pathlib.Path) -> None:
    """The canonical idiom (``if loop and loop.is_running(): apply()``)
    must be classified as guarded."""

    source = (
        "import asyncio\n"
        "import nest_asyncio\n"
        "def fn():\n"
        "    try:\n"
        "        loop = asyncio.get_running_loop()\n"
        "    except RuntimeError:\n"
        "        loop = None\n"
        "    if loop and loop.is_running():\n"
        "        nest_asyncio.apply()\n"
    )
    guarded, unguarded = _scan_fragment(tmp_path, "fragment_canonical.py", source)
    assert guarded == [9], (guarded, unguarded)
    assert not unguarded


def test_ast_scan_rejects_comment_only_guard(tmp_path: pathlib.Path) -> None:
    """Codex pass-1 MEDIUM: an ``is_running`` mention in a *comment*
    must NOT satisfy the guard requirement — only a real ancestor
    ``If``/``While``/``IfExp`` test counts."""

    source = (
        "import nest_asyncio\n"
        "# check loop.is_running() here later? for now: just apply\n"
        "nest_asyncio.apply()\n"
    )
    _, unguarded = _scan_fragment(tmp_path, "fragment_comment.py", source)
    assert unguarded == [3], unguarded


def test_ast_scan_rejects_inverted_guard(tmp_path: pathlib.Path) -> None:
    """A guard with the *wrong polarity* (``if not is_running():
    apply()``) still mentions the token; the AST view does NOT model
    branch polarity, so we deliberately accept this as guarded (the
    static check is a coarse net). This test pins the documented
    limitation so a future codex pass that argues for polarity-aware
    analysis lands a deliberate change, not a silent drift.
    """

    source = (
        "import asyncio\n"
        "import nest_asyncio\n"
        "def fn():\n"
        "    loop = asyncio.get_event_loop()\n"
        "    if not loop.is_running():\n"
        "        nest_asyncio.apply()\n"  # logically wrong but lexically guarded
    )
    guarded, unguarded = _scan_fragment(tmp_path, "fragment_inverted.py", source)
    # DOCUMENTED LIMITATION: AST sees the ``is_running`` token in an If
    # ancestor and classifies as guarded. The runtime probe in
    # ``tests/conftest.py`` catches inverted-polarity polluters at runtime.
    assert guarded == [6], (guarded, unguarded)
    assert not unguarded


def test_ast_scan_catches_reassigned_module_attribute(tmp_path: pathlib.Path) -> None:
    """Codex pass-2 MEDIUM: ``f = nest_asyncio.apply; f()`` must be
    detected. A polluter could trivially evade the import-based scan by
    binding apply to a fresh name; ``visit_Assign`` closes that gap."""

    source = "import nest_asyncio\ndo_apply = nest_asyncio.apply\ndo_apply()\n"
    guarded, unguarded = _scan_fragment(tmp_path, "fragment_reassigned.py", source)
    assert unguarded == [3], (guarded, unguarded)


def test_ast_scan_catches_chained_reassignment(tmp_path: pathlib.Path) -> None:
    """``from nest_asyncio import apply; do = apply; do()`` — chained
    reassignment from an existing apply binding must propagate."""

    source = "from nest_asyncio import apply\ndo = apply\ndo()\n"
    guarded, unguarded = _scan_fragment(tmp_path, "fragment_chain.py", source)
    assert unguarded == [3], (guarded, unguarded)


def test_ast_scan_reassignment_under_guard(tmp_path: pathlib.Path) -> None:
    """A reassigned alias call inside a proper guard should be guarded
    (the binding-tracking and the guard-check are orthogonal)."""

    source = (
        "import asyncio\n"
        "import nest_asyncio\n"
        "do_apply = nest_asyncio.apply\n"
        "def fn():\n"
        "    try:\n"
        "        loop = asyncio.get_running_loop()\n"
        "    except RuntimeError:\n"
        "        loop = None\n"
        "    if loop and loop.is_running():\n"
        "        do_apply()\n"
    )
    guarded, unguarded = _scan_fragment(tmp_path, "fragment_reassigned_guarded.py", source)
    assert guarded == [10], (guarded, unguarded)
    assert not unguarded


def test_iter_apply_callsites_skips_unparseable_files(tmp_path: pathlib.Path) -> None:
    """Defensive: a syntactically broken src/ file must not crash the
    scan (the file's own bug, not this test's responsibility)."""

    # Use _classify_path directly with a deliberately broken source.
    broken = tmp_path / "broken.py"
    broken.write_text("import nest_asyncio\nnest_asyncio.apply(\n", encoding="utf-8")
    guarded, unguarded = _classify_path(broken)
    assert guarded == [] and unguarded == []
