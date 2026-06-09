"""Static AST guard for the two Supabase-client misuse patterns fixed in
issue #821 (same root cause as PR #820's experiment-monitor fix).

Both patterns raise ``TypeError`` at runtime that the surrounding ``except``
swallows, masking a crash as an empty/skipped/fallback result:

1. ``await get_supabase_client()`` — ``get_supabase_client`` is a SYNC factory
   (``def`` at src/memory/services/factories.py); awaiting it raises
   ``TypeError: object Client can't be used in 'await' expression``. The async
   counterpart is ``get_async_supabase_client()``. The legitimate
   ``await _maybe_await(get_supabase_client())`` adapter form is NOT flagged
   (the await operand is the adapter call, not the factory call).

2. ``SomeRepository(client=...)`` — ``BaseRepository.__init__`` takes
   ``supabase_client=`` (it sets ``self.client = supabase_client``); passing
   ``client=`` raises ``TypeError: __init__() got an unexpected keyword
   argument 'client'`` → the repo helper's ``except`` returns ``None`` →
   silent fallback to mock/sample data.

This guard runs in CI by default (no DB needed) and pins the regression to
zero so the latent bug cannot reappear. Faithful real-DB behavioral proof of
the fix lives in tests/integration/test_async_supabase_client_realdb.py
(opt-in, E2I_DB_INTEGRATION=1).
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _iter_src_files() -> list[pathlib.Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(REPO_ROOT))


class _AwaitSyncFactoryVisitor(ast.NodeVisitor):
    """Records ``await get_supabase_client(...)`` callsites.

    Matches when the direct operand of an ``await`` is a ``Call`` to a
    ``get_supabase_client`` name or attribute. Does NOT match
    ``await _maybe_await(get_supabase_client())`` (operand is ``_maybe_await``)
    nor ``await get_async_supabase_client()`` (different name).
    """

    def __init__(self) -> None:
        self.hits: list[int] = []

    def visit_Await(self, node: ast.Await) -> None:
        value = node.value
        if isinstance(value, ast.Call):
            func = value.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "get_supabase_client":
                self.hits.append(node.lineno)
        self.generic_visit(node)


class _RepositoryClientKwargVisitor(ast.NodeVisitor):
    """Records ``*Repository(client=...)`` constructions (wrong kwarg)."""

    def __init__(self) -> None:
        self.hits: list[int] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name and name.endswith("Repository"):
            if any(kw.arg == "client" for kw in node.keywords):
                self.hits.append(node.lineno)
        self.generic_visit(node)


def _scan(visitor_cls: type[ast.NodeVisitor]) -> dict[str, list[int]]:
    violations: dict[str, list[int]] = {}
    for path in _iter_src_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        visitor = visitor_cls()
        visitor.visit(tree)
        if visitor.hits:  # type: ignore[attr-defined]
            violations[_rel(path)] = visitor.hits  # type: ignore[attr-defined]
    return violations


def test_no_await_on_sync_get_supabase_client() -> None:
    """No code in src/ may ``await`` the SYNC ``get_supabase_client()``.

    Use ``await get_async_supabase_client()`` in async contexts, or the
    ``await _maybe_await(get_supabase_client())`` adapter where a path must
    accept either client.
    """
    violations = _scan(_AwaitSyncFactoryVisitor)
    assert not violations, (
        "Found `await get_supabase_client()` (awaiting the SYNC factory) — "
        "swap to `await get_async_supabase_client()`:\n"
        + "\n".join(f"  {f}: lines {lines}" for f, lines in violations.items())
    )


def test_no_repository_constructed_with_client_kwarg() -> None:
    """No ``*Repository`` may be constructed with ``client=`` — the base
    constructor takes ``supabase_client=``; ``client=`` raises TypeError →
    silent None → mock/sample fallback."""
    violations = _scan(_RepositoryClientKwargVisitor)
    assert not violations, (
        "Found `*Repository(client=...)` (wrong kwarg) — use `supabase_client=`:\n"
        + "\n".join(f"  {f}: lines {lines}" for f, lines in violations.items())
    )
