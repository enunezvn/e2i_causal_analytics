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

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Files explicitly exempted from the guard requirement. Empty by default —
# every callsite in src/ MUST be gated. Add entries with a comment justifying
# the carve-out + a tracking issue link.
_GUARDED_BY_DESIGN_EXCEPTIONS: frozenset[str] = frozenset()

# Lexical guard markers. We accept any one of these tokens within the
# preceding window because the gating idiom takes several spellings (some
# files use ``loop and loop.is_running()``, others a try/except around
# ``get_running_loop()``).
_GUARD_TOKENS: tuple[str, ...] = (
    "is_running",
    "get_running_loop",
)
_GUARD_WINDOW_LINES = 8  # codex-confirmed sufficient for all current callsites


def _iter_apply_callsites() -> list[tuple[pathlib.Path, int, str]]:
    """Yield (path, lineno, line) for every ``nest_asyncio.apply()`` call
    in src/ (excluding the test tree)."""

    callsites: list[tuple[pathlib.Path, int, str]] = []
    # Match both ``nest_asyncio.apply(`` and the aliased ``_nest_asyncio.apply(``
    # used by ``src/agents/experiment_designer/graph.py``. Any leading
    # word-char (identifier prefix) is acceptable; we still anchor on
    # ``\.apply\s*\(`` so unrelated identifiers can't drift in.
    apply_pattern = re.compile(r"\bn(?:est_asyncio|est_asyncio_[a-z]*)\.apply\s*\(|\b_?nest_asyncio\.apply\s*\(")
    for path in SRC_ROOT.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            # Skip comments and docstring-style references.
            if stripped.startswith("#"):
                continue
            if apply_pattern.search(line):
                callsites.append((path, lineno, line))
    return callsites


def _has_preceding_guard(path: pathlib.Path, lineno: int) -> bool:
    """Return True if any of ``_GUARD_TOKENS`` appears in the
    ``_GUARD_WINDOW_LINES`` lines preceding ``lineno`` in ``path``."""

    text = path.read_text(encoding="utf-8").splitlines()
    start = max(0, lineno - 1 - _GUARD_WINDOW_LINES)
    window = text[start : lineno - 1]
    joined = "\n".join(window)
    return any(token in joined for token in _GUARD_TOKENS)


def test_no_unconditional_nest_asyncio_apply_in_src() -> None:
    """Every ``nest_asyncio.apply()`` call in src/ must be preceded by an
    ``is_running()`` / ``get_running_loop()`` guard within
    ``_GUARD_WINDOW_LINES`` lines.

    Issue #218: prevents the entire bug class (eager / module-level /
    unconditional apply pollutes every downstream ``asyncio.run`` on the
    same process).
    """

    callsites = _iter_apply_callsites()
    assert callsites, (
        "expected at least one nest_asyncio.apply() callsite in src/ — "
        "regex pattern drift?"
    )

    violations: list[tuple[pathlib.Path, int, str]] = []
    for path, lineno, line in callsites:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _GUARDED_BY_DESIGN_EXCEPTIONS:
            continue
        if _has_preceding_guard(path, lineno):
            continue
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
            "Wrap the call in ``if loop and loop.is_running():`` (or equivalent "
            "``get_running_loop()`` guard) and call apply() only inside that "
            "branch. See src/agents/tool_composer/composer.py for the canonical "
            "pattern."
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
    # ⇒ 11 total. If the count changes, audit the new callsite(s) for
    # proper ``is_running`` / ``get_running_loop`` gating, then update
    # the pin below in the same PR.
    expected = 11
    assert len(callsites) == expected, (
        f"nest_asyncio.apply() callsite count drifted in src/ "
        f"(expected {expected}, found {len(callsites)}). Audit each new "
        f"callsite for an is_running() / get_running_loop() guard and "
        f"update the pin if intentional. Callsites:\n"
        + "\n".join(
            f"  {p.relative_to(REPO_ROOT).as_posix()}:{ln}" for p, ln, _ in callsites
        )
    )
