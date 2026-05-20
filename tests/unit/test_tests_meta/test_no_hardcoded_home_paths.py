"""Drift-guard regression test (issue #410).

Asserts that no test file under ``tests/`` re-introduces a hard-coded
absolute developer/user path (``/home/<user>/...`` or ``/Users/<user>/...``)
inside a ``sys.path.insert(...)`` call or in a bare ``Path(...)`` /
``os.path.*`` literal.

Background
----------
``tests/integration/test_tier0_single_mode_snapshot.py`` previously contained::

    sys.path.insert(0, "/home/enunez/Projects/e2i_causal_analytics")

This was a silent footgun in two distinct ways:

* **In CI**: ``/home/enunez/...`` does not exist on GitHub-hosted runners.
  Python's import machinery skips non-existent ``sys.path`` entries and
  the test passes via ``conftest`` / pytest rootdir fallback.
* **In a git worktree**: the hard-coded path exists (it points at the
  MAIN repo, not the worktree). Python imports the main-repo copy of
  the production module, NOT the worktree copy. PRs that modify the
  imported module then see false-negative drift failures locally even
  though their checkout is correct.

This regression test pins ZERO occurrences of the anti-pattern across
``tests/`` so a future contributor cannot reintroduce it without the CI
deliberately failing.

Detection strategy
------------------
The check uses two complementary passes:

1. **AST pass** — parses each ``.py`` file under ``tests/`` and walks
   every ``ast.Call`` node. Flags any string-literal argument matching
   the developer-path prefix regex. Catches the canonical
   ``sys.path.insert(0, "/home/...")`` shape and any equivalent
   ``Path("/home/...")`` / ``os.path.join("/home/...", ...)`` shapes.
2. **Substring pass** — a defence-in-depth grep for the literal prefixes
   in case AST parsing skips a file (e.g., a stray syntax error) or
   the anti-pattern hides inside an f-string. This pass is intentionally
   noisy-but-safe: it only flags ``/home/`` or ``/Users/`` prefixes that
   appear immediately after a quote character (``"`` or ``'``).

Exclusions
----------
This test file is itself excluded from both passes (it has to literally
contain the offending substrings to describe what it forbids). All
other ``tests/...`` files are in scope.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "tests"

# Files that are allowed to mention the literal anti-pattern strings
# (e.g., this regression test and its docstring).
_SELF_PATH = Path(__file__).resolve()

# Developer / user absolute-path prefixes we never want inside test code.
_FORBIDDEN_PREFIXES: Tuple[str, ...] = ("/home/", "/Users/")

# Pre-compiled substring scanner. Matches a quote char (single or double)
# immediately followed by the forbidden prefix — this avoids flagging
# unrelated mentions inside comments / docstrings of OTHER files that
# happen to discuss filesystem layout.
_QUOTED_PREFIX_RE = re.compile(r"""['"](/home/|/Users/)""")


def _is_forbidden_string(value: object) -> bool:
    """Return True if ``value`` is a string starting with a forbidden prefix."""
    if not isinstance(value, str):
        return False
    return any(value.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES)


def _ast_scan_file(path: Path) -> List[str]:
    """AST pass: return a list of human-readable findings for ``path``.

    Walks every ``ast.Call`` and every ``ast.Constant`` argument, flagging
    string literals that start with a forbidden developer-path prefix.
    """
    findings: List[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem read failure
        return [f"{path}: could not read ({exc!r})"]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - syntax error in test
        return [f"{path}: could not parse ({exc!r})"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and _is_forbidden_string(arg.value):
                    findings.append(
                        f"{path}:{node.lineno}: call argument is a hard-coded "
                        f"developer path {arg.value!r}"
                    )
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and _is_forbidden_string(kw.value.value):
                    findings.append(
                        f"{path}:{node.lineno}: keyword argument {kw.arg!r} "
                        f"is a hard-coded developer path {kw.value.value!r}"
                    )

    return findings


def _substring_scan_file(path: Path) -> List[str]:
    """Substring pass: defence-in-depth grep for quoted forbidden prefixes."""
    findings: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - filesystem read failure
        return [f"{path}: could not read ({exc!r})"]

    for lineno, line in enumerate(lines, start=1):
        if _QUOTED_PREFIX_RE.search(line):
            findings.append(f"{path}:{lineno}: quoted forbidden prefix in line: {line.strip()!r}")
    return findings


def _iter_test_files() -> List[Path]:
    """Return all ``.py`` files under ``tests/`` except this test file."""
    return sorted(p for p in TESTS_DIR.rglob("*.py") if p.resolve() != _SELF_PATH)


def test_tests_dir_contains_no_hardcoded_home_paths_ast() -> None:
    """AST pass: no ``ast.Call`` argument under ``tests/`` is a hard-coded path.

    This is the primary check. It catches the canonical anti-pattern
    ``sys.path.insert(0, "/home/...")`` and any structurally equivalent
    shape (``Path("/home/...")``, ``os.path.join("/home/...", ...)``, etc.).
    """
    assert TESTS_DIR.is_dir(), f"tests/ directory not found at {TESTS_DIR!s}"

    findings: List[str] = []
    for test_file in _iter_test_files():
        findings.extend(_ast_scan_file(test_file))

    assert not findings, (
        "Hard-coded developer/user absolute paths detected in tests/ — "
        "derive the repo root via Path(__file__).resolve().parents[N] instead "
        "(see issue #410 for rationale). Findings:\n  - " + "\n  - ".join(findings)
    )


def test_tests_dir_contains_no_hardcoded_home_paths_substring() -> None:
    """Substring pass: no quoted ``/home/`` or ``/Users/`` literal under ``tests/``.

    Defence-in-depth check that also catches f-strings, multi-line
    string literals, and any future hide-the-pattern attempt that the
    AST pass might miss.
    """
    assert TESTS_DIR.is_dir(), f"tests/ directory not found at {TESTS_DIR!s}"

    findings: List[str] = []
    for test_file in _iter_test_files():
        findings.extend(_substring_scan_file(test_file))

    assert not findings, (
        "Quoted developer/user absolute-path prefixes detected in tests/ — "
        "derive the repo root via Path(__file__).resolve().parents[N] instead "
        "(see issue #410 for rationale). Findings:\n  - " + "\n  - ".join(findings)
    )
