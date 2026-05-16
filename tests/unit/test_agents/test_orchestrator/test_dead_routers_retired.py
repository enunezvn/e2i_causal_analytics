"""Forcing-function tests for issue #256 — dead orchestrator routers retired (Option A).

These tests ensure the two abandoned router iterations stay deleted and that
nothing in the tree imports them again. They are deliberately filesystem +
regex based so the assertions trip even on a `git checkout` that re-adds the
files without re-running the broader test suite.

The active routing path is ``src/agents/orchestrator/nodes/router.py`` —
different module path, different package. These tests only verify the
package-level orphans are gone.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEAD_PATHS = (
    REPO_ROOT / "src" / "agents" / "orchestrator" / "router.py",
    REPO_ROOT / "src" / "agents" / "orchestrator" / "router_v42.py",
)


def test_router_py_retired() -> None:
    """src/agents/orchestrator/router.py was retired per #256 Option A."""
    p = DEAD_PATHS[0]
    assert not p.exists(), (
        f"{p} should be deleted per issue #256 (Option A — lean). "
        "The active routing path is src/agents/orchestrator/nodes/router.py."
    )


def test_router_v42_py_retired() -> None:
    """src/agents/orchestrator/router_v42.py was retired per #256 Option A."""
    p = DEAD_PATHS[1]
    assert not p.exists(), (
        f"{p} should be deleted per issue #256 (Option A — lean). "
        "The MultiFacetedDetector patterns are kept inline in "
        "``INTENT_PATTERNS['multi_faceted']`` within "
        "src/agents/orchestrator/nodes/intent_classifier.py."
    )


def test_no_imports_of_dead_routers() -> None:
    """No Python file imports from the retired router modules.

    The active routing path lives at src/agents/orchestrator/nodes/router.py
    and is imported as ``from .router`` from inside that subpackage —
    that import remains valid because ``.router`` there resolves to
    ``nodes/router.py``, not the deleted ``orchestrator/router.py``.
    """
    dead_import_pattern = re.compile(
        r"(?:^|\s)"
        r"(?:"
        # `from src.agents.orchestrator.router(_v42) import ...`
        r"from\s+src\.agents\.orchestrator\.router(?:_v42)?\s+import"
        # `import src.agents.orchestrator.router(_v42)` (incl. `as alias`)
        r"|import\s+src\.agents\.orchestrator\.router(?:_v42)?(?:\s|$|\.)"
        # `from src.agents.orchestrator import router(_v42)`
        # (parent-package re-import shape; catches `import router, router_v42`)
        r"|from\s+src\.agents\.orchestrator\s+import\s+[^\n#]*\brouter(?:_v42)?\b"
        r")",
        re.MULTILINE,
    )
    scan_roots = (
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "scripts",
    )
    offenders: list[tuple[pathlib.Path, int, str]] = []
    for root in scan_roots:
        if not root.exists():
            continue
        for py_path in root.rglob("*.py"):
            # Skip this test file itself (it references the modules by name
            # in docstrings/comments, not via import).
            if py_path == pathlib.Path(__file__):
                continue
            try:
                contents = py_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in dead_import_pattern.finditer(contents):
                line_no = contents[: match.start()].count("\n") + 1
                offenders.append((py_path, line_no, match.group(0).strip()))

    assert not offenders, "Dead-router imports detected:\n" + "\n".join(
        f"  {p}:{n}  {snippet!r}" for p, n, snippet in offenders
    )
