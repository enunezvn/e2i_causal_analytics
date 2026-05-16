"""Issue #256: pre-deletion confirmation tests for dead orchestrator routers.

These tests assert that ``src/agents/orchestrator/router.py`` and
``src/agents/orchestrator/router_v42.py`` are abandoned earlier iterations
with no live imports anywhere in the codebase. They MUST pass before the
files are deleted (acceptance §1 of the issue).

This file lives at ``tests/unit/test_agents/test_orchestrator/`` so the test
runner picks it up via the existing collection paths.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SEARCH_PATHS = ["src", "tests", "scripts"]


def _grep_python(pattern: str) -> list[str]:
    """Return matching lines from a grep across SEARCH_PATHS, excluding
    self-matches in dead-router files and the nodes/router.py active path.

    Uses ``rg`` if available, else falls back to a Python walk so we don't
    add a system dependency.
    """
    matches: list[str] = []
    for sp in SEARCH_PATHS:
        root = REPO_ROOT / sp
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            # Skip the dead-router files themselves and the active
            # nodes/router.py path.
            relparts = py.relative_to(REPO_ROOT).parts
            if relparts[-1] in ("router.py", "router_v42.py") and "nodes" not in relparts:
                # The dead routers — skip self-matches.
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                # Skip docstring-only references (lines that look like
                # path annotations in comments / docstrings) — match
                # only actual ``import`` / ``from ... import`` lines.
                stripped = line.lstrip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                if re.search(pattern, line):
                    matches.append(f"{py.relative_to(REPO_ROOT)}:{i}: {line.strip()}")
    return matches


def test_no_imports_of_orchestrator_router_top_level() -> None:
    """No live imports of src.agents.orchestrator.router (the dead file)."""
    # Match: ``from src.agents.orchestrator.router import ...`` or
    # ``from src.agents.orchestrator import router``. Exclude
    # ``orchestrator.nodes.router`` (active) and ``orchestrator.router_v42``.
    pattern = (
        r"from\s+src\.agents\.orchestrator\.router\s+import"
        r"|from\s+src\.agents\.orchestrator\s+import\s+router(?!_v42|\.nodes)\b"
    )
    matches = _grep_python(pattern)
    assert matches == [], f"Dead router still imported: {matches}"


def test_no_imports_of_orchestrator_router_v42() -> None:
    """No live imports of src.agents.orchestrator.router_v42 (the dead file)."""
    pattern = (
        r"from\s+src\.agents\.orchestrator\.router_v42\s+import"
        r"|from\s+src\.agents\.orchestrator\s+import\s+router_v42\b"
        r"|import\s+src\.agents\.orchestrator\.router_v42\b"
    )
    matches = _grep_python(pattern)
    assert matches == [], f"Dead router_v42 still imported: {matches}"
