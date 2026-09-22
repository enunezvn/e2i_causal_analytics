"""Shared `.dockerignore` matching semantics for the packaging guards.

Extracted after two independent copies of this matcher drifted apart — one for
the KG Layer-2 caches (#1607), one for the Layer-4 classifier artifact (lane 1,
2026-09-22). That is the exact "same algorithm written twice" shape
``tests/unit/test_docker/conftest.py`` already warns about for the deploy.yml
guards: two solutions to the same problem is the signal it belongs in one
place. One copy now, imported by both packaging tests.

DELIBERATE SEMANTICS — DO NOT "FIX" THIS TO GITIGNORE BEHAVIOUR: a slash-less
Docker pattern (``*.json``, ``*.md``, ...) matches at the BUILD-CONTEXT ROOT
ONLY, not at any depth, unlike ``.gitignore``. Measured on the production
image 2026-09-22: ``scripts/benchmarks/routing/data/agent_contracts.json``
survives the blanket ``*.json`` rule in that same image. ``matches`` models
that root-only semantics on purpose; the pinning test
``tests/unit/test_data/test_causal_role_classifier_packaging.py::test_matcher_models_docker_root_only_semantics_for_slashless_patterns``
guards against this drifting back to an any-depth (gitignore-style) matcher.

This is a plain helper module, not a test module (no ``test_`` prefix, so
pytest's ``python_files = ["test_*.py"]`` never collects it) — no tests live
here; they stay with the assertions in each packaging test file.
"""

from __future__ import annotations

import re
from pathlib import Path


def matches(pattern: str, rel_path: str) -> bool:
    """True when a .dockerignore `pattern` matches `rel_path` or a parent dir."""
    pattern = pattern.rstrip("/")
    if not pattern:
        return False
    # `**` spans separators; `*` does not.
    regex = re.escape(pattern).replace(r"\*\*", "\x00").replace(r"\*", "[^/]*")
    regex = regex.replace("\x00", ".*").replace(r"\?", "[^/]")
    if re.fullmatch(regex, rel_path):
        return True
    # A directory pattern also covers everything beneath it.
    return bool(re.fullmatch(regex + "(/.*)?", rel_path))


def dockerignore_excludes(dockerignore_path: Path, rel_path: str) -> bool:
    """Resolve `rel_path` against `dockerignore_path` with last-match-wins semantics.

    Docker evaluates every pattern in order and the LAST match decides, which is
    exactly the rule the #1607 bug turned on: an un-ignore placed next to a
    blanket exclusion reads correctly but can be silently undone by a later
    rule.
    """
    excluded = False
    for raw in dockerignore_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line[1:] if negate else line
        if matches(pattern, rel_path):
            excluded = not negate
    return excluded
