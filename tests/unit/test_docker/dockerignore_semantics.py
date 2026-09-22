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


def _translate(pattern: str) -> str:
    """Translate one `.dockerignore` glob into a Python regex, TOKEN BY TOKEN.

    Walking the pattern (rather than `re.escape`-then-substitute, the previous
    approach) is what lets a bracket expression survive as a REGEX CHARACTER
    CLASS instead of being escaped into a literal `\\[cod\\]`. Docker's matcher is
    built on Go's `filepath.Match`, and the real `.dockerignore` here has
    `*.py[cod]` (line 19) — `filepath.Match`'s `[abc]` / `[^abc]` / `[a-z]` syntax
    is the same as POSIX/regex character classes (including `^`-negation), so a
    well-formed bracket expression is copied through near-verbatim.

    Token rules, in priority order:
      * `**/` — zero or more WHOLE path segments (moby's compiler); `(?:.*/)?`.
      * `**` (not followed by `/`) — matches anything, including `/`; `.*`.
      * `*` — one path segment, never crosses `/`; `[^/]*`.
      * `?` — one character, never a `/`; `[^/]`.
      * `[...]` (well-formed: closes with a `]`) — copied through as a regex
        character class. An unterminated `[` (no closing `]`) is Go's own
        fallback case — treated as a literal `[`, not a class.
      * anything else — a literal character, `re.escape`d individually.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            # A literal `]` may immediately follow `[` or `[^` without closing the
            # class (both Go and POSIX/regex convention) — skip past it before
            # searching for the real closing bracket.
            j = i + 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            end = pattern.find("]", j)
            if end == -1:
                out.append(re.escape(pattern[i]))
                i += 1
            else:
                out.append(f"[{pattern[i + 1 : end]}]")
                i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


def matches(pattern: str, rel_path: str) -> bool:
    """True when a .dockerignore `pattern` matches `rel_path` or a parent dir."""
    pattern = pattern.rstrip("/")
    if not pattern:
        return False
    regex = _translate(pattern)
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
