"""Walk-up-to-marker project root resolver (Tier-0 Block 6A polish).

Replaces ad-hoc ``Path(__file__).resolve().parents[N]`` calls scattered
across the codebase with a single deterministic helper that walks up to
a canonical marker file (``pyproject.toml`` by default).

Why this exists
---------------
``Path(__file__).resolve().parents[5]`` is brittle: any move of the
caller into a sibling directory silently shifts the resolution by one
or two levels and the caller now references the wrong root. The
walk-up-to-marker idiom is robust to such moves because it terminates
at a structural anchor, not at a depth count. The same idiom already
existed at ``src/skills/loader.py:_find_project_root``; this module
generalises it for shared use.

Multi-marker support
--------------------
Some callers anchor on more than one marker (e.g. the skills loader
also accepts ``.claude/`` as a valid project anchor when a checkout
predates ``pyproject.toml`` adoption). Pass ``markers=(...)`` to opt
into a custom marker set; the walk matches on ANY marker present in
the candidate ancestor.

Optional override
-----------------
Callers that need to point at a different repo (for example tests using
``tmp_path`` to stage a synthetic project tree) can set the
``E2I_CONFIG_DIR`` environment variable. When set, its parent directory
is treated as the project root, mirroring the convention in
``ObservabilityConfig.from_yaml`` where the canonical config directory
sits one level beneath the root. The override is itself validated
against the marker set: if the resolved parent contains no marker, an
actionable :class:`ProjectRootNotFoundError` is raised that names the
env var and the resolved path so the developer can fix the mistake
without a multi-step guess.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

# Canonical marker file. ``pyproject.toml`` is checked into the repo root
# and is unlikely to appear higher up in any reasonable filesystem layout.
_DEFAULT_MARKERS: tuple[str, ...] = ("pyproject.toml",)

# Environment variable that, when set, forces the project root to be the
# parent of the named directory. Used by tests staging a synthetic tree.
_ENV_OVERRIDE = "E2I_CONFIG_DIR"


class ProjectRootNotFoundError(RuntimeError):
    """Raised when no marker file can be found by walking upward."""


def _has_any_marker(directory: Path, markers: Sequence[str]) -> bool:
    """Return True if ``directory`` contains at least one of ``markers``."""
    return any((directory / marker).exists() for marker in markers)


def _walk_for_marker(start: Path, markers: Sequence[str]) -> Optional[Path]:
    """Walk upward from ``start`` looking for any marker.

    Returns the first ancestor (inclusive of ``start`` if it is a
    directory) that contains any marker, or ``None`` if the walk reaches
    the filesystem root without a hit.
    """
    current = start
    while True:
        if _has_any_marker(current, markers):
            return current
        if current.parent == current:
            return None
        current = current.parent


def find_project_root(
    start: Optional[Path] = None,
    *,
    markers: Sequence[str] = _DEFAULT_MARKERS,
) -> Path:
    """Resolve the project root by walking up from ``start`` to a marker.

    Args:
        start: Path to begin the walk from. Defaults to this module's own
            location, which yields the package root containing this util.
            Tests typically pass an explicit ``tmp_path`` here.
        markers: Sequence of filenames/dirnames that mark a project root.
            Matches if ANY of these exist in a candidate ancestor. Defaults
            to ``("pyproject.toml",)``. Callers like the skills loader
            pass ``("pyproject.toml", ".claude")`` to accept either anchor.

    Returns:
        The first ancestor directory that contains at least one of
        ``markers``. When :data:`_ENV_OVERRIDE` is set, its parent
        directory is returned instead — but only after validation that
        the parent itself (or one of its ancestors) contains a marker;
        the env var must point inside an actual project tree.

    Raises:
        ProjectRootNotFoundError: when neither the env-var override nor
            the marker walk yields a usable root. The error message
            names the marker set considered (and, for env-override
            failures, the env var name and resolved path) so the caller
            can diagnose unfamiliar layouts without a multi-step guess.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        # ``E2I_CONFIG_DIR`` points at a directory that lives under the
        # repo root (e.g. ``/path/to/repo/config``). The repo root is its
        # parent. We do NOT require the named directory itself to exist —
        # the caller might be using the override to point at a yet-to-
        # be-created config folder under an existing repo — but we DO
        # require its parent to exist AND to live inside a real project
        # tree (i.e. a marker resolves from there). Otherwise the
        # override would silently consume a wrong root and the
        # downstream config loader would later fail with an opaque
        # ``FileNotFoundError`` that does not name ``E2I_CONFIG_DIR``.
        override_parent = Path(override).expanduser().resolve().parent
        if not override_parent.exists():
            raise ProjectRootNotFoundError(
                f"{_ENV_OVERRIDE}={override!r} resolves to a parent "
                f"directory that does not exist: {override_parent!s}"
            )
        if _walk_for_marker(override_parent, markers) is None:
            raise ProjectRootNotFoundError(
                f"{_ENV_OVERRIDE}={override!r} resolved to "
                f"{override_parent!s} but no project marker "
                f"(any of {tuple(markers)!r}) was found there or in "
                "any ancestor. Point the env var at a directory whose "
                "parent contains a marker, or unset it to fall back to "
                "the marker walk."
            )
        return override_parent

    anchor = (start if start is not None else Path(__file__)).resolve()
    # When ``start`` is a file, walk from its parent. When it's a
    # directory, start from itself.
    walk_start = anchor.parent if anchor.is_file() else anchor

    found = _walk_for_marker(walk_start, markers)
    if found is not None:
        return found

    raise ProjectRootNotFoundError(
        f"Could not locate project root walking up from {anchor!s}; "
        f"no marker from {tuple(markers)!r} found in any ancestor."
    )
