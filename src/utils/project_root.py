"""Walk-up-to-marker project root resolver (Tier-0 Block 6A polish).

Replaces ad-hoc ``Path(__file__).resolve().parents[N]`` calls scattered
across the codebase with a single deterministic helper that walks up to
the canonical marker file (``pyproject.toml``).

Why this exists
---------------
``Path(__file__).resolve().parents[5]`` is brittle: any move of the
caller into a sibling directory silently shifts the resolution by one
or two levels and the caller now references the wrong root. The
walk-up-to-marker idiom is robust to such moves because it terminates
at a structural anchor, not at a depth count. The same idiom already
existed at ``src/skills/loader.py:_find_project_root``; this module
generalises it for shared use.

Optional override
-----------------
Callers that need to point at a different repo (for example tests using
``tmp_path`` to stage a synthetic project tree) can set the
``E2I_CONFIG_DIR`` environment variable. When set, its parent directory
is treated as the project root, mirroring the convention in
``ObservabilityConfig.from_yaml`` where the canonical config directory
sits one level beneath the root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Canonical marker file. ``pyproject.toml`` is checked into the repo root
# and is unlikely to appear higher up in any reasonable filesystem layout.
_MARKER_FILENAMES: tuple[str, ...] = ("pyproject.toml",)

# Environment variable that, when set, forces the project root to be the
# parent of the named directory. Used by tests staging a synthetic tree.
_ENV_OVERRIDE = "E2I_CONFIG_DIR"


class ProjectRootNotFoundError(RuntimeError):
    """Raised when no marker file can be found by walking upward."""


def find_project_root(start: Optional[Path] = None) -> Path:
    """Resolve the project root by walking up from ``start`` to a marker.

    Args:
        start: Path to begin the walk from. Defaults to this module's own
            location, which yields the package root containing this util.
            Tests typically pass an explicit ``tmp_path`` here.

    Returns:
        The first ancestor directory that contains a marker file from
        :data:`_MARKER_FILENAMES`. When :data:`_ENV_OVERRIDE` is set, its
        parent directory is returned instead (after a basic existence
        check).

    Raises:
        ProjectRootNotFoundError: when neither the env-var override nor
            the marker walk yields a usable root. The error message
            names the marker filenames considered so the caller can
            diagnose unfamiliar layouts.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        # ``E2I_CONFIG_DIR`` points at a directory that lives under the
        # repo root (e.g. ``/path/to/repo/config``). The repo root is its
        # parent. We do NOT require the directory to exist — the caller
        # might be using the override to point at a yet-to-be-created
        # config folder under an existing repo — but we DO require its
        # parent to exist, otherwise the override is meaningless.
        override_parent = Path(override).expanduser().resolve().parent
        if not override_parent.exists():
            raise ProjectRootNotFoundError(
                f"{_ENV_OVERRIDE}={override!r} resolves to a parent "
                f"directory that does not exist: {override_parent!s}"
            )
        return override_parent

    anchor = (start if start is not None else Path(__file__)).resolve()
    # When ``start`` is a file, walk from its parent. When it's a
    # directory, start from itself.
    current = anchor.parent if anchor.is_file() else anchor

    while True:
        if any((current / marker).exists() for marker in _MARKER_FILENAMES):
            return current
        if current.parent == current:
            # Reached the filesystem root without finding the marker.
            break
        current = current.parent

    raise ProjectRootNotFoundError(
        f"Could not locate project root walking up from {anchor!s}; "
        f"no marker file from {_MARKER_FILENAMES!r} found in any ancestor."
    )
