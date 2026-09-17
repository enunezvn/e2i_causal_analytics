"""Prove a module under test was imported from the SAME checkout as the test.

Why this exists
---------------
In a git worktree, a file run by path imports ``src`` from the MAIN checkout via the
editable ``.pth``. A test can then assert confidently about code from a different tree
than the one it lives in -- the failure mode ``test_no_hardcoded_home_paths.py``'s
docstring already describes, and the reason the canonical-TRx lane started asserting
which tree it had loaded at all.

Why it is a function and not an inline assertion
------------------------------------------------
The first implementation of that guard was::

    assert ``.worktrees/<lane>`` in _etl.__file__

which is a **proxy for the capability**, not the capability. The capability is "the
module came from this checkout"; the literal is satisfiable in exactly one directory on
one machine and false everywhere else -- including CI, where
``$GITHUB_WORKSPACE`` is ``/home/runner/work/<repo>/<repo>`` and can never contain that
substring. Both tests carrying it would have failed on the first push.

That is this lane's own proxy-vs-capability genus appearing *inside the fix for a
previous instance of the genus*, which is why the corrected form is centralised here
with its own two-directional test rather than re-typed at each call site.

The root is found by STRUCTURE, not by depth or by name: walk up until a directory holds
``src/``, ``tests/`` and ``pyproject.toml``. Depth-based ``parents[3]`` would silently
mean something different if a test file ever moved a level, and a name match would be
another literal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["checkout_root", "assert_same_checkout"]

#: The markers that identify a checkout root. All three must be present: `src` alone
#: appears in plenty of subdirectories, and `pyproject.toml` alone can sit in a nested
#: package.
_ROOT_MARKERS = ("src", "tests", "pyproject.toml")


def checkout_root(start: str | Path) -> Path:
    """Return the checkout root containing ``start``.

    Raises AssertionError rather than returning None: a caller that cannot locate its
    own checkout must not silently continue, because the next thing it would do is
    compare against a root it guessed.
    """
    resolved = Path(start).resolve()
    for candidate in (resolved, *resolved.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise AssertionError(
        f"no checkout root above {resolved} contains all of {_ROOT_MARKERS}; "
        "cannot establish which tree this test belongs to"
    )


def assert_same_checkout(module: Any, test_file: str | Path) -> Path:
    """Assert ``module`` was imported from the same checkout as ``test_file``.

    Args:
        module: the imported module under test (anything with ``__file__``).
        test_file: the calling test's ``__file__``.

    Returns:
        The checkout root, so a caller that wants it does not have to re-derive it.
    """
    root = checkout_root(test_file)
    module_file = getattr(module, "__file__", None)
    assert module_file, f"{module!r} has no __file__; cannot prove which tree it came from"
    module_path = Path(module_file).resolve()
    assert module_path.is_relative_to(root), (
        f"{getattr(module, '__name__', module)!r} was imported from {module_path}, which is "
        f"OUTSIDE the checkout containing this test ({root}). In a worktree the editable "
        f".pth can pull `src` from the MAIN checkout, so this test would be asserting about "
        f"a different tree's code than the one under review."
    )
    return root
