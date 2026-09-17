"""The checkout guard must PASS here and FAIL for a module from another tree.

Both directions are required, and the second is the one that matters. The assertion
this replaces (a ``.worktrees/<lane>`` substring test) failed everywhere
but one directory; the obvious over-correction -- dropping the check, or asserting
something always true -- would pass everywhere and catch nothing. A guard that cannot
fail is worth no more than one that cannot pass.

The "other tree" here is a real directory built by ``tmp_path`` with the same marker
layout, not a synthetic string. That keeps this test free of the environment-specific
literals the guard exists to eliminate -- writing ``/home/enunez/...`` to prove the
point would have reintroduced the family in the test that polices it.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from tests._checkout_guard import assert_same_checkout, checkout_root


def _fake_module(name: str, file: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(__name__=name, __file__=str(file))


def _make_checkout(root: Path) -> Path:
    """A minimal tree carrying all three markers the root finder looks for."""
    (root / "src" / "etl").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='decoy'\n")
    module = root / "src" / "etl" / "decoy_etl.py"
    module.write_text("VALUE = 1\n")
    return module


# =============================================================================
# The passing direction
# =============================================================================


def test_a_module_from_this_checkout_is_accepted() -> None:
    """The real case: src/ and tests/ in the same tree, however that tree is named."""
    from src.etl import territory_metrics_etl as etl

    root = assert_same_checkout(etl, __file__)
    assert (root / "src").is_dir() and (root / "tests").is_dir()


def test_the_root_is_found_by_structure_not_by_depth_or_name(tmp_path: Path) -> None:
    """Depth-based parents[N] breaks silently if a file moves a level, and a name match
    would be another literal. The finder walks up to the markers instead, so it works
    from any nesting depth and under any directory name."""
    module = _make_checkout(tmp_path / "some-other-name")
    deep = tmp_path / "some-other-name" / "tests" / "a" / "b" / "c"
    deep.mkdir(parents=True)
    probe = deep / "test_probe.py"
    probe.write_text("")
    assert checkout_root(probe) == (tmp_path / "some-other-name").resolve()
    assert_same_checkout(_fake_module("decoy_etl", module), probe)


# =============================================================================
# The failing direction -- the positive control
# =============================================================================


def test_a_module_from_a_DIFFERENT_checkout_is_rejected(tmp_path: Path) -> None:
    """The failure mode the guard exists for: the editable .pth pulls `src` from the
    MAIN checkout while the test lives in a worktree, so the test silently asserts
    about another tree's code."""
    here_module = _make_checkout(tmp_path / "worktree")
    there_module = _make_checkout(tmp_path / "main")
    probe = tmp_path / "worktree" / "tests" / "test_probe.py"
    probe.write_text("")

    # Same-tree module: accepted.
    assert_same_checkout(_fake_module("decoy_etl", here_module), probe)

    # Other-tree module: rejected, and the message names both paths.
    with pytest.raises(AssertionError) as exc:
        assert_same_checkout(_fake_module("decoy_etl", there_module), probe)
    message = str(exc.value)
    assert str(there_module.resolve()) in message
    assert str((tmp_path / "worktree").resolve()) in message
    assert "OUTSIDE the checkout" in message


def test_a_sibling_path_that_merely_shares_a_prefix_is_rejected(tmp_path: Path) -> None:
    """``/x/repo2`` must not read as inside ``/x/repo``.

    A string ``startswith`` comparison would accept it; ``is_relative_to`` compares path
    components. This is the bug the obvious implementation has, so it gets its own case.
    """
    _make_checkout(tmp_path / "repo")
    sibling_module = _make_checkout(tmp_path / "repo2")
    probe = tmp_path / "repo" / "tests" / "test_probe.py"
    probe.write_text("")
    with pytest.raises(AssertionError):
        assert_same_checkout(_fake_module("decoy_etl", sibling_module), probe)


def test_a_module_without_a_file_is_rejected(tmp_path: Path) -> None:
    """A namespace package or a C extension has no ``__file__``; the guard must refuse
    rather than skip, because it cannot prove anything about such a module."""
    _make_checkout(tmp_path / "repo")
    probe = tmp_path / "repo" / "tests" / "test_probe.py"
    probe.write_text("")
    with pytest.raises(AssertionError, match="has no __file__"):
        assert_same_checkout(types.SimpleNamespace(__name__="x", __file__=None), probe)


def test_a_path_with_no_checkout_above_it_is_refused(tmp_path: Path) -> None:
    """The finder must not return a guessed root. If no ancestor carries the markers it
    raises, so a caller cannot go on to compare against something it invented."""
    orphan = tmp_path / "nowhere" / "test_probe.py"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("")
    with pytest.raises(AssertionError, match="no checkout root above"):
        checkout_root(orphan)
