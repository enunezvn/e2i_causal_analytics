"""Unit tests for ``src.utils.project_root.find_project_root`` (Block 6A polish).

The helper replaces three ad-hoc ``Path(__file__).resolve().parents[N]``
heuristics scattered across the codebase. These tests lock in:

* the marker walk terminates at the first ancestor containing
  ``pyproject.toml``;
* ``E2I_CONFIG_DIR`` overrides the marker walk and resolves to the
  parent of the named directory;
* a clear error is raised when no marker file exists in any ancestor.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from src.utils.project_root import (
    ProjectRootNotFoundError,
    find_project_root,
)


@pytest.fixture(autouse=True)
def _clear_env_override() -> Iterator[None]:
    """Drop ``E2I_CONFIG_DIR`` between tests so a leak from one test
    cannot poison the next."""
    saved = os.environ.pop("E2I_CONFIG_DIR", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["E2I_CONFIG_DIR"] = saved
        else:
            os.environ.pop("E2I_CONFIG_DIR", None)


class TestMarkerWalk:
    """The walk-up-to-marker idiom must find the root by ``pyproject.toml``."""

    def test_finds_root_from_nested_file(self, tmp_path: Path) -> None:
        """A pyproject.toml at the staged root must be found from any
        descendant."""
        root = tmp_path / "stub_repo"
        root.mkdir()
        (root / "pyproject.toml").write_text("[tool.poetry]\nname='stub'\n")

        nested = root / "src" / "agents" / "demo"
        nested.mkdir(parents=True)
        nested_file = nested / "module.py"
        nested_file.write_text("# stub\n")

        resolved = find_project_root(start=nested_file)
        assert resolved == root.resolve()

    def test_default_start_resolves_repo_root(self) -> None:
        """Calling with no start must resolve a directory that contains
        the canonical marker (a sanity check against the live repo).
        """
        resolved = find_project_root()
        # The actual repo root must contain pyproject.toml.
        assert (resolved / "pyproject.toml").exists(), (
            f"Resolved root {resolved!s} has no pyproject.toml — "
            "the walk-up did not terminate at the right anchor."
        )

    def test_directory_start_walks_from_itself(self, tmp_path: Path) -> None:
        """When ``start`` is a directory, the walk includes that directory."""
        root = tmp_path / "stub"
        root.mkdir()
        (root / "pyproject.toml").write_text("\n")

        resolved = find_project_root(start=root)
        assert resolved == root.resolve()


class TestEnvOverride:
    """``E2I_CONFIG_DIR`` overrides the marker walk."""

    def test_env_override_resolves_to_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override names a config directory; the project root is its
        parent. Mirrors how callers use it in production (``<root>/config``)."""
        repo = tmp_path / "synthetic_repo"
        repo.mkdir()
        # The override target must live inside a real project tree —
        # i.e. a marker must resolve from its parent. This mirrors the
        # validation added in I-2.
        (repo / "pyproject.toml").write_text("\n")
        config_dir = repo / "config"
        config_dir.mkdir()

        monkeypatch.setenv("E2I_CONFIG_DIR", str(config_dir))

        resolved = find_project_root()
        assert resolved == repo.resolve()

    def test_env_override_takes_precedence_over_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the env var is set, the marker walk is NOT consulted —
        the override wins even though a marker would also resolve."""
        repo = tmp_path / "synthetic_repo"
        repo.mkdir()
        # Stage a marker in the repo so the marker walk would also find a
        # root — the override should take precedence regardless.
        (repo / "pyproject.toml").write_text("\n")

        sibling_repo = tmp_path / "sibling_root"
        sibling_repo.mkdir()
        # The sibling repo also needs a marker, otherwise the override
        # validation (I-2) would reject it.
        (sibling_repo / "pyproject.toml").write_text("\n")
        sibling_config = sibling_repo / "config"
        sibling_config.mkdir()

        monkeypatch.setenv("E2I_CONFIG_DIR", str(sibling_config))

        # Walking from repo would normally yield repo.resolve(); the
        # override forces resolution to sibling_repo instead.
        resolved = find_project_root(start=repo)
        assert resolved == sibling_repo.resolve()

    def test_env_override_with_missing_parent_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bogus override (parent does not exist) must raise so the
        caller doesn't silently consume a wrong root."""
        bogus = tmp_path / "does" / "not" / "exist" / "config"
        # Parent ``does/not/exist`` is missing; we don't create it.
        monkeypatch.setenv("E2I_CONFIG_DIR", str(bogus))

        with pytest.raises(ProjectRootNotFoundError, match="does not exist"):
            find_project_root()

    def test_env_override_parent_without_marker_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """I-2: when the parent of the override exists but does NOT
        live inside a project tree (no marker resolves from there or
        any ancestor), raise an actionable error that names the env
        var and the resolved path. Otherwise the downstream config
        loader fails with an opaque ``FileNotFoundError`` and the
        developer can't tell ``E2I_CONFIG_DIR`` is the cause.
        """
        # Create only the parent — no marker anywhere.
        fake_parent = tmp_path / "fake"
        fake_parent.mkdir()
        fake_config = fake_parent / "config"
        # Note: ``fake_config`` is intentionally NOT created — only
        # the parent must exist.

        monkeypatch.setenv("E2I_CONFIG_DIR", str(fake_config))

        with pytest.raises(ProjectRootNotFoundError) as excinfo:
            find_project_root()

        msg = str(excinfo.value)
        # The error must explicitly name the env var so a developer
        # grepping logs can locate the misconfiguration immediately.
        assert "E2I_CONFIG_DIR" in msg
        # The error must include the resolved path to disambiguate
        # which directory was actually consulted.
        assert str(fake_parent.resolve()) in msg
        # The error must name the marker(s) it looked for.
        assert "pyproject.toml" in msg


class TestMultiMarker:
    """``markers=...`` opts a caller into a custom marker set (I-1)."""

    @pytest.mark.parametrize(
        "marker_name",
        [
            ".claude",  # directory marker (skills loader's secondary anchor)
            "pyproject.toml",  # file marker (default)
            "Cargo.toml",  # any-other-marker the caller chooses
        ],
    )
    def test_finds_root_by_custom_marker(
        self, tmp_path: Path, marker_name: str
    ) -> None:
        """Any of the supplied markers — file or directory — must resolve
        the root. The skills loader uses ``(\"pyproject.toml\", \".claude\")``
        because some checkouts predate ``pyproject.toml`` adoption and
        anchor on ``.claude/`` instead.
        """
        root = tmp_path / "stub_repo"
        root.mkdir()
        target = root / marker_name
        # Create as directory if name has no extension, otherwise file.
        if "." in marker_name and not marker_name.startswith("."):
            target.write_text("")
        elif marker_name == ".claude":
            target.mkdir()
        else:
            target.write_text("")

        nested = root / "src" / "module"
        nested.mkdir(parents=True)
        nested_file = nested / "thing.py"
        nested_file.write_text("\n")

        resolved = find_project_root(
            start=nested_file,
            markers=("pyproject.toml", ".claude", "Cargo.toml"),
        )
        assert resolved == root.resolve()

    def test_default_markers_do_not_match_dot_claude(
        self, tmp_path: Path
    ) -> None:
        """Backward compat: with the default marker set, a tree anchored
        only on ``.claude`` must NOT resolve. Callers that need that
        anchor have to opt in via ``markers=...``.
        """
        root = tmp_path / "claude_only_repo"
        root.mkdir()
        (root / ".claude").mkdir()  # only marker — no pyproject.toml

        nested = root / "src" / "module"
        nested.mkdir(parents=True)
        nested_file = nested / "thing.py"
        nested_file.write_text("\n")

        # With default markers (only pyproject.toml), the walk falls
        # off the top of the tmp_path tree. Either it raises (typical
        # on /tmp) or it resolves at some real ancestor with a
        # pyproject.toml — but never at ``root`` itself.
        try:
            resolved = find_project_root(start=nested_file)
        except ProjectRootNotFoundError:
            return  # expected on most filesystems
        assert resolved != root.resolve(), (
            "Default markers must not include .claude — caller must opt in."
        )


class TestNoMarkerFound:
    """When no marker exists anywhere upward, raise a clear error."""

    def test_raises_when_no_marker_in_any_ancestor(self, tmp_path: Path) -> None:
        """A tmp_path tree that does NOT contain pyproject.toml anywhere
        upward (it would, if the repo root happens to be a parent of
        tmp_path — which it usually isn't) must raise.

        ``tmp_path`` typically lives under ``/tmp`` on Linux; the
        filesystem root has no pyproject.toml, so the walk should
        terminate with a ``ProjectRootNotFoundError``.
        """
        nested = tmp_path / "deep" / "no" / "markers" / "here"
        nested.mkdir(parents=True)
        nested_file = nested / "leaf.py"
        nested_file.write_text("\n")

        # If the system happens to have a pyproject.toml in some parent
        # directory of tmp_path (unusual), this test would silently
        # pass-through; that's acceptable — the contract we lock in is
        # that the error is clear *when* it fires.
        try:
            resolved = find_project_root(start=nested_file)
        except ProjectRootNotFoundError as exc:
            assert "pyproject.toml" in str(exc)
            return
        # If the walk somehow terminated at a real root above /tmp, the
        # test still validates the happy path; we don't care which root,
        # only that the error message contains the marker name when it
        # does fire. Skip the strict assertion in that case.
        pytest.skip(
            f"tmp_path tree {tmp_path!s} unexpectedly resolved to "
            f"{resolved!s} via a parent pyproject.toml; cannot exercise "
            "the no-marker branch on this filesystem."
        )
