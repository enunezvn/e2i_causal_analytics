"""Unit tests for ``scripts/check_g2_commit_graph.py``.

Spawns synthetic git repos in tmp_path so the parent-check semantics
can be exercised against real `git merge-base --is-ancestor` output
without depending on the project's commit graph.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_g2_commit_graph as C  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — synthetic git repos under tmp_path
# ---------------------------------------------------------------------------


def _git_init(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=str(tmp_path),
        check=True,
    )


def _git_commit_file(tmp_path: Path, relpath: str, content: str, msg: str) -> str:
    target = tmp_path / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", relpath], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", msg],
        cwd=str(tmp_path),
        check=True,
    )
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _has_git() -> bool:
    return shutil.which("git") is not None


pytestmark = pytest.mark.skipif(
    not _has_git(),
    reason="git not on PATH; commit-graph tests need a real git binary",
)


# ---------------------------------------------------------------------------
# check_parent — happy path + parent constraint violations
# ---------------------------------------------------------------------------


def test_check_parent_passes_for_direct_descendant(tmp_path: Path) -> None:
    """S_prespec is parent of HEAD → exit 0."""
    _git_init(tmp_path)
    s_prespec = _git_commit_file(tmp_path, "memo.md", "spec memo content", "Add memo")
    head = _git_commit_file(tmp_path, "exp.py", "experiment code", "Add experiment")

    rc = C.check_parent(s_prespec, head, cwd=tmp_path)
    assert rc == 0
    # head is strictly newer than s_prespec
    assert head != s_prespec


def test_check_parent_fails_when_experiment_is_ancestor(tmp_path: Path) -> None:
    """Reversed ancestry: S_prespec listed AFTER experiment → fail."""
    _git_init(tmp_path)
    earlier = _git_commit_file(tmp_path, "exp.py", "exp code first", "Add experiment first")
    later = _git_commit_file(tmp_path, "memo.md", "memo second", "Add memo after")

    # Pretend the LATER commit is S_prespec; check parent against the
    # EARLIER commit (which precedes it). The earlier commit is NOT a
    # descendant of the later commit → fail.
    rc = C.check_parent(later, earlier, cwd=tmp_path)
    assert rc == 1


def test_check_parent_fails_when_equal(tmp_path: Path) -> None:
    """S_prespec == experiment — alias is a violation."""
    _git_init(tmp_path)
    s = _git_commit_file(tmp_path, "memo.md", "memo content", "Add memo")

    rc = C.check_parent(s, s, cwd=tmp_path)
    assert rc == 1


def test_check_parent_returns_2_on_empty_sha() -> None:
    """Empty SHA passed → configuration error."""
    rc = C.check_parent("", "abc123", cwd=REPO_ROOT)
    assert rc == 2


def test_check_parent_returns_2_on_unresolvable_sha(tmp_path: Path) -> None:
    """Unresolvable SHA → configuration error."""
    _git_init(tmp_path)
    s = _git_commit_file(tmp_path, "memo.md", "memo", "Add memo")

    # Pass a SHA that doesn't exist in the synthetic repo.
    rc = C.check_parent(s, "0" * 40, cwd=tmp_path)
    assert rc == 2


def test_check_parent_passes_for_indirect_descendant(tmp_path: Path) -> None:
    """S_prespec is a transitive ancestor (3 commits between) → pass."""
    _git_init(tmp_path)
    s_prespec = _git_commit_file(tmp_path, "memo.md", "spec memo", "Add memo")
    _git_commit_file(tmp_path, "filler1.py", "// 1", "filler 1")
    _git_commit_file(tmp_path, "filler2.py", "// 2", "filler 2")
    head = _git_commit_file(tmp_path, "exp.py", "experiment", "Add experiment")

    rc = C.check_parent(s_prespec, head, cwd=tmp_path)
    assert rc == 0
    assert s_prespec != head


def test_check_parent_fails_for_sibling_branch(tmp_path: Path) -> None:
    """Sibling branches (no ancestor relationship) → fail."""
    _git_init(tmp_path)
    base = _git_commit_file(tmp_path, "base.py", "base", "Add base")
    # Create branch_a from base
    subprocess.run(
        ["git", "checkout", "-q", "-b", "branch_a"],
        cwd=str(tmp_path),
        check=True,
    )
    s_prespec = _git_commit_file(tmp_path, "memo.md", "memo on branch_a", "Add memo on branch_a")
    # Switch back to master/main, then branch_b from base
    subprocess.run(
        ["git", "checkout", "-q", base],
        cwd=str(tmp_path),
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-q", "-b", "branch_b"],
        cwd=str(tmp_path),
        check=True,
    )
    experiment_sha = _git_commit_file(
        tmp_path, "exp.py", "experiment on branch_b", "Add experiment on branch_b"
    )

    # s_prespec lives on branch_a; experiment lives on branch_b.
    # Neither is an ancestor of the other.
    rc = C.check_parent(s_prespec, experiment_sha, cwd=tmp_path)
    assert rc == 1


# ---------------------------------------------------------------------------
# _discover_introducing_commit — git log integration
# ---------------------------------------------------------------------------


def test_discover_introducing_commit_returns_first_addition(tmp_path: Path) -> None:
    """The earliest commit that ADDED the file is returned."""
    _git_init(tmp_path)
    introducing = _git_commit_file(tmp_path, "docs/memo.md", "first version", "Add memo first time")
    # Modify the file in subsequent commits — discover should still
    # return the first addition.
    _git_commit_file(tmp_path, "docs/memo.md", "second version", "Edit memo")
    _git_commit_file(tmp_path, "docs/memo.md", "third version", "Edit memo again")

    discovered = C._discover_introducing_commit("docs/memo.md", cwd=tmp_path)
    assert discovered == introducing


def test_discover_introducing_commit_returns_none_for_missing(tmp_path: Path) -> None:
    """File not in history → None."""
    _git_init(tmp_path)
    _git_commit_file(tmp_path, "other.py", "other code", "Add other")

    discovered = C._discover_introducing_commit("docs/missing.md", cwd=tmp_path)
    assert discovered is None


# ---------------------------------------------------------------------------
# main() — argv plumbing + auto-discovery
# ---------------------------------------------------------------------------


def test_main_auto_discovers_prespec(tmp_path: Path) -> None:
    """main() with --repo-root auto-discovers S_prespec from git log."""
    _git_init(tmp_path)
    _git_commit_file(tmp_path, "docs/memo.md", "memo content", "Add memo")
    _git_commit_file(tmp_path, "exp.py", "experiment", "Add experiment")

    rc = C.main(
        [
            "--repo-root",
            str(tmp_path),
            "--prespec-path",
            "docs/memo.md",
            "--experiment-sha",
            "HEAD",
        ]
    )
    assert rc == 0


def test_main_returns_2_when_memo_not_in_history(tmp_path: Path) -> None:
    """If the memo path doesn't exist in history, main() returns 2."""
    _git_init(tmp_path)
    _git_commit_file(tmp_path, "exp.py", "experiment", "Add experiment")

    rc = C.main(
        [
            "--repo-root",
            str(tmp_path),
            "--prespec-path",
            "docs/missing_memo.md",
        ]
    )
    assert rc == 2


def test_main_explicit_prespec_sha_bypasses_discovery(tmp_path: Path) -> None:
    """When --prespec-sha is explicit, discovery is skipped."""
    _git_init(tmp_path)
    s = _git_commit_file(tmp_path, "memo.md", "memo content", "Add memo")
    _git_commit_file(tmp_path, "exp.py", "experiment", "Add experiment")

    rc = C.main(
        [
            "--repo-root",
            str(tmp_path),
            "--prespec-sha",
            s,
            "--experiment-sha",
            "HEAD",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# Project-context smoke — we expect the verifier to find SOMETHING
# pertinent in this repo's history (the actual S_prespec commit).
# This test does NOT assert pass/fail (since the worktree's commit
# state varies); it only asserts the verifier can run without crashing
# in the real repo.
# ---------------------------------------------------------------------------


def test_main_against_real_repo_does_not_crash() -> None:
    """The verifier must run cleanly against the real project repo;
    its return code is determined by the worktree's commit state.
    Acceptable codes are 0 (parent constraint OK) or 1 (parent
    constraint violated) or 2 (memo not yet introduced)."""
    rc = C.main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--prespec-path",
            "docs/specs/tier1b_b2_prespec_20260510.md",
            "--experiment-sha",
            "HEAD",
        ]
    )
    assert rc in (0, 1, 2), f"Unexpected exit code: {rc!r}"
