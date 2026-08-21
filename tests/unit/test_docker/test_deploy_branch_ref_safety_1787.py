"""#1787 — a deploy must never rewind the branch a human left checked out.

``deploy.yml`` guards the working TREE before it hard-resets the droplet checkout::

    # Abort only if a TRACKED file was hot-patched live: that is the change
    # `git reset --hard` would silently clobber, so refusing protects it.
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then

``git status --untracked-files=no`` reports on the working tree. It says nothing about
**where HEAD points**. When HEAD is attached to a branch, ``git reset --hard`` moves that
**branch pointer** — so committed-but-unpushed work is rewound while the guard reports
all-clear.

The inversion is the defect: **a developer who committed their work is LESS protected
than one who left it dirty.** Dirty files abort the deploy; commits do not. A guard that
punishes the more careful behaviour trains people not to commit on the shared checkout,
which is the working-tree state it exists to refuse.

Reachable because PROD == DEV == the same box: ``$PROJECT_DIR`` is both the deploy target
and a shared human working copy. On 2026-08-21 it sat on
``docs/issue-1768-per-worker-cache-measurement`` while a deploy was dispatched; only that
branch happening to be dirty for an unrelated reason stopped the rewind.

``rollback_to_prev()`` already STATES the invariant that makes the reset safe::

    # The droplet's main is a pure mirror of origin/main (never pushed from),
    # so rewinding the branch ref is safe — the next successful deploy's
    # `reset --hard origin/main` moves it forward again. The pre-deploy dirty-
    # tracked-files guard has already run, so there is nothing to clobber.

Both premises hold on ``main`` and fail on a feature branch, which IS pushed from and is
the only ref to that work. Nothing checked that the invariant held. This module pins that
it now does.

Run against a REAL git repository rather than stubbed ``git``. Every question here is
about what git actually does to refs, so a stub would only re-state my belief about it —
and the mechanism is genuinely surprising (see ``test_reattach_survives_an_untracked_collision``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

HELPER = "reattach_to_main"


# --------------------------------------------------------------------------- #
# Extraction (the SHIPPED artifact, verbatim)
# --------------------------------------------------------------------------- #
def _extract_deploy_script() -> str:
    wf: dict = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    for step in wf["jobs"]["deploy"]["steps"]:
        with_ = step.get("with") or {}
        if "script" in with_:
            return str(with_["script"])
    raise AssertionError("deploy.yml has no ssh-action step carrying a `script:`")


def _extract_helper(script: str, name: str) -> str:
    lines = script.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith(f"{name}() {{")),
        None,
    )
    assert start is not None, f"deploy.yml's droplet script defines no {name}() helper"
    indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "}" and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            return "\n".join(ln[indent:] for ln in lines[start : j + 1])
    raise AssertionError(f"{name}() in deploy.yml has no closing brace at its own indent")


# --------------------------------------------------------------------------- #
# A real repository, shaped like the droplet checkout
# --------------------------------------------------------------------------- #
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    """A repo with `main`, plus a `feature` branch carrying an unpushed commit."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", ".")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "f").write_text("base\n")
    _git(repo, "add", "f")
    _git(repo, "commit", "-qm", "base")
    (repo / "only_in_main").write_text("m\n")
    _git(repo, "add", "only_in_main")
    _git(repo, "commit", "-qm", "main-only file")
    _git(repo, "branch", "feature", "HEAD~1")
    return repo


def _run_helper(repo: Path) -> subprocess.CompletedProcess[str]:
    """Execute the SHIPPED reattach_to_main() inside a real checkout."""
    body = _extract_helper(_extract_deploy_script(), HELPER)
    runner = repo.parent / "run.sh"
    runner.write_text("set -e\n" + body + f"\n{HELPER}\n")
    return subprocess.run(
        ["bash", str(runner)], cwd=repo, capture_output=True, text=True, timeout=30
    )


def _checkout_feature_with_work(repo: Path) -> str:
    _git(repo, "checkout", "-q", "feature")
    (repo / "w").write_text("work\n")
    _git(repo, "add", "w")
    _git(repo, "commit", "-qm", "unpushed feature work")
    return _git(repo, "rev-parse", "feature")


# --------------------------------------------------------------------------- #
# 1. The defect itself
# --------------------------------------------------------------------------- #
def test_reset_rewinds_a_checked_out_branch_while_the_guard_reports_clean(
    tmp_path: Path,
) -> None:
    """Characterisation: this is what the deploy did before #1787.

    Not a test of our code — a test of git's semantics, pinned so the fix below is
    understood as protecting against something real rather than hypothetical. If this
    ever stops holding, the whole issue is moot and the fix can go.
    """
    repo = _make_repo(tmp_path)
    feat = _checkout_feature_with_work(repo)
    base = _git(repo, "rev-parse", "main")

    guard_input = _git(repo, "status", "--porcelain", "--untracked-files=no")
    assert guard_input == "", "the deploy's dirty-file guard sees nothing and allows this"

    _git(repo, "reset", "--hard", base)
    assert _git(repo, "rev-parse", "feature") != feat, (
        "reset --hard moved the CHECKED-OUT branch ref, not just the worktree"
    )


# --------------------------------------------------------------------------- #
# 2. The fix
# --------------------------------------------------------------------------- #
def test_reattach_moves_head_to_main_and_leaves_the_feature_ref_alone(
    tmp_path: Path,
) -> None:
    """RED before the fix: deploy.yml defines no reattach_to_main()."""
    repo = _make_repo(tmp_path)
    feat = _checkout_feature_with_work(repo)

    proc = _run_helper(repo)
    assert proc.returncode == 0, proc.stderr

    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main", (
        "HEAD must end up ATTACHED to main — a detached HEAD is the 2026-07-21 state "
        "where the unheld `main` branch got taken over by worktree merges"
    )
    assert _git(repo, "rev-parse", "feature") == feat, (
        "the feature branch must still point at the human's work"
    )


def test_reattach_repairs_a_detached_head(tmp_path: Path) -> None:
    """The 2026-07-21 incident left the droplet detached for two days.

    `rollback_to_prev()` uses `reset --hard` rather than `checkout <sha>` precisely to
    avoid detaching; re-attaching here also repairs the state if it ever happens again.
    """
    repo = _make_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    assert (
        subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo, capture_output=True
        ).returncode
        != 0
    ), "precondition: HEAD really is detached"

    proc = _run_helper(repo)
    assert proc.returncode == 0, proc.stderr
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main"


def test_reattach_is_a_no_op_on_main(tmp_path: Path) -> None:
    """The overwhelmingly common case must cost nothing and move nothing."""
    repo = _make_repo(tmp_path)
    before = _git(repo, "rev-parse", "main")

    proc = _run_helper(repo)
    assert proc.returncode == 0, proc.stderr
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main"
    assert _git(repo, "rev-parse", "main") == before, "a no-op must not move main"


def test_reattach_survives_an_untracked_collision(tmp_path: Path) -> None:
    """The regression this fix could easily have introduced, pinned.

    The dirty guard uses `-uno` DELIBERATELY: untracked ops scratch files must not block
    a deploy, because they previously did. But `git checkout -B main <start-point>`
    ABORTS when an untracked file would be overwritten by the switch — measured:

        error: The following untracked working tree files would be overwritten...
        Please move or remove them before you switch branches.
        Aborting

    That would fail deploys on exactly the files the guard tolerates. `git checkout -B
    main` with NO start point re-points main at the CURRENT HEAD, so the worktree does
    not change and nothing can collide. This test is the reason the helper is written
    that way, and it fails if anyone "tidies" a start-point back in.
    """
    repo = _make_repo(tmp_path)
    feat = _checkout_feature_with_work(repo)
    # untracked here, but tracked on main -> a switch that changes the tree would abort
    (repo / "only_in_main").write_text("ops scratch\n")
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == "", (
        "precondition: the dirty guard still sees a clean tree and allows the deploy"
    )

    proc = _run_helper(repo)
    assert proc.returncode == 0, (
        f"re-attaching must not abort on an untracked collision.\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main"
    assert _git(repo, "rev-parse", "feature") == feat
    assert (repo / "only_in_main").read_text() == "ops scratch\n", (
        "the operator's scratch file must survive untouched"
    )


# --------------------------------------------------------------------------- #
# 3. End to end — re-attach THEN reset, which is what the deploy does
# --------------------------------------------------------------------------- #
def test_the_reset_after_reattaching_moves_main_and_not_the_feature_branch(
    tmp_path: Path,
) -> None:
    """The whole point, in one assertion pair.

    Same starting state as the characterisation test above, but with the helper run
    first: the deploy still converges the checkout onto its target, and the human's
    branch is left where they put it.
    """
    repo = _make_repo(tmp_path)
    feat = _checkout_feature_with_work(repo)
    target = _git(repo, "rev-parse", "main")

    assert _run_helper(repo).returncode == 0
    _git(repo, "reset", "--hard", target)

    assert _git(repo, "rev-parse", "feature") == feat, (
        "the branch a human left checked out must be untouched by a deploy"
    )
    assert _git(repo, "rev-parse", "HEAD") == target, "and the deploy still converged"
    assert _git(repo, "symbolic-ref", "--short", "HEAD") == "main"


# --------------------------------------------------------------------------- #
# 4. Wiring — the helper is useless if it is never called, or called too late
# --------------------------------------------------------------------------- #
def test_reattach_is_called_before_every_hard_reset() -> None:
    """A helper that exists but runs after the reset protects nothing.

    Ordering is the entire contract here, so it is asserted on the shipped script rather
    than left to a reviewer's eye.
    """
    script = _extract_deploy_script()
    lines = script.splitlines()

    call = next(
        (i for i, ln in enumerate(lines) if ln.strip() == HELPER),
        None,
    )
    assert call is not None, f"{HELPER}() is defined but never called"

    resets = [
        i
        for i, ln in enumerate(lines)
        if ln.strip().startswith("git reset --hard")
        # the definition's own body is not a call site
        and not ln.strip().startswith('git reset --hard "$PREV_SHA"')
    ]
    assert resets, "no `git reset --hard` found — has the deploy script been restructured?"
    assert call < min(resets), (
        f"{HELPER}() is called at line {call} but the first `git reset --hard` is at "
        f"{min(resets)} — re-attaching after the reset is too late to protect anything"
    )


@pytest.mark.parametrize(
    "forbidden",
    [
        "git checkout -B main origin/main",
        "git checkout -B main $ORIGIN_SHA",
        'git checkout -B main "$ORIGIN_SHA"',
    ],
)
def test_reattach_never_uses_a_start_point(forbidden: str) -> None:
    """Pinned as a string too, not only behaviourally.

    `test_reattach_survives_an_untracked_collision` catches this, but only if someone
    runs it. The failure it prevents is a deploy outage on an ops scratch file, and the
    edit that causes it looks like a harmless clarification.
    """
    helper = _extract_helper(_extract_deploy_script(), HELPER)
    assert forbidden not in helper, (
        f"{forbidden!r} aborts when an untracked file collides with a path tracked on "
        "main — see the docstring of test_reattach_survives_an_untracked_collision"
    )
