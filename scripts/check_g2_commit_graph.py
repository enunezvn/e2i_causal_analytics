"""Plan v4 Gate G2 — commit-graph parent-check.

Closes G2's threshold-shopping defense for the commit-graph parent
constraint: the experiment commit (the commit that introduces
``scripts/run_tier1b_b2_experiment.py`` AND
``.github/workflows/tier1b_b2_experiment.yml``) MUST be a CHILD of the
``S_prespec`` commit (the commit that introduces
``docs/specs/tier1b_b2_prespec_20260510.md``).

Why this matters (codex-rescue HIGH-2): a determined threshold-shopper
could attempt to commit a "pre-spec" copy on a scratch branch BEFORE
the actual experiment commit, then run the experiment against pre-pinned
thresholds. The parent-check defeats this: the experiment commit's
ancestry MUST contain the introducing commit of the pre-spec memo. Any
commit that is an ancestor OR a sibling (not a descendant) fails this
check.

The check uses ``git merge-base --is-ancestor S_prespec experiment_sha``:

* exits 0 iff ``S_prespec`` is an ancestor of ``experiment_sha`` (i.e.,
  the experiment commit is a descendant — which is what we require)
* exits 1 if NOT an ancestor (parent constraint VIOLATED)
* exits 128 on git error

Usage
-----

CI mode (auto-discover SHAs):
    python scripts/check_g2_commit_graph.py
    # Discovers S_prespec by inspecting git log for the commit that
    # introduced docs/specs/tier1b_b2_prespec_20260510.md. The
    # experiment commit is HEAD by default.

Explicit:
    python scripts/check_g2_commit_graph.py \\
        --prespec-sha <sha> --experiment-sha <sha>

Manual override (the spec memo's introducing commit can be passed
explicitly when CI runs are detached or shallow):
    python scripts/check_g2_commit_graph.py \\
        --prespec-sha 7f616f6f \\
        --experiment-sha HEAD

Behaviour on edge cases
-----------------------

* If the introducing commit cannot be discovered (memo file absent
  from history), exits 2 with a clear pointer.
* If S_prespec == experiment_sha, the parent constraint is VIOLATED
  (the experiment must be a CHILD, not an alias).
* If the path passed to --prespec-sha-from-file is unreadable, exits 2.

Scope (INFO-12 — codex pass-1 informational)
---------------------------------------------

Commit ancestry alone CANNOT prove that no prior exploration occurred.
A determined developer can explore locally on a scratch branch, then
create a clean child commit after S_prespec — the parent-check will
pass even though the developer already knows which thresholds would
pass on the cohort.

The commit-graph parent-check is therefore ONE control in a stack of
four that together close the threshold-shopping defense:

  1. ``check_g2_commit_graph.py`` (this script) — ensures the
     experiment commit DESCENDS from S_prespec.
  2. ``verify_g2_prespec_dataset_hashes.py --prespec-sha`` — ensures
     the pinned hashes are read from the IMMUTABLE S_prespec content,
     not the mutable working tree. Plus the memo-content-unchanged
     check catches edits to load-bearing sections.
  3. Workflow runs verifiers from ``origin/main`` (protected ref) so
     the experiment tag CANNOT weaken the verifier code.
  4. ``.github/g2_runs/<S_prespec>.json`` run registry — first
     ATTEMPTED workflow run is load-bearing; subsequent runs from the
     same S_prespec are rejected. Defeats sequential-testing
     fishing-expedition.

Threshold-shopping is fully prevented only when all four controls
hold. If any control is bypassed or weakened, the threshold-shopping
defense is broken even if the others pass.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _resolve_repo_root() -> Path:
    """Resolve the actual git worktree root.

    NEW HIGH-2 (iter-3) fix: when the workflow copies this script into
    ``governance_checkout/scripts/`` (HIGH-6 protected verifier
    staging), ``Path(__file__).resolve().parents[1]`` resolves to
    ``governance_checkout``, NOT the actual worktree the workflow
    checked out from the experiment tag.

    Resolution order:
      1. ``E2I_GOVERNANCE_REPO_ROOT`` env var — explicit override.
      2. ``git rev-parse --show-toplevel`` from CWD — preferred.
      3. ``Path(__file__).resolve().parents[1]`` — legacy fallback.
    """
    env_root = os.environ.get("E2I_GOVERNANCE_REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
DEFAULT_PRESPEC_PATH = "docs/specs/tier1b_b2_prespec_20260510.md"


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    """Run a git command and return stdout (stripped). Raises
    subprocess.CalledProcessError on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd or REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolve_sha(ref: str, cwd: Optional[Path] = None) -> str:
    """Resolve a ref (HEAD, short SHA, branch) to a full SHA."""
    return _git("rev-parse", ref, cwd=cwd)


def _discover_introducing_commit(memo_relpath: str, cwd: Optional[Path] = None) -> Optional[str]:
    """Discover the commit that INTRODUCED ``memo_relpath`` (i.e., the
    earliest commit in history that contains the file).

    Returns the full SHA, or None if the file is not in history.

    Implementation: ``git log --diff-filter=A --follow --format=%H -- <path>``
    returns commits that ADDED the path; the LAST line is the earliest
    addition.
    """
    try:
        output = _git(
            "log",
            "--diff-filter=A",
            "--follow",
            "--format=%H",
            "--",
            memo_relpath,
            cwd=cwd,
        )
    except subprocess.CalledProcessError:
        return None
    lines = [ln for ln in output.splitlines() if ln.strip()]
    if not lines:
        return None
    # The earliest addition is the LAST line (git log is reverse-chrono).
    return lines[-1]


def _is_ancestor(ancestor_sha: str, descendant_sha: str, cwd: Optional[Path] = None) -> bool:
    """Return True iff ``ancestor_sha`` is an ancestor of
    ``descendant_sha``. Uses ``git merge-base --is-ancestor`` which
    exits 0 for "is an ancestor" and 1 otherwise.
    """
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant_sha],
        cwd=str(cwd or REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        # 128 = git error (e.g., one of the SHAs doesn't exist).
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return result.returncode == 0


def check_parent(
    prespec_sha: str,
    experiment_sha: str,
    cwd: Optional[Path] = None,
) -> int:
    """Verify the parent constraint: prespec_sha is an ancestor of
    experiment_sha AND prespec_sha != experiment_sha.

    Returns 0 on PASS, 1 on FAIL (parent constraint violated), 2 on
    configuration error (one of the SHAs is empty or unresolvable).
    """
    if not prespec_sha or not experiment_sha:
        print(
            "FATAL: empty SHA passed (prespec_sha={prespec_sha!r}, "
            "experiment_sha={experiment_sha!r})".format(
                prespec_sha=prespec_sha, experiment_sha=experiment_sha
            ),
            file=sys.stderr,
        )
        return 2

    try:
        prespec_full = _resolve_sha(prespec_sha, cwd=cwd)
        experiment_full = _resolve_sha(experiment_sha, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"FATAL: failed to resolve SHA: {e.stderr}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("G2 commit-graph parent-check")
    print("=" * 70)
    print(f"  S_prespec:    {prespec_full}")
    print(f"  experiment:   {experiment_full}")

    if prespec_full == experiment_full:
        print()
        print(
            "[FAIL] S_prespec == experiment commit. The experiment "
            "commit MUST be a CHILD of S_prespec (a strict descendant), "
            "not an alias of it."
        )
        return 1

    try:
        is_ancestor = _is_ancestor(prespec_full, experiment_full, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"FATAL: git merge-base failed: {e.stderr}", file=sys.stderr)
        return 2

    if not is_ancestor:
        print()
        print(
            "[FAIL] S_prespec is NOT an ancestor of experiment commit. "
            "The experiment commit must be a CHILD of S_prespec; the "
            "current ancestry does not include S_prespec. Resolution: "
            "rebase the experiment branch onto S_prespec OR re-create "
            "the experiment commit on a fresh branch from S_prespec."
        )
        return 1

    print()
    print("[OK] S_prespec is an ancestor of experiment commit. Parent constraint satisfied.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prespec-sha",
        default=None,
        help=(
            "SHA of the S_prespec commit (the commit that introduces "
            "the pre-spec memo). If omitted, auto-discovered from git "
            "log of the memo path."
        ),
    )
    parser.add_argument(
        "--experiment-sha",
        default="HEAD",
        help="SHA of the experiment commit. Default: HEAD.",
    )
    parser.add_argument(
        "--prespec-path",
        default=DEFAULT_PRESPEC_PATH,
        help=(
            "Repo-relative path to the pre-spec memo whose introducing "
            "commit becomes S_prespec. Default: "
            f"{DEFAULT_PRESPEC_PATH}."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root (for testing with a synthetic git repo).",
    )
    args = parser.parse_args(argv)

    cwd = Path(args.repo_root)

    prespec_sha = args.prespec_sha
    if prespec_sha is None:
        discovered = _discover_introducing_commit(args.prespec_path, cwd=cwd)
        if discovered is None:
            print(
                f"FATAL: could not discover the commit that introduced "
                f"{args.prespec_path!r}. Pass --prespec-sha explicitly "
                "OR ensure the memo is in this branch's history.",
                file=sys.stderr,
            )
            return 2
        prespec_sha = discovered
        print(f"  (auto-discovered S_prespec from git log: {prespec_sha[:12]}…)")

    return check_parent(prespec_sha, args.experiment_sha, cwd=cwd)


if __name__ == "__main__":
    sys.exit(main())
