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
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
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
