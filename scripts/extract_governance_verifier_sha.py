"""Extract the governance verifier SHA from the G2 pre-spec memo.

HIGH-6 (iter-3): the verifier scripts (`check_g2_commit_graph.py`,
`verify_g2_prespec_dataset_hashes.py`, `check_g2_prior_runs.py`) must
be pulled from a reviewed, SHA-pinned governance commit — not from
``origin/main`` (mutable protected ref). The pinned SHA lives in
``docs/specs/tier1b_b2_prespec_20260510.md`` under the
``governance_verifier_sha:`` key.

This script extracts the value so the GitHub workflow can stage
verifiers from that SHA. Stdout: the SHA (or the literal string
``ORIGIN_MAIN_FALLBACK`` if the memo carries the placeholder, in
which case the workflow falls back to origin/main).

Usage:
    python scripts/extract_governance_verifier_sha.py [--prespec-sha <SHA>] [--repo-root <path>]

The ``--prespec-sha`` is optional but RECOMMENDED in CI: with it, the
script reads the memo from ``git show <SHA>:<MEMO>`` (the immutable
S_prespec content) so a child commit cannot edit the SHA pin without
first writing a fresh memo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PLACEHOLDER = "TODO_PIN_AT_FIRST_GREEN_RUN"
FALLBACK_TOKEN = "ORIGIN_MAIN_FALLBACK"
MEMO_RELPATH = "docs/specs/tier1b_b2_prespec_20260510.md"


def _resolve_repo_root() -> Path:
    """Same resolver pattern as the verifier scripts."""
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


def _git_show(sha: str, relpath: str, *, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "show", f"{sha}:{relpath}"],
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prespec-sha",
        default=None,
        help=(
            "Read the memo from git show <SHA>:<MEMO> (recommended in CI) "
            "so a child commit cannot edit the pin without a fresh memo."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override worktree root (mirrors the verifier --repo-root flag).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _resolve_repo_root()

    if args.prespec_sha:
        memo_text = _git_show(args.prespec_sha, MEMO_RELPATH, cwd=repo_root)
        if memo_text is None:
            print(
                f"FATAL: could not load memo at S_prespec={args.prespec_sha} via git show",
                file=sys.stderr,
            )
            return 2
    else:
        memo_path = repo_root / MEMO_RELPATH
        if not memo_path.exists():
            print(f"FATAL: memo not found at {memo_path}", file=sys.stderr)
            return 2
        memo_text = memo_path.read_text(encoding="utf-8")

    # Inline parse so this script remains independent of the verifier
    # module's import surface (avoids picking up its module-level
    # subprocess calls under unusual CWDs).
    import re

    pattern = re.compile(
        r'^\s*governance_verifier_sha\s*:\s*"([^"]*)"',
        re.MULTILINE,
    )
    m = pattern.search(memo_text)
    if m is None:
        # Field absent — treat as fallback (safer than failing CI on a
        # not-yet-pinned memo).
        print(FALLBACK_TOKEN)
        return 0
    value = m.group(1).strip()
    if not value or value == PLACEHOLDER:
        print(FALLBACK_TOKEN)
        return 0
    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
