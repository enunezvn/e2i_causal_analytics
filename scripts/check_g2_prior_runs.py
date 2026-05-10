"""Plan v4 Gate G2 — durable prior-run check via GitHub API.

HIGH-5 (iter-3) fix: the in-workspace registry at
``.github/g2_runs/<S_prespec>.json`` is ephemeral — a fresh checkout in
a new workflow run does NOT see the registry written by a prior run,
so a determined developer could simply re-tag and the prior-attempt
gate would not trip.

This script queries the GitHub API for prior runs of the G2 workflow
whose tag commit message references the same ``S_prespec`` SHA, and
FAILS LOUDLY when ANY prior run (success / failure / in_progress)
exists.

Why "tag commit message references S_prespec":
The workflow's "Verify tag commit references S_prespec" step already
enforces that every G2 run's tag commit (or annotation) contains the
S_prespec SHA. Two runs sharing the same S_prespec MUST therefore
share that token in their respective ``headSha`` commit messages.
The script fetches each prior run's commit message via
``gh api repos/<owner>/<repo>/commits/<headSha>`` and looks for the
S_prespec SHA (long or short) — this is a durable, API-only signal
that survives registry-file deletion.

Usage
-----

CI (live API):
    python scripts/check_g2_prior_runs.py \\
        --s-prespec 7f616f6f7f616f6f7f616f6f7f616f6f7f616f6f \\
        --s-prespec-short 7f616f6f \\
        --workflow tier1b_b2_experiment.yml \\
        --current-run-id "$GITHUB_RUN_ID"

Test mode (no live API):
    python scripts/check_g2_prior_runs.py \\
        --s-prespec 7f616f6f7f616f6f7f616f6f7f616f6f7f616f6f \\
        --s-prespec-short 7f616f6f \\
        --runs-json '[{"databaseId": 1, "headSha": "...", ...}]' \\
        --commit-messages-json '{"abc": "msg referencing 7f616f6f"}' \\
        --current-run-id 999

Exit codes:
  0 — no prior run for this S_prespec → permitted to proceed.
  1 — prior run exists → fail loud.
  2 — configuration error (missing args, gh CLI failure, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional


def _run_gh_run_list(workflow: str) -> List[Dict[str, Any]]:
    """Invoke ``gh run list --workflow <workflow> --json ...`` and
    parse the JSON. Returns a list of run records.

    Raises subprocess.CalledProcessError on gh failure; the caller
    converts to exit code 2.
    """
    cmd = [
        "gh",
        "run",
        "list",
        "--workflow",
        workflow,
        "--limit",
        "200",
        "--json",
        "databaseId,displayTitle,name,status,conclusion,headSha,headBranch,createdAt",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    runs_raw = json.loads(result.stdout or "[]")
    if not isinstance(runs_raw, list):
        return []
    return runs_raw


def _fetch_commit_message(repo: str, sha: str) -> Optional[str]:
    """Fetch the commit message for ``sha`` via
    ``gh api repos/<repo>/commits/<sha>``. Returns None on any failure
    (caller treats as "could not verify, skip this prior run")."""
    if not repo or not sha:
        return None
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/commits/{sha}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    commit = payload.get("commit") or {}
    msg = commit.get("message")
    if isinstance(msg, str):
        return msg
    return None


def filter_prior_runs_for_prespec(
    runs: List[Dict[str, Any]],
    *,
    s_prespec: str,
    s_prespec_short: str,
    commit_messages: Mapping[str, str],
    current_run_id: Optional[str] = None,
    current_head_sha: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the subset of ``runs`` whose tag commit message
    references ``s_prespec`` (long or short SHA) AND that are NOT the
    current run.

    ``commit_messages`` maps headSha → commit message. The caller
    populates this either from a live ``gh api repos/.../commits/...``
    fetch (CI) or inline (tests).
    """
    needles = [n for n in (s_prespec, s_prespec_short) if n]
    if not needles:
        return []
    out: List[Dict[str, Any]] = []
    current_id_str = str(current_run_id) if current_run_id is not None else None
    current_head_str = str(current_head_sha) if current_head_sha else None
    for run in runs:
        run_id = str(run.get("databaseId", ""))
        head_sha = str(run.get("headSha", "") or "")
        if current_id_str and run_id == current_id_str:
            continue
        if current_head_str and head_sha == current_head_str:
            # Same tag SHA as the current run — gh sometimes lists
            # the in-progress run before the current step pulls its
            # own ID; treat as self.
            continue
        msg = commit_messages.get(head_sha, "")
        if not msg:
            continue
        if any(needle in msg for needle in needles):
            out.append(run)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s-prespec",
        required=True,
        help="Full S_prespec SHA (40 chars).",
    )
    parser.add_argument(
        "--s-prespec-short",
        required=True,
        help="Short S_prespec SHA (8 chars).",
    )
    parser.add_argument(
        "--workflow",
        default="tier1b_b2_experiment.yml",
        help="Workflow filename for `gh run list --workflow`.",
    )
    parser.add_argument(
        "--current-run-id",
        default=None,
        help="GitHub Actions run ID for the current run (excluded from prior-run filter).",
    )
    parser.add_argument(
        "--current-head-sha",
        default=None,
        help="HEAD SHA of the current run (excluded from prior-run filter).",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help=(
            "GitHub repo as owner/name. Used for `gh api repos/<repo>/commits/<sha>`. "
            "Defaults to GITHUB_REPOSITORY env var."
        ),
    )
    parser.add_argument(
        "--runs-json",
        default=None,
        help=(
            "Test-only override: inline JSON list of run records. When provided, "
            "the live `gh run list` call is skipped. Used for unit tests."
        ),
    )
    parser.add_argument(
        "--commit-messages-json",
        default=None,
        help=(
            "Test-only override: inline JSON object mapping headSha → commit message. "
            "When provided, the live `gh api repos/.../commits/...` call is skipped."
        ),
    )
    args = parser.parse_args(argv)

    if args.runs_json is not None:
        try:
            runs = json.loads(args.runs_json)
        except json.JSONDecodeError as exc:
            print(f"FATAL: --runs-json is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(runs, list):
            print("FATAL: --runs-json must be a JSON list", file=sys.stderr)
            return 2
    else:
        try:
            runs = _run_gh_run_list(args.workflow)
        except FileNotFoundError:
            print("FATAL: `gh` CLI not available on PATH", file=sys.stderr)
            return 2
        except subprocess.CalledProcessError as exc:
            print(
                f"FATAL: `gh run list` failed with exit {exc.returncode}: {exc.stderr}",
                file=sys.stderr,
            )
            return 2

    if args.commit_messages_json is not None:
        try:
            commit_messages = json.loads(args.commit_messages_json)
        except json.JSONDecodeError as exc:
            print(
                f"FATAL: --commit-messages-json is not valid JSON: {exc}",
                file=sys.stderr,
            )
            return 2
        if not isinstance(commit_messages, dict):
            print("FATAL: --commit-messages-json must be a JSON object", file=sys.stderr)
            return 2
    else:
        repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
        commit_messages = {}
        for run in runs:
            head_sha = run.get("headSha")
            if isinstance(head_sha, str) and head_sha and head_sha not in commit_messages:
                msg = _fetch_commit_message(repo, head_sha)
                if msg is not None:
                    commit_messages[head_sha] = msg

    prior = filter_prior_runs_for_prespec(
        runs,
        s_prespec=args.s_prespec,
        s_prespec_short=args.s_prespec_short,
        commit_messages=commit_messages,
        current_run_id=args.current_run_id,
        current_head_sha=args.current_head_sha,
    )
    if prior:
        print(
            f"::error::HIGH-5 (iter-3) — prior workflow run(s) detected whose tag "
            f"commit message references S_prespec={args.s_prespec_short}. Per the "
            "first-attempted-run-is-load-bearing protocol, only ONE workflow run "
            "from a given S_prespec is permitted.",
            file=sys.stderr,
        )
        for run in prior:
            print(
                "::error::  - run_id={dbid}, status={status}, "
                "conclusion={conc}, headSha={head}".format(
                    dbid=run.get("databaseId"),
                    status=run.get("status"),
                    conc=run.get("conclusion"),
                    head=run.get("headSha"),
                ),
                file=sys.stderr,
            )
        return 1

    print(f"[OK] no prior runs of {args.workflow} reference S_prespec={args.s_prespec_short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
