#!/usr/bin/env python3
"""Is what production is RUNNING the newest thing it should be running?

The gap this closes (#1962, #1963)
----------------------------------
``scripts/deploy/check_image_drift.py`` is the existing drift guard, but it is
invoked from exactly one place -- ``.github/workflows/deploy.yml`` -- i.e. from
*inside* the deploy job. When a flaked test shard makes ``Backend CI Success``
fail or cancel, the deploy job is **skipped**, so the drift check never runs and
nothing anywhere reports that ``origin/main`` moved without production moving
with it. That is precisely how run 34175425118 left prod a commit behind with
eleven green jobs and no red anywhere.

This check is the unattended outside observer: it runs on a schedule,
independent of the deploy it audits, the same way ``maintenance-freshness.yml``
audits the cron layer it must not live inside.

Why the unit of analysis is a FIRST-PARENT commit
-------------------------------------------------
The obvious formulation -- "for every commit between the running sha and
origin/main, did it touch a deploy-trigger path?" -- requires duplicating
``deploy.yml``'s ``paths:`` list, which has 17 patterns and zero negated ones
and has already been got wrong twice by pattern-matching instead of reading.

So this does not re-derive it. **GitHub already evaluates that filter**: a push
to main creates a ``Deploy to Production`` run if and only if the push matched.
The existence of a run is therefore the authoritative answer to "should this
have deployed", with no list to drift from.

The subtlety, measured rather than assumed (this disproved the first design):
runs are keyed to the **push**, not to each commit. A branch's individual
commits arrive on main inside a merge and never had their own push event, so
they have no run of their own -- ``0c2b16579`` touches ``src/repositories/``
(a trigger path) and has no run, because it landed inside merge ``e5becbccf``.
Walking every commit therefore reports a merged branch's interior commits as
"should have deployed but didn't" and the alarm screams forever. Walking
**first-parent** matches the push units exactly. Verified against 12 real
pushes: every one that touched a trigger path has a run, and the two that do
not (``database/`` + ``tests/``, and ``docker/alertmanager/``) are correctly
non-triggering.

Why the running CONTAINER is the anchor, not a job conclusion
-------------------------------------------------------------
A run's conclusion does not tell you what is live. Run 34175425118 reported
``cancelled`` while deploying nothing; 34164682392 reported ``success`` having
deployed a sha that was not its trigger, because ``deploy.yml`` hard-resets the
droplet to ``origin/main`` and walks it newest-first for the newest ancestor
with both images published (``select_built_sha``). So the anchor is
``docker inspect e2i_api --format '{{.Config.Image}}'`` and everything else is
read relative to it.

That same walk is why a lagging image is not automatically drift: when HEAD's
images are still building, deploying an ancestor is correct and temporary.
``--grace-minutes`` is what separates that transient from a real stall, and a
run that has not reached a terminal status is always HOLD.

Verdicts / exit codes
---------------------
``fresh`` (0)   running sha == origin/main, or every push between them was
                non-triggering (a docs-only merge legitimately does not deploy).
``hold`` (0)    a deploy for the missed push is still in flight, or is terminal
                but younger than the grace window. Transient, not drift.
``alarm`` (1)   at least one deploy-triggering push is older than the grace
                window and is still not live. Production is behind.
``unknown`` (2) the question could not be answered (running sha absent from
                history, no runs readable). NEVER reported as drift -- an
                observability outage must not masquerade as a clean bill of
                health, nor as an alarm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

DEPLOY_WORKFLOW = "deploy.yml"
DEPLOY_JOB_NAME = "Deploy to Droplet"
DEFAULT_GRACE_MINUTES = 45
# How far back along first-parent to look before giving up. The deploy's own
# ancestor walk uses a 30-commit window (deploy.yml select_built_sha); going
# meaningfully deeper than that describes a box nobody has deployed to in days,
# which is an alarm on its own terms.
MAX_WALK = 60

FRESH, HOLD, ALARM, UNKNOWN = "fresh", "hold", "alarm", "unknown"
EXIT = {FRESH: 0, HOLD: 0, ALARM: 1, UNKNOWN: 2}


def _run(cmd: list[str], cwd: str | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def first_parent_between(running_sha: str, head_sha: str, cwd: str | None = None) -> list[str]:
    """First-parent commits in (running_sha, head_sha], newest first.

    These are the push units. An empty list means the box is current.
    """
    out = _run(["git", "rev-list", "--first-parent", f"{running_sha}..{head_sha}"], cwd=cwd)
    return [line for line in out.splitlines() if line][:MAX_WALK]


def is_ancestor(maybe_ancestor: str, descendant: str, cwd: str | None = None) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", maybe_ancestor, descendant],
            cwd=cwd,
            capture_output=True,
        ).returncode
        == 0
    )


def commit_exists(sha: str, cwd: str | None = None) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=cwd, capture_output=True
        ).returncode
        == 0
    )


def fetch_deploy_runs(repo: str, per_page: int = 100) -> dict[str, dict]:
    """head_sha -> run, for recent Deploy to Production runs on main.

    Keyed by head_sha because that is the push this run belongs to. Only the
    newest run per sha is kept: a re-run heals the situation (that is how
    34175425118 was recovered), and the newest attempt is the current truth.
    """
    raw = _run(
        [
            "gh",
            "api",
            f"/repos/{repo}/actions/workflows/{DEPLOY_WORKFLOW}/runs"
            f"?branch=main&per_page={per_page}",
        ]
    )
    runs: dict[str, dict] = {}
    for run in json.loads(raw).get("workflow_runs", []):
        sha = run.get("head_sha")
        if not sha:
            continue
        prev = runs.get(sha)
        if prev is None or (run.get("run_number") or 0) > (prev.get("run_number") or 0):
            runs[sha] = run
    return runs


def _age_minutes(run: dict, now: datetime) -> float | None:
    stamp = run.get("updated_at") or run.get("created_at")
    if not stamp:
        return None
    when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return (now - when).total_seconds() / 60.0


def classify(  # noqa: PLR0913
    *,
    running_sha: str,
    head_sha: str,
    missed_pushes: list[str],
    runs_by_sha: dict[str, dict],
    now: datetime,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
) -> dict[str, Any]:
    """Pure verdict function -- every I/O decision is already made by the caller.

    ``missed_pushes`` is first-parent (running_sha, head_sha], newest first.
    """
    if running_sha == head_sha:
        return {
            "verdict": FRESH,
            "reason": "the running image is origin/main",
            "behind": [],
        }

    # Only pushes that actually triggered a deploy count. A docs-only merge
    # creates no run and legitimately never reaches the box.
    triggering = [sha for sha in missed_pushes if sha in runs_by_sha]
    if not triggering:
        return {
            "verdict": FRESH,
            "reason": (
                f"{len(missed_pushes)} push(es) ahead of the running image, none of "
                "which triggered a deploy (no Deploy to Production run exists for "
                "them) -- so nothing is missing from production"
            ),
            "behind": [],
        }

    behind: list[dict[str, Any]] = []
    for sha in triggering:
        run = runs_by_sha[sha]
        behind.append(
            {
                "sha": sha,
                "run_id": run.get("id"),
                "run_url": run.get("html_url"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "age_minutes": _age_minutes(run, now),
            }
        )

    newest = behind[0]
    if newest["status"] != "completed":
        return {
            "verdict": HOLD,
            "reason": (
                f"the newest deploy-triggering push {newest['sha'][:9]} is still "
                f"{newest['status']} -- a build/deploy in flight is not drift"
            ),
            "behind": behind,
        }

    age = newest["age_minutes"]
    if age is not None and age < grace_minutes:
        return {
            "verdict": HOLD,
            "reason": (
                f"the newest deploy-triggering push {newest['sha'][:9]} finished "
                f"{age:.0f}m ago, inside the {grace_minutes}m grace window "
                "(deploy.yml may still be rolling it out)"
            ),
            "behind": behind,
        }

    age_text = f"{age:.0f}m" if age is not None else "an unknown time"
    return {
        "verdict": ALARM,
        "reason": (
            f"{len(behind)} deploy-triggering push(es) are not live; the newest, "
            f"{newest['sha'][:9]}, has a Deploy to Production run that ended "
            f"'{newest['conclusion']}' "
            f"{age_text} ago"
        ),
        "behind": behind,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--running-sha",
        required=True,
        help="sha of the image production is running "
        "(docker inspect e2i_api --format '{{.Config.Image}}' -> tag)",
    )
    ap.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "enunezvn/e2i_causal_analytics"),
    )
    ap.add_argument("--head-ref", default="origin/main")
    ap.add_argument("--grace-minutes", type=int, default=DEFAULT_GRACE_MINUTES)
    ap.add_argument("--repo-dir", default=None, help="git checkout to read (default: cwd)")
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    args = ap.parse_args(argv)

    cwd = args.repo_dir
    running = args.running_sha.strip()

    def bail(reason: str) -> int:
        result: dict[str, Any] = {"verdict": UNKNOWN, "reason": reason, "behind": []}
        _emit(result, args.json)
        return EXIT[UNKNOWN]

    if not running:
        return bail("no running sha was supplied -- the droplet read failed")
    if not commit_exists(running, cwd):
        # Do NOT call this drift: an image built from a commit this checkout
        # cannot see is an observability failure, not a deployment failure.
        return bail(
            f"the running sha {running[:9]} is not in this checkout's history "
            "(shallow clone, or an image built from a commit not on main)"
        )

    head = _run(["git", "rev-parse", args.head_ref], cwd=cwd)

    if not is_ancestor(running, head):
        result: dict[str, Any] = {
            "verdict": ALARM,
            "reason": (
                f"the running sha {running[:9]} is NOT an ancestor of "
                f"{args.head_ref} ({head[:9]}) -- production is on a commit that "
                "is not on main (a rollback that was never followed up, or a "
                "hand-deployed image)"
            ),
            "behind": [],
        }
        _emit(result, args.json)
        return EXIT[ALARM]

    try:
        runs_by_sha = fetch_deploy_runs(args.repo)
    except subprocess.CalledProcessError as exc:
        return bail(f"could not read Deploy to Production runs: {exc.stderr.strip()[:200]}")
    if not runs_by_sha:
        return bail("the deploy workflow returned no runs at all -- API or auth problem")

    result = classify(
        running_sha=running,
        head_sha=head,
        missed_pushes=first_parent_between(running, head, cwd),
        runs_by_sha=runs_by_sha,
        now=datetime.now(timezone.utc),
        grace_minutes=args.grace_minutes,
    )
    result["running_sha"] = running
    result["head_sha"] = head
    _emit(result, args.json)
    return EXIT[result["verdict"]]


def _emit(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return
    print(f"==> verdict: {result['verdict'].upper()}")
    print(f"==> {result['reason']}")
    for row in result.get("behind", []):
        print(
            f"    - {row['sha'][:9]}  run {row['run_id']}  "
            f"{row['status']}/{row['conclusion']}  {row['run_url']}"
        )


if __name__ == "__main__":
    sys.exit(main())
