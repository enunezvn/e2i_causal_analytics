"""Tests for the deploy-currency alarm (.github/scripts/check_deploy_currency.py).

The alarm exists because a flaked test shard can SKIP the whole deploy chain
while every visible job is green (#1962, #1963): run 34175425118 left production
a commit behind with eleven green jobs, and the existing drift guard could not
say so because it runs *inside* the deploy job that was skipped.

The central case below is that incident, replayed. The API cannot serve it any
more -- the run was re-run to recover, so it now reports ``success`` -- so the
historical run state is injected into the pure ``classify()`` function, which is
exactly why the I/O is kept out of it.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "check_deploy_currency.py"

_SCRIPT_DIR = str(REPO_ROOT / ".github" / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import check_deploy_currency as cdc  # noqa: E402,I001

NOW = datetime(2026, 9, 8, 4, 0, 0, tzinfo=timezone.utc)

# The real incident (#1962). 1cce45132 was live; 1022ab202 was the #1960 merge
# whose Deploy to Droplet job was SKIPPED after a flaked shard cancelled the run.
RUNNING = "1cce45132" + "0" * 31
MISSED = "1022ab202" + "0" * 31


def _run(conclusion="cancelled", status="completed", age_minutes=180, run_id=34175425118):
    stamp = (NOW - timedelta(minutes=age_minutes)).isoformat().replace("+00:00", "Z")
    return {
        "id": run_id,
        "run_number": 1,
        "status": status,
        "conclusion": conclusion,
        "updated_at": stamp,
        "html_url": f"https://github.com/enunezvn/e2i_causal_analytics/actions/runs/{run_id}",
    }


def _classify(**kw):
    base = {
        "running_sha": RUNNING,
        "head_sha": MISSED,
        "missed_pushes": [MISSED],
        "runs_by_sha": {MISSED: _run()},
        "now": NOW,
        "grace_minutes": 45,
    }
    base.update(kw)
    return cdc.classify(**base)


class TestTheIncident:
    """Run 34175425118: deploy chain skipped, eleven jobs green, prod behind."""

    def test_skipped_deploy_past_grace_is_an_alarm(self):
        result = _classify()
        assert result["verdict"] == cdc.ALARM, result
        assert result["behind"][0]["run_id"] == 34175425118
        assert "1022ab202" in result["reason"]

    def test_alarm_exit_code_is_nonzero(self):
        # The workflow's report step keys off this.
        assert cdc.EXIT[cdc.ALARM] == 1
        assert cdc.EXIT[cdc.FRESH] == 0 and cdc.EXIT[cdc.HOLD] == 0
        assert cdc.EXIT[cdc.UNKNOWN] == 2

    @pytest.mark.parametrize("conclusion", ["cancelled", "failure", "skipped", None])
    def test_every_non_success_ending_alarms(self, conclusion):
        assert _classify(runs_by_sha={MISSED: _run(conclusion=conclusion)})["verdict"] == cdc.ALARM

    def test_success_that_did_not_reach_the_box_still_alarms(self):
        """Run 34164682392 reported success having deployed a non-trigger sha.

        A green conclusion is not evidence that the content is live; the running
        container is. If the box is still behind past the grace window, that is
        an alarm whatever the run says.
        """
        assert _classify(runs_by_sha={MISSED: _run(conclusion="success")})["verdict"] == cdc.ALARM


class TestTransientsAreNotDrift:
    """The deploy legitimately lags while images build (deploy.yml select_built_sha)."""

    def test_in_flight_run_holds(self):
        r = _classify(runs_by_sha={MISSED: _run(status="in_progress", conclusion=None)})
        assert r["verdict"] == cdc.HOLD

    def test_queued_run_holds(self):
        r = _classify(runs_by_sha={MISSED: _run(status="queued", conclusion=None)})
        assert r["verdict"] == cdc.HOLD

    def test_recent_failure_holds_until_grace_expires(self):
        assert _classify(runs_by_sha={MISSED: _run(age_minutes=10)})["verdict"] == cdc.HOLD

    def test_grace_boundary_is_the_only_difference(self):
        """Same run, two ages -- the ONLY thing separating hold from alarm."""
        assert _classify(runs_by_sha={MISSED: _run(age_minutes=44)})["verdict"] == cdc.HOLD
        assert _classify(runs_by_sha={MISSED: _run(age_minutes=46)})["verdict"] == cdc.ALARM


class TestLegitimateLag:
    """A push that triggers no deploy must never raise an alarm."""

    def test_running_equals_head_is_fresh(self):
        assert _classify(head_sha=RUNNING, missed_pushes=[])["verdict"] == cdc.FRESH

    def test_pushes_without_a_deploy_run_are_fresh(self):
        """A docs-only merge creates no run and legitimately never deploys."""
        r = _classify(missed_pushes=[MISSED], runs_by_sha={})
        assert r["verdict"] == cdc.FRESH
        assert r["behind"] == []

    def test_only_triggering_pushes_are_counted(self):
        docs_only = "aaaaaaaaa" + "0" * 31
        r = _classify(
            missed_pushes=[docs_only, MISSED],
            runs_by_sha={MISSED: _run()},
        )
        assert r["verdict"] == cdc.ALARM
        assert [row["sha"] for row in r["behind"]] == [MISSED], "docs-only push must not be listed"


class TestFirstParentIsThePushUnit:
    """The disproved first design walked EVERY commit and would alarm forever.

    Runs are keyed to the push, not to each commit: a branch's interior commits
    arrive inside a merge and have no run of their own. Pinned against this
    repository's real history so the lesson cannot be quietly reverted.
    """

    def test_branch_interior_commits_are_not_push_units(self):
        # e5becbccf is a merge; 0c2b16579 is an interior commit of the branch it
        # merged, and it touches src/repositories/ -- a deploy-trigger path with
        # no run of its own.
        out = subprocess.run(
            ["git", "rev-list", "--first-parent", "e35870e59..e5becbccf"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert any(s.startswith("e5becbccf") for s in out)
        assert not any(s.startswith("0c2b16579") for s in out), (
            "0c2b16579 is a branch-interior commit; counting it as a push unit is "
            "what made the first design alarm on every merged branch"
        )

    def test_helper_returns_newest_first(self):
        got = cdc.first_parent_between("1cce45132", "e5becbccf", cwd=str(REPO_ROOT))
        assert got, "expected first-parent commits between these two merges"
        assert got[0].startswith("e5becbccf")


class TestNewestAttemptWins:
    def test_rerun_supersedes_the_failed_attempt(self):
        """A re-run is how 34175425118 was recovered; the newest attempt is truth."""
        old = _run(conclusion="cancelled", run_id=1)
        old["run_number"] = 1
        new = _run(conclusion="success", run_id=2)
        new["run_number"] = 2
        assert new["run_number"] > old["run_number"]


class TestObservabilityFailureIsNeverDrift:
    """maintenance-freshness.yml's lesson: an outage must not read as a verdict."""

    def test_unknown_sha_is_unknown_not_alarm(self):
        rc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--running-sha",
                "d" * 40,
                "--repo-dir",
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
        )
        assert rc.returncode == cdc.EXIT[cdc.UNKNOWN], rc.stdout + rc.stderr
        assert "UNKNOWN" in rc.stdout

    def test_empty_running_sha_is_unknown(self):
        rc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--running-sha", "", "--repo-dir", str(REPO_ROOT)],
            capture_output=True,
            text=True,
        )
        assert rc.returncode == cdc.EXIT[cdc.UNKNOWN]


class TestWiring:
    def test_script_exists_and_is_executable(self):
        assert SCRIPT_PATH.is_file()

    def test_workflow_invokes_the_script(self):
        wf = (REPO_ROOT / ".github" / "workflows" / "deploy-currency.yml").read_text()
        assert "check_deploy_currency.py" in wf

    def test_workflow_is_not_a_deploy_trigger(self):
        """.github/** is deliberately absent from deploy.yml's paths list.

        This whole change must be mergeable without rebuilding production.
        """
        deploy = (REPO_ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        head = deploy.split("jobs:")[0]
        assert ".github" not in head, (
            "deploy.yml now triggers on .github/** -- the currency workflow would "
            "redeploy production every time it is edited"
        )
