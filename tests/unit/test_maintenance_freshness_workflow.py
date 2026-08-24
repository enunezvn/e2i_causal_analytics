"""Contract test: the maintenance-freshness workflow is the UNATTENDED caller of
``check_maintenance_freshness.sh`` (#1798, final item).

#1798: the e2i-maintenance cron layer was dead for eight weeks and nothing said
so. #1799/#1802 built a freshness check keyed on success stamps, #1803 gave it a
caller (``health_check.sh``) -- but that caller is manual-only: it is in no
crontab and ``deploy.yml`` never invokes it. So the next silent stop is still
detected only when a human happens to run a script. The alarm needs to fire on
its own, from somewhere independent of the crontab it audits, into a surface
that is actually read. ``slow-tests.yml`` already proves that shape: a scheduled
workflow that files/updates a labelled GitHub issue on failure.

This test pins that contract so a later edit cannot quietly drop the schedule,
swallow the failure before it reaches the reporter, or re-create the #615 class
of muted alarm (no GH_REPO without a checkout; a dedup label nobody created).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOW_PATH = WORKFLOWS / "maintenance-freshness.yml"
DEPLOY_PATH = WORKFLOWS / "deploy.yml"

FRESHNESS_SCRIPT = "check_maintenance_freshness.sh"
DEDUP_LABEL = "maintenance-freshness-failure"
DEPLOY_SSH_SECRETS = {"DEPLOY_HOST", "DEPLOY_USER", "DEPLOY_SSH_KEY"}


def _load(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _triggers(workflow: dict) -> dict:
    # PyYAML parses the bare mapping key ``on:`` as the boolean ``True``.
    return workflow.get("on", workflow.get(True, {})) or {}


def _jobs(workflow: dict) -> dict:
    return workflow.get("jobs") or {}


def _ssh_steps(job: dict) -> list[dict]:
    return [s for s in (job.get("steps") or []) if "appleboy/ssh-action" in str(s.get("uses", ""))]


def _check_job(workflow: dict) -> tuple[str, dict]:
    """The job that actually runs the freshness script over SSH."""
    for name, job in _jobs(workflow).items():
        if any(
            FRESHNESS_SCRIPT in str(s.get("with", {}).get("script", "")) for s in _ssh_steps(job)
        ):
            return name, job
    raise AssertionError(f"no job runs {FRESHNESS_SCRIPT} over SSH")


def _freshness_step(job: dict) -> dict:
    return next(s for s in _ssh_steps(job) if FRESHNESS_SCRIPT in str(s["with"]["script"]))


def _script_lines(step: dict) -> list[str]:
    """The remote script's effective lines: no blanks, no comments."""
    return [
        line.strip()
        for line in str(step["with"]["script"]).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _report_job(workflow: dict) -> tuple[str, dict]:
    for name, job in _jobs(workflow).items():
        if "gh issue create" in str(job):
            return name, job
    raise AssertionError("no job files an issue with `gh issue create`")


def test_workflow_exists() -> None:
    assert WORKFLOW_PATH.exists(), (
        f"{WORKFLOW_PATH.name} is missing -- the freshness check has no unattended caller "
        "and the next silent cron stop is another eight weeks (#1798)"
    )


def test_runs_on_a_schedule_and_can_be_dispatched() -> None:
    triggers = _triggers(_load(WORKFLOW_PATH))
    assert "schedule" in triggers, "must run unattended -- a manual-only caller already exists"
    assert triggers["schedule"], "schedule list is empty"
    assert "workflow_dispatch" in triggers, (
        "must be dispatchable so the alarm can be live-certified"
    )


def test_check_job_runs_the_script_on_the_box_with_the_deploy_secrets() -> None:
    """Same SSH secrets as deploy.yml, so a secret rename cannot orphan one of them."""
    _, job = _check_job(_load(WORKFLOW_PATH))
    steps = _ssh_steps(job)
    assert steps, "the check job has no SSH step"
    for step in steps:
        with_block = step["with"]
        used = {
            key
            for key in DEPLOY_SSH_SECRETS
            if f"secrets.{key}"
            in str(
                with_block.get(
                    {"DEPLOY_HOST": "host", "DEPLOY_USER": "username", "DEPLOY_SSH_KEY": "key"}[
                        key
                    ],
                    "",
                )
            )
        }
        assert used == DEPLOY_SSH_SECRETS, (
            f"SSH step {step.get('name')!r} must use exactly the deploy secrets; uses {sorted(used)}"
        )
    deploy_text = DEPLOY_PATH.read_text()
    for key in DEPLOY_SSH_SECRETS:
        assert f"secrets.{key}" in deploy_text, f"{key} is not what deploy.yml uses any more"


def test_check_job_is_bounded_and_does_not_swallow_its_own_failure() -> None:
    """A swallowed failure never reaches the reporter -- the alarm would be mute."""
    name, job = _check_job(_load(WORKFLOW_PATH))
    assert job.get("timeout-minutes") is not None, f"{name}: needs an explicit timeout"
    assert not job.get("continue-on-error"), f"{name}: continue-on-error would mute the alarm"
    for step in _ssh_steps(job):
        assert not step.get("continue-on-error"), (
            f"{name}: SSH step continue-on-error would mute the alarm"
        )


def test_the_freshness_invocation_cannot_be_swallowed_inside_the_ssh_script() -> None:
    """codex iter-1 (MED): `continue-on-error` is not the only mute. A `|| true` on the
    invocation, or dropping `set -e` above it, silences the alarm with the job green --
    and the removed appleboy `script_stop` input would not catch either."""
    _, job = _check_job(_load(WORKFLOW_PATH))
    lines = _script_lines(_freshness_step(job))
    assert lines[0].startswith("set -") and "e" in lines[0].split()[1] and "pipefail" in lines[0], (
        f"the remote script must open with `set -e ... pipefail`; opens with {lines[0]!r}"
    )
    # The line that EXECUTES the script (first token), not the echo that names it.
    invocation = [line for line in lines if line.split()[0].endswith(FRESHNESS_SCRIPT)]
    assert len(invocation) == 1, f"expected exactly one invocation, got {invocation}"
    for mute in ("|| true", "|| :", "; true", "|| echo", "|| exit 0"):
        assert mute not in invocation[0], (
            f"the invocation is guarded with {mute!r}: {invocation[0]}"
        )


def test_ssh_steps_do_not_rely_on_the_removed_script_stop_input() -> None:
    """appleboy/ssh-action@v1 has no `script_stop` input (its README: 'removed ... add
    `set -e`'). Carrying it reads as a safety contract while doing nothing."""
    _, job = _check_job(_load(WORKFLOW_PATH))
    for step in _ssh_steps(job):
        assert "script_stop" not in (step.get("with") or {}), (
            f"{step.get('name')!r} carries the dead `script_stop` input"
        )


def test_reporter_distinguishes_could_not_check_from_stale() -> None:
    """codex iter-1 (HIGH): an SSH/auth/network failure must not be filed as
    'the cron is stale' -- that sends a human to PAM and stamps when the check never
    ran. The check job classifies its own outcome (preflight SSH step vs. the freshness
    step) and the reporter titles the issue from that verdict."""
    workflow = _load(WORKFLOW_PATH)
    check_name, job = _check_job(workflow)
    ssh = _ssh_steps(job)
    freshness = _freshness_step(job)
    assert ssh.index(freshness) >= 1, "a preflight SSH step must run BEFORE the freshness step"
    preflight = ssh[ssh.index(freshness) - 1]
    assert preflight.get("id") and freshness.get("id"), "both SSH steps need ids to be classified"
    assert FRESHNESS_SCRIPT not in str(preflight["with"]["script"]), (
        "the preflight must not run the check itself, or its failure is ambiguous again"
    )

    verdict_ref = str((job.get("outputs") or {}).get("verdict", ""))
    assert "steps." in verdict_ref and ".outputs.verdict" in verdict_ref, (
        f"{check_name} must expose outputs.verdict from a step; got {verdict_ref!r}"
    )
    classify_id = verdict_ref.split("steps.")[1].split(".")[0]
    classify = next(s for s in job["steps"] if s.get("id") == classify_id)
    assert "always()" in str(classify.get("if", "")), (
        "the classify step must run after a failed step, or the verdict is never set"
    )
    classify_text = str(classify)
    for step in (preflight, freshness):
        assert f"steps.{step['id']}.outcome" in classify_text, (
            f"classify must read steps.{step['id']}.outcome"
        )
    for verdict in ("unreachable", "stale", "fresh"):
        assert verdict in str(classify.get("run", "")), f"classify never emits {verdict!r}"

    _, report = _report_job(workflow)
    report_text = str(report)
    assert f"needs.{check_name}.outputs.verdict" in report_text, (
        "the reporter must read the verdict, not infer 'stale' from any failure"
    )
    assert "unreachable" in report_text and "stale" in report_text, (
        "the reporter must branch its title/body on the verdict"
    )


def test_ssh_step_forwards_every_env_it_declares() -> None:
    """appleboy/ssh-action forwards only what `envs:` names (deploy.yml, ROLLOUT_OUTCOME)."""
    _, job = _check_job(_load(WORKFLOW_PATH))
    for step in _ssh_steps(job):
        declared = set((step.get("env") or {}).keys())
        forwarded = {
            e.strip() for e in str(step.get("with", {}).get("envs", "")).split(",") if e.strip()
        }
        assert declared <= forwarded, (
            f"env declared but not forwarded across SSH: {sorted(declared - forwarded)}"
        )


def test_reporter_fires_on_check_failure_and_can_write_issues() -> None:
    workflow = _load(WORKFLOW_PATH)
    check_name, _ = _check_job(workflow)
    report_name, job = _report_job(workflow)
    needs = job.get("needs")
    needs = [needs] if isinstance(needs, str) else list(needs or [])
    assert check_name in needs, f"{report_name} must depend on {check_name}"
    cond = str(job.get("if", ""))
    assert "always()" in cond, (
        "a `needs`-dependent job is SKIPPED when its dependency fails unless always()"
    )
    assert f"needs.{check_name}.result == 'failure'" in cond, (
        "reporter must key on the check job's failure"
    )
    perms = job.get("permissions") or workflow.get("permissions") or {}
    assert perms.get("issues") == "write", "reporter cannot file an issue without issues: write"


def test_reporter_avoids_the_615_class_of_muted_alarm() -> None:
    """No checkout -> GH_REPO required; `gh issue create --label` needs the label to exist; dedup by label."""
    _, job = _report_job(_load(WORKFLOW_PATH))
    text = str(job)
    assert "GH_REPO" in text, "without a checkout `gh issue` dies 'not a git repository' (#615)"
    assert f"gh label create {DEDUP_LABEL}" in text, (
        "the dedup label must be self-healed before create (#615)"
    )
    assert f"--label {DEDUP_LABEL}" in text, "issues must carry the dedup label"
    assert "gh issue list" in text and "gh issue comment" in text, (
        "must comment on an existing open issue, not pile up duplicates"
    )
    assert not any("actions/checkout" in str(s.get("uses", "")) for s in job.get("steps") or []), (
        "the reporter needs no checkout; if one is added, GH_REPO stops being the thing that saves it"
    )


def test_the_freshness_check_is_still_NOT_installed_as_a_cron_job() -> None:
    """The constraint that motivated the design must survive adding a caller."""
    setup = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "setup_cron.sh"
    assert FRESHNESS_SCRIPT not in setup.read_text()


def _preflight_step(job: dict) -> dict:
    ssh = _ssh_steps(job)
    return ssh[ssh.index(_freshness_step(job)) - 1]


def test_preflight_survives_a_missing_crontab_so_the_script_can_report_it(tmp_path: Path) -> None:
    """codex iter-2 (HIGH): the preflight must be fatal for CONNECT/CHECKOUT only. If it
    dies on an unreadable crontab, the verdict is `unreachable` and the freshness step
    -- whose documented rc=2 is exactly 'crontab unreadable' -- never runs.

    Executes the real preflight script text with only its paths substituted: a
    throwaway git checkout, and a crontab path that does not exist.
    """
    _, job = _check_job(_load(WORKFLOW_PATH))
    script = str(_preflight_step(job)["with"]["script"])
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    missing_crontab = tmp_path / "no-such-crontab"
    assert (
        "/home/enunez/Projects/e2i_causal_analytics" in script
        and "/etc/cron.d/e2i-maintenance" in script
    )
    substituted = script.replace(
        "/home/enunez/Projects/e2i_causal_analytics", str(checkout)
    ).replace("/etc/cron.d/e2i-maintenance", str(missing_crontab))
    result = subprocess.run(["bash", "-c", substituted], capture_output=True, text=True)
    assert result.returncode == 0, (
        "the preflight died on a missing crontab (rc="
        f"{result.returncode}) -- that files 'could not reach the droplet' for an uninstalled "
        f"maintenance layer instead of letting the check say rc=2\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "checkout:" in result.stdout, "the preflight went quiet instead of printing the checkout"


_OUTCOMES_PREFLIGHT = ("success", "failure", "cancelled")
_OUTCOMES_FRESHNESS = ("success", "failure", "skipped", "cancelled")


def _expected_verdict(preflight: str, freshness: str) -> str:
    if preflight != "success":
        return "unreachable"
    return {"success": "fresh", "failure": "stale"}.get(freshness, "unknown")


@pytest.mark.parametrize("preflight", _OUTCOMES_PREFLIGHT)
@pytest.mark.parametrize("freshness", _OUTCOMES_FRESHNESS)
def test_classify_maps_every_outcome_pair_to_the_right_verdict(
    preflight: str, freshness: str, tmp_path: Path
) -> None:
    """codex iter-2 (MED): the presence of the words 'stale'/'unreachable' in the classify
    block does not prove the mapping. Execute the real block over the outcome matrix."""
    _, job = _check_job(_load(WORKFLOW_PATH))
    classify = next(s for s in job["steps"] if s.get("id") == "classify")
    github_output = tmp_path / "out"
    github_output.write_text("")
    env = {
        **os.environ,
        "PREFLIGHT": preflight,
        "FRESHNESS": freshness,
        "GITHUB_OUTPUT": str(github_output),
    }
    result = subprocess.run(
        ["bash", "-c", str(classify["run"])], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (
        github_output.read_text().strip() == f"verdict={_expected_verdict(preflight, freshness)}"
    ), f"preflight={preflight} freshness={freshness}: {github_output.read_text().strip()!r}"
