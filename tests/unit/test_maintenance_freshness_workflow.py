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

from pathlib import Path

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
    step = next(s for s in _ssh_steps(job) if FRESHNESS_SCRIPT in str(s["with"]["script"]))
    with_block = step["with"]
    used = {
        key
        for key in DEPLOY_SSH_SECRETS
        if f"secrets.{key}"
        in str(
            with_block.get(
                {"DEPLOY_HOST": "host", "DEPLOY_USER": "username", "DEPLOY_SSH_KEY": "key"}[key], ""
            )
        )
    }
    assert used == DEPLOY_SSH_SECRETS, (
        f"SSH step must use exactly the deploy secrets; uses {sorted(used)}"
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
