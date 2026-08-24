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
FRESHNESS_SCRIPT_PATH = "scripts/maintenance/" + FRESHNESS_SCRIPT
BOX_CRONTAB = "/etc/cron.d/e2i-maintenance"  # the crontab the check audits on the droplet
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


def _script_lines(step: dict) -> list[str]:
    """The remote script's effective lines: no blanks, no comments."""
    return [
        line.strip()
        for line in str(step["with"]["script"]).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _code_lines(text: str) -> list[str]:
    """Non-blank, non-comment lines of a shell script -- prose must not satisfy a code guard."""
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _invocations(step: dict) -> list[str]:
    """Lines that EXECUTE the freshness script: its path is the first token.

    Naming the script is not running it -- the freshness step echoes its name, and
    the preflight `test -x`es it -- so a substring match would pick the wrong step
    and would call a preflight that merely mentions the script "the check".
    """
    return [line for line in _script_lines(step) if line.split()[0].endswith(FRESHNESS_SCRIPT)]


def _check_job(workflow: dict) -> tuple[str, dict]:
    """The job that actually runs the freshness script over SSH."""
    for name, job in _jobs(workflow).items():
        if any(_invocations(s) for s in _ssh_steps(job)):
            return name, job
    raise AssertionError(f"no job runs {FRESHNESS_SCRIPT} over SSH")


def _freshness_step(job: dict) -> dict:
    return next(s for s in _ssh_steps(job) if _invocations(s))


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
    invocation = _invocations(_freshness_step(job))
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
    assert not _invocations(preflight), (
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
    """No checkout -> GH_REPO required; the dedup label is self-healed; dedup happens.

    #1807 live cert D3 (2026-08-24): `gh label create ... 2>/dev/null || true` hid a
    real failure under GITHUB_TOKEN and `gh issue create --label` then died
    "label not found" -- the alarm filed nothing. An alarm may not discard its
    own errors. (The executed tests below pin the behaviour; this pins the text.)
    """
    _, job = _report_job(_load(WORKFLOW_PATH))
    text = str(job)
    assert "GH_REPO" in text, "without a checkout `gh issue` dies 'not a git repository' (#615)"
    assert DEDUP_LABEL in text and "gh label create" in text, (
        "the dedup label must be self-healed before create (#615)"
    )
    # Code, not prose: the comment explaining D3 is allowed to name the defect.
    code = _code_lines(str(_reporter_step(job)["run"]))
    assert not any("2>/dev/null" in line for line in code), (
        "an alarm must not throw away stderr: that is how D3 filed nothing"
    )
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


def _fake_checkout(root: Path, *, git: bool = True, script: bool = True) -> Path:
    """A directory standing in for the box checkout the preflight `cd`s into.

    `git=False` -> the directory exists but is not a repository; `script=False` ->
    a repository that does not carry the freshness script. Both are shapes the
    preflight must refuse, because a freshness-step failure that follows would
    otherwise be filed as `stale` when the check never ran from the real checkout.
    """
    checkout = root / "checkout"
    checkout.mkdir()
    if git:
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
    if script:
        target = checkout / FRESHNESS_SCRIPT_PATH
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/bash\nexit 0\n")
        target.chmod(0o755)
    return checkout


def _run_preflight(checkout: Path, crontab: Path) -> subprocess.CompletedProcess:
    """Execute the REAL preflight script text with only its two paths substituted."""
    _, job = _check_job(_load(WORKFLOW_PATH))
    step = _preflight_step(job)
    script = str(step["with"]["script"])
    # The box checkout is DERIVED from the script's own `cd` line, not restated
    # here (#410: no developer paths in tests/; and a restated path would drift
    # silently if the checkout ever moved).
    cd_lines = [line for line in _script_lines(step) if line.startswith("cd ")]
    assert len(cd_lines) == 1, f"expected exactly one `cd` in the preflight, got {cd_lines}"
    box_checkout = cd_lines[0].split(None, 1)[1].strip()
    assert box_checkout.startswith("/"), (
        f"the preflight must cd to an absolute path: {cd_lines[0]!r}"
    )
    assert BOX_CRONTAB in script, "the preflight must print the crontab the check will read"
    substituted = script.replace(box_checkout, str(checkout)).replace(BOX_CRONTAB, str(crontab))
    return subprocess.run(["bash", "-c", substituted], capture_output=True, text=True)


def test_preflight_survives_a_missing_crontab_so_the_script_can_report_it(tmp_path: Path) -> None:
    """codex iter-2 (HIGH): the preflight must be fatal for CONNECT/CHECKOUT only. If it
    dies on an unreadable crontab, the verdict is `unreachable` and the freshness step
    -- whose documented rc=2 is exactly 'crontab unreadable' -- never runs.
    """
    result = _run_preflight(_fake_checkout(tmp_path), tmp_path / "no-such-crontab")
    assert result.returncode == 0, (
        "the preflight died on a missing crontab (rc="
        f"{result.returncode}) -- that files 'could not reach the droplet' for an uninstalled "
        f"maintenance layer instead of letting the check say rc=2\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "checkout:" in result.stdout, "the preflight went quiet instead of printing the checkout"


def test_preflight_fails_when_the_directory_is_not_the_checkout(tmp_path: Path) -> None:
    """codex iter-3 (MED): `echo "$(git rev-parse ...)"` cannot fail the step under
    `set -e` -- a failing command substitution used as an echo ARGUMENT is swallowed
    (measured: prints `==> checkout:  ()` and continues, rc=0). A directory that
    exists but is not the checkout would pass preflight, and whatever the freshness
    step then hit would be filed as `stale` -- the wrong diagnosis with the wrong hints.
    """
    # The script IS present, so only the explicit git validation can refuse this.
    result = _run_preflight(_fake_checkout(tmp_path, git=False), tmp_path / "no-such-crontab")
    assert result.returncode != 0, (
        "the preflight passed on a directory that is not a git checkout\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_preflight_fails_when_the_checkout_lacks_the_freshness_script(tmp_path: Path) -> None:
    """Same seam, other half: a checkout without the script (moved, renamed, exec bit
    stripped by a `core.fileMode=false` pull -- #1796) is a checkout the check cannot
    run from, so it is `unreachable`, not `stale`.
    """
    result = _run_preflight(_fake_checkout(tmp_path, script=False), tmp_path / "no-such-crontab")
    assert result.returncode != 0, (
        "the preflight passed on a checkout with no executable "
        f"{FRESHNESS_SCRIPT_PATH}\nstdout={result.stdout}\nstderr={result.stderr}"
    )


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


# --- The reporter, EXECUTED against a `gh` stand-in (#1807 live cert D3) ---------
#
# D3 dispatched the merged workflow with tolerance=0 file_issue=true. The check
# went red and classify said `stale` (correct), the reporter ran (correct) -- and
# filed NOTHING: `gh label create` failed, `2>/dev/null || true` hid why, and
# `gh issue create --label` died "label not found". The hidden error (measured
# on a branch run) was `HTTP 422: description is too long (maximum is 100
# characters)` -- the label description was 103 characters. The self-heal had
# been copied from slow-tests.yml, whose own create path has never run (its
# label was created by hand on 2026-06-02), so copying it proved nothing.
#
# The stand-in models the `gh` behaviours that matter and nothing else:
#   - `label create` can be forbidden, rejects a >100-char description (the
#     real API's limit), or succeeds once;
#   - `issue create --label X` hard-fails when X does not exist (real gh);
#   - `issue list --label X` tolerates a missing X and returns nothing (real gh,
#     which is exactly why the defect hid).
# `--json/--jq` are honoured by handing the real jq the expression the workflow
# wrote, so the dedup query is executed, not eyeballed.

_GH_SHIM = r"""#!/usr/bin/env bash
set -u
STATE="$GH_SHIM_STATE"
printf '%s\n' "$*" >> "$STATE/calls.log"
touch "$STATE/labels" "$STATE/issues"
JQ=""; LABELS=(); TITLE=""; SEARCH=""; DESC=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --jq) JQ="$2"; shift 2 ;;
    --label) LABELS+=("$2"); shift 2 ;;
    --title) TITLE="$2"; shift 2 ;;
    --search) SEARCH="$2"; shift 2 ;;
    --description) DESC="$2"; shift 2 ;;
    --color|--body|--state|--limit|--json) shift 2 ;;
    *) POS+=("$1"); shift ;;
  esac
done
apply_jq() { if [ -n "$JQ" ]; then jq -r "$JQ"; else cat; fi; }
case "${POS[0]} ${POS[1]}" in
  "label create")
    name="${POS[2]}"
    if [ "$GH_SHIM_LABEL_CREATE" = "forbid" ]; then
      echo "HTTP 403: Resource not accessible by integration (https://api.github.com/repos/o/r/labels)" >&2
      exit 1
    fi
    # Measured on the real API (branch run 32755093587): GitHub rejects a label
    # description over 100 characters. That, not the token, was the D3 miss.
    if [ "${#DESC}" -gt 100 ]; then
      echo "HTTP 422: Validation Failed (https://api.github.com/repos/o/r/labels)" >&2
      echo "description is too long (maximum is 100 characters)" >&2
      exit 1
    fi
    if grep -qxF "$name" "$STATE/labels"; then
      echo "HTTP 422: Validation Failed (Label already exists)" >&2; exit 1
    fi
    echo "$name" >> "$STATE/labels"; exit 0 ;;
  "label list")
    grep -F -- "$SEARCH" "$STATE/labels" | jq -R '{name: .}' | jq -s . | apply_jq; exit 0 ;;
  "issue list")
    # issues: number<TAB>title<TAB>label ; a missing --label matches nothing, silently
    want="${LABELS[0]:-}"
    awk -F'\t' -v want="$want" '($3==want || want=="") {print}' "$STATE/issues" \
      | jq -R 'split("\t") | {number: (.[0]|tonumber), title: .[1], labels: [{name: .[2]}]}' \
      | jq -s . | apply_jq; exit 0 ;;
  "issue create")
    for l in "${LABELS[@]:-}"; do
      [ -z "$l" ] && continue
      grep -qxF "$l" "$STATE/labels" || { echo "could not add label: '$l' not found" >&2; exit 1; }
    done
    printf '%s\t%s\t%s\n' 999 "$TITLE" "${LABELS[0]:-}" >> "$STATE/issues"
    echo "https://github.com/o/r/issues/999"; exit 0 ;;
  "issue comment")
    exit 0 ;;
esac
echo "gh shim: unmodelled command: ${POS[*]}" >&2; exit 64
"""


def _reporter_step(job: dict) -> dict:
    return next(s for s in job["steps"] if "gh issue create" in str(s.get("run", "")))


def _run_reporter(
    tmp_path: Path,
    *,
    verdict: str,
    label_create: str,
    labels: tuple[str, ...] = (),
    open_issues: tuple[tuple[int, str, str], ...] = (),
) -> tuple[subprocess.CompletedProcess, list[str], Path]:
    """Execute the REAL reporter script with `gh` replaced by the stand-in."""
    _, job = _report_job(_load(WORKFLOW_PATH))
    step = _reporter_step(job)
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True)
    gh = bindir / "gh"
    gh.write_text(_GH_SHIM)
    gh.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    (state / "labels").write_text("".join(f"{name}\n" for name in labels))
    (state / "issues").write_text(
        "".join(f"{n}\t{title}\t{label}\n" for n, title, label in open_issues)
    )
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "GH_SHIM_STATE": str(state),
        "GH_SHIM_LABEL_CREATE": label_create,
        "GH_TOKEN": "shim",
        "GH_REPO": "o/r",
        "RUN_URL": "https://github.com/o/r/actions/runs/1",
        "EVENT_NAME": "schedule",
        "VERDICT": verdict,
    }
    result = subprocess.run(
        ["bash", "-c", str(step["run"])], env=env, capture_output=True, text=True
    )
    calls = (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else []
    return result, calls, state


def _issue_creates(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("issue create ")]


def _issue_comments(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("issue comment ")]


def test_reporter_files_the_issue_even_when_it_cannot_create_the_dedup_label(
    tmp_path: Path,
) -> None:
    """D3 as it happened: no label, label create forbidden. The alarm must still land,
    and the reason the label is missing must be in the log, not in /dev/null."""
    result, calls, _ = _run_reporter(tmp_path, verdict="stale", label_create="forbid")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    creates = _issue_creates(calls)
    assert len(creates) == 1, f"expected exactly one `issue create`, calls={calls}"
    assert "Droplet maintenance cron is stale" in creates[0]
    assert f"--label {DEDUP_LABEL}" not in creates[0], (
        "a label that does not exist must not be passed to `issue create` (it hard-fails)"
    )
    assert "Resource not accessible" in result.stdout + result.stderr, (
        "the label-create error must be visible in the run log"
    )


def test_reporter_creates_and_uses_the_dedup_label_when_it_can(tmp_path: Path) -> None:
    result, calls, _ = _run_reporter(tmp_path, verdict="unreachable", label_create="allow")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert any(c.startswith(f"label create {DEDUP_LABEL}") for c in calls), calls
    creates = _issue_creates(calls)
    assert len(creates) == 1 and f"--label {DEDUP_LABEL}" in creates[0], calls
    assert "could not reach the droplet" in creates[0]


def test_reporter_does_not_recreate_a_label_that_exists(tmp_path: Path) -> None:
    """The steady state: label present, create would 422. No create, label used."""
    result, calls, _ = _run_reporter(
        tmp_path, verdict="stale", label_create="forbid", labels=(DEDUP_LABEL,)
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert not any(c.startswith("label create") for c in calls), calls
    creates = _issue_creates(calls)
    assert len(creates) == 1 and f"--label {DEDUP_LABEL}" in creates[0], calls


def _created_title(state: Path) -> str:
    rows = [r.split("\t") for r in (state / "issues").read_text().splitlines() if r]
    assert len(rows) == 1, rows
    return rows[0][1]


@pytest.mark.parametrize(
    ("first_verdict", "second_verdict"), [("stale", "unreachable"), ("unreachable", "stale")]
)
def test_reporter_dedups_on_the_open_issue_even_without_the_label(
    tmp_path: Path, first_verdict: str, second_verdict: str
) -> None:
    """Dedup keyed ONLY on the label is dedup that dies with the label: with it
    missing, `gh issue list --label` silently returns nothing and every daily run
    would open a fresh issue. Dedup must find this workflow's own open issue by
    what the workflow controls -- its title -- across both verdicts. The existing
    title is whatever the reporter itself filed, not a restatement."""
    first, _, state = _run_reporter(
        tmp_path / "first", verdict=first_verdict, label_create="forbid"
    )
    assert first.returncode == 0, first.stderr
    existing_title = _created_title(state)
    result, calls, _ = _run_reporter(
        tmp_path / "second",
        verdict=second_verdict,
        label_create="forbid",
        open_issues=((42, existing_title, ""),),
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert _issue_creates(calls) == [], f"must not open a duplicate: {calls}"
    comments = _issue_comments(calls)
    assert len(comments) == 1 and comments[0].startswith("issue comment 42 "), calls
