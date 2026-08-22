"""#1784 + #1785 — gates before cleanup, and a fail-fast on a missing GHCR image.

Two decision-coupled defects in the same inline droplet script.

#1784 — the post-deploy prune ran INSIDE the gated SSH step
--------------------------------------------------------
``docker image prune -a -f || true`` protects against a prune that *errors*. It does
nothing about a prune that is *slow*, because the whole script ran inside one
``appleboy/ssh-action`` invocation with ``command_timeout: 30m`` — and a command timeout
kills the SSH command outright, so ``|| true`` never gets to run.

Measured on run `32507847667` (2026-08-21), read back from the job log::

    18:00:57  ==> Deploying 32259eb
    18:26:23  ==> Health check passed on attempt 1      <- the rollout had CONVERGED
    18:29:50  ==> bentoml is serving the cohort bundles
    18:29:50  ==> Post-deploy prune (unreferenced images + full build cache)
    18:30:26  Deleted Images:                           <- still pruning
    18:30:34  Run Command Timeout
    18:30:34  ##[error]Process completed with exit code 1.
              outcome=failure;conclusion=failure;duration_ms=1802761   (= 30m02s)

The prune got 44 seconds. Two harms followed, and this module pins the fix for both:

* the #1479 image-drift check sat *downstream* of the prune and never ran — a timeout
  there reproduces exactly the pin-drift blindness #1479 was filed to end;
* the run reported FAILED. ``JOB_STATUS: failure`` was handed to the summary step,
  which printed "rollback triggered or post-deploy drift check failed". Both
  attributions were false: nothing rolled back and the drift check never executed.

#1785 — the local-build path was entered silently
-------------------------------------------------
When the resolved sha has no GHCR image the droplet WARNed and started a ~26-minute
local build. That build produces an image that exists only on the box: no rollback
target, and the next deploy resolves the same imageless sha and repeats it. Now that
``ensure-main-image`` (PR #1782) should have guaranteed an image, "no image for the
target" is a broken invariant rather than a routine miss, so it is asserted at the one
moment the answer is actionable — right after ``NEW_SHA`` is resolved.

What this module refuses to be
------------------------------
``.github/**`` is NOT in ``deploy.yml``'s own ``on.push.paths`` and ``deploy.yml`` is
baked into no image, so there is no container marker and no live certification for a
change here. These structural tests are the ONLY gate, which is why every guard below
asserts the value it COMPUTED (the parsed step order, the derived index map, the
rendered summary text) rather than a bare boolean over it, and why both new branches —
the #1785 fail-fast and the three-way summary verdict — are EXECUTED against the
shipped text rather than grepped for.

All matching here is literal (Python ``in`` / ``str.index``), never a regex.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

ROLLOUT_ID = "rollout"
CLEANUP_ID = "cleanup"

# The two cleanup commands, verbatim. Literals, matched literally.
PRUNE_COMMANDS = ("docker image prune -a -f", "docker builder prune -a -f")

DRIFT_CHECK = "scripts/deploy/check_image_drift.py"


# --------------------------------------------------------------------------- #
# Extraction — the SHIPPED artifact, addressed by step id
# --------------------------------------------------------------------------- #
def _load_workflow() -> dict:
    wf: dict = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    return wf


def _deploy_steps() -> list[dict]:
    return list(_load_workflow()["jobs"]["deploy"]["steps"])


def _step_table() -> list[tuple[int, str, str, str]]:
    """The DERIVED step order: (index, id, name, uses/run). Printed by every failure."""
    table = []
    for i, step in enumerate(_deploy_steps()):
        kind = step.get("uses") or ("run:" if "run" in step else "?")
        table.append((i, str(step.get("id", "")), str(step.get("name", "")), str(kind)))
    return table


def _fmt(table: list[tuple[int, str, str, str]]) -> str:
    return "\n".join(f"  [{i}] id={id_!r} name={name!r} {kind}" for i, id_, name, kind in table)


def _step_index(step_id: str) -> int:
    for i, step in enumerate(_deploy_steps()):
        if step.get("id") == step_id:
            return i
    raise AssertionError(
        f"the deploy job has no step with id {step_id!r}. Derived step table:\n"
        + _fmt(_step_table())
    )


def _step(step_id: str) -> dict:
    return _deploy_steps()[_step_index(step_id)]


def _ssh_script(step_id: str) -> str:
    step = _step(step_id)
    with_ = step.get("with") or {}
    assert "script" in with_, (
        f"step id={step_id!r} carries no `script:` — it is not an ssh-action step. "
        f"Derived step table:\n{_fmt(_step_table())}"
    )
    return str(with_["script"])


def _minutes(duration: str) -> int:
    """Parse a Go-style ssh-action duration ('30m', '15m', '900s') into whole minutes."""
    m = re.fullmatch(r"(\d+)([smh])", duration.strip())
    assert m, f"un-parseable command_timeout {duration!r}"
    n, unit = int(m.group(1)), m.group(2)
    return {"s": n // 60, "m": n, "h": n * 60}[unit]


def _command_timeout(step_id: str) -> str:
    with_ = _step(step_id).get("with") or {}
    assert "command_timeout" in with_, (
        f"step id={step_id!r} declares no command_timeout of its own; it would inherit "
        f"the action default. Derived `with` keys: {sorted(with_)}"
    )
    return str(with_["command_timeout"])


def _index_map(script: str, markers: dict[str, str]) -> dict[str, int]:
    """Derive each marker's position. Missing markers surface as -1, never silently."""
    return {label: script.find(needle) for label, needle in markers.items()}


def _prose(script: str) -> str:
    """Comment text with the `#` markers and line wrapping flattened away.

    A phrase in a wrapped comment is split across lines at an arbitrary column, so a
    literal search over the raw script silently misses it — the fail-open shape that
    has bitten this repo repeatedly. Flatten first, then match literally.
    """
    words: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            words.extend(stripped.lstrip("#").split())
    return " ".join(words)


def _last_command_line(script: str) -> str:
    for line in reversed(script.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise AssertionError("the rollout script has no command lines at all")


# --------------------------------------------------------------------------- #
# 1. #1784 option 4 — every gate precedes every cleanup
# --------------------------------------------------------------------------- #
def test_no_cleanup_command_survives_inside_the_gated_rollout_script() -> None:
    """The gated step must contain NO unbounded cleanup work.

    RED before the fix: both prune commands live at the end of the rollout script,
    in front of the #1479 drift check. The assertion prints WHICH prune commands it
    found and where, so a mutation that moves one back is visible as a value, not as
    a flipped boolean.
    """
    script = _ssh_script(ROLLOUT_ID)
    found = {cmd: script.find(cmd) for cmd in PRUNE_COMMANDS if cmd in script}
    assert not found, (
        "#1784 option 4: cleanup must not run inside the gated SSH step — its time "
        "budget sits in front of the #1479 drift check, and a command_timeout there "
        "kills the gate outright (run 32507847667). Found in the rollout script: "
        f"{found}"
    )


def test_rollout_gate_order_is_derived_and_monotonic() -> None:
    """Health gate -> bentoml gate -> drift check, with the drift check LAST.

    The old guard (test_image_drift_check.py) asserted only `invoke_at > health_at`.
    That is a boolean over two indices and stays true no matter what is inserted
    BETWEEN them — which is exactly where the prune sat. This derives the full
    ordered index map and prints it.
    """
    script = _ssh_script(ROLLOUT_ID)
    order = _index_map(
        script,
        {
            "app health gate": "Waiting for health check",
            "bentoml readiness gate": "bentoml is serving the cohort bundles",
            "image drift check (#1479)": DRIFT_CHECK,
        },
    )
    assert all(v >= 0 for v in order.values()), f"a gate vanished from the rollout: {order}"
    positions = list(order.values())
    assert positions == sorted(positions), (
        f"the rollout gates are out of order. Derived index map: {order}"
    )

    last = _last_command_line(script)
    assert DRIFT_CHECK in last, (
        "#1784 option 4: the #1479 drift check must be the LAST command in the gated "
        "step, so nothing can be inserted downstream of the final gate. Derived last "
        f"command line: {last!r}"
    )


def test_drift_check_comment_does_not_assert_a_prune_that_is_no_longer_there() -> None:
    """A comment that is now false must be corrected, not carried over.

    It read "Runs AFTER the prune so disk hygiene still happens when the alarm fires."
    Half is now false — the prune left this script — and half is still required: the
    alarm fails this step, and the disk still needs hygiene when it does. That half now
    lives on the cleanup step's `if:` (see the option-5 guards).

    The distinction this guard has to make is between ASSERTING the old ordering and
    QUOTING it in order to record what changed. A bare "the phrase is absent" check
    cannot tell those apart and would punish the better comment, so the requirement is:
    drop the sentence, or keep it only alongside an explicit correction.
    """
    prose = _prose(_ssh_script(ROLLOUT_ID))
    claim = "Runs AFTER the prune"
    if claim not in prose:
        return
    assert "the prune moved to its own step" in prose, (
        f"the rollout script still carries {claim!r} without saying the prune moved, "
        "so it reads as a live claim about an ordering that no longer exists"
    )


# --------------------------------------------------------------------------- #
# 2. #1784 option 2 — the prune gets its own SSH step, with its own budget
# --------------------------------------------------------------------------- #
def test_prune_lives_in_its_own_ssh_step_after_the_gated_rollout() -> None:
    table = _step_table()
    rollout_at = _step_index(ROLLOUT_ID)
    cleanup_at = _step_index(CLEANUP_ID)
    assert cleanup_at > rollout_at, (
        f"the cleanup step must follow the gated rollout. Derived step table:\n{_fmt(table)}"
    )

    cleanup = _step(CLEANUP_ID)
    assert str(cleanup.get("uses", "")).startswith("appleboy/ssh-action"), (
        f"the cleanup step must be its own SSH invocation; got uses={cleanup.get('uses')!r}"
    )

    cleanup_script = _ssh_script(CLEANUP_ID)
    missing = [cmd for cmd in PRUNE_COMMANDS if cmd not in cleanup_script]
    assert not missing, f"the cleanup step does not run {missing}. Its script is:\n{cleanup_script}"
    # Only the INVOCATIONS, not the echo that announces them: `|| true` is about a
    # prune that ERRORS, which is a different failure from the slow prune #1784 is
    # about, and the older guard must survive this move intact.
    invocations = [
        line.rstrip()
        for line in cleanup_script.splitlines()
        if line.strip().startswith("docker ") and " prune " in line
    ]
    assert len(invocations) == len(PRUNE_COMMANDS), (
        f"expected one invocation per prune command; derived: {invocations}"
    )
    for line in invocations:
        assert line.endswith("|| true"), (
            "each prune keeps its own end-of-line `|| true` guard (#272 discipline) "
            f"so a prune that ERRORS is still best-effort: {line!r}"
        )


def test_job_budget_is_recomputed_from_both_ssh_timeouts() -> None:
    """`timeout-minutes` must exceed the SUM of both command_timeouts.

    #1412's reason for the job-level bound is that a hung deploy must self-cancel
    rather than blockade the serialized deploy-production queue for GitHub's 360-min
    default. Splitting one SSH step into two makes that bound a SUM, not a single
    value — so this recomputes it instead of pinning a literal, and prints the numbers
    it derived. A budget that no longer covers both steps would let the JOB timeout
    fire first, which reports failure on exactly the axis #1784 is trying to separate.
    """
    job = _load_workflow()["jobs"]["deploy"]
    rollout_m = _minutes(_command_timeout(ROLLOUT_ID))
    cleanup_m = _minutes(_command_timeout(CLEANUP_ID))
    job_m = int(job["timeout-minutes"])
    derived = {
        "rollout command_timeout (min)": rollout_m,
        "cleanup command_timeout (min)": cleanup_m,
        "sum (min)": rollout_m + cleanup_m,
        "job timeout-minutes": job_m,
    }
    assert job_m > rollout_m + cleanup_m, (
        "the job budget must exceed BOTH ssh command_timeouts plus checkout/handshake "
        f"overhead, or the job timeout pre-empts them. Derived: {derived}"
    )


# --------------------------------------------------------------------------- #
# 3. #1784 option 5 — a cut-short cleanup must not present as a failed deploy
# --------------------------------------------------------------------------- #
def test_cleanup_failure_cannot_fail_the_job() -> None:
    cleanup = _step(CLEANUP_ID)
    assert cleanup.get("continue-on-error") is True, (
        "#1784 option 5: a prune that overruns its own budget must not turn the job "
        f"red — the rollout already converged and was health-gated. Got: "
        f"continue-on-error={cleanup.get('continue-on-error')!r}"
    )


def test_cleanup_still_runs_when_a_gate_fires() -> None:
    """The surviving intent of the old comment, relocated to where it now lives.

    "Runs AFTER the prune so disk hygiene still happens when the alarm fires" was a
    real requirement, not decoration: a #1479 pin mismatch fails the rollout step, and
    the disk still needs hygiene. Default step semantics (`success()`) would skip the
    cleanup in exactly that case, so the condition must be broader than success.
    """
    cleanup = _step(CLEANUP_ID)
    condition = str(cleanup.get("if", ""))
    assert condition, (
        "the cleanup step has no `if:`, so it inherits success() and is SKIPPED "
        "whenever the drift alarm fires — losing the intent the old ordering carried"
    )
    assert "cancelled()" in condition, (
        "the cleanup must run whether or not the rollout step succeeded (and must NOT "
        "start a fresh SSH prune while the job is being cancelled). Derived condition: "
        f"{condition!r}"
    )
    assert condition.strip() not in ("${{ success() }}", "success()"), (
        f"derived condition {condition!r} still gates cleanup on rollout success"
    )


def _summary_step() -> dict:
    for step in _deploy_steps():
        if "run" in step and "GITHUB_STEP_SUMMARY" in str(step["run"]):
            return step
    raise AssertionError("the deploy job has no step writing GITHUB_STEP_SUMMARY")


def test_summary_is_wired_to_both_step_outcomes_not_the_job_status() -> None:
    """The verdict must be derived from two distinct signals.

    Today the summary reads `job.status`, which is the SSH step's status and nothing
    else — which is why run 32507847667 announced "rollback triggered or post-deploy
    drift check failed" when neither had happened.
    """
    env = {k: str(v) for k, v in (_summary_step().get("env") or {}).items()}
    joined = " ".join(env.values())
    assert f"steps.{ROLLOUT_ID}.outcome" in joined, (
        f"the summary never reads the rollout outcome. Derived env: {env}"
    )
    assert f"steps.{CLEANUP_ID}.outcome" in joined, (
        f"the summary never reads the cleanup outcome. Derived env: {env}"
    )
    assert "job.status" not in joined, (
        "`job.status` is the SSH step's status re-labelled; keeping it alongside the "
        f"per-step outcomes leaves two sources of truth. Derived env: {env}"
    )


def _render_summary(tmp_path: Path, **outcomes: str) -> str:
    """EXECUTE the shipped summary script and read back what it actually wrote."""
    step = _summary_step()
    env_block = {k: str(v) for k, v in (step.get("env") or {}).items()}
    # Map the outcome we want onto the env var the shipped script actually consumes,
    # so the test cannot drift from the wiring it asserts above.
    env: dict[str, str] = {}
    for name, expr in env_block.items():
        for step_id, value in outcomes.items():
            if f"steps.{step_id}.outcome" in expr:
                env[name] = value
        env.setdefault(name, "test")
    summary_file = tmp_path / f"summary_{'_'.join(sorted(outcomes.values()))}.md"
    summary_file.write_text("")
    runner = tmp_path / "summary.sh"
    runner.write_text(str(step["run"]))
    proc = subprocess.run(
        ["bash", str(runner)],
        env={
            "PATH": "/usr/bin:/bin",
            "GITHUB_STEP_SUMMARY": str(summary_file),
            **env,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"summary script failed: {proc.stderr}"
    return summary_file.read_text()


def test_summary_renders_three_distinct_verdicts(tmp_path: Path) -> None:
    """Driven, not grepped: run the shipped script down each of its three branches.

    Executing a guard is not exercising it — a real deploy takes exactly one of these
    paths, so each is forced here directly.
    """
    converged = _render_summary(tmp_path, rollout="success", cleanup="success")
    cut_short = _render_summary(tmp_path, rollout="success", cleanup="failure")
    failed = _render_summary(tmp_path, rollout="failure", cleanup="success")

    rendered = {"converged": converged, "cut_short": cut_short, "failed": failed}

    assert "Deployment Succeeded" in converged, rendered
    assert "Failed" not in converged, rendered
    assert "cleanup" not in converged.lower(), (
        f"a clean deploy must not mention cleanup at all: {rendered}"
    )

    assert "Deployment Succeeded" in cut_short, (
        "#1784 option 5: a converged deploy whose prune was cut short must still read "
        f"as a SUCCESS. Rendered: {rendered}"
    )
    assert "Deployment Failed" not in cut_short, rendered
    assert "cleanup" in cut_short.lower(), (
        f"the cut-short case must NAME the cleanup as what did not finish: {rendered}"
    )

    assert "Deployment Failed" in failed, rendered

    assert len({converged, cut_short, failed}) == 3, (
        f"the three outcomes must render three distinct verdicts: {rendered}"
    )


def test_a_failed_rollout_is_not_attributed_to_a_rollback_that_may_not_have_happened(
    tmp_path: Path,
) -> None:
    """Codex iter-3 HIGH. #1784 is about false attribution, and this PR added a new mode.

    Run 32507847667 announced "rollback triggered or post-deploy drift check failed"
    when neither had occurred — that false attribution is the reason #1784 was filed.
    Option 5 rewrote which VARIABLE the verdict reads but kept the same sentence on the
    failure side, and this PR then added a failure mode that reaches it: the #1785
    fail-fast exits before the migrations, before any flip, before rollback and before
    the drift check, and its own log line says "nothing was flipped or migrated".

    The step outcome cannot tell those apart — `failure` is all there is — so the fix is
    not a better guess. The summary must stop naming a cause it does not have, and
    instead enumerate what the gated half covers, in the order it runs, so a human lands
    in the right part of the log.
    """
    failed = _render_summary(tmp_path, rollout="failure", cleanup="success")
    assert "Deployment Failed" in failed, failed

    lowered = failed.lower()
    assert "published-image" in lowered or "#1785" in failed, (
        "the #1785 fail-fast is a rollout failure that flips and migrates NOTHING; a "
        "summary that never mentions it sends a human looking for a rollback that did "
        f"not happen. Rendered:\n{failed}"
    )
    for claim in ("rollback triggered", "rollback was triggered"):
        assert claim not in lowered, (
            "the summary asserts a rollback as fact from a signal that cannot show one "
            f"— the exact false attribution #1784 was filed over. Rendered:\n{failed}"
        )


@pytest.mark.parametrize("outcome", ["skipped", ""])
def test_a_rollout_that_never_started_is_not_reported_as_a_failed_deploy(
    tmp_path: Path, outcome: str
) -> None:
    """`if: always()` also renders when the rollout never executed.

    A checkout failure or a job that died before the SSH step leaves the summary with
    something that is not `success`, and the shipped script treated every one of them as
    "Deployment Failed — rollback triggered". Nothing was deployed, so nothing was
    rolled back. Unlike the failure branch, this IS derivable from the outcome, so there
    is no excuse for guessing it.
    """
    rendered = _render_summary(tmp_path, rollout=outcome, cleanup="skipped")
    lowered = rendered.lower()
    assert "rollback" not in lowered and "reverted" not in lowered.split("nothing was")[0], (
        f"outcome {outcome!r} means the rollout never started, so there was nothing to "
        f"roll back. Rendered:\n{rendered}"
    )
    assert "did not run" in lowered or "never started" in lowered, (
        f"outcome {outcome!r} must be reported as a rollout that did not run, not as a "
        f"failed deploy. Rendered:\n{rendered}"
    )


def test_a_cancelled_rollout_does_not_claim_the_droplet_was_untouched(tmp_path: Path) -> None:
    """A cancel is not "it never ran" — and my first fix for the above said it was.

    Lumping `cancelled` in with `skipped` produced the sentence "The droplet was never
    reached", which is a FRESH false attribution inside a fix for false attribution.
    `cancel-in-progress` is false for the `deploy-production` group, so a newer deploy
    cannot cancel this one — but a human can, and so can this job's own
    `timeout-minutes`, and either lands wherever the rollout happened to be. The box may
    be part-flipped, which is the state the #1784 prune guard exists for.

    So the summary must say what is known and name the rest as not derivable.
    """
    rendered = _render_summary(tmp_path, rollout="cancelled", cleanup="skipped")
    lowered = rendered.lower()
    assert "cancelled" in lowered, f"the cancel must be named:\n{rendered}"
    assert "never reached" not in lowered and "never started" not in lowered, (
        "a cancel can land mid-rollout, so the summary must not claim the droplet was "
        f"untouched:\n{rendered}"
    )
    assert "not derivable" in lowered, (
        f"the summary must say plainly that how far it got is unknown:\n{rendered}"
    )
    assert "prune" in lowered, (
        "a cancel also skips the post-deploy prune (`if: !cancelled()`), which is part "
        f"of what state the box is left in:\n{rendered}"
    )


# --------------------------------------------------------------------------- #
# 4. #1785 — fail fast when the resolved sha has no GHCR image
# --------------------------------------------------------------------------- #
GATE_AUTH_START = "GATE_AUTH_OK=false"
GATE_AUTH_BRANCH = 'if [ -n "$GATE_STANDDOWN" ]; then'
PROBE_CALL = 'images_verdict "$NEW_SHA" || IMAGES_VERDICT=$?'
FAIL_FAST_OPENER = 'elif [ "$IMAGES_VERDICT" -ne 0 ]; then'


def _line_index(lines: list[str], wanted: str, what: str) -> int:
    idx = next((i for i, ln in enumerate(lines) if ln.strip() == wanted), None)
    assert idx is not None, f"{what} — expected a line {wanted!r} in the rollout script"
    return idx


def _extract_gate_block() -> str:
    """The SHIPPED #1785 gate, verbatim: auth precondition through its own-indent `fi`.

    Spans from the auth flag to the close of the branch chain, because the precondition
    is not decoration — it is the difference between "this image is absent" and "I could
    not ask", and the two must be extracted and executed together.
    """
    script = _ssh_script(ROLLOUT_ID)
    lines = script.splitlines()
    start = _line_index(
        lines,
        GATE_AUTH_START,
        "#1785: the gate has no GHCR-auth precondition, so an unreachable registry is "
        "indistinguishable from a missing image",
    )
    branch = _line_index(
        lines,
        GATE_AUTH_BRANCH,
        "#1785: the gate never branches on whether it got an answer it can stand behind",
    )
    _line_index(
        lines, FAIL_FAST_OPENER, "#1785: the manifest assertion is not the stand-down `elif`"
    )
    assert any(PROBE_CALL in ln for ln in lines), (
        "#1785: the gate does not call the three-way images_verdict — a boolean probe "
        "cannot tell a registry that SAID absent from one that would not answer, which "
        "is the whole finding. Rollout script lines matching 'verdict':\n"
        + "\n".join(f"  {ln}" for ln in lines if "verdict" in ln)
    )
    indent = len(lines[branch]) - len(lines[branch].lstrip())
    for j in range(branch + 1, len(lines)):
        if lines[j].strip() == "fi" and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            return "\n".join(ln[indent:] for ln in lines[start : j + 1])
    raise AssertionError("the #1785 gate has no closing `fi` at its own indent")


def test_manifest_assertion_sits_between_sha_resolution_and_the_expensive_work() -> None:
    """After NEW_SHA is FIXED, before anything expensive.

    Placement is the whole point: earlier and it would probe a sha the reset has not
    landed on yet; later and the DB migrations have already run and the ~26-min build
    has already started. Derives the index map and prints it.
    """
    script = _ssh_script(ROLLOUT_ID)
    order = _index_map(
        script,
        {
            "NEW_SHA resolved": "NEW_SHA=$(git rev-parse HEAD)",
            "#1785 manifest assertion": PROBE_CALL,
            "DB migrations": "bash scripts/run_migrations.sh",
            "GHCR pull / local-build fallback": "$COMPOSE_CMD pull api",
        },
    )
    assert all(v >= 0 for v in order.values()), f"a landmark vanished: {order}"
    positions = list(order.values())
    assert positions == sorted(positions), (
        "#1785: the manifest assertion must sit AFTER NEW_SHA resolution and BEFORE "
        f"the migrations and the pull. Derived index map: {order}"
    )


def _docker_stub(login_rc: int, manifest_rc: int = 1, manifest_stderr: str = "") -> str:
    """A `docker` stub that is faithful about the one thing that matters: it is a BINARY.

    `return`, never `exit`. A first cut of this harness used `exit 1` and every auth
    case silently printed nothing, because a shell FUNCTION that calls `exit` tears the
    whole script down where a real external `docker` only sets an exit status. That is
    a stub-fidelity failure, and it is the shape that hides the very defect being
    hunted, so the correction is recorded here rather than quietly fixed. The `cat`
    drains `docker login`'s piped stdin, as the real binary does.
    """
    return (
        "docker() {\n"
        '  case "$1" in\n'
        f"    login) cat >/dev/null 2>&1 || true; return {login_rc} ;;\n"
        f'    manifest) printf %s\\\\n "{manifest_stderr}" >&2; return {manifest_rc} ;;\n'
        "    *) return 0 ;;\n"
        "  esac\n"
        "}\n"
    )


def _run_gate(tmp_path: Path, preamble: str, **env: str) -> tuple[int, str]:
    """Execute the SHIPPED gate block under a supplied preamble of stubs."""
    runner = tmp_path / "gate.sh"
    runner.write_text(
        "set -e\n" + preamble + _extract_gate_block() + '\necho "REACHED THE EXPENSIVE WORK"\n'
    )
    proc = subprocess.run(
        ["bash", str(runner)],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _run_fail_fast(
    tmp_path: Path, verdict_rc: int, login_rc: int = 0, **env: str
) -> tuple[int, str]:
    """Drive the shipped gate down a chosen branch with the manifest probe stubbed out.

    `verdict_rc` is images_verdict's three-way status: 0 published, 1 the registry NAMED
    it absent, 2 no answer we can stand behind. Auth succeeds by default, so these cases
    exercise the assertion itself. Every case where the ANSWER is in question is covered
    against the REAL probes below, deliberately: stubbing the probe is exactly what made
    the whole class of registry-unreachable defects invisible in the first place.
    """
    preamble = _docker_stub(login_rc) + f"images_verdict() {{ return {verdict_rc}; }}\n"
    runner = tmp_path / "failfast.sh"
    runner.write_text(
        "set -e\n" + preamble + _extract_gate_block() + '\necho "REACHED THE EXPENSIVE WORK"\n'
    )
    proc = subprocess.run(
        ["bash", str(runner)],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout + proc.stderr


SHA = "c" * 40


def test_a_missing_ghcr_image_fails_fast_naming_the_sha(tmp_path: Path) -> None:
    """The failing branch, driven directly (a real deploy almost never takes it)."""
    rc, out = _run_fail_fast(
        tmp_path,
        verdict_rc=1,
        NEW_SHA=SHA,
        IMAGE_OWNER="enunezvn",
        FALLBACK_REASON="GHCR auth SUCCEEDED, but no commit in the window has both images",
    )
    assert rc != 0, f"a missing image must fail the deploy; rc={rc}, output:\n{out}"
    assert "REACHED THE EXPENSIVE WORK" not in out, (
        f"the deploy continued into the local-build path anyway:\n{out}"
    )
    assert SHA in out, f"the message must name the FULL resolved sha:\n{out}"
    assert "e2i-api" in out and "e2i-frontend" in out, (
        f"the message must name both image refs that were expected:\n{out}"
    )
    assert "GHCR auth SUCCEEDED, but no commit in the window has both images" in out, (
        f"the message must carry the sha-selection FALLBACK_REASON forward:\n{out}"
    )


def test_a_present_ghcr_image_lets_the_deploy_proceed(tmp_path: Path) -> None:
    """Positive control: the guard must be a no-op on the path every deploy takes.

    Without this, a fail-fast that ALWAYS fired would satisfy the test above.
    """
    rc, out = _run_fail_fast(tmp_path, verdict_rc=0, NEW_SHA=SHA, IMAGE_OWNER="enunezvn")
    assert rc == 0, f"a published image must not fail the deploy; rc={rc}, output:\n{out}"
    assert "REACHED THE EXPENSIVE WORK" in out, (
        f"the deploy must continue when the image exists:\n{out}"
    )
    assert "ERROR" not in out, f"the guard emitted an error on the happy path:\n{out}"


def test_fail_fast_message_states_the_refusal_and_the_recovery(tmp_path: Path) -> None:
    """The message must be actionable and honest about what it is refusing.

    `manifest unknown` scrolling past mid-run is what made this cost 26 minutes to
    discover. The replacement has to say what it refused (the local build), why (no
    rollback target + an OOM-prone build on a live prod box), and how to recover.
    """
    _, out = _run_fail_fast(tmp_path, verdict_rc=1, NEW_SHA=SHA, IMAGE_OWNER="enunezvn")
    lowered = out.lower()
    for phrase, why in (
        # "local-build path" is this file's own established term for it.
        ("local-build", "must name the path it is refusing"),
        ("rollback", "must state that a box-only image is not a rollback target"),
        ("gh workflow run deploy.yml", "must give the recovery command"),
    ):
        assert phrase in lowered, f"the #1785 message {why}; got:\n{out}"


@pytest.mark.parametrize("reason", ["", "GHCR auth unavailable — no ancestor was probed"])
def test_fail_fast_survives_an_absent_fallback_reason(tmp_path: Path, reason: str) -> None:
    """The message must survive an empty FALLBACK_REASON with its recovery intact.

    A correction worth recording, because I nearly shipped the claim. I wrote the
    conditional as an `if` block believing `[ -n "$X" ] && echo ...` would abort the
    script under `set -e` when X is empty and swallow every line after it, including
    the recovery command. Measured instead of assumed::

        set -e
        if ! false; then
          echo line1
          [ -n "$FR" ] && echo "reason: $FR"
          echo "RECOVERY LINE"
          exit 7
        fi
        $ FR="" bash se.sh
        line1
        RECOVERY LINE
        rc=7

    bash does NOT apply errexit to a short-circuited AND-list, so the `&&` form was
    safe and the `if` form is a readability choice, not a bug fix. Mutation-testing
    confirms it: swapping the shipped `if` back to `&&` leaves both cases below GREEN,
    which is the correct result rather than a gap.

    What this test does pin — and what a mutation DOES break — is the property that
    actually matters: both branches of the optional line still emit the recovery
    instructions, and a FALLBACK_REASON that exists is carried forward to the human.
    """
    rc, out = _run_fail_fast(
        tmp_path, verdict_rc=1, NEW_SHA=SHA, IMAGE_OWNER="enunezvn", FALLBACK_REASON=reason
    )
    assert rc != 0
    assert "gh workflow run deploy.yml" in out, (
        f"the recovery line was swallowed with FALLBACK_REASON={reason!r}:\n{out}"
    )
    if reason:
        assert reason in out, f"a present FALLBACK_REASON must be shown:\n{out}"


# --------------------------------------------------------------------------- #
# 5. #1785 — an UNREACHABLE registry is not a MISSING image
#
# Codex HIGH, reproduced against the real helpers before it was believed.
# `manifest_present()` trusts a DEFINITIVE absent at once ("no such manifest" /
# "manifest unknown" / "not found" / "UNKNOWN") and retries anything else once before
# giving up — so `denied`, `unauthorized` and `manifest unknown` all arrive at the call
# site as the same non-zero. Measured by driving the extracted, unmodified
# manifest_present/image_exists with a stubbed `docker`:
#
#     definitive absent      image_exists -> FALSE
#     GHCR auth denied       image_exists -> FALSE
#     no basic auth creds    image_exists -> FALSE
#     unauthorized           image_exists -> FALSE
#     present (control)      image_exists -> TRUE
#
# Without a precondition the gate therefore converts a GHCR auth blip into a hard
# deploy failure announced as "no published GHCR image", and it does so BEFORE the pull
# step, which performs its OWN fresh `docker login` and was the self-healing path for
# exactly this case. These tests stub `docker`, never `image_exists`: stubbing the
# probe is what made this whole class of defect invisible in the first pass.
# --------------------------------------------------------------------------- #
PROBE_HELPERS_START = "manifest_probe_once() {"
PROBE_HELPERS_END = "# Given candidate SHAs on stdin"


def _extract_probe_helpers() -> str:
    """The REAL manifest_probe_once/manifest_verdict/images_verdict source, dedented."""
    script = _ssh_script(ROLLOUT_ID)
    start = script.find(PROBE_HELPERS_START)
    end = script.find(PROBE_HELPERS_END)
    assert 0 <= start < end, (
        f"could not locate the real manifest probes in the rollout script ({start}, {end})"
    )
    lines = script[start:end].splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(ln[indent:] if ln.strip() else "" for ln in lines) + "\n"


def _run_gate_against_real_probes(
    tmp_path: Path, *, login_rc: int, manifest_rc: int, manifest_stderr: str, **env: str
) -> tuple[int, str]:
    preamble = (
        _docker_stub(login_rc, manifest_rc, manifest_stderr)
        + "sleep() { :; }\n"  # collapse manifest_present's one retry
        + _extract_probe_helpers()
    )
    return _run_gate(tmp_path, preamble, **env)


@pytest.mark.parametrize(
    "manifest_stderr",
    [
        "denied: denied",
        "denied: requested access to the resource is denied",
        "unauthorized: authentication required",
    ],
)
def test_a_ghcr_auth_failure_is_not_reported_as_a_missing_image(
    tmp_path: Path, manifest_stderr: str
) -> None:
    """Login fails -> the gate must NOT fire, and must say why it stood down.

    The deploy has to reach the pull step, whose own fresh `docker login` is the
    self-healing path a transient blip needs. Turning that into a hard failure is a
    strictly worse outcome than the one #1785 set out to prevent.
    """
    rc, out = _run_gate_against_real_probes(
        tmp_path,
        login_rc=1,
        manifest_rc=1,
        manifest_stderr=manifest_stderr,
        NEW_SHA=SHA,
        IMAGE_OWNER="enunezvn",
        GHCR_TOKEN="t",
    )
    assert rc == 0, (
        "a registry we could not authenticate to must not hard-fail the deploy "
        f"(stderr was {manifest_stderr!r}); rc={rc}, output:\n{out}"
    )
    assert "REACHED THE EXPENSIVE WORK" in out, (
        f"the deploy must fall through to the pull's own fresh login:\n{out}"
    )
    assert "no published GHCR image" not in out, (
        f"an auth failure was announced as a missing image:\n{out}"
    )
    lowered = out.lower()
    assert "skipping" in lowered and "login" in lowered, (
        "standing down must be stated, and attributed to the LOGIN that failed rather "
        f"than to the image:\n{out}"
    )


def test_an_authenticated_absent_image_still_fails_fast_through_the_real_probes(
    tmp_path: Path,
) -> None:
    """Positive control for the precondition: it must not neuter the gate.

    Login succeeds and the registry answers a DEFINITIVE absent — the one case #1785
    exists for. A precondition that always stood down would satisfy the test above.
    """
    rc, out = _run_gate_against_real_probes(
        tmp_path,
        login_rc=0,
        manifest_rc=1,
        manifest_stderr="manifest unknown: manifest unknown",
        NEW_SHA=SHA,
        IMAGE_OWNER="enunezvn",
        GHCR_TOKEN="t",
    )
    assert rc != 0, f"an authenticated, definitively-absent image must fail:\n{out}"
    assert "REACHED THE EXPENSIVE WORK" not in out, f"the deploy continued anyway:\n{out}"
    assert SHA in out and "no published GHCR image" in out, (
        f"the hard-fail message must survive the precondition:\n{out}"
    )


def test_an_authenticated_present_image_proceeds_through_the_real_probes(
    tmp_path: Path,
) -> None:
    """The path every healthy deploy takes, with nothing but `docker` stubbed."""
    rc, out = _run_gate_against_real_probes(
        tmp_path,
        login_rc=0,
        manifest_rc=0,
        manifest_stderr="",
        NEW_SHA=SHA,
        IMAGE_OWNER="enunezvn",
        GHCR_TOKEN="t",
    )
    assert rc == 0, f"a published image must not fail the deploy:\n{out}"
    assert "REACHED THE EXPENSIVE WORK" in out, f"the deploy must proceed:\n{out}"
    assert "SKIPPING" not in out, f"the gate stood down on a path where auth worked:\n{out}"


# --------------------------------------------------------------------------- #
# 5b. #1785 — LOGGED IN is not AUTHORIZED TO READ
#
# Codex iter-2 HIGH, reproduced against the shipped gate before it was believed.
# The iter-1 fix asserted only that `docker login` succeeded — and a successful login
# does not mean the manifest can be read. GHCR's login endpoint accepts any valid PAT;
# package-level `read:packages` grants are checked at the MANIFEST, so a scope-reduced
# token, a package unlinked from its repo, or revoked org access all answer
# `denied: requested access to the resource is denied` while the image plainly exists.
# A registry 5xx or a network stall that outlives the single retry lands in the same
# bucket. Measured by driving the SHIPPED gate at 658250acd (login stubbed to 0, the
# manifest read stubbed to a persistent non-definitive error):
#
#     login OK + manifest present (control)   -> PROCEEDS
#     login OK + DEFINITIVE absent            -> HARD-FAIL as missing image   (correct)
#     login OK + persistent DENIED            -> HARD-FAIL as missing image   (WRONG)
#     login OK + persistent UNAUTHORIZED      -> HARD-FAIL as missing image   (WRONG)
#     login OK + network error                -> HARD-FAIL as missing image   (WRONG)
#     login FAILS                             -> stands down                  (correct)
#
# The three WRONG rows are the iter-1 defect one layer in: a converged deploy refused,
# announced as a missing image, sending a human after ensure-main-image when the real
# cause is registry ACCESS. The fix gives the probe a third answer — present /
# DEFINITIVELY absent / no answer at all — and spends the refusal only on the middle one.
#
# These tests stub `docker` and drive the REAL classifier. The per-ref stub below answers
# each of the two image refs separately and counts attempts on DISK, because the probe
# runs inside `$(...)` — a subshell, where an in-memory counter would silently never
# advance and "persistent" would quietly mean "failed once".
# --------------------------------------------------------------------------- #
DEFINITIVE_ABSENT = "manifest unknown: manifest unknown"
INCONCLUSIVE = "denied: requested access to the resource is denied"


def _docker_ref_stub(
    login_rc: int,
    api: list[tuple[int, str]],
    frontend: list[tuple[int, str]],
) -> str:
    """A `docker` stub that answers per-REF and per-ATTEMPT.

    `api`/`frontend` are the successive answers for that ref as ``(rc, stderr)``; the
    LAST entry repeats for any further attempt. Like `_docker_stub`, it uses `return`
    and never `exit` — a shell function that exits tears the harness down where a real
    external binary only sets a status.
    """
    arms = []
    for slot, answers in (("api", api), ("frontend", frontend)):
        for i, (rc, err) in enumerate(answers, start=1):
            label = f"{slot}:{i}" if i < len(answers) else f"{slot}:*"
            emit = f'printf "%s\\n" {shlex.quote(err)} >&2; ' if err else ""
            arms.append(f"    {label}) {emit}return {rc} ;;")
    return (
        "docker() {\n"
        '  if [ "$1" = "login" ]; then\n'
        "    cat >/dev/null 2>&1 || true\n"
        f"    return {login_rc}\n"
        "  fi\n"
        '  case "$3" in\n'
        "    *e2i-api:*) _slot=api ;;\n"
        "    *e2i-frontend:*) _slot=frontend ;;\n"
        '    *) printf "%s\\n" "stub: unexpected ref $3" >&2; return 1 ;;\n'
        "  esac\n"
        '  _f="$STUB_STATE/$_slot"\n'
        "  _n=0\n"
        '  if [ -f "$_f" ]; then _n=$(cat "$_f"); fi\n'
        "  _n=$((_n + 1))\n"
        '  printf "%s\\n" "$_n" > "$_f"\n'
        '  case "${_slot}:${_n}" in\n' + "\n".join(arms) + "\n"
        "  esac\n"
        "  return 1\n"
        "}\n"
    )


def _run_gate_per_ref(
    tmp_path: Path,
    *,
    api: list[tuple[int, str]],
    frontend: list[tuple[int, str]],
    login_rc: int = 0,
    **env: str,
) -> tuple[int, str]:
    state = tmp_path / "stub_state"
    state.mkdir(exist_ok=True)
    preamble = (
        _docker_ref_stub(login_rc, api, frontend)
        + "sleep() { :; }\n"  # collapse the retry's wait; the DISK counter still advances
        + _extract_probe_helpers()
    )
    return _run_gate(
        tmp_path,
        preamble,
        STUB_STATE=str(state),
        NEW_SHA=SHA,
        IMAGE_OWNER="enunezvn",
        GHCR_TOKEN="t",
        **env,
    )


@pytest.mark.parametrize(
    "stderr_text",
    [
        "denied: requested access to the resource is denied",
        "unauthorized: authentication required",
        "Get https://ghcr.io/v2/: dial tcp 140.82.121.33:443: i/o timeout",
        "error parsing HTTP 503 response body",
    ],
)
def test_a_persistent_registry_denial_is_not_reported_as_a_missing_image(
    tmp_path: Path, stderr_text: str
) -> None:
    """Login SUCCEEDS, the manifest read is refused anyway — the gate must stand down.

    This is the iter-2 HIGH. `docker login` proves the credential is valid, not that it
    may read this package, so a gate that treats "logged in" as "the answer is
    trustworthy" still converts a GHCR access failure into a missing-image hard fail.
    """
    rc, out = _run_gate_per_ref(tmp_path, api=[(1, stderr_text)], frontend=[(1, stderr_text)])
    assert rc == 0, (
        "a registry that refused to answer must not hard-fail the deploy "
        f"(stderr was {stderr_text!r}); rc={rc}, output:\n{out}"
    )
    assert "REACHED THE EXPENSIVE WORK" in out, (
        f"the deploy must fall through to the pull's own fresh login:\n{out}"
    )
    assert "no published GHCR image" not in out, (
        f"a registry ACCESS failure was announced as a MISSING IMAGE:\n{out}"
    )
    assert "SKIPPING" in out, f"standing down must be stated, not silent:\n{out}"
    assert "authorized to READ" in out, (
        "the stand-down must name the actual distinction — logged in is not authorized "
        f"to read this package — so a human is not sent after ensure-main-image:\n{out}"
    )


def test_a_transient_denial_that_clears_on_retry_counts_as_published(tmp_path: Path) -> None:
    """Positive control for the retry: one flake must not cost a stand-down.

    The stub's FIRST answer is a denial and its second is the manifest. If the attempt
    counter did not advance — it lives on disk precisely because the probe runs in a
    subshell — this would stand down, and the assertion below says so.
    """
    rc, out = _run_gate_per_ref(
        tmp_path,
        api=[(1, INCONCLUSIVE), (0, "")],
        frontend=[(1, INCONCLUSIVE), (0, "")],
    )
    assert rc == 0, f"a flake that cleared on retry must not fail the deploy:\n{out}"
    assert "REACHED THE EXPENSIVE WORK" in out, f"the deploy must proceed:\n{out}"
    assert "SKIPPING" not in out, (
        f"the retry succeeded, so there was nothing to stand down from:\n{out}"
    )


@pytest.mark.parametrize("absent_side", ["api", "frontend"])
def test_a_definitive_absent_on_either_image_alone_still_fails_fast(
    tmp_path: Path, absent_side: str
) -> None:
    """#1431: BOTH images must be published, so either one missing is the broken invariant.

    The three-way verdict must not weaken the AND that `image_exists` enforced.
    """
    present: list[tuple[int, str]] = [(0, "")]
    absent: list[tuple[int, str]] = [(1, DEFINITIVE_ABSENT)]
    rc, out = _run_gate_per_ref(
        tmp_path,
        api=absent if absent_side == "api" else present,
        frontend=absent if absent_side == "frontend" else present,
    )
    assert rc != 0, f"{absent_side} is definitively absent and the deploy continued:\n{out}"
    assert "no published GHCR image" in out and SHA in out, (
        f"the hard-fail must name the sha:\n{out}"
    )


def test_a_definitive_absent_outranks_an_inconclusive_answer(tmp_path: Path) -> None:
    """One ref unreadable, the other NAMED absent -> still a hard fail.

    A positive fact about one image outweighs a missing answer about the other: whatever
    the api package would have said, the frontend image is not published, so the #1785
    invariant is broken and the refusal is one we can stand behind.
    """
    rc, out = _run_gate_per_ref(
        tmp_path,
        api=[(1, INCONCLUSIVE)],
        frontend=[(1, DEFINITIVE_ABSENT)],
    )
    assert rc != 0, (
        "a DEFINITIVE absent on one image must outrank an inconclusive answer on the "
        f"other — the invariant is broken either way:\n{out}"
    )
    assert "no published GHCR image" in out, f"the hard-fail message must appear:\n{out}"


@pytest.mark.parametrize("odd_status", [127, 126, 3, 130])
def test_an_unrecognised_probe_status_stands_down_rather_than_claiming_published(
    tmp_path: Path, odd_status: int
) -> None:
    """A status images_verdict does not recognise must mean "no answer", not "published".

    Found by a positive control, not by review: a scratch harness that extracted the
    helpers from the wrong starting line left `manifest_verdict` undefined, so every
    call returned 127 (command not found) — and the DEFINITIVE-ABSENT control quietly
    reported PROCEEDS. The combiner was testing for 1 and for 2 and falling through to
    `return 0`, so any other status was read as "both manifests are there".

    The practical outcome of falling through is the same either way — the deploy
    proceeds to the pull, exactly as it did before #1785 — but one of them says so in
    the log and the other claims a fact it does not have. After an iter-1 finding about
    a gate announcing the wrong reason, the silent version is not the one to keep.
    """
    state = tmp_path / "stub_state"
    state.mkdir(exist_ok=True)
    preamble = (
        _docker_ref_stub(0, [(0, "")], [(0, "")])
        + "sleep() { :; }\n"
        + _extract_probe_helpers()
        # Redefined AFTER the real helpers, so this is the definition that binds.
        + f"manifest_verdict() {{ return {odd_status}; }}\n"
    )
    rc, out = _run_gate(
        tmp_path,
        preamble,
        STUB_STATE=str(state),
        NEW_SHA=SHA,
        IMAGE_OWNER="enunezvn",
        GHCR_TOKEN="t",
    )
    assert rc == 0, f"an unrecognised status must not hard-fail the deploy:\n{out}"
    assert "REACHED THE EXPENSIVE WORK" in out, f"the deploy must still proceed:\n{out}"
    assert "SKIPPING" in out, (
        f"status {odd_status} was silently read as 'both images published':\n{out}"
    )


@pytest.mark.parametrize(
    ("label", "answers", "expected_rc", "expected_probes"),
    [
        ("present", [(0, "")], 0, 1),
        ("definitive absent, NOT retried", [(1, DEFINITIVE_ABSENT)], 1, 1),
        ("inconclusive then present", [(1, INCONCLUSIVE), (0, "")], 0, 2),
        ("inconclusive twice", [(1, INCONCLUSIVE), (1, INCONCLUSIVE)], 1, 2),
    ],
)
def test_manifest_present_still_answers_what_the_sha_walk_relied_on(
    tmp_path: Path,
    label: str,
    answers: list[tuple[int, str]],
    expected_rc: int,
    expected_probes: int,
) -> None:
    """`manifest_present` is now expressed via the three-way verdict — pin what it means.

    `select_built_sha`/`image_exists` are the OTHER consumer of these helpers, and
    scripts/test_deploy_sha_selection.sh stubs `image_exists` outright, so nothing else
    in the repo exercises this function against a real probe. Both halves are pinned:
    the boolean, AND the PROBE COUNT — a definitive absent must still cost exactly one
    manifest read, because retrying it is what would slow the walk (and the pathological
    nothing-built window) that the no-retry rule exists to keep fast.
    """
    state = tmp_path / "probe_state"
    state.mkdir(exist_ok=True)
    counter = state / "api"
    runner = tmp_path / "mp.sh"
    runner.write_text(
        "set -e\n"
        + _docker_ref_stub(0, answers, answers)
        + "sleep() { :; }\n"
        + _extract_probe_helpers()
        + '\nif manifest_present "ghcr.io/enunezvn/e2i-api:sha"; then\n'
        '  echo "RESULT=0"\n'
        "else\n"
        '  echo "RESULT=1"\n'
        "fi\n"
        f'printf "PROBES=%s\\n" "$(cat {counter} 2>/dev/null || echo 0)"\n'
    )
    proc = subprocess.run(
        ["bash", str(runner)],
        env={"PATH": "/usr/bin:/bin", "STUB_STATE": str(state)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = proc.stdout + proc.stderr
    got_rc = next((ln for ln in out.splitlines() if ln.startswith("RESULT=")), "RESULT=?")
    got_probes = next((ln for ln in out.splitlines() if ln.startswith("PROBES=")), "PROBES=?")
    assert (got_rc, got_probes) == (f"RESULT={expected_rc}", f"PROBES={expected_probes}"), (
        f"manifest_present changed meaning for {label!r}: expected "
        f"RESULT={expected_rc}/PROBES={expected_probes}, got {got_rc}/{got_probes}.\n{out}"
    )


def test_the_absent_classification_has_exactly_one_source_of_truth() -> None:
    """The definitive-absent string list must exist ONCE in the rollout script.

    The sha walk (`select_built_sha`) and the #1785 gate now ask two different questions
    of the same classification. Two copies of these strings would let them drift into
    disagreeing about what "absent" means — and a gate that calls a definitive absent
    "inconclusive", or the reverse, is this defect's own family. Asserts the COUNT, so a
    failure prints what it actually found.
    """
    script = _ssh_script(ROLLOUT_ID)
    needle = '*"no such manifest"*|*"manifest unknown"*|*"not found"*|*"UNKNOWN"*'
    occurrences = [(i, ln.strip()) for i, ln in enumerate(script.splitlines()) if needle in ln]
    assert len(occurrences) == 1, (
        "the definitive-absent classification must have a single source of truth; found "
        f"{len(occurrences)} copies:\n" + "\n".join(f"  line {i}: {ln}" for i, ln in occurrences)
    )


# --------------------------------------------------------------------------- #
# 6. #1784 — the cleanup step must not prune into a half-recovered box
#
# Codex MED. The prune comment justified `-a` with "any rollback the rollout step
# performed rebuilt/repulled from source and COMPLETED before this step starts". That
# claim does not hold: every `up` inside `rollback_to_prev` is best-effort —
#
#     $COMPOSE_CMD up -d ... || echo "==> WARN: rollback 'up' ... failed — droplet may
#                                     be in a PARTIAL state; manual intervention required"
#
# — so a double fault (the deploy fails AND its own rollback partly fails) leaves
# services with no container at all. `prune -a` spares only images that HAVE a
# container, so precisely in that state it reclaims images an operator is mid-recovery
# with. Splitting the prune into its own always-running step is what made this
# reachable, so the fix belongs in the same PR: the image prune is conditioned on the
# rollout having actually succeeded. The build-cache prune is not — it is never a
# recovery input, and after a local-build attempt it is where the bulk of the garbage
# is. Also covers the LOW: a rollout that never RAN reports `skipped`, which is not
# `success`, so it takes the conservative path by construction.
# --------------------------------------------------------------------------- #
def _run_cleanup(tmp_path: Path, rollout_outcome: str) -> list[str]:
    """Execute the SHIPPED cleanup script and return the docker commands it issued."""
    log = tmp_path / "docker.log"
    runner = tmp_path / "cleanup.sh"
    runner.write_text(
        f'docker() {{ echo "$*" >> "{log}"; return 0; }}\n' + _ssh_script(CLEANUP_ID) + "\n"
    )
    proc = subprocess.run(
        ["bash", str(runner)],
        env={"PATH": "/usr/bin:/bin", "ROLLOUT_OUTCOME": rollout_outcome},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"the cleanup script must never fail its own step; rc={proc.returncode}\n"
        f"{proc.stdout}{proc.stderr}"
    )
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize(
    ("outcome", "expect_image_prune"),
    [
        ("success", True),
        ("failure", False),
        ("cancelled", False),
        ("skipped", False),
        ("", False),
    ],
)
def test_image_prune_runs_only_after_a_rollout_that_actually_succeeded(
    tmp_path: Path, outcome: str, expect_image_prune: bool
) -> None:
    """Prints the commands it OBSERVED, so a wrong verdict shows its own derivation."""
    issued = _run_cleanup(tmp_path, outcome)
    image_prunes = [c for c in issued if c.startswith("image prune")]
    builder_prunes = [c for c in issued if c.startswith("builder prune")]
    assert bool(image_prunes) is expect_image_prune, (
        f"ROLLOUT_OUTCOME={outcome!r}: expected the image prune "
        f"{'to run' if expect_image_prune else 'to be SKIPPED'}; docker calls were {issued}"
    )
    assert builder_prunes, (
        f"ROLLOUT_OUTCOME={outcome!r}: the build cache is never a recovery input and "
        f"must be reclaimed on every path; docker calls were {issued}"
    )


def test_cleanup_step_is_told_the_rollout_outcome() -> None:
    """The condition is worthless if the value never reaches the droplet.

    `env:` alone does not cross the SSH boundary — appleboy/ssh-action forwards only
    what `envs:` names, the same wiring IMAGE_OWNER/GHCR_USER/GHCR_TOKEN already use.
    """
    step = _step(CLEANUP_ID)
    env = step.get("env") or {}
    assert "ROLLOUT_OUTCOME" in env, (
        f"the cleanup step never reads the rollout's outcome; its env is {env}"
    )
    assert "steps.rollout.outcome" in str(env["ROLLOUT_OUTCOME"]), (
        "ROLLOUT_OUTCOME must come from the rollout STEP, not the job; got "
        f"{env['ROLLOUT_OUTCOME']!r}"
    )
    forwarded = [n.strip() for n in str((step.get("with") or {}).get("envs", "")).split(",")]
    assert "ROLLOUT_OUTCOME" in forwarded, (
        "ROLLOUT_OUTCOME is set on the runner but never forwarded over SSH, so the "
        f"droplet sees it empty and prunes as if the rollout failed; envs are {forwarded}"
    )


def test_prune_comment_no_longer_claims_rollback_always_completed() -> None:
    """The false invariant has to go, because it is the argument for deleting the guard.

    A comment asserting "a rollback always completed before this step" reads as a proof
    that the condition above is redundant. It is not true: `rollback_to_prev` WARNs and
    continues on a failed `up`.

    Two corrections are baked into the shape of this guard.

    First, a naive `stale not in prose` punishes the BETTER comment: quoting the old
    sentence in order to contradict it is more useful to the next reader than deleting
    it silently, and the same trap already caught me on the drift-check comment. So the
    claim may survive only inside a window that also contradicts it.

    Second, the load-bearing assertion is the unconditional one. Guarding everything
    behind `if stale in prose` goes vacuous the moment the sentence is deleted — and
    the first corrective marker I reached for ("best-effort") was one lowercasing away
    from being satisfied by the unrelated `Best-effort (|| true)` sentence already in
    the block. Near-miss matching is how a guard fails open.
    """
    prose = _prose(_ssh_script(CLEANUP_ID))
    stale = "completed before this step starts, so dropping old SHA images cannot break a rollback"
    idx = prose.find(stale)
    if idx >= 0:
        window = prose[max(0, idx - 90) : idx + len(stale) + 90]
        contradictions = ("used to say", "not an invariant", "is not true", "does not hold")
        assert any(marker in window for marker in contradictions), (
            "the prune comment still states as a live invariant that a rollback has "
            "completed by the time this step runs. It has not, necessarily. Quote it to "
            f"correct it, or drop it — the text around it says neither:\n{window}"
        )
    assert "rollback_to_prev" in prose, (
        "nothing in the prune comment names `rollback_to_prev`, whose every `up` is "
        "WARN-and-continue — which is the entire reason the image prune is now "
        f"conditional. Unrecorded, the condition reads as redundant:\n{prose}"
    )
