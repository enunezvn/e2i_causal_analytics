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


# --------------------------------------------------------------------------- #
# 4. #1785 — fail fast when the resolved sha has no GHCR image
# --------------------------------------------------------------------------- #
FAIL_FAST_OPENER = 'if ! image_exists "$NEW_SHA"; then'


def _extract_fail_fast_block() -> str:
    """The SHIPPED assertion, verbatim, from opener to its own-indent `fi`."""
    script = _ssh_script(ROLLOUT_ID)
    lines = script.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == FAIL_FAST_OPENER), None)
    assert start is not None, (
        "#1785: the droplet never asserts a GHCR manifest exists for the resolved "
        f"NEW_SHA — expected a line {FAIL_FAST_OPENER!r} in the rollout script"
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "fi" and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            return "\n".join(ln[indent:] for ln in lines[start : j + 1])
    raise AssertionError("the #1785 assertion has no closing `fi` at its own indent")


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
            "#1785 manifest assertion": FAIL_FAST_OPENER,
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


def _run_fail_fast(tmp_path: Path, image_exists_rc: int, **env: str) -> tuple[int, str]:
    """Drive the shipped assertion down a chosen branch with a stubbed probe."""
    body = (
        f"image_exists() {{ return {image_exists_rc}; }}\n"
        + _extract_fail_fast_block()
        + '\necho "REACHED THE EXPENSIVE WORK"\n'
    )
    runner = tmp_path / "failfast.sh"
    runner.write_text("set -e\n" + body)
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
        image_exists_rc=1,
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
    rc, out = _run_fail_fast(tmp_path, image_exists_rc=0, NEW_SHA=SHA, IMAGE_OWNER="enunezvn")
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
    _, out = _run_fail_fast(tmp_path, image_exists_rc=1, NEW_SHA=SHA, IMAGE_OWNER="enunezvn")
    lowered = out.lower()
    for phrase, why in (
        ("local build", "must name the path it is refusing"),
        ("rollback", "must state that a box-only image is not a rollback target"),
        ("gh workflow run deploy.yml", "must give the recovery command"),
    ):
        assert phrase in lowered, f"the #1785 message {why}; got:\n{out}"


@pytest.mark.parametrize("reason", ["", "GHCR auth unavailable — no ancestor was probed"])
def test_fail_fast_survives_an_absent_fallback_reason(tmp_path: Path, reason: str) -> None:
    """`[ -n "$X" ] && echo ...` under `set -e` exits on the empty case and swallows
    every line after it. The message must survive an empty FALLBACK_REASON with its
    recovery instructions intact."""
    rc, out = _run_fail_fast(
        tmp_path, image_exists_rc=1, NEW_SHA=SHA, IMAGE_OWNER="enunezvn", FALLBACK_REASON=reason
    )
    assert rc != 0
    assert "gh workflow run deploy.yml" in out, (
        f"the recovery line was swallowed with FALLBACK_REASON={reason!r}:\n{out}"
    )
    if reason:
        assert reason in out, f"a present FALLBACK_REASON must be shown:\n{out}"
