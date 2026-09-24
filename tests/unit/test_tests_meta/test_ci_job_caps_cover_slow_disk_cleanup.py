"""CI jobs that free runner disk must budget for a slow-disk runner.

2026-09-23: on main c4e191472, ``Agents Unit Tests (3)`` was cancelled at its
25-minute ``timeout-minutes`` while its tests were passing (75% done). The
``jlumbroso/free-disk-space`` step took 20 m 43 s instead of ~2.5 min: every
category was ~10x slower (Android 11 m 44 s vs 1 m 08 s), i.e. a slow-disk runner.
The same commit's deploy gate passed the same shard in 8.5 min.

Policy, per job that runs the cleanup (measured over the last 15 green runs):
  * cap >= green max + SLOW_RUNNER_PENALTY  -> a slow-disk runner still completes;
  * cap >= green max + stall-watchdog window -> on a normal runner a late session
    stall is diagnosed by the watchdog before the job is cancelled.
Every cleanup step carries a FATAL step timeout above the observed 20 m 43 s and
never ``continue-on-error``: proceeding after an incomplete cleanup recreates the
documented ``[Errno 28] No space left on device`` failure.
"""

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

SLOW_RUNNER_PENALTY_MIN = 20  # measured 20m43s cleanup vs ~2.5 min normal (+18.3)
CLEANUP_STEP_TIMEOUT_MIN = 25  # fatal; must exceed the observed 20m43s

# (workflow, job id) -> (green max minutes over 15 runs, stall watchdog minutes)
MEASURED = {
    ("backend-tests.yml", "type-check"): (14, 0),
    ("backend-tests.yml", "integration-tests"): (17, 0),
    ("backend-tests.yml", "agents-tests"): (21, 10),  # E2I_PYTEST_STALL_TIMEOUT 600
    ("backend-tests.yml", "unit-tests"): (24, 10),  # E2I_PYTEST_STALL_TIMEOUT 600
    ("backend-tests.yml", "heavy-unit-tests"): (17, 20),  # E2I_PYTEST_STALL_TIMEOUT 1200
    ("tier1-5-test.yml", "tier1-5-harness"): (9, 0),
    ("slow-tests.yml", "twin-effect-recovery"): (9, 0),
}


def _is_cleanup(step) -> bool:
    return "free-disk-space" in str(step.get("uses", ""))


def _capped_cleanup_jobs():
    """Every job with the cleanup step and an explicit cap (no cap = 360 min default)."""
    found = {}
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in (yaml.safe_load(wf.read_text()).get("jobs") or {}).items():
            if any(_is_cleanup(s) for s in job.get("steps", [])) and "timeout-minutes" in job:
                found[(wf.name, name)] = job
    return found


def test_every_capped_cleanup_job_is_measured():
    assert set(_capped_cleanup_jobs()) == set(MEASURED), (
        "a capped job gained or lost the free-disk-space step; measure its green max "
        f"and watchdog and add it here: found={sorted(_capped_cleanup_jobs())}"
    )


def test_caps_fit_a_slow_runner_and_the_stall_watchdog():
    short = {}
    for key, job in _capped_cleanup_jobs().items():
        green_max, watchdog = MEASURED[key]
        need = green_max + max(SLOW_RUNNER_PENALTY_MIN, watchdog)
        if job["timeout-minutes"] < need:
            short[key] = (job["timeout-minutes"], need)
    assert not short, f"(cap, needed) below policy: {short}"


def test_every_cleanup_step_has_a_fatal_step_timeout():
    bad = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in (yaml.safe_load(wf.read_text()).get("jobs") or {}).items():
            for step in job.get("steps", []):
                if not _is_cleanup(step):
                    continue
                t = step.get("timeout-minutes")
                if t is None or t < CLEANUP_STEP_TIMEOUT_MIN or step.get("continue-on-error"):
                    bad.append((wf.name, name, t, step.get("continue-on-error")))
    assert not bad, f"cleanup steps need timeout-minutes >= {CLEANUP_STEP_TIMEOUT_MIN} and no continue-on-error: {bad}"
