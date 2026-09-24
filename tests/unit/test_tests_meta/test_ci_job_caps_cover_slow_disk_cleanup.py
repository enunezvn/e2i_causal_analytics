"""Backend-tests jobs that free runner disk must budget for a slow runner.

2026-09-23: on main c4e191472, ``Agents Unit Tests (3)`` was cancelled at its
25-minute ``timeout-minutes`` while its tests were passing (75% done). The
``jlumbroso/free-disk-space`` step took 20 m 43 s instead of the usual ~2.5 min:
every category was ~10x slower (Android 11 m 44 s vs 1 m 08 s), i.e. a slow-disk
runner, not one expensive category. The same commit's deploy gate passed the same
shard in 8.5 min. Hang detection belongs to pytest's per-test ``--timeout``; the job
cap only has to fit a real run on a slow runner.

The floor below is (measured job mean over 15 green runs) + 18 min (measured
slow-runner cleanup penalty), rounded up.
"""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "backend-tests.yml"

# job id -> minimum timeout-minutes (mean + slow-cleanup penalty, rounded up)
FLOORS = {
    "type-check": 30,
    "integration-tests": 30,
    "agents-tests": 35,
    "heavy-unit-tests": 40,
    "unit-tests": 40,
}


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _runs_disk_cleanup(job) -> bool:
    return any("free-disk-space" in str(step.get("uses", "")) for step in job.get("steps", []))


def test_every_disk_cleanup_job_has_a_floor():
    cleanup_jobs = {name for name, job in _jobs().items() if _runs_disk_cleanup(job)}
    assert cleanup_jobs == set(FLOORS), (
        "a backend-tests job gained or lost the free-disk-space step; give it a "
        f"slow-runner floor here: cleanup={sorted(cleanup_jobs)} floors={sorted(FLOORS)}"
    )


def test_disk_cleanup_jobs_budget_for_a_slow_runner():
    jobs = _jobs()
    short = {
        name: jobs[name].get("timeout-minutes")
        for name, floor in FLOORS.items()
        if (jobs[name].get("timeout-minutes") or 0) < floor
    }
    assert not short, f"timeout-minutes below the slow-runner floor: {short} (floors {FLOORS})"
