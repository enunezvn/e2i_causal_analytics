"""Contract test: the slow-tests nightly alarm covers EVERY job, incl. Job B.

Background (gap G1)
-------------------
``slow-tests.yml`` is scheduled-only and rotted RED for ~12 days unnoticed
(#594). It already has a ``report-failure`` job that opens/updates a tracking
issue on failure — but its ``needs``/``if`` covered only Jobs A/C/D
(slow-tests, memory-perf-tests, synthetic-regime-tests). Job B
(``excluded-heavy-tests``: test_tier0_e2e + test_adaptive_criteria_e2e +
test_model_trainer_evaluation_modes) is ``continue-on-error: true``, so its
failures never tripped the alarm and the full 7-agent tier0 e2e could regress
silently.

Graduation (#617)
-----------------
Job B's three e2e suites are now stabilized, so it has been promoted from
allowed-to-fail to a hard must-pass: both the job-level and the heavy-step
``continue-on-error: true`` are removed, so a test (or infra) failure red-Xs
the workflow. With the job no longer masked, ``needs.excluded-heavy-tests.result``
is accurate, and the alarm keys off it like Jobs A/C/D — which also covers an
infra failure of the now-blocking job (the ``heavy_result`` output alone would
miss it). The ``heavy_result`` output is retained purely so the SUMMARY can
report the heavy test step's granular outcome distinctly from job-level infra
failures.

This test pins that contract so the alarm can't silently stop covering a job
and the graduation can't be silently reverted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "slow-tests.yml"

# The heavy (now blocking, #617) job and the alarm job.
HEAVY_JOB = "excluded-heavy-tests"
ALARM_JOB = "report-failure"
HEAVY_OUTPUT = "heavy_result"
# Jobs whose failure must open a tracking issue.
ALARMED_JOBS = {
    "slow-tests",
    "memory-perf-tests",
    "synthetic-regime-tests",
    HEAVY_JOB,
}


def _load() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _jobs() -> dict:
    return _load().get("jobs", {})


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    return list(needs)


def test_alarm_job_exists() -> None:
    assert ALARM_JOB in _jobs(), f"{ALARM_JOB} job missing from slow-tests.yml"


def test_alarm_depends_on_every_alarmed_job() -> None:
    """The alarm job must `needs` every job it is supposed to watch — including
    Job B, which was previously omitted (gap G1)."""
    alarm = _jobs()[ALARM_JOB]
    needs = set(_needs(alarm))
    missing = sorted(ALARMED_JOBS - needs)
    assert not missing, (
        f"{ALARM_JOB}.needs is missing {missing}: those jobs' failures cannot "
        "trip the tracking-issue alarm."
    )


def test_alarm_condition_references_every_alarmed_job() -> None:
    """The alarm `if` must reference each watched job's failure signal.

    Post-graduation (#617) Job B is no longer continue-on-error, so its
    ``.result`` is accurate (not masked to success). The alarm keys off
    ``.result == 'failure'`` for ALL FOUR jobs — for Job B this also catches an
    infra failure (e.g. pip install / mlflow boot) that never reached the heavy
    pytest step, which the ``heavy_result`` output alone would miss.
    """
    cond = str(_jobs()[ALARM_JOB].get("if", ""))
    for job in ALARMED_JOBS:
        assert f"needs.{job}.result" in cond, (
            f"alarm `if` must reference needs.{job}.result so its failure alarms."
        )
    # Stays schedule-scoped (we don't want issue spam from manual dispatch).
    assert "github.event_name == 'schedule'" in cond


def test_heavy_job_exposes_outcome_output() -> None:
    """Job B must publish a ``heavy_result`` output sourced from the heavy test
    step's outcome, so the SUMMARY can report the heavy pytest step's granular
    outcome distinctly from a job-level infra failure (retained post-#617)."""
    heavy = _jobs()[HEAVY_JOB]
    outputs = heavy.get("outputs", {})
    assert HEAVY_OUTPUT in outputs, (
        f"{HEAVY_JOB} must declare outputs.{HEAVY_OUTPUT} so the summary can "
        "surface the heavy test step's outcome distinctly."
    )
    assert "steps.heavy.outcome" in str(outputs[HEAVY_OUTPUT]), (
        "heavy_result must be sourced from the heavy test step's .outcome."
    )


def test_excluded_heavy_job_is_blocking() -> None:
    """#617 graduation: Job B must NOT be job-level continue-on-error.

    Its three e2e suites are stabilized; a failure must red-X the workflow (hard
    must-pass) instead of being silently allowed-to-fail."""
    heavy = _jobs()[HEAVY_JOB]
    assert heavy.get("continue-on-error") is not True, (
        f"{HEAVY_JOB} still has job-level continue-on-error: true — Job B was "
        "graduated to a hard must-pass (#617); remove it so a failure is blocking."
    )


def test_heavy_test_step_is_blocking() -> None:
    """The heavy test step must have ``id: heavy`` (so its outcome still feeds
    the ``heavy_result`` output for the summary) and must NOT carry step-level
    ``continue-on-error`` — post-#617 a test failure must fail the job."""
    heavy = _jobs()[HEAVY_JOB]
    steps = heavy.get("steps", [])
    test_steps = [s for s in steps if s.get("id") == "heavy"]
    assert len(test_steps) == 1, (
        "exactly one step in excluded-heavy-tests must have id: heavy "
        "(the pytest step whose outcome the summary keys off)."
    )
    step = test_steps[0]
    assert step.get("continue-on-error") is not True, (
        "the id:heavy step must NOT be continue-on-error post-#617: a test "
        "failure must fail the job so Job B is a real hard gate."
    )
    # Sanity: it is actually the pytest step.
    assert "pytest" in str(step.get("run", ""))


def test_synthetic_pipeline_e2e_runs_in_slow_lane() -> None:
    """The synthetic ATE-recovery e2e must be slow-marked AND not --ignore'd by
    Job A, so its within_tolerance gate actually executes nightly (gap G2).
    Before this, tests/e2e/ ran in no lane at all."""
    e2e = REPO_ROOT / "tests" / "e2e" / "test_synthetic_pipeline_e2e.py"
    src = e2e.read_text()
    assert "pytestmark = pytest.mark.slow" in src or "@pytest.mark.slow" in src, (
        "test_synthetic_pipeline_e2e.py must be slow-marked so Job A "
        "(`pytest tests/ -m slow`) collects its ATE-recovery gate."
    )
    job_a = _jobs()["slow-tests"]
    run_blocks = " ".join(str(s.get("run", "")) for s in job_a.get("steps", []))
    assert "--ignore=tests/e2e" not in run_blocks, (
        "Job A must not --ignore tests/e2e or the synthetic ATE-recovery gate "
        "stops running nightly (gap G2 regression)."
    )


def test_summary_reads_heavy_output_not_masked_result() -> None:
    """The summary must report Job B's REAL outcome via the output, not the
    continue-on-error-masked ``.result`` (which would always say success)."""
    summary = _jobs().get("summary", {})
    step_envs = " ".join(str(s.get("env", {})) for s in summary.get("steps", []))
    assert f"needs.{HEAVY_JOB}.outputs.{HEAVY_OUTPUT}" in step_envs, (
        "summary must surface excluded-heavy-tests via its heavy_result output."
    )


def _job_uses_checkout(job: dict) -> bool:
    return any("actions/checkout" in str(s.get("uses", "")) for s in job.get("steps", []))


def _job_or_steps_set_gh_repo(job: dict) -> bool:
    """GH_REPO present at job level or on any step's env."""
    if "GH_REPO" in (job.get("env") or {}):
        return True
    return any("GH_REPO" in (s.get("env") or {}) for s in job.get("steps", []))


def test_alarm_job_can_resolve_repo() -> None:
    """The alarm must be able to RUN, not just be wired correctly.

    ``report-failure`` invokes ``gh issue list`` / ``gh issue create`` under
    ``set -euo pipefail``. With neither an ``actions/checkout`` step (which gives
    gh a git remote to resolve) nor an explicit ``GH_REPO`` env, gh dies with
    ``failed to run git: fatal: not a git repository`` — and the only nightly
    slow-tests/tier0-e2e rot alarm silently never files an issue (it failed that
    way unnoticed for 5+ days). Require one of the two so a future edit can't
    re-break the alarm's ability to execute (#615).
    """
    alarm = _jobs()[ALARM_JOB]
    assert _job_uses_checkout(alarm) or _job_or_steps_set_gh_repo(alarm), (
        f"{ALARM_JOB} runs gh without a repo context: add `actions/checkout` or "
        "set `env: GH_REPO: ${{ github.repository }}` so gh can resolve the repo "
        "(otherwise it fails 'not a git repository' and no tracking issue is filed)."
    )
