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

Under job-level ``continue-on-error``, ``needs.<job>.result`` reads as
``success`` even when the job's tests fail, so the alarm cannot key off the job
result. The fix captures the heavy test step's ``outcome`` into a job output
(``heavy_result``) and the alarm fires when that output is ``failure`` — Job B
stays allowed-to-fail (no hard must-pass), but a failure is now LOUD.

This test pins that contract so the alarm can't silently stop covering a job.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "slow-tests.yml"

# The heavy/allowed-to-fail job and the alarm job.
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

    Job B is continue-on-error, so its signal is the ``heavy_result`` OUTPUT
    (not ``.result``, which reads success); A/C/D use ``.result == 'failure'``.
    """
    cond = str(_jobs()[ALARM_JOB].get("if", ""))
    # Job B: keyed off its output, since result is masked by continue-on-error.
    assert f"needs.{HEAVY_JOB}.outputs.{HEAVY_OUTPUT}" in cond, (
        "alarm `if` must check the excluded-heavy job's heavy_result OUTPUT — "
        "its .result is masked to success by continue-on-error."
    )
    for job in ALARMED_JOBS - {HEAVY_JOB}:
        assert f"needs.{job}.result" in cond, (
            f"alarm `if` must reference needs.{job}.result so its failure alarms."
        )
    # Stays schedule-scoped (we don't want issue spam from manual dispatch).
    assert "github.event_name == 'schedule'" in cond


def test_heavy_job_exposes_outcome_output() -> None:
    """Job B must publish a ``heavy_result`` output sourced from the heavy test
    step's outcome, so the alarm (and summary) can read its real result despite
    continue-on-error."""
    heavy = _jobs()[HEAVY_JOB]
    outputs = heavy.get("outputs", {})
    assert HEAVY_OUTPUT in outputs, (
        f"{HEAVY_JOB} must declare outputs.{HEAVY_OUTPUT} so its failure is "
        "observable despite continue-on-error."
    )
    assert "steps.heavy.outcome" in str(outputs[HEAVY_OUTPUT]), (
        "heavy_result must be sourced from the heavy test step's .outcome."
    )


def test_heavy_test_step_is_outcome_capturable() -> None:
    """The heavy test step must have ``id: heavy`` and step-level
    ``continue-on-error: true`` so its outcome is recorded as failure (not
    success) and the job proceeds to publish the output."""
    heavy = _jobs()[HEAVY_JOB]
    steps = heavy.get("steps", [])
    test_steps = [s for s in steps if s.get("id") == "heavy"]
    assert len(test_steps) == 1, (
        "exactly one step in excluded-heavy-tests must have id: heavy "
        "(the pytest step whose outcome the alarm keys off)."
    )
    step = test_steps[0]
    assert step.get("continue-on-error") is True, (
        "the id:heavy step needs step-level continue-on-error: true so a test "
        "failure yields outcome=failure while the job still publishes its output."
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
