"""
Unit test: RAGAS evaluation workflow must have a timeout-minutes guard.

Rationale (#504): The ragas-evaluation job is manual-only (workflow_dispatch) and
takes ~96 min on the CI OpenAI key. Without a timeout, a hung run burns the
360-min GitHub Actions default. This test asserts that:
  1. timeout-minutes is present on the ragas-evaluation job.
  2. timeout-minutes is less than 360 (not relying on the default).
"""

from pathlib import Path

import yaml  # noqa: PLC0415

WORKFLOW_PATH = (
    Path(__file__).parent.parent.parent / ".github" / "workflows" / "ragas-evaluation.yml"
)

JOB_KEY = "ragas-evaluation"
MAX_ALLOWED_TIMEOUT = 360  # GitHub Actions default — anything < this is an explicit guard


def _load_job() -> dict:
    with WORKFLOW_PATH.open() as fh:
        workflow = yaml.safe_load(fh)
    jobs = workflow.get("jobs", {})
    assert JOB_KEY in jobs, f"Job '{JOB_KEY}' not found in {WORKFLOW_PATH}"
    return jobs[JOB_KEY]


def test_ragas_evaluation_job_has_timeout_minutes():
    """timeout-minutes must be explicitly set on the ragas-evaluation job."""
    job = _load_job()
    assert "timeout-minutes" in job, (
        f"Job '{JOB_KEY}' is missing 'timeout-minutes'. "
        "A hung manual run will burn the 360-min GitHub default. "
        "Add 'timeout-minutes: 150' (comfortably above the ~96-min real runtime)."
    )


def test_ragas_evaluation_timeout_is_below_github_default():
    """timeout-minutes must be less than 360 (the GitHub Actions default)."""
    job = _load_job()
    timeout = job.get("timeout-minutes")
    if timeout is None:
        return  # Already caught by the first test; avoid duplicate failures.
    assert timeout < MAX_ALLOWED_TIMEOUT, (
        f"Job '{JOB_KEY}' has timeout-minutes={timeout}, "
        f"which equals or exceeds the GitHub default ({MAX_ALLOWED_TIMEOUT}). "
        "Set it to a value below 360 (e.g. 150) to act as a real fail-fast guard."
    )
