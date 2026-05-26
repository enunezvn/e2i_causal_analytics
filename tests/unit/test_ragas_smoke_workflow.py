"""Contract test: the ragas-smoke workflow is the automatic, $0, key-free guard.

#504 made the full gpt-4o RAGAS eval manual-only (it is OpenAI-throughput-bound).
``ragas-smoke.yml`` restores an automatic per-PR signal that the RAGAS dependency
stack still *imports* — the #491 break class — WITHOUT spending on the gpt-4o
judge. This test pins that contract so a future edit can't silently turn it into a
paid job, drop a trigger, or stop running the check.
"""

from pathlib import Path

import yaml  # noqa: PLC0415

WORKFLOW_PATH = Path(__file__).parent.parent.parent / ".github" / "workflows" / "ragas-smoke.yml"

# Paths whose changes can break the RAGAS dependency import (the #491 class).
REQUIRED_PATHS = {
    "src/rag/**",
    "scripts/run_ragas_eval.py",
    "requirements-ragas.txt",
    ".github/workflows/ragas-smoke.yml",
}


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open() as fh:
        return yaml.safe_load(fh)


def _triggers(workflow: dict) -> dict:
    # PyYAML parses the bare mapping key ``on:`` as the boolean ``True``.
    return workflow.get("on", workflow.get(True, {})) or {}


def test_workflow_exists():
    assert WORKFLOW_PATH.exists(), (
        f"{WORKFLOW_PATH} is missing — the automatic smoke guard (#504/#491) is not wired."
    )


def test_triggers_on_pull_request_and_push():
    """Must run automatically on PRs (and pushes), not be manual-only."""
    triggers = _triggers(_load_workflow())
    assert "pull_request" in triggers, "smoke must trigger on pull_request"
    assert "push" in triggers, "smoke must trigger on push"


def test_path_filters_cover_the_ragas_eval_stack():
    """A change to any file that can break the import must re-run the smoke."""
    triggers = _triggers(_load_workflow())
    paths: set[str] = set()
    for event in ("pull_request", "push"):
        cfg = triggers.get(event) or {}
        paths.update(cfg.get("paths", []) or [])
    missing = REQUIRED_PATHS - paths
    assert not missing, f"workflow path filters miss: {sorted(missing)}"


def test_smoke_job_has_timeout_guard():
    """A fail-fast timeout below the 360-min GitHub default."""
    jobs = _load_workflow().get("jobs", {})
    assert jobs, "no jobs defined in ragas-smoke.yml"
    timeouts = [job.get("timeout-minutes") for job in jobs.values()]
    assert any(t is not None and t < 360 for t in timeouts), (
        "ragas-smoke needs a timeout-minutes guard below the 360-min default."
    )


def test_smoke_runs_the_dependency_check():
    """The job must actually invoke the smoke check, not just install deps."""
    assert "--smoke" in WORKFLOW_PATH.read_text(), (
        "workflow must invoke `run_ragas_eval.py --smoke`."
    )


def test_smoke_is_key_free():
    """The smoke is $0 — it must not reference the OpenAI key secret."""
    assert "OPENAI_API_KEY" not in WORKFLOW_PATH.read_text(), (
        "the smoke must be key-free/$0; it must not reference OPENAI_API_KEY."
    )
