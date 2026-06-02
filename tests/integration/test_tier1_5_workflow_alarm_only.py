"""Codify monitored-alarm-only gating for the Tier 1-5 harness — issue #600.

#600 fix has two halves:

1. A committed tier0 fixture (``scripts/tier0_output_cache/latest.pkl``) flips
   ``restore-cache.outputs.found`` to ``true`` so the harness actually RUNS the
   13 agents on every relevant PR (no longer a graceful-skip no-op).

2. Maintainer decision: an *agent / contract* failure must be a MONITORED ALARM
   — a ``::warning`` annotation + the results artifact, NOT a PR-blocking
   failure. *Infra* failures (boot-stack, install-deps) must STILL hard-fail
   (issue #263), and the #263 forcing test must stay green.

The belt-and-braces #263 assertion forbids ``continue-on-error`` on every step,
so alarm-only is implemented by capturing ``make``'s exit code in-shell and
``exit 0``-ing on agent failure (after emitting the warning) — without
``continue-on-error`` and without weakening ``main()``'s ``sys.exit`` (which keeps
``make tier1-5-test`` hard-fail for local / other-workflow use).

These tests pin that wiring. They are complementary to
``test_tier1_5_workflow_hard_fail.py`` (which guards the infra hard-fail
invariants); both must pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "tier1-5-test.yml"
_FIXTURE_PATH = _ROOT / "scripts" / "tier0_output_cache" / "latest.pkl"


def _harness_job() -> dict[str, Any]:
    workflow = cast(dict[str, Any], yaml.safe_load(_WORKFLOW_PATH.read_text()))
    jobs = workflow.get("jobs", {}) or {}
    assert "tier1-5-harness" in jobs
    return cast(dict[str, Any], jobs["tier1-5-harness"])


def _step(step_id: str) -> dict[str, Any]:
    steps = _harness_job().get("steps", []) or []
    matches = [s for s in steps if s.get("id") == step_id]
    assert len(matches) == 1, f"Expected exactly one step id={step_id}; found {len(matches)}."
    return cast(dict[str, Any], matches[0])


def test_run_harness_allows_mock_connector():
    """#606 item A: the run-harness step must set E2I_ALLOW_MOCK_CONNECTOR=1 so
    heterogeneous_optimizer's CATEEstimatorNode can construct (it raises at
    __init__ otherwise — 'MockDataConnector fallback is disabled'). The agent
    still runs on the real tier0 fixture data; the mock only satisfies the eager
    constructor. Matches the agents-tests lane convention."""
    env = _step("run-harness").get("env") or {}
    assert str(env.get("E2I_ALLOW_MOCK_CONNECTOR")) == "1", (
        f"run-harness must set E2I_ALLOW_MOCK_CONNECTOR=1 (issue #606 item A); got env={env!r}"
    )


def test_run_harness_allows_mock_llm():
    """#606 item C: the run-harness step must set E2I_ALLOW_MOCK_LLM=1 so the
    LLM-dependent agents (orchestrator, experiment_designer, tool_composer) use a
    MARKED mock instead of raising at construction in the keyless CI. Prod (no
    flag) stays fail-loud."""
    env = _step("run-harness").get("env") or {}
    assert str(env.get("E2I_ALLOW_MOCK_LLM")) == "1", (
        f"run-harness must set E2I_ALLOW_MOCK_LLM=1 (issue #606 item C); got env={env!r}"
    )


def test_committed_fixture_makes_harness_run_on_prs():
    """Headline #600 fix: the committed fixture exists (so restore-cache reports
    found=true and the harness runs), and the run step still positively gates on
    cache-present (the legitimate skip path is preserved)."""
    assert _FIXTURE_PATH.exists(), (
        "No committed tier0 fixture — the harness would still graceful-skip the "
        "13 agents on every PR (issue #600)."
    )
    run_if = _step("run-harness").get("if") or ""
    assert "steps.restore-cache.outputs.found == 'true'" in run_if


def test_agent_failure_is_a_nonblocking_alarm():
    """The run-harness step must capture make's exit code and exit 0 on agent
    failure (after emitting a ::warning), so an agent/contract regression is a
    monitored alarm rather than a PR block."""
    run = _step("run-harness").get("run") or ""
    assert "make tier1-5-test" in run, "run-harness must still invoke the harness via make."
    # Exit code captured, not allowed to abort the step.
    assert "rc=$?" in run or "rc=${PIPESTATUS" in run, (
        "run-harness must capture make's exit code (e.g. `make tier1-5-test || rc=$?`) "
        "so it can alarm instead of hard-failing the job."
    )
    # Explicit non-blocking exit on the agent-failure branch.
    assert "exit 0" in run, (
        "run-harness must `exit 0` on agent failure (monitored-alarm-only, per #600)."
    )
    # Louder than the skip ::notice.
    assert "::warning" in run, "Agent/contract regression must emit a ::warning alarm."


def test_run_harness_does_not_blanket_enable_set_e():
    """The run step must NOT use ``set -e`` / ``set -euo`` — with errexit, a
    non-zero ``make`` would abort the step before the ::warning + exit 0, which
    would re-block the job. It uses ``set -uo pipefail`` instead."""
    run = _step("run-harness").get("run") or ""
    lines = [ln.strip() for ln in run.splitlines()]
    errexit = [
        ln
        for ln in lines
        if ln == "set -e"
        or ln.startswith("set -e ")
        or ln.startswith("set -eu")
        or ln.startswith("set -euo")
        or ln.startswith("set -ex")
        or "set -o errexit" in ln
    ]
    assert not errexit, (
        f"run-harness enables errexit ({errexit}); that aborts the step before the "
        "alarm/exit 0. Use `set -uo pipefail` and capture via `rc=$?`."
    )
    assert "set -uo pipefail" in run, (
        "run-harness should keep undefined-var + pipe failures loud via `set -uo pipefail`."
    )


def test_run_harness_does_not_use_continue_on_error():
    """Alarm-only must NOT be achieved via continue-on-error (the #263
    belt-and-braces test forbids it on every step); it is done in-shell."""
    assert _step("run-harness").get("continue-on-error") is not True


def test_infra_steps_still_gate_the_harness_run():
    """Infra hard-fail is preserved: the harness only runs when install-deps
    succeeded and boot-stack reported healthy, so the alarm exit 0 can only ever
    mask agent failures, never infra failures."""
    run_if = _step("run-harness").get("if") or ""
    assert "steps.install-deps.outcome == 'success'" in run_if
    assert "steps.boot-stack.outputs.ok == 'true'" in run_if


def test_results_artifact_upload_is_always():
    """The alarm points reviewers at the results JSON; that upload must fire
    even when the harness reports failures (if: always())."""
    steps = _harness_job().get("steps", []) or []
    upload_steps = [
        s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert upload_steps, "Expected an upload-artifact step for the results JSON."
    assert any((s.get("if") or "").strip().startswith("always()") for s in upload_steps), (
        "Results artifact upload must use if: always() so the alarm has evidence on failure."
    )


# ---------------------------------------------------------------------------
# Issue #616 fix#1: honest signal (step summary + expected-fail allow-list)
# ---------------------------------------------------------------------------


def test_report_results_step_writes_github_step_summary():
    """#616: a step must surface the per-agent table to $GITHUB_STEP_SUMMARY so a
    reviewer sees N/13, not a bare green. Today there is NO step summary anywhere;
    this pins the new honest-signal step."""
    step = _step("report-results")
    run = step.get("run") or ""
    assert "GITHUB_STEP_SUMMARY" in run, (
        "report-results step must write to $GITHUB_STEP_SUMMARY (issue #616 visibility)."
    )
    assert "--summarize" in run, (
        "report-results must invoke the harness summarize mode to render the table."
    )


def test_report_results_step_enforces_expected_fail_allow_list():
    """#616: the honest gate must consult the expected-fail allow-list so a NEW
    failure beyond the known set flips to a hard fail. The allow-list lives in a
    documented job env var; the step must pass it through."""
    job = _harness_job()
    env = job.get("env") or {}
    assert "TIER1_5_EXPECTED_FAIL_AGENTS" in env, (
        "harness job must declare TIER1_5_EXPECTED_FAIL_AGENTS (the expected-fail allow-list)."
    )
    run = _step("report-results").get("run") or ""
    assert "--expected-fail" in run and "TIER1_5_EXPECTED_FAIL_AGENTS" in run, (
        "report-results must forward TIER1_5_EXPECTED_FAIL_AGENTS via --expected-fail."
    )


def test_report_results_step_is_not_continue_on_error():
    """#263: the honest gate must NOT be continue-on-error; its exit 1 on a NEW
    failure is the intended hard-fail behaviour."""
    assert _step("report-results").get("continue-on-error") is not True


def test_report_results_runs_when_harness_ran():
    """The honest gate must run whenever the harness actually executed
    (cache-present + infra-OK), even if run-harness alarm-exited 0 — otherwise a
    new failure would never reach the gate."""
    run_if = _step("report-results").get("if") or ""
    assert "always()" in run_if, (
        "report-results must use always() so the alarm-only exit 0 of run-harness "
        "does not skip the honest gate."
    )
    assert "steps.restore-cache.outputs.found == 'true'" in run_if
    assert "steps.boot-stack.outputs.ok == 'true'" in run_if
    assert "steps.install-deps.outcome == 'success'" in run_if


# ---------------------------------------------------------------------------
# Issue #618: trigger paths cover shared schema modules the contracts import
# ---------------------------------------------------------------------------


def _pr_paths() -> list[str]:
    workflow = cast(dict[str, Any], yaml.safe_load(_WORKFLOW_PATH.read_text()))
    # PyYAML parses the bare ``on:`` key as the boolean True.
    on = workflow.get("on") if "on" in workflow else workflow.get(True)
    assert isinstance(on, dict), f"Unexpected 'on' shape: {on!r}"
    return list((on.get("pull_request") or {}).get("paths") or [])


def test_trigger_paths_include_traced_schema_packages():
    """#618: a Pydantic/TypedDict change in shared modules the agent contracts
    import transitively must trigger the harness. The traced (not guessed) set
    includes src/causal_engine (the issue's wrong src/causal), src/data, src/ml,
    src/mlops, etc. Pin the highest-signal ones."""
    paths = _pr_paths()
    # The issue's ``src/causal/**`` does NOT exist; the real package is causal_engine.
    assert "src/causal/**" not in paths, (
        "src/causal/** is not a real package (issue #618 guess was wrong); "
        "use src/causal_engine/** instead."
    )
    required = {
        "src/causal_engine/**",
        "src/data/**",
        "src/ml/**",
        "src/mlops/**",
    }
    missing = required - set(paths)
    assert not missing, f"tier1-5 trigger paths missing traced schema packages: {sorted(missing)}"


def test_trigger_still_covers_original_paths():
    """Broadening must not drop the original high-signal triggers."""
    paths = set(_pr_paths())
    for original in (
        "src/agents/**",
        "src/testing/**",
        "scripts/run_tier1_5_test.py",
        "config/agents/**",
        ".github/workflows/tier1-5-test.yml",
        "docker/docker-compose.yml",
    ):
        assert original in paths, f"original trigger path dropped: {original}"


def test_scheduled_backstop_present():
    """#618: a scheduled backstop guarantees drift in ANY shared module is run at
    least weekly, independent of which paths a PR touched."""
    workflow = cast(dict[str, Any], yaml.safe_load(_WORKFLOW_PATH.read_text()))
    on = workflow.get("on") if "on" in workflow else workflow.get(True)
    assert isinstance(on, dict)
    schedule = on.get("schedule")
    assert schedule, "Expected a schedule: backstop trigger (issue #618)."
    assert any("cron" in entry for entry in schedule)
