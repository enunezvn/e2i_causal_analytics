"""Codify hard-fail mode for the Tier 1-5 agent harness workflow.

Tracker: GitHub issue #263 (filed 2026-05-15 from PR #247). The workflow
at ``.github/workflows/tier1-5-test.yml`` was added in soft-fail mode
with ``continue-on-error: true`` on (a) the job, (b) the boot-stack
step, and (c) the install-deps step. The flip-off was gated on PR #275
(dispatcher wrapped-input fix #260), which merged 2026-05-16 at
``ac68ae41``.

This test is the forcing function for the flip: any future PR that
re-introduces soft-fail markers in the agent harness job will trip
these assertions. The skip-cache notice path (handled via ``if:``
gating on ``steps.restore-cache.outputs.found``) is intentionally
preserved — it's a legitimate skip reason, not a soft-fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "tier1-5-test.yml"


def _load_workflow() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_WORKFLOW_PATH.read_text()))


def _harness_job() -> dict[str, Any]:
    workflow = _load_workflow()
    jobs = workflow.get("jobs", {}) or {}
    assert "tier1-5-harness" in jobs, f"tier1-5-harness job missing. Found jobs: {sorted(jobs)}"
    return cast(dict[str, Any], jobs["tier1-5-harness"])


def test_harness_job_is_not_soft_fail():
    """Job-level ``continue-on-error: true`` must be removed.

    Issue #263 acceptance item 1: "Remove ``continue-on-error: true``
    from the ``tier1-5-harness`` job (line 42)."

    A failing harness must trip the overall PR check — that's the
    whole point of the harness as a forcing function against schema
    drift between tier0 state, the Tier0OutputMapper, and per-agent
    contracts.
    """
    job = _harness_job()
    coe = job.get("continue-on-error")
    assert coe is not True, (
        "tier1-5-harness job has continue-on-error: true (soft-fail). "
        "Issue #263 requires hard-fail since PR #275 landed the "
        "dispatcher wrapped-input fix."
    )


def test_boot_stack_step_is_not_soft_fail():
    """The ``Boot Redis + FalkorDB + MLflow`` step must not soft-fail.

    Issue #263 acceptance item 2: "Remove ``continue-on-error: true``
    from the boot-stack step (line 76)."

    A compose-config regression must surface as a check failure rather
    than be papered over by the step's outcome being marked success.
    """
    job = _harness_job()
    steps = job.get("steps", []) or []
    boot_steps = [s for s in steps if s.get("id") == "boot-stack"]
    assert len(boot_steps) == 1, (
        f"Expected exactly one step with id=boot-stack; found {len(boot_steps)}."
    )
    coe = boot_steps[0].get("continue-on-error")
    assert coe is not True, (
        "boot-stack step has continue-on-error: true. Issue #263 "
        "requires the step to hard-fail; downstream gating on "
        "steps.boot-stack.outputs.ok still allows graceful skip."
    )


def test_install_deps_step_is_not_soft_fail():
    """The ``Install runtime + dev dependencies`` step must not soft-fail.

    Issue #263 acceptance item 3: "Remove ``continue-on-error: true``
    from the install-deps step (line 122)."

    A pip resolver / build regression must trip the check; the harness
    cannot meaningfully run on a broken dep set.
    """
    job = _harness_job()
    steps = job.get("steps", []) or []
    install_steps = [s for s in steps if s.get("id") == "install-deps"]
    assert len(install_steps) == 1, (
        f"Expected exactly one step with id=install-deps; found {len(install_steps)}."
    )
    coe = install_steps[0].get("continue-on-error")
    assert coe is not True, (
        "install-deps step has continue-on-error: true. Issue #263 requires the step to hard-fail."
    )


def test_no_step_in_harness_job_has_continue_on_error_true():
    """Belt-and-braces: no step ANYWHERE in the harness job may use
    ``continue-on-error: true``.

    Covers the case where a future contributor adds a new soft-fail
    step (e.g., to silence a flake) without touching the three steps
    enumerated in the issue. The whole job must trip on real failures.
    """
    job = _harness_job()
    steps = job.get("steps", []) or []
    offenders = [
        s.get("name") or s.get("id") or f"step-{i}"
        for i, s in enumerate(steps)
        if s.get("continue-on-error") is True
    ]
    assert not offenders, (
        f"Steps with continue-on-error: true in tier1-5-harness job: "
        f"{offenders}. Issue #263 requires hard-fail across all steps; "
        "use ``if:`` gating with explicit ``ok`` / ``outputs`` checks "
        "for legitimate graceful-skip paths instead."
    )


def test_boot_stack_does_not_use_set_plus_e_for_compose_up():
    """The boot-stack step must not blanket-disable ``set -e`` via
    ``set +e``. Issue #263 acceptance: only the missing-tier0-cache
    skip is a legitimate graceful-skip — real compose regressions
    (parse errors, missing services, network conflicts, ``docker
    compose up`` non-zero) must trip the job.

    Surfaced by codex gate-on-diff iter-1 (HIGH-2, 2026-05-16). The
    boot-stack step retains a structured health-probe timeout
    graceful-skip via ``outputs.ok=false`` (image-pull rate limits
    can flap), but compose-up and network-create are now hard-fail.
    The script body must use ``set -e``-compatible defaults so a
    real infra regression escapes via shell exit code.
    """
    job = _harness_job()
    steps = job.get("steps", []) or []
    boot_steps = [s for s in steps if s.get("id") == "boot-stack"]
    assert len(boot_steps) == 1
    run = boot_steps[0].get("run") or ""
    # ``set +e`` anywhere in the script disables error propagation
    # globally. The new script uses ``set -euo pipefail`` instead.
    offenders = [
        line.strip()
        for line in run.splitlines()
        if line.strip() == "set +e"
        or line.strip().startswith("set +e ")
        or line.strip().startswith("set +e\t")
    ]
    assert not offenders, (
        f"boot-stack step disables shell error propagation via "
        f"`set +e` (offending lines: {offenders}). Issue #263 "
        "requires real compose / network regressions to hard-fail; "
        "use `set -euo pipefail` and a narrower graceful-skip "
        "(e.g., ALL_HEALTHY=false → outputs.ok=false) for "
        "health-probe timeout only."
    )


def test_install_deps_step_has_no_pipe_or_true_soft_fail():
    """The install-deps step must not mask pip failures via ``|| true``.

    The boot-stack step uses ``set -euo pipefail`` + an
    ``ALL_HEALTHY=false → outputs.ok=false`` health-probe-timeout
    graceful-skip; inside that flow ``docker logs ... || true`` on the
    diagnostic-only path is fine (the script is already past the
    healthy-check decision). install-deps has no such structured
    fallback: any pipe-masking there would silently swallow a real
    pip resolver / build failure on a hard-fail PR.
    """
    job = _harness_job()
    steps = job.get("steps", []) or []
    install_steps = [s for s in steps if s.get("id") == "install-deps"]
    assert len(install_steps) == 1
    run = install_steps[0].get("run") or ""
    offenders = [
        line.strip()
        for line in run.splitlines()
        if line.strip().endswith("|| true") or line.strip().endswith("; true")
    ]
    assert not offenders, (
        f"install-deps step uses shell soft-fail markers: {offenders}. "
        "Issue #263 requires hard-fail."
    )


def test_harness_main_propagates_nonzero_exit_on_agent_failure():
    """The Tier 1-5 harness Python entrypoint
    (``scripts/run_tier1_5_test.py``) must call ``sys.exit(<non-zero>)``
    when any agent fails — otherwise the workflow's ``continue-on-error``
    flip is toothless because ``make tier1-5-test`` will exit 0 even
    while ``summary["failed"] > 0``.

    Surfaced by codex gate-on-diff iter-1 (HIGH-1, 2026-05-16). We
    check the source rather than execute the harness (which requires
    a tier0 cache + docker stack); a structural assert is enough to
    guarantee the failure-exit path exists at all and pin it against
    regressions.
    """
    import ast

    harness = Path(__file__).resolve().parents[2] / "scripts" / "run_tier1_5_test.py"
    tree = ast.parse(harness.read_text())

    # Find the top-level ``main`` function definition.
    main_fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(main_fns) == 1, (
        f"Expected exactly one top-level main() in {harness.name}; found {len(main_fns)}."
    )
    main_src = ast.unparse(main_fns[0])

    # The main function must reach sys.exit at all.
    assert "sys.exit" in main_src, (
        "scripts/run_tier1_5_test.py main() does not call sys.exit. "
        "Issue #263 requires the harness to propagate a non-zero exit "
        "when any agent fails so the workflow check actually fails."
    )

    # The exit decision must be gated on the failed count — exiting
    # unconditionally would always-fail every CI run. We accept any
    # predicate referencing ``failed`` (e.g., ``failed_count > 0``,
    # ``summary['failed']``, ``failed > 0``).
    assert "failed" in main_src.lower(), (
        "scripts/run_tier1_5_test.py main() reaches sys.exit but the "
        "decision is not gated on a 'failed' count. Issue #263 requires "
        "exit code to reflect agent-pass/fail status."
    )


def test_workflow_comment_no_longer_advertises_soft_fail():
    """Issue #263 acceptance item 5: "Update the comment at line 35 to
    remove 'Soft-fail initially' wording."

    Future contributors reading the workflow must not see stale text
    that says soft-fail is the current state, since it isn't.
    """
    text = _WORKFLOW_PATH.read_text()
    assert "Soft-fail initially" not in text, (
        "Workflow YAML still contains stale 'Soft-fail initially' "
        "comment text from the pre-flip era. Issue #263 acceptance "
        "item 5 requires removing it."
    )


def test_skip_cache_notice_path_preserved():
    """Issue #263 acceptance item 4: "Keep the skip-cache notice path
    (steps.restore-cache.outputs.found != 'true') — that's still a
    legitimate skip reason."

    Without a committed tier0 cache or runner-side restore, the harness
    cannot run, and that scenario is a graceful skip — not a failure.
    Encoded as ``if:`` gating on the run step (positive: ``== 'true'``)
    + a notice in the summary step (negative: ``!= 'true'``), not as
    ``continue-on-error``. Targets stable step ``id``s rather than
    display names so a benign step rename doesn't trip the test.
    """
    job = _harness_job()
    steps = job.get("steps", []) or []

    run_steps = [s for s in steps if s.get("id") == "run-harness"]
    assert len(run_steps) == 1, (
        f"Expected exactly one step with id=run-harness; found {len(run_steps)}."
    )
    run_if = run_steps[0].get("if") or ""
    assert "steps.restore-cache.outputs.found == 'true'" in run_if, (
        f"Run step's if-guard must positively gate on cache present "
        f"(steps.restore-cache.outputs.found == 'true'); got: {run_if!r}. "
        "Issue #263 acceptance item 4 requires preserving this skip path."
    )

    summarize_steps = [s for s in steps if s.get("id") == "summarize-skip"]
    assert len(summarize_steps) == 1, (
        f"Expected exactly one step with id=summarize-skip; found {len(summarize_steps)}."
    )
    summarize_if = summarize_steps[0].get("if") or ""
    assert "steps.restore-cache.outputs.found != 'true'" in summarize_if, (
        f"Summarize-skip step's if-guard must negatively gate on cache "
        f"absent (steps.restore-cache.outputs.found != 'true'); got: "
        f"{summarize_if!r}. Issue #263 acceptance item 4 requires the "
        "skip notice to fire on this branch."
    )
