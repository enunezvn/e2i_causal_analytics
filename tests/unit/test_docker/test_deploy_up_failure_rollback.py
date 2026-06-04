"""#563 — deploy.yml ordered-rollout rollback control-flow (hermetic; no docker/DB/deploy).

The prod rollout in ``.github/workflows/deploy.yml`` runs as an appleboy/ssh-action inline
``script:`` block on the droplet under ``set -e`` + ``script_stop: true``. It recreates
``feast`` + ``feast-materializer`` at NEW_SHA FIRST, gates on a fresh materialize, then flips
``api``/``frontend``/``worker_*``/``scheduler``. #563: if the app-services ``up`` itself exits
non-zero (e.g. the #528-B GHCR pull-fallback takes a local ``--build`` and that React build
OOMs), ``set -e`` aborts BETWEEN the app ``up`` and the health check — BEFORE the health-check
rollback — so ``feast`` is stranded at NEW_SHA with no automated rollback.

These tests extract the VERBATIM ``script:`` block (asserting it carries no ``${{ }}``
GitHub-Actions interpolation, so it round-trips), then execute it under ``bash`` with
PATH-shimmed stubs for ``git``/``docker``/``curl``/``date``/``seq``/``sleep``. We force specific
commands to fail and assert the OBSERVABLE rollback effects — which SHA we ``git checkout``,
which services get force-recreated afterward, the operator WARN on a double-fault, and the exit
code — NOT the inline text, so the test survives the ``rollback_to_prev()`` helper refactor.

FAITHFULNESS LIMIT (stated per the cheapest-disproof-in-a-faithful-env discipline): this proves
the shipped script's CONTROL FLOW (the ``set -e`` abort point, which rollback runs, the exit
code, the diagnostic emission) under stubs. It does NOT prove real docker/compose/feast
behavior, ssh-action buffering byte-fidelity, or a real OOM during the React build. The only
fully faithful check is a real deploy exercising the GHCR-pull-fallback build — the off-limits
live 16GB droplet. This is the cheapest faithful-to-control-flow disproof, and decisive for a
missing-rollback / non-guaranteed-exit defect (which IS a control-flow defect).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

# Distinct, recognizable sentinels so CALL_LOG lines are unambiguous.
PREV_SHA = "aaaaaaa000prev"
NEW_SHA = "bbbbbbb111new0"

# The full recreate set the app-up-failure rollback must restore to PREV_SHA.
SERVICES = {
    "feast",
    "feast-materializer",
    "api",
    "frontend",
    "worker_light",
    "worker_medium",
    "scheduler",
}
APP_SERVICES = {"api", "frontend", "worker_light", "worker_medium", "scheduler"}


# --------------------------------------------------------------------------- #
# Extraction (the SHIPPED artifact, verbatim)
# --------------------------------------------------------------------------- #
def _extract_deploy_script() -> str:
    wf = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    for step in wf["jobs"]["deploy"]["steps"]:
        with_ = step.get("with") or {}
        if "script" in with_:
            return with_["script"]
    raise AssertionError("deploy.yml has no ssh-action step carrying a `script:`")


# --------------------------------------------------------------------------- #
# Stubs — PATH-shadowing executables that log argv to $CALL_LOG and dispatch on it
# --------------------------------------------------------------------------- #
_GIT_STUB = r"""#!/usr/bin/env bash
echo "git $*" >> "$CALL_LOG"
if [ "$1" = "rev-parse" ] && [ "$2" = "HEAD" ]; then
  c="$STUB_STATE/rp"
  n=$(cat "$c" 2>/dev/null || echo 0); n=$((n + 1)); printf '%s' "$n" > "$c"
  if [ "$n" -le 1 ]; then printf '%s\n' "$PREV_SHA"; else printf '%s\n' "$NEW_SHA"; fi
fi
# status (clean tree) / fetch / reset / checkout / diff all exit 0 with no stdout.
exit 0
"""

# docker dispatch:
#   * the materializer heartbeat probe prints HEARTBEAT (default fresh) so the
#     freshness gate PASSES and flow REACHES the app-up (else we'd divert to the
#     materializer rollback and never exercise the #563 bug);
#   * any `up -d` AFTER a `git checkout` is a ROLLBACK up (toggle FAIL_ROLLBACK_UP);
#   * a forward `up -d` carrying app services is the app-up (toggle FAIL_APP_UP);
#   * a forward `up -d` without app services is the feast-up (toggle FAIL_FEAST_UP).
_DOCKER_STUB = r"""#!/usr/bin/env bash
echo "docker $*" >> "$CALL_LOG"
args="$*"
case "$args" in
  *materializer_heartbeat*) printf '%s\n' "${HEARTBEAT:-1001}"; exit 0 ;;
esac
if printf '%s' "$args" | grep -q -- 'up -d'; then
  if grep -q 'git checkout' "$CALL_LOG"; then
    [ "${FAIL_ROLLBACK_UP:-0}" = "1" ] && exit 1
    exit 0
  fi
  case "$args" in
    *' api'*) [ "${FAIL_APP_UP:-0}" = "1" ] && exit 1; exit 0 ;;
    *)        [ "${FAIL_FEAST_UP:-0}" = "1" ] && exit 1; exit 0 ;;
  esac
fi
exit 0
"""

_CURL_STUB = r"""#!/usr/bin/env bash
echo "curl $*" >> "$CALL_LOG"
[ "${FAIL_HEALTH:-0}" = "1" ] && exit 1
exit 0
"""

# Deterministic clock so GATE_START is fixed and the heartbeat stub can beat it.
_DATE_STUB = "#!/usr/bin/env bash\nprintf '%s\\n' 1000\n"
# One loop iteration, no real waiting.
_SEQ_STUB = "#!/usr/bin/env bash\nprintf '%s\\n' 1\n"
_SLEEP_STUB = "#!/usr/bin/env bash\nexit 0\n"

_STUBS = {
    "git": _GIT_STUB,
    "docker": _DOCKER_STUB,
    "curl": _CURL_STUB,
    "date": _DATE_STUB,
    "seq": _SEQ_STUB,
    "sleep": _SLEEP_STUB,
}


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a hermetic PROJECT_DIR + stub bin; return (script_file, stub_bin)."""
    project_dir = tmp_path / "repo"
    (project_dir / "docker" / "frontend").mkdir(parents=True)
    # pick_overlay() greps this for `AS production` -> returns "" (base prod, #528-B era).
    (project_dir / "docker" / "frontend" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS production\n"
    )
    # #672 added `bash scripts/run_migrations.sh` to the deploy `script:` (after the
    # `git reset --hard`, before the app-services flip). Stage a no-op stub so the
    # hermetic rollout reaches the rollback control-flow under test — under `set -e`
    # a missing file would abort before any app `up`/health step. This harness
    # validates the ROLLBACK contract, not migration execution, so success is the
    # faithful default (the real droplet always ships scripts/run_migrations.sh).
    (project_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (project_dir / "scripts" / "run_migrations.sh").write_text("#!/usr/bin/env bash\nexit 0\n")

    script = _extract_deploy_script()
    assert "${{" not in script, (
        "deploy `script:` gained GitHub-Actions interpolation; harness extraction is no longer faithful"
    )
    # Redirect ONLY the hardcoded PROJECT_DIR path to the temp tree (infrastructure
    # constant; control-flow text is left verbatim). Assert exactly one substitution.
    script, n = re.subn(r'PROJECT_DIR="[^"]*"', f'PROJECT_DIR="{project_dir}"', script, count=1)
    assert n == 1, "expected exactly one PROJECT_DIR assignment to redirect"

    script_file = project_dir / "_rollout.sh"
    script_file.write_text(script)

    stub_bin = tmp_path / "stubbin"
    stub_bin.mkdir()
    for name, body in _STUBS.items():
        p = stub_bin / name
        p.write_text(body)
        p.chmod(0o755)
    return script_file, stub_bin


def _run(tmp_path: Path, **toggles: str) -> tuple[int, str, list[str]]:
    script_file, stub_bin = _prepare(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    call_log = tmp_path / "calls.log"
    call_log.write_text("")

    env = {
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PREV_SHA": PREV_SHA,
        "NEW_SHA": NEW_SHA,
        "CALL_LOG": str(call_log),
        "STUB_STATE": str(state),
    }
    env.update(toggles)
    # Ensure the migration branch + GHCR pull path stay out of the way.
    for k in ("SUPABASE_DB_URL", "IMAGE_OWNER", "GHCR_USER", "GHCR_TOKEN", "REGISTRY"):
        env.pop(k, None)

    proc = subprocess.run(
        ["bash", str(script_file)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = call_log.read_text().splitlines()
    return proc.returncode, proc.stdout + proc.stderr, calls


# --------------------------------------------------------------------------- #
# Assertion helpers — observable effects, not inline text
# --------------------------------------------------------------------------- #
def _checkout_prev_idx(calls: list[str]) -> int | None:
    for i, c in enumerate(calls):
        if c.startswith("git checkout") and PREV_SHA in c:
            return i
    return None


def _first_up_after(calls: list[str], idx: int) -> str | None:
    for c in calls[idx + 1 :]:
        if "up -d" in c:
            return c
    return None


def _services_in(line: str | None) -> set[str]:
    if line is None:
        return set()
    return {tok for tok in line.split() if tok in SERVICES}


def _app_up_was_invoked(calls: list[str]) -> bool:
    """A forward (pre-checkout) `up -d` carrying app services."""
    for c in calls:
        if c.startswith("git checkout"):
            return False
        if "up -d" in c and " api" in c:
            return True
    return False


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_deploy_script_extracts_verbatim_with_no_gha_interpolation() -> None:
    """The harness can only be faithful if the block round-trips with no `${{ }}`."""
    s = _extract_deploy_script()
    assert "${{" not in s
    assert "set -e" in s and "COMPOSE_CMD=" in s
    assert "up -d" in s


def test_app_up_failure_rolls_back_feast_and_app_to_prev(tmp_path: Path) -> None:
    """#563 core: when the app-services `up` fails, the deploy must checkout PREV_SHA
    and recreate BOTH the feast tier and the app tier, then exit 1.

    RED on the unfixed deploy.yml: `set -e` aborts after the failed app `up` BEFORE the
    health-check rollback, so there is NO `git checkout PREV_SHA` and feast stays at
    NEW_SHA. We first assert it failed for the RIGHT reason, then assert the fix."""
    code, out, calls = _run(tmp_path, FAIL_APP_UP="1")

    # Failed for the right reason: the app `up` really ran and feast was already
    # force-recreated forward (stranded at NEW_SHA) before the failure.
    assert _app_up_was_invoked(calls), "app-services `up` was never reached:\n" + "\n".join(calls)

    ci = _checkout_prev_idx(calls)
    assert ci is not None, (
        "expected a rollback `git checkout " + PREV_SHA + "` after the app-up failure, "
        "but none ran (set -e aborted before any rollback). Calls:\n" + "\n".join(calls)
    )
    rollback_up = _first_up_after(calls, ci)
    assert _services_in(rollback_up) == SERVICES, (
        "the app-up-failure rollback must force-recreate feast+materializer+ALL app "
        f"services at PREV_SHA; got {_services_in(rollback_up)} from: {rollback_up}"
    )
    assert code == 1, f"deploy must exit 1 on app-up failure; got {code}"


def test_double_fault_app_up_and_rollback_up_emits_partial_warning_and_exits_1(
    tmp_path: Path,
) -> None:
    """The load-bearing guarantee: if the rollback `up` ALSO fails (the same OOM-prone
    build, now on an even more pressured box), `set -e` must NOT swallow the failure
    silently. A loud PARTIAL-state diagnostic must print AND the deploy must still
    reach exit 1.

    RED on unfixed code (no rollback at all) AND on a naive Option A without `|| echo`
    guards (set -e aborts before the warning + exit). GREEN only with best-effort guards."""
    code, out, calls = _run(tmp_path, FAIL_APP_UP="1", FAIL_ROLLBACK_UP="1")
    assert "PARTIAL" in out, (
        "a double-fault (rollback `up` also fails) must emit a loud operator WARN about "
        "PARTIAL state needing manual intervention; stdout/stderr was:\n" + out
    )
    assert code == 1, f"deploy must still exit 1 even when the rollback itself fails; got {code}"


def test_materializer_gate_failure_still_rolls_back_feast_only(tmp_path: Path) -> None:
    """Regression-lock (guards the helper refactor): a stale/absent materialize must
    roll feast+materializer back to PREV_SHA and exit 1 — and must NOT have flipped the
    app services. Passes on both the unfixed and the fixed deploy.yml."""
    code, out, calls = _run(tmp_path, HEARTBEAT="0")
    ci = _checkout_prev_idx(calls)
    assert ci is not None, "materializer-gate failure must roll back to PREV_SHA"
    assert {"feast", "feast-materializer"} <= _services_in(_first_up_after(calls, ci))
    assert not _app_up_was_invoked(calls), (
        "app services must not flip when the materializer gate fails"
    )
    assert code == 1


def test_health_check_failure_rolls_back_app_services(tmp_path: Path) -> None:
    """Regression-lock (guards the helper refactor): when the app `up` succeeds but the
    health check never passes, the existing health-path rollback must recreate the app
    services at PREV_SHA and exit 1. Passes on both the unfixed and the fixed deploy.yml."""
    code, out, calls = _run(tmp_path, FAIL_HEALTH="1")
    ci = _checkout_prev_idx(calls)
    assert ci is not None, "health-check failure must roll back to PREV_SHA"
    assert APP_SERVICES <= _services_in(_first_up_after(calls, ci))
    assert code == 1
