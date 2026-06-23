"""T5 — deploy.yml must refresh the BentoML serving container so its schema never
drifts from the API client.

ROOT CAUSE (verified from the repo): the cohort scorer posts ``raw_features`` to
``bentoml:3000/predict_batch`` (src/api/routes/predictions.py), but the running
``bentoml`` container serves whatever schema it was last BUILT/started with. The
serving entrypoint is ``scripts/bentoml/e2i_serving_service.py`` — and that path is
NOT in deploy.yml's ``on.push.paths`` (which lists ``src/**`` but not ``scripts/``),
so a serving-schema change never even triggers a deploy. Worse, even when a deploy
DOES fire, deploy.yml force-recreates feast + the app tier but NEVER the ``bentoml``
service, so the live serving schema silently drifts every deploy → the #predict 400.

This test pins the durable fix WITHOUT a live droplet, mirroring
``test_deploy_up_failure_rollback.py``:
  * a structural check that the trigger ``paths:`` cover the serving inputs, and
  * a hermetic run of the VERBATIM deploy ``script:`` under PATH-shimmed stubs for
    git/docker/curl/date/seq/sleep, asserting OBSERVABLE effects — the ``bentoml``
    force-recreate fires only when serving inputs changed, the ``/healthz`` gate +
    ``rollback_to_prev bentoml`` run on failure, and the exit code — not inline text.

FAITHFULNESS LIMIT (cheapest-disproof-in-a-faithful-env discipline): this proves the
shipped script's CONTROL FLOW under stubs (which service is recreated, the gate, the
rollback, the exit code). It does NOT prove real docker/compose behavior or that the
recreated BentoML actually accepts ``raw_features`` — the only fully faithful check is
a real deploy against the off-limits live droplet. This is the cheapest faithful-to-
control-flow disproof, decisive for a missing-refresh / missing-rollback defect (which
IS a control-flow defect).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"

PREV_SHA = "aaaaaaa000prev"
NEW_SHA = "bbbbbbb111new0"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _load_workflow() -> dict:
    return yaml.safe_load(DEPLOY_WORKFLOW.read_text())


def _trigger_paths() -> list[str]:
    wf = _load_workflow()
    # PyYAML parses the bare `on:` key as the boolean True.
    push = (wf.get("on") or wf.get(True))["push"]
    return list(push["paths"])


def _extract_deploy_script() -> str:
    wf = _load_workflow()
    for step in wf["jobs"]["deploy"]["steps"]:
        with_ = step.get("with") or {}
        if "script" in with_:
            return with_["script"]
    raise AssertionError("deploy.yml has no ssh-action step carrying a `script:`")


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
# `git diff --name-only PREV NEW` prints $DIFF_FILES so the rebuild + the bentoml
# change-detectors see a real changed-file list. rev-parse HEAD returns PREV then NEW.
_GIT_STUB = r"""#!/usr/bin/env bash
echo "git $*" >> "$CALL_LOG"
if [ "$1" = "rev-parse" ] && [ "$2" = "HEAD" ]; then
  c="$STUB_STATE/rp"
  n=$(cat "$c" 2>/dev/null || echo 0); n=$((n + 1)); printf '%s' "$n" > "$c"
  if [ "$n" -le 1 ]; then printf '%s\n' "$PREV_SHA"; else printf '%s\n' "$NEW_SHA"; fi
fi
if [ "$1" = "diff" ]; then
  printf '%s\n' "${DIFF_FILES:-scripts/bentoml/e2i_serving_service.py}"
fi
# status (clean) / fetch / reset / checkout all exit 0 silently.
exit 0
"""

# docker dispatch:
#   * materializer heartbeat -> fresh (>= GATE_START) so flow reaches app + bentoml;
#   * a forward `up -d` carrying ` bentoml` is the bentoml refresh (toggle FAIL_BENTOML_UP);
#   * a forward `up -d` carrying ` api`     is the app flip;
#   * any `up -d` AFTER a `git checkout` is a ROLLBACK up.
_DOCKER_STUB = r"""#!/usr/bin/env bash
echo "docker $*" >> "$CALL_LOG"
args="$*"
case "$args" in
  *materializer_heartbeat*) printf '%s\n' "${HEARTBEAT:-1001}"; exit 0 ;;
esac
if printf '%s' "$args" | grep -q -- 'up -d'; then
  if grep -q 'git checkout' "$CALL_LOG"; then
    exit 0
  fi
  case "$args" in
    *' bentoml'*) [ "${FAIL_BENTOML_UP:-0}" = "1" ] && exit 1; exit 0 ;;
    *) exit 0 ;;
  esac
fi
exit 0
"""

# Port-aware: bentoml health is :3000, the app health is :8000. Failing one must
# not fail the other (we must still REACH the bentoml step with a healthy app).
_CURL_STUB = r"""#!/usr/bin/env bash
echo "curl $*" >> "$CALL_LOG"
args="$*"
case "$args" in
  *localhost:3000*) [ "${FAIL_BENTOML_HEALTH:-0}" = "1" ] && exit 1; exit 0 ;;
  *localhost:8000*) [ "${FAIL_HEALTH:-0}" = "1" ] && exit 1; exit 0 ;;
esac
exit 0
"""

_DATE_STUB = "#!/usr/bin/env bash\nprintf '%s\\n' 1000\n"
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
    project_dir = tmp_path / "repo"
    (project_dir / "docker" / "frontend").mkdir(parents=True)
    # pick_overlay() greps this for `AS production` -> returns "" (base prod).
    (project_dir / "docker" / "frontend" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS production\n"
    )
    (project_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (project_dir / "scripts" / "run_migrations.sh").write_text("#!/usr/bin/env bash\nexit 0\n")

    script = _extract_deploy_script()
    assert "${{" not in script, (
        "deploy `script:` gained GitHub-Actions interpolation; harness extraction is no longer faithful"
    )
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
# Assertion helpers
# --------------------------------------------------------------------------- #
def _checkout_prev_idx(calls: list[str]) -> int | None:
    for i, c in enumerate(calls):
        if c.startswith("git checkout") and PREV_SHA in c:
            return i
    return None


def _forward_bentoml_up_idx(calls: list[str]) -> int | None:
    """Index of a forward (pre-checkout) `up -d ... bentoml`."""
    for i, c in enumerate(calls):
        if c.startswith("git checkout"):
            return None
        if "up -d" in c and " bentoml" in c:
            return i
    return None


def _first_up_after(calls: list[str], idx: int) -> str | None:
    for c in calls[idx + 1 :]:
        if "up -d" in c:
            return c
    return None


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_trigger_paths_cover_bentoml_serving_inputs() -> None:
    """A serving-schema change must TRIGGER a deploy. The entrypoint lives under
    scripts/bentoml/ and the image under docker/bentoml/ — neither is covered by
    src/**, so both must be explicit trigger paths."""
    paths = _trigger_paths()
    assert "scripts/bentoml/**" in paths, (
        "deploy.yml does not trigger on scripts/bentoml/** — a serving-schema change "
        f"(e2i_serving_service.py) would not deploy. paths={paths}"
    )
    assert "docker/bentoml/**" in paths, (
        f"deploy.yml does not trigger on docker/bentoml/** (image inputs). paths={paths}"
    )


def test_recreates_bentoml_when_serving_inputs_change(tmp_path: Path) -> None:
    """Happy path: when PREV..NEW touched a bentoml serving input, the deploy must
    force-recreate the `bentoml` service (so the live schema matches the client) and
    succeed. RED on the unfixed deploy.yml: nothing ever recreates bentoml."""
    code, out, calls = _run(tmp_path, DIFF_FILES="scripts/bentoml/e2i_serving_service.py")
    fwd = _forward_bentoml_up_idx(calls)
    assert fwd is not None, (
        "expected a forward `up -d ... --force-recreate bentoml` when a serving input "
        "changed, but none ran. Calls:\n" + "\n".join(calls)
    )
    assert "force-recreate" in calls[fwd], (
        "bentoml must be force-recreated (not a no-op `up`):\n" + calls[fwd]
    )
    assert code == 0, f"a healthy bentoml refresh must exit 0; got {code}\n{out}"


def test_skips_bentoml_recreate_when_no_serving_change(tmp_path: Path) -> None:
    """Surgical: a deploy that did NOT touch bentoml inputs must NOT bounce the
    serving container (it has a 60s start_period; a needless restart is a per-deploy
    serving blip). Locks the change-gated conditional."""
    code, out, calls = _run(tmp_path, DIFF_FILES="frontend/src/App.tsx")
    assert _forward_bentoml_up_idx(calls) is None, (
        "bentoml was recreated on a deploy that did not touch any serving input:\n"
        + "\n".join(calls)
    )
    assert code == 0, f"a no-bentoml-change deploy must still succeed; got {code}\n{out}"


def test_bentoml_health_failure_rolls_back_bentoml_and_exits_1(tmp_path: Path) -> None:
    """If the recreated bentoml never reports healthy at /healthz, the deploy must
    roll bentoml back to PREV_SHA and FAIL loud (exit 1) — never leave a wedged
    serving container while reporting success."""
    code, out, calls = _run(
        tmp_path, DIFF_FILES="scripts/bentoml/e2i_serving_service.py", FAIL_BENTOML_HEALTH="1"
    )
    fwd = _forward_bentoml_up_idx(calls)
    assert fwd is not None, "the forward bentoml recreate must have run:\n" + "\n".join(calls)
    ci = _checkout_prev_idx(calls)
    assert ci is not None and ci > fwd, (
        "a failed bentoml health gate must roll back via `git checkout " + PREV_SHA + "` "
        "AFTER the forward recreate. Calls:\n" + "\n".join(calls)
    )
    rollback_up = _first_up_after(calls, ci)
    assert rollback_up is not None and " bentoml" in rollback_up, (
        "the rollback must force-recreate bentoml at PREV_SHA:\n" + str(rollback_up)
    )
    assert code == 1, f"a failed bentoml refresh must exit 1; got {code}\n{out}"


def test_bentoml_up_failure_rolls_back_bentoml_and_exits_1(tmp_path: Path) -> None:
    """If the recreate `up` itself fails (e.g. a bad serving build), same contract:
    roll bentoml back to PREV_SHA and exit 1."""
    code, out, calls = _run(
        tmp_path, DIFF_FILES="scripts/bentoml/e2i_serving_service.py", FAIL_BENTOML_UP="1"
    )
    ci = _checkout_prev_idx(calls)
    assert ci is not None, "a failed bentoml `up` must roll back to PREV_SHA. Calls:\n" + "\n".join(
        calls
    )
    rollback_up = _first_up_after(calls, ci)
    assert rollback_up is not None and " bentoml" in rollback_up, (
        "the rollback must recreate bentoml at PREV_SHA:\n" + str(rollback_up)
    )
    assert code == 1, f"a failed bentoml `up` must exit 1; got {code}\n{out}"


def test_deploy_script_still_round_trips_without_gha_interpolation() -> None:
    s = _extract_deploy_script()
    assert "${{" not in s
    assert "force-recreate" in s and "bentoml" in s
