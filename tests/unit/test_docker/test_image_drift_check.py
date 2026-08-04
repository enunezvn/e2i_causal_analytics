"""#1479 — post-deploy image-drift check: pin drift must FAIL LOUDLY, never silently.

ROOT CAUSE (measured 2026-08-04): deploy.yml's ordered rollout recreates
feast/feast-materializer, the app tier (api/frontend/worker_light/worker_medium/
scheduler) and conditionally bentoml — but NEVER mlflow (or any other pinned
sidecar). Two mlflow pin bumps (#442 v3.11.1, #1477 v3.15.1) therefore never
reached the live server: ``e2i_mlflow_dev`` still ran ``ghcr.io/mlflow/mlflow:v3.1.0``
five weeks after the first bump, and nothing ever noticed.

The recurrence guard is ``scripts/deploy/check_image_drift.py``: after the rollout
converges, compare every compose-pinned service's RUNNING image (docker inspect
``.Config.Image`` via compose labels) against the compose-resolved pin
(``$COMPOSE_CMD config --format json``). Any mismatch not covered by an explicit,
dated, ticketed allowlist entry (``scripts/deploy/image_drift_allowlist.json`` —
mirrors the pip-audit ``--ignore-vuln`` carve-out idiom in security.yml) fails the
deploy run. Auto-recreating mlflow instead (issue option (a)) is deliberately NOT
done: first boot of v3.15.1 against the existing store runs a ONE-WAY sqlite
migration on volume ``e2i_mlflow_db``, so the recreate must stay a deliberate
quiet-window step with a backup — the check makes the drift loud in the meantime.

Test layers (mirrors the test_docker idiom):
  * pure-logic tests on ``evaluate()`` + allowlist parsing (fixture data, hermetic);
  * structural wiring guards on deploy.yml (the check step cannot be silently
    dropped later — #618-style);
  * a hermetic run of the VERBATIM deploy ``script:`` under PATH-shimmed stubs
    (the test_deploy_up_failure_rollback.py harness) proving the control flow:
    a drift-check failure fails the deploy WITHOUT triggering any rollback
    (the rollout converged; the check is an alarm, not a gate on the flip);
  * a live-docker mapping test that runs the REAL script against the REAL daemon
    + compose files (skips cleanly off-box / in CI where no e2i project runs).

FAITHFULNESS LIMIT: the hermetic layers prove logic + wiring + control flow, not
real docker/compose behavior. The live layer proves service->container mapping on
the real daemon. The decisive real-world red/green (mlflow v3.1.0 vs pin v3.15.1
detected with an empty allowlist; green with the #1479 entry; red again for a
synthetic unknown drift) was captured on the droplet — transcripts in PR.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy.yml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "deploy" / "check_image_drift.py"
ALLOWLIST_PATH = REPO_ROOT / "scripts" / "deploy" / "image_drift_allowlist.json"

# Import the module under test (scripts/ is not a package — tests/unit/test_scripts
# convention: put the dir on sys.path and import the module directly).
_SCRIPT_DIR = str(REPO_ROOT / "scripts" / "deploy")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import check_image_drift as cid  # noqa: E402,I001  (scripts/deploy on sys.path above)


# --------------------------------------------------------------------------- #
# Fixture data — shapes mirror the REAL live box (measured 2026-08-04)
# --------------------------------------------------------------------------- #
MLFLOW_RUNNING = "ghcr.io/mlflow/mlflow:v3.1.0"
MLFLOW_PINNED = "ghcr.io/mlflow/mlflow:v3.15.1"


def _mlflow_entry() -> dict:
    return {
        "service": "mlflow",
        "running": MLFLOW_RUNNING,
        "pinned": MLFLOW_PINNED,
        "issue": "#1479",
        "added": "2026-08-04",
        "reason": "known drift; deliberate quiet-window recreate pending",
    }


# --------------------------------------------------------------------------- #
# Pure logic — evaluate()
# --------------------------------------------------------------------------- #
def test_detects_drift_with_empty_allowlist() -> None:
    """The killer case: the REAL drift on the live box must fail the check."""
    report = cid.evaluate(
        pinned={"mlflow": MLFLOW_PINNED},
        running={"mlflow": [MLFLOW_RUNNING]},
        allowlist=[],
    )
    assert report.failed, "mlflow v3.1.0 vs pin v3.15.1 must FAIL with an empty allowlist"
    assert ("mlflow", MLFLOW_RUNNING, MLFLOW_PINNED) in [
        (d.service, d.running, d.pinned) for d in report.drift
    ]


def test_matching_running_and_pinned_is_ok() -> None:
    report = cid.evaluate(
        pinned={"redis": "redis:7-alpine"},
        running={"redis": ["redis:7-alpine"]},
        allowlist=[],
    )
    assert not report.failed
    assert "redis" in [o.service for o in report.ok]


def test_exact_allowlist_entry_suppresses_known_drift() -> None:
    """The #1479 idiom: the KNOWN, ticketed drift keeps deploys green."""
    report = cid.evaluate(
        pinned={"mlflow": MLFLOW_PINNED},
        running={"mlflow": [MLFLOW_RUNNING]},
        allowlist=[_mlflow_entry()],
    )
    assert not report.failed
    assert ("mlflow", MLFLOW_RUNNING, MLFLOW_PINNED) in [
        (a.service, a.running, a.pinned) for a in report.allowlisted
    ]


def test_allowlist_stops_matching_after_further_pin_bump() -> None:
    """The load-bearing property: an entry pins the EXACT (service, running,
    pinned) triple. A FURTHER mlflow bump (pin moves past v3.15.1) must fail —
    otherwise the allowlist would silently re-institutionalize the drift."""
    report = cid.evaluate(
        pinned={"mlflow": "ghcr.io/mlflow/mlflow:v3.16.0"},
        running={"mlflow": [MLFLOW_RUNNING]},
        allowlist=[_mlflow_entry()],
    )
    assert report.failed, "a NEW pin past the allowlisted one must fail the check"


def test_allowlist_does_not_cover_other_services() -> None:
    """An entry never bleeds across services, even with identical image strings."""
    entry = _mlflow_entry()
    report = cid.evaluate(
        pinned={"grafana": MLFLOW_PINNED},
        running={"grafana": [MLFLOW_RUNNING]},
        allowlist=[entry],
    )
    assert report.failed


def test_not_running_service_is_info_not_failure() -> None:
    """Measured on the live box: prometheus/grafana/loki/worker_heavy are compose-
    defined but intentionally not running. Liveness is the health gates' job;
    failing on not-running would false-alarm on day one."""
    report = cid.evaluate(
        pinned={"prometheus": "prom/prometheus:v3.2.1"},
        running={},
        allowlist=[],
    )
    assert not report.failed
    assert "prometheus" in [n.service for n in report.not_running]


def test_build_only_service_is_skipped() -> None:
    """bentoml/feast/feast-materializer have build: but no image: pin — there is
    no pin to drift from, so they are reported as skipped, never compared."""
    report = cid.evaluate(
        pinned={"bentoml": None},
        running={"bentoml": ["e2i-causal-analytics-bentoml"]},
        allowlist=[],
    )
    assert not report.failed
    assert "bentoml" in [s.service for s in report.skipped]


def test_multi_replica_service_checks_every_container() -> None:
    """worker_light runs 2 replicas on the live box: ONE stale replica (a failed
    recreate) must fail even when the other matches."""
    report = cid.evaluate(
        pinned={"worker_light": "ghcr.io/enunezvn/e2i-api:newsha"},
        running={
            "worker_light": ["ghcr.io/enunezvn/e2i-api:newsha", "ghcr.io/enunezvn/e2i-api:oldsha"]
        },
        allowlist=[],
    )
    assert report.failed
    assert (
        "worker_light",
        "ghcr.io/enunezvn/e2i-api:oldsha",
        "ghcr.io/enunezvn/e2i-api:newsha",
    ) in [(d.service, d.running, d.pinned) for d in report.drift]


def test_stale_allowlist_entry_warns_but_does_not_fail() -> None:
    """After the deliberate recreate lands, the running image matches the pin and
    the #1479 entry goes stale. The recreate happens on the box; removing the
    entry is a PR — hard-failing on staleness would gate deploys on that timing,
    so staleness is a WARN with a removal instruction, not a failure."""
    report = cid.evaluate(
        pinned={"mlflow": MLFLOW_PINNED},
        running={"mlflow": [MLFLOW_PINNED]},
        allowlist=[_mlflow_entry()],
    )
    assert not report.failed
    assert len(report.stale_entries) == 1


# --------------------------------------------------------------------------- #
# Allowlist parsing — fail closed
# --------------------------------------------------------------------------- #
def test_malformed_allowlist_entry_fails_closed() -> None:
    """An entry missing its required provenance (issue/added/reason) must raise —
    a broken allowlist must never silently allow everything."""
    bad = _mlflow_entry()
    del bad["issue"]
    with pytest.raises(cid.AllowlistError):
        cid.parse_allowlist(json.dumps([bad]))


def test_allowlist_must_be_a_json_list() -> None:
    with pytest.raises(cid.AllowlistError):
        cid.parse_allowlist(json.dumps({"service": "mlflow"}))


def test_shipped_allowlist_file_is_valid_and_ticketed() -> None:
    """The committed allowlist must always parse and every entry must carry a
    real issue reference + ISO date (the pip-audit carve-out discipline)."""
    entries = cid.parse_allowlist(ALLOWLIST_PATH.read_text())
    for e in entries:
        assert re.fullmatch(r"#\d+", e["issue"]), f"issue ref must be '#<n>': {e}"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", e["added"]), f"added must be ISO date: {e}"
        assert len(e["reason"]) >= 20, f"reason must actually explain the hold: {e}"


# --------------------------------------------------------------------------- #
# Structural wiring guards on deploy.yml (#618-style: the step cannot be
# silently dropped later)
# --------------------------------------------------------------------------- #
def _load_workflow() -> dict:
    wf: dict = yaml.safe_load(DEPLOY_WORKFLOW.read_text())
    return wf


def _extract_deploy_script() -> str:
    wf = _load_workflow()
    for step in wf["jobs"]["deploy"]["steps"]:
        with_ = step.get("with") or {}
        if "script" in with_:
            return str(with_["script"])
    raise AssertionError("deploy.yml has no ssh-action step carrying a `script:`")


def test_deploy_script_invokes_drift_check_after_rollout() -> None:
    """deploy.yml must run the drift check AFTER the rollout + health gates
    (the check verifies the CONVERGED state) and pass it the deploy's own
    $COMPOSE_CMD so both sides resolve the same compose file set."""
    script = _extract_deploy_script()
    assert "scripts/deploy/check_image_drift.py" in script, (
        "deploy.yml no longer invokes the #1479 image-drift check"
    )
    invoke_at = script.index("scripts/deploy/check_image_drift.py")
    health_at = script.index("Waiting for health check")
    assert invoke_at > health_at, "drift check must run after the app health gate"
    invocation = next(
        line
        for line in script.splitlines()
        if "check_image_drift.py" in line and not line.lstrip().startswith("#")
    )
    assert '"$COMPOSE_CMD"' in invocation, (
        "the check must receive the deploy's own $COMPOSE_CMD (same overlay set): " + invocation
    )


def test_trigger_paths_cover_drift_check_inputs() -> None:
    """A change to the check script/allowlist must itself trigger a deploy
    (mirrors the scripts/bentoml/** precedent): the script is consumed at
    deploy time from the droplet checkout, not baked into an image."""
    wf = _load_workflow()
    # PyYAML parses the bare `on:` key as the boolean True.
    trigger = wf.get("on") or wf.get(True)
    assert trigger is not None, "deploy.yml has no trigger block"
    push = trigger["push"]
    assert "scripts/deploy/**" in list(push["paths"]), (
        "deploy.yml does not trigger on scripts/deploy/** — an allowlist edit "
        "(e.g. removing the #1479 entry after the recreate) would not deploy"
    )


def test_script_and_allowlist_files_exist() -> None:
    assert SCRIPT_PATH.is_file(), "scripts/deploy/check_image_drift.py missing"
    assert ALLOWLIST_PATH.is_file(), "scripts/deploy/image_drift_allowlist.json missing"


# --------------------------------------------------------------------------- #
# Hermetic control-flow harness (mirrors test_deploy_up_failure_rollback.py):
# a drift-check failure must fail the deploy LOUDLY but WITHOUT any rollback —
# the rollout converged; recreating anything would not fix a pin mismatch.
# --------------------------------------------------------------------------- #
PREV_SHA = "aaaaaaa000prev"
NEW_SHA = "bbbbbbb111new0"

_GIT_STUB = r"""#!/usr/bin/env bash
echo "git $*" >> "$CALL_LOG"
if [ "$1" = "rev-parse" ] && [ "$2" = "HEAD" ]; then
  c="$STUB_STATE/rp"
  n=$(cat "$c" 2>/dev/null || echo 0); n=$((n + 1)); printf '%s' "$n" > "$c"
  if [ "$n" -le 1 ]; then printf '%s\n' "$PREV_SHA"; else printf '%s\n' "$NEW_SHA"; fi
fi
exit 0
"""

_DOCKER_STUB = r"""#!/usr/bin/env bash
echo "docker $*" >> "$CALL_LOG"
case "$*" in
  *materializer_heartbeat*) printf '%s\n' "${HEARTBEAT:-1001}"; exit 0 ;;
esac
exit 0
"""

_CURL_STUB = '#!/usr/bin/env bash\necho "curl $*" >> "$CALL_LOG"\nexit 0\n'
_DATE_STUB = "#!/usr/bin/env bash\nprintf '%s\\n' 1000\n"
_SEQ_STUB = "#!/usr/bin/env bash\nprintf '%s\\n' 1\n"
_SLEEP_STUB = "#!/usr/bin/env bash\nexit 0\n"

# The drift-check stand-in: logs its invocation, honours DRIFT_EXIT so a test can
# force the drift-detected path without real docker/compose.
_DRIFT_STUB = r"""#!/usr/bin/env python3
import os, sys
with open(os.environ["CALL_LOG"], "a") as f:
    f.write("drift-check " + " ".join(sys.argv[1:]) + "\n")
code = int(os.environ.get("DRIFT_EXIT", "0"))
if code:
    print("IMAGE DRIFT CHECK: FAILED (stub)")
sys.exit(code)
"""

_STUBS = {
    "git": _GIT_STUB,
    "docker": _DOCKER_STUB,
    "curl": _CURL_STUB,
    "date": _DATE_STUB,
    "seq": _SEQ_STUB,
    "sleep": _SLEEP_STUB,
}


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    project_dir = tmp_path / "repo"
    (project_dir / "docker" / "frontend").mkdir(parents=True)
    # pick_overlay() greps this for `AS production` -> returns "" (base prod).
    (project_dir / "docker" / "frontend" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS production\n"
    )
    (project_dir / "scripts" / "deploy").mkdir(parents=True, exist_ok=True)
    (project_dir / "scripts" / "run_migrations.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    drift_stub = project_dir / "scripts" / "deploy" / "check_image_drift.py"
    drift_stub.write_text(_DRIFT_STUB)
    drift_stub.chmod(0o755)

    script = _extract_deploy_script()
    assert "${{" not in script
    script, n = re.subn(r'PROJECT_DIR="[^"]*"', f'PROJECT_DIR="{project_dir}"', script, count=1)
    assert n == 1

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
        ["bash", str(script_file)], env=env, capture_output=True, text=True, timeout=60
    )
    return proc.returncode, proc.stdout + proc.stderr, call_log.read_text().splitlines()


def test_healthy_deploy_runs_drift_check(tmp_path: Path) -> None:
    """RED before the fix: nothing in deploy.yml ever invokes the check."""
    code, out, calls = _run(tmp_path)
    assert any(c.startswith("drift-check") for c in calls), (
        "a converged deploy must invoke the image-drift check. Calls:\n" + "\n".join(calls)
    )
    assert code == 0, f"a clean deploy + clean drift check must exit 0; got {code}\n{out}"


def test_drift_failure_fails_deploy_without_rollback(tmp_path: Path) -> None:
    """Drift found -> deploy run FAILS (loud) but NO rollback fires: every service
    was already recreated + health-gated; a pin mismatch on a never-recreated
    sidecar is fixed by a deliberate operator recreate, not by rolling back code."""
    code, out, calls = _run(tmp_path, DRIFT_EXIT="1")
    assert any(c.startswith("drift-check") for c in calls)
    assert code != 0, "an unallowlisted drift must fail the deploy run"
    assert not any(c.startswith("git reset") and PREV_SHA in c for c in calls), (
        "a drift-check failure must NOT trigger a rollback:\n" + "\n".join(calls)
    )


# --------------------------------------------------------------------------- #
# Live layer — REAL docker daemon + REAL compose files (skips off-box)
# --------------------------------------------------------------------------- #
def _live_project_present() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "label=com.docker.compose.project=e2i-causal-analytics",
                "--format",
                "{{.ID}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


@pytest.mark.skipif(
    not _live_project_present(),
    reason="no live e2i-causal-analytics compose project on this docker daemon",
)
def test_live_service_to_container_mapping_end_to_end() -> None:
    """On the real box: the script must resolve compose pins from the REAL compose
    file and map services to REAL running containers via compose labels (the live
    mlflow container is named e2i_mlflow_dev, NOT the base compose's e2i_mlflow —
    name-based mapping would silently miss exactly the drift #1479 is about).

    Asserts the MECHANICS (mapping + report shape), not the box's current drift
    state, so it stays green after the deliberate mlflow recreate lands."""
    env = dict(os.environ)
    # Compose interpolation: required vars may be absent in a worktree (.env is
    # not checked out). Dummy values are fine — image refs don't depend on them.
    for var in (
        "FALKORDB_PASSWORD",
        "GRAFANA_ADMIN_PASSWORD",
        "REDIS_PASSWORD",
        "SUPABASE_POSTGRES_PASSWORD",
    ):
        env.setdefault(var, "drift-check-dummy")
    env.setdefault("IMAGE_OWNER", "enunezvn")
    env.setdefault("IMAGE_TAG", "latest")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--compose-cmd",
            f"docker compose -f {REPO_ROOT / 'docker' / 'docker-compose.yml'}",
            "--allowlist",
            str(ALLOWLIST_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode in (0, 1), (
        f"infra error (exit {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
    )
    mlflow_lines = [line for line in proc.stdout.splitlines() if re.search(r"\bmlflow\b", line)]
    assert mlflow_lines, "the report must carry a line for the mlflow service:\n" + proc.stdout
    # The mapping worked iff mlflow resolved to a RUNNING image ref (any state:
    # OK, DRIFT or ALLOWED — but never NOTRUN, which would mean label mapping broke).
    assert not any(line.startswith("NOTRUN") for line in mlflow_lines), (
        "mlflow has a live container — NOTRUN means service->container label "
        "mapping is broken:\n" + proc.stdout
    )
