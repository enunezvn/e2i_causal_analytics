"""Optional (profile-gated) services must not be reported as outages.

Follow-up to #1798/#1805. `health_check.sh` probed 9 services that NO deploy step
starts: the 7 monitoring services and the 2 Opik endpoints. `SYSTEM STATUS` was
therefore permanently DEGRADED for structural reasons, which buries every real
signal inside it -- including the maintenance-freshness alarm #1798 exists to
raise. A permanently-firing alarm is the same defect as a silent one.

The fix declares intent in compose (`profiles: [monitoring]`, following the
`falkordb-browser`/`debug` precedent already in this file) and has health_check
DERIVE its skip-set from `docker compose config --services`, which lists only the
services a default `up` would start. No second list to drift, and enabling the
profile re-arms the probes automatically.

Deployments are unaffected by construction: every deploy `up` is `--no-deps` with
an explicit service list, and none of those lists names a monitoring service.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DEPLOY = REPO_ROOT / ".github" / "workflows" / "deploy.yml"
HEALTH_CHECK = REPO_ROOT / "scripts" / "health_check.sh"

MONITORING = [
    "prometheus",
    "grafana",
    "loki",
    "alertmanager",
    "promtail",
    "node-exporter",
    "postgres-exporter",
]


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=dict(os.environ), timeout=90
    )


def _extract_shell_function(script: str, name: str) -> str:
    lines = script.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(f"{name}() {{")), None)
    assert start is not None, f"{name}() not found"
    indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "}" and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            return "\n".join(ln[indent:] for ln in lines[start : j + 1])
    raise AssertionError(f"{name}() has no closing brace at its own indent")


def _service_blocks() -> dict[str, list[str]]:
    """Map service name -> its YAML lines, without a YAML parser.

    docker-compose.yml carries custom `!override` / `!reset` tags that trip the
    default loader, so this walks the two-space service headers instead.
    """
    lines = COMPOSE.read_text().splitlines()
    starts = [
        (i, m.group(1))
        for i, ln in enumerate(lines)
        if (m := re.match(r"^  ([A-Za-z0-9._-]+):\s*$", ln))
    ]
    blocks: dict[str, list[str]] = {}
    for idx, (i, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks[name] = lines[i:end]
    return blocks


def _profiles_of(block: list[str]) -> list[str]:
    out: list[str] = []
    for k, ln in enumerate(block):
        if re.match(r"^\s+profiles:", ln):
            for nxt in block[k + 1 :]:
                m = re.match(r"^\s+-\s*(\S+)\s*$", nxt)
                if not m:
                    break
                out.append(m.group(1).strip("\"'"))
            inline = re.match(r"^\s+profiles:\s*\[(.+)\]\s*$", ln)
            if inline:
                out += [p.strip().strip("\"'") for p in inline.group(1).split(",")]
    return out


# --------------------------------------------------------------------------- #
# Declaring the intent in compose
# --------------------------------------------------------------------------- #


def test_the_seven_monitoring_services_are_profile_gated() -> None:
    blocks = _service_blocks()
    missing = [s for s in MONITORING if "monitoring" not in _profiles_of(blocks.get(s, []))]
    assert not missing, (
        f"{missing} are started by a default `up` but no deploy step manages them, "
        "so health_check reports them as an outage forever"
    )


def test_no_unprofiled_service_depends_on_a_profiled_one() -> None:
    """A default-profile service depending on a gated one breaks `docker compose up`."""
    blocks = _service_blocks()
    gated = {name for name, b in blocks.items() if _profiles_of(b)}
    offenders: list[str] = []
    for name, block in blocks.items():
        if name in gated:
            continue
        in_dep = False
        for ln in block:
            if re.match(r"^\s+depends_on:", ln):
                in_dep = True
                continue
            if in_dep:
                m = re.match(r"^\s+-?\s*([A-Za-z0-9._-]+):?\s*$", ln)
                if m and m.group(1) in gated:
                    offenders.append(f"{name} -> {m.group(1)}")
                elif re.match(r"^\s{0,4}\S", ln) or re.match(r"^\s+[a-z_]+:\s*\S", ln):
                    in_dep = False
    assert not offenders, f"unprofiled service depends on a gated one: {offenders}"


def test_the_deploy_never_names_a_monitoring_service() -> None:
    """The 'without affecting deployments' guarantee, pinned.

    Naming a profiled service on a compose command line implicitly enables its
    profile -- so this is what actually keeps the gate effective.
    """
    text = DEPLOY.read_text()
    named = [s for s in MONITORING if re.search(rf"\b{re.escape(s)}\b", text)]
    assert not named, (
        f"deploy.yml names {named}; naming a profiled service on the command line "
        "enables its profile and would start it"
    )


def test_compose_excludes_monitoring_from_the_default_service_set() -> None:
    """Behavioural companion: the flag form matters.

    `config --services` honours profiles, but `--no-interpolate` bypasses the
    filter -- so this pins the exact invocation health_check.sh relies on.
    """
    if not shutil.which("docker"):
        import pytest

        pytest.skip("docker not available")
    res = _bash(f'docker compose -f "{COMPOSE}" config --services')
    if res.returncode != 0:
        import pytest

        pytest.skip(f"docker compose config unavailable: {res.stderr[:200]}")
    active = set(res.stdout.split())
    assert active, "positive control: the active set must not be empty"
    assert "api" in active, "positive control: a required service must still be listed"
    leaked = sorted(set(MONITORING) & active)
    assert not leaked, f"{leaked} still appear in the default service set"


# --------------------------------------------------------------------------- #
# health_check.sh behaviour
# --------------------------------------------------------------------------- #


def _run_probe(active: str, svc: str, url: str = "http://127.0.0.1:9/dead") -> str:
    src = HEALTH_CHECK.read_text()
    harness = textwrap.dedent(
        f"""
        GREEN=''; RED=''; YELLOW=''; NC=''
        HEALTHY=0; UNHEALTHY=0; SKIPPED=0
        ACTIVE_SERVICES=$'{active}'   # $'' so \n is a real newline, not literal
        {_extract_shell_function(src, "is_optional_service")}
        {_extract_shell_function(src, "check_http")}
        check_http "{url}" "Probe" 1 "{svc}" || true
        echo "COUNTS healthy=$HEALTHY unhealthy=$UNHEALTHY skipped=$SKIPPED"
        """
    )
    return _bash(harness).stdout


def test_an_absent_OPTIONAL_service_is_skipped_not_unhealthy() -> None:
    out = _run_probe(active="api\\nredis", svc="prometheus")
    assert "skipped=1" in out, out
    assert "unhealthy=0" in out, out


def test_an_absent_REQUIRED_service_is_still_unhealthy() -> None:
    """Positive control: a gate that skipped everything would pass the test above."""
    out = _run_probe(active="api\\nredis", svc="api")
    assert "unhealthy=1" in out, out
    assert "skipped=0" in out, out


def test_an_undeterminable_active_set_fails_toward_PROBING() -> None:
    """If we cannot read compose we must not silently skip -- that hides outages."""
    out = _run_probe(active="", svc="prometheus")
    assert "unhealthy=1" in out, out
    assert "skipped=0" in out, out
