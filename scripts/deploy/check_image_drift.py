#!/usr/bin/env python3
"""#1479 — post-deploy image-drift check: pinned image vs what actually runs.

WHY THIS EXISTS (measured 2026-08-04): deploy.yml's ordered rollout recreates
feast/feast-materializer, the app tier and (conditionally) bentoml — but never
the other compose-pinned sidecars. Two mlflow pin bumps (#442 -> v3.11.1,
#1477 -> v3.15.1) therefore never reached the live server: ``e2i_mlflow_dev``
still ran ``ghcr.io/mlflow/mlflow:v3.1.0`` five weeks later, silently. This
check runs at the END of the deploy script, after the rollout converged, and
FAILS THE RUN (exit 1) on any pin/running mismatch not covered by an explicit
allowlist entry.

Deliberately NOT the alternative fix (adding mlflow to the rollout): first boot
of v3.15.1 against the existing store runs a ONE-WAY sqlite schema migration on
volume ``e2i_mlflow_db`` — that recreate must stay a deliberate quiet-window
step with a backup first (issue #1479). Until it lands, the KNOWN drift is held
by a dated, ticketed entry in ``image_drift_allowlist.json`` (mirrors the
pip-audit ``--ignore-vuln`` carve-out idiom in security.yml). An entry matches
only its EXACT (service, running, pinned) triple, so any FURTHER pin bump — or
drift on any other service — fails loudly.

Scope and semantics:
  * Services are taken from the deploy's own compose resolution
    (``$COMPOSE_CMD config --format json``), so the check always compares
    against the file set the deploy actually used.
  * ALL compose profiles are included (``config --profiles`` is enumerated and
    every profile passed back via ``--profile``): a profile-gated service like
    ``falkordb-browser`` (profiles: [debug]) is otherwise OMITTED from the
    default resolution while its container runs 24/7 on this box — the exact
    mlflow fail-open class again (codex audit finding, 2026-08-04).
  * A RUNNING container in this compose project whose service is absent even
    from the profile-inclusive resolution (e.g. a container from a different
    overlay era) is reported UNMANAGED — info, not failure: its pin is not
    resolvable from the deploy's file set, but it must never silently vanish
    from the report.
  * Service -> container mapping uses compose labels
    (com.docker.compose.project / .service), NEVER container names: the live
    mlflow container is ``e2i_mlflow_dev`` (dev-overlay era) while the base
    compose names it ``e2i_mlflow`` — name-based mapping would miss exactly
    the drift this guard exists for.
  * Build-only services (no ``image:`` pin — bentoml/feast/feast-materializer)
    are skipped: there is no pin to drift from.
  * Services with no running container are INFO, not failure: liveness is the
    deploy health gates' job, and compose-defined-but-stopped services
    (prometheus stack, worker_heavy) are intentional on this box.
  * Comparison is by image REFERENCE (what ``docker inspect .Config.Image``
    reports the container was created from). A floating tag (redis:7-alpine)
    drifting by DIGEST is out of scope.
  * Stale allowlist entries (matching no current drift) WARN but do not fail:
    the recreate happens on the box, removing the entry is a PR — hard-failing
    would gate deploys on that ordering.

Exit codes: 0 = no unallowlisted drift; 1 = drift detected; 2 = infra/usage
error (fail closed — the deploy run must not go green on a broken check).

Usage:
    python3 scripts/deploy/check_image_drift.py --compose-cmd "$COMPOSE_CMD"
    # optional: --allowlist <path>  (default: image_drift_allowlist.json
    #                                next to this script)
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ALLOWLIST = Path(__file__).resolve().parent / "image_drift_allowlist.json"
_REQUIRED_ENTRY_KEYS = ("service", "running", "pinned", "issue", "added", "reason")


class AllowlistError(ValueError):
    """The allowlist is malformed — fail closed, never allow-by-accident."""


@dataclass(frozen=True)
class Ok:
    service: str
    running: str


@dataclass(frozen=True)
class Drift:
    service: str
    running: str
    pinned: str


@dataclass(frozen=True)
class Allowed:
    service: str
    running: str
    pinned: str
    issue: str
    added: str


@dataclass(frozen=True)
class NotRunning:
    service: str
    pinned: str


@dataclass(frozen=True)
class Skipped:
    service: str


@dataclass(frozen=True)
class Unmanaged:
    service: str
    running: str


@dataclass
class Report:
    ok: list[Ok] = field(default_factory=list)
    drift: list[Drift] = field(default_factory=list)
    allowlisted: list[Allowed] = field(default_factory=list)
    not_running: list[NotRunning] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    unmanaged: list[Unmanaged] = field(default_factory=list)
    stale_entries: list[dict] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.drift)


def parse_allowlist(text: str) -> list[dict]:
    """Parse + validate the allowlist JSON. Every entry must carry full
    provenance (issue + date + reason) — the pip-audit carve-out discipline."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AllowlistError(f"allowlist is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise AllowlistError("allowlist must be a JSON list of entry objects")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise AllowlistError(f"allowlist entry {i} is not an object: {entry!r}")
        missing = [k for k in _REQUIRED_ENTRY_KEYS if not entry.get(k)]
        if missing:
            raise AllowlistError(
                f"allowlist entry {i} missing required key(s) {missing}: {entry!r}"
            )
    return data


def evaluate(
    pinned: dict[str, str | None],
    running: dict[str, list[str]],
    allowlist: list[dict],
) -> Report:
    """Pure comparison core (unit-testable without docker).

    pinned:  service -> compose-resolved image ref, or None for build-only.
    running: service -> image refs of its RUNNING containers (replicas => many).
    """
    report = Report()
    used_entries: set[int] = set()
    for service in sorted(pinned):
        pin = pinned[service]
        if pin is None:
            report.skipped.append(Skipped(service))
            continue
        refs = running.get(service, [])
        if not refs:
            report.not_running.append(NotRunning(service, pin))
            continue
        for ref in refs:
            if ref == pin:
                report.ok.append(Ok(service, ref))
                continue
            matched = False
            for i, entry in enumerate(allowlist):
                if (
                    entry["service"] == service
                    and entry["running"] == ref
                    and entry["pinned"] == pin
                ):
                    report.allowlisted.append(
                        Allowed(service, ref, pin, entry["issue"], entry["added"])
                    )
                    used_entries.add(i)
                    matched = True
                    break
            if not matched:
                report.drift.append(Drift(service, ref, pin))
    # Running containers of this project whose service is not in the (profile-
    # inclusive) resolution: no pin to compare, but never silently omit them.
    for service in sorted(set(running) - set(pinned)):
        for ref in running[service]:
            report.unmanaged.append(Unmanaged(service, ref))
    report.stale_entries = [e for i, e in enumerate(allowlist) if i not in used_entries]
    return report


# --------------------------------------------------------------------------- #
# Docker/compose wrappers (thin; everything above is pure)
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: int = 120) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def config_argv(compose_cmd: list[str], profiles: list[str]) -> list[str]:
    """Pure: the ``config --format json`` argv with every profile included.

    ``docker compose config`` omits profile-gated services unless their profile
    is passed — which would fail-open on e.g. ``falkordb-browser`` (profiles:
    [debug], running 24/7 on this box). ``--profile`` is a root ``docker
    compose`` flag, so it goes before the ``config`` subcommand.
    """
    argv = list(compose_cmd)
    for p in profiles:
        argv += ["--profile", p]
    return [*argv, "config", "--format", "json"]


def list_profiles(compose_cmd: list[str]) -> list[str]:
    """Profile names defined by the compose file set (may be empty)."""
    return _run([*compose_cmd, "config", "--profiles"]).split()


def resolve_pins(compose_cmd: list[str]) -> tuple[str, dict[str, str | None]]:
    """Return (compose project name, service -> pinned image ref or None),
    including services gated behind ANY compose profile."""
    out = _run(config_argv(compose_cmd, list_profiles(compose_cmd)))
    cfg = json.loads(out)
    project = cfg.get("name")
    if not project:
        raise RuntimeError("compose config carries no project `name`")
    services = cfg.get("services") or {}
    return project, {svc: body.get("image") for svc, body in services.items()}


def resolve_running(project: str) -> dict[str, list[str]]:
    """service -> image refs of RUNNING containers, mapped via compose labels."""
    ids = _run(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ]
    ).split()
    if not ids:
        return {}
    fmt = '{{index .Config.Labels "com.docker.compose.service"}}\t{{.Config.Image}}'
    out = _run(["docker", "inspect", "--format", fmt, *ids])
    running: dict[str, list[str]] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        service, _, image = line.partition("\t")
        running.setdefault(service, []).append(image)
    return running


def render(report: Report) -> str:
    lines: list[str] = []
    for d in report.drift:
        lines.append(f"DRIFT     {d.service}: running={d.running} pinned={d.pinned}")
    for a in report.allowlisted:
        lines.append(
            f"ALLOWED   {a.service}: running={a.running} pinned={a.pinned} "
            f"({a.issue}, added {a.added})"
        )
    for o in report.ok:
        lines.append(f"OK        {o.service}: {o.running}")
    for n in report.not_running:
        lines.append(f"NOTRUN    {n.service}: pinned={n.pinned} (no running container — info only)")
    for s in report.skipped:
        lines.append(f"NOPIN     {s.service}: build-only service, no image pin (skipped)")
    for u in report.unmanaged:
        lines.append(
            f"UNMANAGED {u.service}: running={u.running} (service not in the deploy's "
            "compose resolution — info only)"
        )
    for e in report.stale_entries:
        lines.append(
            f"STALE     allowlist entry for {e['service']} ({e['issue']}) matches no "
            "current drift — remove it from image_drift_allowlist.json"
        )
    if report.failed:
        lines.append(f"IMAGE DRIFT CHECK: FAILED ({len(report.drift)} unallowlisted drift(s))")
        lines.append("A running container's image does not match its compose pin. Either the")
        lines.append("rollout skipped it (recreate it deliberately — for stateful services do a")
        lines.append("backup first, cf. #1479) or add a dated, ticketed allowlist entry in")
        lines.append("scripts/deploy/image_drift_allowlist.json referencing an open issue.")
    else:
        lines.append("IMAGE DRIFT CHECK: PASSED")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose-cmd",
        required=True,
        help='the deploy\'s compose command, e.g. "docker compose -f docker/docker-compose.yml"',
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help=f"path to the drift allowlist JSON (default: {DEFAULT_ALLOWLIST})",
    )
    args = parser.parse_args(argv)

    try:
        allowlist = parse_allowlist(args.allowlist.read_text())
        compose_cmd = shlex.split(args.compose_cmd)
        if not compose_cmd:
            raise RuntimeError("--compose-cmd is empty")
        project, pinned = resolve_pins(compose_cmd)
        running = resolve_running(project)
    except (AllowlistError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"IMAGE DRIFT CHECK: ERROR — {exc}", file=sys.stderr)
        return 2  # fail closed: a broken check must not pass the deploy

    report = evaluate(pinned, running, allowlist)
    print(
        f"Image drift check for compose project '{project}' "
        f"({len(pinned)} services, {sum(len(v) for v in running.values())} running containers)"
    )
    print(render(report))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
