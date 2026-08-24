"""Wiring the freshness check into a non-cron caller (#1798) and refusing to
attempt a reap that cannot work (#1801).

#1798 built ``check_maintenance_freshness.sh`` but nothing called it. A staleness
check nobody runs is the same as no staleness check -- and it must not be
installed into the crontab it audits, because then it dies exactly when the thing
it watches dies. ``health_check.sh`` is the seam: already manual/on-demand,
already has HEALTHY/UNHEALTHY counters, already exits 1 on degraded, and already
performs a non-HTTP check (``.env`` permissions). Verified it needs no root.

#1801: all 40 zombies on the box are children of ``supabase-meta``'s PID 1, which
is a containerized ``node`` with no init shim. A host-side ``kill -SIGCHLD`` at a
containerized PID 1 cannot reap them -- PID 1 in a namespace ignores signals it
has no handler for. The script had been attempting that every 15 minutes and, as
of #1799, honestly logging ``REAPED 0 of 40`` each time. Honest but useless
noise, and noise is what trained everyone to stop reading the log that hid #1798
for eight weeks.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MAINTENANCE = REPO_ROOT / "scripts" / "maintenance"
HEALTH_CHECK = REPO_ROOT / "scripts" / "health_check.sh"
CLEANUP_ORPHANS = MAINTENANCE / "cleanup_orphans.sh"


def _bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=90,
    )


def _extract_shell_function(script: str, name: str) -> str:
    """Slice a shell function out by its own-indent closing brace."""
    lines = script.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith(f"{name}() {{")),
        None,
    )
    assert start is not None, f"{name}() not found"
    indent = len(lines[start]) - len(lines[start].lstrip())
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "}" and (len(lines[j]) - len(lines[j].lstrip())) == indent:
            return "\n".join(ln[indent:] for ln in lines[start : j + 1])
    raise AssertionError(f"{name}() has no closing brace at its own indent")


# --------------------------------------------------------------------------- #
# #1798 -- the freshness check must have a caller
# --------------------------------------------------------------------------- #


def test_health_check_calls_the_freshness_check() -> None:
    text = HEALTH_CHECK.read_text()
    assert "check_maintenance_freshness.sh" in text, (
        "nothing calls check_maintenance_freshness.sh -- a staleness check with no "
        "caller is the same as no staleness check (#1798)"
    )


def test_the_freshness_check_is_still_NOT_installed_as_a_cron_job() -> None:
    """The constraint that motivated the whole design must survive the wiring."""
    setup = (MAINTENANCE / "setup_cron.sh").read_text()
    assert "check_maintenance_freshness.sh" not in setup, (
        "the freshness check must never run from the crontab it audits -- it would "
        "be dead exactly when the thing it watches is dead"
    )


def _run_maintenance_section(
    tmp_path: Path, stub_rc: int, stub_out: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run health_check.sh's maintenance block against a stub freshness script."""
    block = _extract_shell_function(HEALTH_CHECK.read_text(), "check_maintenance")
    stub = tmp_path / "check_maintenance_freshness.sh"
    stub.write_text(f"#!/bin/bash\necho '{stub_out}'\nexit {stub_rc}\n")
    stub.chmod(0o755)
    harness = textwrap.dedent(
        f"""
        GREEN=''; RED=''; YELLOW=''; NC=''
        HEALTHY=0; UNHEALTHY=0; SKIPPED=0
        FRESHNESS_SCRIPT="{stub}"
        {block}
        check_maintenance
        echo "COUNTS healthy=$HEALTHY unhealthy=$UNHEALTHY skipped=$SKIPPED"
        """
    )
    return _bash(harness)


def test_a_stale_maintenance_job_makes_the_health_check_report_unhealthy(tmp_path: Path) -> None:
    res = _run_maintenance_section(tmp_path, stub_rc=1, stub_out="docker_cleanup.log: STALE")
    assert "unhealthy=1" in res.stdout, f"stdout={res.stdout!r}"


def test_fresh_maintenance_jobs_count_as_healthy(tmp_path: Path) -> None:
    """Positive control: a checker that always reported unhealthy would pass above."""
    res = _run_maintenance_section(tmp_path, stub_rc=0, stub_out="OK: 3 maintenance job(s) fresh.")
    assert "unhealthy=0" in res.stdout, f"stdout={res.stdout!r}"
    assert "healthy=1" in res.stdout, f"stdout={res.stdout!r}"


def test_a_missing_freshness_script_is_SKIPPED_not_a_failure(tmp_path: Path) -> None:
    """On a dev box with no cron installed, absence must not fail the health check."""
    block = _extract_shell_function(HEALTH_CHECK.read_text(), "check_maintenance")
    harness = textwrap.dedent(
        f"""
        GREEN=''; RED=''; YELLOW=''; NC=''
        HEALTHY=0; UNHEALTHY=0; SKIPPED=0
        FRESHNESS_SCRIPT="{tmp_path}/does_not_exist.sh"
        {block}
        check_maintenance
        echo "COUNTS healthy=$HEALTHY unhealthy=$UNHEALTHY skipped=$SKIPPED"
        """
    )
    res = _bash(harness)
    assert "unhealthy=0" in res.stdout, f"stdout={res.stdout!r}"
    assert "skipped=1" in res.stdout, f"stdout={res.stdout!r}"


# --------------------------------------------------------------------------- #
# #1801 -- do not attempt a reap that cannot work
# --------------------------------------------------------------------------- #


def _fake_proc(tmp_path: Path, pid: str, cgroup: str) -> Path:
    d = tmp_path / "proc" / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "cgroup").write_text(cgroup)
    return tmp_path / "proc"


HOST_CGROUP = "0::/user.slice/user-1000.slice/session-1166.scope\n"
CONTAINER_CGROUP = "0::/system.slice/docker-b8bf87cca642d04e6ea6fbf7693383b55104a9f9c78740a86d545684e9ee1b37.scope\n"


def _parent_is_containerized(
    tmp_path: Path, pid: str, cgroup: str
) -> subprocess.CompletedProcess[str]:
    proc = _fake_proc(tmp_path, pid, cgroup)
    fn = _extract_shell_function(CLEANUP_ORPHANS.read_text(), "parent_is_containerized")
    return _bash(
        f'PROC_ROOT="{proc}"\n{fn}\nif parent_is_containerized {pid}; then echo YES; else echo NO; fi'
    )


def test_a_containerized_parent_is_detected(tmp_path: Path) -> None:
    res = _parent_is_containerized(tmp_path, "3846", CONTAINER_CGROUP)
    assert "YES" in res.stdout, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_a_host_parent_is_NOT_flagged_as_containerized(tmp_path: Path) -> None:
    """Positive control: a detector that always said YES would pass the test above,
    and would silently stop the script reaping zombies it genuinely can reap."""
    res = _parent_is_containerized(tmp_path, "4242", HOST_CGROUP)
    assert "NO" in res.stdout, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_an_unreadable_cgroup_is_NOT_assumed_containerized(tmp_path: Path) -> None:
    """Fail toward doing the work: if we cannot tell, still try the reap."""
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    fn = _extract_shell_function(CLEANUP_ORPHANS.read_text(), "parent_is_containerized")
    res = _bash(
        f'PROC_ROOT="{proc}"\n{fn}\nif parent_is_containerized 9999; then echo YES; else echo NO; fi'
    )
    assert "NO" in res.stdout, f"stdout={res.stdout!r}"


def test_zombies_under_a_containerized_parent_are_reported_not_signalled() -> None:
    """The script must say why it is standing down, not silently skip."""
    text = CLEANUP_ORPHANS.read_text()
    zombie_block = text[text.index("# 3.") : text.index("# 4.")]
    code = "\n".join(ln for ln in zombie_block.splitlines() if not ln.strip().startswith("#"))
    assert "parent_is_containerized" in code, (
        "the zombie branch must skip parents it cannot signal (#1801) -- a host-side "
        "SIGCHLD at a containerized PID 1 cannot reap anything"
    )
    assert "not host-reapable" in text or "NOT HOST-REAPABLE" in text, (
        "standing down must be reported, not silent"
    )


# --------------------------------------------------------------------------- #
# The summary must COUNT the maintenance result, not just print it afterwards
#
# The first cut of the #1798 wiring appended check_maintenance AFTER the Summary
# block. The counters it incremented had already been printed, so a STALE job
# rendered as "Unhealthy: 0 ... SYSTEM STATUS: DEGRADED" -- a report contradicting
# its own verdict. The nine tests above all passed, because every one of them
# harnesses check_maintenance in ISOLATION and reads the counter VARIABLES.
# Testing a function's effect on state is blind to WHERE that state is rendered.
# These two assert on the RENDERED OUTPUT instead.
# --------------------------------------------------------------------------- #


def _run_summary_tail(
    tmp_path: Path, healthy: int, unhealthy: int, skipped: int, stub_rc: int
) -> subprocess.CompletedProcess[str]:
    """Run health_check.sh from the SUMMARY banner to EOF with preset counters."""
    lines = HEALTH_CHECK.read_text().splitlines()
    # Anchor on whichever of the two blocks comes FIRST, so the harness reproduces
    # the real relative order in either layout -- that is what is under test.
    summary = next(i for i, ln in enumerate(lines) if ln.strip() == "# SUMMARY")
    maint = next(
        i for i, ln in enumerate(lines) if ln.startswith("# --- Maintenance cron freshness")
    )
    tail = "\n".join(lines[min(summary, maint) :])

    stub = tmp_path / "check_maintenance_freshness.sh"
    stub.write_text(f"#!/bin/bash\necho 'stub'\nexit {stub_rc}\n")
    stub.chmod(0o755)

    harness = tmp_path / "tail.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "GREEN=''; RED=''; YELLOW=''; NC=''\n"
        f"HEALTHY={healthy}; UNHEALTHY={unhealthy}; SKIPPED={skipped}\n"
        f'FRESHNESS_SCRIPT="{stub}"\n' + tail + "\n"
    )
    harness.chmod(0o755)
    return _bash(f'bash "{harness}"')


def test_a_stale_job_is_COUNTED_in_the_printed_unhealthy_total(tmp_path: Path) -> None:
    """The printed summary must not contradict the SYSTEM STATUS verdict."""
    res = _run_summary_tail(tmp_path, healthy=5, unhealthy=0, skipped=0, stub_rc=1)
    assert "Unhealthy: 1" in res.stdout, (
        "a STALE maintenance job printed as 'Unhealthy: 0' while the verdict said "
        f"DEGRADED -- the summary must count it, not trail it. stdout={res.stdout!r}"
    )
    assert "Total Services: 6" in res.stdout, f"stdout={res.stdout!r}"
    assert "SYSTEM STATUS: DEGRADED" in res.stdout, f"stdout={res.stdout!r}"


def test_a_fresh_job_is_COUNTED_in_the_printed_healthy_total(tmp_path: Path) -> None:
    """Positive control: pinning only the stale case would pass on a hardcoded +1."""
    res = _run_summary_tail(tmp_path, healthy=5, unhealthy=0, skipped=0, stub_rc=0)
    assert "Healthy: 6" in res.stdout, f"stdout={res.stdout!r}"
    assert "Unhealthy: 0" in res.stdout, f"stdout={res.stdout!r}"
    assert "SYSTEM STATUS: HEALTHY" in res.stdout, f"stdout={res.stdout!r}"


def test_the_maintenance_check_runs_BEFORE_the_counts_are_printed() -> None:
    """Structural companion: names the defect directly if the rendered tests fail."""
    lines = HEALTH_CHECK.read_text().splitlines()
    call = next(i for i, ln in enumerate(lines) if ln.strip() == "check_maintenance")
    printed = next(i for i, ln in enumerate(lines) if "Healthy: $HEALTHY" in ln)
    assert call < printed, (
        f"check_maintenance is invoked at line {call + 1} but the counts are printed "
        f"at line {printed + 1} -- the maintenance result can never appear in them"
    )
