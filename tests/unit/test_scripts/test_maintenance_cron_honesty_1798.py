"""Guards for the maintenance-cron defects found in #1798.

Context: ``/etc/cron.d/e2i-maintenance`` silently stopped executing on 2026-06-30
because root's password entered a forced-change state and ``pam_unix`` rejected
every job. Re-enabling the jobs surfaced two defects in what they *do*, plus the
absence of anything that would have reported the outage:

1. ``cleanup_orphans.sh`` reported ``Processes killed: 40`` while 40 zombies
   remained -- it counted ``SIGCHLD`` signals *sent* as processes *killed*.
2. ``substr($2,1,2) > 1`` is a **lexical** comparison (``substr`` returns a
   string), so the "npm older than 1 hour" rule actually fired at 10 hours.
3. Nothing checked whether these jobs were running at all.

These tests execute the real shell against fixtures. They do not stub the
comparison logic -- a stub would re-implement the bug under test.
"""

from __future__ import annotations

import os
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MAINTENANCE = REPO_ROOT / "scripts" / "maintenance"
CLEANUP_ORPHANS = MAINTENANCE / "cleanup_orphans.sh"
FRESHNESS = MAINTENANCE / "check_maintenance_freshness.sh"


def _bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=60,
    )


# --------------------------------------------------------------------------- #
# Defect 2 -- the npm age rule must compare NUMERICALLY
# --------------------------------------------------------------------------- #
# The shipped expression is:
#     $2 ~ /^[0-9]+-/ || $2 ~ /^[0-9]+:[0-9]+:[0-9]+/ && substr($2,1,2) > 1
# `substr` returns a string, so "02" > 1 is a LEXICAL compare and is false.
# These cases pin the boundary the rule is supposed to enforce.


@pytest.mark.parametrize(
    ("etime", "hours_threshold", "should_select"),
    [
        # --- the bug: hours 01..09 were silently exempt ---
        ("02:49:50", 1, True),  # 2h49m vs 1h threshold -> must select
        ("09:59:59", 1, True),  # 9h59m vs 1h threshold -> must select
        ("01:00:01", 1, True),
        # --- genuinely under threshold ---
        ("00:30:00", 1, False),
        ("00:59:59", 1, False),
        # --- a higher threshold must actually hold ---
        ("02:49:50", 10, False),  # 2h49m vs 10h -> spared
        ("10:00:01", 10, True),
        ("09:59:59", 10, False),
        # --- days form is always over any hour threshold ---
        ("3-04:00:00", 1, True),
        ("3-04:00:00", 10, True),
    ],
)
def test_npm_age_rule_compares_numerically(
    etime: str, hours_threshold: int, should_select: bool
) -> None:
    """A process's age must be compared as a number, not as a string.

    Red on the shipped code for every ``01..09``-hour case: ``"02" > "1"`` is
    false because ``'0' < '1'``.
    """
    text = CLEANUP_ORPHANS.read_text()
    awk_expr = None
    for line in text.splitlines():
        if "old_npm=" in line and "awk" in line:
            awk_expr = line
            break
    assert awk_expr is not None, "cleanup_orphans.sh no longer has the old_npm awk selector"

    # Feed one synthetic `ps` row through the SHIPPED selector line.
    probe = textwrap.dedent(
        f"""
        NPM_MAX_AGE_HOURS={hours_threshold}
        ps() {{ printf '%s\\n' '12345 {etime} npm'; }}
        export NPM_MAX_AGE_HOURS
        {awk_expr}
        echo "SELECTED=[$old_npm]"
        """
    )
    res = _bash(probe)
    selected = "12345" in res.stdout
    assert selected is should_select, (
        f"etime={etime} threshold={hours_threshold}h: "
        f"expected selected={should_select}, got {should_select is not selected and 'the opposite' or selected}. "
        f"stdout={res.stdout!r}"
    )


def test_npm_threshold_is_a_named_knob_not_a_magic_number() -> None:
    """The age threshold must be a named variable.

    It was an inline ``> 1`` whose comment said "1 hour" while the code did 10.
    A named knob makes the value a deliberate decision instead of a typo away
    from killing live tooling.
    """
    text = CLEANUP_ORPHANS.read_text()
    assert "NPM_MAX_AGE_HOURS" in text, (
        "cleanup_orphans.sh must name its npm age threshold (NPM_MAX_AGE_HOURS)"
    )


# --------------------------------------------------------------------------- #
# Defect 1 -- the zombie reap must not report kills it did not achieve
# --------------------------------------------------------------------------- #


def test_zombie_reap_reports_what_it_actually_reaped_not_signals_sent() -> None:
    """Signalling a parent is not reaping a zombie.

    The 2026-08-23 run logged ``Processes killed: 40`` while the zombie count
    stayed at 40. Red on the shipped code, which increments ``KILLED_COUNT``
    once per ``kill -SIGCHLD`` regardless of outcome.
    """
    text = CLEANUP_ORPHANS.read_text()

    # Locate the zombie section.
    assert "zombie" in text.lower(), "cleanup_orphans.sh no longer handles zombies"
    zombie_block = text[text.index("# 3.") :]
    zombie_block = zombie_block[: zombie_block.index("# 4.")]
    # Discriminate CODE from PROSE. The fix's own comment quotes the defective
    # line to explain what it used to do; a raw grep over the block therefore
    # reports FAIL on a correct fix. Strip comment lines before asserting.
    zombie_code = "\n".join(
        ln for ln in zombie_block.splitlines() if not ln.strip().startswith("#")
    )

    assert "KILLED_COUNT=$((KILLED_COUNT + 1))" not in zombie_code, (
        "the zombie branch still counts a SIGCHLD as a kill -- signalling a parent "
        "does not reap the zombie, and reporting it as killed is a success claim "
        "with no outcome behind it (#1798 defect 1)"
    )
    assert "REAPED" in zombie_code, "the zombie branch must report what was actually reaped"


def test_zombie_reap_recounts_after_signalling() -> None:
    """The count must be derived from a re-observation, not from the loop."""
    text = CLEANUP_ORPHANS.read_text()
    zombie_block = text[text.index("# 3.") : text.index("# 4.")]
    zombie_code = "\n".join(
        ln for ln in zombie_block.splitlines() if not ln.strip().startswith("#")
    )
    # A re-count means the zombie list is observed a second time after signalling.
    assert zombie_code.count("count_zombies") >= 2, (
        "the zombie branch must re-observe the zombie set after signalling parents, "
        "otherwise the number it reports cannot reflect what was reaped"
    )


# --------------------------------------------------------------------------- #
# The staleness check -- the thing that would have caught the 8-week outage
# --------------------------------------------------------------------------- #


@pytest.fixture
def cron_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A crontab shaped like /etc/cron.d/e2i-maintenance, plus a log dir."""
    logdir = tmp_path / "log"
    logdir.mkdir()
    cron = tmp_path / "e2i-maintenance"
    cron.write_text(
        textwrap.dedent(
            f"""\
            # E2I Causal Analytics - Maintenance Cron Jobs
            SHELL=/bin/bash
            PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

            */15 * * * * root /opt/e2i/cleanup_orphans.sh >> {logdir}/orphan_cleanup.log 2>&1
            */5 * * * * root /opt/e2i/memory_monitor.sh --auto-cleanup >> {logdir}/memory_monitor.log 2>&1
            0 2 * * * root find /var/log/e2i -name "*.log" -size +10M -exec truncate -s 1M {{}} \\;
            0 3 * * 0 root /opt/e2i/docker_cleanup.sh >> {logdir}/docker_cleanup.log 2>&1
            """
        )
    )
    return cron, logdir


def _touch(path: Path, age_seconds: float) -> None:
    path.write_text("x")
    when = time.time() - age_seconds
    os.utime(path, (when, when))


def _stamp(logdir: Path, script: str, age_seconds: float) -> None:
    """Write the success stamp a completed run leaves behind.

    #1798 follow-up: the checker keys on this, not on the log's mtime. A log is
    written by anything that runs the script -- including a --dry-run and a run
    that aborts halfway -- so log mtime answers "was this file written", not
    "did this job complete".
    """
    _touch(logdir / f".{script}.success", age_seconds)


def test_freshness_script_exists() -> None:
    assert FRESHNESS.exists(), (
        "scripts/maintenance/check_maintenance_freshness.sh must exist -- nothing "
        "reported the 8-week cron outage (#1798)"
    )
    assert os.access(FRESHNESS, os.X_OK), "check_maintenance_freshness.sh must be executable"


def test_all_fresh_logs_exit_zero(cron_fixture: tuple[Path, Path]) -> None:
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)  # 1 min old, interval 15 min
    _touch(logdir / "memory_monitor.log", 60)  # 1 min old, interval 5 min
    _touch(logdir / "docker_cleanup.log", 3600)  # 1 h old, interval 1 week
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "memory_monitor", 60)
    _stamp(logdir, "docker_cleanup", 3600)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode == 0, f"expected fresh -> rc 0. stdout={res.stdout} stderr={res.stderr}"


def test_a_stale_log_is_detected_and_exits_nonzero(cron_fixture: tuple[Path, Path]) -> None:
    """The exact 2026-06-30 shape: a */5 job whose log is 8 weeks old."""
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "memory_monitor.log", 8 * 7 * 24 * 3600)  # 8 weeks
    _touch(logdir / "docker_cleanup.log", 3600)
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "memory_monitor", 8 * 7 * 24 * 3600)
    _stamp(logdir, "docker_cleanup", 3600)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode != 0, "an 8-week-stale */5 job must fail the check"
    assert "memory_monitor.log" in res.stdout, "the stale job must be named"


def test_a_missing_log_is_reported_not_silently_passed(cron_fixture: tuple[Path, Path]) -> None:
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "docker_cleanup.log", 3600)
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "docker_cleanup", 3600)
    # memory_monitor never ran: no log, no stamp

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode != 0, "a missing log must not pass"
    assert "memory_monitor.log" in res.stdout


def test_intervals_are_DERIVED_from_the_crontab_not_hardcoded(
    cron_fixture: tuple[Path, Path],
) -> None:
    """Changing the schedule in the crontab must change the verdict.

    A second hardcoded copy of the intervals would drift from the real crontab
    exactly the way the deploy-guard tables did (#1791). This test rewrites the
    schedule and requires the verdict to follow it.
    """
    cron, logdir = cron_fixture
    for name in ("orphan_cleanup", "memory_monitor", "docker_cleanup"):
        _touch(logdir / f"{name}.log", 3600)  # all exactly 1 h old
    for script in ("cleanup_orphans", "memory_monitor", "docker_cleanup"):
        _stamp(logdir, script, 3600)

    # With memory_monitor on */5, a 1 h old log is stale.
    res_stale = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res_stale.returncode != 0
    assert "memory_monitor.log" in res_stale.stdout

    # Rewrite ONLY that job to daily; the same 1 h old log is now fresh.
    text = cron.read_text().replace(
        "*/5 * * * * root /opt/e2i/memory_monitor.sh",
        "0 4 * * * root /opt/e2i/memory_monitor.sh",
    )
    cron.write_text(text)
    res_fresh = _bash(f"{FRESHNESS} --crontab {cron}")
    assert "memory_monitor.log" not in res_fresh.stdout.replace("memory_monitor.log:", ""), (
        "verdict did not follow the crontab -- intervals look hardcoded rather than derived"
    )


def test_unparseable_schedule_is_reported_as_unknown_not_guessed(
    cron_fixture: tuple[Path, Path],
) -> None:
    """Omit-or-report: never fabricate an interval for a schedule we can't read."""
    cron, logdir = cron_fixture
    for name in ("orphan_cleanup", "memory_monitor", "docker_cleanup"):
        _touch(logdir / f"{name}.log", 60)
    for script in ("cleanup_orphans", "memory_monitor", "docker_cleanup"):
        _stamp(logdir, script, 60)
    text = cron.read_text().replace(
        "*/5 * * * * root /opt/e2i/memory_monitor.sh",
        "1-59/7,3 2-5 * * * root /opt/e2i/memory_monitor.sh",
    )
    cron.write_text(text)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert "UNKNOWN" in res.stdout.upper(), (
        "an unrecognised cron schedule must be reported as unknown, not silently "
        "assigned an interval"
    )


def test_a_job_with_no_log_redirect_is_not_reported_as_stale(
    cron_fixture: tuple[Path, Path],
) -> None:
    """The log-rotation job writes no log; absence of a log is not a failure."""
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "memory_monitor.log", 60)
    _touch(logdir / "docker_cleanup.log", 3600)
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "memory_monitor", 60)
    _stamp(logdir, "docker_cleanup", 3600)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode == 0
    assert "truncate" not in res.stdout, (
        "a job with no log redirect must not be checked for staleness"
    )


def test_freshness_check_does_not_itself_depend_on_cron() -> None:
    """The checker must be callable from a non-cron caller.

    A staleness check installed *into* the same crontab cannot detect that the
    crontab stopped running -- which is the entire failure mode of #1798.
    """
    text = FRESHNESS.read_text()
    assert "--crontab" in text, (
        "the checker must accept an explicit crontab path so it can be run anywhere"
    )
    setup = (MAINTENANCE / "setup_cron.sh").read_text()
    assert "check_maintenance_freshness.sh" not in setup, (
        "the freshness check must NOT be installed as a cron job -- a checker that "
        "runs from the cron it is checking is dead whenever the thing it checks is dead"
    )


def test_zombie_reap_reports_ZERO_when_the_parent_ignores_sigchld() -> None:
    """Behavioural reproduction of the 2026-08-23 observation.

    40 zombies, 40 parents signalled, 40 zombies still present afterwards -- and
    the log said ``Processes killed: 40``. This executes the real zombie branch
    against a parent that does not reap, and requires the reported number to be
    the number that actually went away: zero.

    The structural tests above assert the *shape* of the fix; this one asserts
    the *behaviour*, and is the one that would survive a fix that re-counts but
    still reports the wrong figure.
    """
    text = CLEANUP_ORPHANS.read_text()
    block = text[text.index("# 3.") : text.index("# 4.")]

    harness = textwrap.dedent(
        """
        DRY_RUN=false
        TOTAL_FOUND=0
        KILLED_COUNT=0
        log() { echo "$*"; }
        sleep() { :; }
        # The parent ignores SIGCHLD, so the zombies never go away.
        kill() { return 0; }
        ps() {
          if [[ "${1:-}" == "aux" ]]; then
            echo "root 111 0.0 0.0 0 0 ? Z 10:00 0:00 [a] <defunct>"
            echo "root 222 0.0 0.0 0 0 ? Z 10:00 0:00 [b] <defunct>"
            echo "root 333 0.0 0.0 0 0 ? Z 10:00 0:00 [c] <defunct>"
          else
            echo " 3846"
          fi
        }
        """
    )
    res = _bash(harness + "\n" + block + '\necho "FINAL_KILLED_COUNT=$KILLED_COUNT"')

    assert "FINAL_KILLED_COUNT=0" in res.stdout, (
        "the parent never reaped, so nothing was killed -- but the script "
        f"still counted some. stdout={res.stdout!r}"
    )
    assert "REAPED 0 of 3" in res.stdout, (
        f"expected an honest 'REAPED 0 of 3'. stdout={res.stdout!r}"
    )
    assert "3 remain" in res.stdout, (
        f"the remaining zombies must be reported. stdout={res.stdout!r}"
    )


def test_zombie_reap_reports_the_TRUE_count_when_the_parent_does_reap() -> None:
    """Positive control: when zombies do go away, the number must reflect it.

    Without this, a fix that always reports 0 would pass the test above.
    """
    text = CLEANUP_ORPHANS.read_text()
    block = text[text.index("# 3.") : text.index("# 4.")]

    harness = textwrap.dedent(
        """
        DRY_RUN=false
        TOTAL_FOUND=0
        KILLED_COUNT=0
        STATE=/tmp/e2i_zombie_probe_$$
        echo first > "$STATE"
        log() { echo "$*"; }
        sleep() { :; }
        kill() { return 0; }
        ps() {
          if [[ "${1:-}" == "aux" ]]; then
            if [[ "$(cat "$STATE")" == "first" ]]; then
              echo "root 111 0.0 0.0 0 0 ? Z 10:00 0:00 [a] <defunct>"
              echo "root 222 0.0 0.0 0 0 ? Z 10:00 0:00 [b] <defunct>"
              echo "root 333 0.0 0.0 0 0 ? Z 10:00 0:00 [c] <defunct>"
              echo second > "$STATE"
            else
              # two were reaped; one parent was stuck
              echo "root 333 0.0 0.0 0 0 ? Z 10:00 0:00 [c] <defunct>"
            fi
          else
            echo " 3846"
          fi
        }
        """
    )
    res = _bash(
        harness
        + "\n"
        + block
        + '\necho "FINAL_KILLED_COUNT=$KILLED_COUNT"; rm -f /tmp/e2i_zombie_probe_*'
    )

    assert "REAPED 2 of 3" in res.stdout, f"expected 'REAPED 2 of 3'. stdout={res.stdout!r}"
    assert "FINAL_KILLED_COUNT=2" in res.stdout, f"stdout={res.stdout!r}"


# --------------------------------------------------------------------------- #
# The false-OK: a written log is not a completed run
# --------------------------------------------------------------------------- #


def test_a_fresh_log_with_NO_success_stamp_is_not_reported_fresh(
    cron_fixture: tuple[Path, Path],
) -> None:
    """The exact false-OK I produced on 2026-08-23.

    Running ``docker_cleanup.sh --dry-run`` by hand wrote four lines to the real
    log and then aborted. That reset the log's mtime, and the checker reported
    ``docker_cleanup.log: OK`` for a job that had not run since 2026-06-28.

    A log is written by anything that invokes the script. Only a completed,
    non-dry run leaves a success stamp.
    """
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "memory_monitor.log", 60)
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "memory_monitor", 60)
    # docker_cleanup: log freshly written (a dry-run), but it never completed
    _touch(logdir / "docker_cleanup.log", 5)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode != 0, (
        "a freshly-written log with no success stamp must NOT pass -- that is the "
        f"false-OK this guard exists for. stdout={res.stdout!r}"
    )
    assert "docker_cleanup" in res.stdout
    # Pin the BRANCH, not just the verdict. Without this the test passes even when
    # the missing-stamp branch is dead: execution falls through to `stat` on a
    # nonexistent file, mtime becomes 0, the age is astronomically over the limit,
    # and it fails as STALE -- the right answer for the wrong reason, which a
    # mutation of that branch would sail straight through.
    assert "NEVER COMPLETED" in res.stdout, (
        "a missing stamp must be reported as never-completed, not incidentally "
        f"caught by an age comparison against mtime 0. stdout={res.stdout!r}"
    )


def test_a_stale_stamp_fails_even_when_the_log_was_just_written(
    cron_fixture: tuple[Path, Path],
) -> None:
    """Log mtime must not be able to mask a stale stamp."""
    cron, logdir = cron_fixture
    for name in ("orphan_cleanup", "memory_monitor", "docker_cleanup"):
        _touch(logdir / f"{name}.log", 1)  # all just written
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "docker_cleanup", 3600)
    _stamp(logdir, "memory_monitor", 8 * 7 * 24 * 3600)  # 8 weeks stale

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode != 0
    assert "memory_monitor" in res.stdout


def test_a_fresh_stamp_passes_even_if_the_log_is_older(
    cron_fixture: tuple[Path, Path],
) -> None:
    """Positive control: the stamp is the signal, so it must be able to pass.

    Without this, a checker that always failed would satisfy both tests above.
    """
    cron, logdir = cron_fixture
    for name in ("orphan_cleanup", "memory_monitor", "docker_cleanup"):
        _touch(logdir / f"{name}.log", 100000)  # ancient logs
    _stamp(logdir, "cleanup_orphans", 60)
    _stamp(logdir, "memory_monitor", 60)
    _stamp(logdir, "docker_cleanup", 3600)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode == 0, f"stdout={res.stdout!r}"


# --------------------------------------------------------------------------- #
# docker_cleanup.sh --dry-run
# --------------------------------------------------------------------------- #

DOCKER_CLEANUP = MAINTENANCE / "docker_cleanup.sh"


def _docker_stub_bin(tmp_path: Path) -> Path:
    """Put a fake `docker` FIRST on PATH.

    It must be a real executable on PATH, not a shell function: the script runs
    as its own `bash` process, and functions do not cross a subprocess boundary.
    Getting this wrong once already ran the real cleanup against the live daemon.

    It rejects `ps --filter until=...` the way the real daemon does
    ("invalid filter 'until'"), because that rejection under `set -e` IS the bug.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    docker = bindir / "docker"
    docker.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        "  *\"system df\"*--format*) printf 'Images\\t19.81GB\\nContainers\\t0B\\nLocal Volumes\\t4.316GB\\nBuild Cache\\t0B\\n' ;;\n"
        "  *\"system df\"*) printf 'TYPE TOTAL ACTIVE SIZE RECLAIMABLE\\n' ;;\n"
        "  *ps*-a*until=*) echo \"Error response from daemon: invalid filter 'until'\" >&2; exit 1 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    docker.chmod(0o755)
    return bindir


def _run_docker_cleanup(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run docker_cleanup.sh fully isolated: fake docker on PATH, log in tmp."""
    bindir = _docker_stub_bin(tmp_path)
    log = tmp_path / "docker_cleanup.log"
    env = {
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "LOG_FILE": str(log),
        "HOME": str(tmp_path),
    }
    return subprocess.run(
        ["bash", str(DOCKER_CLEANUP), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_the_docker_stub_actually_shadows_the_real_binary(tmp_path: Path) -> None:
    """Guard the guard.

    If the stub is not first on PATH, every test below silently exercises the
    REAL docker daemon -- which is exactly what happened on 2026-08-23 and
    pruned 8 volumes off the live box.
    """
    bindir = _docker_stub_bin(tmp_path)
    res = subprocess.run(
        ["bash", "-c", "command -v docker"],
        capture_output=True,
        text=True,
        env={"PATH": f"{bindir}:{os.environ.get('PATH', '')}"},
        timeout=30,
    )
    assert res.stdout.strip() == str(bindir / "docker"), (
        f"the stub does not shadow the real docker: resolved {res.stdout.strip()!r}"
    )


def test_dry_run_completes_instead_of_aborting_on_the_until_filter(tmp_path: Path) -> None:
    """`until` is not a valid `docker ps` filter; the real daemon rejects it.

    Red on the shipped script: `set -euo pipefail` plus
    `exited=$(docker ps -a --filter until=24h ...)` kills the run at step 3, so
    --dry-run has never reached the volume or network steps.
    """
    res = _run_docker_cleanup(tmp_path, "--dry-run")
    assert "Pruning dangling volumes" in res.stdout, (
        f"--dry-run aborted before the volume step. rc={res.returncode} stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    assert "Pruning unused networks" in res.stdout


def test_dry_run_reports_the_BUILD_CACHE_row_not_the_images_row(tmp_path: Path) -> None:
    """`docker system df --format '{{.Reclaimable}}' | head -1` takes Images.

    Red on the shipped script, which reported "Build cache reclaimable: 19.81GB"
    while build cache was 0B.
    """
    res = _run_docker_cleanup(tmp_path, "--dry-run")
    line = [ln for ln in res.stdout.splitlines() if "uild cache reclaimable" in ln]
    assert line, f"no build-cache line. stdout={res.stdout!r}"
    assert "19.81GB" not in line[0], f"reported the Images row as build cache: {line[0]!r}"
    assert "0B" in line[0], f"expected the Build Cache row (0B): {line[0]!r}"


def test_dry_run_does_NOT_write_a_success_stamp(tmp_path: Path) -> None:
    """A preview must never satisfy the freshness check."""
    _run_docker_cleanup(tmp_path, "--dry-run")
    stamp = tmp_path / ".docker_cleanup.success"
    assert not stamp.exists(), "a --dry-run wrote a success stamp; that is the false-OK again"


def test_a_real_run_DOES_write_a_success_stamp(tmp_path: Path) -> None:
    """Positive control for the stamp: it must be writable on a real run."""
    res = _run_docker_cleanup(tmp_path)
    stamp = tmp_path / ".docker_cleanup.success"
    assert stamp.exists(), (
        f"a completed real run must leave a success stamp. rc={res.returncode} stdout={res.stdout[-500:]!r}"
    )


def test_dry_run_lists_only_what_the_real_prune_would_actually_remove(tmp_path: Path) -> None:
    """The preview must match the action.

    ``docker volume prune -f`` (no ``--all``) removes ONLY anonymous volumes --
    confirmed on Docker 29.1.3, where ``-a`` is documented as "Remove all unused
    volumes, not just anonymous ones", and empirically by a real run that left
    every named volume intact.

    But the preview listed ``docker volume ls -f dangling=true``, which includes
    unused NAMED volumes. On this box that meant the dry-run named
    ``e2i_grafana_data``, ``e2i_loki_data``, ``e2i_prometheus_data`` and
    ``e2i_promtail_positions`` as removal candidates that the real prune would
    never touch -- a preview that tells an operator their observability data is
    about to be deleted when it is not.
    """
    bindir = _docker_stub_bin(tmp_path)
    # A stub whose `volume ls -f dangling=true` returns one anonymous volume
    # (64 hex chars) and one NAMED volume, the way the real daemon does.
    anon = "a" * 64
    (bindir / "docker").write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        "  *\"system df\"*--format*) printf 'Build Cache\\t0B\\n' ;;\n"
        "  *\"system df\"*) printf 'TYPE TOTAL ACTIVE SIZE RECLAIMABLE\\n' ;;\n"
        '  *ps*-a*until=*) echo "invalid filter" >&2; exit 1 ;;\n'
        f"  *\"volume ls\"*dangling*) printf '{anon}\\ne2i_grafana_data\\n' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    (bindir / "docker").chmod(0o755)

    log = tmp_path / "docker_cleanup.log"
    res = subprocess.run(
        ["bash", str(DOCKER_CLEANUP), "--dry-run"],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "LOG_FILE": str(log),
            "HOME": str(tmp_path),
        },
        timeout=120,
    )
    vol_section = res.stdout[res.stdout.find("dangling volumes") :]
    vol_section = vol_section[: vol_section.find("unused networks")] or vol_section

    assert "e2i_grafana_data" not in vol_section, (
        "the preview listed a NAMED volume the real `docker volume prune -f` would "
        f"never remove. section={vol_section!r}"
    )
    assert anon[:12] in vol_section, (
        f"the preview must still list anonymous volumes, which ARE removed. section={vol_section!r}"
    )


def test_dry_run_lists_only_networks_that_have_no_connected_containers(tmp_path: Path) -> None:
    """Third instance of the same shape in this script's preview.

    ``docker network prune`` removes only networks with NO connected containers,
    but the preview listed ``docker network ls --filter type=custom`` -- i.e.
    every custom network. On this box that named ``e2i_network`` and
    ``supabase-network``, which carry the entire running stack.
    """
    bindir = _docker_stub_bin(tmp_path)
    (bindir / "docker").write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        "  *\"system df\"*--format*) printf 'Build Cache\\t0B\\n' ;;\n"
        "  *\"system df\"*) printf 'TYPE TOTAL ACTIVE SIZE RECLAIMABLE\\n' ;;\n"
        '  *ps*-a*until=*) echo "invalid filter" >&2; exit 1 ;;\n'
        "  *\"network ls\"*) printf 'busy_net\\nidle_net\\n' ;;\n"
        '  *"network inspect"*busy_net*) echo 2 ;;\n'
        '  *"network inspect"*idle_net*) echo 0 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    (bindir / "docker").chmod(0o755)

    res = subprocess.run(
        ["bash", str(DOCKER_CLEANUP), "--dry-run"],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "LOG_FILE": str(tmp_path / "docker_cleanup.log"),
            "HOME": str(tmp_path),
        },
        timeout=120,
    )
    net = res.stdout[res.stdout.find("unused networks") :]
    assert "busy_net" not in net, (
        f"the preview listed a network with connected containers. section={net!r}"
    )
    assert "idle_net" in net, f"the preview must list genuinely unused networks. section={net!r}"
