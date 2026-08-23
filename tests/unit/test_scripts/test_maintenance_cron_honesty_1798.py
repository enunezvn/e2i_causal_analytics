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
import shutil
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


def _extract_shell_function(script: str, name: str) -> str:
    """Slice a shell function out by its own-indent closing brace.

    Same algorithm the deploy guards use (#1796); duplicated here rather than
    imported because that conftest is scoped to ``tests/unit/test_docker``.
    """
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
def test_npm_age_rule_compares_numerically(etime: str, hours_threshold: int, should_select: bool) -> None:
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
    zombie_block = text[text.index("# 3."):]
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
    assert "REAPED" in zombie_code, (
        "the zombie branch must report what was actually reaped"
    )


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

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode == 0, f"expected fresh -> rc 0. stdout={res.stdout} stderr={res.stderr}"


def test_a_stale_log_is_detected_and_exits_nonzero(cron_fixture: tuple[Path, Path]) -> None:
    """The exact 2026-06-30 shape: a */5 job whose log is 8 weeks old."""
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "memory_monitor.log", 8 * 7 * 24 * 3600)  # 8 weeks
    _touch(logdir / "docker_cleanup.log", 3600)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode != 0, "an 8-week-stale */5 job must fail the check"
    assert "memory_monitor.log" in res.stdout, "the stale job must be named"


def test_a_missing_log_is_reported_not_silently_passed(cron_fixture: tuple[Path, Path]) -> None:
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "docker_cleanup.log", 3600)
    # memory_monitor.log never created

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode != 0, "a missing log must not pass"
    assert "memory_monitor.log" in res.stdout


def test_intervals_are_DERIVED_from_the_crontab_not_hardcoded(cron_fixture: tuple[Path, Path]) -> None:
    """Changing the schedule in the crontab must change the verdict.

    A second hardcoded copy of the intervals would drift from the real crontab
    exactly the way the deploy-guard tables did (#1791). This test rewrites the
    schedule and requires the verdict to follow it.
    """
    cron, logdir = cron_fixture
    for name in ("orphan_cleanup", "memory_monitor", "docker_cleanup"):
        _touch(logdir / f"{name}.log", 3600)  # all exactly 1 h old

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


def test_unparseable_schedule_is_reported_as_unknown_not_guessed(cron_fixture: tuple[Path, Path]) -> None:
    """Omit-or-report: never fabricate an interval for a schedule we can't read."""
    cron, logdir = cron_fixture
    for name in ("orphan_cleanup", "memory_monitor", "docker_cleanup"):
        _touch(logdir / f"{name}.log", 60)
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


def test_a_job_with_no_log_redirect_is_not_reported_as_stale(cron_fixture: tuple[Path, Path]) -> None:
    """The log-rotation job writes no log; absence of a log is not a failure."""
    cron, logdir = cron_fixture
    _touch(logdir / "orphan_cleanup.log", 60)
    _touch(logdir / "memory_monitor.log", 60)
    _touch(logdir / "docker_cleanup.log", 3600)

    res = _bash(f"{FRESHNESS} --crontab {cron}")
    assert res.returncode == 0
    assert "truncate" not in res.stdout, "a job with no log redirect must not be checked for staleness"


def test_freshness_check_does_not_itself_depend_on_cron() -> None:
    """The checker must be callable from a non-cron caller.

    A staleness check installed *into* the same crontab cannot detect that the
    crontab stopped running -- which is the entire failure mode of #1798.
    """
    text = FRESHNESS.read_text()
    assert "--crontab" in text, "the checker must accept an explicit crontab path so it can be run anywhere"
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
    res = _bash(harness + "\n" + block + "\necho \"FINAL_KILLED_COUNT=$KILLED_COUNT\"")

    assert "FINAL_KILLED_COUNT=0" in res.stdout, (
        "the parent never reaped, so nothing was killed -- but the script "
        f"still counted some. stdout={res.stdout!r}"
    )
    assert "REAPED 0 of 3" in res.stdout, (
        f"expected an honest 'REAPED 0 of 3'. stdout={res.stdout!r}"
    )
    assert "3 remain" in res.stdout, f"the remaining zombies must be reported. stdout={res.stdout!r}"


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
    res = _bash(harness + "\n" + block + "\necho \"FINAL_KILLED_COUNT=$KILLED_COUNT\"; rm -f /tmp/e2i_zombie_probe_*")

    assert "REAPED 2 of 3" in res.stdout, f"expected 'REAPED 2 of 3'. stdout={res.stdout!r}"
    assert "FINAL_KILLED_COUNT=2" in res.stdout, f"stdout={res.stdout!r}"
