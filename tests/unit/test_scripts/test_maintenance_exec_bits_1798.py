"""Every script cron invokes must be executable IN GIT (#1798 follow-up).

Incident, 2026-08-23 22:30 -- caused by merging PR #1799:

    /bin/bash: line 1: .../scripts/maintenance/cleanup_orphans.sh: Permission denied

This repo sets ``core.fileMode = false``, so git ignores on-disk mode changes:
a ``chmod +x`` never reaches the index. Every maintenance script was tracked
``100644`` and only ran because ``setup_cron.sh`` chmods them at install time.

Git rewrites a working-tree file whenever a pull changes its content, and it
writes it with the *index* mode. So the moment a PR touched
``cleanup_orphans.sh``, pulling that PR stripped the exec bit and the ``*/15``
cron job began failing with "Permission denied" -- silently, since the failure
goes to the cron log nobody reads. That is the same silent-failure class as the
8-week outage this issue is about.

The guard derives the script list from ``setup_cron.sh`` (which owns what gets
installed) rather than hardcoding it, so a newly scheduled script is covered the
day it is added.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_CRON = REPO_ROOT / "scripts" / "maintenance" / "setup_cron.sh"


def _cron_invoked_scripts() -> set[str]:
    """Scripts that the installed crontab actually executes.

    Derived from the crontab body ``setup_cron.sh`` writes, so this cannot drift
    from what is really scheduled.
    """
    found = set()
    for line in SETUP_CRON.read_text().splitlines():
        # Only real crontab entries: five schedule fields then a user then the
        # command. A bare mention elsewhere in the installer (a chmod, an alias,
        # a self-reference) is NOT something cron executes.
        if not re.match(
            r"^\s*[\d*/,\-]+\s+[\d*/,\-]+\s+[\d*/,\-]+\s+[\d*/,\-]+\s+[\d*/,\-]+\s+\w+\s", line
        ):
            continue
        for match in re.finditer(r"scripts/maintenance/([A-Za-z0-9_]+\.sh)", line):
            found.add(match.group(1))
    return found


def _index_mode(rel_path: str) -> str:
    out = subprocess.run(
        ["git", "ls-files", "-s", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out, f"{rel_path} is not tracked by git"
    return out.split()[0]


def test_setup_cron_references_are_discoverable() -> None:
    """Positive control.

    If the regex stops matching, every assertion below passes vacuously over an
    empty set -- which is exactly how a guard goes green while checking nothing.
    """
    scripts = _cron_invoked_scripts()
    assert len(scripts) >= 3, (
        f"expected to find the cron-invoked maintenance scripts in setup_cron.sh, got {scripts}"
    )
    assert "cleanup_orphans.sh" in scripts
    assert "memory_monitor.sh" in scripts
    assert "docker_cleanup.sh" in scripts


def test_every_cron_invoked_script_is_executable_in_the_git_index() -> None:
    """core.fileMode=false means a chmod that is not in the index does not exist.

    Red on ``fe7da66d0``: cleanup_orphans.sh, memory_monitor.sh and
    docker_cleanup.sh were all ``100644``.
    """
    offenders = []
    for name in sorted(_cron_invoked_scripts()):
        rel = f"scripts/maintenance/{name}"
        mode = _index_mode(rel)
        if mode != "100755":
            offenders.append(f"{rel} is {mode}")

    assert not offenders, (
        "these scripts are executed by cron but are not executable in the git index, "
        "so any pull that rewrites them strips the exec bit and the job fails with "
        "'Permission denied':\n  " + "\n  ".join(offenders) + "\n"
        "Fix with: git update-index --chmod=+x <path>"
    )


def test_the_freshness_checker_is_executable_too() -> None:
    """It is not cron-invoked (deliberately), but callers still exec it."""
    assert _index_mode("scripts/maintenance/check_maintenance_freshness.sh") == "100755"
