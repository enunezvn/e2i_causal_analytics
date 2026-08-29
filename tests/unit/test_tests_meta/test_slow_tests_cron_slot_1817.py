"""Contract test: the slow-tests nightly must not sample ChEMBL in EBI's hot hour.

Background (#1817)
------------------
``slow-tests.yml`` Job A runs the live clinical-context suite, which calls
EBI's ChEMBL API with no retry (a seconds-scale retry cannot cross the
5–7 min ``500`` windows measured on 08-18/24/25). With the cron at
``0 7 * * *`` and GitHub's usual 30–60 min schedule delay, those calls landed
at 07:42–08:09 UTC every night — and 4 of 11 nightlies went red on the same
EBI ``500`` while the client, the suite and production were all clean.

A 48 h probe of the exact failing query from the droplet (2026-08-26 18:01 →
08-28 17:51 UTC) found two shapes: a day-long outage that no slot escapes
(08-27, 00:00–11:54 UTC), and — on the good day — a single cluster at
08:00–08:40 UTC (09:00 BST, UK start of day; 5/28 bad) with 05h/06h/07h
fully clean (0/38). The cron therefore moved to 05:00 UTC so the calls land
~05:40–06:40 UTC.

This test pins the slot OUT of the 07–08h UTC window so a later edit cannot
silently walk the nightly back into it. It deliberately does not pin the
exact hour: any daily slot whose calls stay clear of 07:00–09:00 UTC is fine.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "slow-tests.yml"

# Cron HOURS (UTC) whose ChEMBL calls — cron + ~35–70 min — would land inside
# EBI's measured 07:30–09:00 UTC instability window.
FORBIDDEN_CRON_HOURS_UTC = {7, 8}


def _schedule_crons() -> list[str]:
    data = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML parses the bare ``on:`` key as boolean True.
    triggers = data.get("on") or data.get(True) or {}
    schedule = triggers.get("schedule") or []
    return [entry["cron"] for entry in schedule]


def test_slow_tests_has_exactly_one_daily_cron() -> None:
    crons = _schedule_crons()
    assert len(crons) == 1, f"expected one schedule entry in slow-tests.yml, got {crons}"
    fields = crons[0].split()
    assert len(fields) == 5, f"malformed cron {crons[0]!r}"
    minute, hour, dom, month, dow = fields
    assert minute.isdigit() and hour.isdigit(), (
        f"cron {crons[0]!r} must name a fixed minute and hour so the ChEMBL sample "
        "time is predictable"
    )
    assert (dom, month, dow) == ("*", "*", "*"), (
        f"cron {crons[0]!r} is no longer nightly — the live suite is the only thing "
        "that exercises the ChEMBL contract, so it must run every day"
    )


def test_slow_tests_cron_avoids_ebi_morning_window() -> None:
    """The nightly's ChEMBL calls must not land in EBI's 07:30–09:00 UTC hot hour."""
    (cron,) = _schedule_crons()
    hour = int(cron.split()[1])
    assert hour not in FORBIDDEN_CRON_HOURS_UTC, (
        f"slow-tests.yml cron {cron!r} fires at {hour:02d}:xx UTC; with GitHub's "
        "30–60 min schedule delay Job A's ChEMBL calls land inside EBI's measured "
        "07:30–09:00 UTC instability window (#1817: 4 of 11 nightlies red on an "
        "upstream 500). Pick a slot whose calls stay clear of that window."
    )
