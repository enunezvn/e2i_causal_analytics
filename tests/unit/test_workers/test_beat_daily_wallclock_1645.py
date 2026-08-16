"""CI guard: daily beat entries must use a wall-clock crontab, not a bare interval (#1645).

The defect
----------
``celery beat`` measures an interval schedule (``"schedule": 86400.0``) from
``last_run_at``, which ``PersistentScheduler`` keeps in a state file. That file
lived in the scheduler container's *ephemeral* ``/tmp`` tmpfs, so every deploy
destroyed it and reset ``last_run_at`` to boot time. An interval entry therefore
only becomes due once *uptime > interval* — and on a box that deploys several
times a day, **no 24-hour entry can ever fire**. Measured in the issue: over a
5-hour container life only the <=4h entries fired, and the two 4h entries fired
exactly once each. The observable casualty was #1649 —
``sync_operational_corpus`` was scheduled, implemented, and its queue consumed,
and it had still never run.

The fix has two halves and needs both:

1. ``docker/docker-compose.yml`` puts the state file on the ``celerybeat_state``
   named volume, so ``last_run_at`` survives a container recreate.
   Guarded by ``tests/unit/test_docker/test_compose_beat_state_volume_1645.py``.
2. Daily entries move to ``crontab(hour=..., minute=...)`` so the work lands at a
   predictable hour instead of "deploy time + 24h", and so the ordering the
   surrounding comments assert (per-HCP before territory; rollups before corpus
   sync) is real rather than incidental — under intervals every daily entry came
   due on the same tick.

This module guards half (2) and the celery semantics both halves rely on.
"""

from __future__ import annotations

import datetime as dt

from celery.schedules import crontab, schedule

import src.etl  # noqa: F401 — registers the src.etl.* rollup tasks
import src.tasks  # noqa: F401 — registers all src.tasks.* task modules
from src.workers.celery_app import celery_app

# The 11 entries that were ``86400.0`` before #1645, with the slot each now owns.
# Rationale for every slot lives in the WALL-CLOCK SLOT MAP comment above
# ``celery_app.conf.beat_schedule``; the pairs here are the machine-checkable half.
DAILY_SLOTS_UTC: dict[str, tuple[int, int]] = {
    "drift-history-cleanup": (0, 45),
    "ab-interim-analysis-check": (1, 15),
    "feedback-loop-medium-window": (2, 10),
    "feedback-loop-drift-analysis": (2, 40),
    "business-metrics-per-hcp-rollup": (3, 15),
    "patient-adherence-rollup": (3, 30),
    "territory-metrics-rollup": (3, 45),
    "sync-operational-corpus": (4, 0),
    "sync-chunk-corpus": (4, 15),
    "dspy-prompt-optimization-daily": (6, 0),
    "insight-lifecycle-consolidate": (6, 30),
}

# Windows this box already spends on something else (measured 2026-08-16 from the
# host crontab and the job logs). A daily beat slot inside one of these would be
# contending with a pg_dump or a full synthetic reseed.
#   backup : `0 2 * * *`   scripts/backup_cron.sh   -> 02:00:01..02:01:26 observed
#   reseed : `0 3 * * 1`   scripts/reseed_synthetic.sh -> 03:00:07..03:01:08 observed
# 5 minutes of headroom on each measured ~90s window.
RESERVED_HOST_WINDOWS_UTC: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "host backup (scripts/backup_cron.sh)": ((2, 0), (2, 5)),
    "host reseed (scripts/reseed_synthetic.sh, Mondays)": ((3, 0), (3, 5)),
}

# Beat entries that were already on a wall clock before #1645 and whose comments
# say, in so many words, "stay off the other windows". New daily slots must not
# land on top of them either.
PREEXISTING_BEAT_SLOTS_UTC: dict[str, tuple[int, int]] = {
    "routing-label-nightly": (4, 30),
    "chatbot-optimization-drain": (5, 30),
}

# Bare interval schedules that are longer than a day and are deliberately left
# alone by #1645. They are the SAME failure class and are only rescued by the
# state-file volume (half 1) — a 7-day interval could never fire off ephemeral
# state either. Their comments claim a Sunday cadence that an interval cannot
# honour, which is a follow-up, not this change. Listing them here means a NEW
# long interval fails this guard instead of joining them silently.
ACKNOWLEDGED_LONG_INTERVALS: dict[str, float] = {
    "feast-materialize-full-weekly": 604800.0,
    "ab-results-cleanup": 604800.0,
    "feedback-loop-long-window": 604800.0,
    "nppes-refresh-monthly": 2592000.0,
}

_ONE_DAY_SECONDS = 86400.0


def _minute_of_day(hour: int, minute: int) -> int:
    return hour * 60 + minute


def test_every_daily_entry_uses_a_wallclock_crontab() -> None:
    """Each formerly-``86400.0`` entry must be a crontab pinned to its slot."""
    beat_schedule = celery_app.conf.beat_schedule

    missing = sorted(set(DAILY_SLOTS_UTC) - set(beat_schedule))
    assert not missing, (
        f"beat entries named in DAILY_SLOTS_UTC are gone from beat_schedule: {missing}. "
        "If an entry was intentionally removed, drop it from this map too."
    )

    wrong: dict[str, str] = {}
    for name, (hour, minute) in DAILY_SLOTS_UTC.items():
        sched = beat_schedule[name]["schedule"]
        if not isinstance(sched, crontab):
            wrong[name] = f"{sched!r} (not a crontab)"
        elif sched.hour != {hour} or sched.minute != {minute}:
            wrong[name] = f"hour={sorted(sched.hour)} minute={sorted(sched.minute)}"

    assert not wrong, (
        "daily beat entries must be wall-clock crontab schedules at their documented "
        f"slot (#1645). Offenders: {wrong}. A bare interval float restores the "
        "original defect: the interval is measured from last_run_at, so a task with a "
        "period longer than the container's uptime never becomes due."
    )


def test_no_new_bare_daily_interval_is_introduced() -> None:
    """No beat entry may carry a bare >=24h interval outside the known allow-list."""
    offenders: dict[str, float] = {}
    for name, entry in celery_app.conf.beat_schedule.items():
        sched = entry["schedule"]
        if isinstance(sched, (int, float)) and float(sched) >= _ONE_DAY_SECONDS:
            if ACKNOWLEDGED_LONG_INTERVALS.get(name) != float(sched):
                offenders[name] = float(sched)

    assert not offenders, (
        f"beat entries use a bare interval of >=24h: {offenders}. Express a daily "
        "cadence as crontab(hour=..., minute=...) instead — see #1645. If a longer "
        "interval is genuinely wanted, add it to ACKNOWLEDGED_LONG_INTERVALS with "
        "the reasoning."
    )


def test_daily_slots_are_distinct() -> None:
    """11 tasks stacked on one minute would defeat the point of staggering them."""
    by_slot: dict[tuple[int, int], list[str]] = {}
    for name, slot in DAILY_SLOTS_UTC.items():
        by_slot.setdefault(slot, []).append(name)
    collisions = {slot: names for slot, names in by_slot.items() if len(names) > 1}
    assert not collisions, f"daily beat slots collide: {collisions}"


def test_daily_slots_avoid_reserved_windows() -> None:
    """Slots must stay off the host cron windows and the pre-existing beat slots."""
    offenders: list[str] = []

    for name, (hour, minute) in DAILY_SLOTS_UTC.items():
        at = _minute_of_day(hour, minute)
        for label, (start, end) in RESERVED_HOST_WINDOWS_UTC.items():
            if _minute_of_day(*start) <= at < _minute_of_day(*end):
                offenders.append(f"{name} @ {hour:02d}:{minute:02d} inside {label}")
        for other, slot in PREEXISTING_BEAT_SLOTS_UTC.items():
            if (hour, minute) == slot:
                offenders.append(f"{name} @ {hour:02d}:{minute:02d} collides with {other}")

    assert not offenders, (
        f"daily beat slots land on windows this box already spends elsewhere: {offenders}"
    )


def test_preexisting_wallclock_entries_still_hold_their_slots() -> None:
    """#1645 must not have moved the two entries that were already on a wall clock."""
    for name, (hour, minute) in PREEXISTING_BEAT_SLOTS_UTC.items():
        sched = celery_app.conf.beat_schedule[name]["schedule"]
        assert isinstance(sched, crontab), f"{name} is no longer a crontab: {sched!r}"
        assert sched.hour == {hour} and sched.minute == {minute}, (
            f"{name} moved off its documented {hour:02d}:{minute:02d} slot"
        )


def test_beat_timezone_is_utc() -> None:
    """The slot map, the host crontab and the job logs are all UTC — keep them aligned."""
    assert celery_app.conf.timezone == "UTC"
    assert celery_app.conf.enable_utc is True


def test_crontab_catches_up_after_a_missed_slot_but_an_interval_does_not() -> None:
    """The celery semantics both halves of the #1645 fix rely on.

    Pins the actual behaviour rather than asserting it in a comment, so a celery
    upgrade that changes it fails here instead of silently in production.
    """
    now = dt.datetime(2026, 8, 16, 4, 0, tzinfo=dt.timezone.utc)

    daily = crontab(hour=3, minute=15, app=celery_app, nowfun=lambda: now)

    # Scheduler was down across yesterday's slot; last_run_at survived on the
    # named volume. The missed slot must fire immediately on the next start.
    missed = dt.datetime(2026, 8, 15, 3, 15, tzinfo=dt.timezone.utc)
    assert daily.is_due(missed).is_due is True

    # Already ran today -> not due again.
    ran_today = dt.datetime(2026, 8, 16, 3, 15, tzinfo=dt.timezone.utc)
    assert daily.is_due(ran_today).is_due is False

    # The defect, reproduced: a 24h interval whose last_run_at was reset by a
    # deploy an hour ago is not due, and will not be due until uptime exceeds 24h.
    interval = schedule(run_every=_ONE_DAY_SECONDS, app=celery_app, nowfun=lambda: now)
    deploy_reset = dt.datetime(2026, 8, 16, 3, 0, tzinfo=dt.timezone.utc)
    assert interval.is_due(deploy_reset).is_due is False
