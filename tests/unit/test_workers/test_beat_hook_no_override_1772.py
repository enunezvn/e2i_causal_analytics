"""CI guard: no ``on_after_finalize`` hook may write into ``beat_schedule`` (#1772).

The defect
----------
``src/tasks/ab_testing_tasks.py`` carried an ``@celery_app.on_after_finalize``
hook that called ``sender.add_periodic_task(..., name="ab-interim-analysis-check")``
— the *same key* the declared ``celery_app.conf.beat_schedule`` uses. Celery's
``add_periodic_task`` writes straight into ``conf.beat_schedule`` under that key,
so at every worker/beat boot the hook **replaced** the #1645 wall-clock
``crontab(hour=1, minute=15)`` with a bare ``86400`` interval. That is exactly the
failure #1645 fixed: an interval is measured from ``last_run_at``, so an entry
whose period exceeds the container's uptime never becomes due.

All four entries the hook installed collided with a declared key, and all four
also lost their declared ``options={"queue": "quick"}`` (``add_periodic_task``
rebuilds the entry from the signature, whose options are empty).

Why the obvious check misses it
-------------------------------
Importing ``src.workers.celery_app`` alone does **not** reproduce it: the hook
lives in a task module, so the receiver is never connected and the crontab looks
intact. Reproduction needs the real boot path — ``autoregister``/``autodiscover``
imports ``src.tasks``, and then ``finalize()`` fires the receivers. Connection is
not execution: ``conf.beat_schedule`` stays pristine until ``finalize()``.

Why this is not the existing guard
----------------------------------
``test_beat_schedule_registration.py::test_no_task_is_beat_scheduled_twice``
catches *two different keys scheduling one task* (the 2026-07-04 drift alert
storm). A same-key overwrite keeps the entry count at one, so it slips straight
through. This module guards the complementary case.

Single source of truth
----------------------
The codebase already decided this: the identical hook in
``drift_monitoring_tasks.py`` was removed and its entries moved into
``celery_app.conf.beat_schedule`` (see the NOTE at the bottom of that module and
the ``monitor-drift`` / ``drift-history-cleanup`` comments in ``celery_app.py``).
``setup_feedback_loop_periodic_tasks`` says the same in prose: "Static schedule is
preferred for production stability." #1772 is the one place that had not caught up.
"""

from __future__ import annotations

import pytest
from celery import Celery
from celery.schedules import crontab
from celery.utils.dispatch.signal import NONE_ID

import src.etl  # noqa: F401 — registers the src.etl.* rollup tasks
import src.tasks  # noqa: F401 — connects the on_after_finalize receivers
from src.workers.celery_app import celery_app

# Snapshot the DECLARED schedule at import (pytest collection), which is always
# before any test body can call finalize(). _FINALIZED_AT_IMPORT makes that
# assumption checkable rather than assumed: if it is ever True, the snapshot
# could already contain hook-written keys and the guard would be comparing
# against a polluted baseline, so the tests fail loudly instead of going green.
_FINALIZED_AT_IMPORT = celery_app.finalized
_DECLARED_KEYS = frozenset(celery_app.conf.beat_schedule)


def _fingerprint(beat_schedule: dict) -> dict[str, tuple[str, str, str]]:
    """A comparable, mutation-proof summary of a beat schedule.

    ``repr`` rather than the values themselves for two reasons: the entry dicts are
    live objects a hook could mutate in place (so holding references would compare a
    thing to itself), and ``43200.0 == 43200`` is True while the reprs differ — the
    float-to-int coercion ``add_periodic_task`` performs is a real change to the
    declared entry even where it is numerically identical.
    """
    return {
        key: (str(entry.get("task")), repr(entry.get("schedule")), repr(entry.get("options")))
        for key, entry in beat_schedule.items()
    }


_DECLARED_FINGERPRINT = _fingerprint(celery_app.conf.beat_schedule)

# The four keys #1772 was about, with the schedule celery_app.py declares.
_AB_ENTRIES: list[tuple[str, object]] = [
    ("ab-enrollment-health-check", 43200.0),
    ("ab-interim-analysis-check", crontab(hour=1, minute=15)),
    ("ab-results-cleanup", 604800.0),
    ("ab-srm-detection-sweep", 21600.0),
]


def _hook_registered_entries() -> dict[str, dict]:
    """Replay every connected ``on_after_finalize`` receiver against a probe app.

    Runs the real hook code (nothing is mocked) but sends it a throwaway
    ``Celery`` instance as ``sender``, so whatever the hooks register lands in the
    probe's ``beat_schedule`` and the production app is left untouched. That also
    makes this check independent of whether some earlier test already finalized
    the real app.

    Two things this deliberately does NOT see, both covered elsewhere in this module:

    * a receiver connected as ``connect(fn, sender=celery_app)`` — celery's
      ``Signal._live_receivers`` filters on ``id(sender)``, so a sender-filtered
      receiver fires on the real ``finalize()`` and is skipped here (measured: with
      one filtered and one unfiltered receiver connected, ``send(sender=probe)``
      fired 1 of 2, ``send(sender=celery_app)`` fired 2 of 2).
      ``test_every_finalize_receiver_is_unfiltered`` fails if such a receiver exists,
      so this function's coverage gap cannot open silently.
    * a hook that assigns into ``sender.conf.beat_schedule`` directly instead of
      calling ``add_periodic_task``.

    ``test_finalize_does_not_change_the_declared_beat_schedule`` catches both, and
    every other mechanism, by comparing the real app across the real ``finalize()``.
    The probe is kept because it names the offending keys before any of that runs.
    """
    probe = Celery("beat_hook_probe_1772")
    celery_app.on_after_finalize.send(sender=probe)
    # A fresh app is not `configured`, so add_periodic_task queues into
    # _pending_periodic_tasks; finalize() flushes them into conf.beat_schedule.
    probe.finalize()
    return dict(probe.conf.beat_schedule)


def _assert_probe_is_not_blind() -> None:
    """Positive control: a null result is only evidence if the probe can see a hit.

    Plants a receiver that registers a declared key, confirms the harness catches
    it, then disconnects. Without this, "no hook registered anything" would be
    indistinguishable from "the harness never looked".
    """
    victim = next(iter(sorted(_DECLARED_KEYS)))

    def _control(sender, **_kwargs):
        sender.add_periodic_task(
            60.0,
            celery_app.signature("src.tasks.collect_queue_metrics"),
            name=victim,
        )

    celery_app.on_after_finalize.connect(_control)
    try:
        caught = _hook_registered_entries()
    finally:
        celery_app.on_after_finalize.disconnect(_control)

    assert victim in caught, (
        "positive control failed: the probe did not observe a deliberately planted "
        "add_periodic_task registration, so a clean run below would prove nothing. "
        "Fix the harness before trusting this module."
    )


def test_no_finalize_hook_overwrites_a_declared_beat_key() -> None:
    """The #1772 invariant: a hook must never write a key ``celery_app.py`` declares."""
    assert not _FINALIZED_AT_IMPORT, (
        "celery_app was already finalized when this module was imported, so "
        "_DECLARED_KEYS may include hook-written entries. The baseline is no longer "
        "trustworthy — find what finalizes the app at import time."
    )
    assert len(_DECLARED_KEYS) >= 20, (
        f"declared beat_schedule unexpectedly small ({len(_DECLARED_KEYS)} entries) — "
        "an empty-vs-empty comparison would pass this test vacuously."
    )

    _assert_probe_is_not_blind()

    collisions = sorted(set(_hook_registered_entries()) & _DECLARED_KEYS)
    assert not collisions, (
        f"on_after_finalize hooks register beat keys that celery_app.py already "
        f"declares: {collisions}. add_periodic_task writes into conf.beat_schedule "
        "under that key, so the hook silently REPLACES the declared entry at every "
        "worker/beat boot — losing its crontab slot and its options (queue). Declare "
        "the schedule in celery_app.conf.beat_schedule and delete the hook (#1772)."
    )


def test_no_finalize_hook_registers_any_beat_entry() -> None:
    """Stronger SSOT invariant: all beat scheduling lives in ``conf.beat_schedule``.

    A hook-added entry is invisible in the declared dict, so every guard that reads
    ``celery_app.conf.beat_schedule`` at import time is blind to it — that is how
    both this defect and the 2026-07-04 drift double-fire survived. Keeping the
    schedule in one dict is what makes those guards meaningful.
    """
    _assert_probe_is_not_blind()

    registered = _hook_registered_entries()
    assert not registered, (
        f"on_after_finalize hooks add beat entries at runtime: {sorted(registered)}. "
        "Beat scheduling is declared in celery_app.conf.beat_schedule only — a "
        "runtime-added entry bypasses every guard that inspects that dict "
        "(test_beat_schedule_registration, test_beat_daily_wallclock_1645). See the "
        "NOTE at the end of src/tasks/drift_monitoring_tasks.py for the precedent."
    )


def test_every_finalize_receiver_is_unfiltered() -> None:
    """Keep ``_hook_registered_entries`` exhaustive, or fail loudly.

    Celery's ``Signal`` stores each receiver under ``(id(receiver), id(sender))`` and
    ``_live_receivers`` dispatches only to those whose sender id is ``NONE_ID`` (no
    filter) or matches the sender. A receiver connected as
    ``on_after_finalize.connect(fn, sender=celery_app)`` therefore runs on the real
    ``finalize()`` but is invisible to a probe-app replay. Rather than let the probe
    quietly under-report, assert the condition that makes it complete.
    """
    filtered = [
        receiver
        for (_receiver_id, sender_id), receiver in celery_app.on_after_finalize.receivers
        if sender_id != NONE_ID
    ]
    assert not filtered, (
        f"on_after_finalize receivers are connected with a sender filter: {filtered}. "
        "Those fire on the real finalize() but not on the probe-app replay in "
        "_hook_registered_entries(), so the two hook guards in this module would "
        "under-report. Either connect them without a sender= filter, or extend "
        "_hook_registered_entries to cover them."
    )


def test_finalize_does_not_change_the_declared_beat_schedule() -> None:
    """The faithful, mechanism-agnostic guard: real app, real ``finalize()``.

    Everything above inspects hooks. This one ignores how a change is made and asks
    only whether the schedule production runs still equals the schedule
    ``celery_app.py`` declares. It therefore covers the cases the probe cannot —
    sender-filtered receivers, and hooks that assign into ``conf.beat_schedule``
    directly instead of calling ``add_periodic_task``.
    """
    assert not _FINALIZED_AT_IMPORT, (
        "celery_app was already finalized when this module was imported, so "
        "_DECLARED_FINGERPRINT is a post-hook baseline and comparing against it "
        "proves nothing. Find what finalizes the app at import time."
    )
    assert len(_DECLARED_FINGERPRINT) >= 20, (
        f"declared beat_schedule unexpectedly small ({len(_DECLARED_FINGERPRINT)} "
        "entries) — an empty-vs-empty comparison would pass this test vacuously."
    )

    # Positive control for the comparator: a schedule that differs the way #1772
    # differed must not fingerprint equal. Without this, an over-lossy _fingerprint
    # would make every comparison below trivially true.
    victim = "ab-interim-analysis-check"
    mutated = dict(celery_app.conf.beat_schedule)
    mutated[victim] = {"task": "src.tasks.check_all_active_experiments", "schedule": 86400}
    assert _fingerprint(mutated) != _DECLARED_FINGERPRINT, (
        "_fingerprint() does not distinguish the #1772 overwrite from the declared "
        "entry, so it cannot detect the defect it exists to detect."
    )

    celery_app.finalize()

    actual = _fingerprint(celery_app.conf.beat_schedule)
    changed = {
        key: (_DECLARED_FINGERPRINT.get(key), actual.get(key))
        for key in set(_DECLARED_FINGERPRINT) | set(actual)
        if _DECLARED_FINGERPRINT.get(key) != actual.get(key)
    }
    assert not changed, (
        f"finalize() changed the declared beat schedule (declared -> after): {changed}. "
        "celery beat finalizes the app at startup, so this is the schedule production "
        "runs — it must be the one celery_app.py declares (#1772)."
    )


@pytest.mark.parametrize(("name", "expected"), _AB_ENTRIES)
def test_ab_entry_survives_finalize(name: str, expected: object) -> None:
    """End-to-end regression on the real app: finalize, then re-read the entry.

    This is the state production actually runs — ``celery beat`` finalizes the app
    at startup. Calling ``finalize()`` here rather than relying on some earlier test
    to have done it keeps the assertion order-independent.
    """
    celery_app.finalize()
    entry: dict[str, object] = celery_app.conf.beat_schedule[name]

    assert entry["schedule"] == expected, (
        f"{name} no longer holds its declared schedule after finalize: "
        f"{entry['schedule']!r} != {expected!r}. An on_after_finalize hook is "
        "overwriting it (#1772)."
    )
    assert entry.get("options") == {"queue": "quick"}, (
        f"{name} lost its declared queue routing after finalize: "
        f"{entry.get('options')!r}. add_periodic_task rebuilds the entry from the "
        "signature, whose options are empty, so the declared queue is dropped."
    )
