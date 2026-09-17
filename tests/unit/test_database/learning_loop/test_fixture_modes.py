"""The fixture's two modes, decided from prod's ledger rather than a hard-coded lane (#2065).

``base_db`` is prod's CURRENT schema with prod's ledger. What the fixture upgrades is DERIVED:
``pending`` is every key the runner would apply that the ledger does not hold, in runner order.

* **Behaviour mode** (always): tests run on the base with every pending migration applied —
  the schema production will have once the deploy's ``run_migrations.sh`` has run.
* **Upgrade-path mode**: a test marked ``realdb_upgrade`` runs on the base itself, before the
  pending migrations, and only when the migrations it upgrades through ARE pending. Otherwise it
  skips with a reason naming what it waits for. It never xfails and never silently passes.

These are pure decisions, so they run everywhere, CI included, without a database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_database.learning_loop import _pg

RUNNER = ["100_a.sql", "101_b.sql", "ml/040_x.sql", "ml/041_y.sql", "ml/044_z.sql", "audit/001.sql"]


# ---------------------------------------------------------------------------
# pending = runner keys - prod ledger, in runner order
# ---------------------------------------------------------------------------


def test_pending_is_what_the_runner_would_apply_in_its_order():
    ledger = ["ml/041_y.sql", "100_a.sql", "101_b.sql", "retired/old.sql"]
    assert _pg.derive_pending(RUNNER, ledger) == ["ml/040_x.sql", "ml/044_z.sql", "audit/001.sql"]


def test_nothing_is_pending_when_the_ledger_holds_every_runner_key():
    assert _pg.derive_pending(RUNNER, list(reversed(RUNNER))) == []


def test_a_simulated_key_is_pending_even_though_prod_applied_it():
    assert _pg.derive_pending(RUNNER, RUNNER, simulated="ml/044_z.sql") == ["ml/044_z.sql"]


# ---------------------------------------------------------------------------
# migrate(): ledgered keys are no-ops, pending keys apply in runner order
# ---------------------------------------------------------------------------


def test_plan_applies_only_unapplied_keys_through_upto_in_runner_order():
    applied = {"100_a.sql", "ml/041_y.sql"}
    assert _pg.migration_plan(RUNNER, applied, "ml/044_z.sql") == [
        "101_b.sql",
        "ml/040_x.sql",
        "ml/044_z.sql",
    ]


def test_plan_through_a_ledgered_key_is_a_no_op():
    assert _pg.migration_plan(RUNNER, set(RUNNER[:4]), "ml/041_y.sql") == []


def test_plan_through_everything_is_the_pending_list():
    applied = {"100_a.sql", "101_b.sql", "ml/040_x.sql"}
    assert _pg.migration_plan(RUNNER, applied, _pg.ALL_PENDING) == _pg.derive_pending(
        RUNNER, sorted(applied)
    )


def test_plan_refuses_a_key_the_runner_does_not_know():
    with pytest.raises(ValueError, match="not a runner migration key"):
        _pg.migration_plan(RUNNER, set(), "ml/999_nope.sql")


# ---------------------------------------------------------------------------
# realdb_upgrade: skip unless what the test upgrades through is pending
# ---------------------------------------------------------------------------


def test_upgrade_test_with_no_keys_needs_some_pending_migration():
    reason = _pg.upgrade_skip_reason((), [])
    assert reason is not None
    assert "no migration is pending" in reason and "not coverage" in reason
    assert _pg.upgrade_skip_reason((), ["ml/045_new.sql"]) is None


def test_upgrade_test_through_named_keys_runs_only_when_all_of_them_are_pending():
    keys = ("ml/040_x.sql", "ml/041_y.sql")
    assert _pg.upgrade_skip_reason(keys, ["ml/040_x.sql", "ml/041_y.sql", "ml/044_z.sql"]) is None

    reason = _pg.upgrade_skip_reason(keys, ["ml/041_y.sql"])
    assert reason is not None
    # Names what it waits for, and which of those prod already carries.
    assert "ml/040_x.sql" in reason and "ml/041_y.sql" in reason
    assert "already applied: ['ml/040_x.sql']" in reason
    assert "not coverage" in reason


def test_a_later_unrelated_pending_migration_does_not_wake_a_lane_upgrade_test():
    # The next deploy adding ml/045 must not run ml/040's upgrade test against a base that is
    # no longer pre-040: that would block the deploy on a test whose premise is gone.
    reason = _pg.upgrade_skip_reason(("ml/040_x.sql",), ["ml/045_new.sql"])
    assert reason is not None and "already applied: ['ml/040_x.sql']" in reason


# ---------------------------------------------------------------------------
# E2I_DB_SIMULATE_PENDING: rehearse upgrade mode on the droplet, only through a real rollback
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "database" / "ml").mkdir(parents=True)
    (tmp_path / "database" / "migrations").mkdir(parents=True)
    (tmp_path / "database" / "ml" / "rollback_044.sql").write_text("-- rollback\n")
    return tmp_path


def test_simulation_is_off_by_default(tmp_path):
    root = _repo(tmp_path)
    assert _pg.simulated_pending_key(None, RUNNER, RUNNER, root) is None
    assert _pg.simulated_pending_key("", RUNNER, RUNNER, root) is None


def test_simulation_accepts_an_applied_key_that_has_a_rollback_file(tmp_path):
    root = _repo(tmp_path)
    assert _pg.simulated_pending_key("ml/044_z.sql", RUNNER, RUNNER, root) == "ml/044_z.sql"
    assert _pg.rollback_file("ml/044_z.sql", root) == root / "database" / "ml" / "rollback_044.sql"
    assert (
        _pg.rollback_file("145_x.sql", root)
        == root / "database" / "migrations" / "rollback_145.sql"
    )


@pytest.mark.parametrize(
    "key, message",
    [
        ("ml/041_y.sql", "no rollback file"),
        ("ml/099_never.sql", "not in prod's ledger"),
    ],
)
def test_simulation_refuses_a_key_it_cannot_faithfully_un_apply(tmp_path, key, message):
    with pytest.raises(ValueError, match=message):
        _pg.simulated_pending_key(key, RUNNER, RUNNER, _repo(tmp_path))


def test_simulation_is_refused_while_a_real_migration_is_pending(tmp_path):
    # Its purpose is rehearsing upgrade mode when nothing is pending. With a real pending key the
    # base is already pre-upgrade, and a simulated one on top leaves no template equal to prod.
    ledger = RUNNER[:-1]  # audit/001.sql is genuinely pending
    with pytest.raises(ValueError, match=r"already pending: \['audit/001.sql'\]"):
        _pg.simulated_pending_key("ml/044_z.sql", ledger, RUNNER, _repo(tmp_path))
