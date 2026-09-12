"""Static checks on ml/042, which persists a twin simulation's estimate scope (#2053).

No database: CI has none for this table, and the deploy applies the file with
``scripts/run_migrations.sh`` inside ``--single-transaction`` with its ledger row. These pin what a
real apply depends on: the columns are additive, nullable and default-less (a default or a
backfill would stamp legacy rows with a scope nobody recorded), every statement re-applies
cleanly, the runner wraps the file, the rollback is never auto-applied, and every column the
repository writes exists in the schema the migrations build.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.integration.test_migrations_no_inner_txn import _scan_for_bare_txn
from tests.unit.test_database.learning_loop import _pg

DATABASE = _pg.REPO_ROOT / "database"
KEY = "ml/042_twin_simulations_estimate_scope.sql"
MIGRATION = DATABASE / KEY
ROLLBACK = DATABASE / "ml" / "rollback_042.sql"

SCOPE_COLUMNS = {
    "effect_scope_regions": "text[]",
    "cohort_ate": "double precision",
    "cohort_ci_lower": "double precision",
    "cohort_ci_upper": "double precision",
}


def _code(path: Path) -> str:
    """SQL with ``--`` and ``/* */`` comments removed."""
    if not path.exists():
        pytest.fail(f"{path.relative_to(_pg.REPO_ROOT)} is missing")
    text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.DOTALL)
    return "\n".join(re.sub(r"--.*$", "", line) for line in text.splitlines())


@pytest.mark.unit
@pytest.mark.parametrize("column, sql_type", sorted(SCOPE_COLUMNS.items()))
def test_adds_each_scope_column_nullable_without_a_default(column, sql_type):
    code = _code(MIGRATION)
    m = re.search(
        rf"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+{column}\s+([^,;]+)", code, flags=re.IGNORECASE
    )
    assert m, f"{column} is not added idempotently"
    declared = " ".join(m.group(1).split()).lower()
    assert declared == sql_type, f"{column} declared as {declared!r}"


@pytest.mark.unit
def test_does_not_backfill_legacy_rows():
    code = _code(MIGRATION)
    assert not re.search(r"\bUPDATE\b|\bDEFAULT\b|\bNOT\s+NULL\b", code, flags=re.IGNORECASE)


@pytest.mark.unit
def test_every_added_constraint_is_dropped_first_so_a_reapply_is_clean():
    code = _code(MIGRATION)
    added = re.findall(r"ADD\s+CONSTRAINT\s+(\w+)", code, flags=re.IGNORECASE)
    assert added, "expected the scope invariants to be enforced in the table"
    for name in added:
        drop = re.search(rf"DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+{name}\b", code, re.IGNORECASE)
        add = re.search(rf"ADD\s+CONSTRAINT\s+{name}\b", code, re.IGNORECASE)
        assert drop and add and drop.start() < add.start(), name


@pytest.mark.unit
def test_runner_applies_it_wrapped_with_no_transaction_control_of_its_own():
    assert KEY in _pg.runner_migration_keys()
    assert _pg.runner_unwraps(MIGRATION.read_text()) is False
    assert _scan_for_bare_txn(MIGRATION) == []


@pytest.mark.unit
def test_rollback_is_never_auto_applied_and_undoes_the_columns_and_ledger_row():
    code = _code(ROLLBACK)
    assert not [k for k in _pg.runner_migration_keys() if k.endswith(ROLLBACK.name)]
    assert _scan_for_bare_txn(ROLLBACK) == []
    for column in SCOPE_COLUMNS:
        assert re.search(rf"DROP\s+COLUMN\s+IF\s+EXISTS\s+{column}\b", code, re.IGNORECASE), column
    assert f"filename = '{KEY}'" in code


def _schema_columns() -> set[str]:
    """Columns of twin_simulations as ml/012's CREATE TABLE plus every later ADD COLUMN build it."""
    create = _code(DATABASE / "ml" / "012_digital_twin_tables.sql")
    body = re.search(
        r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+twin_simulations\s*\((.*?)\n\);", create, re.DOTALL
    )
    assert body, "twin_simulations CREATE TABLE not found in ml/012"
    columns = {
        m.group(1)
        for m in re.finditer(r"^\s*([a-z_][a-z0-9_]*)\s+\S", body.group(1), re.MULTILINE)
        if m.group(1).upper() != "CONSTRAINT"
    }
    for path in sorted(DATABASE.rglob("*.sql")):
        if path.name.startswith("rollback_"):
            continue
        code = _code(path)
        for stmt in re.findall(
            r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:public\.)?twin_simulations\b(.*?);",
            code,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            columns.update(
                re.findall(r"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", stmt, re.IGNORECASE)
            )
    return columns


@pytest.mark.unit
def test_every_column_save_simulation_writes_exists_in_the_migrated_schema():
    """A written column no migration adds fails every insert on the deployed table."""
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        SimulationRecommendation,
        SimulationResult,
    )
    from src.digital_twin.twin_repository import SimulationRepository

    sink: dict = {}

    class _Client:
        def table(self, _name):
            return self

        def insert(self, row):
            sink["row"] = row
            return self

        async def execute(self):
            return SimpleNamespace(data=[sink["row"]])

    result = SimulationResult(
        model_id=uuid4(),
        intervention_config=InterventionConfig(intervention_type="email_campaign"),
        twin_count=1,
        simulated_ate=0.1,
        simulated_ci_lower=0.0,
        simulated_ci_upper=0.2,
        simulated_std_error=0.05,
        target_regions=["northeast"],
        cohort_ate=0.1,
        cohort_ci_lower=0.0,
        cohort_ci_upper=0.2,
        recommendation=SimulationRecommendation.REFINE,
        recommendation_rationale="r",
        simulation_confidence=0.5,
        execution_time_ms=1,
    )
    asyncio.run(SimulationRepository(supabase_client=_Client()).save_simulation(result, "Kisqali"))

    missing = sorted(set(sink["row"]) - _schema_columns())
    assert missing == []
    assert set(SCOPE_COLUMNS) <= set(sink["row"])
